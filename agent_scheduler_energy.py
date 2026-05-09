"""Scheduler — energy budget, step estimation, context overflow exception."""
from __future__ import annotations
import os
import math
import time
from typing import Dict

from config import Config
from agent_kernel_glue import SubAgentEventType

class ContextTooLongError(Exception):
    pass

class StepEstimator:
    """Gamma-Exponential 共轭：根据已完成子任务步数预测剩余步数"""
    def __init__(self, total_subtasks: int, prior_mean: float = 3.0, prior_strength: float = 1.0):
        self.N = total_subtasks
        self.alpha = prior_mean * prior_strength
        self.beta = prior_strength
        self.completed = 0

    def update(self, steps_taken: int):
        self.alpha += steps_taken
        self.beta += 1
        self.completed += 1

    def predict_remaining(self) -> tuple:
        """返回 (期望, 下限5%, 上限95%)"""
        remaining = self.N - self.completed
        if remaining <= 0:
            return 0, 0, 0
        shape = remaining * self.alpha
        rate = self.beta
        mean = shape / rate
        std = math.sqrt(shape) / rate
        return mean, max(0, mean - 1.645 * std), mean + 1.645 * std

    def predict_total(self, steps_done: int) -> tuple:
        rem_mean, rem_low, rem_high = self.predict_remaining()
        return steps_done + rem_mean, steps_done + rem_low, steps_done + rem_high

class BayesianEnergyManager:
    """双层能量预算：软约束（流动资金）+ 硬约束（总预算）+ 投资-收益模型"""
    def __init__(self, total_energy: float = 154800.0, step_overhead: int = 1000,
                 cost_step: float = 2.0,
                 cost_tool: float = 0.2,
                 explore_roi: float = 0.5):
        # 双层约束
        self.total_energy = total_energy   # 硬上限（不可突破）
        self.total_spent = 0.0             # 已净消耗（不可逆，只增不减）
        self.energy = total_energy          # 流动资金（可预扣、可返还、可获奖金）
        self.step_overhead = step_overhead  # 每步固定 token 开销
        self.cost_step = cost_step
        self.cost_tool = cost_tool
        # Token 使用统计
        self._total_tokens = 0
        self._total_input = 0
        self._total_output = 0
        # Spawn 保留栈：每次 spawn 扣 20% 作为父节点保底，子返回后释放
        self._reserve_stack: list[float] = []
        # 探索投资回报率
        self.explore_roi = explore_roi
        # Beta(α, β) 后验：任务完成概率
        self.done_alpha = 1.0
        self.done_beta = 1.0
        # 子任务成功率
        self.subtask_beta: Dict[str, tuple] = {}
        # 派生成功率
        self.spawn_alpha = 2.0
        self.spawn_beta = 2.0
        # 连续无进展计数
        self._no_progress_count = 0
        # 失败计数
        self.failure_count = 0
        # 命令耗时先验 (Gamma): pattern -> (alpha_sum_time, beta_count)
        self.cmd_time_prior: Dict[str, tuple] = {}
        self.energy_per_second = Config.CMD_REFUND_PER_SEC  # 每秒退款速率
        # 命令模式连续失败计数
        self._cmd_fail_streak: Dict[str, int] = {}
        # 累积追踪（调参用，不参与决策）
        self._cumulative_input: int = 0

    # ---- 累积追踪 ----
    def track_input(self, tokens: int):
        self._cumulative_input += tokens

    @property
    def estimated_context_usage(self) -> float:
        """累积输入 token / 模型窗口 — 上下文占用估算（仅供参考）。"""
        if self.total_energy <= 0:
            return 0.0
        return min(1.0, self._cumulative_input / self.total_energy)

    # ---- 统一计费接口 ----
    def charge(self, amount: float, *, check: bool = True) -> bool:
        """扣除流动资金。check=False 时无条件扣除（用于已发生的 API 成本）。
        返回 False 表示资金不足（仅 check=True 时可能）。"""
        if check and self.energy < amount:
            return False
        self.energy -= amount
        return True

    def spend(self, amount: float):
        """确认不可逆净消耗，累加到 total_spent。"""
        self.total_spent += amount

    def credit(self, amount: float):
        """返还/奖励流动资金。"""
        self.energy += amount

    def expand_budget(self, amount: float):
        """膨胀预算上限（用于奖励），不超过安全上限。"""
        self.total_energy += amount
        max_total = Config.CONTEXT_BUDGET - Config.CONTEXT_RESERVE // 2
        if self.total_energy > max_total:
            self.total_energy = max_total

    # Token 使用统计
    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_input_tokens(self) -> int:
        return self._total_input

    @property
    def total_output_tokens(self) -> int:
        return self._total_output

    def add_tokens(self, input_t: int, output_t: int):
        self._total_tokens += input_t + output_t
        self._total_input += input_t
        self._total_output += output_t

    def update_subtask(self, sub_id: str, success: bool):
        a, b = self.subtask_beta.get(sub_id, (1.0, 1.0))
        if success:
            a += 1
            self._no_progress_count = 0
        else:
            b += 1
            self._no_progress_count += 1
        self.subtask_beta[sub_id] = (a, b)

    def update_done(self, is_done: bool):
        if is_done:
            self.done_alpha += 1
            self._no_progress_count = 0
        else:
            self.done_beta += 0.2  # 软惩罚

    def update_spawn(self, success: bool):
        if success:
            self.spawn_alpha += 1
        else:
            self.spawn_beta += 1

    def p_done(self) -> float:
        return self.done_alpha / (self.done_alpha + self.done_beta)

    def p_subtask_success(self, sub_id: str) -> float:
        a, b = self.subtask_beta.get(sub_id, (1.0, 1.0))
        return a / (a + b)

    def should_spawn(self, energy_ratio: float = 0.3) -> bool:
        p = self.spawn_alpha / (self.spawn_alpha + self.spawn_beta)
        return (p > energy_ratio * 0.5
                and self.energy > self.total_energy * energy_ratio
                and self.total_spent < self.total_energy * 0.8)

    # ── 能量感知调度 ──

    def get_role_probability(self) -> float:
        """返回 explore 概率（高能量+低进展→多探索，反之→多利用）"""
        energy_ratio = self.energy / self.total_energy if self.total_energy > 0 else 0.0
        p_done = self.p_done()
        fail_penalty = min(self._no_progress_count, 5) * 0.05
        explore_prob = energy_ratio * 0.5
        if p_done < 0.3:
            explore_prob += 0.2
        elif p_done > 0.7:
            explore_prob -= 0.2
        explore_prob -= fail_penalty
        return max(0.1, min(0.7, explore_prob))

    def should_stop(self) -> tuple:
        # 硬约束：总预算耗尽
        if self.total_spent >= self.total_energy:
            return True, f"Total budget exhausted ({self.total_spent:.1f}/{self.total_energy:.1f})"
        # 软约束：流动资金耗尽
        if self.energy <= 0:
            return True, "Working capital depleted"
        # 任务完成概率高
        if self.p_done() > 0.9:
            return True, f"Task likely done (P={self.p_done():.2f})"
        # 连续无进展
        if self._no_progress_count >= 5:
            return True, f"No progress for {self._no_progress_count} rounds"
        # 失败过多
        if self.failure_count >= 5:
            return True, f"Too many failures ({self.failure_count})"
        return False, ""

    def should_stop_with_estimator(self, estimator: StepEstimator) -> tuple:
        basic, reason = self.should_stop()
        if basic:
            return basic, reason
        # 能量前瞻：用期望值比较，给 80% 缓冲
        # 仅在已有执行数据时才前瞻（至少跑过一轮）
        if estimator and estimator.completed > 0:
            rem_mean, _, _ = estimator.predict_remaining()
            required = rem_mean * self.step_overhead * 0.8
            if self.energy < required:
                return True, f"Insufficient energy for remaining steps ({self.energy:.0f} < {required:.0f})"
        return False, ""

    # ---- 命令超时预估 ----
    def _extract_pattern(self, command: str) -> str:
        """提取命令模式：程序名 + 子命令（不含 URL/路径/文件名）"""
        parts = command.strip().split()
        if not parts:
            return "empty"
        prog = os.path.basename(parts[0]).lower()
        for ext in ('.exe', '.cmd', '.bat', '.ps1'):
            if prog.endswith(ext):
                prog = prog[:-len(ext)]
        if len(parts) < 2:
            return prog
        second = parts[1]
        # URL / 路径 / 文件 / flag → 只保留程序名
        if (second.startswith(('http://', 'https://', '-', '/', '\\', '.'))
                or '\\' in second or '/' in second
                or second.endswith(('.py', '.sh', '.js', '.json', '.txt'))):
            return prog
        # 子命令（如 npx playwright, pip install）
        return f"{prog} {second}"

    def estimate_cmd_time(self, command: str) -> tuple:
        """返回 (期望秒, 上限秒95%)，Gamma 后验 + Wilson-Hilferty 近似"""
        pattern = self._extract_pattern(command)
        alpha_sum, beta_n = self.cmd_time_prior.get(pattern, (5.0, 1.0))
        mean = alpha_sum / beta_n
        shape = alpha_sum
        if shape > 1:
            # Wilson-Hilferty 近似 Gamma 95% 分位数
            z = 1.645
            factor = 1 - 1/(9*shape) + z * math.sqrt(1/(9*shape))
            upper = (shape / beta_n) * max(0.5, factor) ** 3
        else:
            upper = mean * 3  # 数据不足，宽裕估计
        return mean, min(upper, 60.0)

    def pre_consume_for_cmd(self, command: str):
        """预扣能量（基于预估上限 + 连续失败惩罚），返回预扣量或 False"""
        _, upper = self.estimate_cmd_time(command)
        pre_cost = upper * self.energy_per_second
        pattern = self._extract_pattern(command)
        streak = self._cmd_fail_streak.get(pattern, 0)
        penalty = min(streak, 5) * 500
        total_pre = pre_cost + penalty
        return total_pre if self.charge(total_pre) else False

    def refund_for_cmd(self, command: str, pre_cost: float,
                       actual_seconds: float, success: bool) -> float:
        """根据实际耗时和成败返还能量。失败只返还 50% 差额。"""
        actual_cost = actual_seconds * self.energy_per_second
        refund_ratio = 1.0 if success else 0.5
        refund = max(0, pre_cost - actual_cost) * refund_ratio
        net = pre_cost - refund
        self.credit(refund)
        self.spend(net)
        # 更新后验
        pattern = self._extract_pattern(command)
        alpha_sum, beta_n = self.cmd_time_prior.get(pattern, (5.0, 1.0))
        self.cmd_time_prior[pattern] = (alpha_sum + actual_seconds, beta_n + 1)
        if success:
            self._cmd_fail_streak[pattern] = 0
        else:
            self._cmd_fail_streak[pattern] = self._cmd_fail_streak.get(pattern, 0) + 1
        return refund

    # ---- 投资-收益模型（spawn_agent 专用） ----
    def grant_terminal_reward(self, task_success: bool,
                              plan_complexity: int = 1,
                              actual_steps: int = 1,
                              expected_steps: float = 1.0,
                              tool_calls_count: int = 0):
        """动态终端奖励：base × 难度 × 效率 × 工具调用阶梯系数。"""
        if not task_success:
            return 0.0
        base = Config.REWARD_BASE
        difficulty = min(2.0, 1.0 + 0.3 * (plan_complexity - 1))
        spent_ratio = self.total_spent / self.total_energy if self.total_energy > 0 else 0
        efficiency = max(0.3, 1.0 - spent_ratio)
        tier = Config.REWARD_STEP_TIER
        step_mult = 1.0 if tool_calls_count <= tier else max(0.8, 1.0 - 0.02 * (tool_calls_count - tier))
        reward = base * difficulty * efficiency * step_mult
        self.credit(reward)
        self.expand_budget(reward)
        tier_label = f"tools={tool_calls_count}≤{tier}:全额" if step_mult == 1.0 else f"tools={tool_calls_count}>{tier}:×{step_mult:.2f}"
        print(f"  [Reward] +{reward:.0f}E (base={base:.0f} × diff={difficulty:.2f} × eff={efficiency:.2f} [{tier_label}])")
        return reward

    def process_event(self, event: 'SubAgentEvent'):
        """处理子 Agent 事件，更新贝叶斯后验"""
        if event.type == SubAgentEventType.STEP_COMPLETED:
            pass  # 开销已在 execute_next_step 中通过 consume_step_overhead 扣除
        elif event.type == SubAgentEventType.TOOL_FAILED:
            self.failure_count += 1
            self._no_progress_count += 1
        elif event.type == SubAgentEventType.TASK_COMPLETED:
            self.update_done(True)
        elif event.type == SubAgentEventType.FATAL_ERROR:
            self.failure_count += 1
            self._no_progress_count += 1
            self.update_done(False)
        elif event.type == SubAgentEventType.MAX_STEPS_REACHED:
            self.update_done(False)
        elif event.type == SubAgentEventType.TIMEOUT:
            self.failure_count += 1
            self.update_done(False)


# ============================================================================
# BayesianOptimizer — 在线贝叶斯优化（边用边学）
# ============================================================================

class BayesianOptimizer:
    """Thompson sampling 优化配置参数。嵌入模板引擎，每次执行自动学习。

    用法:
        opt = BayesianOptimizer("rigidity", ["rigid", "soft", "open"])
        best, idx = opt.sample()          # 选下一组配置
        opt.update(idx, success=True)     # 结果反馈
        if opt.converged(): ...           # 收敛 → 固定最优配置
    """

    def __init__(self, param_name: str, values: list, prior_strength: float = 2.0):
        self.param_name = param_name
        self.values = list(values)
        self.n = len(values)
        self._alpha = [prior_strength] * self.n    # Beta α = successes + prior
        self._beta = [prior_strength] * self.n     # Beta β = failures + prior
        self._trials = [0] * self.n

    def sample(self) -> tuple[str, int]:
        """Thompson 采样：返回 (配置值, 索引)。"""
        import random
        samples = [random.betavariate(self._alpha[i], self._beta[i])
                   for i in range(self.n)]
        best = max(range(self.n), key=lambda i: samples[i])
        return self.values[best], best

    def update(self, idx: int, success: bool):
        self._trials[idx] += 1
        if success:
            self._alpha[idx] += 1
        else:
            self._beta[idx] += 1

    def converged(self, threshold: float = 0.9) -> bool:
        means = [self._alpha[i] / (self._alpha[i] + self._beta[i])
                 for i in range(self.n)]
        best = max(range(self.n), key=lambda i: means[i])
        total = max(sum(self._trials), 1)
        confidence = means[best] * min(1.0, self._trials[best] / (total / self.n))
        return confidence >= threshold

    @property
    def best(self) -> str:
        means = [self._alpha[i] / (self._alpha[i] + self._beta[i])
                 for i in range(self.n)]
        return self.values[max(range(self.n), key=lambda i: means[i])]

    @property
    def summary(self) -> dict:
        means = [self._alpha[i] / (self._alpha[i] + self._beta[i])
                 for i in range(self.n)]
        return {
            "param": self.param_name,
            "values": self.values,
            "trials": self._trials,
            "rates": [round(m, 3) for m in means],
            "best": self.best,
            "converged": self.converged(),
        }

