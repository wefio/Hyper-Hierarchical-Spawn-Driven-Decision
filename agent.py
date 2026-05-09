from __future__ import annotations
import sys
import io
import os
import json
import time
import random
import math
import re
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Optional, Dict, Any, List
from pathlib import Path

import anthropic
import httpx

from config import Config, smart_truncate, _PLUGINS
from agent_memory_frame import StackFrame, SavepointMeta
from skill import Skill, SkillManager

from agent_kernel_glue import (APIClient, AnthropicCacheProvider, CacheProvider,
    FlowchartRecorder, AgentEventBus, _get_event_bus, SavepointManager,
    PlanState, PlanContext, SubAgentEventType, SubAgentEvent)
from agent_process_executor import (ToolExecutor, TOOL_DEFINITIONS,
    _build_cached_tools, energy_hooks, _spawn_deduct, _spawn_settle,
    _cmd_deduct, _cmd_refund, _skill_deduct)
from agent_scheduler_energy import BayesianEnergyManager, StepEstimator, ContextTooLongError
from agent_memory_l2 import L2Cache
from agent_memory_report import NodeReport
from agent_kernel_router import get_router

_http_client = httpx.Client(proxy=None, timeout=60.0, follow_redirects=True, trust_env=False)

def measure_stack(stack) -> int:
    return sum(len(f.content) for f in stack)

# ============ Agent ============
class Agent:
    def __init__(self, system_prompt: str = "你是一个AI编程助手。",
                 work_dir: str = None,
                 skills: Optional[List[str]] = None,
                 auto_skill: bool = False,
                 depth: int = 0,
                 parent: 'Agent' = None,
                 energy_mgr: 'BayesianEnergyManager' = None,
                 l2_cache: 'L2Cache' = None,
                 model_spec: dict = None,
                 can_spawn: bool = True,
                 context_budget: int = None):
        self.work_dir = work_dir or Config.WORK_DIR
        os.makedirs(self.work_dir, exist_ok=True)
        self.depth = depth
        self.agent_id = f"agent_{random.randint(1000,9999)}"
        self.parent = parent
        self.l2_cache = l2_cache
        self.can_spawn = can_spawn
        self.child_model_spec = None
        self.intervention_budget = int(os.getenv("INTERVENTION_BUDGET", "3"))  # 人类干预预算
        self.can_prune = True  # 允许剪枝（机械+未来 LLM 剪枝的权限开关）

        # ── 模型路由 ──
        router = get_router()
        cb = context_budget or Config.CONTEXT_BUDGET
        self._model_id = "default"
        if model_spec and (model_spec.get("id") or model_spec.get("tier") or model_spec.get("tags")):
            sticky = f"{getattr(parent, 'agent_id', '')}:{self.agent_id}"
            res = router.resolve_for_agent(model_spec, sticky_key=sticky,
                                           required_context=min(cb, 8000))
            self.client = res["client"]
            cb = res["context_budget"]
            self.can_spawn = res["can_spawn"] and can_spawn
            self._model_id = res["model_id"]
            self._model_spec = res.get("model_spec")
            self._extra_params = res.get("extra_params", {})
        else:
            self.client = anthropic.Anthropic(
                api_key=Config.ANTHROPIC_API_KEY,
                base_url=Config.ANTHROPIC_BASE_URL,
                http_client=_http_client,
            )
            self._extra_params = {}
        self._router = router
        self._context_budget = cb

        # Skill 管理（先于工具加载，放在稳定前缀中）
        self.skill_manager = SkillManager()
        self.active_skills: list[Skill] = []
        if skills:
            for name in skills:
                skill = self.skill_manager.get(name)
                if skill:
                    self.active_skills.append(skill)
                else:
                    print(f"  [Warning] Skill not found: {name}")
        self.auto_skill = auto_skill

        # 构建系统约束：Skill 在前（稳定），工具说明在后
        skill_section = ""
        if self.active_skills:
            skill_section = "\n\n" + "\n\n".join(f"## Skill: {s.name}\n{s.content}" for s in self.active_skills)

        self.stack: deque[StackFrame] = deque()
        self.stack.append(StackFrame("constraint", system_prompt + skill_section, level=0))
        self.stack.append(StackFrame("plan", "Pending plan...", level=1))

        self._plan = PlanContext()
        self.step_counter = 0
        self.max_tool_rounds = Config.MAX_TOOL_ROUNDS  # 默认安全网，下面能量初始化后重算
        self.subtask_queue: list[dict] = []
        self.conversation_history: list[dict] = []
        self.step_estimator: Optional[StepEstimator] = None
        # 缓存 provider（子 agent 继承父级）
        self._cache_provider = parent._cache_provider if parent else AnthropicCacheProvider()
        # 启动时间（固定写入 system prompt，避免每轮变化破坏缓存）
        self.start_time = parent.start_time if parent else None
        if self.start_time is None:
            from datetime import datetime
            self.start_time = datetime.now().strftime("%Y-%m-%d %p").replace("AM", "上午").replace("PM", "下午")
        # 能量管理：优先用传入的独立实例（并行 spawn），其次共享父级，顶层新建
        # 使用模型 context 窗口动态计算总能量，保留值按窗口比例（默认 25%）
        reserve = min(Config.CONTEXT_RESERVE, int(self._context_budget * Config.CONTEXT_RESERVE_RATIO))
        total_e = max(self._context_budget - reserve, 4000)
        if energy_mgr is not None:
            self.energy_manager = energy_mgr
        elif parent:
            self.energy_manager = parent.energy_manager
        else:
            self.energy_manager = BayesianEnergyManager(total_energy=total_e)
        # ── Pointer 磁盘缓存系统 ──
        # Scope 链：每个 agent 继承父 scope + 自己的 id
        self.scope = f"{parent.scope}.{self.agent_id}" if parent and hasattr(parent, 'scope') else "root"
        # PointerStore：CoW 共享（同一 agent 树共享同一实例）
        if parent and hasattr(parent, 'pointer_store'):
            self.pointer_store = parent.pointer_store
        else:
            from pointer_store import PointerStore
            self.pointer_store = PointerStore(Config.ARCHIVE_DIR, self.agent_id, self.scope)
        # per-agent 工具轮数上限，跟能量预算挂钩
        self.max_tool_rounds = max(5, int(self.energy_manager.energy / 3000))
        # API 客户端
        self._api = APIClient(self.client, Config, self._cache_provider, self.energy_manager,
                              model_id=self._model_id,
                              model_spec=getattr(self, '_model_spec', None))
        self._api._interventions_left = self.intervention_budget
        # 事件系统
        self.event_log: list[SubAgentEvent] = []
        # 节点报告
        self.reports_history: list[NodeReport] = []
        self._current_report: Optional[NodeReport] = None
        # 流程图记录器：子 Agent 复用父级实例，避免覆盖
        if parent and hasattr(parent, 'flowchart') and parent.flowchart:
            self.flowchart = parent.flowchart
        else:
            self.flowchart = FlowchartRecorder(self.work_dir)
        # 共享文件缓存：父 agent 传递已完成的文件路径，子 agent 可直接读
        self.shared_files: dict[str, str] = {}  # {relative_path: absolute_path}
        if parent and hasattr(parent, 'shared_files'):
            self.shared_files = dict(parent.shared_files)
        # 共享缓存目录
        self._cache_dir = os.path.join(self.work_dir, "cache")
        os.makedirs(self._cache_dir, exist_ok=True)
        # 记录启动节点（仅顶层 Agent）
        if not parent:
            try:
                self.flowchart.add_node("start", "Agent 启动", shape="start")
            except Exception:
                pass

        # 子任务验证反馈：跨 subtask 累积，注入下一步 prompt
        self._verify_feedback: list[str] = []
        # 验证上下文注入：tool 执行后等待注入 LLM 上下文
        self._pending_verify_feedback: str = ""
        # Tool 调用记录：当前任务中所有 tool 调用的序列（供技能保存使用）
        self._current_tool_calls: list[dict] = []
        # 事件总线初始化 + 插件钩子
        if not parent:
            bus = _get_event_bus()
            for plugin in _PLUGINS:
                plugin.on_agent_init(self, bus)
        # 经验元数据：仅根 Agent 从文件加载，子 Agent 从空开始（避免继承兄弟姐妹的经验导致指数爆炸）
        self._pending_experiences: list[dict] = []
        if not parent:
            self._load_pending_experiences()

        # 存档点系统
        self.active_savepoint: Optional[SavepointMeta] = None
        self.savepoint_history: list[SavepointMeta] = []
        self._just_popped = False  # pop 后中断当前 tool loop

    # ---- 状态持久化 ----
    @classmethod
    def load_state(cls, work_dir: str = None):
        work_dir = work_dir or Config.WORK_DIR
        state_file = os.path.join(work_dir, "state.json")
        if not os.path.exists(state_file):
            raise FileNotFoundError(f"State file not found: {state_file}")
        with open(state_file, "r", encoding='utf-8') as f:
            state = json.load(f)
        agent = cls.__new__(cls)
        agent.work_dir = work_dir
        agent.stack = deque([StackFrame(**frame) for frame in state["stack"]])
        agent.step_counter = state["step_counter"]
        agent.subtask_queue = state.get("subtask_queue", [])
        agent.conversation_history = state.get("conversation_history", [])
        agent._pending_experiences = state.get("_pending_experiences", [])
        agent.depth = 0
        agent.agent_id = f"agent_{random.randint(1000,9999)}"
        agent.parent = None
        agent.event_log = []
        agent.energy_manager = BayesianEnergyManager()
        agent.max_tool_rounds = max(5, int(agent.energy_manager.energy / 3000))
        agent.step_estimator = None
        agent._cache_provider = AnthropicCacheProvider()
        agent.client = anthropic.Anthropic(
            api_key=Config.ANTHROPIC_API_KEY,
            base_url=Config.ANTHROPIC_BASE_URL,
            http_client=_http_client,
        )
        agent._api = APIClient(agent.client, Config, agent._cache_provider, agent.energy_manager)
        agent.skill_manager = SkillManager()
        agent.active_skills = []
        agent.auto_skill = False
        agent._load_pending_experiences()
        return agent

    def _save_state(self):
        state = {
            "stack": [asdict(f) for f in self.stack],
            "step_counter": self.step_counter,
            "subtask_queue": self.subtask_queue,
            "conversation_history": self.conversation_history,
            "_pending_experiences": self._pending_experiences,
        }
        path = os.path.join(self.work_dir, "state.json")
        with open(path, "w", encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def _load_pending_experiences(self):
        path = os.path.join(self.work_dir, "pending_exp.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self._pending_experiences = json.load(f)
                # 安全上限：超过200条截断到最近200条
                if len(self._pending_experiences) > 200:
                    print(f"  [Experience] truncating loaded {len(self._pending_experiences)} -> 200 records")
                    self._pending_experiences = self._pending_experiences[-200:]
            except (json.JSONDecodeError, Exception):
                self._pending_experiences = []
        else:
            self._pending_experiences = []

    def _save_pending_experiences(self):
        path = os.path.join(self.work_dir, "pending_exp.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._pending_experiences, f, ensure_ascii=False, indent=2)

    def _inject_experience(self, task: str):
        """从经验库搜索相关经验，注入到历史帧"""
        try:
            from experience_store import ExperienceStore
            store = ExperienceStore()
            results = store.search(task, limit=3)
            if not results:
                return
            parts = []
            for r in results:
                status = "OK" if r["success"] else "FAIL"
                line = f"- [{status}] {r['task'][:60]} -> {r['summary'][:100] if r['summary'] else '(no summary)'}"
                if r.get("lessons"):
                    line += f"\n  Lesson: {r['lessons'][:120]}"
                parts.append(line)
            exp_text = "## Relevant Past Experience\n" + "\n".join(parts)
            # 替换或追加到 history 帧
            self.stack = deque(f for f in self.stack if f.type != "experience")
            self.stack.append(StackFrame("experience", exp_text, level=0))
            print(f"  [Experience] injected {len(results)} relevant records")
            # 流程图：经验注入
            if self.flowchart and not self.parent:
                try:
                    self.flowchart.add_node("exp_inject", f"注入 {len(results)} 条经验", shape="energy")
                    self.flowchart.add_edge("start", "exp_inject", label="经验")
                except Exception:
                    pass

            # 同时注入技能库建议
            skills = store.search_skills(task, limit=2)
            if skills:
                skill_lines = ["## Skill Suggestions (from experience store)"]
                for sk in skills:
                    tag = "[内置]" if sk.get("is_builtin") else "[学习]"
                    skill_lines.append(f"\n### {tag} {sk['name']} (w={sk.get('weight', 1.0):.0%})")
                    seq = sk.get("tool_sequence", [])
                    if seq:
                        for step in seq:
                            t = step.get("tool", "?")
                            a = step.get("action", "?")
                            h = step.get("hint", "")
                            skill_lines.append(f"  {step.get('step', '?')}. {t}(action=\"{a}\") — {h}")
                skill_text = "\n".join(skill_lines)
                self.stack.append(StackFrame("skill", skill_text, level=0))
                print(f"  [Experience] injected {len(skills)} skills")
        except Exception as e:
            pass  # 经验库不可用不应阻塞执行

    def _save_history(self, task: str, result: str, steps: int, success: bool):
        """保存运行结果到 history/ 目录"""
        try:
            hist_dir = Path(Config.HISTORY_DIR)
            hist_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = hist_dir / f"{ts}.md"
            status = "SUCCESS" if success else "PARTIAL"
            content = f"# {task[:80]}\n\n"
            content += f"- **Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            content += f"- **Steps**: {steps}\n"
            content += f"- **Status**: {status}\n"
            content += f"- **Depth**: {self.depth}\n\n"
            content += f"## Result\n\n{result or '(no result)'}\n"
            fname.write_text(content, encoding='utf-8')
        except Exception:
            pass

    def _report(self, severity: str, category: str, message: str, data: dict = None):
        """结构化事件上报（能量告警/空响应/spawn失败/异常吞没等）。"""
        ts = time.strftime("%H:%M:%S")
        data_str = f" {json.dumps(data, ensure_ascii=False)}" if data else ""
        print(f"  [{severity.upper()}] [{category}] {ts} {message}{data_str}")
        self.event_log.append(SubAgentEvent(
            agent_id=self.agent_id,
            type=SubAgentEventType.FATAL_ERROR if severity == "error" else SubAgentEventType.STEP_COMPLETED,
            message=f"[{category}] {message}",
            data=data or {},
        ))

    def emit_event(self, event_type: SubAgentEventType, message: str = "", data: dict = None):
        event = SubAgentEvent(self.agent_id, event_type, message, data)
        self.event_log.append(event)
        # 能量管理器处理事件
        if self.energy_manager:
            self.energy_manager.process_event(event)

    def _collect_experience_metadata(self, task: str, plan: str, step_count: int, success: bool, final_result: str):
        """轻量收集经验元数据，不调用 LLM"""
        last_summary = ""
        for f in reversed(self.stack):
            if f.type == "summary" and f.level == 2:
                last_summary = f.content
                break
        self._pending_experiences.append({
            "task": task,
            "plan": plan,
            "step_count": step_count,
            "success": success,
            "last_summary": last_summary[:500],
            "final_result_preview": (final_result or "")[:300],
            "timestamp": time.time(),
            "tool_calls": list(self._current_tool_calls),
        })
        self._save_pending_experiences()

    def flush_experiences(self):
        """批量处理待处理经验：仅对步数>=3或失败的调用 LLM 提取教训"""
        if not self._pending_experiences:
            return
        # 安全上限：超过100条仅保留最近100条
        if len(self._pending_experiences) > 100:
            print(f"  [Experience] truncating {len(self._pending_experiences)} -> 100 records")
            self._pending_experiences = self._pending_experiences[-100:]
        try:
            from experience_store import ExperienceStore
            store = ExperienceStore()
            count = 0
            for exp in self._pending_experiences:
                # 优先用栈摘要，空则用结果预览
                summary = exp["last_summary"] or exp["final_result_preview"]
                # 仅失败或复杂任务才提取教训（节省 API 调用）
                lesson = ""
                if not exp["success"] or exp["step_count"] >= 3:
                    lesson = self.extract_lesson_from_text(summary)
                store.record(
                    task=exp["task"], plan=exp["plan"], summary=summary,
                    lessons=lesson, step_count=exp["step_count"], success=exp["success"],
                    tool_calls=exp.get("tool_calls", [])
                )
                # 经验权重更新
                store.update_weights(exp["task"], exp["plan"], exp["success"])
                count += 1
            self._pending_experiences = []
            self._save_pending_experiences()
            print(f"  [Experience] flushed {count} records")
        except Exception as e:
            print(f"  [Experience] flush failed: {e}")

    def extract_lesson_from_text(self, text: str) -> str:
        """从文本中提取教训（调用 LLM，仅复杂任务触发）"""
        if not text or len(text) < 20:
            return ""
        response = self._api.call(
            [{"role": "user", "content":
                f"从以下任务结果中提取一个可复用的经验教训（100字以内）：\n\n{text}"}],
            system="你是一个经验总结助手。只输出经验教训。",
            max_tokens=200
        )
        return self._extract_text(response)

    def _extract_text(self, response) -> str:
        """从 Message 对象提取文本"""
        return "\n".join(b.text for b in response.content if hasattr(b, 'text'))

    # ---- 自适应上下文管理 ----
    def _count_tokens(self, messages: list, system_blocks: list = None) -> int:
        """精确 token 计数。SDK 支持 -> count_tokens()，否则 -> char/4 估算。"""
        try:
            if hasattr(self.client, 'messages') and hasattr(self.client.messages, 'count_tokens'):
                params = {"messages": messages, "model": getattr(self, '_model_name', Config.MODEL)}
                if system_blocks:
                    params["system"] = system_blocks
                result = self.client.messages.count_tokens(**params)
                return getattr(result, 'input_tokens', 0)
        except Exception:
            pass
        # Fallback: 估算
        print(f"  [Token] count_tokens() unavailable, using char/4 estimate")
        total = 0
        for m in messages:
            c = m.get("content", "")
            if isinstance(c, str):
                total += len(c)
            elif isinstance(c, list):
                total += len(json.dumps(c, ensure_ascii=False))
        if system_blocks:
            for s in system_blocks:
                total += len(s.get("text", ""))
        return total // 4

    def _estimate_context_length(self) -> int:
        """估算当前 context 的 token 数。优先用 API 精确计数，回退到本地估算。"""
        try:
            msgs, sys = self._build_messages()
            return self._count_tokens(msgs, sys)
        except Exception:
            print(f"  [Token] _build_messages failed for estimate, using raw stack char/4")
        # 无法构建消息时的粗略估算
        return sum(len(f.content) for f in self.stack) // 4

    def _compress(self, text: str, step_id: int) -> str:
        # ── STORE：归档完整内容到磁盘 ──
        pid = ""
        if len(text) > 200:
            task_desc = self.subtask_queue[0]['desc'][:60] if self.subtask_queue else "unknown"
            pid = self.pointer_store.store(
                text, task=task_desc, level=0,
                frame_type="step_detail", step_id=step_id,
                parent_scope=self.scope,
            )
            if pid:
                tokens = len(text) // 4
                self.energy_manager.credit(tokens)
                print(f"  [Archive] {pid}: {tokens}E refunded ({len(text)} chars)")
                # 流程图：归档节点
                if self.flowchart:
                    try:
                        node_id = f"ptr_store_{self.depth}_{self.agent_id}_{step_id}"
                        self.flowchart.add_node(node_id, f"STORE {pid} (+{tokens}E)", shape="pointer")
                        step_ref = f"s{self.depth}_{self.agent_id}_{step_id}"
                        self.flowchart.add_edge(step_ref, node_id, label="归档")
                    except Exception:
                        pass
        # ── LLM 压缩 ──
        prompt = f"将以下内容压缩为一句话摘要（不超过100字），保留关键动作和结果：\n\n{text}"
        response = self._api.call(
            [{"role": "user", "content": prompt}],
            system="你是一个摘要助手。只输出摘要。",
            max_tokens=300,
            bypass_energy=True
        )
        summary = self._extract_text(response)
        if pid:
            return f"[Step {step_id}] 摘要:{pid} — {summary}"
        return f"[Step {step_id}]: {summary}"

    # ---- 能量回收：弹出栈顶帧 ----
    def _reclaim_energy(self, target_amount: float) -> float:
        """从栈中弹出帧回收能量。策略：
        0. 优先驱逐不活跃存档点（历史记录，按 context_size 降序）
        1. 回收 reclaimable 帧（recall 借阅内容）
        2. 弹空壳子 agent 结果（内容 ≤ 50 chars，无实质内容）
        3. 弹 summary/merge 帧（use_count=0 的最久未用项优先）
        4. 最后弹 step_detail（损失最大）
        不弹 constraint/plan/history/experience。
        弹出前 STORE 到磁盘（保留 pointer stub），返还能量。"""
        reclaimed = 0.0
        MIN_STACK = 3  # 至少保留 3 帧
        _reclaim_seq = 0  # 用于生成唯一节点 ID

        def _evict_frame(frame, idx):
            """STORE 再弹出（或替换为 pointer stub）。"""
            nonlocal reclaimed, _reclaim_seq
            estimated_tokens = len(frame.content) / 4
            # reclaimable 帧：无需 STORE（已在磁盘），直接删除
            if getattr(frame, 'reclaimable', False):
                del self.stack[idx]
                reclaimed += estimated_tokens
                print(f"  [Reclaim] Reclaimable {frame.type} Step {frame.step_id} "
                      f"({len(frame.content)} chars ≈ {estimated_tokens:.0f}E)")
                return
            # 已是 pointer stub 或内容极小：直接删除
            if frame.type == "pointer" or len(frame.content) <= 50:
                del self.stack[idx]
                reclaimed += estimated_tokens
                print(f"  [Reclaim] Popped {frame.type} Step {frame.step_id} "
                      f"({len(frame.content)} chars ≈ {estimated_tokens:.0f}E)")
                return
            # 实质性内容：STORE 到磁盘，留 pointer stub
            task_desc = self.subtask_queue[0]['desc'][:60] if self.subtask_queue else "unknown"
            pid = self.pointer_store.store(
                frame.content, task=task_desc, level=0,
                frame_type=frame.type, step_id=frame.step_id,
                parent_scope=self.scope,
            )
            if pid:
                tokens = len(frame.content) // 4
                self.energy_manager.credit(tokens)
                stub = StackFrame(
                    "pointer", f"摘要:{pid}",
                    step_id=frame.step_id, level=2,
                    agent_id=frame.agent_id,
                    pointer_id=pid, reclaimable=False,
                    use_count=getattr(frame, 'use_count', 0),
                    last_used_step=self.step_counter,
                )
                self.stack[idx] = stub
                reclaimed += estimated_tokens
                print(f"  [Reclaim] Archived {frame.type} Step {frame.step_id} -> {pid} "
                      f"({len(frame.content)} chars ≈ {estimated_tokens:.0f}E, +{tokens}E refunded)")
            else:
                # STORE 失败，回退到直接删除
                del self.stack[idx]
                reclaimed += estimated_tokens
                print(f"  [Reclaim] Popped {frame.type} Step {frame.step_id} "
                      f"(STORE failed, {len(frame.content)} chars ≈ {estimated_tokens:.0f}E)")
            # 记录到流程图
            if self.flowchart:
                try:
                    _reclaim_seq += 1
                    node_id = f"reclaim_d{self.depth}_{_reclaim_seq}"
                    label = f"弹出 {frame.type} (≈{estimated_tokens:.0f}E)"
                    self.flowchart.add_node(node_id, label, shape="reclaim")
                    aid = frame.agent_id or self.agent_id
                    step_ref = f"s{self.depth}_{aid}_{frame.step_id}" if frame.step_id else None
                    if step_ref:
                        self.flowchart.add_edge(step_ref, node_id, label=f"-{estimated_tokens:.0f}E")
                    # 如果 STORE 成功，额外记录 pointer 节点
                    if pid:
                        ptr_node = f"ptr_reclaim_d{self.depth}_{_reclaim_seq}"
                        self.flowchart.add_node(ptr_node, f"STORE {pid} (+{tokens}E)", shape="pointer")
                        self.flowchart.add_edge(node_id, ptr_node, label="归档")
                except Exception:
                    pass

        # Phase 0: 驱逐不活跃存档点（按 context_size 降序，优先回收最大的）
        if reclaimed < target_amount and hasattr(self, 'savepoint_history') and self.savepoint_history:
            for sp in sorted(self.savepoint_history, key=lambda s: s.context_size, reverse=True):
                if reclaimed >= target_amount:
                    break
                if sp.status == "committed" and sp.context_size > 0:
                    reclaimed += sp.context_size
                    self.savepoint_history.remove(sp)
                    print(f"  [Reclaim] Evicted savepoint '{sp.name}' "
                          f"(ctx≈{sp.context_size}E, committed summary: {sp.summary[:50]})")
                    if self.flowchart:
                        try:
                            _reclaim_seq += 1
                            node_id = f"reclaim_d{self.depth}_{_reclaim_seq}"
                            self.flowchart.add_node(node_id, f"驱逐存档点 {sp.name} (≈{sp.context_size}E)", shape="reclaim")
                        except Exception:
                            pass

        # Phase 1: 回收 reclaimable 帧（recall 借阅内容，优先释放）
        if reclaimed < target_amount:
            for i in range(len(self.stack) - 1, -1, -1):
                if len(self.stack) <= MIN_STACK:
                    break
                f = self.stack[i]
                if getattr(f, 'reclaimable', False):
                    _evict_frame(f, i)
                    if reclaimed >= target_amount:
                        break

        # Phase 2: 弹空壳（≤50 chars 的 step_detail/summary）
        if reclaimed < target_amount:
            for i in range(len(self.stack) - 1, -1, -1):
                if len(self.stack) <= MIN_STACK:
                    break
                f = self.stack[i]
                if f.type in ("step_detail", "summary") and len(f.content) <= 50:
                    _evict_frame(f, i)
                    if reclaimed >= target_amount:
                        break

        # Phase 3: 弹 summary/merge（use_count=0 的最久未用优先）
        if reclaimed < target_amount:
            # 按 use_count 升序 + last_used_step 升序排序候选帧
            candidates = [(i, f) for i, f in enumerate(self.stack)
                          if f.type in ("summary", "merge") and f.level >= 2]
            candidates.sort(key=lambda x: (getattr(x[1], 'use_count', 0),
                                           getattr(x[1], 'last_used_step', 0)))
            for i, f in candidates:
                if len(self.stack) <= MIN_STACK:
                    break
                if reclaimed >= target_amount:
                    break
                _evict_frame(f, i)

        # Phase 4: 弹 step_detail（信息损失最大，最后手段）
        if reclaimed < target_amount:
            candidates = [(i, f) for i, f in enumerate(self.stack)
                          if f.type == "step_detail" and f.level >= 2]
            candidates.sort(key=lambda x: (getattr(x[1], 'use_count', 0),
                                           getattr(x[1], 'last_used_step', 0)))
            for i, f in candidates:
                if len(self.stack) <= MIN_STACK:
                    break
                if reclaimed >= target_amount:
                    break
                _evict_frame(f, i)

        if reclaimed > 0:
            self.energy_manager.energy += reclaimed
            print(f"  [Reclaim] Total recovered: {reclaimed:.0f}E")
        return reclaimed

    # ---- 消息总量截断 ----
    def _trim_messages(self, messages: list, max_chars: int) -> list:
        """截断 messages 列表使其总字符数低于 max_chars。
        策略：从最旧的 user 消息（通常是工具结果）开始截断内容。"""
        total = sum(len(str(m)) for m in messages)
        if total <= max_chars:
            return messages
        for i, msg in enumerate(messages):
            if total <= max_chars:
                break
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 500:
                old_len = len(content)
                content = content[:500] + f"\n... [truncated from {old_len} chars]"
                messages[i] = {**msg, "content": content}
                total -= (old_len - len(content))
            elif isinstance(content, list):
                # tool_result 格式: [{"type": "tool_result", "content": "..."}]
                for j, item in enumerate(content):
                    if isinstance(item, dict) and len(str(item.get("content", ""))) > 500:
                        old_len = len(str(item["content"]))
                        item["content"] = str(item["content"])[:500] + f"\n... [truncated from {old_len}]"
                        total -= (old_len - 500)
            if total <= max_chars:
                break
        print(f"  [Trim] Messages trimmed to ~{total} chars (limit {max_chars})")
        return messages

    # ---- 子 Agent 结果吸收 ----
    def _make_summary(self, text: str, max_len: int = 300) -> str:
        """不调 LLM 的快速摘要：取首尾 + 元数据"""
        if len(text) <= max_len:
            return text
        head = text[:max_len // 2]
        tail = text[-(max_len // 3):]
        return f"{head}\n... [省略 {len(text) - len(head) - len(tail)} 字] ...\n{tail}"

    def _build_failure_experience(self, max_items: int = 5) -> str:
        """从栈帧中提取失败经验，传递给子 Agent。"""
        failures = []
        # 从最近栈帧中找包含 error/失败 的内容
        for frame in list(self.stack)[-10:]:
            content = frame.content[:300]
            if any(kw in content.lower() for kw in ["error", "失败", "fail", "not found", "不存在"]):
                failures.append(content)
        if not failures:
            return ""
        lines = failures[-max_items:]
        exp = "## 父级执行经验（请避免重复以下失败操作）\n"
        for i, f in enumerate(lines):
            exp += f"{i+1}. {f}\n"
        exp += "\n建议：使用不同的方法或命令重试。\n"
        return exp

    def _absorb_child_result(self, child_result: str, subtask_desc: str,
                              child_agent=None):
        """吸收子 Agent 结果。

        预算 = 父 freeze / 3 + 子剩余 energy * 0.3（对齐窗口模型）
        """
        em = self.energy_manager
        # 子剩余 energy（从子获取，或估算）
        child_remaining = getattr(child_agent, 'energy_manager', None)
        if child_remaining and hasattr(child_remaining, 'energy'):
            child_remaining = child_remaining.energy
        else:
            child_remaining = em.energy * 0.5  # 估算
        # 父 freeze：从子携带的 spawn 参数获取
        freeze = getattr(child_agent, '_spawn_freeze', 0) if child_agent else 0
        if freeze <= 0:
            freeze = em.energy * Config.SPAWN_RESERVE_RATIO  # 估算
        budget_tokens = int(freeze / 3 + child_remaining * 0.30)
        budget_chars = max(200, budget_tokens // 4)
        is_full = len(child_result) <= budget_chars
        if is_full:
            self.stack.append(StackFrame("step_detail", child_result, self.step_counter, level=2, agent_id=self.agent_id))
            print(f"  [Absorb] Full ({len(child_result)}c <= budget {budget_chars}c, freeze={freeze:.0f}E child_rem={child_remaining:.0f}E)")
        else:
            summary = self._make_summary(child_result, max_len=min(budget_chars, 500))
            self.stack.append(StackFrame("summary", summary, self.step_counter, level=2, agent_id=self.agent_id))
            print(f"  [Absorb] Compressed ({len(summary)}/{len(child_result)}c, budget {budget_chars}c)")
        # 生命周期计数器：记录吸收类型（不再创建单独节点）
        if self.flowchart:
            try:
                child_depth = self.depth + 1
                self.flowchart.record_lifecycle(child_depth,
                    absorb_full=1 if is_full else 0,
                    absorb_summary=0 if is_full else 1)
            except Exception:
                pass

    # ---- 子 Agent 上下文摘要 ----
    def _summarize_via_child(self, messages: list, task_hint: str = "") -> str:
        """将完整上下文注入子 Agent 生成结构化摘要。
        子 Agent 拥有独立上下文，摘要后 parent 只保留摘要文本。
        成本：子 Agent 一次 API 调用（含全量上下文），parent 后续每轮省 ~full_context tokens。"""
        # 提取 messages 中的关键信息构建摘要输入
        context_parts = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            if isinstance(content, str) and content:
                context_parts.append(f"[{role}]\n{content}")
            elif isinstance(content, list):
                # tool_result 格式
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("content", "")
                        if text:
                            context_parts.append(f"[tool_result]\n{text}")
        full_context = "\n\n---\n\n".join(context_parts)

        if not full_context.strip():
            return ""

        # 构建摘要 prompt
        prompt = (
            f"以下是某个任务执行过程中积累的全部上下文。请生成结构化摘要（不超过800字），包含：\n"
            f"1. 已完成的关键操作和结果\n"
            f"2. 已获取的重要信息\n"
            f"3. 当前进展状态\n"
            f"4. 待解决的剩余问题\n"
        )
        if task_hint:
            prompt += f"\n重点关注与「{task_hint}」相关的信息。\n"
        prompt += f"\n--- 原始上下文 ---\n{full_context}"

        # 创建极简子 Agent，不做工具调用，纯摘要
        child = Agent(
            system_prompt="你是摘要助手。只输出结构化摘要，不超过800字。不要输出其他内容。",
            depth=self.depth + 1,
            parent=self
        )
        child.max_tool_rounds = 0  # 不允许工具调用
        try:
            summary = child.run(prompt, max_steps=1)
            # 子 Agent 的栈已销毁，只有 summary 文本返回
            summary_len = len(summary) if summary else 0
            print(f"  [ChildSummary] {len(full_context)} chars -> {summary_len} chars "
                  f"(saved ~{len(full_context) // 4} tokens/round)")
            return summary or ""
        except Exception as e:
            print(f"  [ChildSummary] Failed: {e}")
            # 降级为简单摘要
            return self._make_summary(full_context, max_len=1000)

    def _adapt_context(self):
        threshold = Config.CONTEXT_BUDGET * Config.COMPRESSION_THRESHOLD
        _compress_seq = 0
        # Phase 0: 优先回收 reclaimable 帧（recall 借阅的内容）
        while True:
            est = self._estimate_context_length()
            if est < threshold:
                return
            reclaimed_any = False
            for i, frame in enumerate(self.stack):
                if getattr(frame, 'reclaimable', False) and len(self.stack) > 3:
                    # reclaimable 帧无需再 STORE（数据已在磁盘）
                    del self.stack[i]
                    reclaimed_tokens = len(frame.content) // 4
                    print(f"  [ReclaimReclaimable] Step {frame.step_id}: {reclaimed_tokens}E freed ({est}/{Config.CONTEXT_BUDGET})")
                    reclaimed_any = True
                    break
            if not reclaimed_any:
                break
        # Phase 1: 循环压缩 step_detail -> STORE + _compress -> summary
        while True:
            est = self._estimate_context_length()
            if est < threshold:
                return
            compressed_any = False
            for i, frame in enumerate(self.stack):
                if frame.type == "step_detail" and frame.level >= 2:
                    compressed = self._compress(frame.content, frame.step_id)
                    print(f"  [Compress] Step {frame.step_id}: {len(frame.content)} -> {len(compressed)} chars ({est}/{Config.CONTEXT_BUDGET})")
                    self.stack[i] = StackFrame("summary", compressed, frame.step_id, level=2, agent_id=frame.agent_id)
                    # 记录到流程图
                    if self.flowchart:
                        try:
                            _compress_seq += 1
                            node_id = f"compress_d{self.depth}_{_compress_seq}"
                            label = f"压缩 Step {frame.step_id} ({len(frame.content)}→{len(compressed)})"
                            self.flowchart.add_node(node_id, label, shape="reclaim")
                            aid = frame.agent_id or self.agent_id
                            step_ref = f"s{self.depth}_{aid}_{frame.step_id}" if frame.step_id else None
                            if step_ref:
                                self.flowchart.add_edge(step_ref, node_id, label="压缩")
                        except Exception:
                            pass
                    compressed_any = True
                    break  # 每次只压缩一帧，重新评估
            if not compressed_any:
                break
        # Phase 2: 截断 — STORE 后替换为 pointer stub（非删除）
        est = self._estimate_context_length()
        if est > Config.CONTEXT_BUDGET:
            for i, frame in enumerate(self.stack):
                if frame.level >= 2:
                    if len(frame.content) > 50 and frame.type != "pointer":
                        # STORE 到磁盘，留 pointer stub
                        task_desc = self.subtask_queue[0]['desc'][:60] if self.subtask_queue else "unknown"
                        pid = self.pointer_store.store(
                            frame.content, task=task_desc, level=0,
                            frame_type=frame.type, step_id=frame.step_id,
                            parent_scope=self.scope,
                        )
                        if pid:
                            tokens = len(frame.content) // 4
                            self.energy_manager.credit(tokens)
                            stub = StackFrame(
                                "pointer", f"摘要:{pid}",
                                step_id=frame.step_id, level=2,
                                agent_id=frame.agent_id,
                                pointer_id=pid, reclaimable=False,
                                use_count=getattr(frame, 'use_count', 0),
                                last_used_step=self.step_counter,
                            )
                            self.stack[i] = stub
                            print(f"  [Truncate] Step {frame.step_id}: archived as {pid}, {tokens}E refunded ({est}/{Config.CONTEXT_BUDGET})")
                        else:
                            # STORE 失败，回退到直接删除
                            del self.stack[i]
                            print(f"  [Truncate] Step {frame.step_id}: delete (STORE failed) ({est}/{Config.CONTEXT_BUDGET})")
                    else:
                        # pointer 或极小帧，直接删除
                        del self.stack[i]
                        print(f"  [Truncate] Step {frame.step_id}: remove pointer stub ({est}/{Config.CONTEXT_BUDGET})")
                    # 记录到流程图
                    if self.flowchart:
                        try:
                            _compress_seq += 1
                            node_id = f"truncate_d{self.depth}_{_compress_seq}"
                            label = f"截断 Step {frame.step_id}"
                            self.flowchart.add_node(node_id, label, shape="truncate")
                            aid = frame.agent_id or self.agent_id
                            step_ref = f"s{self.depth}_{aid}_{frame.step_id}" if frame.step_id else None
                            if step_ref:
                                self.flowchart.add_edge(step_ref, node_id, label="截断")
                            # 如果 STORE 成功，额外记录 pointer 节点
                            if pid:
                                ptr_node = f"ptr_trunc_d{self.depth}_{_compress_seq}"
                                self.flowchart.add_node(ptr_node, f"STORE {pid} (+{tokens}E)", shape="pointer")
                                self.flowchart.add_edge(node_id, ptr_node, label="归档")
                        except Exception:
                            pass
                    return

    def _maintain_pointer_table(self):
        """指针表维护：当一级 pointer 超过阈值时，合并同 task pointer。"""
        count = self.pointer_store.primary_count()
        threshold = Config.MAX_PRIMARY_POINTERS * Config.POINTER_MERGE_THRESHOLD
        if count < threshold:
            return
        # 按 task 分组
        primaries = self.pointer_store._index.primary_entries()
        by_task: dict[str, list] = {}
        for entry in primaries:
            by_task.setdefault(entry.task, []).append(entry)
        # 合并 task 下 ≥3 个 pointer
        merged_count = 0
        for task_prefix, entries in by_task.items():
            if len(entries) < 3:
                continue
            merged_id = self.pointer_store.merge_pointers(task_prefix)
            if merged_id:
                merged_count += 1
                # 更新栈中对应帧
                for f in self.stack:
                    if getattr(f, 'pointer_id', '') in [e.id for e in entries]:
                        f.pointer_id = merged_id
                        f.content = f"摘要:{merged_id}"
        if merged_count:
            print(f"  [PointerTable] Merged {merged_count} task groups "
                  f"({count} -> {self.pointer_store.primary_count()} primary pointers)")
            # 流程图：MERGE 节点
            if self.flowchart:
                try:
                    node_id = f"ptr_merge_{self.depth}_{self.agent_id}_{self.step_counter}"
                    self.flowchart.add_node(
                        node_id,
                        f"MERGE {merged_count} groups ({count}→{self.pointer_store.primary_count()})",
                        shape="pointer")
                    task_ref = self._fc_task_node if hasattr(self, '_fc_task_node') else None
                    if task_ref:
                        self.flowchart.add_edge(task_ref, node_id, label="合并")
                except Exception:
                    pass

    # ---- 上下文构建（单字符串，用于估算长度等） ----
    def _build_context(self) -> str:
        frames = list(self.stack)
        budget = Config.CONTEXT_BUDGET
        parts = []

        history_frames = [f for f in frames if f.type == "history"]
        if history_frames:
            parts.append(history_frames[-1].content)

        exp_frames = [f for f in frames if f.type == "experience"]
        if exp_frames:
            parts.append(exp_frames[-1].content)

        plan_frames = [f for f in frames if f.level == 1]
        for f in plan_frames:
            parts.append(f"## Plan\n{f.content}")

        merge_frames = [f for f in frames if f.level == 3]
        if merge_frames:
            for f in merge_frames:
                parts.append(f"## Phase Summary\n{f.content}")

        summary_frames = [f for f in frames if f.level == 2]
        used = sum(len(p) for p in parts)
        remaining = budget - used
        if remaining > 0 and summary_frames:
            history_parts = []
            for f in reversed(summary_frames):
                entry = f"Step {f.step_id}: {f.content}"
                if remaining - len(entry) < 0:
                    break
                history_parts.append(entry)
                remaining -= len(entry)
            if history_parts:
                parts.append("## Completed Steps\n" + "\n".join(reversed(history_parts)))

        return "\n\n".join(parts)

    # ---- 多轮消息构建（缓存友好） ----
    def _build_messages(self) -> tuple[list, list]:
        """从栈帧生成多轮消息列表 + system blocks，供缓存友好的 API 调用。
        返回 (messages, system_blocks)，system_blocks 含 cache_control 断点。"""
        cc = getattr(self._api._adapter, 'cache_control', lambda: {})()
        PLATFORM_TEXT = (
            f"Program start time: {self.start_time}\n"
            "Platform: Windows. Use `python` not `python3`. "
            "Prefer `start` or PowerShell for browser/file operations. "
            "When running scripts, use `python scripts/NAME.py` (files auto-routed to scripts/). "
            "CRITICAL: Every `python -c` command MUST include print() to show output. "
            "Think first, then write ONE correct command instead of trial-and-error. "
            "完成任务后在末尾单独一行写 DONE，否则写 CONTINUE。"
        )
        # Plugin prompt fragments
        for plugin in _PLUGINS:
            for fragment in plugin.get_platform_prompt_fragments():
                PLATFORM_TEXT += fragment
        PLATFORM_TEXT += (
            "\n## 并行工具调用\n"
            "你可以在单次回复中同时调用多个**独立**工具（无依赖关系），系统会批量执行后一次性返回所有结果。\n"
            "有依赖关系的工具不能并行，必须顺序调用。"
        )
        # System blocks（缓存断点 #1: tools 已通过 _build_cached_tools 标记）
        system_blocks = [
            {"type": "text", "text": self.stack[0].content},
        ]
        platform_block = {"type": "text", "text": PLATFORM_TEXT}
        if cc:
            platform_block["cache_control"] = cc  # 缓存断点 #2: system
        system_blocks.append(platform_block)

        messages = []
        frames = list(self.stack)

        # 经验注入
        exp = [f for f in frames if f.type == "experience"]
        if exp:
            messages.append({"role": "user", "content": exp[-1].content})
            messages.append({"role": "assistant", "content": "经验已纳入参考。"})

        # 计划
        plans = [f for f in frames if f.level == 1]
        if plans:
            messages.append({"role": "user", "content": f"## Plan\n{plans[0].content}"})
            messages.append({"role": "assistant", "content": "计划已确认，开始执行。"})

        # 父级上下文
        hist = [f for f in frames if f.type == "history"]
        if hist:
            messages.append({"role": "user", "content": hist[-1].content})

        # 阶段合并 (level 3)
        merges = [f for f in frames if f.level == 3]
        for f in merges:
            messages.append({"role": "user", "content": "汇总阶段成果"})
            messages.append({"role": "assistant", "content": f.content})

        # 已完成步骤 (level 2) — 配对 user(subtask desc)/assistant(result)
        summaries = [f for f in frames if f.level == 2]
        done_subs = [s for s in self.subtask_queue if s.get('done')]
        for i, f in enumerate(summaries):
            desc = done_subs[i]['desc'] if i < len(done_subs) else f"Step {f.step_id}"
            messages.append({"role": "user", "content": f"执行: {desc}"})
            if getattr(f, 'failed', False):
                # 失败步骤：只留一行摘要，不保留完整 tool 调用链
                fail_preview = f.content[:100].replace("\n", " ")
                messages.append({"role": "assistant", "content": f"[Step {f.step_id} FAILED] {fail_preview}"})
            elif f.type == "pointer":
                if getattr(f, 'expanded', False):
                    messages.append({"role": "assistant", "content": f.content})
                else:
                    ptr_id = getattr(f, 'pointer_id', '')
                    label = f"[Archived] {f.content}"
                    if ptr_id:
                        entry = self.pointer_store._index.get(ptr_id)
                        tok_info = f" ({entry.tokens} tokens)" if entry else ""
                        label += f"{tok_info} (use recall tool to retrieve)"
                    messages.append({"role": "assistant", "content": label})
            else:
                label = f"[Step {f.step_id}] " if f.type == "summary" else ""
                messages.append({"role": "assistant", "content": label + f.content})

        return messages, system_blocks

    # ---- 单步执行（原生工具循环） ----
    def execute_next_step(self, task: str) -> tuple[str, bool]:
        self.step_counter += 1
        if not self.energy_manager.charge(self.energy_manager.step_overhead):
            print(f"  [Step {self.step_counter}] Energy depleted (overhead), skipping step")
            return "(energy depleted)", False
        self.energy_manager.spend(self.energy_manager.step_overhead)
        # 记录步骤节点
        step_node = f"s{self.depth}_{self.agent_id}_{self.step_counter}"
        if self.flowchart:
            try:
                overhead = int(self.energy_manager.step_overhead)
                self.flowchart.add_node(step_node, f"步骤 {self.step_counter}: {task[:30]}...", shape="step")
                if hasattr(self, '_fc_task_node') and self._fc_task_node:
                    self.flowchart.add_edge(getattr(self, "_fc_current_sub_node", self._fc_task_node), step_node, label=f"-{overhead}E")
            except Exception:
                pass
        self._current_step_node = step_node
        # 多轮消息格式（缓存友好）
        base_messages, system_blocks = self._build_messages()
        cached_tools = _build_cached_tools(self._cache_provider)
        # Root 有完整描述，子节点继承理解 -> 去所有 desc
        if self.depth > 0:
            def _strip_desc(t):
                t = {k: v for k, v in t.items() if k != "description"}
                if "input_schema" in t and "properties" in t["input_schema"]:
                    t["input_schema"] = dict(t["input_schema"])
                    t["input_schema"]["properties"] = {
                        pn: {pk: pv for pk, pv in pi.items() if pk != "description"}
                        for pn, pi in t["input_schema"]["properties"].items()}
                return t
            cached_tools = [_strip_desc(t) for t in cached_tools]
        # can_spawn=False -> 移除 spawn_agent（叶子节点）
        if not self.can_spawn:
            cached_tools = [t for t in cached_tools
                           if t.get("name") != "spawn_agent"]
        cc = getattr(self._api._adapter, 'cache_control', lambda: {})()

        # 当前步骤指令（缓存断点 #3: 最新 user 消息）
        # Plan pending -> 嵌入 plan prompt，一次 API 返回 A (plan) + B (tool calls)
        plan_prefix = self._plan.prompt
        if plan_prefix:
            step_text = f"{plan_prefix}\n\n[Progress] Step {self.step_counter}"
        else:
            step_text = f"[Task]\n{task}\n\n[Progress] Step {self.step_counter}"
        if cc:
            step_msg = {"role": "user", "content": [
                {"type": "text", "text": step_text, "cache_control": cc}
            ]}
        else:
            step_msg = {"role": "user", "content": step_text}

        messages = base_messages + [step_msg]
        final_text = ""

        for round_num in range(self.max_tool_rounds):
            # 能量不足则停止工具循环
            if self.energy_manager.energy <= 0:
                print(f"  [Tool loop] Energy depleted after {round_num} rounds")
                # 流程图：工具循环耗尽
                if self.flowchart:
                    try:
                        depl_id = f"depl_d{self.depth}_{self.step_counter}_{round_num}"
                        label = f"能量耗尽 (round {round_num})"
                        self.flowchart.add_node(depl_id, label, shape="decision")
                        self.flowchart.add_edge(self._current_step_node, depl_id, label="耗尽")
                    except Exception:
                        pass
                break

            # Pre-call 能量估算：chars / 4 ≈ tokens，成本 = tokens * 1.0
            msg_chars = sum(len(str(m)) for m in messages)
            est_input_tokens = msg_chars // 4
            est_cost = est_input_tokens * Config.TOKEN_COST_INPUT + 500  # +500 估算 output
            if est_cost > self.energy_manager.energy:
                # 优先尝试弹栈回收
                reclaimed = self._reclaim_energy(est_cost - self.energy_manager.energy)
                if est_cost > self.energy_manager.energy and round_num > 0 and msg_chars > Config.MAX_MESSAGE_CHARS // 2:
                    # 上下文过大且能量不足：用子 Agent 摘要替代全量消息
                    print(f"  [Context] Summarizing {msg_chars} chars via child agent...")
                    summary = self._summarize_via_child(messages, task_hint=task)
                    if summary:
                        # 用摘要替换 base_messages 中的历史
                        messages = [
                            {"role": "user", "content": f"[执行摘要]\n{summary}"},
                            {"role": "assistant", "content": "已了解，继续执行。"},
                            step_msg
                        ]
                        # 重算估算成本
                        msg_chars = sum(len(str(m)) for m in messages)
                        est_input_tokens = msg_chars // 4
                        est_cost = est_input_tokens * Config.TOKEN_COST_INPUT + 500
                        print(f"  [Context] After summary: {msg_chars} chars, est_cost={est_cost:.0f}")
                if est_cost > self.energy_manager.energy:
                    print(f"  [Tool loop] Energy low ({self.energy_manager.energy:.0f} < est {est_cost:.0f}), stopping")
                    break

            try:
                response = self._api.call(
                    messages, system=system_blocks, tools=cached_tools,
                    extra_params=getattr(self, '_extra_params', None),
                )
            except ContextTooLongError:
                self._adapt_context()
                self._maintain_pointer_table()
                messages.append({"role": "assistant", "content": "[Context too long, compressed. Retry.]"})
                continue
            finally:
                self._api.release()

            # ── A/B 分离：首次响应含 plan (A) + 执行 (B) ──────────────────
            plan_just_parsed = False
            if self._plan.state == PlanState.PENDING:
                text_blocks = [b for b in response.content if b.type == "text"]
                plan_text = "\n".join(b.text for b in text_blocks if hasattr(b, 'text'))
                if plan_text.strip():
                    self._plan.reasoning = plan_text  # run() 中 _finalize_plan 消费
                self._plan.state = PlanState.PARSED
                self._plan.prompt = ""  # 清除，避免后续步骤重复注入
                plan_just_parsed = True

            if response.stop_reason == "tool_use":
                tool_blocks = [b for b in response.content if b.type == "tool_use"]
                # 厂商适配：提取 reasoning_content（DeepSeek/Kimi 思考链跨轮保留）
                extra_msg = self._api._adapter.post_process(response)
                # A/B 分离：plan 文本进 stack[1]（通过 PlanContext.reasoning），消息里只留简短标记
                if plan_just_parsed:
                    clean_content = [{"type": "text", "text": "[Plan recorded. Executing...]"}]
                    # 保留 thinking 块（DeepSeek/Kimi 跨轮必须）
                    for b in response.content:
                        if b.type == "thinking":
                            clean_content.append({"type": "thinking",
                                                  "thinking": b.thinking,
                                                  "signature": b.signature})
                        elif b.type == "tool_use":
                            clean_content.append({"type": "tool_use",
                                                  "id": b.id, "input": b.input, "name": b.name})
                    am = {"role": "assistant", "content": clean_content}
                    if extra_msg.get("reasoning_content"):
                        am["reasoning_content"] = extra_msg["reasoning_content"]
                    messages.append(am)
                else:
                    am = {"role": "assistant", "content": response.content}
                    if extra_msg.get("reasoning_content"):
                        am["reasoning_content"] = extra_msg["reasoning_content"]
                    messages.append(am)

                # 并行工具补贴：无论多少个工具，只计 1 次，费用 ×3
                # 合并 plan+execute 时免除工具费（省下的 plan 调用抵消）
                tool_count = len(tool_blocks)
                if plan_just_parsed:
                    print(f"  [Tool] {tool_count} tools (plan-merged, tool cost WAIVED)")
                    # 工具费完全免除——省下的 plan API 调用已抵消
                elif tool_count == 1:
                    if not self.energy_manager.charge(Config.TOOL_ENERGY_COST):
                        print(f"  [Tool] Energy depleted, cannot start tool")
                        break
                    self.energy_manager.spend(Config.TOOL_ENERGY_COST)
                elif tool_count > 1:
                    batch_cost = Config.TOOL_ENERGY_COST * Config.BATCH_TOOL_COST_MULT
                    if not self.energy_manager.charge(batch_cost):
                        print(f"  [Tool] Energy depleted, cannot start tool batch ({tool_count} tools, cost={batch_cost})")
                        break
                    self.energy_manager.spend(batch_cost)
                    saved = tool_count - 3
                    saved_str = f"(省 {saved} 次)" if saved > 0 else (f"(亏 {-saved} 次)" if saved < 0 else "(持平)")
                    print(f"  [Tool] 并行 {tool_count} 个工具，按 1 次计费 ×3 = {batch_cost} {saved_str}")

                # ── 并行 spawn 检测 ──────────────────────────────────────
                spawn_blocks = [b for b in tool_blocks if b.name == "spawn_agent"]
                other_blocks = [b for b in tool_blocks if b.name != "spawn_agent"]
                parallel_spawn_results = {}  # {block_id: result_dict}

                agent_ctx = {
                    "depth": self.depth,
                    "system_prompt": self.stack[0].content,
                    "parent_agent": self
                }
                if len(spawn_blocks) > 1:
                    em = self.energy_manager
                    spawn_n = len(spawn_blocks)
                    if em.should_spawn():
                        invest_per = max(float(Config.SPAWN_INVEST_MIN), em.energy * 0.05)
                        reserve = em.energy * Config.SPAWN_RESERVE_RATIO
                        total_deduct = invest_per * spawn_n + reserve
                        if em.charge(total_deduct):
                            em._reserve_stack.append(reserve)
                            child_pool_energy = em.energy  # 扣留后的父流动资金 = 子 CoW 基准
                            child_total = child_pool_energy * (1 - Config.SPAWN_RESERVE_RATIO)
                            print(f"  [ParallelSpawn] {spawn_n} children, reserve={reserve:.0f}E, "
                                  f"invest each={invest_per:.0f}E, child pool={child_total:.0f}E")

                            # L2 同级共享缓存：每层分叉一个独立命名空间
                            ns_id = f"l2_{self.agent_id}_{int(time.time()) % 100000}"
                            l2_cache = L2Cache(ns_id, self.scope)
                            print(f"  [L2] Created {ns_id} for {spawn_n} siblings")

                            def _run_one_child(block, agent_ctx):
                                args = dict(block.input)
                                args["_parallel_mode"] = True
                                args["_child_total_energy"] = child_total
                                args["_l2_cache"] = l2_cache
                                return block.id, ToolExecutor._spawn_agent(
                                    args, Config.TOOL_RESULT_BUDGET, agent_ctx)

                            with ThreadPoolExecutor(max_workers=spawn_n) as pool:
                                futures = {pool.submit(_run_one_child, b, agent_ctx): b for b in spawn_blocks}
                                for future in as_completed(futures):
                                    block_id, result = future.result()
                                    parallel_spawn_results[block_id] = result

                            # 释放保底
                            if em._reserve_stack:
                                r = em._reserve_stack.pop()
                                em.credit(r)
                                print(f"  [ParallelSpawn] Reserve released: {r:.0f}E -> energy {em.energy:.0f}")
                            # 结算投资（explore/exploit）
                            for bid, r in parallel_spawn_results.items():
                                mode = r.get("_mode", "exploit")
                                success = r.get("success", False)
                                if mode == "explore":
                                    if success:
                                        em.credit(invest_per + invest_per * em.explore_roi)
                                    else:
                                        em.spend(invest_per)
                                else:
                                    if success:
                                        em.credit(invest_per)
                                    else:
                                        em.credit(invest_per * 0.5)
                                        em.spend(invest_per * 0.5)
                                if not success:
                                    em.update_spawn(False)
                            # 吸收结果
                            for bid, r in parallel_spawn_results.items():
                                if r.get("success") and self:
                                    output = r.get("output", "")
                                    task_desc = "parallel subtask"
                                    for sb in spawn_blocks:
                                        if sb.id == bid:
                                            task_desc = sb.input.get("task", task_desc)[:80]
                                            break
                                    self._absorb_child_result(output, task_desc)
                                    self.energy_manager.update_spawn(True)
                                    # 经验上提
                                    child_exp = r.get("_child_experiences", [])
                                    if child_exp:
                                        self._pending_experiences.extend(child_exp)
                                        self._save_pending_experiences()
                                        print(f"  [Experience] propagated {len(child_exp)} records from parallel child")
                                    child_rep = r.get("_child_reports", [])
                                    if child_rep:
                                        for rd in child_rep:
                                            try:
                                                self.reports_history.append(NodeReport.from_dict(rd))
                                            except Exception:
                                                pass
                            # 同期组完成，销毁 L2
                            try:
                                l2_cache.destroy()
                                print(f"  [L2] Destroyed {ns_id}")
                            except Exception:
                                pass
                        else:
                            for b in spawn_blocks:
                                parallel_spawn_results[b.id] = {"success": False, "output": "Insufficient energy for parallel spawn"}
                    else:
                        for b in spawn_blocks:
                            parallel_spawn_results[b.id] = {"success": False, "output": "Spawn conditions not met (low energy/budget)"}

                # ── 执行所有工具 ──────────────────────────────────────
                # ── 只读工具并行（read_file, list_dir, recall, read_peer）──
                READ_ONLY_TOOLS = {"read_file", "list_dir", "recall", "read_peer"}
                read_blocks = [b for b in tool_blocks if b.name in READ_ONLY_TOOLS]
                write_blocks = [b for b in tool_blocks if b.name not in READ_ONLY_TOOLS
                                and b.id not in parallel_spawn_results]
                spawn_pre = [b for b in tool_blocks if b.id in parallel_spawn_results]

                def _exec_tool(block, budget):
                    return ToolExecutor.execute(
                        block.name, dict(block.input),
                        budget=budget, agent_context=agent_ctx)

                # 并行执行只读工具
                tool_result_map = {}  # block_id -> result
                if len(read_blocks) > 1:
                    msg_total = sum(len(str(m)) for m in messages)
                    rbudget = max(1000, min(Config.TOOL_RESULT_BUDGET,
                                            Config.CONTEXT_BUDGET - msg_total))
                    with ThreadPoolExecutor(max_workers=len(read_blocks)) as rpool:
                        rfutures = {rpool.submit(_exec_tool, b, rbudget): b for b in read_blocks}
                        for f in as_completed(rfutures):
                            b = rfutures[f]
                            tool_result_map[b.id] = f.result()
                    print(f"  [Tool] {len(read_blocks)} read tools executed in parallel")
                elif len(read_blocks) == 1:
                    b = read_blocks[0]
                    tool_result_map[b.id] = _exec_tool(b, Config.TOOL_RESULT_BUDGET)
                # 串行执行写工具
                for b in write_blocks:
                    tool_result_map[b.id] = _exec_tool(b, Config.TOOL_RESULT_BUDGET)
                # 并行 spawn 预执行结果
                for bid, result in parallel_spawn_results.items():
                    tool_result_map[bid] = result

                # 按原始顺序组装 tool_results
                tool_results = []
                for block in tool_blocks:
                    result = tool_result_map.get(block.id, {
                        "success": False,
                        "output": f"Tool execution missing: {block.name}"
                    })
                    args_str = json.dumps(block.input, ensure_ascii=False)[:100]
                    status = "ok" if result["success"] else "error"
                    print(f"  [Tool result] {block.name}({args_str}) status={status}: "
                          f"{str(result.get('output', ''))[:80]}...")

                    merged_node = self.flowchart.merged_tool(
                        step_node, block.name, self.step_counter,
                        depth=self.depth) if self.flowchart else None
                    if merged_node and self.flowchart and not result["success"]:
                        try:
                            self.flowchart._append(f"    class {merged_node} toolFail")
                        except Exception:
                            pass

                    tr = {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result.get("output", ""),
                        "is_error": not result["success"]
                    }
                    if result.get("subtask_complete"):
                        tr["subtask_complete"] = True
                    tool_results.append(tr)

                messages.append({"role": "user", "content": tool_results})

                # 检测 subtask_complete 信号：若任一工具返回该标记，立即结束工具循环
                subtask_done = False
                for t_result in tool_results:
                    if isinstance(t_result, dict) and t_result.get("subtask_complete"):
                        subtask_done = True
                        break
                if subtask_done:
                    final_text = "✅ 任务完成"
                    print(f"  [Auto-complete] subtask_complete signal detected")
                    break

                # Post-append 消息总量守卫：超过 MAX_MESSAGE_CHARS 则截断最旧工具结果
                total_chars = sum(len(str(m)) for m in messages)
                if total_chars > Config.MAX_MESSAGE_CHARS:
                    messages = self._trim_messages(messages, Config.MAX_MESSAGE_CHARS)
                # 工具轮结束，显示内联进度
                em = self.energy_manager
                budget_pct = em.total_spent / em.total_energy * 100 if em.total_energy > 0 else 0
                tok_info = f" | Tokens {em.total_tokens}" if em.total_tokens > 0 else ""
                ctx_info = f" | ctx~{em.estimated_context_usage:.0%}" if em._cumulative_input > 0 else ""
                print(f"  [{round_num+1}/{self.max_tool_rounds} rounds | Budget {budget_pct:.0f}% | Energy {em.energy:.0f}{tok_info}{ctx_info}]")

                # 存档点 pop：状态已恢复，中断工具循环
                if getattr(self, '_just_popped', False):
                    print(f"  [Savepoint] Breaking tool loop after pop — next step uses restored state")
                    break

                continue

            # stop_reason == "end_turn"
            final_text = self._extract_text(response)
            # plan 刚解析但无工具调用 -> 标记需要重放，让 run() 重新执行第一个子任务
            if plan_just_parsed:
                self._plan.state = PlanState.NEEDS_REPLAY
                print(f"  [Plan] Plan-only response (no tool calls), will replay first subtask")
            break

        if not final_text:
            final_text = "(no response)"

        # 存档点 pop 后跳过 step_detail 写入（栈已恢复到存档点状态）
        if getattr(self, '_just_popped', False):
            self._just_popped = False
            # 不追加 step_detail，不 emit event，不 adapt_context
            # 直接返回 — run() 将继续下一个 subtask
            return "(savepoint popped)", False

        self.stack.append(StackFrame("step_detail", final_text, self.step_counter, level=2, agent_id=self.agent_id))
        self.emit_event(SubAgentEventType.STEP_COMPLETED, f"Step {self.step_counter}",
                        {"steps": 1, "tool_rounds": round_num + 1})
        self._adapt_context()
        self._maintain_pointer_table()

        # ── L2 发布：步骤指针写入同级共享缓存 ──
        if self.l2_cache:
            try:
                # 从栈中找本次步骤产生的 pointer_id
                ptr_id = ""
                token_est = 0
                for f in self.stack:
                    if hasattr(f, 'pointer_id') and f.pointer_id and f.step_id == self.step_counter:
                        ptr_id = f.pointer_id
                        token_est = len(f.content) // 4
                        break
                summary = final_text[:300].replace("\n", " ")
                self.l2_cache.publish(self.agent_id, self.step_counter,
                                       summary, ptr_id, token_est)
            except Exception:
                pass

        # ── 节点报告：机械字段自动收集 ──
        try:
            report = NodeReport.collect_from_agent(self, task_desc=task[:200])
            report.summary = final_text[:300].replace("\n", " ")
            self._current_report = report
            self.reports_history.append(report)
            report.save()  # 持久化到磁盘
        except Exception:
            pass

        self._save_state()

        last_line_upper = final_text.rstrip().split("\n")[-1].upper()
        tail_text = final_text.rstrip().split("\n")[-5:]  # last 5 lines
        # DONE 标记：标准 "DONE" 或中文完成标记（不同模型格式不同）
        is_done = ("DONE" in last_line_upper
                   or any("任务完成" in l for l in tail_text)
                   or "TASK COMPLETE" in last_line_upper)
        # 记录步骤完成节点
        if self.flowchart:
            try:
                done_node = f"s{self.depth}_{self.agent_id}_{self.step_counter}_done"
                self.flowchart.add_node(done_node, f"步骤 {self.step_counter} {'完成' if is_done else '继续'}", shape="end")
                self.flowchart.add_edge(self._current_step_node, done_node)
            except Exception:
                pass

        return final_text, is_done

    # ---- 汇合 ----
    def merge_level(self, level: int):
        frames = [f for f in self.stack if f.level == level]
        if len(frames) < 2:
            return
        merged_content = "\n".join(f"Step {f.step_id}: {f.content}" for f in frames)
        prompt = f"将以下多个步骤的执行结果整合为一段连贯的阶段总结（不超过200字）：\n\n{merged_content}"
        response = self._api.call(
            [{"role": "user", "content": prompt}],
            system="你是一个摘要助手。只输出总结。",
            max_tokens=400
        )
        summary = self._extract_text(response)
        new_level = level + 1
        self.stack.append(StackFrame("merge", summary, step_id=0, level=new_level))
        self.stack = deque(f for f in self.stack if f.level != level or f.type in ("constraint", "plan"))
        self._save_state()
        print(f"  [Merge] {len(frames)} level={level} frames -> level={new_level}")

    # ---- 自动剪枝 ----
    def _auto_prune(self):
        """机械剪枝 — 同时上报关键事件。"""
        if not self.can_prune:
            return
        em = self.energy_manager
        # 1. 能量紧急
        if em.energy / max(em.total_energy, 1) < 0.1 and em._no_progress_count >= 3:
            self._report("warn", "energy", f"Emergency: {em.energy:.0f}E remaining, {em._no_progress_count} no-progress rounds")
            print(f"  [Prune] Emergency: energy={em.energy:.0f}, no_progress={em._no_progress_count}")
            # 截断 subtask_queue，只保留已完成和当前
            done_subs = [s for s in self.subtask_queue if s.get('done')]
            if done_subs:
                pending = [s for s in self.subtask_queue if not s.get('done')]
                if pending:
                    # 保留第一个未完成的，其余丢弃
                    self.subtask_queue = done_subs + [pending[0]]
                    print(f"  [Prune] Trimmed {len(pending)-1} pending subtasks")

        # 2. 连续失败：同一子任务失败 3 次 -> 标记取消
        failures = [r for r in self.reports_history[-10:]
                    if r.system_verify.get("severity") in ("fail", "warn")]
        if len(failures) >= 3:
            # 检查是否同一 subtree
            failed_tasks = set(f.task_summary[:60] for f in failures)
            if len(failed_tasks) == 1:
                print(f"  [Prune] Stuck on '{list(failed_tasks)[0]}' — {len(failures)} failures")
                # 如果还有并行兄弟没完成，等它们；否则标记跳过
                pending = [s for s in self.subtask_queue if not s.get('done')]
                if len(pending) == 1 and pending[0]['desc'][:60] == list(failed_tasks)[0]:
                    pending[0]['done'] = True  # 标记跳过
                    pending[0]['_pruned'] = True
                    print(f"  [Prune] Skipped stuck subtask")

        # 3. 低质量报告 -> 过滤（不影响执行，只影响汇总）
        bad_reports = [r for r in self.reports_history
                       if r.energy_used > 0 and r.tokens_output == 0]
        if len(bad_reports) > 5:
            # 截断到最近 20 条好报告
            good = [r for r in self.reports_history if r.tokens_output > 0]
            if len(good) > 10:
                self.reports_history = good[-20:]
                print(f"  [Prune] Filtered bad reports, kept {len(good)} good")

    # ---- 聚合 (Aggregator 节点) ----
    def aggregate_children(self, children_data: list[dict],
                           action: str = "none",
                           output_to: str = "parent",
                           task_hint: str = "") -> dict:
        """统一聚合节点：收集子节点报告 -> 评判 -> 可选整理/合并 -> 路由输出。

        Args:
            children_data: [{report: NodeReport_dict, output_text: str}, ...]
            action: "none" | "collate" | "merge"
            output_to: "parent" | "peer" | "child"

        Returns:
            {judgment: dict, merged_result: str (if action!=none), output_to: str}
        """
        if not children_data:
            return {"judgment": {}, "merged_result": "", "output_to": output_to}

        # ── 1. 机械评判报告：纯事实陈述，不调 LLM ──
        comparison_rows = []
        for i, cd in enumerate(children_data):
            r = cd.get("report", {})
            row = {
                "index": i + 1,
                "agent_id": r.get("agent_id", f"child_{i}"),
                "step_count": r.get("step_count", 0),
                "tokens": r.get("tokens_input", 0) + r.get("tokens_output", 0),
                "energy_used": r.get("energy_used", 0),
                "tool_call_count": len(r.get("tool_calls", [])),
                "files": r.get("files_produced", []),
                "system_verify": r.get("system_verify", {}).get("severity", "?"),
                "exit_code": r.get("exit_code", 0),
            }
            comparison_rows.append(row)

        # 分组统计
        total_tokens = sum(r["tokens"] for r in comparison_rows)
        total_energy = sum(r["energy_used"] for r in comparison_rows)
        all_verified = all(r["system_verify"] == "pass" for r in comparison_rows)

        judgment = {
            "children_count": len(children_data),
            "comparison": comparison_rows,
            "summary": {
                "total_tokens": total_tokens,
                "total_energy": total_energy,
                "all_system_verify_pass": all_verified,
            },
        }

        # 保存评判报告到磁盘
        try:
            jr = NodeReport(
                agent_id=self.agent_id,
                node_name=task_hint[:60],
                node_type="aggregator",
                step_count=len(children_data),
                children=comparison_rows,
                tool_calls=[],
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            jr.save()
            self.reports_history.append(jr)
        except Exception:
            pass

        # ── 2. 整理/合并（仅 action != none 时调 LLM） ──
        merged_result = ""
        if action in ("collate", "merge") and len(children_data) > 0:
            # 收集子节点输出
            parts = []
            for i, cd in enumerate(children_data):
                output = cd.get("output_text", "")
                if output:
                    parts.append(f"--- 节点 {i+1} ---\n{output[:2000]}")

            if parts:
                if action == "collate":
                    system = "你是整合助手。将以下互补成果拼接为一份连贯的产出。不丢失关键信息。"
                else:  # merge
                    system = "你是去重助手。将以下平行成果合并为一份，去除重复内容，统一表述。"

                prompt = (f"任务: {task_hint}\n\n" +
                          f"以下评判数据供参考:\n"
                          f"  子节点数={len(children_data)}, "
                          f"总token={total_tokens}, 系统验证={'全部通过' if all_verified else '部分预警'}\n\n" +
                          "\n".join(parts) +
                          f"\n\n请生成{'整合' if action=='collate' else '合并'}后的产出（不超过800字）。")

                try:
                    resp = self._api.call(
                        [{"role": "user", "content": prompt}],
                        system=system,
                        max_tokens=1200,
                    )
                    merged_result = self._extract_text(resp)
                    print(f"  [Aggregator] {action} -> {len(merged_result)} chars")
                except Exception as e:
                    print(f"  [Aggregator] {action} failed: {e}")
                    merged_result = f"(aggregation error: {e})"

        # ── 3. 路由输出 ──
        if output_to == "parent":
            # 结果入栈，向上级汇报
            output = merged_result or f"Aggregation judgment: {len(children_data)} children, {total_tokens} tokens"
            self.stack.append(StackFrame("step_detail", output, self.step_counter, level=2,
                                          agent_id=self.agent_id))
        elif output_to == "peer":
            # 发布到 L2，同级可读
            if self.l2_cache and merged_result:
                # STORE 到 pointer，发布 stub
                pid = self.pointer_store.store(
                    merged_result, task=task_hint[:60], level=0,
                    frame_type="aggregated", step_id=self.step_counter,
                    parent_scope=self.scope, extra_tags=["aggregated"],
                )
                summary = merged_result[:300].replace("\n", " ")
                self.l2_cache.publish(self.agent_id, self.step_counter,
                                       summary, pid or "", len(merged_result) // 4)
        elif output_to == "child":
            # 作为新的子任务下发——暂存到 subtask_queue
            if merged_result:
                self.subtask_queue.append({
                    "desc": f"Aggregated: {task_hint[:80]}",
                    "done": False,
                    "_aggregated_content": merged_result,
                })

        return {
            "judgment": judgment,
            "merged_result": merged_result,
            "output_to": output_to,
        }

    # ---- 显示 ----
    def print_stack(self):
        total = measure_stack(self.stack)
        print(f"\n[Stack] {len(self.stack)} frames, {total}/{Config.CONTEXT_BUDGET} chars, depth={self.depth}")
        for i, f in enumerate(self.stack):
            preview = f.content[:40] + "..." if len(f.content) > 40 else f.content
            print(f"  [{i}] L{f.level} {f.type} ({len(f.content)}): {preview}")

    def _progress_bar(self, current: int, total: int, width: int = 20) -> str:
        if total <= 0:
            return "[" + "-" * width + "] 0%"
        ratio = min(current / total, 1.0)
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        percent = int(ratio * 100)
        return f"[{bar}] {percent}% ({current}/{total})"

    def print_progress(self, max_steps: int = None):
        """一行式进度"""
        em = self.energy_manager
        total_sub = len(self.subtask_queue)
        done_sub = sum(1 for s in self.subtask_queue if s.get('done'))
        max_s = max_steps or Config.MAX_STEPS
        budget_pct = em.total_spent / em.total_energy * 100 if em.total_energy > 0 else 0

        parts = [f"Step {self.step_counter}"]
        if total_sub > 1:
            parts.append(f"Sub {done_sub}/{total_sub}")
        parts.append(f"Budget {budget_pct:.0f}%")
        parts.append(f"Energy {em.energy:.0f}")

        # Token 统计
        if em.total_tokens > 0:
            parts.append(f"Tokens {em.total_tokens}")

        if self.step_estimator and done_sub > 0:
            est_total, _, _ = self.step_estimator.predict_total(self.step_counter)
            parts.append(f"~{est_total:.0f}steps")

        print(f"  [{'] ['.join(parts)}]")

    # ---- 交互模式 ----
    def interact(self, initial_task: str = None, max_steps: int = None):
        max_steps = max_steps or Config.MAX_STEPS

        print("\n" + "=" * 50)
        print("  Agent Interactive Mode")
        print("  /status   — show stack")
        print("  /progress — show progress bar")
        print("  /reset    — reset conversation")
        print("  /exit     — exit")
        print("=" * 50)

        if initial_task:
            self.run(initial_task, max_steps)

        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue
            if user_input == "/exit":
                self.flush_experiences()
                print("Bye!")
                break
            elif user_input == "/status":
                self.print_stack()
                continue
            elif user_input == "/progress":
                self.print_progress()
                continue
            elif user_input == "/reset":
                self._reset_stack()
                print("Reset done.")
                continue

            self._inject_followup(user_input)
            self.run(user_input, max_steps)

    def _reset_stack(self):
        constraint = self.stack[0] if self.stack else StackFrame("constraint", "你是一个AI编程助手。", level=0)
        self.stack = deque([constraint, StackFrame("plan", "Pending plan...", level=1)])
        self.step_counter = 0
        self.subtask_queue = []
        self.conversation_history = []
        self._save_state()

    def _inject_followup(self, question: str):
        self.stack = deque(f for f in self.stack if f.level == 0)

        history_parts = []
        for i, turn in enumerate(self.conversation_history):
            q = turn["q"]
            a = turn["a"]
            a_preview = a[:300] + "..." if len(a) > 300 else a
            history_parts.append(f"[Turn {i+1}] User: {q}\nAssistant: {a_preview}")

        if history_parts:
            history_text = "## Conversation History\n" + "\n\n".join(history_parts)
            self.stack.append(StackFrame("history", history_text, level=0))

        self.stack.append(StackFrame("plan", "Pending plan...", level=1))
        self.step_counter = 0

    # ---- 子任务去重 ----
    def _merge_subtasks(self, subtasks: list[dict], threshold: float = None) -> list[dict]:
        """子任务去重：相似子任务合并，保留更详细的描述"""
        if len(subtasks) <= 1:
            return subtasks
        threshold = threshold or Config.SUBTASK_DEDUP_THRESHOLD

        def _word_set(text: str) -> set:
            """提取中文单字 + 英文单词集合"""
            return set(re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text.lower()))

        def _jaccard(a: set, b: set) -> float:
            if not a and not b:
                return 0.0
            return len(a & b) / len(a | b)

        word_sets = [_word_set(s['desc']) for s in subtasks]
        avg_len = sum(len(s['desc']) for s in subtasks) / len(subtasks)

        def _quality_score(idx: int) -> float:
            specificity = len(subtasks[idx]['desc']) / max(avg_len, 1)
            if self.energy_manager:
                success_rate = self.energy_manager.p_subtask_success(f"sub_{idx}")
            else:
                success_rate = 0.5
            return specificity * success_rate

        # 构建合并映射：低质量 -> 高质量
        n = len(subtasks)
        merge_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                sim = _jaccard(word_sets[i], word_sets[j])
                if sim > threshold:
                    bi, bj = _quality_score(i), _quality_score(j)
                    if bi >= bj:
                        merge_map[j] = i
                    else:
                        merge_map[i] = j

        if not merge_map:
            return subtasks

        # 解析传递链：A->B->C 变成 A->C
        def _resolve(idx, visited=None):
            if visited is None:
                visited = set()
            if idx not in merge_map or idx in visited:
                return idx
            visited.add(idx)
            return _resolve(merge_map[idx], visited)
        merge_map = {k: _resolve(v) for k, v in merge_map.items()}

        merged = []
        consumed = set()
        for i in range(n):
            if i in consumed:
                continue
            if i in merge_map:
                target = merge_map[i]
                if target not in consumed and target != i:
                    # 基于单词的唯一部分提取
                    unique_words = word_sets[i] - word_sets[target]
                    words_in_desc = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', subtasks[i]['desc'])
                    unique_parts = [w for w in words_in_desc if w.lower() in unique_words]
                    new_desc = subtasks[target]['desc']
                    if unique_parts:
                        new_desc += f"; {' '.join(unique_parts)}"
                    merged.append({"desc": new_desc, "done": False})
                    consumed.add(i)
                    consumed.add(target)
                    print(f"  [Dedup] Merged subtask {i+1} -> {target+1} (sim>{threshold:.1f})")
                else:
                    merged.append(subtasks[i])
            else:
                merged.append(subtasks[i])

        return merged if merged else subtasks

    # ---- 计划生成 ----
    @staticmethod
    def _is_simple_task(task: str) -> bool:
        """判断任务是否足够简单，可以跳过 plan（省 1 次 API 调用）。"""
        # 短任务
        if len(task) < 20:
            return True
        # 单一动作：无连词 = 无多步骤依赖
        complex_indicators = ["并", "然后", "接着", "之后", "再",
                              "and", "then", "after", "before",
                              "先", "后", "同时", "最后"]
        if not any(w in task for w in complex_indicators):
            return True
        return False

    def _setup_flowchart_plan(self, task: str):
        """初始化流程图的计划节点（plan 解析前画占位，解析后更新）"""
        if not self.flowchart:
            return
        try:
            if not self.parent:
                task_label = task[:40].replace('"', "'")
                self.flowchart.add_node("task_start", f"任务: {task_label}...", shape="task")
                self.flowchart.add_edge("start", "task_start")
                self._fc_task_node = "task_start"
                for sub_idx, sub in enumerate(self.subtask_queue):
                    sub_label = sub["desc"][:40].replace('"', "'")
                    sub_node = f"sub_{sub_idx}"
                    self.flowchart.add_node(sub_node, f"子任务 {sub_idx+1}: {sub_label}", shape="subtask")
                    self.flowchart.add_edge("task_start", sub_node)
                    setattr(self, f"_fc_sub_{sub_idx}_node", sub_node)
            else:
                spawn_node = getattr(self, "_fc_parent_spawn_node", None)
                if spawn_node:
                    self._fc_task_node = spawn_node
                    for sub_idx, sub in enumerate(self.subtask_queue):
                        sub_label = sub["desc"][:30].replace('"', "'")
                        sub_node = f"d{self.depth}_{self.agent_id}_sub_{sub_idx}"
                        self.flowchart.add_node(sub_node, f"子任务 {sub_idx+1}: {sub_label}", shape="subtask")
                        self.flowchart.add_edge(spawn_node, sub_node)
                        setattr(self, f"_fc_sub_{sub_idx}_node", sub_node)
        except Exception:
            pass

    @staticmethod
    def _parse_plan_steps(text: str) -> list[dict]:
        """从 plan 文本解析 STEP: 行，返回步骤列表。"""
        steps = []
        for line in text.split('\n'):
            line = line.strip()
            if line.upper().startswith('STEP:') or line.upper().startswith('STEP '):
                desc = re.sub(r'^STEP\s*:?\s*', '', line, flags=re.IGNORECASE).strip()
                if desc:
                    steps.append({"desc": desc, "done": False})
        return steps

    def _finalize_plan(self, plan_text: str):
        """plan 解析完成后：写入 stack[1]，更新 subtask_queue 和流程图。"""
        steps = self._parse_plan_steps(plan_text)
        self._plan.reasoning = plan_text  # 保留供调试/流程图

        if steps:
            if Config.ENABLE_SUBTASK_DEDUP:
                steps = self._merge_subtasks(steps)
            self.subtask_queue = steps
            # 更新流程图（先清理旧节点，再重建）
            self._setup_flowchart_plan("")  # re-init with parsed steps

        plan_display = "\n".join(f"  {i+1}. {s['desc']}" for i, s in enumerate(self.subtask_queue))
        full_plan = plan_text + "\n\n## Steps:\n" + plan_display
        self.stack[1] = StackFrame("plan", full_plan, level=1)
        print(f"[Plan] Parsed {len(self.subtask_queue)} subtasks:\n{plan_display}")

    # ---- 主循环 ----
    def run(self, task: str, max_steps: int = None, skip_plan: bool = False) -> str:
        max_steps = max_steps or Config.MAX_STEPS

        # Plugin lifecycle hooks
        for plugin in _PLUGINS:
            plugin.on_run_start(self, task)

        # 初始化能量管理器（顶层 Agent），使用模型 context 窗口，保留值按比例
        if not self.parent:
            cb = self._context_budget or Config.CONTEXT_BUDGET
            reserve = min(Config.CONTEXT_RESERVE, int(cb * 0.25))
            total_e = max(cb - reserve, 4000)
            self.energy_manager = BayesianEnergyManager(
                total_energy=total_e,
                step_overhead=Config.STEP_OVERHEAD,
                cost_tool=Config.TOOL_ENERGY_COST,
            )
            # 自动剪枝（基于机械事实 + 贝叶斯数据）
            self._auto_prune()
            # 重置当前任务 tool 调用记录
            self._current_tool_calls = []
            # 注入相关历史经验
            self._inject_experience(task)
            # 注入内置流程技能模板（作为建议参考）
            for plugin in _PLUGINS:
                plugin.inject_procedure_template(self, task)
            # TLB 预热：经验注入的指针
            warm_ids = [f.pointer_id for f in self.stack
                        if hasattr(f, 'pointer_id') and f.pointer_id]
            if warm_ids:
                self.pointer_store.warm_tlb(warm_ids)

        # 自动匹配 Skill（仅顶层 agent 匹配，子 agent 通过父级 context 继承）
        if self.auto_skill and not self.active_skills and not self.parent:
            matched = self.skill_manager.match(task, top_n=1)
            if matched:
                self.active_skills = matched
                self._skill_name = matched[0].name  # 供结算时使用
                skill_text = f"\n\n## Skill: {matched[0].name}\n{matched[0].content}"
                old = self.stack[0].content
                self.stack[0] = StackFrame("constraint", old + skill_text, level=0)
                print(f"  [Auto-skill] activated: {matched[0].name}")

        # 决定是否需要 plan（AUTO_PLAN 策略 + skip_plan 覆盖）
        should_plan = not skip_plan  # skip_plan 优先（--no-plan 或父级传递）
        if should_plan and Config.AUTO_PLAN == "smart":
            should_plan = not self._is_simple_task(task)
        elif should_plan and Config.AUTO_PLAN == "never":
            should_plan = False
        self._skip_plan_resolved = not should_plan  # 供 _spawn_agent 继承

        if should_plan:
            # 合并 plan + 第一步执行：plan prompt 嵌入第一个 step_text，一次 API 返回 A+B
            self._plan.state = PlanState.PENDING
            self._plan.prompt = (
                f"First, analyze the task and output a plan:\n\n"
                f"1. FINAL_GOAL: What is the concrete final outcome?\n"
                f"2. BACKWARD: What must be true just before the goal? (2-3 levels)\n"
                f"3. FORWARD_STEPS: Concrete actions (max 3 steps).\n"
                f"   Format: STEP: <action>\n\n"
                f"Then, immediately call tools to execute the first step.\n\n"
                f"Task: {task}"
            )
            print(f"\n[Plan] Embedded in first step (A+B, saves 1 API call)")
            self.subtask_queue = [{"desc": task, "done": False}]
        else:
            reason = "skip_plan" if skip_plan else f"AUTO_PLAN={Config.AUTO_PLAN}"
            print(f"\n[Plan] Skipping ({reason}, task: {task[:60]})")
            self.subtask_queue = [{"desc": task, "done": False}]
        # 子任务去重
        if Config.ENABLE_SUBTASK_DEDUP:
            self.subtask_queue = self._merge_subtasks(self.subtask_queue)
        plan_text = "\n".join(f"  {i+1}. {s['desc']}" for i, s in enumerate(self.subtask_queue))
        # 记录计划节点到流程图（plan pending 时先画占位节点，plan 解析后更新）
        self._setup_flowchart_plan(task)
        if not should_plan:
            print(f"[Plan] {len(self.subtask_queue)} subtasks:\n{plan_text}")
            self.stack[1] = StackFrame("plan", plan_text, level=1)

        # 基于计划复杂度调整能量预算（保证至少能执行一轮）
        if not self.parent:
            min_energy = len(self.subtask_queue) * 3 * self.energy_manager.step_overhead * 1.2
            if self.energy_manager.total_energy < min_energy:
                deficit = min_energy - self.energy_manager.total_energy
                self.energy_manager.total_energy += deficit
                self.energy_manager.energy += deficit
                print(f"  [Energy] Budget adjusted to {self.energy_manager.total_energy:.0f}")

        # 初始化贝叶斯预估器
        self.step_estimator = StepEstimator(len(self.subtask_queue))
        self._save_state()

        last_result = ""
        total_steps = 0
        consecutive_failures = 0

        sub_idx = 0
        while sub_idx < len(self.subtask_queue):
            subtask = self.subtask_queue[sub_idx]
            # 能量停止检查（主要停止条件）
            should_stop, reason = self.energy_manager.should_stop_with_estimator(self.step_estimator)
            if should_stop:
                print(f"\n[Stopped] {reason}")
                if self.flowchart and not self.parent:
                    try:
                        self.flowchart.add_node("stop_energy", f"停止: {reason[:30]}", shape="decision")
                        last_sub = getattr(self, f"_fc_sub_{sub_idx-1}_node", self._fc_task_node) if sub_idx > 0 else self._fc_task_node
                        if last_sub and last_sub in self.flowchart.nodes:
                            self.flowchart.add_edge(last_sub, "stop_energy", label="能量耗尽")
                    except Exception:
                        pass
                break

            # 步数安全网（防止能量充足但无限循环）
            if total_steps >= max_steps:
                print(f"\n[Halted] Safety net: {max_steps} steps reached")
                self.emit_event(SubAgentEventType.MAX_STEPS_REACHED, f"{total_steps}/{max_steps}")
                if self.flowchart and not self.parent:
                    try:
                        self.flowchart.add_node("stop_halted", f"停止: {max_steps}步安全网", shape="decision")
                    except Exception:
                        pass
                break

            print(f"\n{'='*40}")
            # 记录当前子任务上下文
            self._fc_current_sub_idx = sub_idx
            self._fc_current_sub_node = getattr(self, f"_fc_sub_{sub_idx}_node", self._fc_task_node)

            print(f"  Subtask {sub_idx+1}/{len(self.subtask_queue)}: {subtask['desc']}")
            self.print_progress(max_steps)
            print(f"{'='*40}")

            progress = self._format_progress(sub_idx)
            # Plugin: inject pending verification feedback
            for plugin in _PLUGINS:
                plugin.on_subtask_loop(self, sub_idx)
            # 注入上轮验证反馈
            feedback_text = ""
            if self._verify_feedback:
                feedback_text = "\n\n[验证反馈]\n" + "\n".join(self._verify_feedback)
                self._verify_feedback.clear()
            step_task = f"{task}\n\nCurrent subtask: {subtask['desc']}\n{progress}{feedback_text}"

            steps_before = self.step_counter
            try:
                # Plugin: check procedure fallback on high consecutive failures
                fallback_result = None
                for plugin in _PLUGINS:
                    fallback_result = plugin.check_procedure_fallback(self, task, consecutive_failures)
                    if fallback_result:
                        break
                if fallback_result:
                    result = fallback_result.get("output", str(fallback_result))
                    done = True
                    for plugin in _PLUGINS:
                        plugin.on_subtask_loop(self, sub_idx)
                    consecutive_failures = 0
                else:
                    result, done = self.execute_next_step(step_task)

                # ── Plan 合并：第一步返回后解析 plan，更新 subtask_queue ──
                plan_was_parsed = False
                plan_state = self._plan.state
                if plan_state in (PlanState.PARSED, PlanState.NEEDS_REPLAY) and self._plan.reasoning:
                    self._finalize_plan(self._plan.reasoning)
                    self._plan.reasoning = ""  # 消费后清除
                    self.step_estimator = StepEstimator(len(self.subtask_queue))
                    plan_was_parsed = True
                    # subtask 引用重新绑定（队列已被 _finalize_plan 替换）
                    subtask = self.subtask_queue[sub_idx] if sub_idx < len(self.subtask_queue) else self.subtask_queue[-1]

                # Plan-only 重放：首次响应只有 plan 文本无工具调用 -> 重试当前子任务
                if self._plan.state == PlanState.NEEDS_REPLAY:
                    self._plan.state = PlanState.IDLE
                    # 重新构建 step_task（subtask_queue 可能已更新，subtask 引用已变）
                    subtask = self.subtask_queue[sub_idx] if sub_idx < len(self.subtask_queue) else self.subtask_queue[-1]
                    step_task = f"{task}\n\nCurrent subtask: {subtask['desc']}\n{self._format_progress(sub_idx)}{feedback_text}"
                    continue
                elif plan_was_parsed:
                    self._plan.state = PlanState.IDLE

                # 空响应检测：长度 < 50 视为实质失败
                if not result or len(result.strip()) < 50:
                    print(f"  [Warning] Empty/minimal response ({len(result.strip()) if result else 0} chars)")
                    self.energy_manager.update_subtask(f"sub_{sub_idx}", False)
                    subtask['done'] = False
                    consecutive_failures += 1
                    # Plugin: check procedure fallback
                    fallback_result = None
                    for plugin in _PLUGINS:
                        fallback_result = plugin.check_procedure_fallback(self, task, consecutive_failures)
                        if fallback_result:
                            break
                    if fallback_result:
                        result = fallback_result.get("output", str(fallback_result))
                        for plugin in _PLUGINS:
                            plugin.on_subtask_loop(self, sub_idx)
                        consecutive_failures = 0  # reset after fallback
                else:
                    # ── Dual-auth verification ───────────────────────────────
                    # Agent decides (self-judgment, drives next action)
                    # System verifies independently (review, anti-cheating)
                    sys_v = self._system_verify(subtask['desc'], result)
                    agent_done = done  # Agent's own claim

                    if agent_done and sys_v["severity"] == "pass":
                        # Both agree — subtask truly done
                        subtask['done'] = True
                        self.energy_manager.update_subtask(f"sub_{sub_idx}", True)
                        consecutive_failures = 0
                        print(f"  [Dual-Auth] Agent OK  System OK  — subtask complete")

                    elif agent_done and sys_v["severity"] in ("warn", "fail"):
                        subtask['done'] = True
                        self.energy_manager.update_subtask(f"sub_{sub_idx}", True)
                        consecutive_failures = 0
                        # 标记失败步骤 -> 跨节点时压缩（只留摘要，不保留完整 tool 调用）
                        for f in self.stack:
                            if f.type in ("step_detail", "summary") and f.step_id == self.step_counter:
                                f.failed = True
                        print(f"  [Dual-Auth] Agent OK  System WARN — Agent over-claim? ({sys_v['reason']})")
                        self._verify_feedback.append(
                            f"[系统复盘提醒] {sys_v['feedback']} "
                            f"（Agent 标记完成，但系统检测到异常。请确认是否确实完成。）"
                        )

                    elif not agent_done and sys_v["severity"] == "pass":
                        # Agent thinks not done, but System sees success
                        if sys_v.get("reason") == "no verification needed":
                            # 纯对话/无动作任务 — 无需验证，直接视为完成
                            subtask['done'] = True
                            self.energy_manager.update_subtask(f"sub_{sub_idx}", True)
                            consecutive_failures = 0
                            print(f"  [Dual-Auth] Agent FAIL  System OK  — conversational task, auto-done ({sys_v['reason']})")
                        else:
                            # Agent 过于保守 — nudge
                            subtask['done'] = False  # respect Agent's caution
                            self.energy_manager.update_subtask(f"sub_{sub_idx}", False)
                            consecutive_failures += 1
                            print(f"  [Dual-Auth] Agent FAIL  System OK  — Agent too conservative? ({sys_v['reason']})")
                            self._verify_feedback.append(
                                f"[系统复盘提醒] 系统检测到子任务可能已完成（{sys_v['reason']}）。"
                                f"如果你认为确实完成，可以直接声明完成。"
                            )

                    else:
                        # Both agree not done
                        subtask['done'] = False
                        self.energy_manager.update_subtask(f"sub_{sub_idx}", False)
                        consecutive_failures += 1
                        if sys_v["feedback"]:
                            self._verify_feedback.append(sys_v["feedback"])
                steps_consumed = self.step_counter - steps_before
                total_steps += steps_consumed
                last_result = result
                # 流程图：子任务完成状态
                if self.flowchart and subtask.get('done') is not None:
                    try:
                        sub_node = f"sub_{sub_idx}"
                        if sub_node in self.flowchart.nodes:
                            self.flowchart.update_subtask_status(
                                sub_node, subtask['desc'], subtask['done'])
                    except Exception:
                        pass
                # 更新能量管理器
                self.energy_manager.update_done(done)
                # 更新贝叶斯预估器
                self.step_estimator.update(steps_consumed)
                try:
                    print(f"\nResult:\n{result[:500]}")
                except UnicodeEncodeError:
                    print(f"\nResult:\n{result[:500].encode('ascii',errors='replace').decode()}")
                sub_idx += 1
            except RuntimeError as e:
                print(f"Error: {e}")
                self.emit_event(SubAgentEventType.FATAL_ERROR, str(e))
                self.energy_manager.update_subtask(f"sub_{sub_idx}", False)
                self._save_state()
                raise

        # 汇合
        level2_frames = [f for f in self.stack if f.level == 2]
        if len(level2_frames) >= 2:
            print(f"\n{'='*40}")
            print(f"  Merging {len(level2_frames)} execution results")
            print(f"{'='*40}")
            self.merge_level(2)

        # 紧急交付：如果没完成但有部分成果，用 bypass_energy 生成最终交付物
        done_count = sum(1 for s in self.subtask_queue if s.get('done'))
        if not self.parent and done_count > 0 and not all(s.get('done') for s in self.subtask_queue):
            print(f"\n{'='*40}")
            print(f"  [Emergency] Delivering partial results ({done_count}/{len(self.subtask_queue)} done)")
            print(f"{'='*40}")
            # 流程图：紧急交付
            if self.flowchart:
                try:
                    self.flowchart.add_node("emergency", f"紧急交付 ({done_count}/{len(self.subtask_queue)})", shape="decision")
                except Exception:
                    pass
            # 收集已有成果
            collected = []
            for f in self.stack:
                if f.level >= 2 and f.content and len(f.content.strip()) > 50:
                    collected.append(f.content[:500])
            if collected:
                summary_prompt = (f"任务: {task}\n\n以下是部分执行结果，请整合为最终交付物（包含已完成部分，"
                                  f"标注未完成部分）:\n\n" + "\n---\n".join(collected))
                try:
                    resp = self._api.call(
                        [{"role": "user", "content": summary_prompt}],
                        system="你是交付助手。整理已有成果为完整报告，标注缺失部分。",
                        max_tokens=2000,
                        bypass_energy=True
                    )
                    last_result = self._extract_text(resp)
                    print(f"  [Emergency] Delivered {len(last_result)} chars")
                except Exception as e:
                    print(f"  [Emergency] Failed: {e}")

        # 经验记录（轻量收集，不调 LLM）
        self._collect_experience_metadata(
            task=task, plan=plan_text, step_count=total_steps,
            success=all(s.get('done') for s in self.subtask_queue),
            final_result=last_result
        )

        # 保存对话历史
        self.conversation_history.append({"q": task, "a": last_result})
        self._save_state()

        # 保存到历史目录
        self._save_history(task, last_result, total_steps,
                           all(s.get('done') for s in self.subtask_queue))

        # 自动刷新经验到数据库（仅顶层 Agent）
        if not self.parent and self._pending_experiences:
            self.flush_experiences()

        # 终端奖励 + Skill 押金结算（仅顶层 Agent）
        if not self.parent:
            task_success = (all(s.get('done') for s in self.subtask_queue)
                            and last_result and len(last_result) > 20)
            em = self.energy_manager
            # 结算 Skill 押金
            skill_name = getattr(self, '_skill_name', '')
            if skill_name:
                deposit = float(Config.SPAWN_INVEST_MIN)
                if task_success:
                    em.credit(deposit)
                    print(f"  [Skill] '{skill_name}' deposit {deposit:.0f}E refunded")
                else:
                    em.spend(deposit)
                    print(f"  [Skill] '{skill_name}' deposit {deposit:.0f}E forfeited")
                # 流程图：Skill 押金结算
                if self.flowchart:
                    try:
                        sk_id = "skill_settle"
                        label = f"Skill {'退回' if task_success else '罚没'} {deposit:.0f}E"
                        self.flowchart.add_node(sk_id, label, shape="energy")
                        self.flowchart.add_edge(self._fc_task_node, sk_id, label="Skill")
                    except Exception:
                        pass
            # 动态终端奖励
            # IK fallback 检查：如果任何 compute_ik 使用了 fallback 方向，不给奖励
            if task_success and self._current_tool_calls:
                for tc in self._current_tool_calls:
                    if tc.get("ik_fallback"):
                        print(f"  [Reward] DENIED: IK fallback detected "
                              f"(requested={tc.get('requested_direction','')}, "
                              f"actual={tc.get('actual_direction','')})")
                        task_success = False
                        break
            expected = self.step_estimator.predict_total(0)[0] if self.step_estimator else float(total_steps)
            self.energy_manager.grant_terminal_reward(
                task_success,
                plan_complexity=len(self.subtask_queue),
                actual_steps=total_steps,
                expected_steps=expected,
                tool_calls_count=len(self._current_tool_calls),
            )
            # 流程图：奖励节点
            if self.flowchart:
                try:
                    reward = 0
                    # 从 grant_terminal_reward 的返回值估算
                    if task_success:
                        base = Config.REWARD_BASE
                        diff = min(2.0, 1.0 + 0.3 * (len(self.subtask_queue) - 1))
                        spent_ratio = em.total_spent / em.total_energy if em.total_energy > 0 else 0
                        eff = max(0.3, 1.0 - spent_ratio)
                        reward = base * diff * eff
                    rw_id = "reward"
                    label = f"{'OK 成功' if task_success else '✗ 失败'} +{reward:.0f}E奖励"
                    self.flowchart.add_node(rw_id, label, shape="decision")
                    # 连到最后一个 sub 节点或 task_start
                    last_done_sub = None
                    for si, s in enumerate(self.subtask_queue):
                        if s.get('done'):
                            last_done_sub = f"sub_{si}"
                    anchor = last_done_sub or self._fc_task_node
                    if anchor and anchor in self.flowchart.nodes:
                        self.flowchart.add_edge(anchor, rw_id, label="完成")
                except Exception:
                    pass
            # 最终状态
            print(f"  [Final] Energy: {em.energy:.0f} | Spent: {em.total_spent:.0f}/{em.total_energy:.0f}")

        # 清理 ipython 会话
        if hasattr(self, '_ipy_sessions') and self._ipy_sessions:
            for sid in list(self._ipy_sessions):
                del self._ipy_sessions[sid]
        print("\nDone!")
        # 关闭流程图
        if self.flowchart:
            try:
                self.flowchart.finalize()
            except Exception:
                pass
        _root_span = getattr(self, '_plugin_root_span', None)
        if _root_span:
            try:
                _root_span.__exit__(None, None, None)
            except Exception:
                pass
        # 主动关闭 httpx client，避免 Python GC 关机清理时 C++ 扩展触发 abort
        try:
            _http_client.close()
        except Exception:
            pass
        return last_result

    def _format_progress(self, current_idx: int) -> str:
        lines = []
        for i, sub in enumerate(self.subtask_queue):
            status = "OK" if sub['done'] else ("▶" if i == current_idx else "○")
            lines.append(f"{status} {i+1}. {sub['desc']}")
        return "Progress:\n" + "\n".join(lines)

    def _system_verify(self, subtask_desc: str, result_text: str) -> dict:
        """通用任务验证 — 基于工具调用链的验证分派。

        从 _current_tool_calls 提取工具类别，分派到对应验证器：
        - 领域插件 -> 通过 register_verifier 注册的验证器
        - write_file -> 文件验证（存在+非空）
        - run_command -> 命令验证（退出码）
        - 默认 -> 关键词验证

        Returns: {"verified": bool, "reason": str, "feedback": str, "severity": str}
        severity: "pass" | "warn" | "fail"
        """
        try:
            from agent_kernel_verify import TaskVerifier
            if not hasattr(self, '_task_verifier') or self._task_verifier is None:
                self._task_verifier = TaskVerifier(self)
            result = self._task_verifier.verify(subtask_desc, result_text)
            # 流程图：验证节点
            if self.flowchart:
                try:
                    fc_step = getattr(self, '_current_step_node', None)
                    if fc_step:
                        v_node = f"verify_{self.depth}_{self.agent_id}_{self.step_counter}"
                        sev = result.get("severity", "pass")
                        cat = result.get("reason", "verify")[:30]
                        label = f"验证: {cat} ({sev})"
                        self.flowchart.add_node(v_node, label, shape="verify")
                        self.flowchart.add_edge(fc_step, v_node, label="验证")
                except Exception:
                    pass
            return result
        except ImportError:
            print("  [Verify] TaskVerifier unavailable — falling back to keyword check")
            return self._keyword_verify(result_text)

    @staticmethod
    def _keyword_verify(result_text: str) -> dict:
        """关键词验证（仅作微弱信号，不作权威判断）。"""
        result_lower = result_text.lower()
        # 只认强信号：DONE/TASK COMPLETE 是 Agent 自己写的，不是验证
        has_failure = any(kw in result_lower for kw in [
            '"success": false', '"success":false', 'success: false',
            '失败', 'error:', 'failed', 'traceback', 'exception',
        ])
        if has_failure:
            return {"verified": False, "reason": "failure signal detected",
                    "feedback": "[系统复盘] 检测到失败信号。建议检查并重试。",
                    "severity": "fail"}
        return {"verified": False, "reason": "unverified (no tool-chain verifier available)",
                "feedback": "[系统复盘] 无法通过工具链验证。请人工审查结果。",
                "severity": "warn"}

