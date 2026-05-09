"""Template engine — YAML 分形模板加载与执行。

模板 = 结构定义 + 可复用缓存。引擎按三种分叉策略派发节点，
集成多级页表指针复用、自动聚合、统计追踪。

Usage:
    engine = TemplateEngine(pointer_store, global_pointer_store)
    template = engine.load("templates/research.yaml")
    result = engine.execute(agent, template, inputs={"topic": "AI"})
"""

from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import yaml

from config import Config
from agent_scheduler_energy import BayesianOptimizer


# ============================================================================
# Template dataclass (thin wrapper over parsed YAML)
# ============================================================================

class TemplateNode:
    """模板中一个节点的解析结果。"""

    def __init__(self, raw: dict, parent_template: str = ""):
        self.id: str = raw.get("id", "")
        self.type: str = raw.get("type", "executor")  # executor | composite
        self.phase: str = raw.get("phase", "execute")  # diverge | execute | converge
        self.prompt: str = raw.get("prompt", "")
        self.inherits: list[str] = raw.get("inherits", [])
        self.depends_on: list[str] = raw.get("depends_on", [])
        self.rigidity: str = raw.get("rigidity", "soft")  # rigid | soft | open

        # model routing
        model = raw.get("model", {})
        self.model_provider: str = model.get("provider", "")
        self.model_name: str = model.get("name", "")
        self.model_fallback: str = model.get("fallback", "")

        # branching (composite nodes only)
        branching = raw.get("branching", {})
        self.branching_strategy: str = branching.get("strategy", "natural")
        self.max_children: int = branching.get("max_children", 5)
        self.replicate: int = branching.get("replicate", 3)
        self.children_template: str = branching.get("children_template", "")

        # aggregation
        aggregate = raw.get("aggregate", {})
        self.aggregate_action: str = aggregate.get("action", "none")
        self.aggregate_output_to: str = aggregate.get("output_to", "parent")

        # pointer reuse
        reuse = raw.get("reuse", {})
        self.reuse_enabled: bool = bool(reuse)
        self.reuse_freshness_days: int = reuse.get("freshness_days", 30)

        # experiment: online Bayesian optimization
        experiment = raw.get("experiment", {})
        self.experiment_param: str = experiment.get("param", "")
        self.experiment_values: list = experiment.get("values", [])

        # runtime (populated by engine)
        self._parent_template = parent_template
        self._raw = raw

    def input_hash(self, inputs: dict) -> str:
        """计算输入参数的哈希（L1 精确查找键）。"""
        raw = f"{self._parent_template}:{self.id}:{json.dumps(inputs, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class Template:
    """加载后的模板。"""

    def __init__(self, raw: dict, source_path: str = ""):
        self.name: str = raw.get("name", source_path)
        self.description: str = raw.get("description", "")
        self.phases: list[str] = raw.get("phases", ["execute"])
        self.nodes: list[TemplateNode] = [
            TemplateNode(n, parent_template=self.name)
            for n in raw.get("nodes", [])
        ]
        self._source = source_path

    def get_node(self, node_id: str) -> Optional[TemplateNode]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    @classmethod
    def load(cls, path: str) -> "Template":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(raw, source_path=path)


# ============================================================================
# Template stats (runtime, not written to template file)
# ============================================================================

class TemplateStats:
    """模板运行时统计，存入 agent_state/template_stats.yaml。

    写入不要求时效性 — 使用队列收集操作，由主线程 flush() 统一落盘。
    线程安全：队列 append 是原子的（GIL）。
    """

    def __init__(self, template_name: str, stats_dir: str = None):
        self.template_name = template_name
        base = Path(stats_dir or os.path.join(Config.WORK_DIR, "template_stats"))
        base.mkdir(parents=True, exist_ok=True)
        self._path = base / f"{template_name}.yaml"
        self.data = self._load()
        self._dirty = False
        self._pending: list[tuple] = []  # (node_id, level, action, reused)

    def _load(self) -> dict:
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                return yaml.safe_load(f.read()) or {}
        return {}

    def _apply_pending(self):
        """应用队列中的操作到内存数据。"""
        for node_id, level, action, reused in self._pending:
            ns = self.data.setdefault(node_id, {
                "total_lookups": 0, "total_reuses": 0, "last_level": 0,
            })
            ns["total_lookups"] += 1
            if reused:
                ns["total_reuses"] += 1
            ns["last_level"] = level
            ns["last_action"] = action
            ns["last_used_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._pending.clear()

    def record_lookup(self, node_id: str, level: int, action: str, reused: bool):
        """线程安全：追加到队列，不立即写盘。"""
        self._pending.append((node_id, level, action, reused))

    def flush(self):
        """应用队列并落盘。由主线程调用。"""
        if not self._pending and not self._dirty:
            return
        self._apply_pending()
        with open(self._path, "w", encoding="utf-8") as f:
            yaml.dump(self.data, f, allow_unicode=True, sort_keys=False)
        self._dirty = False

    @property
    def hit_rate(self) -> float:
        lookups = sum(n.get("total_lookups", 0) for n in self.data.values())
        reuses = sum(n.get("total_reuses", 0) for n in self.data.values())
        return reuses / lookups if lookups > 0 else 0.0


# ============================================================================
# Template engine
# ============================================================================

class TemplateEngine:
    """分形模板执行引擎。

    聚合了：模板加载 → 指针复用查找 → 分叉派发 → 自动聚合 → 统计更新。
    """

    def __init__(self, pointer_store=None, global_pointer_store=None):
        from pointer_store import PointerStore, GlobalPointerStore
        self.pointer_store = pointer_store
        self.global_store = global_pointer_store
        self._page_tables: dict[str, str] = {}
        self._stats: dict[str, TemplateStats] = {}
        self._optimizers: dict[str, BayesianOptimizer] = {}  # node_id → optimizer

    def _get_optimizer(self, node: TemplateNode) -> Optional[BayesianOptimizer]:
        """获取节点的贝叶斯优化器（边用边学）。"""
        if not node.experiment_param or not node.experiment_values:
            return None
        key = node.id
        if key not in self._optimizers:
            self._optimizers[key] = BayesianOptimizer(
                node.experiment_param, node.experiment_values)
        return self._optimizers[key]

    def _get_stats(self, template_name: str) -> TemplateStats:
        if template_name not in self._stats:
            self._stats[template_name] = TemplateStats(template_name)
        return self._stats[template_name]

    def _ensure_page_table(self, node: TemplateNode, scope: str) -> str:
        """确保节点有对应的 L2 页表。"""
        key = f"{node._parent_template}:{node.id}"
        if key not in self._page_tables and self.pointer_store:
            self._page_tables[key] = self.pointer_store.register_page_table(
                key, scope=scope)
        return self._page_tables.get(key, "")

    def load(self, path: str) -> Template:
        return Template.load(path)

    # ========================================================================
    # 指针复用（四级查找）
    # ========================================================================

    def try_reuse(self, node: TemplateNode, inputs: dict,
                  agent_scope: str) -> dict:
        """四级查找，返回 {action, level, pointers}。"""
        if not node.reuse_enabled or not self.pointer_store:
            return {"action": "execute", "level": 0, "pointers": []}

        input_h = node.input_hash(inputs)
        table_id = self._ensure_page_table(node, agent_scope)
        result = self.pointer_store.find_reusable(
            table_id, input_h, json.dumps(inputs))

        # L4: experience_store 兜底
        if result["action"] == "execute" and self.global_store:
            global_hits = self.global_store.search(
                json.dumps(inputs), limit=3)
            if global_hits:
                result = {"level": 4, "action": "experience",
                          "pointers": global_hits}

        stats = self._get_stats(node._parent_template)
        stats.record_lookup(node.id, result["level"], result["action"],
                            result["action"] == "skip")
        return result

    def register_result(self, node: TemplateNode, inputs: dict,
                        ptr_id: str, report: dict, agent_scope: str):
        """执行完成后注册指针到页表和全局库。"""
        if not ptr_id or not self.pointer_store:
            return
        table_id = self._ensure_page_table(node, agent_scope)
        input_h = node.input_hash(inputs)
        self.pointer_store.add_to_page_table(
            table_id, input_h, ptr_id,
            summary=report.get("summary", "")[:300],
            verification=report.get("system_verify", {}).get("severity", "pass"),
            freshness_days=node.reuse_freshness_days,
        )
        if self.global_store:
            self.global_store.register(report, ptr_id=ptr_id,
                                       task_desc=json.dumps(inputs)[:200])

    # ========================================================================
    # 模板执行
    # ========================================================================

    def execute(self, agent, template: Template,
                inputs: dict = None, max_steps: int = None) -> str:
        """执行模板。对根节点递归展开。"""
        inputs = inputs or {}
        if not template.nodes:
            return "(empty template)"
        root = template.nodes[0]
        try:
            return self._execute_node(agent, root, inputs, template,
                                      max_steps or Config.MAX_STEPS)
        finally:
            # 确保统计落盘
            stats = self._get_stats(template.name)
            stats.flush()

    def _execute_node(self, agent, node: TemplateNode, inputs: dict,
                      template: Template, max_steps: int) -> str:
        """执行单个模板节点。"""
        task_text = node.prompt.format(**inputs) if node.prompt else str(inputs)

        if node.type == "executor":
            return self._execute_executor(agent, node, task_text, inputs)

        elif node.type == "composite":
            return self._execute_composite(agent, node, task_text, inputs,
                                           template, max_steps)
        return "(unknown node type)"

    def _execute_executor(self, agent, node: TemplateNode,
                          task_text: str, inputs: dict) -> str:
        """执行 executor 节点：在线贝叶斯优化 → 查复用 → 运行。"""
        # 在线优化：试验参数 → Thompson 采样
        opt = self._get_optimizer(node)
        opt_active = False
        if opt and not opt.converged():
            best_val, best_idx = opt.sample()
            opt_active = True
            # 注入采样配置
            if node.experiment_param == "rigidity":
                node.rigidity = best_val
                task_text = f"[{best_val}] {task_text}"
            print(f"  [Optimize] {node.id}: {node.experiment_param}={best_val} "
                  f"(trial #{opt._trials[best_idx]+1})")

        # 指针复用
        reuse = self.try_reuse(node, inputs, agent.scope)
        if reuse["action"] == "skip" and reuse["pointers"]:
            ptr = reuse["pointers"][0]
            print(f"  [Template] {node.id}: L{reuse['level']} REUSE {ptr['ptr_id']}")
            output = f"[Reused pointer {ptr['ptr_id']}] {ptr['summary']}"
            from agent_memory_frame import StackFrame
            agent.stack.append(StackFrame(
                "step_detail", output, agent.step_counter, level=2,
                agent_id=agent.agent_id))
            return output

        if reuse["action"] == "reference" and reuse["pointers"]:
            refs = "\n".join(
                f"- `{p['ptr_id']}`: {p.get('summary', '')[:100]}"
                for p in reuse["pointers"][:3])
            task_text = f"{task_text}\n\n[历史参考]\n{refs}"

        # 执行
        from agent import Agent
        child = Agent(
            system_prompt=agent.stack[0].content,
            depth=agent.depth + 1,
            parent=agent,
        )
        result = child.run(task_text, max_steps=Config.MAX_STEPS)

        # 在线优化：反馈结果
        if opt_active and opt:
            success = bool(result and len(result) > 10 and "error" not in result.lower()[:200])
            opt.update(best_idx, success)
            if opt.converged():
                print(f"  [Optimize] {node.id}: CONVERGED → {opt.best} "
                      f"(trials: {opt._trials})")

        # 注册指针
        if hasattr(child, 'reports_history') and child.reports_history:
            report = child.reports_history[-1]
            ptr_id = self._find_ptr_from_report(report)
            if ptr_id:
                self.register_result(node, inputs, ptr_id,
                                     report.to_dict(), agent.scope)
            agent.reports_history.extend(child.reports_history)

        return result or "(no output)"

    def _execute_composite(self, agent, node: TemplateNode,
                           task_text: str, inputs: dict,
                           template: Template, max_steps: int) -> str:
        """执行 composite 节点：按分叉策略派发子节点。"""
        strategy = node.branching_strategy

        if strategy == "multi_parallel":
            return self._branch_multi_parallel(agent, node, task_text,
                                               inputs, template, max_steps)
        elif strategy == "template":
            return self._branch_template(agent, node, inputs,
                                         template, max_steps)
        else:  # natural
            return self._branch_natural(agent, node, task_text,
                                        inputs, template, max_steps)

    # ── 三种分叉 ──

    def _branch_natural(self, agent, node: TemplateNode, task_text: str,
                        inputs: dict, template: Template,
                        max_steps: int) -> str:
        """自然分叉：LLM 动态决定子任务拆分。"""
        if node.rigidity == "rigid":
            # 刚性自然分叉：LLM 必须生成 plan 并严格遵循
            task_text = (
                "[rigid] 必须生成完整的任务分解计划，并严格按计划执行每一步。\n"
                "[rigid] 不允许跳过或替换计划中的子任务。\n" + task_text)
        elif node.rigidity == "open":
            task_text = (
                "[open] 可以自由探索，不需要严格遵循某个固定计划。\n"
                "[open] 可以随时调整、增减子任务。\n" + task_text)
        return agent.run(task_text, max_steps=max_steps)

    def _branch_template(self, agent, node: TemplateNode, inputs: dict,
                         template: Template, max_steps: int) -> str:
        """模板分叉：按预设子节点展开，按依赖顺序执行。"""
        children = [n for n in template.nodes
                    if n.id != node.id and n.id in node.depends_on
                    or n.id in (node.children_template.split(",") if node.children_template else [])]
        if not children:
            return agent.run(node.prompt.format(**inputs), max_steps=max_steps)

        # 刚性：必须全部执行；软性：可以跳过；开放：可以增减
        if node.rigidity == "rigid":
            print(f"  [Template] rigid: {len(children)} children must all complete")
        elif node.rigidity == "open":
            prompt = node.prompt.format(**inputs) if node.prompt else str(inputs)
            prompt = f"[open] 可以自由调整以下子任务列表，增减或修改子节点:\n{prompt}"
            return agent.run(prompt, max_steps=max_steps)

        results = {}
        for child in children:
            child_inputs = dict(inputs)
            result = self._execute_node(agent, child, child_inputs,
                                        template, max_steps)
            results[child.id] = result

        children_data = [{"report": {}, "output_text": v}
                         for k, v in results.items()]
        agg = agent.aggregate_children(
            children_data,
            action=node.aggregate_action,
            output_to=node.aggregate_output_to,
            task_hint=node.prompt or node.id,
        )
        return agg.get("merged_result", "\n".join(results.values()))

    def _branch_multi_parallel(self, agent, node: TemplateNode,
                               task_text: str, inputs: dict,
                               template: Template,
                               max_steps: int) -> str:
        """多并行分叉：复制 N 份并行执行，强制聚合。"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from agent import Agent
        from agent_scheduler_energy import BayesianEnergyManager

        spawn_n = node.replicate
        if node.rigidity == "open" and spawn_n > 2:
            spawn_n = max(2, spawn_n - 2)  # 开放模式允许减少
        elif node.rigidity == "rigid":
            print(f"  [Template] rigid multi_parallel: {spawn_n} copies locked")
        em = agent.energy_manager
        children_data = []

        invest = max(float(Config.SPAWN_INVEST_MIN), em.energy * 0.05)
        reserve = em.energy * Config.SPAWN_RESERVE_RATIO
        if not em.charge(invest * spawn_n + reserve):
            return "(insufficient energy for multi-parallel)"
        em._reserve_stack.append(reserve)
        child_pool = em.energy * (1 - Config.SPAWN_RESERVE_RATIO)

        def _run_copy(i):
            child_em = BayesianEnergyManager(
                total_energy=child_pool,
                step_overhead=Config.STEP_OVERHEAD,
                cost_tool=Config.TOOL_ENERGY_COST,
            )
            child = Agent(
                system_prompt=agent.stack[0].content,
                depth=agent.depth + 1,
                parent=agent,
                energy_mgr=child_em,
            )
            result = child.run(task_text, max_steps=max_steps)
            report_dict = child.reports_history[-1].to_dict() if child.reports_history else {}
            return {"output_text": result, "report": report_dict,
                    "agent_id": child.agent_id}

        with ThreadPoolExecutor(max_workers=spawn_n) as pool:
            futures = [pool.submit(_run_copy, i) for i in range(spawn_n)]
            for f in as_completed(futures):
                children_data.append(f.result())

        if em._reserve_stack:
            em.credit(em._reserve_stack.pop())

        print(f"  [Template] multi_parallel: {spawn_n} copies -> aggregating")

        agg = agent.aggregate_children(
            children_data,
            action=node.aggregate_action or "merge",
            output_to=node.aggregate_output_to,
            task_hint=node.prompt or node.id,
        )
        return agg.get("merged_result", "(aggregation failed)")

    def _find_ptr_from_report(self, report) -> str:
        """从 NodeReport 关联的指针中提取 ptr_id。"""
        # 检查 pointer_id 字段
        if hasattr(report, 'pointer_id') and report.pointer_id:
            return report.pointer_id
        # 检查 system_verify 中关联的文件
        return ""
