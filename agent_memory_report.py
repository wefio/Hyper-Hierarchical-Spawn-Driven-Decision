"""NodeReport — 节点执行报告（运行时数据，不是模板产物）。

每个 Agent 完成 step 或整个任务时生成。只陈述事实，不做判断。
- 机械字段：引擎自动填，不调 LLM
- 内容字段：可选 LLM 生成（异常时触发）
- 子节点字段：仅 aggregator 节点使用

存储：
- 磁盘：agent_state/reports/{agent_id}.yaml
- 上下文：pointer stub（摘要 + report_id, 轻量注入）
- L2：同级查询时返回摘要列表（已有 L2Cache）
"""

from __future__ import annotations
import os
import time
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from config import Config


# ============================================================================
# Dataclass
# ============================================================================

@dataclass
class NodeReport:
    # -- identity --
    agent_id: str = ""
    node_name: str = ""
    node_type: str = "executor"      # executor | aggregator

    # -- mechanical (引擎自动填) --
    step_count: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cache_read: int = 0
    energy_used: float = 0.0
    energy_remaining: float = 0.0
    tool_calls: list = field(default_factory=list)
    files_produced: list = field(default_factory=list)
    system_verify: dict = field(default_factory=dict)
    exit_code: int = 0
    timing_ttft_ms: float = 0.0
    timing_total_ms: float = 0.0
    model_provider: str = ""
    model_name: str = ""
    task_summary: str = ""           # 子任务描述
    created_at: str = ""

    # -- content (LLM 生成，可选) --
    summary: str = ""                # 步骤结果摘要
    decisions: list = field(default_factory=list)   # 偏离软指令时记录理由
    suggestions: str = ""            # 对上下游的提示
    rigidity: str = "soft"           # 节点刚性: rigid | soft | open
    rigidity_compliance: dict = field(default_factory=dict)  # {rigid_passed: bool, soft_deviations: [], open_explorations: []}

    # -- aggregator only --
    children: list = field(default_factory=list)    # 子节点机械报告摘要列表
    aggregated_output_ptr: str = ""  # 合并产出的 pointer_id

    # ========================================================================
    # Serialization
    # ========================================================================

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NodeReport":
        # filter out keys not in the dataclass (backward compat)
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in field_names})

    # ========================================================================
    # I/O
    # ========================================================================

    def save(self, report_dir: str = None) -> str:
        """Write report to disk. Returns file path."""
        base = Path(report_dir or os.path.join(Config.WORK_DIR, "reports"))
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{self.agent_id}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)
        return str(path)

    @classmethod
    def load(cls, agent_id: str, report_dir: str = None) -> Optional["NodeReport"]:
        base = Path(report_dir or os.path.join(Config.WORK_DIR, "reports"))
        path = base / f"{agent_id}.yaml"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    # ========================================================================
    # 轻量摘要（给上下文注入 / L2 查询用）
    # ========================================================================

    def to_pointer_stub(self) -> str:
        """返回一段短文本，适合作为 pointer stub 注入 context。
        不占 context 窗口（<80 chars）。"""
        status = "OK" if self.system_verify.get("severity", "pass") == "pass" else "WARN"
        return (f"[Report {self.agent_id}] step={self.step_count} "
                f"tokens={self.tokens_input + self.tokens_output} "
                f"{status}")

    def to_l2_summary(self) -> dict:
        """返回 L2 共享池中用到的轻量摘要。"""
        return {
            "agent_id": self.agent_id,
            "step_count": self.step_count,
            "tokens": self.tokens_input + self.tokens_output,
            "files": self.files_produced,
            "verify": self.system_verify.get("severity", "pass"),
            "summary": (self.summary or self.task_summary)[:200],
        }

    # ========================================================================
    # 机械数据收集（从 agent 实例提取，不调 LLM）
    # ========================================================================

    @classmethod
    def collect_from_agent(cls, agent, task_desc: str = "",
                           timing: dict = None) -> "NodeReport":
        """从 agent 实例收集所有机械字段。"""
        em = agent.energy_manager
        tc = getattr(agent, "_current_tool_calls", []) or []

        # 提取产生的文件路径
        files = []
        for t in tc:
            if t.get("tool") == "write_file" and t.get("success"):
                p = t.get("params", {}).get("path", "")
                if p:
                    files.append(p)

        # 系统验证结果
        verify = {}
        if tc:
            try:
                from agent_kernel_verify import TaskVerifier
                tv = TaskVerifier(agent)
                verify = tv.verify(task_desc, "")
            except Exception:
                pass

        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        timing = timing or {}

        # 从 APIClient 读取上次调用的模型署名 + 时序
        api_meta = {}
        if hasattr(agent, '_api') and getattr(agent._api, 'last_call_meta', None):
            api_meta = agent._api.last_call_meta

        return cls(
            agent_id=agent.agent_id,
            node_name=getattr(agent, "_skill_name", "") or task_desc[:60],
            node_type="executor",
            step_count=agent.step_counter,
            tokens_input=api_meta.get("input_tokens", em.total_input_tokens),
            tokens_output=api_meta.get("output_tokens", em.total_output_tokens),
            tokens_cache_read=api_meta.get("cache_read_tokens", 0),
            energy_used=em.total_spent,
            energy_remaining=em.energy,
            tool_calls=[{"tool": t.get("tool"), "action": t.get("action"),
                         "success": t.get("success")} for t in tc],
            files_produced=files,
            system_verify=verify,
            timing_ttft_ms=timing.get("ttft_ms", 0),
            timing_total_ms=timing.get("total_ms", api_meta.get("total_ms", 0)),
            model_provider=api_meta.get("model", getattr(Config, "MODEL", ""))[:60],
            model_name=api_meta.get("model", getattr(Config, "MODEL", ""))[:60],
            task_summary=task_desc[:200],
            created_at=now,
        )
