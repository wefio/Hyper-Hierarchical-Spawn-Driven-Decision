"""Foundation types for the context stack — StackFrame (page table entry) and SavepointMeta (checkpoint metadata)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class StackFrame:
    type: str      # "constraint", "plan", "step_detail", "summary", "merge", "history", "pointer"
    content: str
    step_id: int = 0
    level: int = 0
    agent_id: str = ""
    pointer_id: str = ""       # 关联 archive pointer id
    reclaimable: bool = False  # recall 注入的内容，优先回收
    expanded: bool = False     # pointer stub 已展开为全文（原地替换）
    failed: bool = False       # system_verify 失败 → 跨节点压缩时跳过
    use_count: int = 0         # 被引用次数（LRU-K）
    last_used_step: int = 0    # 最后使用 step


@dataclass
class SavepointMeta:
    """存档点元数据 — 活跃存档点 + 历史记录共用"""
    name: str
    path: str                    # 全量快照磁盘路径
    agent_id: str = ""
    status: str = "active"       # "active", "committed", "popped"
    energy_at_save: float = 0.0  # create 时刻的流动资金
    total_spent_at_save: float = 0.0
    step_counter_at_save: int = 0
    context_chars_at_save: int = 0
    summary: str = ""            # commit → 结论摘要, pop → 失败原因
    created_at: float = 0.0
    context_size: int = 0        # context_chars_at_save 的摘要（用于排序驱逐）
