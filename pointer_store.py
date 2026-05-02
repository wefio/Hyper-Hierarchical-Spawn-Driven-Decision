"""Pointer Store — 磁盘缓存层（虚拟内存的 page table + swap disk）

把 context window 中即将被淘汰的内容持久化到本地文件，原地替换为 `摘要:ptr_id`。
支持多级 pointer、作用域隔离、能量联动、LRU-K 追踪。
"""

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import Config


# ---------------------------------------------------------------------------
# 1. PointerEntry — 单条归档元数据
# ---------------------------------------------------------------------------

@dataclass
class PointerEntry:
    id: str                      # e.g. "ptr_a1b2c3d4"
    file: str                    # 相对 archive 根的路径
    summary: str                 # 前 300 字摘要，供搜索
    timestamp: float
    task: str
    level: int                   # 0=原始内容, 1=段落合并, 2=二次合并
    children: List[str] = field(default_factory=list)
    scope: str = "root"          # "root.sub_0"
    tokens: int = 0              # len(content)/4
    tags: List[str] = field(default_factory=list)
    agent_id: str = ""
    frame_type: str = "step_detail"
    step_id: int = 0
    use_count: int = 0
    last_used_step: int = 0
    stored_at_step: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PointerEntry":
        return cls(**d)


# ---------------------------------------------------------------------------
# 2. PointerIndex — 内存中的索引（页表）
# ---------------------------------------------------------------------------

class PointerIndex:
    """内存中的索引字典，启动时从 index.json 加载，STORE 时原子写入。"""

    def __init__(self, archive_dir: Path):
        self._archive_dir = archive_dir
        self._entries: Dict[str, PointerEntry] = {}
        self._dirty = False
        self._index_path = archive_dir / "index.json"
        self._bak_path = archive_dir / "index.json.bak"

    # ---- 持久化 ----

    def load(self) -> None:
        """加载 index.json，损坏时 fallback 到 .bak，否则空索引启动。"""
        for src in (self._index_path, self._bak_path):
            if not src.exists():
                continue
            try:
                with open(src, "r", encoding="utf-8") as f:
                    data = json.load(f)
                version = data.get("version", 1)
                entries = data.get("entries", {})
                self._entries = {k: PointerEntry.from_dict(v) for k, v in entries.items()}
                self._dirty = False
                print(f"  [PointerIndex] Loaded {len(self._entries)} entries from {src.name}")
                return
            except (json.JSONDecodeError, OSError, KeyError) as e:
                print(f"  [PointerIndex] {src.name} corrupt ({e}), trying fallback...")
                continue
        print("  [PointerIndex] Starting with empty index")
        self._entries = {}
        self._dirty = False

    def save(self) -> None:
        """原子写：.tmp → rename → .bak"""
        if not self._dirty:
            return
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._archive_dir / "index.json.tmp"
        data = {
            "version": 1,
            "updated": time.time(),
            "stats": self._stats(),
            "entries": {k: v.to_dict() for k, v in self._entries.items()},
        }
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(self._index_path)
            shutil.copy2(self._index_path, self._bak_path)
            self._dirty = False
        except OSError as e:
            print(f"  [PointerIndex] Save failed: {e}")

    def sync(self) -> None:
        self.save()

    # ---- CRUD ----

    def add(self, entry: PointerEntry) -> None:
        self._entries[entry.id] = entry
        self._dirty = True

    def get(self, ptr_id: str) -> Optional[PointerEntry]:
        return self._entries.get(ptr_id)

    def remove(self, ptr_id: str) -> bool:
        if ptr_id in self._entries:
            del self._entries[ptr_id]
            self._dirty = True
            return True
        return False

    def update(self, ptr_id: str, **kwargs) -> bool:
        entry = self._entries.get(ptr_id)
        if not entry:
            return False
        for k, v in kwargs.items():
            if hasattr(entry, k):
                setattr(entry, k, v)
        self._dirty = True
        return True

    # ---- 查询 ----

    def search_summary(self, query: str, limit: int = 5) -> List[PointerEntry]:
        q = query.lower()
        results = []
        for e in self._entries.values():
            if q in e.summary.lower() or q in e.task.lower() or any(q in t.lower() for t in e.tags):
                results.append(e)
                if len(results) >= limit:
                    break
        return results

    def by_scope(self, scope: str) -> List[PointerEntry]:
        """返回 scope 精确匹配或 scope 前缀匹配的条目（父可见子）。"""
        return [e for e in self._entries.values()
                if e.scope == scope or e.scope.startswith(scope + ".")]

    def by_task(self, task_prefix: str) -> List[PointerEntry]:
        return [e for e in self._entries.values() if e.task.startswith(task_prefix)]

    def primary_entries(self) -> List[PointerEntry]:
        return [e for e in self._entries.values() if e.level == 0]

    def count(self) -> int:
        return len(self._entries)

    def primary_count(self) -> int:
        return len(self.primary_entries())

    def _stats(self) -> dict:
        by_level = {}
        total_bytes = 0
        for e in self._entries.values():
            by_level[e.level] = by_level.get(e.level, 0) + 1
            try:
                fp = self._archive_dir / e.file
                total_bytes += fp.stat().st_size
            except OSError:
                pass
        return {
            "total_entries": len(self._entries),
            "total_bytes_on_disk": total_bytes,
            "by_level": by_level,
        }


# ---------------------------------------------------------------------------
# 3. PointerStore — 门面：文件 I/O + 索引 + 能量联动
# ---------------------------------------------------------------------------

class PointerStore:
    """
    门面类，提供 STORE / RECALL / MERGE / SEARCH。
    与能量系统联动：STORE 返还能量，RECALL 消耗能量。
    """

    def __init__(self, archive_dir: str, agent_id: str, scope: str):
        self._archive_dir = Path(archive_dir)
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        self.agent_id = agent_id
        self.scope = scope
        self._index = PointerIndex(self._archive_dir)
        self._index.load()

    # ---- 内部工具 ----

    @staticmethod
    def _make_id(content: str) -> str:
        """基于内容哈希 + 时间戳生成 8 位 ID。"""
        import hashlib
        ts = str(time.time())
        h = hashlib.sha256((content[:200] + ts).encode()).hexdigest()[:8]
        return f"ptr_{h}"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def _sanitize(name: str) -> str:
        return re.sub(r"[^\w\-]+", "_", name).strip("_")[:40] or "task"

    def _file_path(self, date_dir: str, task_dir: str, ptr_id: str) -> Path:
        d = self._archive_dir / date_dir / task_dir
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{ptr_id}.md"

    # ---- STORE ----

    def store(self, content: str, *, task: str, level: int = 0,
              children: Optional[List[str]] = None,
              frame_type: str = "step_detail", step_id: int = 0,
              parent_scope: Optional[str] = None,
              extra_tags: Optional[List[str]] = None,
              tokens: Optional[int] = None) -> Optional[str]:
        """
        将 content 持久化到磁盘，创建 index 条目，返回 pointer_id。
        失败（磁盘满等）返回 None，调用方回退到无存储模式。
        """
        if not content or len(content) < 10:
            return None

        ptr_id = self._make_id(content)
        tokens_est = tokens or self._estimate_tokens(content)
        date_dir = time.strftime("%Y-%m-%d")
        task_dir = self._sanitize(task)
        file_path = self._file_path(date_dir, task_dir, ptr_id)

        # 检查是否已存在（幂等）
        if self._index.get(ptr_id):
            return ptr_id

        # 生成摘要
        summary = content[:300].replace("\n", " ")

        # 写入 .md 文件
        meta_header = (
            f"# Pointer: {ptr_id}\n\n"
            f"- **Task**: {task}\n"
            f"- **Step**: {step_id}\n"
            f"- **Frame type**: {frame_type}\n"
            f"- **Scope**: {parent_scope or self.scope}\n"
            f"- **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- **Tokens**: {tokens_est}\n"
            f"- **Level**: {level}\n"
            f"- **Children**: {', '.join(children or [])}\n\n"
            f"---\n\n"
        )
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(meta_header)
                f.write(content)
        except OSError as e:
            print(f"  [PointerStore] STORE failed (disk full?): {e}")
            return None

        # 更新索引
        entry = PointerEntry(
            id=ptr_id,
            file=str(file_path.relative_to(self._archive_dir)),
            summary=summary,
            timestamp=time.time(),
            task=task,
            level=level,
            children=children or [],
            scope=parent_scope or self.scope,
            tokens=tokens_est,
            tags=list(set((extra_tags or []) + [task_dir])),
            agent_id=self.agent_id,
            frame_type=frame_type,
            step_id=step_id,
        )
        self._index.add(entry)
        self._index.save()
        print(f"  [PointerStore] STORE {ptr_id}: {tokens_est} tokens -> {file_path}")
        return ptr_id

    # ---- RECALL ----

    def recall(self, pointer_id: str, *, scope: Optional[str] = None,
               offset: int = 0, max_tokens: int = 2000) -> Optional[tuple[str, dict]]:
        """
        根据 pointer_id 召回内容。
        - scope 检查：caller 的 scope 必须是 entry scope 的前缀（父可见子）
        - offset/max_tokens：分页召回，防止 token 冲击
        - 返回 (content, metadata_dict) 或 None
        """
        entry = self._index.get(pointer_id)
        if not entry:
            return None

        # 作用域检查
        caller_scope = scope or self.scope
        if not (entry.scope == caller_scope
                or entry.scope.startswith(caller_scope + ".")
                or caller_scope.startswith(entry.scope + ".")):
            print(f"  [PointerStore] RECALL denied: {pointer_id} scope={entry.scope} vs caller={caller_scope}")
            return None

        # 读取文件
        file_path = self._archive_dir / entry.file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read()
        except (OSError, UnicodeDecodeError) as e:
            print(f"  [PointerStore] RECALL read failed: {e}")
            return None

        # 去掉元数据头（--- 之前的部分）
        parts = raw.split("\n---\n", 1)
        content = parts[1] if len(parts) > 1 else raw
        content = content.strip()

        # 分页：按字符近似（4 chars ≈ 1 token）
        char_offset = offset * 4
        char_budget = max_tokens * 4
        total_tokens = self._estimate_tokens(content)

        if char_offset >= len(content):
            return None

        sliced = content[char_offset:char_offset + char_budget]
        truncated = (char_offset + char_budget) < len(content)

        if truncated:
            next_offset = offset + max_tokens
            sliced += (
                f"\n\n... [内容已截断，共 {total_tokens} tokens，"
                f"offset={next_offset} 可继续召回] ..."
            )

        # 更新使用计数
        self._index.update(pointer_id, use_count=entry.use_count + 1)
        self._index.save()

        meta = {
            "id": pointer_id,
            "task": entry.task,
            "scope": entry.scope,
            "tokens": total_tokens,
            "injected_tokens": self._estimate_tokens(sliced),
            "offset": offset,
            "truncated": truncated,
            "step_id": entry.step_id,
            "frame_type": entry.frame_type,
        }
        print(f"  [PointerStore] RECALL {pointer_id}: offset={offset}, max={max_tokens}, "
              f"injected={meta['injected_tokens']}/{total_tokens} tokens")
        return sliced, meta

    # ---- MERGE ----

    def merge_pointers(self, task_prefix: str) -> Optional[str]:
        """
        将同 task 的 level-0 pointer 合并为 level-1 pointer。
        新 pointer 的 content 是一个索引表：id -> summary。
        返回新 pointer_id，或 None（无可合并）。
        """
        targets = [e for e in self._index.primary_entries()
                   if e.task.startswith(task_prefix)]
        if len(targets) < 2:
            return None

        # 按 timestamp 排序
        targets.sort(key=lambda e: e.timestamp)

        # 生成合并内容
        lines = [f"# Merged Pointer: {task_prefix}\n"]
        child_ids = []
        total_tokens = 0
        for e in targets:
            lines.append(f"\n## {e.id} ({e.tokens} tokens)\n")
            lines.append(f"- Summary: {e.summary[:200]}\n")
            lines.append(f"- Step: {e.step_id}, Scope: {e.scope}\n")
            child_ids.append(e.id)
            total_tokens += e.tokens

        merged_content = "\n".join(lines)
        merged_id = self._make_id(merged_content)

        # 写入合并文件
        date_dir = time.strftime("%Y-%m-%d")
        task_dir = self._sanitize(task_prefix)
        file_path = self._file_path(date_dir, task_dir, merged_id)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(merged_content)
        except OSError:
            return None

        entry = PointerEntry(
            id=merged_id,
            file=str(file_path.relative_to(self._archive_dir)),
            summary=f"Merged {len(targets)} pointers for {task_prefix}",
            timestamp=time.time(),
            task=task_prefix,
            level=1,
            children=child_ids,
            scope=self.scope,
            tokens=total_tokens,
            tags=["merged", task_dir],
            agent_id=self.agent_id,
            frame_type="merge",
        )
        self._index.add(entry)
        self._index.save()
        print(f"  [PointerStore] MERGE {merged_id}: {len(targets)} pointers -> {task_prefix}")
        return merged_id

    # ---- SEARCH ----

    def search_keywords(self, query: str, scope: str, limit: int = 5) -> List[dict]:
        """轻量级搜索，返回候选列表（含 tokens 大小，供 LLM 决策）。"""
        entries = self._index.search_summary(query, limit=limit * 2)
        # 过滤 scope
        results = []
        for e in entries:
            if e.scope == scope or e.scope.startswith(scope + ".") or scope.startswith(e.scope + "."):
                results.append({
                    "id": e.id,
                    "summary": e.summary[:200],
                    "tokens": e.tokens,
                    "task": e.task,
                    "step_id": e.step_id,
                    "use_count": e.use_count,
                })
                if len(results) >= limit:
                    break
        return results

    # ---- 统计 ----

    def primary_count(self) -> int:
        return self._index.primary_count()

    def sync(self) -> None:
        self._index.sync()

    def stats(self) -> dict:
        return self._index._stats()
