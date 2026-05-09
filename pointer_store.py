"""Pointer Store — 磁盘缓存层（虚拟内存的 page table + swap disk）

把 context window 中即将被淘汰的内容持久化到本地文件，原地替换为 `摘要:ptr_id`。
支持多级 pointer、作用域隔离、能量联动、LRU-K 追踪。
"""

import hashlib
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
    level: int                   # 0=原始内容, 1=输入hash分组, 2=模板索引, 3=全局索引
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
    # ── 多级页表 + 复用追踪 ──
    input_hash: str = ""         # hash(template_id + input_args), L1 查找键
    template_id: str = ""        # 所属模板，L2 关联
    freshness_days: int = 30     # 保鲜期
    verification: str = ""       # system_verify: "pass" | "warn" | "fail"
    hits: int = 0                # 被查找次数
    reuses: int = 0              # 实际复用次数
    last_hit_at: float = 0.0
    last_reused_at: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.reuses / self.hits if self.hits > 0 else 0.0

    @property
    def is_fresh(self) -> bool:
        if not self.timestamp or self.freshness_days <= 0:
            return True  # no expiry set
        age_days = (time.time() - self.timestamp) / 86400.0
        return age_days <= self.freshness_days

    @property
    def is_reliable(self) -> bool:
        return self.verification in ("", "pass")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PointerEntry":
        # filter unknown keys for backward compat with old index files
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in field_names})


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

# ============================================================================
# RecallTLB — 快表：ptr_id → (content, meta) LRU 缓存
# ============================================================================

class RecallTLB:
    """recall 地址映射缓存。小容量（5-20 条），LRU 淘汰。

    命中 → 直接返回，零 I/O。
    预热 → 引擎确定性预加载（Agent 启动时 / 经验注入时 / recall 返回时）。
    """

    def __init__(self, max_entries: int = None):
        self.max_entries = max_entries or Config.HOT_POINTER_LIMIT
        self._cache: dict[str, tuple[str, dict]] = {}  # ptr_id → (content, meta)
        self._lru: list[str] = []

    def get(self, ptr_id: str) -> Optional[tuple[str, dict]]:
        if ptr_id in self._cache:
            self._lru.remove(ptr_id)
            self._lru.append(ptr_id)
            return self._cache[ptr_id]
        return None

    def put(self, ptr_id: str, content: str, meta: dict):
        while len(self._cache) >= self.max_entries:
            oldest = self._lru.pop(0)
            del self._cache[oldest]
        self._cache[ptr_id] = (content, meta)
        self._lru.append(ptr_id)

    def warm(self, ptr_ids: list[str]):
        """批量预加载到 TLB。已缓存的跳过，不存在的静默跳过。"""
        for pid in ptr_ids:
            if pid and pid not in self._cache:
                # 标记为待加载 — 实际加载由 PointerStore 在 recall 时完成
                pass  # warm 本身不触发 I/O；recall 调用时自然填充

    def invalidate(self, ptr_id: str):
        if ptr_id in self._cache:
            del self._cache[ptr_id]
            self._lru.remove(ptr_id)

    def __contains__(self, ptr_id: str) -> bool:
        return ptr_id in self._cache

    def __len__(self) -> int:
        return len(self._cache)


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
        self._tlb = RecallTLB()
        self._write_lock = __import__("threading").Lock()  # 并发写保护

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
        with self._write_lock:
            try:
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(meta_header)
                    f.write(content)
                    f.write(f"\n\n--- HASH:{content_hash}")
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
        - TLB 优先：命中直接返回，零 I/O
        - scope 检查：caller 的 scope 必须是 entry scope 的前缀
        - offset/max_tokens：分页召回，防止 token 冲击
        - 返回 (content, metadata_dict) 或 None
        """
        # ── TLB 查找 ──
        if offset == 0:  # 分页请求不走 TLB
            hit = self._tlb.get(pointer_id)
            if hit:
                print(f"  [TLB] Hit {pointer_id}")
                return hit

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

        # 文件完整性校验：提取并验证 SHA256 哈希
        if "\n--- HASH:" in content:
            content_part, hash_part = content.rsplit("\n--- HASH:", 1)
            stored_hash = hash_part.strip()
            computed = hashlib.sha256(content_part.encode("utf-8")).hexdigest()
            if stored_hash != computed:
                print(f"  [PointerStore] RECALL INTEGRITY FAIL: {pointer_id} hash mismatch")
                # 标记 entry 失效
                self._index.update(pointer_id, verification="tampered")
                self._index.save()
                return None
            content = content_part.strip()

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

        # ── TLB 写入 ──
        if offset == 0 and len(sliced) < 200_000:  # 只缓存全量截取且不太大的
            self._tlb.put(pointer_id, sliced, self._build_recall_meta(
                pointer_id, entry, total_tokens))

        # ── 确定性预热：step+1, children[0] ──
        next_ids = []
        if entry.step_id > 0:
            for sib in self._index.by_scope(entry.scope):
                if sib.step_id == entry.step_id + 1 and sib.id != pointer_id:
                    next_ids.append(sib.id)
                    break
        if entry.children:
            next_ids.append(entry.children[0])
        for nid in next_ids[:3]:
            if nid not in self._tlb:
                self._prefetch(nid)

        meta = self._build_recall_meta(pointer_id, entry, total_tokens,
                                        offset, truncated, sliced)
        print(f"  [PointerStore] RECALL {pointer_id}: offset={offset}, max={max_tokens}, "
              f"injected={meta['injected_tokens']}/{total_tokens} tokens")
        return sliced, meta

    def _build_recall_meta(self, ptr_id: str, entry, total_tokens: int,
                           offset: int = 0, truncated: bool = False,
                           sliced: str = "") -> dict:
        return {
            "id": ptr_id,
            "task": entry.task,
            "scope": entry.scope,
            "tokens": total_tokens,
            "injected_tokens": self._estimate_tokens(sliced) if sliced else 0,
            "offset": offset,
            "truncated": truncated,
            "step_id": entry.step_id,
            "frame_type": entry.frame_type,
        }

    def warm_tlb(self, ptr_ids: list[str]):
        """批量预热 TLB：预加载 ptr_ids 中未缓存的指针。"""
        for pid in ptr_ids:
            if pid and pid not in self._tlb:
                self._prefetch(pid)

    def _prefetch(self, ptr_id: str):
        """预加载 ptr_id 内容到 TLB（出错静默跳过）。"""
        try:
            entry = self._index.get(ptr_id)
            if not entry or not entry.file:
                return
            fp = self._archive_dir / entry.file
            with open(fp, encoding="utf-8") as f:
                raw = f.read()
            parts = raw.split("\n---\n", 1)
            content = (parts[1] if len(parts) > 1 else raw).strip()
            if len(content) < 200_000:
                meta = self._build_recall_meta(ptr_id, entry,
                                               self._estimate_tokens(content))
                self._tlb.put(ptr_id, content, meta)
        except Exception:
            pass

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

    # ========================================================================
    # 多级页表操作
    # ========================================================================

    def register_page_table(self, template_id: str, scope: str = "root") -> str:
        """创建 L2 模板索引页表，返回 table_ptr_id。"""
        ptr_id = self._make_id(f"pt_{template_id}_{time.time()}")
        entry = PointerEntry(
            id=ptr_id,
            file="",  # 页表不存文件，只在内存
            summary=f"Template page table: {template_id}",
            timestamp=time.time(),
            task=template_id,
            level=2,                  # L2 = 模板索引
            children=[],
            scope=scope,
            tokens=0,
            tags=["page_table", template_id],
            agent_id=self.agent_id,
            frame_type="page_table",
            template_id=template_id,
        )
        self._index.add(entry)
        self._index.save()
        return ptr_id

    def add_to_page_table(self, table_ptr_id: str, input_hash: str,
                          child_ptr_id: str, summary: str = "",
                          verification: str = "pass",
                          freshness_days: int = 30) -> Optional[str]:
        """在模板索引下注册 L1 条目（input_hash 组）。返回 L1 ptr_id。"""
        table = self._index.get(table_ptr_id)
        if not table:
            return None
        # 检查是否已有同 hash 的 L1 条目
        for cid in table.children:
            existing = self._index.get(cid)
            if existing and existing.input_hash == input_hash:
                # 追加子指针
                if child_ptr_id not in existing.children:
                    existing.children.append(child_ptr_id)
                existing.timestamp = time.time()
                existing.verification = verification
                existing.freshness_days = freshness_days
                existing.tokens += self._index.get(child_ptr_id).tokens if self._index.get(child_ptr_id) else 0
                self._index.update(cid, children=existing.children,
                                   timestamp=existing.timestamp,
                                   verification=verification,
                                   freshness_days=freshness_days,
                                   tokens=existing.tokens)
                self._index.save()
                return cid
        # 新建 L1 条目
        l1_id = self._make_id(f"l1_{input_hash[:16]}")
        entry = PointerEntry(
            id=l1_id,
            file="",
            summary=summary[:300],
            timestamp=time.time(),
            task=table.task,
            level=1,               # L1 = input hash 分组
            children=[child_ptr_id],
            scope=table.scope,
            tokens=self._index.get(child_ptr_id).tokens if self._index.get(child_ptr_id) else 0,
            tags=["input_group", input_hash[:16]],
            agent_id=self.agent_id,
            frame_type="page_table_entry",
            input_hash=input_hash,
            template_id=table.template_id,
            freshness_days=freshness_days,
            verification=verification,
        )
        self._index.add(entry)
        # 更新 L2 的子指针列表
        table.children.append(l1_id)
        self._index.update(table_ptr_id, children=table.children)
        self._index.save()
        return l1_id

    def lookup(self, table_ptr_id: str, input_hash: str) -> Optional[PointerEntry]:
        """L1 精确查找：在模板索引中按 input_hash 匹配。"""
        table = self._index.get(table_ptr_id)
        if not table:
            return None
        for cid in table.children:
            entry = self._index.get(cid)
            if entry and entry.input_hash == input_hash:
                return entry
        return None

    def list_page_entries(self, table_ptr_id: str) -> list:
        """返回模板索引下所有 L1 条目（用于 L2 级别查找）。"""
        table = self._index.get(table_ptr_id)
        if not table:
            return []
        result = []
        for cid in table.children:
            e = self._index.get(cid)
            if e:
                result.append({
                    "id": e.id, "input_hash": e.input_hash,
                    "summary": e.summary[:200], "tokens": e.tokens,
                    "verification": e.verification,
                    "freshness_days": e.freshness_days,
                    "is_fresh": e.is_fresh, "is_reliable": e.is_reliable,
                    "hits": e.hits, "reuses": e.reuses,
                    "hit_rate": e.hit_rate, "children_count": len(e.children),
                })
        return result

    def validate_entry(self, ptr_id: str) -> str:
        """有效性校验。返回 "ok" | "stale" | "suspect" | "missing"。"""
        entry = self._index.get(ptr_id)
        if not entry:
            return "missing"
        if not entry.is_fresh:
            return "stale"
        if not entry.is_reliable:
            return "suspect"
        # 文件完整性：L0 条目需要文件存在
        if entry.level == 0 and entry.file:
            if not (self._archive_dir / entry.file).exists():
                return "missing"
        return "ok"

    def record_hit(self, ptr_id: str, reused: bool = False):
        """记录一次查找或复用。"""
        entry = self._index.get(ptr_id)
        if not entry:
            return
        now = time.time()
        kwargs = {"hits": entry.hits + 1, "last_hit_at": now}
        if reused:
            kwargs["reuses"] = entry.reuses + 1
            kwargs["last_reused_at"] = now
        self._index.update(ptr_id, **kwargs)
        self._index.save()

    def evict_cold(self, table_ptr_id: str, min_hit_rate: float = 0.1,
                   min_hits: int = 20) -> int:
        """淘汰低命中率 L1 条目。返回淘汰数量。"""
        table = self._index.get(table_ptr_id)
        if not table:
            return 0
        removed = 0
        survivors = []
        for cid in list(table.children):
            entry = self._index.get(cid)
            if entry and entry.hits >= min_hits and entry.hit_rate < min_hit_rate:
                # 回收 L0 子指针的能量
                for child_id in entry.children:
                    self._index.remove(child_id)
                self._index.remove(cid)
                removed += 1
            else:
                survivors.append(cid)
        if removed:
            self._index.update(table_ptr_id, children=survivors)
            self._index.save()
        return removed

    def find_reusable(self, table_ptr_id: str, input_hash: str,
                      task_desc: str, limit: int = 3) -> dict:
        """四级查找：返回 {level: int, action: str, pointers: list}。

        L1: 精确 hash 匹配 → action="skip" (复用，跳过执行)
        L2: 同模板其他条目  → action="reference" (注入参考)
        L3: scope 上溯全局   → action="experience" (经验注入)
        L4: 返回空，交由 experience_store FTS5 兜底
        """
        result = {"level": 0, "action": "execute", "pointers": []}

        # ── L1: 精确匹配 ──
        l1_entry = self.lookup(table_ptr_id, input_hash)
        if l1_entry:
            validity = self.validate_entry(l1_entry.id)
            if validity == "ok":
                self.record_hit(l1_entry.id, reused=True)
                return {"level": 1, "action": "skip",
                        "pointers": [{
                            "ptr_id": l1_entry.children[-1] if l1_entry.children else l1_entry.id,
                            "input_hash": l1_entry.input_hash,
                            "summary": l1_entry.summary[:200],
                            "tokens": l1_entry.tokens,
                            "verification": l1_entry.verification,
                            "is_fresh": True,
                        }]}
            else:
                # 降级但仍返回
                self.record_hit(l1_entry.id, reused=False)
                result = {"level": 1, "action": "reference",
                          "pointers": [{
                              "ptr_id": l1_entry.children[-1] if l1_entry.children else l1_entry.id,
                              "input_hash": l1_entry.input_hash,
                              "summary": l1_entry.summary[:200],
                              "tokens": l1_entry.tokens,
                              "verification": l1_entry.verification,
                              "is_fresh": False,
                              "validity": validity,
                          }]}
                return result

        # ── L2: 同模板其他条目 ──
        siblings = self.list_page_entries(table_ptr_id)
        if siblings:
            self.record_hit(table_ptr_id)  # record hit on the template itself
            return {"level": 2, "action": "reference",
                    "pointers": siblings[:limit]}

        # ── L3: scope 上溯 ──
        table = self._index.get(table_ptr_id)
        scope = table.scope if table else self.scope
        scope_parts = scope.split(".")
        for depth in range(len(scope_parts) - 1, 0, -1):
            ancestor = ".".join(scope_parts[:depth])
            candidates = self._index.by_scope(ancestor)
            if candidates:
                result_list = []
                for e in candidates[:limit]:
                    result_list.append({
                        "ptr_id": e.id,
                        "summary": e.summary[:200],
                        "tokens": e.tokens,
                        "task": e.task,
                        "is_fresh": e.is_fresh,
                    })
                return {"level": 3, "action": "experience",
                        "pointers": result_list}

        return result  # L4: 交由 experience_store


# ============================================================================
# GlobalPointerStore — 全局指针库（L3 跨模板搜索）
# ============================================================================

class GlobalPointerStore:
    """全局指针库：包装 PointerStore，提供跨模板搜索和自动注册。

    内部使用 L3 scope="root" 页表，所有注册的 NodeReport 指针按关键词
    索引。提供与 experience_store 互补的快速指针查找。
    """

    def __init__(self, store: PointerStore):
        self._store = store
        self._ensure_global_table()

    def _ensure_global_table(self) -> str:
        """确保全局 L3 索引页表存在。"""
        existing = self._store._index.by_scope("root")
        for e in existing:
            if e.level == 3 and e.frame_type == "global_index":
                return e.id
        return self._store.register_page_table("__global__", scope="root")

    def register(self, report_dict: dict, ptr_id: str = "",
                 task_desc: str = "") -> Optional[str]:
        """从 NodeReport 注册指针到全局库。"""
        if not ptr_id:
            return None
        summary = report_dict.get("summary", task_desc)[:300]
        if not summary:
            summary = report_dict.get("task_summary", "")[:300]
        verification = report_dict.get("system_verify", {}).get("severity", "pass")
        tokens = report_dict.get("tokens_input", 0) + report_dict.get("tokens_output", 0)
        # 用 task 关键词作 input_hash
        input_h = hashlib.sha256(task_desc[:200].encode()).hexdigest()[:16]
        table_id = self._ensure_global_table()
        return self._store.add_to_page_table(
            table_id, input_h, ptr_id,
            summary=summary, verification=verification,
        )

    def search(self, task_desc: str, limit: int = 5) -> list[dict]:
        """L3 全局搜索：基于关键词 + scope 上溯。"""
        entries = self._store._index.search_summary(task_desc, limit=limit * 2)
        results = []
        for e in entries:
            if e.summary:
                results.append({
                    "ptr_id": e.id,
                    "summary": e.summary[:200],
                    "tokens": e.tokens,
                    "task": e.task,
                    "verification": e.verification,
                    "is_fresh": e.is_fresh,
                    "is_reliable": e.is_reliable,
                    "hit_rate": e.hit_rate,
                    "timestamp": e.timestamp,
                })
                if len(results) >= limit:
                    break
        return results

    def list_recent(self, limit: int = 10) -> list[dict]:
        """最近注册的指针（按时间排序）。"""
        table_id = self._ensure_global_table()
        entries = self._store.list_page_entries(table_id)
        entries.sort(key=lambda e: e.get("hits", 0), reverse=True)
        return entries[:limit]

    @property
    def stats(self) -> dict:
        table_id = self._ensure_global_table()
        all_entries = self._store.list_page_entries(table_id)
        total_hits = sum(e.get("hits", 0) for e in all_entries)
        total_reuses = sum(e.get("reuses", 0) for e in all_entries)
        return {
            "total_entries": len(all_entries),
            "total_hits": total_hits,
            "total_reuses": total_reuses,
            "global_hit_rate": total_reuses / total_hits if total_hits > 0 else 0.0,
        }
