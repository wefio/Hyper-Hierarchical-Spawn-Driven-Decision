"""经验留存库 — SQLite + FTS5 + 经验权重 + memory.md 目录"""
import sqlite3
import json
from pathlib import Path
from typing import List, Dict


class ExperienceStore:
    def __init__(self, store_dir: str = None):
        from config import Config
        self.store_dir = Path(store_dir or Config.EXPERIENCE_DIR)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.store_dir / "experience.db"
        self.memory_path = self.store_dir / "memory.md"
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                plan TEXT,
                summary TEXT,
                lessons TEXT,
                tools_used TEXT,
                step_count INTEGER,
                success INTEGER DEFAULT 1,
                weight REAL DEFAULT 1.0,
                created_at TEXT DEFAULT (datetime('now'))
            )""")
            # 迁移：旧列名 pheromone → weight
            try:
                conn.execute("ALTER TABLE experiences RENAME COLUMN pheromone TO weight")
            except sqlite3.OperationalError:
                pass  # 列已是 weight 或不存在
            try:
                conn.execute("ALTER TABLE experiences ADD COLUMN weight REAL DEFAULT 1.0")
            except sqlite3.OperationalError:
                pass  # 列已存在
            # 迁移：tool_calls 列（JSON 格式记录 tool 调用序列）
            try:
                conn.execute("ALTER TABLE experiences ADD COLUMN tool_calls TEXT")
            except sqlite3.OperationalError:
                pass  # 列已存在
            try:
                conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS experiences_fts USING fts5(
                    task, summary, lessons,
                    content=experiences,
                    content_rowid=id
                )""")
                conn.execute("""CREATE TRIGGER IF NOT EXISTS experiences_ai AFTER INSERT ON experiences BEGIN
                    INSERT INTO experiences_fts(rowid, task, summary, lessons)
                    VALUES (new.id, new.task, new.summary, new.lessons);
                END""")
            except sqlite3.OperationalError:
                pass
            # ── skills 表 ──
            conn.execute("""CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                task_pattern TEXT,
                tool_sequence TEXT NOT NULL,
                preconditions TEXT,
                success_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                usage_count INTEGER DEFAULT 0,
                weight REAL DEFAULT 1.0,
                source_experience_ids TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                last_used TEXT
            )""")
            try:
                conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
                    name, description, task_pattern,
                    content=skills,
                    content_rowid=id
                )""")
                conn.execute("""CREATE TRIGGER IF NOT EXISTS skills_ai AFTER INSERT ON skills BEGIN
                    INSERT INTO skills_fts(rowid, name, description, task_pattern)
                    VALUES (new.id, new.name, new.description, new.task_pattern);
                END""")
            except sqlite3.OperationalError:
                pass
        # 从 @procedure 导入内置流程作为种子技能
        self._seed_builtin_skills()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def record(self, task: str, plan: str = "", summary: str = "",
               lessons: str = "", tools_used: list = None,
               step_count: int = 0, success: bool = True,
               tool_calls: list = None):
        tools_json = json.dumps(tools_used or [], ensure_ascii=False)
        tool_calls_json = json.dumps(tool_calls or [], ensure_ascii=False)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO experiences (task, plan, summary, lessons, tools_used, step_count, success, tool_calls) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (task, plan, summary, lessons, tools_json, step_count, int(success), tool_calls_json)
            )
        self._update_memory()

    def _update_memory(self):
        """更新 memory.md 目录文件"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, task, summary, step_count, success, weight, created_at "
                "FROM experiences ORDER BY id DESC LIMIT 20"
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

        lines = ["# Experience Store Index\n"]
        lines.append(f"Total records: {total} (showing latest 20)\n")
        for r in rows:
            status = "OK" if r[4] else "FAIL"
            ph_str = f" w={r[5]:.2f}" if r[5] != 1.0 else ""
            lines.append(f"- [#{r[0]}] [{status}] {r[6]} | {r[1][:60]}{ph_str}")
            if r[2]:
                lines.append(f"  > {r[2][:80]}")
            if r[3]:
                lines.append(f"  > Steps: {r[3]}")

        self.memory_path.write_text("\n".join(lines), encoding='utf-8')

    # ── 经验权重 ──

    def _fts_query(self, q: str) -> str:
        """中文拆字 + 英文 token 化，供 FTS5 使用"""
        tokens = []
        for ch in q:
            if '\u4e00' <= ch <= '\u9fff':
                tokens.append(f'"{ch}"')
        if tokens:
            return ' OR '.join(tokens)
        return q

    def update_weights(self, task: str, plan: str, success: bool):
        """成功/失败后更新相关记录的权重（targeted boost + 全局衰减）"""
        boost = 1.2 if success else 0.8
        fts_q = self._fts_query(task)
        try:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE experiences SET weight = weight * ? "
                    "WHERE id IN (SELECT rowid FROM experiences_fts WHERE experiences_fts MATCH ? LIMIT 5)",
                    (boost, fts_q)
                )
                conn.execute("UPDATE experiences SET weight = weight * 0.99")
        except sqlite3.OperationalError:
            pass

    def search(self, query: str, limit: int = 3) -> List[Dict]:
        fts_q = self._fts_query(query)
        try:
            with self._conn() as conn:
                rows = conn.execute("""
                    SELECT e.task, e.summary, e.lessons, e.tools_used, e.step_count, e.success, e.weight
                    FROM experiences_fts f
                    JOIN experiences e ON e.id = f.rowid
                    WHERE experiences_fts MATCH ?
                    ORDER BY (e.weight * -rank) DESC
                    LIMIT ?
                """, (fts_q, limit)).fetchall()
                if rows:
                    return [
                        {"task": r[0], "summary": r[1], "lessons": r[2],
                         "tools_used": r[3], "steps": r[4], "success": bool(r[5]),
                         "weight": r[6]}
                        for r in rows
                    ]
        except sqlite3.OperationalError:
            pass

        # LIKE 回退：搜索 task、summary、lessons
        with self._conn() as conn:
            pattern = f"%{query}%"
            rows = conn.execute(
                "SELECT task, summary, lessons, tools_used, step_count, success, weight "
                "FROM experiences WHERE task LIKE ? OR summary LIKE ? OR lessons LIKE ? "
                "ORDER BY weight DESC LIMIT ?",
                (pattern, pattern, pattern, limit)
            ).fetchall()
            return [
                {"task": r[0], "summary": r[1], "lessons": r[2],
                 "tools_used": r[3], "steps": r[4], "success": bool(r[5]),
                 "weight": r[6]}
                for r in rows
            ]

    def recent(self, limit: int = 5) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT task, summary, success, created_at FROM experiences ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [{"task": r[0], "summary": r[1], "success": bool(r[2]), "time": r[3]} for r in rows]

    # ── 技能库 ──

    def _seed_builtin_skills(self):
        """从 @procedure 装饰器注册的内置流程导入种子技能。"""
        try:
            from so100_tools import PROCEDURES
        except ImportError:
            return
        for name, proc in PROCEDURES.items():
            with self._conn() as conn:
                existing = conn.execute(
                    "SELECT id FROM skills WHERE name = ?", (name,)
                ).fetchone()
                if existing:
                    continue
                conn.execute(
                    "INSERT INTO skills (name, description, task_pattern, tool_sequence, "
                    "success_count) VALUES (?, ?, ?, ?, -1)",
                    (name, proc.get("description", ""),
                     proc.get("task_pattern", ""),
                     json.dumps(proc.get("skill_template", []), ensure_ascii=False))
                )

    def save_skill(self, name: str, task: str, tool_sequence: list,
                   description: str = "", source_exp_ids: list = None):
        """保存技能到 skills 表。"""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO skills (name, description, task_pattern, tool_sequence, "
                "source_experience_ids) VALUES (?, ?, ?, ?, ?)",
                (name, description, task, json.dumps(tool_sequence, ensure_ascii=False),
                 json.dumps(source_exp_ids or [], ensure_ascii=False))
            )

    def extract_skill(self, min_occurrences: int = 2, min_success_rate: float = 0.7):
        """从成功经验中提取可复用的技能。

        查找具有相似 tool_calls 的成功经验，按 task 分组，
        提取共同的操作序列存入 skills 表。
        """
        with self._conn() as conn:
            # 查找 tool_calls 非空的成功经验
            rows = conn.execute(
                "SELECT id, task, tool_calls FROM experiences "
                "WHERE success = 1 AND tool_calls IS NOT NULL AND tool_calls != '[]' "
                "ORDER BY id DESC LIMIT 200"
            ).fetchall()

        if len(rows) < min_occurrences:
            return

        # 按 task 关键词分组
        from collections import defaultdict
        groups = defaultdict(list)
        for row in rows:
            exp_id, task, tool_calls_json = row
            try:
                tcs = json.loads(tool_calls_json)
            except (json.JSONDecodeError, TypeError):
                continue
            if not tcs:
                continue
            # 提取 tool+action 签名
            sig = tuple((tc.get("tool", ""), tc.get("action", "")) for tc in tcs)
            if not sig:
                continue
            # 简化 task 为关键词
            key = task[:40]
            groups[key].append((exp_id, sig, len(tcs), tcs))

        for key, entries in groups.items():
            if len(entries) < min_occurrences:
                continue
            success_all = len(entries)
            # 取最常见的 tool_sequence
            from collections import Counter
            sig_counter = Counter(e[1] for e in entries)
            most_common_sig, count = sig_counter.most_common(1)[0]
            rate = count / success_all if success_all > 0 else 0
            if rate < min_success_rate:
                continue

            # 构建技能模板
            template = [
                {"step": i + 1, "tool": tc[0], "action": tc[1], "hint": ""}
                for i, tc in enumerate(most_common_sig)
            ]
            source_ids = [e[0] for e in entries if e[1] == most_common_sig]

            # 检查是否已存在
            with self._conn() as conn:
                existing = conn.execute(
                    "SELECT id FROM skills WHERE name = ?", (key,)
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE skills SET tool_sequence = ?, success_count = ?, "
                        "source_experience_ids = ?, weight = ? WHERE name = ?",
                        (json.dumps(template, ensure_ascii=False), count,
                         json.dumps(source_ids, ensure_ascii=False), rate, key)
                    )
                else:
                    conn.execute(
                        "INSERT INTO skills (name, description, task_pattern, tool_sequence, "
                        "success_count, source_experience_ids, weight) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (key, f"Extracted from {count} experiences", key,
                         json.dumps(template, ensure_ascii=False), count,
                         json.dumps(source_ids, ensure_ascii=False), rate)
                    )

    def search_skills(self, task: str, limit: int = 3) -> list[dict]:
        """检索与任务匹配的技能。返回操作序列作为参考。

        同时搜索:
        - @procedure 注册的内置流程（success_count=-1）
        - 从经验中提取的学习技能（success_count>=0）
        """
        results = []
        # FTS5 搜索
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT s.name, s.description, s.tool_sequence, s.success_count, "
                    "s.usage_count, s.weight, s.task_pattern "
                    "FROM skills_fts f "
                    "JOIN skills s ON s.id = f.rowid "
                    "WHERE skills_fts MATCH ? "
                    "ORDER BY (s.weight * -rank) DESC "
                    "LIMIT ?",
                    (task, limit)
                ).fetchall()
                for r in rows:
                    try:
                        tool_seq = json.loads(r[2])
                    except (json.JSONDecodeError, TypeError):
                        tool_seq = []
                    results.append({
                        "name": r[0],
                        "description": r[1],
                        "tool_sequence": tool_seq,
                        "success_count": r[3],
                        "usage_count": r[4],
                        "weight": r[5],
                        "task_pattern": r[6],
                        "is_builtin": r[3] < 0,
                    })
        except sqlite3.OperationalError:
            pass

        # LIKE 回退：按单词分别匹配
        if not results:
            with self._conn() as conn:
                # 拆 task 为关键词，分别 LIKE 匹配
                keywords = [w.strip() for w in task.replace(",", " ").split() if len(w.strip()) > 1]
                if not keywords:
                    keywords = [task]
                conditions = " OR ".join(["task_pattern LIKE ?" for _ in keywords])
                params = [f"%{kw}%" for kw in keywords]
                rows = conn.execute(
                    f"SELECT name, description, tool_sequence, success_count, "
                    f"usage_count, weight, task_pattern "
                    f"FROM skills WHERE {conditions} "
                    f"ORDER BY weight DESC LIMIT ?",
                    params + [limit]
                ).fetchall()
                for r in rows:
                    try:
                        tool_seq = json.loads(r[2])
                    except (json.JSONDecodeError, TypeError):
                        tool_seq = []
                    results.append({
                        "name": r[0],
                        "description": r[1],
                        "tool_sequence": tool_seq,
                        "success_count": r[3],
                        "usage_count": r[4],
                        "weight": r[5],
                        "task_pattern": r[6],
                        "is_builtin": r[3] < 0,
                    })

        # 更新 usage_count
        if results:
            with self._conn() as conn:
                names = [r["name"] for r in results]
                conn.execute(
                    "UPDATE skills SET usage_count = usage_count + 1, "
                    "last_used = datetime('now') WHERE name IN ({})".format(
                        ",".join("?" for _ in names)),
                    names
                )

        return results
