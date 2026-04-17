"""经验留存库 — SQLite + FTS5 + 经验权重 + memory.md 目录"""
import sqlite3
import json
from pathlib import Path
from typing import List, Dict


class ExperienceStore:
    def __init__(self, store_dir: str = None):
        from agent import Config
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

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def record(self, task: str, plan: str = "", summary: str = "",
               lessons: str = "", tools_used: list = None,
               step_count: int = 0, success: bool = True):
        tools_json = json.dumps(tools_used or [], ensure_ascii=False)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO experiences (task, plan, summary, lessons, tools_used, step_count, success) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (task, plan, summary, lessons, tools_json, step_count, int(success))
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
