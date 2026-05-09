"""Skill system — loads and matches SKILL.md files from skills/ directory."""
import re
from pathlib import Path
from typing import Dict, List, Optional

from config import Config


class Skill:
    def __init__(self, name: str, description: str, keywords: list[str], content: str):
        self.name = name
        self.description = description
        self.keywords = keywords
        self.content = content


class SkillManager:
    _skills_cache: Dict[str, 'Skill'] = {}
    _loaded = False

    def __init__(self, skills_dir: str = None):
        self.skills_dir = Path(skills_dir or Config.SKILLS_DIR)
        self.skills = SkillManager._skills_cache
        if not SkillManager._loaded:
            self._load_all()
            SkillManager._loaded = True

    def _load_all(self):
        if not self.skills_dir.exists():
            return
        for skill_path in sorted(self.skills_dir.iterdir()):
            if not skill_path.is_dir():
                continue
            md_file = skill_path / "SKILL.md"
            if not md_file.exists():
                continue
            try:
                skill = self._parse(md_file)
                self.skills[skill.name] = skill
                print(f"  [Skill] {skill.name}: {skill.description[:50]}")
            except Exception as e:
                print(f"  [Skill] parse failed {md_file.name}: {e}")

    @staticmethod
    def _parse(filepath: Path) -> Skill:
        text = filepath.read_text(encoding='utf-8')
        m = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', text, re.DOTALL)
        if not m:
            raise ValueError("Missing frontmatter")
        meta_text, body = m.group(1), m.group(2).strip()
        meta = {}
        for line in meta_text.split('\n'):
            if ':' in line:
                k, v = line.split(':', 1)
                meta[k.strip()] = v.strip()
        name = meta.get('name', filepath.parent.name)
        desc = meta.get('description', '')
        kw_str = meta.get('keywords', '')
        keywords = [k.strip().lower() for k in kw_str.split(',') if k.strip()]
        return Skill(name, desc, keywords, body)

    def get(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def match(self, task: str, top_n: int = 1) -> list[Skill]:
        task_lower = task.lower()
        scored = []
        for skill in self.skills.values():
            score = 0
            if skill.name.lower() in task_lower:
                score += 10
            for kw in skill.keywords:
                if kw in task_lower:
                    score += 3
            for word in skill.description.lower().split():
                if len(word) > 2 and word in task_lower:
                    score += 1
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_n]]

    def list_names(self) -> list[str]:
        return list(self.skills.keys())
