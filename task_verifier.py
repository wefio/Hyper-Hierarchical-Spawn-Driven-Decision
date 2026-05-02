"""通用任务验证框架 — 基于工具调用链的验证分派。

核心思路：agent 用了什么工具，就按什么标准验证任务完成。
- 文件操作   → 文件验证（存在+非空）
- 命令执行   → 命令验证（退出码+输出）
- 领域插件   → 通过 register_verifier() 注册（如 SO100 抓取验证）
- 默认       → 关键词验证（success/failure 信号检测）
"""

import os
from typing import Dict, List, Callable

# ── 工具 → 验证类别映射（内置类别） ──────────────────────────────────────────

TOOL_CATEGORY: Dict[str, str] = {
    # 文件操作
    "write_file": "file",
    "read_file": "file",
    # 命令执行
    "run_command": "command",
}

# 验证优先级：file > command > 注册的领域 > default
PRIORITY = ["file", "command"]

# 注册的领域验证器（由插件通过 register_verifier 添加）
_REGISTERED_VERIFIERS: Dict[str, Callable] = {}  # {category: verify_fn}


def register_category(category: str, tool_names: list[str]):
    """注册工具类别映射（由插件调用）。

    Args:
        category: 类别名（如 "so100"）
        tool_names: 属于此类的工具名列表
    """
    for name in tool_names:
        TOOL_CATEGORY[name] = category
    if category not in PRIORITY:
        # 领域验证器优先级高于内置（file/command），低于 default
        PRIORITY.insert(len(PRIORITY), category)


def register_verifier(category: str, verifier_fn: Callable):
    """注册领域验证器函数（由插件调用）。

    Args:
        category: 类别名（需先通过 register_category 注册）
        verifier_fn: (subtask_desc, result_text) -> dict
    """
    _REGISTERED_VERIFIERS[category] = verifier_fn


class TaskVerifier:
    """通用任务验证器：从 _current_tool_calls 提取工具类别，分派到对应验证器。"""

    def __init__(self, agent):
        self.agent = agent

    def verify(self, subtask_desc: str, result_text: str) -> dict:
        """
        从工具调用链推断验证策略，返回统一的验证结果。

        Returns:
            {"verified": bool, "reason": str, "feedback": str, "severity": str}
            severity: "pass" | "warn" | "fail"
        """
        categories = self._detect_categories()

        # 按优先级验证
        priority_order = PRIORITY + ["default"]
        last_result = None
        for cat in priority_order:
            if cat not in categories:
                continue
            verifier = getattr(self, f"_verify_{cat}", None)
            if not verifier:
                continue
            result = verifier(subtask_desc, result_text)
            last_result = result
            if result["severity"] == "fail":
                return result  # 一票否决

        return last_result or self._verify_default(subtask_desc, result_text)

    def _detect_categories(self) -> set:
        """从 _current_tool_calls 提取去重的工具类别。"""
        categories = set()
        for tc in self.agent._current_tool_calls:
            cat = TOOL_CATEGORY.get(tc.get("tool", ""), "default")
            if cat != "default":
                categories.add(cat)
        return categories or {"default"}

    # ── 文件验证器 ────────────────────────────────────────────────────

    def _verify_file(self, subtask_desc: str, result_text: str) -> dict:
        """
        文件任务验证：
        1. 从 write_file 调用记录中提取文件路径
        2. 检查文件存在 + 非空
        """
        written_files = []
        for tc in self.agent._current_tool_calls:
            if tc.get("tool") == "write_file" and tc.get("success"):
                path = tc.get("params", {}).get("path", "")
                if path:
                    written_files.append(path)

        if not written_files:
            return self._verify_default(subtask_desc, result_text)

        for fpath in written_files:
            if not os.path.exists(fpath):
                return {
                    "verified": False,
                    "reason": f"File not found: {fpath}",
                    "feedback": f"[系统复盘] 文件 {fpath} 不存在。写入可能失败。",
                    "severity": "fail",
                }
            try:
                size = os.path.getsize(fpath)
                if size == 0:
                    return {
                        "verified": False,
                        "reason": f"File is empty: {fpath}",
                        "feedback": f"[系统复盘] 文件 {fpath} 为空。",
                        "severity": "warn",
                    }
            except OSError:
                pass

        paths_str = ", ".join(written_files[:3])
        return {
            "verified": True,
            "reason": f"Files verified: {paths_str}",
            "feedback": "",
            "severity": "pass",
        }

    # ── 命令验证器 ────────────────────────────────────────────────────

    def _verify_command(self, subtask_desc: str, result_text: str) -> dict:
        """
        命令任务验证：
        1. 从 run_command 调用记录中提取退出码
        2. 检查退出码 == 0 且输出非空
        """
        command_results = []
        for tc in self.agent._current_tool_calls:
            if tc.get("tool") == "run_command":
                command_results.append(tc)

        if not command_results:
            return self._verify_default(subtask_desc, result_text)

        last_cmd = command_results[-1]
        exit_code = last_cmd.get("params", {}).get("exit_code")
        success = last_cmd.get("success", False)

        if not success:
            return {
                "verified": False,
                "reason": f"Command failed (exit_code={exit_code})",
                "feedback": "[系统复盘] 命令执行失败。检查命令语法和参数。",
                "severity": "fail",
            }

        return {
            "verified": True,
            "reason": f"Command succeeded (exit_code={exit_code or 0})",
            "feedback": "",
            "severity": "pass",
        }

    # ── 动态验证器分派 ────────────────────────────────────────────────

    def __getattr__(self, name: str):
        """拦截 _verify_{category} 调用，分派到注册的领域验证器。"""
        if name.startswith("_verify_") and name != "_verify_default":
            cat = name[len("_verify_"):]
            fn = _REGISTERED_VERIFIERS.get(cat)
            if fn:
                return lambda desc, text: fn(self.agent, desc, text)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    # ── 关键词验证器（默认/fallback）──────────────────────────────────

    def _verify_default(self, subtask_desc: str, result_text: str) -> dict:
        """
        通用关键词验证：检查结果文本中的成功/失败信号。
        """
        result_lower = result_text.lower()

        has_success = any(kw in result_lower for kw in [
            '"success": true', '"success":true', 'success: true',
            '✅', '成功', '已完成', 'confirmed', '任务完成', 'placed at',
        ])
        has_failure = any(kw in result_lower for kw in [
            '"success": false', '"success":false', 'success: false',
            '失败', 'error:', 'failed',
        ])

        if has_success:
            return {"verified": True, "reason": "success signal detected",
                    "feedback": "", "severity": "pass"}

        if has_failure:
            return {"verified": False, "reason": "failure signal detected",
                    "feedback": "[系统复盘] 检测到失败信号。建议检查并重试。",
                    "severity": "fail"}

        return {"verified": True, "reason": "no verification needed",
                "feedback": "", "severity": "pass"}
