"""OPCode 轻量引擎 — 精简工具调用协议 + 解析 + 分发 + 执行。

用 @opcode 指令替代 JSON tool_use，省 token、易解析、可混用。
独立模块，零外部依赖（仅 re, uuid, time）。

格式：@opcode arg1 [arg2] [key=value ...]
示例：
    @r src/main.py                  → read_file(path="src/main.py")
    @ls src/ depth=2                → list_dir(path="src/", depth="2")
    @c echo hello world             → run_command(command="echo hello world")
    @rs deploy.sh bg=true           → run_script(path="deploy.sh", background=True)

用法：
    engine = OPCodeEngine()
    engine.register('r', 'read_file', positional=['path'], flags={'range': 'line_range'})
    calls, remaining = engine.parse(text)
    results = engine.dispatch(calls, handler_dict)
"""
import re
import uuid
import time as _time
from typing import Optional, Callable, Any

# ── 默认 Opcode 定义表 ──
DEFAULT_OPCODE_TABLE: dict[str, dict] = {
    'r': {
        'tool': 'read_file',
        'positional': ['path'],
        'flags': {'range': 'line_range'},
    },
    'ls': {
        'tool': 'list_dir',
        'positional': ['path'],
        'flags': {'depth': 'depth'},
    },
    'w': {
        'tool': 'write_file',
        'positional': ['path', 'content'],
        'flags': {},
    },
    'c': {
        'tool': 'run_command',
        'positional': ['command'],
        'flags': {'timeout': 'timeout'},
    },
    's': {
        'tool': 'spawn_agent',
        'positional': ['task'],
        'flags': {'ctx': 'context', 'mode': 'mode', 'sp': 'skip_plan'},
    },
    'sp': {
        'tool': 'savepoint',
        'positional': ['action'],
        'flags': {'name': 'name', 'label': 'label'},
    },
    'recall': {
        'tool': 'recall',
        'positional': ['pointer_id'],
        'flags': {'id': 'pointer_id', 'kw': 'keywords'},
    },
    'vi': {
        'tool': 'view_image',
        'positional': ['path'],
        'flags': {'prompt': 'prompt'},
    },
    'skill': {
        'tool': 'use_skill',
        'positional': ['skill_name'],
        'flags': {'args': 'args'},
    },
    'peer': {
        'tool': 'read_peer',
        'positional': ['peer_id'],
        'flags': {},
    },
    'rs': {
        'tool': 'run_script',
        'positional': ['path'],
        'flags': {'action': 'action', 'args': 'args', 'lang': 'lang',
                  'bg': 'background', 'delay': 'delay', 'interval': 'interval',
                  'timeout': 'timeout', 'job': 'job_id', 'code': 'content'},
    },
}

# 匹配 @opcode 开头的行
_RE_LINE = re.compile(r'^@(\w+)\s*(.*?)\s*$')

# 匹配 key=value（支持引号）
_RE_KV = re.compile(r'''(\w+)=("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[^\s]+)''')


def _strip_quotes(val: str) -> str:
    if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
        return val[1:-1]
    return val


def _tokenize_args(s: str) -> list[str]:
    """将参数字符串拆分为 token，支持引号包裹。"""
    tokens = []
    i = 0
    while i < len(s):
        if s[i] in ' \t':
            i += 1
            continue
        if s[i] in '"\'':
            quote = s[i]
            j = i + 1
            while j < len(s) and s[j] != quote:
                if s[j] == '\\':
                    j += 1
                j += 1
            tokens.append(s[i:j + 1] if j < len(s) else s[i:])
            i = j + 1
        else:
            j = i
            while j < len(s) and s[j] not in ' \t':
                j += 1
            tokens.append(s[i:j])
            i = j
    return tokens


# ── OPCode 引擎 ──

class OPCodeEngine:
    """轻量 OPCode 引擎：注册 opcode → 解析文本 → 分发执行。"""

    def __init__(self, table: dict = None):
        self.table: dict[str, dict] = dict(table or DEFAULT_OPCODE_TABLE)
        self._reverse: dict[str, str] = {v['tool']: k for k, v in self.table.items()}

    # ── 注册 ──

    def register(self, opcode: str, tool: str,
                 positional: list[str] = None, flags: dict[str, str] = None):
        """动态注册一个 opcode。"""
        self.table[opcode] = {
            'tool': tool,
            'positional': positional or [],
            'flags': flags or {},
        }
        self._reverse[tool] = opcode

    def unregister(self, opcode: str):
        """移除一个 opcode。"""
        spec = self.table.pop(opcode, None)
        if spec:
            self._reverse.pop(spec['tool'], None)

    # ── 解析 ──

    def parse_line(self, line: str) -> Optional[dict]:
        """解析单行 → {tool, args, opcode} 或 None。"""
        m = _RE_LINE.match(line.strip())
        if not m:
            return None
        op, args_str = m.group(1), m.group(2)
        spec = self.table.get(op)
        if not spec:
            return None

        # 分离 key=value 和位置参数
        named = {}
        for km in _RE_KV.finditer(args_str):
            k, v = km.group(1), _strip_quotes(km.group(2))
            named[k] = v
        remaining = _RE_KV.sub('', args_str).strip()

        pos_tokens = _tokenize_args(remaining) if remaining else []
        pos_names = spec['positional']
        args = {}

        # 填充位置参数（最后一个消费所有剩余 token）
        for i, pname in enumerate(pos_names):
            if i < len(pos_tokens):
                if i == len(pos_names) - 1:
                    args[pname] = ' '.join(_strip_quotes(t) for t in pos_tokens[i:])
                else:
                    args[pname] = _strip_quotes(pos_tokens[i])

        # 命名参数（覆盖位置映射）
        flags = spec['flags']
        for k, v in named.items():
            if k in flags:
                args[flags[k]] = v
            elif k in pos_names:
                args[k] = v

        return {'tool': spec['tool'], 'args': args, 'opcode': op}

    def parse(self, text: str) -> tuple[list[dict], str]:
        """从文本中提取所有 opcode 行。

        Returns:
            (tool_calls, remaining_text)
        """
        lines = text.split('\n')
        tool_calls = []
        remaining = []
        for line in lines:
            parsed = self.parse_line(line)
            if parsed:
                tool_calls.append(parsed)
            else:
                remaining.append(line)
        return tool_calls, '\n'.join(remaining).strip()

    # ── 格式转换 ──

    @staticmethod
    def to_tool_use(tc: dict) -> dict:
        """解析结果 → Anthropic tool_use content block。"""
        return {
            "type": "tool_use",
            "id": f"opc_{uuid.uuid4().hex[:12]}",
            "name": tc['tool'],
            "input": tc['args'],
        }

    @staticmethod
    def to_simple_namespace(tc: dict):
        """解析结果 → SimpleNamespace（兼容 tool_blocks 列表）。"""
        from types import SimpleNamespace
        return SimpleNamespace(
            type="tool_use", id=f"opc_{uuid.uuid4().hex[:12]}",
            name=tc['tool'], input=tc['args'])

    # ── 分发执行 ──

    def dispatch(self, tool_calls: list[dict],
                 handlers: dict[str, Callable],
                 budget: int = 4000,
                 agent_context: dict = None,
                 parallel_tools: set[str] = None) -> list[dict]:
        """执行解析出的工具调用。

        Args:
            tool_calls: parse() 返回的列表
            handlers: {tool_name: handler_fn(args, budget, agent_context) -> dict}
            budget: 工具结果字符预算
            agent_context: 传递给 handler 的上下文
            parallel_tools: 可并行执行的工具名集合（None=全部串行）

        Returns:
            [{tool, args, opcode, success, output}, ...]
        """
        if not parallel_tools:
            return [self._exec_one(tc, handlers, budget, agent_context) for tc in tool_calls]

        # 分组：可并行 vs 串行
        parallel = [tc for tc in tool_calls if tc['tool'] in parallel_tools]
        sequential = [tc for tc in tool_calls if tc['tool'] not in parallel_tools]
        results = []

        # 并行执行
        if parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=len(parallel)) as pool:
                futures = {pool.submit(self._exec_one, tc, handlers, budget, agent_context): tc
                           for tc in parallel}
                for f in as_completed(futures):
                    results.append(f.result())

        # 串行执行
        for tc in sequential:
            results.append(self._exec_one(tc, handlers, budget, agent_context))

        return results

    def _exec_one(self, tc: dict, handlers: dict, budget: int, ctx: dict) -> dict:
        """执行单个工具调用。"""
        handler = handlers.get(tc['tool'])
        if not handler:
            return {**tc, 'success': False, 'output': f"No handler for {tc['tool']}"}
        try:
            result = handler(tc['args'], budget, ctx or {})
            return {**tc, 'success': result.get('success', False),
                    'output': result.get('output', '')}
        except Exception as e:
            return {**tc, 'success': False, 'output': f"Error: {e}"}

    def execute(self, text: str, handlers: dict[str, Callable],
                budget: int = 4000, agent_context: dict = None,
                parallel_tools: set[str] = None) -> tuple[list[dict], str]:
        """一步完成：解析 + 分发执行。

        Returns:
            (results, remaining_text)
        """
        calls, remaining = self.parse(text)
        if not calls:
            return [], remaining
        results = self.dispatch(calls, handlers, budget, agent_context, parallel_tools)
        return results, remaining

    # ── System prompt 注入 ──

    def reference(self) -> str:
        """生成 opcode 参考表（注入 system prompt）。"""
        lines = ["OPCode 精简工具调用格式（每行一条，@ 开头）："]
        for op, spec in self.table.items():
            pos = ' '.join(f'<{p}>' for p in spec['positional'])
            flags = ' '.join(f'[{k}=...]' for k in spec['flags'])
            parts = f"@{op} {pos}"
            if flags:
                parts += f" {flags}"
            lines.append(f"  {parts:<40s} → {spec['tool']}")
        lines.append("命名参数用 key=value，值含空格时用引号包裹。")
        lines.append("OPCode 与标准工具调用可混用。")
        return '\n'.join(lines)

    def hint(self, tool_name: str) -> Optional[str]:
        """为指定工具生成 opcode 用法提示。"""
        op = self._reverse.get(tool_name)
        if not op:
            return None
        spec = self.table[op]
        pos = ' '.join(f'<{p}>' for p in spec['positional'])
        flags = ' '.join(f'[{k}=...]' for k in spec['flags'])
        parts = f"@{op} {pos}"
        if flags:
            parts += f" {flags}"
        return parts

    @property
    def tool_to_opcode(self) -> dict[str, str]:
        """工具名 → opcode 反向映射。"""
        return dict(self._reverse)


# ── 模块级便利函数（向后兼容）──

_default_engine = OPCodeEngine()

# 导出模块级 API
OPCODE_TABLE = _default_engine.table
TOOL_TO_OPCODE = _default_engine._reverse

parse_opcode_line = _default_engine.parse_line
parse_opcodes = _default_engine.parse
opcode_to_tool_call = OPCodeEngine.to_tool_use
generate_opcode_reference = _default_engine.reference
format_opcode_hint = _default_engine.hint


if __name__ == '__main__':
    # 引擎自测
    engine = OPCodeEngine()
    print(engine.reference())
    print()

    tests = [
        '@r src/main.py',
        '@r config.py range=L10-L20',
        '@ls src/',
        '@ls . depth=2',
        '@w out.txt "hello world"',
        '@c echo hello world',
        '@s "summarize the doc" ctx=brief',
        '@sp create name=checkpoint1',
        '@recall id=ptr_abc123',
        '@vi screenshot.png prompt=OCR',
        '@skill web_search args="python async"',
        '@rs deploy.sh bg=true delay=10',
    ]
    for t in tests:
        r = engine.parse_line(t)
        print(f"{t:<50s} → {r}")

    # 测试 dispatch（mock handler）
    print("\n--- Dispatch test ---")
    mock_handlers = {
        'read_file': lambda a, b, c: {'success': True, 'output': f"content of {a['path']}"},
        'list_dir': lambda a, b, c: {'success': True, 'output': f"files in {a.get('path', '.')}"},
    }
    text = "Check the file:\n@r agent.py\n@ls config/\n\nDone."
    results, remaining = engine.execute(text, mock_handlers,
                                        parallel_tools={'read_file', 'list_dir'})
    for r in results:
        print(f"  {r['opcode']} → {r['tool']}: success={r['success']} output={r['output']}")
    print(f"  remaining: {repr(remaining)}")
