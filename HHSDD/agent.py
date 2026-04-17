import sys
import io
import os
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
import subprocess
import json
import time
import random
import math
import argparse
import re
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
from enum import Enum, auto

import anthropic

# ============ 配置 ============
class Config:
    # ---- API ----
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")
    MODEL = os.getenv("MODEL", "MiniMax-M2.7")
    MINIMAX_API_HOST = os.getenv("MINIMAX_API_HOST", "https://api.minimax.chat")
    MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
    MINIMAX_API_HOST = os.getenv("MINIMAX_API_HOST", "https://api.minimax.chat")
    RATE_LIMIT = float(os.getenv("RATE_LIMIT", "1"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

    # ---- 能量系统 ----
    # 能量 = token，1E = 1 token。总能量 = CONTEXT_BUDGET - CONTEXT_RESERVE
    CONTEXT_BUDGET = int(os.getenv("CONTEXT_BUDGET", "204800"))       # 模型上下文窗口上限
    CONTEXT_RESERVE = int(os.getenv("CONTEXT_RESERVE", "50000"))      # 保留给 system/tools/紧急交付
    STEP_OVERHEAD = int(os.getenv("STEP_OVERHEAD", "1000"))           # 每步固定能量开销
    TOKEN_COST_INPUT = float(os.getenv("TOKEN_COST_INPUT", "1.0"))    # 输入能量系数
    TOKEN_COST_OUTPUT = float(os.getenv("TOKEN_COST_OUTPUT", "3.0"))  # 输出能量系数
    SPAWN_INVEST_MIN = int(os.getenv("SPAWN_INVEST_MIN", "2000"))     # spawn 最低投资
    SPAWN_RESERVE_RATIO = float(os.getenv("SPAWN_RESERVE_RATIO", "0.15"))  # spawn 预留比例
    REWARD_BASE = float(os.getenv("REWARD_BASE", "5000"))             # 终端奖励基数
    CMD_REFUND_PER_SEC = float(os.getenv("CMD_REFUND_PER_SEC", "500"))  # 命令超时退款速率
    TOOL_ENERGY_COST = float(os.getenv("TOOL_ENERGY_COST", "200"))      # 单次工具调用预扣能量

    # ---- 上下文管理 ----
    COMPRESSION_THRESHOLD = float(os.getenv("COMPRESSION_THRESHOLD", "0.9"))  # 上下文占比触发压缩
    TOOL_RESULT_BUDGET = int(os.getenv("TOOL_RESULT_BUDGET", "4000"))         # 单次工具结果字符上限
    MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "120000"))         # 消息列表总字符上限

    # ---- 安全网（正常由能量系统停止） ----
    MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))
    MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "50"))
    MAX_SPAWN_DEPTH = int(os.getenv("MAX_SPAWN_DEPTH", "0"))  # 0=不限制

    # ---- 路径 ----
    WORK_DIR = os.getenv("AGENT_WORK_DIR", "./agent_state")
    SKILLS_DIR = os.getenv("SKILLS_DIR", "./skills")
    EXPERIENCE_DIR = os.getenv("EXPERIENCE_DIR", "./experience_store")
    SCRIPTS_DIR = os.getenv("SCRIPTS_DIR", "./scripts")
    HISTORY_DIR = os.getenv("HISTORY_DIR", "./history")
    DEBUG_DIR = os.getenv("AGENT_DEBUG_DIR", "./debug_messages")

    # ---- 策略 ----
    ENABLE_SUBTASK_DEDUP = os.getenv("ENABLE_SUBTASK_DEDUP", "true").lower() == "true"
    SUBTASK_DEDUP_THRESHOLD = float(os.getenv("SUBTASK_DEDUP_THRESHOLD", "0.6"))

    @classmethod
    def load_env(cls, env_file: str = ".env"):
        env_path = Path(env_file)
        if not env_path.exists():
            return
        with open(env_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

Config.load_env()
# .env 加载后再读 API key（.env 可能覆盖环境变量）
Config.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", Config.ANTHROPIC_API_KEY)
Config.ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", Config.ANTHROPIC_BASE_URL)

# ============ 上下文长度管理 ============
def smart_truncate(text: str, budget: int, label: str = "") -> str:
    """智能截断：内容在预算内原样返回，超出则保留首尾，中间省略"""
    if len(text) <= budget:
        return text
    tag = f"\n... [{label}共{len(text)}字，截断至{budget}字] ...\n" if label else "\n... [已截断] ...\n"
    tag_len = len(tag)
    head_size = (budget - tag_len) // 2
    tail_size = budget - tag_len - head_size
    return text[:head_size] + tag + text[-tail_size:]

def measure_stack(stack) -> int:
    return sum(len(f.content) for f in stack)

# ============ 栈帧 ============
@dataclass
class StackFrame:
    type: str      # "constraint", "plan", "step_detail", "summary", "merge", "history"
    content: str
    step_id: int = 0
    level: int = 0
    agent_id: str = ""

# ============ 缓存抽象 ============
class APIClient:
    """Anthropic API 封装：限流、重试、计能、调试转储"""

    def __init__(self, client: anthropic.Anthropic, config: type,
                 cache_provider: CacheProvider, energy_mgr):
        self.client = client
        self.cfg = config
        self.cache = cache_provider
        self.em = energy_mgr
        self._last_call_time = 0.0
        self._debug_dir = Path(config.DEBUG_DIR)

    def call(self, messages: list, system=None, tools=None,
             max_tokens: int = 4096, bypass_energy: bool = False):
        """调用 API，返回 Message 对象"""
        kwargs = {"model": self.cfg.MODEL, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        self._dump("req", kwargs)

        elapsed = time.time() - self._last_call_time
        if elapsed < self.cfg.RATE_LIMIT:
            time.sleep(self.cfg.RATE_LIMIT - elapsed)

        max_retries = self.cfg.MAX_RETRIES
        overload_retries = 0
        max_overload = max_retries * 3

        for attempt in range(max_retries + max_overload):
            try:
                result = self.client.messages.create(timeout=120.0, **kwargs)
                self._last_call_time = time.time()
                self._dump("res", result)

                if not bypass_energy and hasattr(result, 'usage') and self.em:
                    inp = getattr(result.usage, 'input_tokens', 0) or 0
                    out = getattr(result.usage, 'output_tokens', 0) or 0
                    m = self.cache.extract_metrics(result)
                    cost = self.cache.compute_cost(inp, out, m)
                    self.em.consume(cost, force=True)
                    self.em.commit_spent(cost)
                    self.em.add_tokens(inp, out)
                    if m.cache_read_tokens > 0:
                        print(f"  [Cache] read {m.cache_read_tokens} tokens (saved {m.cache_read_tokens * 0.9:.0f} cost)")

                return result

            except anthropic.RateLimitError:
                backoff = (2 ** min(attempt, 5)) + random.uniform(0, 1)
                print(f"  [Rate limit] retry in {backoff:.1f}s ({attempt+1}/{max_retries})...")
                time.sleep(backoff)
                if attempt >= max_retries - 1:
                    raise RuntimeError(f"Rate limited after {max_retries} retries")
                continue

            except anthropic.APIStatusError as e:
                if e.status_code in (529, 500, 502, 503, 504):
                    overload_retries += 1
                    backoff = min(30, (2 ** min(overload_retries, 5)) * 2) + random.uniform(0, 3)
                    print(f"  [ServerError {e.status_code}] retry in {backoff:.1f}s ({overload_retries}/{max_overload})...")
                    time.sleep(backoff)
                    if overload_retries < max_overload:
                        continue
                raise

            except (anthropic.APIConnectionError, TimeoutError) as e:
                backoff = (2 ** attempt) + random.uniform(0, 2)
                print(f"  [Timeout/Connection] {type(e).__name__}: retry in {backoff:.1f}s ({attempt+1}/{self.cfg.MAX_RETRIES})...")
                time.sleep(backoff)
                continue

            except anthropic.APIError as e:
                if "too long" in str(e).lower() or "context" in str(e).lower():
                    raise ContextTooLongError()
                raise RuntimeError(f"API Error: {e}")

        raise RuntimeError(f"Failed after {self.cfg.MAX_RETRIES} retries")

    def _dump(self, kind: str, data):
        from datetime import datetime
        try:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(self._debug_dir.glob("*.json"))
            if len(existing) > 200:
                for old in existing[:50]:
                    old.unlink()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = self._debug_dir / f"{kind}_{ts}.json"
            content = json.dumps(data, default=str, ensure_ascii=False, indent=2)
            path.write_text(content, encoding='utf-8')
        except Exception:
            pass


@dataclass
class CacheMetrics:
    """Provider-agnostic 缓存指标"""
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

class CacheProvider:
    """缓存提供商抽象基类。各 provider 实现差异化 token 费率。"""
    def extract_metrics(self, response) -> CacheMetrics:
        return CacheMetrics()

    def cache_control(self) -> dict:
        """返回 cache_control 标记（如 {"type": "ephemeral"}）"""
        return {}

    def compute_cost(self, input_tokens: int, output_tokens: int,
                     metrics: CacheMetrics) -> float:
        """计算含缓存的 token 成本"""
        return input_tokens * Config.TOKEN_COST_INPUT + output_tokens * Config.TOKEN_COST_OUTPUT

class AnthropicCacheProvider(CacheProvider):
    """Anthropic 兼容 API 主动缓存：cache_read=0.1x, cache_write=1.25x, 非缓存=1x"""
    def extract_metrics(self, response) -> CacheMetrics:
        usage = getattr(response, 'usage', None)
        if not usage:
            return CacheMetrics()
        return CacheMetrics(
            cache_read_tokens=getattr(usage, 'cache_read_input_tokens', 0) or 0,
            cache_write_tokens=getattr(usage, 'cache_creation_input_tokens', 0) or 0,
        )

    def cache_control(self) -> dict:
        return {"type": "ephemeral"}

    def compute_cost(self, input_tokens: int, output_tokens: int,
                     metrics: CacheMetrics) -> float:
        # input_tokens / cache_read / cache_write 三者互不重叠
        cost = (input_tokens * Config.TOKEN_COST_INPUT      # 非缓存 input
                + metrics.cache_read_tokens * 0.1           # 缓存读取 0.1x
                + metrics.cache_write_tokens * 1.25         # 缓存写入 1.25x
                + output_tokens * Config.TOKEN_COST_OUTPUT) # output
        return cost

# ============ 原生工具定义 ============
TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path. Returns file content as text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative or absolute file path"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "list_dir",
        "description": "List directory contents. Shows files with sizes and subdirectories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path", "default": "."}
            }
        }
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Content to write"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "view_image",
        "description": "Analyze an image using a vision model. Supports local file paths and URLs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Image file path or URL"},
                "question": {"type": "string", "description": "Question about the image", "default": "Describe the content of this image"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "spawn_agent",
        "description": "Spawn a sub-agent to independently handle a subtask. The sub-agent gets its own execution context and returns a compressed summary. Use 'explore' mode for uncertain/investigative tasks (high risk, high reward). Use 'exploit' mode for well-understood tasks (low risk, stable).",
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The subtask description for the spawned agent"},
                "context": {"type": "string", "description": "Optional additional context from the parent task"},
                "mode": {"type": "string", "enum": ["explore", "exploit"], "description": "explore=risky but rewarding on success; exploit=safe and stable. Default: exploit"}
            },
            "required": ["task"]
        }
    },
    {
        "name": "run_command",
        "description": "Execute a shell command and return its stdout/stderr. Use for running scripts, installing packages, making HTTP requests, etc. Commands run in the project root directory with a 30-second timeout.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)", "default": 30}
            },
            "required": ["command"]
        }
    },
    {
        "type": "skill_executor",
        "name": "use_skill",
        "description": "Activate a loaded skill for the current task. Costs a deposit which is fully refunded on task success, forfeited on failure. Prefer skills over ad-hoc approaches.",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "Name of the skill to activate (must be loaded)"}
            },
            "required": ["skill_name"]
        }
    },
]

def _build_cached_tools(cache_provider: CacheProvider) -> list:
    """构建带 cache_control 的工具定义（缓存结果，避免重复 deepcopy）"""
    cc = cache_provider.cache_control()
    if not cc:
        return TOOL_DEFINITIONS
    # 仅复制需要修改的最后一个元素
    tools = TOOL_DEFINITIONS[:-1] + [{**TOOL_DEFINITIONS[-1], "cache_control": cc}]
    return tools

def _get_energy_mgr(agent_context: dict) -> 'BayesianEnergyManager':
    """从 agent_context 中提取能量管理器"""
    parent = agent_context.get("parent_agent") if agent_context else None
    return parent.energy_manager if parent else None


def energy_hooks(*, deduct=None, post=None, settle=None, abort_msg="Insufficient energy"):
    """通用能量生命周期装饰器。

    Args:
        deduct: (args, em) -> cost | False — 预扣能量，返回False则中止
        post: (args, result, em, cost) -> None — 正常执行后返费/记录
        settle: (args, result, em, cost, agent_context) -> None — 始终执行（finally），用于结算
        abort_msg: deduct返回False时的错误消息
    """
    from functools import wraps
    def decorator(handler):
        @wraps(handler)
        def wrapper(args, budget, agent_context=None):
            em = _get_energy_mgr(agent_context)
            cost = None
            if em and deduct:
                cost = deduct(args, em)
                if cost is False:
                    return {"success": False, "output": abort_msg}
            result = None
            try:
                result = handler(args, budget, agent_context)
                if em and post:
                    post(args, result, em, cost)
                return result
            finally:
                if em and settle:
                    settle(args, result, em, cost, agent_context)
        return wrapper
    return decorator


# ── 能量钩子定义 ──
def _cmd_deduct(args, em):
    return em.pre_consume_for_cmd(args.get("command", ""))

def _cmd_refund(args, result, em, cost):
    if cost and cost > 0:
        refund = em.refund_for_cmd(
            args.get("command", ""), cost,
            result.get("execution_time", 0), result.get("success", False))
        if refund > 0.01:
            actual = result.get("execution_time", 0)
            print(f"  [Energy] +{refund:.1f} refunded ({actual:.1f}s, {'ok' if result['success'] else 'FAIL'})")

def _skill_deduct(args, em):
    deposit = float(Config.SPAWN_INVEST_MIN)
    if not em.invest(deposit):
        return False
    return deposit

def _skill_post(args, result, em, cost):
    em._skill_deposit = 2000.0
    em._skill_name = args.get("skill_name", "")

def _spawn_deduct(args, em):
    if not em.should_spawn():
        return False
    # 投资额也按比例：当前能量的 5%，最低 2000
    invest_amount = max(float(Config.SPAWN_INVEST_MIN), em.energy * 0.05)
    if not em.invest(invest_amount):
        return False
    if not em.consume_spawn_overhead():
        return False
    return invest_amount

def _spawn_settle(args, result, em, cost, agent_context=None):
    if cost is None:
        return
    parent_agent = (agent_context or {}).get("parent_agent")
    spawn_depth = parent_agent.depth + 1 if parent_agent else 1
    fc = getattr(parent_agent, 'flowchart', None) if parent_agent else None
    # 释放父节点的保留能量
    reserve = em._reserve_stack[-1] if em._reserve_stack else 0
    em.release_spawn_reserve()
    success = result.get("success", False) if result else False
    mode = result.get("_mode", "exploit") if result else "exploit"
    em.settle_investment(cost, success, mode)
    if not success:
        em.update_spawn(False)
    # 生命周期计数器：记录释放 + 结算（不再创建单独节点）
    if fc:
        try:
            fc.record_lifecycle(spawn_depth, release=reserve, settle=cost)
        except Exception:
            pass




# ============ Mermaid 流程图记录器 ============
class FlowchartRecorder:
    """增量式流程图记录器，每次追加到文件，不重写。"""
    def __init__(self, work_dir: str):
        from pathlib import Path
        self.file_path = Path(work_dir) / "flowchart.md"
        self.nodes = set()
        self.edges = set()
        self.tool_calls: dict[tuple, int] = {}  # (step, tool_name) -> count
        # spawn 生命周期计数器：合并 absorb/release/settle/exp_prop 为每深度汇总
        self.spawn_seq: dict[int, int] = {}  # depth -> 序号
        self._lifecycle: dict[int, dict] = {}  # depth -> {count, full, summary, release, settle, exp, modes}
        self._init_file()

    def _init_file(self):
        try:
            self.file_path.write_text("```mermaid\ngraph TD\n", encoding='utf-8')
        except Exception:
            pass

    def _append(self, text: str):
        try:
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(text + "\n")
        except Exception:
            pass

    def add_node(self, node_id: str, label: str, shape: str = "rect"):
        """添加节点。shape: start | end | task | subtask | step | tool | agent | reclaim | truncate | rect"""
        if node_id in self.nodes:
            return
        self.nodes.add(node_id)
        indent = "    "
        if shape == "start":
            line = f'{indent}{node_id}(["{label}"])'
        elif shape == "end":
            line = f'{indent}{node_id}(["{label}"])'
        elif shape == "task":
            line = f'{indent}{node_id}["{label}"]'
        elif shape == "subtask":
            line = f'{indent}{node_id}["{label}"]'
        elif shape == "step":
            line = f'{indent}{node_id}["{label}"]'
        elif shape == "tool":
            line = f'{indent}{node_id}{{"{label}"}}'
        elif shape == "agent":
            line = f'{indent}{node_id}{{{{"{label}"}}}}'
        elif shape == "reclaim":
            line = f'{indent}{node_id}[/"{label}"/]'
        elif shape == "truncate":
            line = f'{indent}{node_id}{{"{label}"}}'
        elif shape == "energy":
            line = f'{indent}{node_id}("{label}")'
        elif shape == "absorb":
            line = f'{indent}{node_id}[["{label}"]]'
        elif shape == "decision":
            line = f'{indent}{node_id}{{{{"{label}"}}}}'
        else:
            line = f'{indent}{node_id}["{label}"]'
        self._append(line)

    def add_edge(self, from_id: str, to_id: str, label: str = ""):
        edge = (from_id, to_id, label)
        if edge in self.edges:
            return
        self.edges.add(edge)
        indent = "    "
        if label:
            line = f'{indent}{from_id} -->|{label}| {to_id}'
        else:
            line = f'{indent}{from_id} --> {to_id}'
        self._append(line)


    def merged_tool(self, from_id: str, tool_name: str, step_counter: int, depth: int = 0) -> str:
        """同一 step 内合并同名工具调用，首次创建节点，后续只计数。
        返回节点 ID（用于后续连接）。
        """
        key = (depth, step_counter, tool_name)
        self.tool_calls[key] = self.tool_calls.get(key, 0) + 1
        count = self.tool_calls[key]
        node_id = f"tool_d{depth}_{step_counter}_{tool_name}"
        label = f"{tool_name} (x{count})"
        if count == 1:
            # 首次调用：创建节点并连接
            self.add_node(node_id, f"工具: {label}", shape="tool")
            self.add_edge(from_id, node_id)
        return node_id

    def add_note(self, node_id: str, note: str):
        """为节点添加注释"""
        self._append(f'    note for {node_id} "{note}"')

    def next_spawn_seq(self, depth: int) -> int:
        """返回该深度下一个 spawn 序号（1-based）"""
        self.spawn_seq[depth] = self.spawn_seq.get(depth, 0) + 1
        return self.spawn_seq[depth]

    def record_lifecycle(self, depth: int, **kw):
        """累加 spawn 生命周期事件，finalize 时统一写入汇总节点。
        kw: spawn=1, absorb_full=1, absorb_summary=1, release=float, settle=float, exp=int, mode=str
        """
        if depth not in self._lifecycle:
            self._lifecycle[depth] = {
                "count": 0, "full": 0, "summary": 0,
                "release": 0.0, "settle": 0.0, "exp": 0, "modes": []
            }
        d = self._lifecycle[depth]
        if "spawn" in kw:
            d["count"] += kw["spawn"]
        if "absorb_full" in kw:
            d["full"] += 1
        if "absorb_summary" in kw:
            d["summary"] += 1
        if "release" in kw:
            d["release"] += kw["release"]
        if "settle" in kw:
            d["settle"] += kw["settle"]
        if "exp" in kw:
            d["exp"] += kw["exp"]
        if "mode" in kw:
            d["modes"].append(kw["mode"])

    def finalize(self):
        # 写入每深度 spawn 生命周期汇总节点
        for depth in sorted(self._lifecycle):
            d = self._lifecycle[depth]
            if d["count"] == 0:
                continue
            explore_n = d["modes"].count("explore")
            exploit_n = d["modes"].count("exploit")
            parts = [f"{d['count']}次spawn"]
            if explore_n:
                parts.append(f"{explore_n}探索")
            if exploit_n:
                parts.append(f"{exploit_n}利用")
            if d["full"]:
                parts.append(f"{d['full']}完整吸收")
            if d["summary"]:
                parts.append(f"{d['summary']}摘要吸收")
            if d["release"] > 0:
                parts.append(f"释放{d['release']:.0f}E")
            if d["exp"] > 0:
                parts.append(f"经验{d['exp']}条")
            label = f"深度{depth} 生命周期: {', '.join(parts)}"
            node_id = f"lifecycle_d{depth}"
            self.add_node(node_id, label, shape="energy")
        self._append("```")


# ============ 工具执行器 ============
class ToolExecutor:
    @staticmethod
    def execute(name: str, args: Dict[str, Any], budget: int,
                agent_context: dict = None) -> Dict[str, Any]:
        handlers = {
            "read_file": ToolExecutor._read_file,
            "list_dir": ToolExecutor._list_dir,
            "write_file": ToolExecutor._write_file,
            "view_image": ToolExecutor._view_image,
            "spawn_agent": energy_hooks(deduct=_spawn_deduct, settle=_spawn_settle,
                                        abort_msg="Insufficient energy to spawn sub-agent")(ToolExecutor._spawn_agent),
            "run_command": energy_hooks(deduct=_cmd_deduct, post=_cmd_refund)(ToolExecutor._run_command),
            "use_skill": energy_hooks(deduct=_skill_deduct, post=_skill_post,
                                      abort_msg="Insufficient energy for skill deposit")(ToolExecutor._use_skill),
        }
        handler = handlers.get(name)
        if not handler:
            return {"success": False, "output": f"Unknown tool: {name}. Available: {', '.join(handlers.keys())}"}
        try:
            return handler(args, budget, agent_context or {})
        except Exception as e:
            return {"success": False, "output": f"Tool error: {e}"}
    def _read_file(args: dict, budget: int, agent_context: dict = None) -> dict:
        path = args.get("path", "")
        if not path:
            return {"success": False, "output": "Missing path parameter"}
        path = str(path)
        if not os.path.exists(path):
            return {"success": False, "output": f"File not found: {path}"}
        if os.path.isdir(path):
            return {"success": False, "output": f"Is a directory: {path}, use list_dir"}
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        total = len(content)
        output = smart_truncate(content, budget, label=f"{os.path.basename(path)} ")
        return {"success": True, "output": output, "truncated": total > budget, "original_size": total}

    @staticmethod
    def _list_dir(args: dict, budget: int, agent_context: dict = None) -> dict:
        path = str(args.get("path", "."))
        if not os.path.exists(path):
            return {"success": False, "output": f"Path not found: {path}"}
        if not os.path.isdir(path):
            return {"success": False, "output": f"Not a directory: {path}"}
        entries = []
        for entry in sorted(os.listdir(path)):
            full = os.path.join(path, entry)
            if os.path.isdir(full):
                entries.append(f"  {entry}/")
            else:
                size = os.path.getsize(full)
                entries.append(f"  {entry}  ({size} bytes)")
        output = "\n".join(entries) or "(empty)"
        return {"success": True, "output": smart_truncate(output, budget, label="dir ")}

    @staticmethod
    def _write_file(args: dict, budget: int, agent_context: dict = None) -> dict:
        path = args.get("path", "")
        content = args.get("content", "")
        if not path:
            return {"success": False, "output": "Missing path parameter"}
        path = str(path)
        # 脚本文件自动路由到 scripts/ 目录
        if path.endswith(('.py', '.sh', '.bat', '.ps1')) and not os.path.dirname(path):
            scripts_dir = Path(Config.SCRIPTS_DIR)
            scripts_dir.mkdir(parents=True, exist_ok=True)
            path = str(scripts_dir / path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(str(content))
        # 更新共享文件缓存（供子 agent 直接复用）
        if agent_context:
            parent_agent = agent_context.get("parent_agent")
            if parent_agent and hasattr(parent_agent, 'shared_files'):
                rel = os.path.relpath(path, parent_agent.work_dir)
                parent_agent.shared_files[rel] = os.path.abspath(path)
        return {"success": True, "output": f"Wrote {path} ({len(str(content))} chars). Run with: {path}"}

    @staticmethod
    def _view_image(args: dict, budget: int, agent_context: dict = None) -> dict:
        """调用 MiniMax VLM API 分析图片（伪装 MCP 来源）"""
        import requests as _req
        import base64
        path = args.get("path", "")
        prompt = args.get("question", "Describe the content of this image in detail")
        if not path:
            return {"success": False, "output": "Missing path parameter"}
        path = str(path)
        is_url = path.startswith(("http://", "https://"))
        if not is_url and not os.path.exists(path):
            return {"success": False, "output": f"File not found: {path}"}
        try:
            # 图片 → base64 data URL
            if is_url:
                resp = _req.get(path, timeout=15)
                resp.raise_for_status()
                ct = resp.headers.get("content-type", "").lower()
                fmt = "png" if "png" in ct else "webp" if "webp" in ct else "jpeg"
                b64 = base64.b64encode(resp.content).decode()
            else:
                ext = os.path.splitext(path)[1].lower()
                fmt = ".png" if ext == ".png" else ".webp" if ext == ".webp" else "jpeg"
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
            data_url = f"data:image/{fmt};base64,{b64}"
            # 调 MiniMax VLM（伪装 MCP）
            r = _req.post(
                f"{Config.MINIMAX_API_HOST}/v1/coding_plan/vlm",
                json={"prompt": prompt, "image_url": data_url},
                headers={
                    "Authorization": f"Bearer {Config.ANTHROPIC_API_KEY}",
                    "Content-Type": "application/json",
                    "MM-API-Source": "Minimax-MCP",
                },
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            content = data.get("content", "")
            if not content:
                return {"success": False, "output": f"VLM returned empty: {data}"}
            return {"success": True, "output": smart_truncate(content, budget, label="image ")}
        except Exception as e:
            return {"success": False, "output": f"Image analysis error: {e}"}

    @staticmethod
    def _run_command(args: dict, budget: int, agent_context: dict = None) -> dict:
        """执行 shell 命令（纯执行，能量预扣/返还由 execute 层处理）"""
        command = args.get("command", "")
        timeout = min(args.get("timeout", 30), 60)
        if not command:
            return {"success": False, "output": "Missing command parameter", "execution_time": 0}

        start_time = time.time()
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                encoding='utf-8', errors='replace', timeout=timeout,
                cwd=os.getcwd(),
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "LANG": "en_US.UTF-8"}
            )
            actual_seconds = time.time() - start_time
            cmd_success = result.returncode == 0
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                stderr_clean = result.stderr[:500]
                output += ("\nSTDERR:\n" + stderr_clean) if output else stderr_clean
            if not output:
                output = f"(exit code {result.returncode}, no output)"
            return {"success": cmd_success, "output": smart_truncate(output, budget, label="cmd "),
                    "exit_code": result.returncode, "execution_time": actual_seconds}
        except subprocess.TimeoutExpired:
            return {"success": False, "output": f"Command timed out after {timeout}s",
                    "execution_time": timeout}
        except Exception as e:
            return {"success": False, "output": f"Command error: {e}",
                    "execution_time": time.time() - start_time}

    @staticmethod
    def _use_skill(args: dict, budget: int, agent_context: dict) -> dict:
        """激活 Skill（纯逻辑，押金由 execute 层处理）"""
        skill_name = args.get("skill_name", "")
        if not skill_name:
            return {"success": False, "output": "Missing skill_name parameter"}
        parent = agent_context.get("parent_agent") if agent_context else None
        if not parent:
            return {"success": False, "output": "No agent context"}
        # 检查 Skill 是否已加载
        loaded = [s.name for s in parent.active_skills]
        if skill_name not in loaded:
            available = ", ".join(loaded) if loaded else "(none loaded)"
            return {"success": False, "output": f"Skill '{skill_name}' not loaded. Available: {available}"}
        # 将 Skill 内容注入到约束帧（如果尚未包含完整内容）
        for skill in parent.active_skills:
            if skill.name == skill_name:
                constraint = parent.stack[0].content
                if skill.content not in constraint:
                    parent.stack[0] = StackFrame("constraint",
                                                  constraint + f"\n\n## Skill: {skill.name}\n{skill.content}", level=0)
                break
        return {"success": True, "output": f"Skill '{skill_name}' activated."}

    @staticmethod
    def _spawn_agent(args: dict, budget: int, agent_context: dict) -> dict:
        """派生子 Agent（纯执行，能量由装饰器管理）"""
        task = args.get("task", "")
        context = args.get("context", "")
        if not task:
            return {"success": False, "output": "Missing task parameter", "_mode": "exploit"}

        current_depth = agent_context.get("depth", 0)
        if Config.MAX_SPAWN_DEPTH > 0 and current_depth >= Config.MAX_SPAWN_DEPTH:
            return {"success": False, "output": f"Maximum spawn depth ({Config.MAX_SPAWN_DEPTH}) reached", "_mode": "exploit"}

        parent_agent = agent_context.get("parent_agent")
        energy_mgr = parent_agent.energy_manager if parent_agent else None

        # 能量感知：决定 explore/exploit 模式
        llm_specified_mode = "mode" in args
        base_mode = args.get("mode", "exploit")
        if energy_mgr and not llm_specified_mode:
            explore_prob = energy_mgr.get_role_probability()
            mode = "explore" if random.random() < explore_prob else "exploit"
        elif energy_mgr and llm_specified_mode:
            explore_prob = energy_mgr.get_role_probability()
            suggested = "explore" if random.random() < explore_prob else "exploit"
            if suggested != base_mode:
                print(f"  [EnergyAware] LLM={base_mode}, energy suggests {suggested} (p_explore={explore_prob:.2f})")
            mode = base_mode
        else:
            mode = base_mode

        try:
            child_system = agent_context.get("system_prompt", "你是一个AI编程助手。")
            child = Agent(
                system_prompt=child_system,
                depth=current_depth + 1,
                parent=parent_agent
            )
            # 记录子 Agent 节点（带序号）
            spawn_node = f"spawn_{current_depth+1}_{child.agent_id}"
            args["_child_id"] = child.agent_id  # 供 spawn_settle 定位节点
            if parent_agent and hasattr(parent_agent, 'flowchart'):
                try:
                    seq = parent_agent.flowchart.next_spawn_seq(current_depth + 1)
                    parent_agent.flowchart.add_node(spawn_node, f"子 Agent #{seq} (深度 {current_depth+1})", shape="agent")
                    current_step = getattr(parent_agent, "_current_step_node", None)
                    if current_step:
                        parent_agent.flowchart.add_edge(current_step, spawn_node)
                    # 传递 spawn 节点给子 Agent，使其步骤挂在该节点下
                    child._fc_parent_spawn_node = spawn_node
                    # 记录 spawn 生命周期事件
                    parent_agent.flowchart.record_lifecycle(current_depth + 1, spawn=1, mode=mode)
                except Exception:
                    pass

            child.emit_event(SubAgentEventType.STARTED, f"Spawning ({mode}) for: {task[:50]}")
            # 传递父级 context
            if context:
                child.stack.append(StackFrame("history", f"Parent context: {context}", level=0))
            # 共享文件路径（父已完成文件，子可直接读）
            if parent_agent and parent_agent.shared_files:
                files_lines = ["## Shared files (completed by parent agents):"]
                for rel, abspath in parent_agent.shared_files.items():
                    files_lines.append(f"- {rel}: {abspath}")
                child.stack.append(StackFrame("history", "\n".join(files_lines), level=0))
            # Skills 继承（单例 SkillManager 只加载一次，auto_skill 只在顶层匹配）
            if parent_agent and parent_agent.active_skills:
                skill_lines = ["## Skills (inherited from parent):"]
                for s in parent_agent.active_skills:
                    skill_lines.append(f"### {s.name}\n{s.content}")
                child.stack.append(StackFrame("history", "\n".join(skill_lines), level=0))
            # 失败经验传递：将父级最近的失败工具调用告知子 Agent
            if parent_agent:
                parent_frame = parent_agent._build_failure_experience()
                if parent_frame:
                    child.stack.append(StackFrame("experience", parent_frame, level=0))

            result = child.run(task, max_steps=Config.MAX_STEPS)
            child.emit_event(SubAgentEventType.TASK_COMPLETED, f"Completed: {task[:50]}")

            if energy_mgr:
                energy_mgr.update_spawn(True)

            # 子 Agent 结果吸收：父 Agent 根据能量决定保留完整内容或仅摘要
            output = result or "(no output)"
            if parent_agent:
                parent_agent._absorb_child_result(output, task)
                # 子 Agent 的经验上提到父 Agent，最终由根 Agent 统一 flush
                if child._pending_experiences:
                    n = len(child._pending_experiences)
                    parent_agent._pending_experiences.extend(child._pending_experiences)
                    parent_agent._save_pending_experiences()
                    print(f"  [Experience] propagated {n} records from child")
                    child._pending_experiences = []
                    # 生命周期计数器：经验上提
                    if parent_agent.flowchart:
                        try:
                            parent_agent.flowchart.record_lifecycle(child.depth, exp=n)
                        except Exception:
                            pass
            return {"success": True, "output": smart_truncate(output, budget, label="sub-agent "),
                    "_mode": mode}
        except Exception as e:
            if parent_agent:
                parent_agent.energy_manager.process_event(
                    SubAgentEvent("child", SubAgentEventType.FATAL_ERROR, str(e)))
            return {"success": False, "output": f"Sub-agent error: {e}", "_mode": mode}

# ============ Skill 系统 ============
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

# ============ 异常 ============
class ContextTooLongError(Exception):
    pass

# ============ 贝叶斯步数预估器 ============
class StepEstimator:
    """Gamma-Exponential 共轭：根据已完成子任务步数预测剩余步数"""
    def __init__(self, total_subtasks: int, prior_mean: float = 3.0, prior_strength: float = 1.0):
        self.N = total_subtasks
        self.alpha = prior_mean * prior_strength
        self.beta = prior_strength
        self.completed = 0

    def update(self, steps_taken: int):
        self.alpha += steps_taken
        self.beta += 1
        self.completed += 1

    def predict_remaining(self) -> tuple:
        """返回 (期望, 下限5%, 上限95%)"""
        remaining = self.N - self.completed
        if remaining <= 0:
            return 0, 0, 0
        shape = remaining * self.alpha
        rate = self.beta
        mean = shape / rate
        std = math.sqrt(shape) / rate
        return mean, max(0, mean - 1.645 * std), mean + 1.645 * std

    def predict_total(self, steps_done: int) -> tuple:
        rem_mean, rem_low, rem_high = self.predict_remaining()
        return steps_done + rem_mean, steps_done + rem_low, steps_done + rem_high

# ============ 子 Agent 事件系统 ============
class SubAgentEventType(Enum):
    STARTED = auto()
    STEP_COMPLETED = auto()
    TOOL_FAILED = auto()
    FATAL_ERROR = auto()
    TASK_COMPLETED = auto()
    MAX_STEPS_REACHED = auto()
    TIMEOUT = auto()

@dataclass
class SubAgentEvent:
    agent_id: str
    type: SubAgentEventType
    message: str = ""
    data: dict = None
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

# ============ 能量管理 ============
class BayesianEnergyManager:
    """双层能量预算：软约束（流动资金）+ 硬约束（总预算）+ 投资-收益模型"""
    def __init__(self, total_energy: float = 154800.0, step_overhead: int = 1000,
                 cost_step: float = 2.0,
                 cost_tool: float = 0.2,
                 explore_roi: float = 0.5):
        # 双层约束
        self.total_energy = total_energy   # 硬上限（不可突破）
        self.total_spent = 0.0             # 已净消耗（不可逆，只增不减）
        self.energy = total_energy          # 流动资金（可预扣、可返还、可获奖金）
        self.step_overhead = step_overhead  # 每步固定 token 开销
        self.cost_step = cost_step
        self.cost_tool = cost_tool
        # Token 使用统计
        self._total_tokens = 0
        self._total_input = 0
        self._total_output = 0
        # Spawn 保留栈：每次 spawn 扣 20% 作为父节点保底，子返回后释放
        self._reserve_stack: list[float] = []
        # 探索投资回报率
        self.explore_roi = explore_roi
        # Beta(α, β) 后验：任务完成概率
        self.done_alpha = 1.0
        self.done_beta = 1.0
        # 子任务成功率
        self.subtask_beta: Dict[str, tuple] = {}
        # 派生成功率
        self.spawn_alpha = 2.0
        self.spawn_beta = 2.0
        # 连续无进展计数
        self._no_progress_count = 0
        # 失败计数
        self.failure_count = 0
        # 命令耗时先验 (Gamma): pattern -> (alpha_sum_time, beta_count)
        self.cmd_time_prior: Dict[str, tuple] = {}
        self.energy_per_second = Config.CMD_REFUND_PER_SEC  # 每秒退款速率
        # 命令模式连续失败计数
        self._cmd_fail_streak: Dict[str, int] = {}

    # ---- 基础能量操作 ----
    def consume(self, amount: float, *, force: bool = False) -> bool:
        """预扣能量（流动资金），不累加 total_spent。
        force=True: 无条件扣除（用于已发生的 API 调用成本记账），但不低于 -total_energy*0.1。
        force=False: 能量不足时不扣除，返回 False。"""
        if not force and self.energy < amount:
            return False
        new_energy = self.energy - amount
        if force:
            floor = -self.total_energy * 0.1  # 下限：总预算的 -10%
            new_energy = max(new_energy, floor)
        self.energy = new_energy
        return True

    def commit_spent(self, amount: float):
        """确认净消耗，累加到 total_spent（不可逆）"""
        self.total_spent += amount

    # Token 使用统计
    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_input_tokens(self) -> int:
        return self._total_input

    @property
    def total_output_tokens(self) -> int:
        return self._total_output

    def add_tokens(self, input_t: int, output_t: int):
        self._total_tokens += input_t + output_t
        self._total_input += input_t
        self._total_output += output_t

    def consume_step_overhead(self) -> bool:
        """每步固定 token 开销，能量不足时不扣"""
        if not self.consume(self.step_overhead):
            return False
        self.commit_spent(self.step_overhead)
        return True

    def consume_tool(self) -> bool:
        """每次工具调用的额外开销，能量不足时不扣"""
        if not self.consume(self.cost_tool):
            return False
        self.commit_spent(self.cost_tool)
        return True

    def consume_spawn_overhead(self) -> bool:
        """每次派生子 Agent：扣留当前剩余能量的 15% 作为父节点保底。
        子 agent 获得剩余 85%。子返回后通过 release_spawn_reserve() 释放。
        动态阈值：保留量低于 max(energy*0.01, 500) 时拒绝 spawn。"""
        cost = self.energy * Config.SPAWN_RESERVE_RATIO
        min_threshold = max(self.energy * 0.01, 500)
        if cost < min_threshold:
            return False
        self.energy -= cost
        self._reserve_stack.append(cost)
        print(f"  [Spawn] Reserved: {cost:.0f}E (15% of {self.energy + cost:.0f}), "
              f"child pool: {self.energy:.0f}")
        return True

    def release_spawn_reserve(self):
        """子 agent 返回后释放父节点的保留能量。"""
        if self._reserve_stack:
            reserve = self._reserve_stack.pop()
            self.energy += reserve
            print(f"  [Spawn] Reserve released: {reserve:.0f}E -> energy {self.energy:.0f}")

    def update_subtask(self, sub_id: str, success: bool):
        a, b = self.subtask_beta.get(sub_id, (1.0, 1.0))
        if success:
            a += 1
            self._no_progress_count = 0
        else:
            b += 1
            self._no_progress_count += 1
        self.subtask_beta[sub_id] = (a, b)

    def update_done(self, is_done: bool):
        if is_done:
            self.done_alpha += 1
            self._no_progress_count = 0
        else:
            self.done_beta += 0.2  # 软惩罚

    def update_spawn(self, success: bool):
        if success:
            self.spawn_alpha += 1
        else:
            self.spawn_beta += 1

    def p_done(self) -> float:
        return self.done_alpha / (self.done_alpha + self.done_beta)

    def p_subtask_success(self, sub_id: str) -> float:
        a, b = self.subtask_beta.get(sub_id, (1.0, 1.0))
        return a / (a + b)

    def should_spawn(self, energy_ratio: float = 0.3) -> bool:
        p = self.spawn_alpha / (self.spawn_alpha + self.spawn_beta)
        return (p > energy_ratio * 0.5
                and self.energy > self.total_energy * energy_ratio
                and self.total_spent < self.total_energy * 0.8)

    # ── 能量感知调度 ──

    def get_role_probability(self) -> float:
        """返回 explore 概率（高能量+低进展→多探索，反之→多利用）"""
        energy_ratio = self.energy / self.total_energy if self.total_energy > 0 else 0.0
        p_done = self.p_done()
        fail_penalty = min(self._no_progress_count, 5) * 0.05
        explore_prob = energy_ratio * 0.5
        if p_done < 0.3:
            explore_prob += 0.2
        elif p_done > 0.7:
            explore_prob -= 0.2
        explore_prob -= fail_penalty
        return max(0.1, min(0.7, explore_prob))

    def should_stop(self) -> tuple:
        # 硬约束：总预算耗尽
        if self.total_spent >= self.total_energy:
            return True, f"Total budget exhausted ({self.total_spent:.1f}/{self.total_energy:.1f})"
        # 软约束：流动资金耗尽
        if self.energy <= 0:
            return True, "Working capital depleted"
        # 任务完成概率高
        if self.p_done() > 0.9:
            return True, f"Task likely done (P={self.p_done():.2f})"
        # 连续无进展
        if self._no_progress_count >= 5:
            return True, f"No progress for {self._no_progress_count} rounds"
        # 失败过多
        if self.failure_count >= 5:
            return True, f"Too many failures ({self.failure_count})"
        return False, ""

    def should_stop_with_estimator(self, estimator: StepEstimator) -> tuple:
        basic, reason = self.should_stop()
        if basic:
            return basic, reason
        # 能量前瞻：用期望值比较，给 80% 缓冲
        # 仅在已有执行数据时才前瞻（至少跑过一轮）
        if estimator and estimator.completed > 0:
            rem_mean, _, _ = estimator.predict_remaining()
            required = rem_mean * self.step_overhead * 0.8
            if self.energy < required:
                return True, f"Insufficient energy for remaining steps ({self.energy:.0f} < {required:.0f})"
        return False, ""

    # ---- 命令超时预估 ----
    def _extract_pattern(self, command: str) -> str:
        """提取命令模式：程序名 + 子命令（不含 URL/路径/文件名）"""
        parts = command.strip().split()
        if not parts:
            return "empty"
        prog = os.path.basename(parts[0]).lower()
        for ext in ('.exe', '.cmd', '.bat', '.ps1'):
            if prog.endswith(ext):
                prog = prog[:-len(ext)]
        if len(parts) < 2:
            return prog
        second = parts[1]
        # URL / 路径 / 文件 / flag → 只保留程序名
        if (second.startswith(('http://', 'https://', '-', '/', '\\', '.'))
                or '\\' in second or '/' in second
                or second.endswith(('.py', '.sh', '.js', '.json', '.txt'))):
            return prog
        # 子命令（如 npx playwright, pip install）
        return f"{prog} {second}"

    def estimate_cmd_time(self, command: str) -> tuple:
        """返回 (期望秒, 上限秒95%)，Gamma 后验 + Wilson-Hilferty 近似"""
        pattern = self._extract_pattern(command)
        alpha_sum, beta_n = self.cmd_time_prior.get(pattern, (5.0, 1.0))
        mean = alpha_sum / beta_n
        shape = alpha_sum
        if shape > 1:
            # Wilson-Hilferty 近似 Gamma 95% 分位数
            z = 1.645
            factor = 1 - 1/(9*shape) + z * math.sqrt(1/(9*shape))
            upper = (shape / beta_n) * max(0.5, factor) ** 3
        else:
            upper = mean * 3  # 数据不足，宽裕估计
        return mean, min(upper, 60.0)

    def pre_consume_for_cmd(self, command: str):
        """预扣能量（基于预估上限 + 连续失败惩罚），返回预扣量或 False"""
        _, upper = self.estimate_cmd_time(command)
        pre_cost = upper * self.energy_per_second
        # 连续失败惩罚：每多一次失败加 500 token 基础惩罚
        pattern = self._extract_pattern(command)
        streak = self._cmd_fail_streak.get(pattern, 0)
        penalty = min(streak, 5) * 500
        total_pre = pre_cost + penalty
        if not self.consume(total_pre):
            return False
        return total_pre

    def refund_for_cmd(self, command: str, pre_cost: float,
                       actual_seconds: float, success: bool) -> float:
        """根据实际耗时和成败返还能量。失败只返还 50% 差额。净消耗记入 total_spent。"""
        actual_cost = actual_seconds * self.energy_per_second
        refund_ratio = 1.0 if success else 0.5
        refund = max(0, pre_cost - actual_cost) * refund_ratio
        self.energy += refund
        # 净消耗 = 预扣 - 返还（不可逆）
        net = pre_cost - refund
        self.total_spent += net
        # 更新后验
        pattern = self._extract_pattern(command)
        alpha_sum, beta_n = self.cmd_time_prior.get(pattern, (5.0, 1.0))
        self.cmd_time_prior[pattern] = (alpha_sum + actual_seconds, beta_n + 1)
        # 更新连续失败计数
        if success:
            self._cmd_fail_streak[pattern] = 0
        else:
            self._cmd_fail_streak[pattern] = self._cmd_fail_streak.get(pattern, 0) + 1
        return refund

    # ---- 投资-收益模型（spawn_agent 专用） ----
    def invest(self, amount: float) -> bool:
        """预扣投资额（仅扣流动资金），返回是否资金充足"""
        if self.energy < amount:
            return False
        self.energy -= amount
        return True

    def settle_investment(self, amount: float, success: bool, mode: str = "exploit"):
        """结算投资：根据成败和模式决定返还 + 记录净消耗

        mode="explore": 成功→返还本金+ROI奖金; 失败→血本无归
        mode="exploit": 成功→返还本金;   失败→返还50%
        """
        if mode == "explore":
            if success:
                bonus = amount * self.explore_roi
                self.energy += amount + bonus   # 返还本金 + 奖金
                self.commit_spent(0)            # 净消耗为 0
            else:
                # 血本无归，不返还
                self.commit_spent(amount)       # 全额计入不可逆消耗
        else:  # exploit
            if success:
                self.energy += amount           # 返还全部本金
                self.commit_spent(0)
            else:
                refund = amount * 0.5
                self.energy += refund
                self.commit_spent(amount - refund)  # 净消耗 50%

    def grant_terminal_reward(self, task_success: bool,
                              plan_complexity: int = 1,
                              actual_steps: int = 1,
                              expected_steps: float = 1.0):
        """动态终端奖励：base × 难度 - 消耗惩罚。越省能量奖励越高"""
        if not task_success:
            return 0.0
        base = Config.REWARD_BASE
        difficulty = min(2.0, 1.0 + 0.3 * (plan_complexity - 1))
        # 用实际能量消耗作为效率指标：消耗越少，奖励越高
        spent_ratio = self.total_spent / self.total_energy if self.total_energy > 0 else 0
        efficiency = max(0.3, 1.0 - spent_ratio)
        reward = base * difficulty * efficiency
        self.energy += reward
        self.total_energy += reward
        max_total = Config.CONTEXT_BUDGET - Config.CONTEXT_RESERVE // 2
        if self.total_energy > max_total:
            self.total_energy = max_total
        print(f"  [Reward] +{reward:.0f}E (base={base:.0f} × diff={difficulty:.2f} × eff={efficiency:.2f})")
        return reward

    def process_event(self, event: 'SubAgentEvent'):
        """处理子 Agent 事件，更新贝叶斯后验"""
        if event.type == SubAgentEventType.STEP_COMPLETED:
            pass  # 开销已在 execute_next_step 中通过 consume_step_overhead 扣除
        elif event.type == SubAgentEventType.TOOL_FAILED:
            self.failure_count += 1
            self._no_progress_count += 1
        elif event.type == SubAgentEventType.TASK_COMPLETED:
            self.update_done(True)
        elif event.type == SubAgentEventType.FATAL_ERROR:
            self.failure_count += 1
            self._no_progress_count += 1
            self.update_done(False)
        elif event.type == SubAgentEventType.MAX_STEPS_REACHED:
            self.update_done(False)
        elif event.type == SubAgentEventType.TIMEOUT:
            self.failure_count += 1
            self.update_done(False)

# ============ Agent ============
class Agent:
    def __init__(self, system_prompt: str = "你是一个AI编程助手。",
                 work_dir: str = None,
                 skills: Optional[List[str]] = None,
                 auto_skill: bool = False,
                 depth: int = 0,
                 parent: 'Agent' = None):
        self.work_dir = work_dir or Config.WORK_DIR
        os.makedirs(self.work_dir, exist_ok=True)
        self.depth = depth
        self.agent_id = f"agent_{random.randint(1000,9999)}"
        self.parent = parent

        # Anthropic SDK 客户端
        self.client = anthropic.Anthropic(
            api_key=Config.ANTHROPIC_API_KEY,
            base_url=Config.ANTHROPIC_BASE_URL
        )

        # Skill 管理（先于工具加载，放在稳定前缀中）
        self.skill_manager = SkillManager()
        self.active_skills: list[Skill] = []
        if skills:
            for name in skills:
                skill = self.skill_manager.get(name)
                if skill:
                    self.active_skills.append(skill)
                else:
                    print(f"  [Warning] Skill not found: {name}")
        self.auto_skill = auto_skill

        # 构建系统约束：Skill 在前（稳定），工具说明在后
        skill_section = ""
        if self.active_skills:
            skill_section = "\n\n" + "\n\n".join(f"## Skill: {s.name}\n{s.content}" for s in self.active_skills)

        self.stack: deque[StackFrame] = deque()
        self.stack.append(StackFrame("constraint", system_prompt + skill_section, level=0))
        self.stack.append(StackFrame("plan", "Pending plan...", level=1))

        self.step_counter = 0
        self.max_tool_rounds = Config.MAX_TOOL_ROUNDS  # 默认安全网，下面能量初始化后重算
        self.subtask_queue: list[dict] = []
        self.conversation_history: list[dict] = []
        self.step_estimator: Optional[StepEstimator] = None
        # 缓存 provider（子 agent 继承父级）
        self._cache_provider = parent._cache_provider if parent else AnthropicCacheProvider()
        # 启动时间（固定写入 system prompt，避免每轮变化破坏缓存）
        self.start_time = parent.start_time if parent else None
        if self.start_time is None:
            from datetime import datetime
            self.start_time = datetime.now().strftime("%Y-%m-%d %p").replace("AM", "上午").replace("PM", "下午")
        # 能量管理：共享父级（子孙同池），顶层新建
        self.energy_manager = parent.energy_manager if parent else BayesianEnergyManager()
        # per-agent 工具轮数上限，跟能量预算挂钩
        self.max_tool_rounds = max(5, int(self.energy_manager.energy / 3000))
        # API 客户端
        self._api = APIClient(self.client, Config, self._cache_provider, self.energy_manager)
        # 事件系统
        self.event_log: list[SubAgentEvent] = []
        # 流程图记录器：子 Agent 复用父级实例，避免覆盖
        if parent and hasattr(parent, 'flowchart') and parent.flowchart:
            self.flowchart = parent.flowchart
        else:
            self.flowchart = FlowchartRecorder(self.work_dir)
        # 共享文件缓存：父 agent 传递已完成的文件路径，子 agent 可直接读
        self.shared_files: dict[str, str] = {}  # {relative_path: absolute_path}
        if parent and hasattr(parent, 'shared_files'):
            self.shared_files = dict(parent.shared_files)
        # 共享缓存目录
        self._cache_dir = os.path.join(self.work_dir, "cache")
        os.makedirs(self._cache_dir, exist_ok=True)
        # 记录启动节点（仅顶层 Agent）
        if not parent:
            try:
                self.flowchart.add_node("start", "Agent 启动", shape="start")
            except Exception:
                pass

        # 经验元数据：仅根 Agent 从文件加载，子 Agent 从空开始（避免继承兄弟姐妹的经验导致指数爆炸）
        self._pending_experiences: list[dict] = []
        if not parent:
            self._load_pending_experiences()

    # ---- 状态持久化 ----
    @classmethod
    def load_state(cls, work_dir: str = None):
        work_dir = work_dir or Config.WORK_DIR
        state_file = os.path.join(work_dir, "state.json")
        if not os.path.exists(state_file):
            raise FileNotFoundError(f"State file not found: {state_file}")
        with open(state_file, "r", encoding='utf-8') as f:
            state = json.load(f)
        agent = cls.__new__(cls)
        agent.work_dir = work_dir
        agent.stack = deque([StackFrame(**frame) for frame in state["stack"]])
        agent.step_counter = state["step_counter"]
        agent.subtask_queue = state.get("subtask_queue", [])
        agent.conversation_history = state.get("conversation_history", [])
        agent._pending_experiences = state.get("_pending_experiences", [])
        agent.depth = 0
        agent.agent_id = f"agent_{random.randint(1000,9999)}"
        agent.parent = None
        agent.event_log = []
        agent.energy_manager = BayesianEnergyManager()
        agent.max_tool_rounds = max(5, int(agent.energy_manager.energy / 3000))
        agent.step_estimator = None
        agent._cache_provider = AnthropicCacheProvider()
        agent.client = anthropic.Anthropic(
            api_key=Config.ANTHROPIC_API_KEY,
            base_url=Config.ANTHROPIC_BASE_URL
        )
        agent._api = APIClient(agent.client, Config, agent._cache_provider, agent.energy_manager)
        agent.skill_manager = SkillManager()
        agent.active_skills = []
        agent.auto_skill = False
        agent._load_pending_experiences()
        return agent

    def _save_state(self):
        state = {
            "stack": [asdict(f) for f in self.stack],
            "step_counter": self.step_counter,
            "subtask_queue": self.subtask_queue,
            "conversation_history": self.conversation_history,
            "_pending_experiences": self._pending_experiences,
        }
        path = os.path.join(self.work_dir, "state.json")
        with open(path, "w", encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def _load_pending_experiences(self):
        path = os.path.join(self.work_dir, "pending_exp.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self._pending_experiences = json.load(f)
                # 安全上限：超过200条截断到最近200条
                if len(self._pending_experiences) > 200:
                    print(f"  [Experience] truncating loaded {len(self._pending_experiences)} -> 200 records")
                    self._pending_experiences = self._pending_experiences[-200:]
            except (json.JSONDecodeError, Exception):
                self._pending_experiences = []
        else:
            self._pending_experiences = []

    def _save_pending_experiences(self):
        path = os.path.join(self.work_dir, "pending_exp.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._pending_experiences, f, ensure_ascii=False, indent=2)

    def _inject_experience(self, task: str):
        """从经验库搜索相关经验，注入到历史帧"""
        try:
            from experience_store import ExperienceStore
            store = ExperienceStore()
            results = store.search(task, limit=3)
            if not results:
                return
            parts = []
            for r in results:
                status = "OK" if r["success"] else "FAIL"
                line = f"- [{status}] {r['task'][:60]} -> {r['summary'][:100] if r['summary'] else '(no summary)'}"
                if r.get("lessons"):
                    line += f"\n  Lesson: {r['lessons'][:120]}"
                parts.append(line)
            exp_text = "## Relevant Past Experience\n" + "\n".join(parts)
            # 替换或追加到 history 帧
            self.stack = deque(f for f in self.stack if f.type != "experience")
            self.stack.append(StackFrame("experience", exp_text, level=0))
            print(f"  [Experience] injected {len(results)} relevant records")
            # 流程图：经验注入
            if self.flowchart and not self.parent:
                try:
                    self.flowchart.add_node("exp_inject", f"注入 {len(results)} 条经验", shape="energy")
                    self.flowchart.add_edge("start", "exp_inject", label="经验")
                except Exception:
                    pass
        except Exception as e:
            pass  # 经验库不可用不应阻塞执行

    def _save_history(self, task: str, result: str, steps: int, success: bool):
        """保存运行结果到 history/ 目录"""
        try:
            hist_dir = Path(Config.HISTORY_DIR)
            hist_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = hist_dir / f"{ts}.md"
            status = "SUCCESS" if success else "PARTIAL"
            content = f"# {task[:80]}\n\n"
            content += f"- **Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            content += f"- **Steps**: {steps}\n"
            content += f"- **Status**: {status}\n"
            content += f"- **Depth**: {self.depth}\n\n"
            content += f"## Result\n\n{result or '(no result)'}\n"
            fname.write_text(content, encoding='utf-8')
        except Exception:
            pass

    def emit_event(self, event_type: SubAgentEventType, message: str = "", data: dict = None):
        event = SubAgentEvent(self.agent_id, event_type, message, data)
        self.event_log.append(event)
        # 能量管理器处理事件
        if self.energy_manager:
            self.energy_manager.process_event(event)

    def _collect_experience_metadata(self, task: str, plan: str, step_count: int, success: bool, final_result: str):
        """轻量收集经验元数据，不调用 LLM"""
        last_summary = ""
        for f in reversed(self.stack):
            if f.type == "summary" and f.level == 2:
                last_summary = f.content
                break
        self._pending_experiences.append({
            "task": task,
            "plan": plan,
            "step_count": step_count,
            "success": success,
            "last_summary": last_summary[:500],
            "final_result_preview": (final_result or "")[:300],
            "timestamp": time.time()
        })
        self._save_pending_experiences()

    def flush_experiences(self):
        """批量处理待处理经验：仅对步数>=3或失败的调用 LLM 提取教训"""
        if not self._pending_experiences:
            return
        # 安全上限：超过100条仅保留最近100条
        if len(self._pending_experiences) > 100:
            print(f"  [Experience] truncating {len(self._pending_experiences)} -> 100 records")
            self._pending_experiences = self._pending_experiences[-100:]
        try:
            from experience_store import ExperienceStore
            store = ExperienceStore()
            count = 0
            for exp in self._pending_experiences:
                summary = exp["last_summary"] or exp["final_result_preview"]
                lesson = ""
                # 仅失败或复杂任务才提取教训
                if not exp["success"] or exp["step_count"] >= 3:
                    lesson = self.extract_lesson_from_text(exp["final_result_preview"])
                store.record(
                    task=exp["task"], plan=exp["plan"], summary=summary,
                    lessons=lesson, step_count=exp["step_count"], success=exp["success"]
                )
                # 经验权重更新
                store.update_weights(exp["task"], exp["plan"], exp["success"])
                count += 1
            self._pending_experiences = []
            self._save_pending_experiences()
            print(f"  [Experience] flushed {count} records")
        except Exception as e:
            print(f"  [Experience] flush failed: {e}")

    def extract_lesson_from_text(self, text: str) -> str:
        """从文本中提取教训（调用 LLM，仅复杂任务触发）"""
        if not text or len(text) < 20:
            return ""
        response = self._api.call(
            [{"role": "user", "content":
                f"从以下任务结果中提取一个可复用的经验教训（100字以内）：\n\n{text}"}],
            system="你是一个经验总结助手。只输出经验教训。",
            max_tokens=200
        )
        return self._extract_text(response)

    def _extract_text(self, response) -> str:
        """从 Message 对象提取文本"""
        return "\n".join(b.text for b in response.content if hasattr(b, 'text'))

    # ---- 自适应上下文管理 ----
    def _estimate_context_length(self) -> int:
        frames = list(self.stack)
        budget = Config.CONTEXT_BUDGET
        length = 0

        for f in frames:
            if f.level == 1:
                length += len(f"## Plan\n{f.content}\n\n")
        for f in frames:
            if f.level == 3:
                length += len(f"## Summary\n{f.content}\n\n")

        summary_frames = [f for f in frames if f.level == 2]
        remaining = budget - length
        for f in reversed(summary_frames):
            entry = f"Step {f.step_id}: {f.content}\n"
            if remaining - len(entry) < 0:
                break
            length += len(entry)
            remaining -= len(entry)
        return length

    def _compress(self, text: str, step_id: int) -> str:
        prompt = f"将以下内容压缩为一句话摘要（不超过100字），保留关键动作和结果：\n\n{text}"
        response = self._api.call(
            [{"role": "user", "content": prompt}],
            system="你是一个摘要助手。只输出摘要。",
            max_tokens=300,
            bypass_energy=True
        )
        summary = self._extract_text(response)
        return f"[Step {step_id}]: {summary}"

    # ---- 能量回收：弹出栈顶帧 ----
    def _reclaim_energy(self, target_amount: float) -> float:
        """从栈中弹出帧回收能量。策略：
        1. 优先弹空壳子 agent 结果（内容 ≤ 50 chars，无实质内容）
        2. 再弹 summary/merge 帧（压缩过的，信息密度低）
        3. 最后弹 step_detail（损失最大）
        不弹 constraint/plan/history/experience。"""
        reclaimed = 0.0
        MIN_STACK = 3  # 至少保留 3 帧
        _reclaim_seq = 0  # 用于生成唯一节点 ID

        def _pop_frame(frame, idx):
            nonlocal reclaimed, _reclaim_seq
            estimated_tokens = len(frame.content) / 4
            del self.stack[idx]
            reclaimed += estimated_tokens
            print(f"  [Reclaim] Popped {frame.type} Step {frame.step_id} "
                  f"({len(frame.content)} chars ≈ {estimated_tokens:.0f}E)")
            # 记录到流程图
            if self.flowchart:
                try:
                    _reclaim_seq += 1
                    node_id = f"reclaim_d{self.depth}_{_reclaim_seq}"
                    label = f"弹出 {frame.type} (≈{estimated_tokens:.0f}E)"
                    self.flowchart.add_node(node_id, label, shape="reclaim")
                    aid = frame.agent_id or self.agent_id
                    step_ref = f"s{self.depth}_{aid}_{frame.step_id}" if frame.step_id else None
                    if step_ref:
                        self.flowchart.add_edge(step_ref, node_id, label=f"-{estimated_tokens:.0f}E")
                except Exception:
                    pass

        # Phase 1: 弹空壳（≤50 chars 的 step_detail/summary）
        if reclaimed < target_amount:
            for i in range(len(self.stack) - 1, -1, -1):
                if len(self.stack) <= MIN_STACK:
                    break
                f = self.stack[i]
                if f.type in ("step_detail", "summary") and len(f.content) <= 50:
                    _pop_frame(f, i)
                    if reclaimed >= target_amount:
                        break

        # Phase 2: 弹 summary/merge（压缩过的低密度帧）
        if reclaimed < target_amount:
            for i in range(len(self.stack) - 1, -1, -1):
                if len(self.stack) <= MIN_STACK:
                    break
                f = self.stack[i]
                if f.type in ("summary", "merge") and f.level >= 2:
                    _pop_frame(f, i)
                    if reclaimed >= target_amount:
                        break

        # Phase 3: 弹 step_detail（信息损失最大，最后手段）
        if reclaimed < target_amount:
            for i in range(len(self.stack) - 1, -1, -1):
                if len(self.stack) <= MIN_STACK:
                    break
                f = self.stack[i]
                if f.type == "step_detail" and f.level >= 2:
                    _pop_frame(f, i)
                    if reclaimed >= target_amount:
                        break

        if reclaimed > 0:
            self.energy_manager.energy += reclaimed
            print(f"  [Reclaim] Total recovered: {reclaimed:.0f}E")
        return reclaimed

    # ---- 消息总量截断 ----
    def _trim_messages(self, messages: list, max_chars: int) -> list:
        """截断 messages 列表使其总字符数低于 max_chars。
        策略：从最旧的 user 消息（通常是工具结果）开始截断内容。"""
        total = sum(len(str(m)) for m in messages)
        if total <= max_chars:
            return messages
        for i, msg in enumerate(messages):
            if total <= max_chars:
                break
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 500:
                old_len = len(content)
                content = content[:500] + f"\n... [truncated from {old_len} chars]"
                messages[i] = {**msg, "content": content}
                total -= (old_len - len(content))
            elif isinstance(content, list):
                # tool_result 格式: [{"type": "tool_result", "content": "..."}]
                for j, item in enumerate(content):
                    if isinstance(item, dict) and len(str(item.get("content", ""))) > 500:
                        old_len = len(str(item["content"]))
                        item["content"] = str(item["content"])[:500] + f"\n... [truncated from {old_len}]"
                        total -= (old_len - 500)
            if total <= max_chars:
                break
        print(f"  [Trim] Messages trimmed to ~{total} chars (limit {max_chars})")
        return messages

    # ---- 子 Agent 结果吸收 ----
    def _make_summary(self, text: str, max_len: int = 300) -> str:
        """不调 LLM 的快速摘要：取首尾 + 元数据"""
        if len(text) <= max_len:
            return text
        head = text[:max_len // 2]
        tail = text[-(max_len // 3):]
        return f"{head}\n... [省略 {len(text) - len(head) - len(tail)} 字] ...\n{tail}"

    def _build_failure_experience(self, max_items: int = 5) -> str:
        """从栈帧中提取失败经验，传递给子 Agent。"""
        failures = []
        # 从最近栈帧中找包含 error/失败 的内容
        for frame in list(self.stack)[-10:]:
            content = frame.content[:300]
            if any(kw in content.lower() for kw in ["error", "失败", "fail", "not found", "不存在"]):
                failures.append(content)
        if not failures:
            return ""
        lines = failures[-max_items:]
        exp = "## 父级执行经验（请避免重复以下失败操作）\n"
        for i, f in enumerate(lines):
            exp += f"{i+1}. {f}\n"
        exp += "\n建议：使用不同的方法或命令重试。\n"
        return exp

    def _absorb_child_result(self, child_result: str, subtask_desc: str):
        """根据父 Agent 能量决定保留完整内容还是仅摘要，将结果入栈。"""
        energy_ratio = self.energy_manager.energy / self.energy_manager.total_energy if self.energy_manager.total_energy > 0 else 0
        is_full = energy_ratio > 0.3 and len(child_result) < 2000
        if is_full:
            # 能量充足且内容不大，保留完整内容
            self.stack.append(StackFrame("step_detail", child_result, self.step_counter, level=2, agent_id=self.agent_id))
            print(f"  [Absorb] Full result ({len(child_result)} chars, energy {energy_ratio:.0%})")
        else:
            # 能量紧张或内容过大，仅保留摘要
            summary = self._make_summary(child_result)
            self.stack.append(StackFrame("summary", summary, self.step_counter, level=2, agent_id=self.agent_id))
            print(f"  [Absorb] Summary only ({len(summary)}/{len(child_result)} chars, energy {energy_ratio:.0%})")
        # 生命周期计数器：记录吸收类型（不再创建单独节点）
        if self.flowchart:
            try:
                child_depth = self.depth + 1
                self.flowchart.record_lifecycle(child_depth,
                    absorb_full=1 if is_full else 0,
                    absorb_summary=0 if is_full else 1)
            except Exception:
                pass

    # ---- 子 Agent 上下文摘要 ----
    def _summarize_via_child(self, messages: list, task_hint: str = "") -> str:
        """将完整上下文注入子 Agent 生成结构化摘要。
        子 Agent 拥有独立上下文，摘要后 parent 只保留摘要文本。
        成本：子 Agent 一次 API 调用（含全量上下文），parent 后续每轮省 ~full_context tokens。"""
        # 提取 messages 中的关键信息构建摘要输入
        context_parts = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            if isinstance(content, str) and content:
                context_parts.append(f"[{role}]\n{content}")
            elif isinstance(content, list):
                # tool_result 格式
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("content", "")
                        if text:
                            context_parts.append(f"[tool_result]\n{text}")
        full_context = "\n\n---\n\n".join(context_parts)

        if not full_context.strip():
            return ""

        # 构建摘要 prompt
        prompt = (
            f"以下是某个任务执行过程中积累的全部上下文。请生成结构化摘要（不超过800字），包含：\n"
            f"1. 已完成的关键操作和结果\n"
            f"2. 已获取的重要信息\n"
            f"3. 当前进展状态\n"
            f"4. 待解决的剩余问题\n"
        )
        if task_hint:
            prompt += f"\n重点关注与「{task_hint}」相关的信息。\n"
        prompt += f"\n--- 原始上下文 ---\n{full_context}"

        # 创建极简子 Agent，不做工具调用，纯摘要
        child = Agent(
            system_prompt="你是摘要助手。只输出结构化摘要，不超过800字。不要输出其他内容。",
            depth=self.depth + 1,
            parent=self
        )
        child.max_tool_rounds = 0  # 不允许工具调用
        try:
            summary = child.run(prompt, max_steps=1)
            # 子 Agent 的栈已销毁，只有 summary 文本返回
            summary_len = len(summary) if summary else 0
            print(f"  [ChildSummary] {len(full_context)} chars -> {summary_len} chars "
                  f"(saved ~{len(full_context) // 4} tokens/round)")
            return summary or ""
        except Exception as e:
            print(f"  [ChildSummary] Failed: {e}")
            # 降级为简单摘要
            return self._make_summary(full_context, max_len=1000)

    def _adapt_context(self):
        threshold = Config.CONTEXT_BUDGET * Config.COMPRESSION_THRESHOLD
        _compress_seq = 0
        # 循环压缩，直到低于阈值或没有可压缩的帧
        while True:
            est = self._estimate_context_length()
            if est < threshold:
                return
            compressed_any = False
            for i, frame in enumerate(self.stack):
                if frame.type == "step_detail" and frame.level >= 2:
                    compressed = self._compress(frame.content, frame.step_id)
                    print(f"  [Compress] Step {frame.step_id}: {len(frame.content)} -> {len(compressed)} chars ({est}/{Config.CONTEXT_BUDGET})")
                    self.stack[i] = StackFrame("summary", compressed, frame.step_id, level=2, agent_id=frame.agent_id)
                    # 记录到流程图
                    if self.flowchart:
                        try:
                            _compress_seq += 1
                            node_id = f"compress_d{self.depth}_{_compress_seq}"
                            label = f"压缩 Step {frame.step_id} ({len(frame.content)}→{len(compressed)})"
                            self.flowchart.add_node(node_id, label, shape="reclaim")
                            aid = frame.agent_id or self.agent_id
                            step_ref = f"s{self.depth}_{aid}_{frame.step_id}" if frame.step_id else None
                            if step_ref:
                                self.flowchart.add_edge(step_ref, node_id, label="压缩")
                        except Exception:
                            pass
                    compressed_any = True
                    break  # 每次只压缩一帧，重新评估
            if not compressed_any:
                break
        # 最终兜底：如果还超，删除最旧的 summary 帧
        est = self._estimate_context_length()
        if est > Config.CONTEXT_BUDGET:
            for i, frame in enumerate(self.stack):
                if frame.level == 2:
                    print(f"  [Truncate] remove Step {frame.step_id} ({est}/{Config.CONTEXT_BUDGET})")
                    del self.stack[i]
                    # 记录到流程图
                    if self.flowchart:
                        try:
                            _compress_seq += 1
                            node_id = f"truncate_d{self.depth}_{_compress_seq}"
                            label = f"截断 Step {frame.step_id}"
                            self.flowchart.add_node(node_id, label, shape="truncate")
                            aid = frame.agent_id or self.agent_id
                            step_ref = f"s{self.depth}_{aid}_{frame.step_id}" if frame.step_id else None
                            if step_ref:
                                self.flowchart.add_edge(step_ref, node_id, label="截断")
                        except Exception:
                            pass
                    return

    # ---- 上下文构建（单字符串，用于估算长度等） ----
    def _build_context(self) -> str:
        frames = list(self.stack)
        budget = Config.CONTEXT_BUDGET
        parts = []

        history_frames = [f for f in frames if f.type == "history"]
        if history_frames:
            parts.append(history_frames[-1].content)

        exp_frames = [f for f in frames if f.type == "experience"]
        if exp_frames:
            parts.append(exp_frames[-1].content)

        plan_frames = [f for f in frames if f.level == 1]
        for f in plan_frames:
            parts.append(f"## Plan\n{f.content}")

        merge_frames = [f for f in frames if f.level == 3]
        if merge_frames:
            for f in merge_frames:
                parts.append(f"## Phase Summary\n{f.content}")

        summary_frames = [f for f in frames if f.level == 2]
        used = sum(len(p) for p in parts)
        remaining = budget - used
        if remaining > 0 and summary_frames:
            history_parts = []
            for f in reversed(summary_frames):
                entry = f"Step {f.step_id}: {f.content}"
                if remaining - len(entry) < 0:
                    break
                history_parts.append(entry)
                remaining -= len(entry)
            if history_parts:
                parts.append("## Completed Steps\n" + "\n".join(reversed(history_parts)))

        return "\n\n".join(parts)

    # ---- 多轮消息构建（缓存友好） ----
    def _build_messages(self) -> tuple[list, list]:
        """从栈帧生成多轮消息列表 + system blocks，供缓存友好的 API 调用。
        返回 (messages, system_blocks)，system_blocks 含 cache_control 断点。"""
        cc = self._cache_provider.cache_control()
        PLATFORM_TEXT = (
            f"Program start time: {self.start_time}\n"
            "Platform: Windows. Use `python` not `python3`. "
            "Prefer `start` or PowerShell for browser/file operations. "
            "When running scripts, use `python scripts/NAME.py` (files auto-routed to scripts/). "
            "CRITICAL: Every `python -c` command MUST include print() to show output. "
            "Think first, then write ONE correct command instead of trial-and-error. "
            "完成任务后在末尾单独一行写 DONE，否则写 CONTINUE。"
        )
        # System blocks（缓存断点 #1: tools 已通过 _build_cached_tools 标记）
        system_blocks = [
            {"type": "text", "text": self.stack[0].content},
        ]
        platform_block = {"type": "text", "text": PLATFORM_TEXT}
        if cc:
            platform_block["cache_control"] = cc  # 缓存断点 #2: system
        system_blocks.append(platform_block)

        messages = []
        frames = list(self.stack)

        # 经验注入
        exp = [f for f in frames if f.type == "experience"]
        if exp:
            messages.append({"role": "user", "content": exp[-1].content})
            messages.append({"role": "assistant", "content": "经验已纳入参考。"})

        # 计划
        plans = [f for f in frames if f.level == 1]
        if plans:
            messages.append({"role": "user", "content": f"## Plan\n{plans[0].content}"})
            messages.append({"role": "assistant", "content": "计划已确认，开始执行。"})

        # 父级上下文
        hist = [f for f in frames if f.type == "history"]
        if hist:
            messages.append({"role": "user", "content": hist[-1].content})

        # 阶段合并 (level 3)
        merges = [f for f in frames if f.level == 3]
        for f in merges:
            messages.append({"role": "user", "content": "汇总阶段成果"})
            messages.append({"role": "assistant", "content": f.content})

        # 已完成步骤 (level 2) — 配对 user(subtask desc)/assistant(result)
        summaries = [f for f in frames if f.level == 2]
        done_subs = [s for s in self.subtask_queue if s.get('done')]
        for i, f in enumerate(summaries):
            desc = done_subs[i]['desc'] if i < len(done_subs) else f"Step {f.step_id}"
            messages.append({"role": "user", "content": f"执行: {desc}"})
            label = f"[Step {f.step_id}] " if f.type == "summary" else ""
            messages.append({"role": "assistant", "content": label + f.content})

        return messages, system_blocks

    # ---- 单步执行（原生工具循环） ----
    def execute_next_step(self, task: str) -> tuple[str, bool]:
        self.step_counter += 1
        if not self.energy_manager.consume_step_overhead():
            print(f"  [Step {self.step_counter}] Energy depleted (overhead), skipping step")
            return "(energy depleted)", False
        # 记录步骤节点
        step_node = f"s{self.depth}_{self.agent_id}_{self.step_counter}"
        if self.flowchart:
            try:
                self.flowchart.add_node(step_node, f"步骤 {self.step_counter}: {task[:30]}...", shape="step")
                if hasattr(self, '_fc_task_node') and self._fc_task_node:
                    self.flowchart.add_edge(getattr(self, "_fc_current_sub_node", self._fc_task_node), step_node)
            except Exception:
                pass
        self._current_step_node = step_node
        # 多轮消息格式（缓存友好）
        base_messages, system_blocks = self._build_messages()
        cached_tools = _build_cached_tools(self._cache_provider)
        cc = self._cache_provider.cache_control()

        # 当前步骤指令（缓存断点 #3: 最新 user 消息）
        step_text = f"[Task]\n{task}\n\n[Progress] Step {self.step_counter}"
        if cc:
            step_msg = {"role": "user", "content": [
                {"type": "text", "text": step_text, "cache_control": cc}
            ]}
        else:
            step_msg = {"role": "user", "content": step_text}

        messages = base_messages + [step_msg]
        final_text = ""

        for round_num in range(self.max_tool_rounds):
            # 能量不足则停止工具循环
            if self.energy_manager.energy <= 0:
                print(f"  [Tool loop] Energy depleted after {round_num} rounds")
                # 流程图：工具循环耗尽
                if self.flowchart:
                    try:
                        depl_id = f"depl_d{self.depth}_{self.step_counter}_{round_num}"
                        label = f"能量耗尽 (round {round_num})"
                        self.flowchart.add_node(depl_id, label, shape="decision")
                        self.flowchart.add_edge(self._current_step_node, depl_id, label="耗尽")
                    except Exception:
                        pass
                break

            # Pre-call 能量估算：chars / 4 ≈ tokens，成本 = tokens * 1.0
            msg_chars = sum(len(str(m)) for m in messages)
            est_input_tokens = msg_chars // 4
            est_cost = est_input_tokens * Config.TOKEN_COST_INPUT + 500  # +500 估算 output
            if est_cost > self.energy_manager.energy:
                # 优先尝试弹栈回收
                reclaimed = self._reclaim_energy(est_cost - self.energy_manager.energy)
                if est_cost > self.energy_manager.energy and round_num > 0 and msg_chars > Config.MAX_MESSAGE_CHARS // 2:
                    # 上下文过大且能量不足：用子 Agent 摘要替代全量消息
                    print(f"  [Context] Summarizing {msg_chars} chars via child agent...")
                    summary = self._summarize_via_child(messages, task_hint=task)
                    if summary:
                        # 用摘要替换 base_messages 中的历史
                        messages = [
                            {"role": "user", "content": f"[执行摘要]\n{summary}"},
                            {"role": "assistant", "content": "已了解，继续执行。"},
                            step_msg
                        ]
                        # 重算估算成本
                        msg_chars = sum(len(str(m)) for m in messages)
                        est_input_tokens = msg_chars // 4
                        est_cost = est_input_tokens * Config.TOKEN_COST_INPUT + 500
                        print(f"  [Context] After summary: {msg_chars} chars, est_cost={est_cost:.0f}")
                if est_cost > self.energy_manager.energy:
                    print(f"  [Tool loop] Energy low ({self.energy_manager.energy:.0f} < est {est_cost:.0f}), stopping")
                    break

            try:
                response = self._api.call(
                    messages, system=system_blocks, tools=cached_tools
                )
            except ContextTooLongError:
                self._adapt_context()
                messages.append({"role": "assistant", "content": "[Context too long, compressed. Retry.]"})
                continue

            if response.stop_reason == "tool_use":
                tool_blocks = [b for b in response.content if b.type == "tool_use"]
                # 必须把完整的 assistant content 放入消息（包含 text + tool_use blocks）
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in tool_blocks:
                    if not self.energy_manager.consume_tool():
                        print(f"  [Tool] Energy depleted, skipping remaining tools")
                        break
                    args_str = json.dumps(block.input, ensure_ascii=False)[:100]
                    print(f"  [Tool] {block.name}({args_str})")

                    msg_total = sum(len(str(m)) for m in messages)
                    tool_budget = min(Config.TOOL_RESULT_BUDGET, Config.CONTEXT_BUDGET - msg_total)
                    tool_budget = max(tool_budget, 1000)
                    # 记录工具节点（同一 step 内同名工具合并计数）
                    merged_node = self.flowchart.merged_tool(
                        step_node, block.name, self.step_counter, depth=self.depth) if self.flowchart else None


                    agent_ctx = {
                        "depth": self.depth,
                        "system_prompt": self.stack[0].content,
                        "parent_agent": self
                    }
                    result = ToolExecutor.execute(
                        block.name, dict(block.input),
                        budget=tool_budget, agent_context=agent_ctx
                    )

                    status = "ok" if result["success"] else "error"
                    trunc = f" (truncated from {result.get('original_size', '?')})" if result.get("truncated") else ""
                    print(f"  [Tool result] {status}: {result['output'][:80]}...{trunc}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result["output"],
                        "is_error": not result["success"]
                    })

                messages.append({"role": "user", "content": tool_results})

                # Post-append 消息总量守卫：超过 MAX_MESSAGE_CHARS 则截断最旧工具结果
                total_chars = sum(len(str(m)) for m in messages)
                if total_chars > Config.MAX_MESSAGE_CHARS:
                    messages = self._trim_messages(messages, Config.MAX_MESSAGE_CHARS)
                # 工具轮结束，显示内联进度
                em = self.energy_manager
                budget_pct = em.total_spent / em.total_energy * 100 if em.total_energy > 0 else 0
                tok_info = f" | Tokens {em.total_tokens}" if em.total_tokens > 0 else ""
                print(f"  [{round_num+1}/{self.max_tool_rounds} rounds | Budget {budget_pct:.0f}% | Energy {em.energy:.0f}{tok_info}]")
                continue

            # stop_reason == "end_turn"
            final_text = self._extract_text(response)
            break

        if not final_text:
            final_text = "(no response)"

        self.stack.append(StackFrame("step_detail", final_text, self.step_counter, level=2, agent_id=self.agent_id))
        self.emit_event(SubAgentEventType.STEP_COMPLETED, f"Step {self.step_counter}",
                        {"steps": 1, "tool_rounds": round_num + 1})
        self._adapt_context()
        self._save_state()

        is_done = "DONE" in final_text.rstrip().split("\n")[-1].upper()
        # 记录步骤完成节点
        if self.flowchart:
            try:
                done_node = f"s{self.depth}_{self.agent_id}_{self.step_counter}_done"
                self.flowchart.add_node(done_node, f"步骤 {self.step_counter} {'完成' if is_done else '继续'}", shape="end")
                self.flowchart.add_edge(self._current_step_node, done_node)
            except Exception:
                pass

        return final_text, is_done

    # ---- 汇合 ----
    def merge_level(self, level: int):
        frames = [f for f in self.stack if f.level == level]
        if len(frames) < 2:
            return
        merged_content = "\n".join(f"Step {f.step_id}: {f.content}" for f in frames)
        prompt = f"将以下多个步骤的执行结果整合为一段连贯的阶段总结（不超过200字）：\n\n{merged_content}"
        response = self._api.call(
            [{"role": "user", "content": prompt}],
            system="你是一个摘要助手。只输出总结。",
            max_tokens=400
        )
        summary = self._extract_text(response)
        new_level = level + 1
        self.stack.append(StackFrame("merge", summary, step_id=0, level=new_level))
        self.stack = deque(f for f in self.stack if f.level != level or f.type in ("constraint", "plan"))
        self._save_state()
        print(f"  [Merge] {len(frames)} level={level} frames -> level={new_level}")

    # ---- 经验提取 ----
    def generate_summary(self) -> str:
        """压缩整次任务执行为摘要"""
        all_steps = [f for f in self.stack if f.level >= 2]
        if not all_steps:
            return ""
        steps_text = "\n".join(f"Step {f.step_id}: {f.content[:200]}" for f in all_steps)
        response = self._api.call(
            [{"role": "user", "content":
                f"将以下任务执行过程总结为一段200字以内的摘要，包含：任务目标、关键操作、最终结果：\n\n{steps_text}"}],
            system="你是一个摘要助手。只输出摘要。",
            max_tokens=300
        )
        return self._extract_text(response)

    def extract_lesson(self) -> str:
        """提取可复用的经验教训"""
        all_steps = [f for f in self.stack if f.level >= 2]
        if not all_steps:
            return ""
        steps_text = "\n".join(f"Step {f.step_id}: {f.content[:200]}" for f in all_steps)
        response = self._api.call(
            [{"role": "user", "content":
                f"从以下任务执行过程中提取一个可复用的经验教训（100字以内）：\n\n{steps_text}"}],
            system="你是一个经验总结助手。只输出经验教训。",
            max_tokens=200
        )
        return self._extract_text(response)

    # ---- 显示 ----
    def print_stack(self):
        total = measure_stack(self.stack)
        print(f"\n[Stack] {len(self.stack)} frames, {total}/{Config.CONTEXT_BUDGET} chars, depth={self.depth}")
        for i, f in enumerate(self.stack):
            preview = f.content[:40] + "..." if len(f.content) > 40 else f.content
            print(f"  [{i}] L{f.level} {f.type} ({len(f.content)}): {preview}")

    def _progress_bar(self, current: int, total: int, width: int = 20) -> str:
        if total <= 0:
            return "[" + "-" * width + "] 0%"
        ratio = min(current / total, 1.0)
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        percent = int(ratio * 100)
        return f"[{bar}] {percent}% ({current}/{total})"

    def print_progress(self, max_steps: int = None):
        """一行式进度"""
        em = self.energy_manager
        total_sub = len(self.subtask_queue)
        done_sub = sum(1 for s in self.subtask_queue if s.get('done'))
        max_s = max_steps or Config.MAX_STEPS
        budget_pct = em.total_spent / em.total_energy * 100 if em.total_energy > 0 else 0

        parts = [f"Step {self.step_counter}"]
        if total_sub > 1:
            parts.append(f"Sub {done_sub}/{total_sub}")
        parts.append(f"Budget {budget_pct:.0f}%")
        parts.append(f"Energy {em.energy:.0f}")

        # Token 统计
        if em.total_tokens > 0:
            parts.append(f"Tokens {em.total_tokens}")

        if self.step_estimator and done_sub > 0:
            est_total, _, _ = self.step_estimator.predict_total(self.step_counter)
            parts.append(f"~{est_total:.0f}steps")

        print(f"  [{'] ['.join(parts)}]")

    # ---- 交互模式 ----
    def interact(self, initial_task: str = None, max_steps: int = None):
        max_steps = max_steps or Config.MAX_STEPS

        print("\n" + "=" * 50)
        print("  Agent Interactive Mode")
        print("  /status   — show stack")
        print("  /progress — show progress bar")
        print("  /reset    — reset conversation")
        print("  /exit     — exit")
        print("=" * 50)

        if initial_task:
            self.run(initial_task, max_steps)

        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue
            if user_input == "/exit":
                self.flush_experiences()
                print("Bye!")
                break
            elif user_input == "/status":
                self.print_stack()
                continue
            elif user_input == "/progress":
                self.print_progress()
                continue
            elif user_input == "/reset":
                self._reset_stack()
                print("Reset done.")
                continue

            self._inject_followup(user_input)
            self.run(user_input, max_steps)

    def _reset_stack(self):
        constraint = self.stack[0] if self.stack else StackFrame("constraint", "你是一个AI编程助手。", level=0)
        self.stack = deque([constraint, StackFrame("plan", "Pending plan...", level=1)])
        self.step_counter = 0
        self.subtask_queue = []
        self.conversation_history = []
        self._save_state()

    def _inject_followup(self, question: str):
        self.stack = deque(f for f in self.stack if f.level == 0)

        history_parts = []
        for i, turn in enumerate(self.conversation_history):
            q = turn["q"]
            a = turn["a"]
            a_preview = a[:300] + "..." if len(a) > 300 else a
            history_parts.append(f"[Turn {i+1}] User: {q}\nAssistant: {a_preview}")

        if history_parts:
            history_text = "## Conversation History\n" + "\n\n".join(history_parts)
            self.stack.append(StackFrame("history", history_text, level=0))

        self.stack.append(StackFrame("plan", "Pending plan...", level=1))
        self.step_counter = 0

    # ---- 子任务去重 ----
    def _merge_subtasks(self, subtasks: list[dict], threshold: float = None) -> list[dict]:
        """子任务去重：相似子任务合并，保留更详细的描述"""
        if len(subtasks) <= 1:
            return subtasks
        threshold = threshold or Config.SUBTASK_DEDUP_THRESHOLD

        def _word_set(text: str) -> set:
            """提取中文单字 + 英文单词集合"""
            return set(re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text.lower()))

        def _jaccard(a: set, b: set) -> float:
            if not a and not b:
                return 0.0
            return len(a & b) / len(a | b)

        word_sets = [_word_set(s['desc']) for s in subtasks]
        avg_len = sum(len(s['desc']) for s in subtasks) / len(subtasks)

        def _quality_score(idx: int) -> float:
            specificity = len(subtasks[idx]['desc']) / max(avg_len, 1)
            if self.energy_manager:
                success_rate = self.energy_manager.p_subtask_success(f"sub_{idx}")
            else:
                success_rate = 0.5
            return specificity * success_rate

        # 构建合并映射：低质量 -> 高质量
        n = len(subtasks)
        merge_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                sim = _jaccard(word_sets[i], word_sets[j])
                if sim > threshold:
                    bi, bj = _quality_score(i), _quality_score(j)
                    if bi >= bj:
                        merge_map[j] = i
                    else:
                        merge_map[i] = j

        if not merge_map:
            return subtasks

        # 解析传递链：A->B->C 变成 A->C
        def _resolve(idx, visited=None):
            if visited is None:
                visited = set()
            if idx not in merge_map or idx in visited:
                return idx
            visited.add(idx)
            return _resolve(merge_map[idx], visited)
        merge_map = {k: _resolve(v) for k, v in merge_map.items()}

        merged = []
        consumed = set()
        for i in range(n):
            if i in consumed:
                continue
            if i in merge_map:
                target = merge_map[i]
                if target not in consumed and target != i:
                    # 基于单词的唯一部分提取
                    unique_words = word_sets[i] - word_sets[target]
                    words_in_desc = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', subtasks[i]['desc'])
                    unique_parts = [w for w in words_in_desc if w.lower() in unique_words]
                    new_desc = subtasks[target]['desc']
                    if unique_parts:
                        new_desc += f"; {' '.join(unique_parts)}"
                    merged.append({"desc": new_desc, "done": False})
                    consumed.add(i)
                    consumed.add(target)
                    print(f"  [Dedup] Merged subtask {i+1} -> {target+1} (sim>{threshold:.1f})")
                else:
                    merged.append(subtasks[i])
            else:
                merged.append(subtasks[i])

        return merged if merged else subtasks

    # ---- 计划生成 ----
    def _generate_plan(self, task: str) -> list[dict]:
        """以终为始规划：先明确目标，反向推导前置条件，正向输出步骤。"""
        prompt = f"""{self.stack[0].content}

Analyze this task using backward reasoning, then output execution steps.

1. FINAL_GOAL: What is the concrete final outcome?
2. BACKWARD: What must be true just before the goal? And before that? (2-3 levels)
3. FORWARD_STEPS: Concrete actions to achieve the goal.

Rules:
- Maximum 3 steps. Combine related actions into one step.
- If the task is simple (query, search, single action) → output exactly 1 STEP.
- Each step must be independently executable with available tools.

Task: {task}

Format:
FINAL_GOAL: <one sentence>
BACKWARD:
- Before goal: <condition A>
- Before A: <condition B>
FORWARD_STEPS:
STEP: <action>
STEP: <action>

If the task is simple, output just one STEP line."""
        try:
            response = self._api.call(
                [{"role": "user", "content": prompt}],
                max_tokens=768
            )
            text = self._extract_text(response)
            # 记录完整推理过程（供后续步骤参考）
            self._plan_reasoning = text
            # 解析正向步骤
            steps = []
            for line in text.split('\n'):
                line = line.strip()
                if line.upper().startswith('STEP:') or line.upper().startswith('STEP '):
                    desc = re.sub(r'^STEP\s*:?\s*', '', line, flags=re.IGNORECASE).strip()
                    if desc:
                        steps.append({"desc": desc, "done": False})
        except RuntimeError:
            print("  [Plan] API unavailable, using task as single step")
            steps = []
        if not steps:
            steps = [{"desc": task, "done": False}]
        return steps

    # ---- 主循环 ----
    def run(self, task: str, max_steps: int = None) -> str:
        max_steps = max_steps or Config.MAX_STEPS

        # 初始化能量管理器（顶层 Agent）
        if not self.parent:
            self.energy_manager = BayesianEnergyManager(
                total_energy=Config.CONTEXT_BUDGET - Config.CONTEXT_RESERVE,  # 154800E
                step_overhead=Config.STEP_OVERHEAD,
                cost_tool=Config.TOOL_ENERGY_COST,
            )
            # 注入相关历史经验
            self._inject_experience(task)

        # 自动匹配 Skill（仅顶层 agent 匹配，子 agent 通过父级 context 继承）
        if self.auto_skill and not self.active_skills and not self.parent:
            matched = self.skill_manager.match(task, top_n=1)
            if matched:
                self.active_skills = matched
                skill_text = f"\n\n## Skill: {matched[0].name}\n{matched[0].content}"
                old = self.stack[0].content
                self.stack[0] = StackFrame("constraint", old + skill_text, level=0)
                print(f"  [Auto-skill] activated: {matched[0].name}")

        # 生成计划
        print(f"\n[Plan] Analyzing task: {task}")
        self.subtask_queue = self._generate_plan(task)
        # 子任务去重
        if Config.ENABLE_SUBTASK_DEDUP:
            self.subtask_queue = self._merge_subtasks(self.subtask_queue)
        plan_text = "\n".join(f"  {i+1}. {s['desc']}" for i, s in enumerate(self.subtask_queue))
        # 记录计划节点到流程图
        if self.flowchart:
            try:
                if not self.parent:
                    # 根 Agent：创建完整的 task_start + subtask 结构
                    task_label = task[:40].replace('"', "'")
                    self.flowchart.add_node("task_start", f"任务: {task_label}...", shape="task")
                    self.flowchart.add_edge("start", "task_start")
                    self._fc_task_node = "task_start"
                    for sub_idx, sub in enumerate(self.subtask_queue):
                        sub_label = sub["desc"][:40].replace('"', "'")
                        sub_node = f"sub_{sub_idx}"
                        self.flowchart.add_node(sub_node, f"子任务 {sub_idx+1}: {sub_label}", shape="subtask")
                        self.flowchart.add_edge("task_start", sub_node)
                        setattr(self, f"_fc_sub_{sub_idx}_node", sub_node)
                else:
                    # 子 Agent：步骤直接挂在父 spawn 节点下，ID 含 agent_id 防止冲突
                    spawn_node = getattr(self, "_fc_parent_spawn_node", None)
                    if spawn_node:
                        # 子 Agent 的 task_node 就是 spawn 节点
                        self._fc_task_node = spawn_node
                        for sub_idx, sub in enumerate(self.subtask_queue):
                            sub_label = sub["desc"][:30].replace('"', "'")
                            # agent_id 后缀防止同深度多 agent 冲突
                            sub_node = f"d{self.depth}_{self.agent_id}_sub_{sub_idx}"
                            self.flowchart.add_node(sub_node, f"子任务 {sub_idx+1}: {sub_label}", shape="subtask")
                            self.flowchart.add_edge(spawn_node, sub_node)
                            setattr(self, f"_fc_sub_{sub_idx}_node", sub_node)
            except Exception:
                pass

        print(f"[Plan] {len(self.subtask_queue)} subtasks:\n{plan_text}")

        # plan 帧包含反向推理 + 正向步骤（供后续执行参考）
        reasoning = getattr(self, '_plan_reasoning', '')
        if reasoning:
            full_plan = reasoning + "\n\n## Steps:\n" + plan_text
        else:
            full_plan = plan_text
        self.stack[1] = StackFrame("plan", full_plan, level=1)

        # 基于计划复杂度调整能量预算（保证至少能执行一轮）
        if not self.parent:
            min_energy = len(self.subtask_queue) * 3 * self.energy_manager.step_overhead * 1.2
            if self.energy_manager.total_energy < min_energy:
                deficit = min_energy - self.energy_manager.total_energy
                self.energy_manager.total_energy += deficit
                self.energy_manager.energy += deficit
                print(f"  [Energy] Budget adjusted to {self.energy_manager.total_energy:.0f}")

        # 初始化贝叶斯预估器
        self.step_estimator = StepEstimator(len(self.subtask_queue))
        self._save_state()

        last_result = ""
        total_steps = 0

        for sub_idx, subtask in enumerate(self.subtask_queue):
            # 能量停止检查（主要停止条件）
            should_stop, reason = self.energy_manager.should_stop_with_estimator(self.step_estimator)
            if should_stop:
                print(f"\n[Stopped] {reason}")
                if self.flowchart and not self.parent:
                    try:
                        self.flowchart.add_node("stop_energy", f"停止: {reason[:30]}", shape="decision")
                        last_sub = getattr(self, f"_fc_sub_{sub_idx-1}_node", self._fc_task_node) if sub_idx > 0 else self._fc_task_node
                        if last_sub and last_sub in self.flowchart.nodes:
                            self.flowchart.add_edge(last_sub, "stop_energy", label="能量耗尽")
                    except Exception:
                        pass
                break

            # 步数安全网（防止能量充足但无限循环）
            if total_steps >= max_steps:
                print(f"\n[Halted] Safety net: {max_steps} steps reached")
                self.emit_event(SubAgentEventType.MAX_STEPS_REACHED, f"{total_steps}/{max_steps}")
                if self.flowchart and not self.parent:
                    try:
                        self.flowchart.add_node("stop_halted", f"停止: {max_steps}步安全网", shape="decision")
                    except Exception:
                        pass
                break

            print(f"\n{'='*40}")
            # 记录当前子任务上下文
            self._fc_current_sub_idx = sub_idx
            self._fc_current_sub_node = getattr(self, f"_fc_sub_{sub_idx}_node", self._fc_task_node)

            print(f"  Subtask {sub_idx+1}/{len(self.subtask_queue)}: {subtask['desc']}")
            self.print_progress(max_steps)
            print(f"{'='*40}")

            progress = self._format_progress(sub_idx)
            step_task = f"{task}\n\nCurrent subtask: {subtask['desc']}\n{progress}"

            steps_before = self.step_counter
            try:
                result, done = self.execute_next_step(step_task)
                # 空响应检测：长度 < 50 视为实质失败
                if not result or len(result.strip()) < 50:
                    print(f"  [Warning] Empty/minimal response ({len(result.strip()) if result else 0} chars)")
                    self.energy_manager.update_subtask(f"sub_{sub_idx}", False)
                    subtask['done'] = False
                else:
                    subtask['done'] = True
                    self.energy_manager.update_subtask(f"sub_{sub_idx}", True)
                steps_consumed = self.step_counter - steps_before
                total_steps += steps_consumed
                last_result = result
                # 更新能量管理器
                self.energy_manager.update_done(done)
                # 更新贝叶斯预估器
                self.step_estimator.update(steps_consumed)
                print(f"\nResult:\n{result}")
            except RuntimeError as e:
                print(f"Error: {e}")
                self.emit_event(SubAgentEventType.FATAL_ERROR, str(e))
                self.energy_manager.update_subtask(f"sub_{sub_idx}", False)
                self._save_state()
                raise

        # 汇合
        level2_frames = [f for f in self.stack if f.level == 2]
        if len(level2_frames) >= 2:
            print(f"\n{'='*40}")
            print(f"  Merging {len(level2_frames)} execution results")
            print(f"{'='*40}")
            self.merge_level(2)

        # 紧急交付：如果没完成但有部分成果，用 bypass_energy 生成最终交付物
        done_count = sum(1 for s in self.subtask_queue if s.get('done'))
        if not self.parent and done_count > 0 and not all(s.get('done') for s in self.subtask_queue):
            print(f"\n{'='*40}")
            print(f"  [Emergency] Delivering partial results ({done_count}/{len(self.subtask_queue)} done)")
            print(f"{'='*40}")
            # 流程图：紧急交付
            if self.flowchart:
                try:
                    self.flowchart.add_node("emergency", f"紧急交付 ({done_count}/{len(self.subtask_queue)})", shape="decision")
                except Exception:
                    pass
            # 收集已有成果
            collected = []
            for f in self.stack:
                if f.level >= 2 and f.content and len(f.content.strip()) > 50:
                    collected.append(f.content[:500])
            if collected:
                summary_prompt = (f"任务: {task}\n\n以下是部分执行结果，请整合为最终交付物（包含已完成部分，"
                                  f"标注未完成部分）:\n\n" + "\n---\n".join(collected))
                try:
                    resp = self._api.call(
                        [{"role": "user", "content": summary_prompt}],
                        system="你是交付助手。整理已有成果为完整报告，标注缺失部分。",
                        max_tokens=2000,
                        bypass_energy=True
                    )
                    last_result = self._extract_text(resp)
                    print(f"  [Emergency] Delivered {len(last_result)} chars")
                except Exception as e:
                    print(f"  [Emergency] Failed: {e}")

        # 经验记录（轻量收集，不调 LLM）
        self._collect_experience_metadata(
            task=task, plan=plan_text, step_count=total_steps,
            success=all(s.get('done') for s in self.subtask_queue),
            final_result=last_result
        )

        # 保存对话历史
        self.conversation_history.append({"q": task, "a": last_result})
        self._save_state()

        # 保存到历史目录
        self._save_history(task, last_result, total_steps,
                           all(s.get('done') for s in self.subtask_queue))

        # 自动刷新经验到数据库（仅顶层 Agent）
        if not self.parent and self._pending_experiences:
            self.flush_experiences()

        # 终端奖励 + Skill 押金结算（仅顶层 Agent）
        if not self.parent:
            task_success = (all(s.get('done') for s in self.subtask_queue)
                            and last_result and len(last_result) > 20)
            em = self.energy_manager
            # 结算 Skill 押金
            if hasattr(em, '_skill_deposit'):
                deposit = em._skill_deposit
                skill_name = em._skill_name
                if task_success:
                    em.energy += deposit
                    print(f"  [Skill] '{skill_name}' deposit {deposit:.0f}E refunded")
                else:
                    em.commit_spent(deposit)
                    print(f"  [Skill] '{skill_name}' deposit {deposit:.0f}E forfeited")
                del em._skill_deposit
                del em._skill_name
                # 流程图：Skill 押金结算
                if self.flowchart:
                    try:
                        sk_id = "skill_settle"
                        label = f"Skill {'退回' if task_success else '罚没'} {deposit:.0f}E"
                        self.flowchart.add_node(sk_id, label, shape="energy")
                        self.flowchart.add_edge(self._fc_task_node, sk_id, label="Skill")
                    except Exception:
                        pass
            # 动态终端奖励
            expected = self.step_estimator.predict_total(0)[0] if self.step_estimator else float(total_steps)
            self.energy_manager.grant_terminal_reward(
                task_success,
                plan_complexity=len(self.subtask_queue),
                actual_steps=total_steps,
                expected_steps=expected
            )
            # 流程图：奖励节点
            if self.flowchart:
                try:
                    reward = 0
                    # 从 grant_terminal_reward 的返回值估算
                    if task_success:
                        base = Config.REWARD_BASE
                        diff = min(2.0, 1.0 + 0.3 * (len(self.subtask_queue) - 1))
                        spent_ratio = em.total_spent / em.total_energy if em.total_energy > 0 else 0
                        eff = max(0.3, 1.0 - spent_ratio)
                        reward = base * diff * eff
                    rw_id = "reward"
                    label = f"{'✓ 成功' if task_success else '✗ 失败'} +{reward:.0f}E奖励"
                    self.flowchart.add_node(rw_id, label, shape="decision")
                    # 连到最后一个 sub 节点或 task_start
                    last_done_sub = None
                    for si, s in enumerate(self.subtask_queue):
                        if s.get('done'):
                            last_done_sub = f"sub_{si}"
                    anchor = last_done_sub or self._fc_task_node
                    if anchor and anchor in self.flowchart.nodes:
                        self.flowchart.add_edge(anchor, rw_id, label="完成")
                except Exception:
                    pass
            # 最终状态
            print(f"  [Final] Energy: {em.energy:.0f} | Spent: {em.total_spent:.0f}/{em.total_energy:.0f}")

        print("\nDone!")
        # 关闭流程图
        if self.flowchart:
            try:
                self.flowchart.finalize()
            except Exception:
                pass
        return last_result

    def _format_progress(self, current_idx: int) -> str:
        lines = []
        for i, sub in enumerate(self.subtask_queue):
            status = "✓" if sub['done'] else ("▶" if i == current_idx else "○")
            lines.append(f"{status} {i+1}. {sub['desc']}")
        return "Progress:\n" + "\n".join(lines)

# ============ 连接测试（轻量） ============
def check_connection(client: anthropic.Anthropic) -> bool:
    """用最短请求验证 API 可达，529 过载自动重试"""
    print("Testing API connection...")
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=Config.MODEL,
                max_tokens=5,
                messages=[{"role": "user", "content": "OK"}]
            )
            if response.content:
                print("✓ API connected")
                return True
        except anthropic.APIStatusError as e:
            if e.status_code == 529:
                wait = 2 ** attempt + random.uniform(0, 2)
                print(f"  API overloaded, retry {attempt+1}/3 in {wait:.1f}s...")
                time.sleep(wait)
                continue
            print(f"✗ Connection failed: {e}")
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            break
    return False

# ============ CLI ============
def main(t):
    parser = argparse.ArgumentParser(description="Autonomous Agent")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    parser.add_argument("--task", type=str, help="Task description")
    parser.add_argument("--system", type=str, help="System prompt")
    parser.add_argument("--steps", type=int, help="Max steps")
    parser.add_argument("--skill", type=str, action='append', help="Specify Skill")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--auto-skill", action="store_true", help="Auto-match Skills")
    parser.add_argument("--list-skills", action="store_true", help="List available Skills")
    args = parser.parse_args()

    if args.list_skills:
        sm = SkillManager()
        names = sm.list_names()
        if names:
            print("Available Skills:")
            for name in names:
                s = sm.get(name)
                print(f"  {name}: {s.description}")
        else:
            print("No Skills found (create in ./skills/)")
        return

    # 启动时检测 state 文件膨胀（超过 1MB 自动清除重置）
    for fname in ("state.json", "pending_exp.json"):
        fpath = os.path.join(Config.WORK_DIR, fname)
        try:
            if os.path.exists(fpath) and os.path.getsize(fpath) > 1_048_576:
                sz_mb = os.path.getsize(fpath) / 1_048_576
                print(f"  [Sanity] {fname} is {sz_mb:.1f}MB (>1MB), resetting")
                os.remove(fpath)
        except Exception:
            pass

    if args.resume:
        print("Resuming saved state...")
        try:
            agent = Agent.load_state()
            print(f"✓ Resumed at Step {agent.step_counter}")
        except FileNotFoundError:
            print("No saved state, creating new Agent")
            agent = Agent(args.system or "你是一个AI编程助手。",
                                skills=args.skill, auto_skill=args.auto_skill)
    else:
        agent = Agent(args.system or "你是一个AI编程助手。",
                            skills=args.skill, auto_skill=args.auto_skill)

    # 连接测试复用 Agent 的 client（省一次预热调用）
    if not check_connection(agent.client):
        print("\nCheck API config in .env (ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL)")
        sys.exit(1)

    task = args.task or t
    if args.interactive:
        agent.interact(initial_task=task if task else None, max_steps=args.steps)
    else:
        agent.run(task, args.steps)

if __name__ == "__main__":
    main("34254e576a0356ee46877635dc03bd4fb414378e.jpg这是什么")
