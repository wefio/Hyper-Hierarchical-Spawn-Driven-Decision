"""HHSDD Agent configuration — single source of truth.

Extracted from agent.py to break the circular import:
  experience_store.py needs Config, but importing agent.py to get it
  would pull in the entire 3400-line module.
"""
import os
from pathlib import Path


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
    TOKEN_COST_INPUT = float(os.getenv("TOKEN_COST_INPUT", "1.0"))    # token→能量 1:1
    SPAWN_INVEST_MIN = int(os.getenv("SPAWN_INVEST_MIN", "2000"))     # spawn 最低投资
    SPAWN_INVEST_RATIO = float(os.getenv("SPAWN_INVEST_RATIO", "0.01"))  # spawn 投资比例
    SPAWN_RESERVE_RATIO = float(os.getenv("SPAWN_RESERVE_RATIO", "0.15"))  # spawn 预留比例
    SAVEPOINT_BACKUP_COST = float(os.getenv("SAVEPOINT_BACKUP_COST", "0.1"))  # savepoint 备份成本系数
    BATCH_TOOL_COST_MULT = float(os.getenv("BATCH_TOOL_COST_MULT", "3"))  # 并行工具批量倍率
    CONTEXT_RESERVE_RATIO = float(os.getenv("CONTEXT_RESERVE_RATIO", "0.25"))  # context 保留比例
    REWARD_BASE = float(os.getenv("REWARD_BASE", "5000"))             # 终端奖励基数
    CMD_REFUND_PER_SEC = float(os.getenv("CMD_REFUND_PER_SEC", "500"))  # 命令超时退款速率
    TOOL_ENERGY_COST = float(os.getenv("TOOL_ENERGY_COST", "200"))      # 单次工具调用预扣能量

    # ---- 上下文管理 ----
    COMPRESSION_THRESHOLD = float(os.getenv("COMPRESSION_THRESHOLD", "0.9"))  # 上下文占比触发压缩
    TOOL_RESULT_BUDGET = int(os.getenv("TOOL_RESULT_BUDGET", "4000"))         # 单次工具结果字符上限
    TOOL_RESULT_MIN = int(os.getenv("TOOL_RESULT_MIN", "2000"))              # 工具结果字符下限（100 token/档）
    MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "120000"))         # 消息列表总字符上限

    # ---- 安全网（正常由能量系统停止） ----
    MAX_STEPS = int(os.getenv("MAX_STEPS", "100"))
    MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "100"))
    REWARD_STEP_TIER = int(os.getenv("REWARD_STEP_TIER", "50"))  # ≤此步数全额奖励
    MAX_SPAWN_DEPTH = int(os.getenv("MAX_SPAWN_DEPTH", "0"))  # 0=不限制

    # ---- 路径 (基于脚本所在目录, 非 CWD) ----
    _BASE = os.path.dirname(os.path.abspath(__file__))
    WORK_DIR = os.getenv("AGENT_WORK_DIR", os.path.join(_BASE, "agent_state"))
    SKILLS_DIR = os.getenv("SKILLS_DIR", os.path.join(_BASE, "skills"))
    EXPERIENCE_DIR = os.getenv("EXPERIENCE_DIR", os.path.join(_BASE, "experience_store"))
    SCRIPTS_DIR = os.getenv("SCRIPTS_DIR", os.path.join(_BASE, "scripts"))
    HISTORY_DIR = os.getenv("HISTORY_DIR", os.path.join(_BASE, "history"))
    DEBUG_DIR = os.getenv("AGENT_DEBUG_DIR", os.path.join(_BASE, "debug_messages"))
    SAVEPOINT_DIR = os.getenv("SAVEPOINT_DIR", os.path.join(_BASE, "savepoints"))

    # ---- Pointer 磁盘缓存系统 ----
    ARCHIVE_DIR = os.getenv("ARCHIVE_DIR", os.path.join(_BASE, "agent_state", "archive"))
    MAX_PRIMARY_POINTERS = int(os.getenv("MAX_PRIMARY_POINTERS", "20"))
    POINTER_MERGE_THRESHOLD = float(os.getenv("POINTER_MERGE_THRESHOLD", "0.8"))
    ARCHIVE_MAX_SIZE_MB = int(os.getenv("ARCHIVE_MAX_SIZE_MB", "500"))
    POINTER_CONTENT_BUDGET = int(os.getenv("POINTER_CONTENT_BUDGET", "500000"))
    RECALL_DEFAULT_TOKENS = int(os.getenv("RECALL_DEFAULT_TOKENS", "2000"))
    RECALL_MAX_TOKENS = int(os.getenv("RECALL_MAX_TOKENS", "8000"))
    HOT_POINTER_LIMIT = int(os.getenv("HOT_POINTER_LIMIT", "5"))

    # ---- 插件 ----
    PLUGINS = os.getenv("PLUGINS", "").split(",")  # 逗号分隔插件模块名，如 "so100_plugin"

    # ---- 策略 ----
    AUTO_PLAN = os.getenv("AUTO_PLAN", "smart")  # always / smart / never
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


# ============ 插件加载 ============
import importlib as _importlib
_PLUGINS: list = []

def _load_plugins():
    """从 Config.PLUGINS 动态加载插件模块。"""
    global _PLUGINS
    for name in Config.PLUGINS:
        name = name.strip()
        if not name:
            continue
        try:
            mod = _importlib.import_module(name)
            _PLUGINS.append(mod)
            print(f"  [Plugin] loaded: {name}")
        except ImportError as e:
            print(f"  [Plugin] skipped: {name} ({e})")

_load_plugins()


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
