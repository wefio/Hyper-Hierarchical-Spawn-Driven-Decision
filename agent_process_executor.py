"""Process executor — ToolExecutor, tool definitions, energy hooks."""
from __future__ import annotations
import json
import os
import re
import time
import random
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from config import Config, smart_truncate, _PLUGINS
from agent_memory_frame import StackFrame, SavepointMeta
from agent_kernel_glue import CacheProvider, SubAgentEvent, SubAgentEventType, SavepointManager
from agent_scheduler_energy import BayesianEnergyManager

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
                "mode": {"type": "string", "enum": ["explore", "exploit"], "description": "explore=risky but rewarding on success; exploit=safe and stable. Default: exploit"},
                "_child_can_spawn": {"type": "boolean", "description": "Child agent can spawn its own sub-agents (default true, set false for leaf nodes)"},
                "skip_plan": {"type": "boolean", "description": "Skip sub-agent's planning phase (saves 1 API call). Set true for simple, well-defined subtasks.", "default": False}
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
    {
        "name": "recall",
        "description": (
            "Recalls archived content by pointer_id or keyword query. "
            "Use offset/max_tokens to read partial content (demand paging, like mmap). "
            "Recalling consumes energy proportional to injected tokens. "
            "Content is marked reclaimable (recycled first on next context pressure)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pointer_id": {
                    "type": "string",
                    "description": "Pointer ID to recall, e.g. 'ptr_a1b2c3d4'"
                },
                "query": {
                    "type": "string",
                    "description": "Keyword search to find pointer IDs when you forgot the exact ID. Returns candidate list with summary and token size."
                },
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "description": "Token offset for partial recall (like mmap paging). Default: 0 (start)."
                },
                "max_tokens": {
                    "type": "integer",
                    "default": 2000,
                    "description": "Max tokens to inject. Default 2000, max 8000. Prevents token shock."
                }
            },
            "required": []
        }
    },
    {
        "name": "savepoint",
        "description": "Savepoint tool for iterative exploration within the current agent. "
        "create: snapshot current state to disk (one active savepoint per agent). "
        "commit: keep exploration result as conclusion summary, savepoint becomes history. "
        "pop: discard exploration, restore to snapshot state, refund all energy spent since create. "
        "list: show active and historical savepoints. "
        "Use create before risky/uncertain operations, commit on success, pop on failure to retry differently.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "commit", "pop", "list"],
                    "description": "create=snapshot state, commit=keep result, pop=restore+refund, list=show all"
                },
                "name": {
                    "type": "string",
                    "description": "Optional savepoint name (auto-generated if omitted)"
                },
                "summary": {
                    "type": "string",
                    "description": "Conclusion summary (for commit) or failure reason (for pop)"
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "read_peer",
        "description": (
            "Read peer agent results from the L2 shared cache (same spawn group siblings only). "
            "Returns a list of pointers (agent_id, step_id, summary, ptr_id, tokens). "
            "Use this to check sibling progress. To load full content, use recall with ptr_id. "
            "Results are signature-verified; tampered entries are dropped."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Filter by specific sibling agent_id (optional)"
                },
                "query": {
                    "type": "string",
                    "description": "Keyword search in summaries (optional)"
                }
            },
            "required": []
        }
    },

    {
        "name": "ipython",
        "description": (
            "Execute Python code. Three modes: isolated (fresh subprocess), "
            "session (shared variables within same session_id), shared (L2-linked). "
            "Set timeout for long-running code. Output returned as text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "mode": {"type": "string", "enum": ["isolated", "session", "close"], "description": "isolated=fresh process, session=persistent kernel"},
                "session_id": {"type": "string", "description": "Session name for persistent kernel (session mode)"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"}
            },
            "required": ["code", "mode"]
        }
    },
]

# Merge plugin tool definitions
for _plugin in _PLUGINS:
    TOOL_DEFINITIONS.extend(_plugin.get_tool_definitions())

def _build_cached_tools(cache_provider: CacheProvider) -> list:
    """构建带 cache_control 的工具定义（缓存结果，避免重复 deepcopy）"""
    cc = cache_provider.cache_control()
    if not cc:
        return TOOL_DEFINITIONS
    # 收集需要 cache_control 的工具名前缀
    cache_prefixes = set()
    for p in _PLUGINS:
        cache_prefixes |= p.get_cache_rules()
    tools = []
    for t in TOOL_DEFINITIONS:
        if any(t.get("name", "").startswith(prefix) for prefix in cache_prefixes):
            tools.append({**t, "cache_control": cc})
        else:
            tools.append(t)
    return tools

def _get_energy_mgr(agent_context: dict) -> 'BayesianEnergyManager':
    """从 agent_context 中提取能量管理器"""
    parent = agent_context.get("parent_agent") if agent_context else None
    return parent.energy_manager if parent else None


def energy_hooks(*, deduct=None, settle=None, abort_msg="Insufficient energy"):
    """通用能量生命周期装饰器。

    Args:
        deduct: (args, em) -> cost | False — 预扣能量，返回False则中止
        settle: (args, result, em, cost, agent_context) -> None — 始终执行（finally），用于结算/返费
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
                return result
            finally:
                if em and settle:
                    settle(args, result, em, cost, agent_context)
        return wrapper
    return decorator


# ── 能量钩子定义 ──
def _cmd_deduct(args, em):
    return em.pre_consume_for_cmd(args.get("command", ""))

def _cmd_refund(args, result, em, cost, agent_context=None):
    if cost and cost > 0:
        refund = em.refund_for_cmd(
            args.get("command", ""), cost,
            result.get("execution_time", 0), result.get("success", False))
        if refund > 0.01:
            actual = result.get("execution_time", 0)
            print(f"  [Energy] +{refund:.1f} refunded ({actual:.1f}s, {'ok' if result['success'] else 'FAIL'})")

def _skill_deduct(args, em):
    deposit = float(Config.SPAWN_INVEST_MIN)
    if not em.charge(deposit):
        return False
    return deposit

def _spawn_deduct(args, em):
    if not em.should_spawn():
        return False
    # 窗口对齐：父池 vs 子模型窗口，取较小
    child_model_ctx = args.get("_child_context_window", em.total_energy)
    effective = min(em.energy, child_model_ctx)
    invest = max(float(Config.SPAWN_INVEST_MIN), effective * Config.SPAWN_INVEST_RATIO)
    after_invest = effective - invest
    freeze = after_invest * Config.SPAWN_RESERVE_RATIO
    if freeze < max(em.energy * 0.01, 500):
        return False
    if not em.charge(invest + freeze):
        return False
    em._reserve_stack.append(freeze)
    child_budget = after_invest * (1 - Config.SPAWN_RESERVE_RATIO)
    em._last_spawn = {"invest": invest, "freeze": freeze, "aligned": effective, "child_pool": child_budget}
    # 将 freeze 存入 args 供 child 携带（吸收时用）
    args["_spawn_freeze"] = freeze
    print(f"  [Spawn] aligned={effective:.0f}E invest={invest:.0f}E freeze={freeze:.0f}E child_pool={child_budget:.0f}E")
    return child_budget  # 子的 total_energy

def _spawn_settle(args, result, em, cost, agent_context=None):
    if cost is None:
        return
    parent_agent = (agent_context or {}).get("parent_agent")
    spawn_depth = parent_agent.depth + 1 if parent_agent else 1
    fc = getattr(parent_agent, 'flowchart', None) if parent_agent else None
    last = getattr(em, '_last_spawn', {})
    invest = last.get("invest", cost)
    reserve = em._reserve_stack.pop() if em._reserve_stack else 0
    em.credit(reserve)
    print(f"  [Spawn] freeze released: {reserve:.0f}E -> energy {em.energy:.0f}")
    success = result.get("success", False) if result else False
    mode = result.get("_mode", "exploit") if result else "exploit"
    if mode == "explore":
        if success:
            em.credit(invest + invest * em.explore_roi)
        else:
            em.spend(invest)
    else:
        if success:
            em.credit(invest)
        else:
            em.credit(invest * 0.5)
            em.spend(invest * 0.5)
    if not success:
        em.update_spawn(False)
    if fc:
        try:
            fc.record_lifecycle(spawn_depth, release=reserve, settle=invest)
        except Exception:
            pass





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
            "run_command": energy_hooks(deduct=_cmd_deduct, settle=_cmd_refund)(ToolExecutor._run_command),
            "use_skill": energy_hooks(deduct=_skill_deduct,
                                      abort_msg="Insufficient energy for skill deposit")(ToolExecutor._use_skill),
            "savepoint": ToolExecutor._savepoint,
            "recall": ToolExecutor._recall,
            "read_peer": ToolExecutor._read_peer,
            "ipython": ToolExecutor._ipython,
        }
        # Merge plugin handlers
        for plugin in _PLUGINS:
            handlers.update(plugin.get_tool_handlers())

        handler = handlers.get(name)
        if not handler:
            return {"success": False, "output": f"Unknown tool: {name}. Available: {', '.join(handlers.keys())}"}
        try:
            result = handler(args, budget, agent_context or {})
            # ── 记录 tool 调用序列（供技能保存）──
            parent = (agent_context or {}).get("parent_agent")
            if parent and hasattr(parent, "_current_tool_calls"):
                record = {
                    "tool": name,
                    "action": args.get("action", ""),
                    "params": {k: v for k, v in args.items() if k != "action"},
                    "success": result.get("success", False),
                    "timestamp": time.time(),
                }
                # Track IK fallback for reward penalty
                if result.get("ik_fallback"):
                    record["ik_fallback"] = True
                    record["requested_direction"] = result.get("requested_direction", "")
                    record["actual_direction"] = result.get("approach_direction", "")
                # Track verified field for task verification
                if "verified" in result:
                    record["verified"] = result["verified"]
                parent._current_tool_calls.append(record)

            # Plugin post-tool hook (IK settlement, etc.)
            if parent:
                for plugin in _PLUGINS:
                    plugin.on_tool_executed(parent, name, result)

            return result
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
        # 禁用代理 — requests 默认读取 HTTP_PROXY，会阻断内网 API 调用
        _NO_PROXY = {"http": None, "https": None}
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
                resp = _req.get(path, timeout=15, proxies=_NO_PROXY)
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
                proxies=_NO_PROXY,
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
    def _recall(args: dict, budget: int, agent_context: dict = None) -> dict:
        """Recall archived content by pointer_id or keyword query."""
        parent = (agent_context or {}).get("parent_agent")
        if not parent or not hasattr(parent, 'pointer_store'):
            return {"success": False, "output": "Pointer store not available"}

        pointer_id = args.get("pointer_id", "")
        query = args.get("query", "")
        offset = int(args.get("offset", 0))
        max_tokens = min(int(args.get("max_tokens", Config.RECALL_DEFAULT_TOKENS)),
                         Config.RECALL_MAX_TOKENS)

        # 模式 1: 关键词搜索
        if query and not pointer_id:
            results = parent.pointer_store.search_keywords(query, parent.scope, limit=5)
            if not results:
                return {"success": True, "output": "No matching pointers found."}
            lines = ["## Matching Pointers\n"]
            for r in results:
                lines.append(f"- `{r['id']}`: {r['summary'][:120]} ({r['tokens']} tokens, used {r['use_count']}x)")
            return {"success": True, "output": "\n".join(lines)}

        # 模式 2: 精确召回
        if not pointer_id:
            return {"success": False, "output": "Provide pointer_id or query"}

        result = parent.pointer_store.recall(
            pointer_id, scope=parent.scope,
            offset=offset, max_tokens=max_tokens,
        )
        if result is None:
            return {"success": False, "output": f"Pointer '{pointer_id}' not found or scope inaccessible."}

        content, meta = result
        injected_tokens = meta["injected_tokens"]

        # 流程图：RECALL 节点
        if parent.flowchart:
            try:
                fc_step = getattr(parent, '_current_step_node', None)
                if fc_step:
                    ptr_node = f"ptr_recall_{parent.depth}_{parent.agent_id}_{pointer_id}"
                    parent.flowchart.add_node(
                        ptr_node,
                        f"RECALL {pointer_id} ({injected_tokens}E)",
                        shape="pointer")
                    parent.flowchart.add_edge(fc_step, ptr_node, label="召回")
            except Exception:
                pass

        # 消耗能量（recall = swap in，有代价）
        if injected_tokens > 0:
            parent.energy_manager.charge(injected_tokens)

        # 原地展开：替换 pointer stub 为全文（位置不变，上下文不漂移）
        expanded = False
        for f in parent.stack:
            if getattr(f, 'pointer_id', '') == pointer_id:
                f.content = smart_truncate(content, budget, label="recall ")
                f.use_count = getattr(f, 'use_count', 0) + 1
                f.last_used_step = parent.step_counter
                f.reclaimable = True     # 原文在磁盘有备份，优先回收
                f.expanded = True        # _build_messages 输出全文
                expanded = True
                print(f"  [Recall] Expanded {pointer_id} in-place ({injected_tokens}E)")
                break

        # 未找到原始 stub → 回退到追加模式
        if not expanded:
            parent.stack.append(StackFrame(
                "step_detail", content,
                step_id=meta.get("step_id", 0),
                level=2, agent_id=parent.agent_id,
                pointer_id=pointer_id, reclaimable=True,
                use_count=1, last_used_step=parent.step_counter,
            ))
            print(f"  [Recall] Appended {pointer_id} (stub not found, {injected_tokens}E)")

        output = smart_truncate(content, budget, label="recall ")
        if meta["truncated"]:
            output += f"\n\n... [内容已截断，共 {meta['tokens']} tokens，offset={offset + max_tokens} 可继续召回] ..."
        output += f"\n\n[Recalled {injected_tokens} tokens from {pointer_id}, -{injected_tokens}E charged. Content is reclaimable.]"
        print(f"  [Recall] {pointer_id}: {injected_tokens} tokens injected, -{injected_tokens}E charged")
        return {"success": True, "output": output}

    @staticmethod
    def _read_peer(args: dict, budget: int, agent_context: dict = None) -> dict:
        """Read sibling agent results from L2 shared cache (same spawn group only)."""
        parent = (agent_context or {}).get("parent_agent")
        if not parent or not getattr(parent, 'l2_cache', None):
            return {"success": True, "output": "L2 cache not available (not in a parallel spawn group)."}
        l2 = parent.l2_cache
        results = l2.read_peer(
            agent_id=args.get("agent_id"),
            query=args.get("query"),
        )
        if not results:
            return {"success": True, "output": "No matching peer pointers found."}
        lines = [f"## Peer Pointers ({len(results)})"]
        for r in results:
            ptr_info = f"ptr_id={r['ptr_id']}" if r.get("ptr_id") else "no ptr"
            lines.append(
                f"- `{r['agent_id']}` step {r['step_id']}: {r['summary'][:120]} "
                f"({ptr_info}, {r.get('tokens', 0)} tokens)"
            )
        return {"success": True, "output": "\n".join(lines)}

    @staticmethod
    def _ipython(args: dict, budget: int, agent_context: dict = None) -> dict:
        """IPython 沙盒：isolated（子进程）| session（持久内核）| close（销毁会话）。"""
        code = args.get("code", "")
        mode = args.get("mode", "isolated")
        session_id = args.get("session_id", "default")
        timeout = min(args.get("timeout", 30), 120)
        parent = (agent_context or {}).get("parent_agent")
        if not parent:
            return {"success": False, "output": "No agent context"}

        # close 模式：显式销毁会话
        if mode == "close":
            sessions = getattr(parent, '_ipy_sessions', {})
            if session_id in sessions:
                del sessions[session_id]
                return {"success": True, "output": f"Session '{session_id}' closed"}
            return {"success": True, "output": f"Session '{session_id}' not found"}

        if not code:
            return {"success": False, "output": "Missing code parameter"}

        parent = (agent_context or {}).get("parent_agent")
        if not parent:
            return {"success": False, "output": "No agent context"}

        if mode == "session":
            # 持久内核：同一 agent 同一 session_id 复用
            if not hasattr(parent, '_ipy_sessions'):
                parent._ipy_sessions = {}
            if session_id not in parent._ipy_sessions:
                import code as _code_mod
                import io
                stdout = io.StringIO()
                stderr = io.StringIO()
                console = _code_mod.InteractiveConsole(locals={})
                parent._ipy_sessions[session_id] = {
                    "console": console, "stdout": stdout, "stderr": stderr,
                }
            session = parent._ipy_sessions[session_id]
            console = session["console"]
            out_buf = session["stdout"]
            err_buf = session["stderr"]
            # 临时替换 stdout
            import sys
            old_out, old_err = sys.stdout, sys.stderr
            try:
                sys.stdout = out_buf
                sys.stderr = err_buf
                # 检查是否单行表达式
                code_stripped = code.strip()
                if "\n" not in code_stripped and not code_stripped.startswith(("for ", "if ", "while ", "def ", "class ", "import ", "from ", "try:", "with ")):
                    console.push(code_stripped)
                else:
                    console.runsource(code_stripped, "<ipython>", "exec")
            except Exception as e:
                err_buf.write(str(e))
            finally:
                sys.stdout, sys.stderr = old_out, old_err
            output = out_buf.getvalue()
            out_buf.truncate(0)
            out_buf.seek(0)
            return {"success": True, "output": output or "(no output)"}
        else:
            # isolated: 子进程执行
            import subprocess, tempfile, os as _os
            tmp = tempfile.mkdtemp(prefix="ipy_")
            try:
                pyfile = _os.path.join(tmp, "script.py")
                with open(pyfile, "w", encoding="utf-8") as f:
                    f.write(code)
                result = subprocess.run(
                    ["python", pyfile], capture_output=True, text=True,
                    timeout=timeout, cwd=tmp,
                    encoding="utf-8", errors="replace",
                )
                output = result.stdout
                if result.stderr:
                    output += "\n[stderr]\n" + result.stderr[:500]
                if not output.strip():
                    output = f"(exit {result.returncode})"
                return {"success": result.returncode == 0, "output": output,
                        "exit_code": result.returncode}
            except subprocess.TimeoutExpired:
                return {"success": False, "output": f"Timeout after {timeout}s"}
            except Exception as e:
                return {"success": False, "output": f"Error: {e}"}
            finally:
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def _savepoint(args: dict, budget: int, agent_context: dict = None) -> dict:
        """存档点工具 — create / commit / pop / list"""
        action = args.get("action", "list")
        agent = (agent_context or {}).get("parent_agent")
        if not agent:
            return {"success": False, "output": "No agent context — savepoint requires a running agent"}

        if action == "create":
            return SavepointManager.create(agent, args.get("name", None))
        elif action == "commit":
            return SavepointManager.commit(agent, args.get("summary", ""))
        elif action == "pop":
            return SavepointManager.pop(agent, args.get("reason", ""))
        elif action == "list":
            return SavepointManager.list_savepoints(agent)
        else:
            return {"success": False,
                    "output": f"Unknown savepoint action: {action}. Use create/commit/pop/list"}

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
            from agent import Agent  # late import to break circular dependency
            child_system = agent_context.get("system_prompt", "你是一个AI编程助手。")

            # 并行模式：使用独立能量管理器（CoW），不共享父级
            kw = {}
            # 模型路由：允许 LLM 指定子 Agent 使用的模型
            if args.get("_child_model") and isinstance(args["_child_model"], dict):
                kw["model_spec"] = args["_child_model"]
            if args.get("_parallel_mode"):
                from agent_scheduler_energy import BayesianEnergyManager as BEM
                child_em = BEM(
                    total_energy=args["_child_total_energy"],
                    step_overhead=Config.STEP_OVERHEAD,
                    cost_tool=Config.TOOL_ENERGY_COST,
                )
                kw["energy_mgr"] = child_em
                # L2 同级共享缓存（同 spawn 组兄弟共用）
                if args.get("_l2_cache") is not None:
                    kw["l2_cache"] = args["_l2_cache"]
            # 父声明子不能派生（叶子节点）
            if "_child_can_spawn" in args:
                kw["can_spawn"] = args["_child_can_spawn"]
            elif "_parallel_mode" in args:
                kw["can_spawn"] = False  # 并行子默认不派生

            child = Agent(
                system_prompt=child_system,
                depth=current_depth + 1,
                parent=parent_agent,
                **kw
            )
            child._spawn_freeze = args.get("_spawn_freeze", 0)  # 吸收预算用
            # 记录子 Agent 节点（带序号）
            spawn_node = f"spawn_{current_depth+1}_{child.agent_id}"
            args["_child_id"] = child.agent_id  # 供 spawn_settle 定位节点
            if parent_agent and hasattr(parent_agent, 'flowchart'):
                try:
                    seq = parent_agent.flowchart.next_spawn_seq(current_depth + 1)
                    mode_tag = "🔍" if mode == "explore" else "🔧"
                    parent_agent.flowchart.add_node(spawn_node, f"子 Agent #{seq} (深度 {current_depth+1}, {mode_tag}{mode})", shape="agent")
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

            # 继承父级 skip_plan 决策：显式参数 > 父级决策 > False
            inherit_skip = args.get("skip_plan", False) or getattr(parent_agent, '_skip_plan_resolved', False)
            result = child.run(task, max_steps=Config.MAX_STEPS, skip_plan=inherit_skip)
            child.emit_event(SubAgentEventType.TASK_COMPLETED, f"Completed: {task[:50]}")

            output = result or "(no output)"

            if args.get("_parallel_mode"):
                # 并行模式：不吸收、不上提经验，由调用方统一处理
                child_experiences = list(child._pending_experiences) if child._pending_experiences else []
                child_reports = [r.to_dict() for r in child.reports_history] if child.reports_history else []
                return {"success": True, "output": smart_truncate(output, budget, label="sub-agent "),
                        "_mode": mode, "_child_experiences": child_experiences,
                        "_child_reports": child_reports}
            else:
                # 串行模式：直接吸收到父级
                if energy_mgr:
                    energy_mgr.update_spawn(True)
                if parent_agent:
                    parent_agent._absorb_child_result(output, task, child_agent=child)
                    # 子节点报告上提
                    if child.reports_history:
                        parent_agent.reports_history.extend(child.reports_history)
                    if child._pending_experiences:
                        n = len(child._pending_experiences)
                        parent_agent._pending_experiences.extend(child._pending_experiences)
                        parent_agent._save_pending_experiences()
                        print(f"  [Experience] propagated {n} records from child")
                        child._pending_experiences = []
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


