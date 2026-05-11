"""Kernel glue — API client, cache, monitoring, events, checkpoints, plan state machine."""
from __future__ import annotations
import json
import os
import time
import random
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, Any

import anthropic

from config import Config
from agent_memory_frame import StackFrame, SavepointMeta

# ============ 缓存抽象 ============
class APIClient:
    """Anthropic API 封装：限流、重试、计能、调试转储"""

    def __init__(self, client: anthropic.Anthropic, config: type,
                 cache_provider: CacheProvider, energy_mgr,
                 model_id: str = "default",
                 model_spec = None):
        self.client = client
        self.cfg = config
        self.cache = cache_provider
        self.em = energy_mgr
        self._last_call_time = 0.0
        self._debug_dir = Path(config.DEBUG_DIR)
        self._model_id = model_id
        self._model_name = model_spec.name if model_spec else config.MODEL
        self.last_call_meta: dict = {}
        # 厂商适配器
        if model_spec:
            from agent_kernel_router import get_adapter
            self._adapter = get_adapter(model_spec.provider)
        else:
            from agent_kernel_router import ProviderAdapter
            self._adapter = ProviderAdapter()
        self._model_spec = model_spec

    def _record_usage(self, result, bypass_energy: bool = False, total_ms: float = 0):
        raw_meta = {}
        raw_meta = self._adapter.parse_usage(result, raw_meta)
        inp = raw_meta.get("input_tokens", 0)
        out = raw_meta.get("output_tokens", 0)
        self.last_call_meta = {
            "model": getattr(result, 'model', self._model_name),
            "total_ms": total_ms,
            "input_tokens": inp, "output_tokens": out,
            "cache_read_tokens": raw_meta.get("cache_read_tokens", 0),
            "cache_write_tokens": raw_meta.get("cache_write_tokens", 0),
        }
        if not bypass_energy and hasattr(result, 'usage') and self.em:
            cost = (inp + out) * Config.TOKEN_COST_INPUT
            self.em.charge(cost, check=False)
            self.em.spend(cost)
            self.em.add_tokens(inp, out)
            self.em.track_input(inp)
            if raw_meta.get("cache_read_tokens", 0) > 0:
                print(f"  [Cache] read {raw_meta['cache_read_tokens']} tokens")

    def release(self):
        if self._model_id and self._model_id.startswith("human"):
            return
        try:
            from agent_kernel_router import get_router
            get_router().release(self._model_id)
        except Exception:
            pass

    # ── 人类模型交互 ──

    def _call_human(self, messages: list, system=None, tools=None):
        """阻塞等待人类输入，返回模拟的 API Message。"""
        import sys
        from types import SimpleNamespace

        # 预算耗尽：自动通过
        left = getattr(self, '_interventions_left', 1)
        if left <= 0:
            mock_text = SimpleNamespace(type="text", text="[人类预算耗尽，自动通过]")
            mock_msg = SimpleNamespace(
                content=[mock_text], stop_reason="end_turn", model="human",
                usage=SimpleNamespace(input_tokens=0, output_tokens=0),
            )
            self.last_call_meta = {"model": "human", "total_ms": 0,
                "input_tokens": 0, "output_tokens": 0,
                "cache_read_tokens": 0, "cache_write_tokens": 0}
            print("\n! 干预预算耗尽，自动通过")
            return mock_msg

        # ── 构建结构化摘要 ──
        sys_text = ""
        if system:
            s = system[0] if isinstance(system, list) and system else system
            sys_text = s.get("text", str(s)) if isinstance(s, dict) else str(s)

        # 提取消息中的关键信息
        user_msgs = []
        assistant_summaries = []
        for m in messages:
            role = m.get("role", "")
            c = m.get("content", "")
            text = ""
            if isinstance(c, str):
                text = c
            elif isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text":
                        text += b.get("text", "") + " "
            if role == "user":
                user_msgs.append(text[:300].strip())
            elif role == "assistant":
                # 提取工具调用
                tool_calls = []
                if isinstance(c, list):
                    for b in c:
                        if isinstance(b, dict) and b.get("type") == "tool_use":
                            tool_calls.append(b.get("name", "?"))
                        elif hasattr(b, 'type') and b.type == "tool_use":
                            tool_calls.append(b.name)
                summary = text[:200].strip()
                if tool_calls:
                    summary += f" [tools: {', '.join(tool_calls[:5])}]"
                if summary:
                    assistant_summaries.append(summary)

        # 工具列表
        tool_names = [t.get("name", "?") for t in (tools or [])]

        print("\n" + "=" * 60)
        print("[Human Review]")
        print("=" * 60)
        if sys_text:
            print(f"角色: {sys_text[:120]}")
        print(f"干预剩余: {getattr(self, '_interventions_left', '?')} 次")

        # 任务列表
        print(f"\n[Context] 上下文 ({len(messages)} 条消息):")
        for i, um in enumerate(user_msgs[-3:]):
            print(f"  [{i+1}] {um[:200]}")
        if assistant_summaries:
            recent = assistant_summaries[-3:]
            print(f"\n[Steps] 已完成步骤 ({len(assistant_summaries)} 步):")
            for i, s in enumerate(recent):
                print(f"  {'OK' if i < len(recent)-1 else '>'} {s[:200]}")

        print(f"\n[Tools] 可用工具 ({len(tool_names)}): {', '.join(tool_names[:8])}")
        print("-" * 60)
        print("[A]pprove  [R]eject+原因  suggest:探索描述  自由输入")
        print("空行 = Approve  |  Ctrl+D = 结束")

        lines = []
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    break
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            pass

        response = "\n".join(lines) if lines else "APPROVED"

        # 干预预算递减
        interventions = getattr(self, '_interventions_left', None)
        if interventions is not None and interventions > 0:
            self._interventions_left -= 1

        # 构造模拟响应
        mock_text = SimpleNamespace(type="text", text=f"[人类响应] {response}")
        mock_msg = SimpleNamespace(
            content=[mock_text], stop_reason="end_turn", model="human",
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
        )
        self.last_call_meta = {
            "model": "human", "total_ms": 0,
            "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_write_tokens": 0,
        }
        try:
            print(f"[Human] {response[:100]}")
        except UnicodeEncodeError:
            print(f"[Human] {response[:100].encode('ascii', errors='replace').decode()}")
        return mock_msg

    def call_stream(self, messages: list, system=None, tools=None,
                    max_tokens: int = 4096, extra_params: dict = None):
        """流式调用 API，yield text_delta 字符串。"""
        if self._model_id and self._model_id.startswith("human"):
            yield "[Human model does not support streaming]"
            return

        kwargs = {"model": self._model_name, "max_tokens": max_tokens,
                  "messages": messages}
        if system: kwargs["system"] = system
        if tools: kwargs["tools"] = tools
        if self._model_spec:
            kwargs = self._adapter.build_request(kwargs, self._model_spec)
        elif extra_params:
            kwargs["extra_body"] = extra_params

        try:
            with self.client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
                final = stream.get_final_message()
                if final:
                    self._record_usage(final)
        finally:
            self.release()

    def call(self, messages: list, system=None, tools=None,
             max_tokens: int = 4096, bypass_energy: bool = False,
             extra_params: dict = None, stop_sequences: list = None,
             text_callback=None):
        """调用 API，返回 Message 对象。text_callback 可选，支持流式。"""
        # ── 人类模型：展示上下文，阻塞读 stdin ──
        if self._model_id and self._model_id.startswith("human"):
            return self._call_human(messages, system, tools)

        # 基础 kwargs
        kwargs = {"model": self._model_name, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences
        # 厂商适配器：请求转换
        if self._model_spec:
            kwargs = self._adapter.build_request(kwargs, self._model_spec)
        elif extra_params:
            kwargs["extra_body"] = extra_params

        self._dump("req", kwargs)

        elapsed = time.time() - self._last_call_time
        if elapsed < self.cfg.RATE_LIMIT:
            time.sleep(self.cfg.RATE_LIMIT - elapsed)

        max_retries = self.cfg.MAX_RETRIES
        overload_retries = 0
        max_overload = max_retries * 3

        # ── 流式模式：text_callback + stream.get_final_message() ──
        if text_callback:
            kwargs.pop("stop_sequences", None)  # streaming 不支持 stop_sequences
            with self.client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    text_callback(text)
                result = stream.get_final_message()
            if result and hasattr(result, 'usage') and self.em:
                self._record_usage(result, bypass_energy)
            self.release()
            return result

        for attempt in range(max_retries + max_overload):
            try:
                t0 = time.time()
                result = self.client.messages.create(timeout=120.0, **kwargs)
                total_ms = (time.time() - t0) * 1000
                self._last_call_time = time.time()
                self._dump("res", result)

                total_ms = (time.time() - t0) * 1000
                self._last_call_time = time.time()
                self._dump("res", result)
                self._record_usage(result, bypass_energy, total_ms)

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
        # 所有 token 等权 — energy 计的是 context 窗口占用
        return (input_tokens + output_tokens) * Config.TOKEN_COST_INPUT


class FlowchartRecorder:
    """增量式流程图记录器。写文件用队列，flush 时统一落盘。"""

    def __init__(self, work_dir: str):
        from pathlib import Path
        self.file_path = Path(work_dir) / "flowchart.md"
        self.nodes = set()
        self.edges = set()
        self.tool_calls: dict[tuple, int] = {}
        self.spawn_seq: dict[int, int] = {}
        self._lifecycle: dict[int, dict] = {}
        self._pending: list[str] = []  # 写队列
        self._init_file()

    def _init_file(self):
        try:
            self.file_path.write_text("```mermaid\ngraph TD\n", encoding='utf-8')
        except Exception:
            pass

    def _queue(self, line: str):
        """追加到写队列（GIL 保护 list append，无锁）。"""
        self._pending.append(line)

    def flush(self):
        """落盘队列中的所有行。主线程调用。"""
        if not self._pending:
            return
        lines = self._pending[:]
        self._pending.clear()
        try:
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass

    # ── 节点/边格式保持不变，只把 _append 改成 _queue ──

    @staticmethod
    def _node_line(node_id: str, label: str, shape: str) -> str:
        indent = "    "
        cls = f":::{shape}"
        if shape == "start":
            return f'{indent}{node_id}(["{label}"]):::startEnd'
        elif shape == "end":
            return f'{indent}{node_id}(["{label}"]):::startEnd'
        elif shape == "task":
            return f'{indent}{node_id}["{label}"]{cls}'
        elif shape == "subtask":
            return f'{indent}{node_id}["{label}"]{cls}'
        elif shape == "step":
            return f'{indent}{node_id}["{label}"]{cls}'
        elif shape == "tool":
            return f'{indent}{node_id}{{"{label}"}}{cls}'
        elif shape == "agent":
            return f'{indent}{node_id}{{{{"{label}"}}}}{cls}'
        elif shape == "reclaim":
            return f'{indent}{node_id}[/"{label}"/]{cls}'
        elif shape == "truncate":
            return f'{indent}{node_id}{{"{label}"}}{cls}'
        elif shape == "energy":
            return f'{indent}{node_id}("{label}"){cls}'
        elif shape == "absorb":
            return f'{indent}{node_id}[["{label}"]]{cls}'
        elif shape == "decision":
            return f'{indent}{node_id}{{{{"{label}"}}}}{cls}'
        elif shape == "pointer":
            return f'{indent}{node_id}[("{label}")]{cls}'
        elif shape == "verify":
            return f'{indent}{node_id}{{"{label}"}}{cls}'
        return f'{indent}{node_id}["{label}"]'

    def add_node(self, node_id: str, label: str, shape: str = "rect"):
        if node_id in self.nodes:
            return
        self.nodes.add(node_id)
        self._queue(self._node_line(node_id, label, shape))

    def add_edge(self, from_id: str, to_id: str, label: str = ""):
        edge = (from_id, to_id, label)
        if edge in self.edges:
            return
        self.edges.add(edge)
        indent = "    "
        line = f'{indent}{from_id} -->|{label}| {to_id}' if label else f'{indent}{from_id} --> {to_id}'
        self._queue(line)


    def merged_tool(self, from_id: str, tool_name: str, step_counter: int, depth: int = 0,
                    success: bool = True) -> str:
        key = (depth, step_counter, tool_name)
        self.tool_calls[key] = self.tool_calls.get(key, 0) + 1
        count = self.tool_calls[key]
        node_id = f"tool_d{depth}_{step_counter}_{tool_name}"
        label = f"{tool_name} (x{count}){' OK' if success else ' ✗'}"
        if count == 1:
            self.add_node(node_id, f"工具: {label}", shape="tool")
            self.add_edge(from_id, node_id)
        elif not success:
            self._queue(f"    class {node_id} toolFail")
        return node_id

    def add_note(self, node_id: str, note: str):
        """为节点添加注释"""
        self._queue(f'    note for {node_id} "{note}"')

    def update_subtask_status(self, sub_node: str, desc: str, success: bool):
        """更新子任务节点的完成状态。Mermaid 不支持更新，追加状态标注节点。"""
        if sub_node not in self.nodes:
            return
        tag = " OK" if success else " ✗"
        status_node = f"{sub_node}_status"
        # 追加一个状态节点，连到子任务节点
        self.add_node(status_node, f"{desc[:20]}{tag}", shape="subtask")
        self.add_edge(sub_node, status_node, label="结果")
        if not success:
            self._queue(f"    class {status_node} toolFail")

    def next_spawn_seq(self, depth: int) -> int:
        self.spawn_seq[depth] = self.spawn_seq.get(depth, 0) + 1
        return self.spawn_seq[depth]

    def record_lifecycle(self, depth: int, **kw):
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
        self.flush()
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
        # 样式定义
        self._queue("    classDef startEnd fill:#e1f5e1,stroke:#2e7d32,stroke-width:2px")
        self._queue("    classDef task fill:#fff3e0,stroke:#ef6c00,stroke-width:2px")
        self._queue("    classDef step fill:#e3f2fd,stroke:#1565c0,stroke-width:1px")
        self._queue("    classDef tool fill:#fce4ec,stroke:#c2185b,stroke-width:1px")
        self._queue("    classDef toolFail fill:#ffcdd2,stroke:#b71c1c,stroke-width:2px")
        self._queue("    classDef agent fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px")
        self._queue("    classDef reclaim fill:#ffebee,stroke:#c62828,stroke-width:1px,stroke-dasharray: 5 5")
        self._queue("    classDef truncate fill:#fffde7,stroke:#f9a825,stroke-width:1px,stroke-dasharray: 3 3")
        self._queue("    classDef energy fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px")
        self._queue("    classDef pointer fill:#e0f7fa,stroke:#00838f,stroke-width:1px")
        self._queue("    classDef verify fill:#fbe9e7,stroke:#d84315,stroke-width:2px")
        self._queue("    classDef decision fill:#fff8e1,stroke:#ff8f00,stroke-width:2px")
        self._queue("```")



class AgentEventBus:
    """Simple event bus for decoupling components within the agent."""

    def __init__(self):
        self._subscribers: dict[str, list[callable]] = {}

    def subscribe(self, event_type: str, handler: callable):
        self._subscribers.setdefault(event_type, []).append(handler)

    def emit(self, event_type: str, data: dict):
        for handler in self._subscribers.get(event_type, []):
            try:
                handler(data)
            except Exception as e:
                print(f"[EventBus] Handler error for '{event_type}': {e}")


# Shared event bus — initialized in Agent.__init__
_EVENT_BUS: AgentEventBus | None = None


def _get_event_bus() -> AgentEventBus:
    global _EVENT_BUS
    if _EVENT_BUS is None:
        _EVENT_BUS = AgentEventBus()
    return _EVENT_BUS



class SavepointManager:
    """存档点管理器 — 支持 agent 内部迭代探索。

    每个 agent 只有一个活跃存档点。
    历史存档点按 context_size 排序，驱逐时优先回收最大的。

    create → 快照当前状态到磁盘，扣除备份能量
    commit → 保留结论摘要，存档点进入历史
    pop → 从磁盘恢复快照，返还探索消耗的能量
    """

    @staticmethod
    def _dir(agent) -> str:
        d = os.path.join(Config.SAVEPOINT_DIR, agent.agent_id)
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def create(agent, name: str = None) -> dict:
        if agent.active_savepoint is not None:
            return {"success": False, "output": f"已有活跃存档点 '{agent.active_savepoint.name}'，请先 commit 或 pop"}

        name = name or f"sp_{agent.step_counter}_{int(time.time()) % 10000}"
        save_dir = SavepointManager._dir(agent)
        path = os.path.join(save_dir, f"{name}.json")

        # 全量快照
        snapshot = {
            "stack": [asdict(f) for f in agent.stack],
            "step_counter": agent.step_counter,
            "subtask_queue": agent.subtask_queue,
            "conversation_history": agent.conversation_history,
            "_verify_feedback": list(agent._verify_feedback),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)

        # 估算当前上下文大小
        ctx_chars = sum(len(fr.content) for fr in agent.stack)
        ctx_tokens_est = ctx_chars // 4

        sp = SavepointMeta(
            name=name, path=path, agent_id=agent.agent_id,
            status="active",
            energy_at_save=agent.energy_manager.energy,
            total_spent_at_save=agent.energy_manager.total_spent,
            step_counter_at_save=agent.step_counter,
            context_chars_at_save=ctx_chars,
            context_size=ctx_tokens_est,
            created_at=time.time(),
        )
        agent.active_savepoint = sp

        # 扣除备份能量（快照占用 context 的估算成本）
        backup_cost = ctx_tokens_est * Config.TOKEN_COST_INPUT * Config.SAVEPOINT_BACKUP_COST
        agent.energy_manager.charge(backup_cost, check=False)
        agent.energy_manager.spend(backup_cost)

        print(f"  [Savepoint] Created '{name}' at step {agent.step_counter}, "
              f"energy={sp.energy_at_save:.0f}E, ctx≈{ctx_tokens_est} tokens, "
              f"backup_cost={backup_cost:.0f}E")
        return {"success": True, "output": f"Savepoint '{name}' created (step {agent.step_counter}, "
                                          f"energy {sp.energy_at_save:.0f}E, "
                                          f"ctx ~{ctx_tokens_est} tokens)"}

    @staticmethod
    def commit(agent, summary: str = "") -> dict:
        sp = agent.active_savepoint
        if sp is None:
            return {"success": False, "output": "No active savepoint to commit"}

        sp.status = "committed"
        sp.summary = summary or "(no summary)"

        # 结论摘要入栈（极小上下文开销）
        conclusion = f"[Savepoint '{sp.name}' committed] {sp.summary}"
        agent.stack.append(StackFrame("summary", conclusion,
                                       agent.step_counter, level=2,
                                       agent_id=agent.agent_id))

        # 移入历史（不活跃）
        agent.savepoint_history.append(sp)
        agent.active_savepoint = None

        print(f"  [Savepoint] Committed '{sp.name}': {sp.summary[:80]}")
        return {"success": True, "output": f"Savepoint '{sp.name}' committed. Summary: {sp.summary[:200]}"}

    @staticmethod
    def pop(agent, reason: str = "") -> dict:
        sp = agent.active_savepoint
        if sp is None:
            return {"success": False, "output": "No active savepoint to pop"}

        # 从磁盘读取快照
        try:
            with open(sp.path, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
        except Exception as e:
            return {"success": False, "output": f"Failed to load snapshot: {e}"}

        # 计算返还能量
        energy_before_pop = agent.energy_manager.energy
        energy_spent_during = sp.energy_at_save - energy_before_pop
        refund_amount = max(0, energy_spent_during)

        total_spent_during = agent.energy_manager.total_spent - sp.total_spent_at_save

        # 恢复状态
        agent.stack = deque([StackFrame(**frame) for frame in snapshot["stack"]])
        agent.step_counter = snapshot["step_counter"]
        agent.subtask_queue = snapshot["subtask_queue"]
        agent.conversation_history = snapshot["conversation_history"]
        agent._verify_feedback = snapshot.get("_verify_feedback", [])

        # 返还探索消耗的能量（context 已释放）
        agent.energy_manager.energy = sp.energy_at_save
        # total_spent 不回调（API 已实际消耗），但流动资金恢复

        # 标记存档点被弹出，移入历史
        sp.status = "popped"
        sp.summary = reason or "(no reason)"
        sp.context_size = 0  # 几乎不占 context（只有 reason 字符串）
        agent.savepoint_history.append(sp)
        agent.active_savepoint = None

        # 通知 tool loop 停止（状态已恢复，下一步直接使用新栈）
        agent._just_popped = True

        print(f"  [Savepoint] POPPED '{sp.name}': {reason[:80] if reason else 'no reason'}. "
              f"Refunded {refund_amount:.0f}E (spent {energy_spent_during:.0f}E during exploration, "
              f"total_spent={total_spent_during:.0f}E consumed)")
        return {"success": True,
                "output": f"Savepoint '{sp.name}' popped. Energy refunded: {refund_amount:.0f}E. "
                          f"Reason: {sp.summary[:200]}"}

    @staticmethod
    def list_savepoints(agent) -> dict:
        lines = []
        if agent.active_savepoint:
            sp = agent.active_savepoint
            lines.append(f"[ACTIVE] {sp.name} (step {sp.step_counter_at_save}, "
                        f"energy_at_save={sp.energy_at_save:.0f}E, "
                        f"ctx≈{sp.context_chars_at_save} chars)")
        if agent.savepoint_history:
            # 按 context_size 降序列出
            sorted_hist = sorted(agent.savepoint_history,
                                key=lambda s: s.context_size, reverse=True)
            lines.append(f"--- History ({len(sorted_hist)}) ---")
            for sp in sorted_hist:
                lines.append(f"  [{sp.status.upper()}] {sp.name}: {sp.summary[:60]}")
        if not lines:
            lines.append("(no savepoints)")
        return {"success": True, "output": "\n".join(lines)}


from skill import Skill, SkillManager


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

# ============ Plan 状态机 ============
class PlanState(Enum):
    IDLE = "idle"               # 无 plan 或已消费
    PENDING = "pending"         # plan prompt 已嵌入首步，等待 API 响应
    PARSED = "parsed"           # plan 文本已提取，等待 _finalize_plan 消费
    NEEDS_REPLAY = "needs_replay"  # plan 已解析但无工具调用，需重放首个子任务

class PlanContext:
    __slots__ = ('state', 'prompt', 'reasoning')
    def __init__(self):
        self.state = PlanState.IDLE
        self.prompt = ""
        self.reasoning = ""

