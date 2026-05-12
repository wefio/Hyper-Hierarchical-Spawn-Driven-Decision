"""Chainlit frontend for HHSDD Agent — thinking Step + streaming text + progress."""
import chainlit as cl
import json as _json
import sys, os, tempfile, time as _time, queue, asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Windows GBK 编码会导致 print 崩溃，强制 UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')
from agent import Agent
from agent_kernel_router import get_router



@cl.on_chat_start
async def start():
    r = get_router(); models = []
    for t in ["opus", "sonnet", "haiku", "fallback"]:
        for mid in r.registry.tiers.get(t, []):
            s = r.registry.get(mid)
            if s: models.append((mid, t, f"{s.context_window//1000}k"))
    ws = tempfile.mkdtemp(prefix="hhsdd_")
    cl.user_session.set("workspace", ws)
    ml = "\n".join(f"- `{mid}` [{t.upper()}] ({ctx})" for mid, t, ctx in models)
    await cl.Message(content=f"Models:\n{ml}\n\n`/model <id>` to switch").send()


@cl.on_message
async def on_message(msg: cl.Message):
    mid = cl.user_session.get("model_id", "deepseek:v4-flash")
    ws = cl.user_session.get("workspace", ".")
    t = msg.content.strip()

    if t.startswith("/model "):
        req = t[7:].strip()
        if get_router().registry.get(req):
            cl.user_session.set("model_id", req)
            await cl.Message(content=f"-> `{req}`").send()
        else:
            await cl.Message(content=f"Unknown: `{req}`").send()
        return

    # ── 监控 sink：每条消息清空后独立写入 ──
    _monitor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "public", "monitor.json")
    _monitor_state = {}
    # 清空上一条消息的残留
    try:
        with open(_monitor_path, "w", encoding="utf-8") as f:
            _json.dump({}, f)
    except Exception:
        pass

    def _monitor_sink(data):
        if "tool" in data:
            _monitor_state.setdefault("tools", []).append(data["tool"])
        _monitor_state.update({k: v for k, v in data.items() if k != "tool"})
        try:
            with open(_monitor_path, "w", encoding="utf-8") as f:
                _json.dump(_monitor_state, f)
        except Exception:
            pass

    # ── 流式文本队列（线程→异步桥接）──
    text_queue = queue.Queue()
    active = [True]

    def _text_callback(chunk):
        text_queue.put(chunk)

    agent = Agent(
        system_prompt="\n".join([
            "输出格式：", "思考和正文用 Markdown 格式输出",
            "思考过程：用 > 引用块，每个独立步骤前加 - ",
            "工具返回结果：用 ``` 代码块包裹", "任务结束：末尾单独一行 [DONE]",
            "", "工具并行：", "互不依赖的工具可一起发起",
            "", "上下文：", "用 recall 取回历史指针内容",
            "用 read_peer 查看同级 Agent 进度", "用 spawn_agent 拆分复杂任务",
        ]),
        depth=0, model_spec={"id": mid}, can_spawn=True, work_dir=ws,
        monitor_sink=_monitor_sink,
        text_callback=_text_callback,
    )

    # ── 工具调用：执行时实时推送到流式输出 + 侧边栏 ──
    tool_elements = []

    from agent_process_executor import ToolExecutor as _TE
    _orig_exec = _TE.execute
    def _wrap_exec(name, args, budget, agent_context):
        result = _orig_exec(name, args, budget, agent_context)
        inp = _json.dumps(args, ensure_ascii=False)[:500]
        out = str(result.get("output", ""))[:500]
        ok = "✅" if result.get("success") else "❌"
        _text_callback(f"> {ok} **{name}**")
        tool_elements.append(cl.Text(
            name=f"{name} {ok}",
            content=f"```json\n{inp}\n```\n→\n```\n{out}\n```",
            display="side",
        ))
        return result
    _TE.execute = staticmethod(_wrap_exec)

    # ── 流式读取器：从队列消费文本，实时更新 UI ──
    msg_obj = cl.Message(content="")
    streamed = [False]

    async def _reader():
        while active[0]:
            try:
                chunk = text_queue.get_nowait()
                if not streamed[0]:
                    streamed[0] = True
                    await msg_obj.stream_token(chunk)
                else:
                    await msg_obj.stream_token("\n\n" + chunk)
            except queue.Empty:
                await asyncio.sleep(0.1)
        # 排空剩余
        while not text_queue.empty():
            try:
                chunk = text_queue.get_nowait()
                if not streamed[0]:
                    streamed[0] = True
                    await msg_obj.stream_token(chunk)
                else:
                    await msg_obj.stream_token("\n\n" + chunk)
            except queue.Empty:
                break

    reader_task = asyncio.create_task(_reader())

    # ── 运行 Agent ──
    try:
        result = await cl.make_async(agent.run)(t, max_steps=20)
        if isinstance(result, tuple):
            final_text = result[0] if result else ""
        else:
            final_text = str(result) if result else ""
    finally:
        _TE.execute = _orig_exec
        active[0] = False
        await reader_task

    # ── 发送最终结果 ──
    if not streamed[0]:
        msg_obj.content = final_text or ""
    msg_obj.elements = tool_elements if tool_elements else []
    await msg_obj.send() if not streamed[0] else await msg_obj.update()

    # ── 侧边栏：流程图 + 工具 ──
    sidebar_el = []
    fc_path = os.path.join(ws, "flowchart.md")
    if os.path.exists(fc_path):
        fc = open(fc_path, encoding="utf-8").read()
        if len(fc) > 100:
            sidebar_el.append(cl.Text(name="Flowchart", content=fc, display="side"))
    if tool_elements:
        sidebar_el.extend(tool_elements)
    if sidebar_el:
        await cl.ElementSidebar.set_elements(sidebar_el)
        await cl.ElementSidebar.set_title("Details")
