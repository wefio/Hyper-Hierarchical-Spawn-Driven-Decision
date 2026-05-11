"""Chainlit frontend for HHSDD Agent — thinking Steps + streaming text."""
import chainlit as cl
import json as _json
import sys, os, tempfile, time as _time, asyncio, queue

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    )

    # ── Streaming message ──
    msg_obj = cl.Message(content="")
    await msg_obj.send()

    text_queue = queue.Queue()
    active = [True]

    def on_text(chunk):
        text_queue.put(chunk)

    async def reader():
        while active[0] or not text_queue.empty():
            try:
                chunk = text_queue.get(timeout=0.3)
                await msg_obj.stream_token(chunk)
            except queue.Empty:
                continue

    reader_task = asyncio.create_task(reader())

    # ── Capture thinking + tools ──
    thinks, timeline, tool_elements = [], [], []
    orig = agent._api.call

    def wrap_api(*a, **kw):
        kw["text_callback"] = on_text
        t0 = _time.time(); resp = orig(*a, **kw)
        ms = int((_time.time() - t0) * 1000)
        if hasattr(resp, "content"):
            for b in resp.content:
                if getattr(b, "type", "") == "thinking" and getattr(b, "thinking", ""):
                    thinks.append({"text": b.thinking, "ms": ms})
        return resp
    agent._api.call = wrap_api

    from agent_process_executor import ToolExecutor as _TE
    _orig_exec = _TE.execute
    def _wrap_exec(name, args, budget, agent_context):
        timeline.append(("tool", f"- `{name}`"))
        result = _orig_exec(name, args, budget, agent_context)
        inp = _json.dumps(args, ensure_ascii=False)[:500]
        out = str(result.get("output", ""))
        tool_elements.append(cl.Text(
            name=f"{name} ({'OK' if result.get('success') else 'FAIL'})",
            content=f"Input:\n```json\n{inp}\n```\nOutput:\n```\n{out}\n```",
            display="side",
        ))
        return result
    _TE.execute = staticmethod(_wrap_exec)

    try:
        r = await cl.make_async(agent.run)(t, max_steps=20)
    finally:
        _TE.execute = _orig_exec
        active[0] = False
        await reader_task

    # ── Thinking as collapsible Steps ──
    for th in thinks:
        async with cl.Step(name=f"Thought for {th['ms']/1000:.1f}s") as step:
            await step.stream_token(th["text"])

    # ── Finalize with tool summary ──
    if timeline:
        await msg_obj.stream_token("\n\n" + "\n".join(e[1] for e in timeline))
    msg_obj.elements = tool_elements if tool_elements else None
    await msg_obj.update()

    # ── Sidebar: flowchart + tools ──
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
