"""Chainlit frontend for HHSDD Agent."""
import chainlit as cl
import sys, os, tempfile

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
    ws = os.path.join(tempfile.gettempdir(), "hhsdd_workspace")
    os.makedirs(ws, exist_ok=True)
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
            "输出格式：",
            "思考和正文用 Markdown 格式输出",
            "思考过程：用 > 引用块，每个独立步骤前加 - ",
            "工具返回结果：用 ``` 代码块包裹",
            "任务结束：末尾单独一行 [DONE]",
            "",
            "工具并行：",
            "互不依赖的工具可一起发起",
            "",
            "上下文：",
            "用 recall 取回历史指针内容",
            "用 read_peer 查看同级 Agent 进度",
            "用 spawn_agent 拆分复杂任务",
        ]),
        depth=0, model_spec={"id": mid}, work_dir=ws)

    # Capture timeline: thinking blocks + tool calls interleaved
    import time as _time
    timeline = []
    think_msgs = []
    orig = agent._api.call
    def wrap(*a, **kw):
        t0 = _time.time(); r = orig(*a, **kw)
        ms = int((_time.time()-t0)*1000)
        if hasattr(r,'content'):
            for b in r.content:
                if getattr(b,'type','')=='thinking' and getattr(b,'thinking',''):
                    think_text = b.thinking
                    timeline.append(('think', f"**已思考（{ms/1000:.1f}s）**\n> " + think_text.replace('\n','\n> ')))
                    # Also send as separate message for collapsible effect
                    think_msgs.append((ms, think_text))
        return r
    agent._api.call = wrap

    # Intercept tool execution: inline timeline + sidebar details
    from agent_process_executor import ToolExecutor as _TE
    import json as _json
    _orig_exec = _TE.execute
    tool_elements = []
    def _wrap_exec(name, args, budget, agent_context):
        timeline.append(('tool', f"- `{name}`"))
        result = _orig_exec(name, args, budget, agent_context)
        inp = _json.dumps(args, ensure_ascii=False)[:500]
        out = str(result.get('output', ''))
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

    parts = [e[1] for e in timeline] + [r or "(no response)"]
    await cl.Message(content="\n\n".join(parts), elements=tool_elements if tool_elements else None).send()
