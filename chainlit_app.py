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

    agent = Agent(system_prompt="你是AI助手。用中文。完成输出[DONE]。",
                  depth=0, model_spec={"id": mid}, work_dir=ws)
    r = await cl.make_async(agent.run)(t, max_steps=20)
    await cl.Message(content=r or "(no response)").send()
