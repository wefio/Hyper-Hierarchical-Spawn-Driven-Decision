"""CLI entry point and connection test."""
from __future__ import annotations
import sys
import os
import io
import argparse
import time
import random

import anthropic
import httpx

from config import Config
from skill import SkillManager
from agent import Agent

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
    parser.add_argument("--no-plan", action="store_true", help="Skip planning phase (save API call)")
    args = parser.parse_args()

    # 清除代理环境变量 — 防止内网 API 调用被路由到代理服务器
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                "ALL_PROXY", "all_proxy", "no_proxy", "NO_PROXY"):
        os.environ.pop(key, None)

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
        agent.run(task, args.steps, skip_plan=args.no_plan)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    import sys as _sys
    main(_sys.argv[1] if len(_sys.argv) > 1 else "hi")
