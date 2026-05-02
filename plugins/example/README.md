# HHSDD Plugin API

Plugins extend the HHSDD Agent with domain-specific tools and behaviors.

## Quick Start

```python
# my_plugin.py — place it in the same directory as agent.py

def get_tool_definitions():
    return [{"name": "my_tool", "description": "...", "input_schema": {...}}]

def get_tool_handlers():
    return {"my_tool": lambda agent, params: "result"}

def get_cache_rules():
    return {"my_"}  # prefix match on tool names

def get_platform_prompt_fragments():
    return ["\nExtra instructions for the LLM.\n"]
```

Enable it:
```bash
PLUGINS=my_plugin python agent.py --task "..."
```

## API Reference

A plugin is any Python module that implements some or all of these functions:

### Registration (called once at startup)

| Function | Return | Purpose |
|----------|--------|---------|
| `get_tool_definitions()` | `list[dict]` | Anthropic tool_use schemas |
| `get_tool_handlers()` | `dict[str, callable]` | `{tool_name: (agent, params) -> str}` |
| `get_cache_rules()` | `set[str]` | Tool name prefixes for cache_control |
| `get_platform_prompt_fragments()` | `list[str]` | Extra system prompt text |

### Lifecycle Hooks (all optional)

| Hook | When | Typical Use |
|------|------|-------------|
| `on_agent_init(agent, bus=None)` | `Agent.__init__` | Event bus subscriptions, verifier registration |
| `on_run_start(agent, task)` | `Agent.run()` start | Trace spans, initialization |
| `on_tool_executed(agent, tool_name, result)` | After every tool call | Energy settlement, post-processing |
| `on_subtask_loop(agent, sub_idx)` | Each subtask iteration | Feedback injection, state checks |

### Task Verification (via `task_verifier`)

Plugins can register domain-specific verifiers:

```python
from task_verifier import register_category, register_verifier

def on_agent_init(agent, bus=None):
    register_category("my_domain", ["my_tool_a", "my_tool_b"])
    register_verifier("my_domain", lambda agent, desc, text: {
        "verified": True,
        "reason": "looks good",
        "feedback": "",
        "severity": "pass",
    })
```

Verifiers are called after each subtask. Priority: file > command > registered domains > default (keyword match).

## Loading

Set `PLUGINS` in `.env` or environment:
```bash
PLUGINS=plugin_a,plugin_b    # comma-separated module names
PLUGINS=                      # empty = no plugins
```

Modules are loaded via `importlib.import_module()`, so they must be importable (same directory or on `PYTHONPATH`).

## Example

See `example_plugin.py` in this directory for a minimal working plugin.
