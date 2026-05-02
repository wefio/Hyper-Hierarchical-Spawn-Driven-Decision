"""Example Plugin for HHSDD Agent.

A minimal plugin demonstrating the plugin API.
Plugins are Python modules discovered via Config.PLUGINS (comma-separated module names).

Usage:
    PLUGINS=example_plugin python agent.py --task "..."
"""

# ---- 1. Tool Definitions (Anthropic tool_use schema) ----

def get_tool_definitions() -> list:
    """Return tool schemas to register with the Agent."""
    return [
        {
            "name": "example_hello",
            "description": "Say hello. A minimal example tool.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name to greet",
                    }
                },
                "required": ["name"],
            },
        },
    ]


# ---- 2. Tool Handlers ----

def _handle_hello(agent, params: dict) -> str:
    """Handler for the example_hello tool."""
    name = params.get("name", "world")
    return f"Hello, {name}! This is a response from example_plugin."


def get_tool_handlers() -> dict:
    """Return {tool_name: handler_fn} mapping.

    Handler signature: (agent, params: dict) -> str
    - agent: the Agent instance (read-only access to state)
    - params: parsed tool input parameters
    - returns: string result sent back to the LLM
    """
    return {
        "example_hello": _handle_hello,
    }


# ---- 3. Cache Rules ----

def get_cache_rules() -> set:
    """Return tool name prefixes that should get cache_control markers.

    Tools whose names start with any returned prefix will have
    ephemeral cache_control added to their definition.
    """
    return set()  # No special caching for this example


# ---- 4. Platform Prompt Fragments ----

def get_platform_prompt_fragments() -> list[str]:
    """Return extra text appended to the system prompt."""
    return [
        "\n\n## Example Plugin\nYou have an `example_hello` tool available for testing.\n",
    ]


# ---- 5. Lifecycle Hooks (all optional) ----

def on_agent_init(agent, bus=None):
    """Called once during Agent.__init__. Use for event subscriptions, verifier registration, etc."""
    print("  [example_plugin] initialized")


def on_run_start(agent, task: str):
    """Called at the start of Agent.run()."""
    print(f"  [example_plugin] run started: {task[:50]}")


def on_tool_executed(agent, tool_name: str, result: str):
    """Called after every tool execution. Use for post-processing, energy adjustments, etc."""
    pass


def on_subtask_loop(agent, sub_idx: int):
    """Called at the start of each subtask loop iteration. Use for feedback injection, etc."""
    pass
