# HHSDD Plugins

This directory contains HHSDD agent plugins.

## Available Plugins

| Plugin | Description |
|--------|-------------|
| `example` | Minimal example demonstrating the plugin API |

## Creating a Plugin

See [example/README.md](example/README.md) for the full API reference.

Quick start: create a `.py` file with `get_tool_definitions()` and `get_tool_handlers()`, then set `PLUGINS=your_module_name`.

## External Plugins

Plugins outside this repo (like `so100_plugin`) can be loaded by ensuring the module is on `PYTHONPATH` and setting `PLUGINS=so100_plugin`.
