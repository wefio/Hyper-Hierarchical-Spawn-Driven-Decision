# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install anthropic          # only dependency
python main.py --task "..."   # run with a task
python main.py -i --task "..."  # interactive mode
python main.py --resume       # resume from agent_state/state.json
PLUGINS=example_plugin python main.py --task "..."  # load plugins
```

No test suite, linter, or type checker.

## Conceptual architecture

整个系统的核心资源是 **LLM 上下文窗口（context window）**。所有设计围绕"在有限 context 内完成复杂任务"展开，大量借用了 OS 内存管理概念。

| 概念 | 对应 OS 概念 | 在项目中 |
|------|------------|---------|
| Context 窗口 | RAM | `CONTEXT_BUDGET`（默认 204800 tokens） |
| StackFrame 栈 | 内存分段+页表 | 层级结构：constraint→plan→exec→merge |
| use_count / last_used_step | 页表访问位 | LRU-K 驱逐决策 |
| PointerStore | Disk swap | `archive/*.md`，栈内留 stub `摘要:ptr_id` |
| recall 工具 | Page fault | 从磁盘取回内容，注入 reclaimable 帧 |
| 子 Agent 前缀继承 | CoW / 共享页表 | 缓存命中，递归分解可行 |
| L2Cache | CPU L2 缓存 | 同级 agent 共享指针列表，内存中，零信任签名 |
| Savepoint | PDA 栈快照 | 回退状态，保留推导历史 |
| Energy | 资源配额 + RL reward | 阻止无限递归，奖励成功 |

## 代码文件

```
agent.py                    # Agent 主类（进程本体）
main.py                     # CLI 入口
config.py                   # Config 单例 + _PLUGINS 加载 + smart_truncate()
skill.py                    # Skill + SkillManager

agent_memory_frame.py       # StackFrame, SavepointMeta (基础类型)
agent_memory_l2.py          # L2Cache — 同级共享内存缓存 (CPU L2 类比)
agent_memory_report.py      # NodeReport — 节点执行报告 (引擎自动收集机械字段)

agent_process_executor.py   # ToolExecutor + TOOL_DEFINITIONS + energy_hooks

agent_scheduler_energy.py   # BayesianEnergyManager + StepEstimator

agent_kernel_glue.py        # APIClient + CacheProvider + FlowchartRecorder +
                            #   EventBus + SavepointManager + PlanContext + SubAgentEvent
agent_kernel_router.py      # ModelRouter — 异构模型路由 + 并发控制
agent_kernel_verify.py      # TaskVerifier (原 task_verifier.py)

opcode_parser.py            # @opcode 精简工具调用解析器
agent_template.py           # 分形模板引擎 — YAML 模板 + 四级指针复用 + 三种分叉
pointer_store.py            # 上下文磁盘归档 + 多级页表 + 全局指针库

experience_store.py         # SQLite + FTS5 经验留存
agent_template.py           # 分形模板引擎 — YAML 模板 + 四级指针复用 + 三种分叉
pointer_store.py            # 上下文磁盘归档 + 多级页表 + 全局指针库
plugins/                    # importlib 动态加载的插件模块
```

## Core loop

1. `main()` → `Agent()` → `agent.run(task)`
2. `Agent.__init__()` 构建初始栈：constraint（system prompt + skills）+ plan（占位），连接所有子系统
3. 子 Agent 和父 Agent 是**同一个类、同一种执行循环**——Spawn 只是 `ToolExecutor` 中的一个工具处理器，没有特殊通道
4. 子 Agent 共享父级的 `cache_provider`、`energy_manager`、`flowchart`、`pointer_store`

`Agent.run()`（line 3172）的流程：
1. 注入相关经验（`_inject_experience`）→ 可选 plan 生成（A+B 合并到第一步）→ 循环执行 subtask_queue
2. plan 解析后 `_merge_subtasks()` 按 Jaccard 相似度去重——核心目的是防止子 Agent 无限外包（阻断重复子任务链），是能量系统之外的又一道防线
3. 每个 subtask 调用 `execute_next_step()` → `_build_messages()` 栈→API 消息 → 工具调用循环
4. 每步后：`_adapt_context()` 维持 context 在预算内 → `_maintain_pointer_table()` 合并指针
5. 完成后：merge → 紧急交付 → 终端奖励 → 经验收集 → flush

## Context 栈

`StackFrame` dataclass（line 50）：`type`, `content`, `step_id`, `level`, `agent_id`, `pointer_id`, `reclaimable`, `use_count`, `last_used_step`。

```
level 0: constraint   — system prompt + skills（永久，缓存前缀）
level 1: plan         — 任务计划（生成后极少修改）
level 2: step_detail / summary / pointer — 执行轨迹（动态压缩/归档）
level 3: merge        — 阶段汇合
```

`_build_messages()`（line 2561）将栈帧按类型组装为 Anthropic API 消息对（user/assistant），constraint 作为 system_blocks 发送。pointer 帧显示摘要 + token 大小提示，引导 LLM 使用 recall 工具。

`_adapt_context()`（line 2379）每步后三阶段维持 context：
1. 回收 reclaimable 帧（recall 借阅的内容，数据已在磁盘）
2. 压缩 step_detail → summary（先 `pointer_store.store()` 归档，再 LLM 压缩为 ≤100 字）
3. 截断为 pointer stub（STORE 到磁盘，栈内替换为 `摘要:ptr_id`）

`_reclaim_energy()`（line 2097）当能量不足以支付下次 API 调用时，五阶段弹栈回收：
0. 驱逐不活跃存档点历史 1. reclaimable 帧 2. 空壳（≤50 字符）3. summary/merge（按 LRU-K）4. step_detail

## PointerStore — 上下文磁盘缓存 + 多级页表（`pointer_store.py`）

三层：`PointerEntry`（元数据，含复用追踪字段）→ `PointerIndex`（内存索引/页表）→ `PointerStore`（门面）+ `GlobalPointerStore`（全局指针库）

`PointerEntry` 新增字段：`input_hash`, `template_id`, `freshness_days`, `verification`, `hits`, `reuses`, `last_hit_at`, `last_reused_at`, `hit_rate`（property）。

- `store()` / `recall()` / `merge_pointers()` / `search_keywords()` — 原有磁盘缓存
- `register_page_table()` / `add_to_page_table()` / `lookup()` — 多级页表（L0=原文, L1=input_hash组, L2=模板索引, L3=全局）
- `find_reusable()` — 四级查找：L1 精确复用 → L2 同模板参考 → L3 全局经验 → L4 FTS5
- `validate_entry()` / `record_hit()` / `evict_cold()` — 有效性 + 命中率驱动淘汰
- `GlobalPointerStore` — 跨模板全局搜索 + NodeReport 自动注册

## Energy — 上下文约束（`BayesianEnergyManager`, line 1447）

主要目的：**防止无限 spawn 递归**，同时提供 RL 式的奖励信号。

双层计数器：
- `energy`（软，流动资金）：可预扣、返还、奖励，允许透支至 -10%
- `total_spent`（硬，不可逆累加）：触顶即停

所有操作消耗以 token 计量（1:1）。所有 token 等权 1:1。Energy 计的是 context 窗口占用，不做费率区分。

贝叶斯后验：Beta(α,β) 估计任务完成/子任务成功/spawn 成功率；Gamma 估计命令耗时。`should_stop()` 检查：硬预算耗尽、流动资金耗尽、P(done)>0.9、5 轮无进展、5 次失败。`should_stop_with_estimator()` 额外前瞻剩余步数所需能量。

Spawn 投资-收益：explore 成功→本金+50% ROI，失败血本无归；exploit 按规则正常扣费。`get_role_probability()` 在高能量低完成率时倾向 explore。

## Spawn — 递归上下文分解（`_spawn_agent`, line 1034）

Spawn 是内置工具之一，不是特殊机制。每个 Agent（父/子）结构完全相同。

- 子 Agent `Agent(depth+1, parent=self)`，共享 `cache_provider`、`energy_manager`、`flowchart`、`pointer_store`
- 上下文继承：父的 system prompt + skills + shared files + failure experience 作为子栈帧注入，最大化缓存命中
- `_spawn_deduct`：预留父级 15% 能量作为 `_reserve_stack` 保底，子运行于剩余 pool；返回后 `_spawn_settle` 释放
- `_absorb_child_result()`：能量充足+内容<2000 字→完整保留，否则仅 `_make_summary()` 摘要
- `_summarize_via_child()`：保底措施——创建 `max_tool_rounds=0` 的无工具子 Agent 专门压缩全量 context
- 子 Agent 的 `_pending_experiences` 上提到父，根 Agent 统一 flush
- 同步阻塞，`MAX_SPAWN_DEPTH` 硬上限（默认 0=不限）
- 深度由能量预算自然约束（指数衰减）

## Savepoint — PDA 栈快照（`SavepointManager`, line 1145）

支持同 Agent 内部迭代探索——基于下推自动机理念（保留所有过程，回退状态不丢推导）。

- `create`：快照 stack + step_counter + subtask_queue + _verify_feedback 到磁盘
- `pop`：恢复快照状态（`total_spent` 不回退，API 已消耗），设 `_just_popped=True` 中断当前 tool loop
- `commit`：标记完成，结论入 history
- 同一时刻只有一个活跃存档点
- 快照不包含 `_pending_experiences` 和 `_current_tool_calls`——探索期间收集的经验不回退

## Experience + Skill — 上下文复用（`experience_store.py`）

- **experiences 表**（+FTS5）：全文检索，中文拆单字 + 英文单词。排序 `weight × -rank`，成功权重 ×1.2 / 失败 ×0.8 / 全局日衰减 ×0.99（原蚁群信息素设计）
- **skills 表**（+FTS5）：`extract_skill()` 自动从成功经验挖掘重复 tool+action 序列；`_seed_builtin_skills()` 从外部 `so100_tools.PROCEDURES` 导入
- `_inject_experience()`：在 `run()` 启动时将经验+技能注入为 level 0 帧（仅 root agent 执行一次）
- 轻量收集→`pending_exp.json`→累积≥3 条→批量 LLM 提取教训→写入 SQLite

## Built-in tools（`ToolExecutor`, line 714）

| 工具 | 特点 |
|------|------|
| `read_file` | smart_truncate 截断 |
| `list_dir` | 显示文件大小 |
| `write_file` | .py/.sh/.bat/.ps1 自动路由到 scripts/ 目录 |
| `view_image` | 调 MiniMax VLM API（非 Anthropic） |
| `spawn_agent` | 递归子 Agent，`@energy_hooks(deduct=_spawn_deduct, settle=_spawn_settle)` |
| `run_command` | Gamma 预估耗时→预扣→退还差额 |
| `use_skill` | 预扣押金，成功返还；实际工作流在外部插件项目中 |
| `recall` | 按 ptr_id 或关键词从 PointerStore 取回，注入 reclaimable 帧 |
| `savepoint` | create/commit/pop/list |
| `read_peer` | L2 同级共享缓存读取 |
| `ipython` | Python 沙盒执行（isolated/session/close） |
| `run_script` | 通用脚本执行：内联内容/已有文件 + 后台/定时调度 + 状态轮询 |

`@energy_hooks(deduct, settle)` 装饰器：deduct 在 handler 前（返回 False 则中止），settle 在 finally 中。

## 补充型工具定义 — Tier 分级（`_tier_tool_def`）

`Config.COMPACT_OPCODES=True` 启用。每个工具初始 Tier 1（最小定义），失败或模型请求时升级。

- Tier 1: name + 参数名+type + `_tier` 隐藏参数（~55-75% 压缩）
- Tier 2: + 首句描述
- Tier 3: 完整定义

升级机制：
- 主动：模型传 `_tier=3` → 立即升级，返回完整定义作为工具结果
- 被动：工具执行失败 → 自动 `cur+1`

`agent.py` 中 `_tool_tiers: dict[str, int]` 存储状态，`_build_messages()` 按 tier 组装工具列表。

## OPCode 精简工具调用（`opcode_parser.py`）

模型可在文本中用 `@opcode` 格式直接调用工具，替代 JSON tool_use 协议。两种触发场景：

1. `stop_reason == "tool_use"`：扫描文本中的 @opcode，与 tool_use 块合并执行
2. `stop_reason == "end_turn"`：检测到 @opcode 后执行工具，结果追加到 messages，继续循环

格式：`@opcode arg1 [arg2] [key=value ...]`

| opcode | 工具 | 示例 |
|--------|------|------|
| `@r` | read_file | `@r src/main.py` |
| `@ls` | list_dir | `@ls src/ depth=2` |
| `@w` | write_file | `@w out.txt "content"` |
| `@c` | run_command | `@c echo hello` |
| `@s` | spawn_agent | `@s "task" ctx=brief` |
| `@sp` | savepoint | `@sp create name=cp1` |
| `@recall` | recall | `@recall ptr123` |
| `@vi` | view_image | `@vi img.png prompt=OCR` |

最后一个位置参数消费所有剩余 token（如 `@c echo hello world` → `command="echo hello world"`）。

## Task verification — 双重验证（`task_verifier.py`）

`_system_verify()` 按 `_current_tool_calls` 中的工具类别分派：file→文件存在+非空，command→退出码，插件注册→领域验证器，default→中英文关键词。Agent 自评（DONE 声明）始终决定是否推进，系统反馈作为警告注入。

## Plugin system

`importlib.import_module()` 从 `Config.PLUGINS` 加载。接口：`get_tool_definitions()`, `get_tool_handlers()`, `get_cache_rules()`, `get_platform_prompt_fragments()`, `on_agent_init()`, `on_run_start()`, `on_tool_executed()`, `on_subtask_loop()`, `check_procedure_fallback()`, `inject_procedure_template()`。

## 代码约定

- 中文注释、完成检测（`"任务完成"`）、平台提示
- `agent_context` 是 grab-bag dict（`depth`, `system_prompt`, `parent_agent`），通过 `parent_agent` 引用可触达所有子系统
- `.env` 在 `config.py` import 时加载（模块级 `Config.load_env()`）
- `FlowchartRecorder` 实时追加写入 `flowchart.md`，纯给人 debug 和看进度用
- `_just_popped` 标志：savepoint pop 后跳过当前 step 的栈写入和 context 适配
- `CacheProvider` 抽象：`AnthropicCacheProvider` 用 `cache_control: {"type": "ephemeral"}`，三级缓存断点
- A+B Plan 合并：plan prompt 嵌入第一步 user 消息，一次 API 返回 plan+工具调用，省一次 API
- 紧急交付：根 Agent 部分完成时 `bypass_energy=True` 最后调用一次 LLM 整合产物
- `_current_tool_calls` 中的 IK fallback 检测（`ik_fallback` 字段）是 SO100 机械臂项目的残留，终端奖励中检测到 IK fallback 会降级为失败
