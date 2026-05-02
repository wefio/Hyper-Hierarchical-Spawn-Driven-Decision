# HHSDD

**Hyper-Hierarchical Spawn-Driven Decision** / 超分层派生驱动决策系统

自主 AI Agent，核心设计目标：在固定的上下文窗口内，通过 token 预算硬约束管理递归深度，并基于贝叶斯后验优化执行决策。基于 Anthropic tool_use 协议，支持递归子 Agent、能量预算管理、经验留存和自动压缩。通过插件系统扩展领域工具。

## 设计约束与假设

- **API 协议**：Anthropic Messages API（支持 `tool_use` / `tool_result`）
- **运行环境**：Python 3.10+，核心依赖仅 `anthropic` SDK
- **核心假设**：Skill 文件在启动时一次性加载，运行期间不再变更；子 Agent 通过消息前缀继承父级上下文

## 快速开始

```bash
# 1. 安装依赖
pip install anthropic

# 2. 配置 API
cp .env.example .env
# 编辑 .env 填入 ANTHROPIC_API_KEY

# 3. 运行
python agent.py --task "你的任务描述"

# 交互模式
python agent.py -i --task "你的任务描述"

# 加载插件
PLUGINS=example_plugin python agent.py --task "你好"

# 指定 Skill
python agent.py --skill investigation-first --task "调研 Gazebo 仿真框架"

# 恢复上次状态
python agent.py --resume
```

## 核心机制

### 递归子 Agent（Spawn）

`spawn_agent` 是原生工具之一。子 Agent 隔离执行，返回压缩摘要，父 Agent 基于能量状态决定保留全文或仅摘要。

- **上下文继承**：子 Agent 继承父级的 `system_blocks` + 计划 + 已完成步骤，最大化 prompt cache 命中率
- **能量隔离**：每次 spawn 扣留父级当前能量的 15% 作为保底，子 Agent 在剩余 85% 内独立运行，返回后释放
- **深度限制**：由能量预算自然约束（指数衰减），可通过 `MAX_SPAWN_DEPTH` 设硬上限

### Token 预算系统

所有资源统一以 token 计量，作为硬约束引擎：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CONTEXT_BUDGET` | 204800 | 模型上下文窗口上限 |
| `CONTEXT_RESERVE` | 50000 | 保留给 system prompt、工具定义、缓存开销、紧急交付 |
| **可用预算** | **154800** | `CONTEXT_BUDGET - CONTEXT_RESERVE` |
| `STEP_OVERHEAD` | 1000 | 每步固定 token 开销 |
| `TOKEN_COST_INPUT` | 1.0 | Input token 能量系数 |
| `TOKEN_COST_OUTPUT` | 3.0 | Output token 能量系数（输出更贵） |
| `SPAWN_RESERVE_RATIO` | 0.15 | Spawn 预留比例 |

**双层约束**：
- **软约束（`energy`）**：流动资金，可预扣、返还、奖励，允许临时透支至 -10%
- **硬约束（`total_spent`）**：不可逆累加，触顶即停

**投资-收益模型**（Spawn 专用）：
- `explore` 模式：成功返还本金 + 50% ROI；失败血本无归
- `exploit` 模式：成功返还本金；失败返还 50%
- `use_skill` 押金：任务成功全额返还，失败没收

### 上下文分层栈

采用栈式结构管理消息历史，与 API 消息列表顺序天然对齐：

```
[0] constraint  ← 系统提示词 + Skills（永久保留，缓存前缀）
[1] plan        ← 任务计划（生成后极少修改）
[2] step_detail ← 详细执行轨迹（动态压缩）
[2] summary     ← 压缩后的步骤摘要
[3] merge       ← 阶段汇合总结
```

**缓存优化**：三级缓存断点结构
```
[System Blocks] → [Plan] → [Completed Steps] → [Latest User]
     断点1          断点2        (动态)            断点3
```

### 自适应压缩与回收

- **压缩触发**：上下文长度超过 `CONTEXT_BUDGET × COMPRESSION_THRESHOLD`（默认 90%）
- **压缩策略**：`step_detail → summary`，每步压缩为 100 字以内摘要，循环压缩直到低于阈值
- **能量回收**：当能量不足以支付下次 API 调用时，按优先级弹栈：空壳结果 → summary/merge → step_detail
- **极端兜底**：截断最旧的 summary 帧

### 贝叶斯决策辅助

利用共轭先验在数据稀疏时提供稳健估计：

- **任务完成概率**：`Beta(α, β)` 先验，观测后更新。用于 `should_stop` 判断
- **子任务成功率**：每个子任务独立的 `Beta` 后验，用于去重时的质量评分
- **命令耗时预估**：`Gamma(α, β)` 先验，按命令模式（程序名+子命令）聚合历史，给出期望耗时和 95% 上限
- **剩余步数预估**：`Gamma` 共轭预测剩余子任务所需步数，用于能量前瞻

### 经验留存

- **存储**：SQLite + FTS5 全文检索
- **检索**：基于中文单字 + 英文单词的 FTS5 查询，失败时回退到 LIKE
- **权重机制**：成功记录权重 ×1.2，失败 ×0.8，全局每日衰减 1%（`×0.99`）
- **检索排序**：`weight × -rank`，成功路径优先
- **写入策略**：轻量元数据即时写入；仅失败或步数≥3 的任务才调用 LLM 提取结构化教训

### Pointer 磁盘缓存

将即将被淘汰的上下文持久化到本地文件，原地替换为摘要指针。支持多级 pointer、作用域隔离、LRU-K 追踪、能量联动。

### 任务验证框架

基于工具调用链的动态验证分派：
- 文件操作 → 文件存在 + 非空检查
- 命令执行 → 退出码检查
- 领域插件 → 通过 `register_verifier()` 注册
- 默认 → 关键词信号检测（success/failure）

### Dual-Auth 双通道验证

每个子任务同时由 LLM 自评和系统验证：
- Agent 自评：LLM 判断子任务是否完成
- 系统验证：基于工具调用链的自动化验证
- 对话类任务（无工具调用）自动标记为完成

## 插件系统

HHSDD 通过插件系统扩展领域工具。插件是普通 Python 模块，实现特定接口函数即可被 Agent 加载。

```bash
PLUGINS=example_plugin    # 加载示例插件
PLUGINS=so100_plugin      # 加载 SO100 机械臂插件
PLUGINS=                  # 禁用所有插件
```

插件 API：

| 函数 | 阶段 | 用途 |
|------|------|------|
| `get_tool_definitions()` | 启动 | 工具 schema（Anthropic tool_use 格式） |
| `get_tool_handlers()` | 启动 | `{tool_name: (agent, params) -> str}` |
| `get_cache_rules()` | 启动 | 需要缓存标记的工具名前缀 |
| `get_platform_prompt_fragments()` | 启动 | 追加到系统提示的文本 |
| `on_agent_init(agent, bus)` | Agent 初始化 | 事件订阅、验证器注册 |
| `on_run_start(agent, task)` | 任务开始 | 初始化（如 trace span） |
| `on_tool_executed(agent, tool_name, result)` | 工具执行后 | 后处理、能量结算 |
| `on_subtask_loop(agent, sub_idx)` | 子任务循环 | 反馈注入、状态检查 |

详见 [plugins/example/README.md](plugins/example/README.md)。

## 配置

所有参数通过 `.env` 文件或环境变量配置。详见 `.env.example`。

关键参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `PLUGINS` | (空) | 逗号分隔的插件模块名 |
| `MAX_STEPS` | 100 | 最大步骤数 |
| `CONTEXT_BUDGET` | 204800 | 上下文窗口 |
| `TOOL_ENERGY_COST` | 200 | 单次工具调用预扣能量 |
| `AUTO_PLAN` | smart | 计划模式：always/smart/never |

## 项目结构

```
├── agent.py              # 核心：Agent / 能量 / 工具 / 栈帧 / 流程图 / 贝叶斯预估
├── config.py             # 配置（单例，解决循环导入）
├── experience_store.py   # 经验库：SQLite + FTS5 + 权重更新
├── pointer_store.py      # Pointer 磁盘缓存：多级归档 + 作用域隔离
├── task_verifier.py      # 任务验证框架：工具链分派 + 动态注册
├── plugins/              # 插件目录
│   └── example/          # 示例插件
├── skills/               # Skill 定义（Markdown + YAML frontmatter）
├── scripts/              # 脚本文件自动路由目录
├── agent_state/          # 运行时状态持久化
├── experience_store/     # 经验数据库
├── history/              # 运行历史
└── .env                  # 运行配置
```

## 限制

1. **API 协议**：依赖 Anthropic Messages API 的 `tool_use` 格式及 `cache_control` 扩展
2. **Spawn 信息损耗**：深层递归下子 Agent 返回的摘要对父级而言是有损压缩，复杂依赖链可能出现语义漂移
3. **经验统计有效性**：贝叶斯先验在任务样本极少时（<5 次）主要起平滑作用，不具备统计显著性
4. **无并行 Spawn**：当前 spawn_agent 为同步阻塞调用，子 Agent 完成后父级才继续

## 架构详情

详见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 相关项目

- [so100-skill](https://github.com/wefio/so100-skill) — SO100 机械臂自主抓取技能，基于 HHSDD 插件系统
