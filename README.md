# HHSDD

**Hyper-Hierarchical Spawn-Driven Decision** / 超分层派生驱动决策系统

自主 AI Agent，单文件实现。核心设计目标：在固定的 204K 上下文窗口内，通过 token 预算硬约束管理递归深度，并基于贝叶斯后验优化执行决策。基于 Anthropic tool_use 协议，支持递归子 Agent、能量预算管理、经验留存和自动压缩。<br>
本项目完全基于minimax token plan制作，其他请自行适配

## 设计约束与假设

- **目标模型**：MiniMax-M2.7（204800 token 上下文）
- **API 协议**：Anthropic Messages API（支持 `tool_use` / `tool_result`）
- **运行环境**：单文件 Python，依赖仅 `anthropic` SDK
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

# 指定 Skill
python agent.py --skill investigation-first --task "调研 Gazebo 仿真框架"

# 自动匹配 Skill
python agent.py --auto-skill --task "分析这张图片"

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

### 流程图追踪

运行时生成 Mermaid 流程图，记录：
- 任务分解与子任务节点
- 工具调用（同 step 内同名工具合并计数）
- Spawn 生命周期（按深度汇总：次数、探索/利用模式、完整/摘要吸收、能量释放、经验上提）
- 能量事件（回收、压缩、截断、紧急交付、Skill 结算、终端奖励）

## 配置

所有参数通过 `.env` 文件配置。详见 `.env.example`。

## 项目结构

```
├── agent.py              # 核心：Agent / 能量 / 工具 / 栈帧 / 流程图 / 贝叶斯预估
├── experience_store.py   # 经验库：SQLite + FTS5 + 权重更新
├── .env                  # 运行配置
├── ARCHITECTURE.md       # 架构设计文档
├── skills/               # Skill 定义（Markdown + YAML frontmatter）
├── agent_state/          # 运行时状态持久化
│   ├── state.json        # 栈帧、步骤计数器、子任务队列
│   ├── flowchart.md      # Mermaid 流程图
│   └── pending_exp.json  # 待处理经验元数据
├── experience_store/     # 经验数据库
│   ├── experience.db     # SQLite
│   └── memory.md         # 最近 20 条记录索引
├── history/              # 运行历史（Markdown 格式）
└── scripts/              # 脚本文件自动路由目录
```

## 限制

1. **模型绑定**：默认面向 minimax 204K 上下文优化，其他模型需调整 `CONTEXT_BUDGET` 和 `CONTEXT_RESERVE`
2. **API 协议**：依赖 Anthropic Messages API 的 `tool_use` 格式及 `cache_control` 扩展
3. **Spawn 信息损耗**：深层递归下子 Agent 返回的摘要对父级而言是有损压缩，复杂依赖链可能出现语义漂移
4. **经验统计有效性**：贝叶斯先验在任务样本极少时（<5 次）主要起平滑作用，不具备统计显著性
5. **无并行 Spawn**：当前 spawn_agent 为同步阻塞调用，子 Agent 完成后父级才继续

```

## 架构详情

详见 [[ARCHITECTURE.md](ARCHITECTURE.md)](https://github.com/wefio/Hyper-Hierarchical-Spawn-Driven-Decision/blob/2403b42a8506b91c9bff40df3fe319ec3a9c105c/ARCHITECTURE.md)。
