# HHSDD

**Hyper-Hierarchical Spawn-Driven Decision** / 超分层派生驱动决策系统

自主 AI Agent，单文件实现。基于 Anthropic tool_use 协议，支持递归子 Agent、能量预算管理、经验留存和自动压缩。

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

## 核心特性

### 递归子 Agent

`spawn_agent` 是一个工具调用。子 Agent 隔离执行，返回压缩摘要。主 Agent 上下文零污染。
子 Agent 可继续 spawn，形成树状探索结构，深度由能量预算自动限制。

### 能量即 Token

能量直接用 token 计量（1E = 1 token），物理约束不可绕过：
- 每步固定开销 1000E
- API 调用按 input × 1 + output × 3 扣费
- Spawn 预留 15% 能量，返回后释放
- 能量耗尽 = 任务终止

### 经验留存

每次任务自动收集经验元数据，失败或复杂任务提取结构化教训。
新任务启动时检索相关经验注入上下文。蚁群信息素机制让成功路径更容易被复用。

### 自适应压缩

上下文接近预算时自动压缩早期步骤（step_detail → summary），循环压缩直到低于阈值。
极端情况截断最旧帧。

### Skill 系统

Markdown + YAML frontmatter 定义 Skill，支持关键词匹配和自动激活。
内置 10 个 Skill：调研优先、批评与自我批评、持久策略、浏览器自动化等。

### 流程图追踪

自动生成 Mermaid 流程图，记录任务分解、工具调用、子 Agent 生命周期。
同步骤工具调用合并计数，生命周期事件汇总为每深度统计。

## 配置

所有参数通过 `.env` 文件配置，详见 `.env`。关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CONTEXT_BUDGET` | 204800 | 模型上下文窗口上限 |
| `CONTEXT_RESERVE` | 50000 | 保留给 system/tools |
| `STEP_OVERHEAD` | 1000 | 每步固定能量开销 |
| `TOKEN_COST_INPUT` | 1.0 | 输入能量系数 |
| `TOKEN_COST_OUTPUT` | 3.0 | 输出能量系数 |
| `COMPRESSION_THRESHOLD` | 0.9 | 上下文占比触发压缩 |
| `MAX_SPAWN_DEPTH` | 0 | 子 Agent 最大深度（0=不限） |

## 项目结构

```
├── agent.py              # 核心逻辑（Agent / 能量 / 工具 / 栈帧 / 流程图）
├── experience_store.py   # 经验库（SQLite + FTS5）
├── pso_tuner.py          # PSO 参数调优
├── .env                  # 运行配置
├── ARCHITECTURE.md       # 架构设计文档
├── skills/               # Skill 定义
├── agent_state/          # 运行时状态
│   ├── state.json        # 栈帧/进度
│   ├── flowchart.md      # Mermaid 流程图
│   └── pending_exp.json  # 待处理经验
├── experience_store/     # 经验数据库
├── history/              # 运行历史
└── scripts/              # 辅助脚本
```

## 架构详情

详见 [ARCHITECTURE.md](ARCHITECTURE.md)。
