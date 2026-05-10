# HHSDD

**Hyper-Hierarchical Spawn-Driven Decision** / 超分层派生驱动决策系统

自主 AI Agent。核心思路：把 LLM 的上下文窗口当作唯一的硬件资源来管理——自动组装合适上下文、最大化利用每一段 context、通过递归子 Agent 分解复杂任务、用能量约束防止无限派生、复用历史经验。

## 设计假设

- 协议：Anthropic Messages API（`tool_use` / `tool_result`）
- 环境：Python 3.10+，依赖 `anthropic` + `pyyaml`

## 快速开始

```bash
pip install anthropic pyyaml
cp models.yaml.template models.yaml   # 填入你的模型
python main.py --task "你的任务"
python main.py -i --task "..."        # 交互模式
python main.py --resume               # 恢复状态
```

## 上下文管理

上下文窗口是核心资源。所有设计围绕"在有限 context 内完成复杂任务"。

### 自动组装

每个 step 前，`_build_messages` 从栈帧自动组装 API 消息：system prompt + 经验 + plan + 已完成步骤 + 当前任务。子 Agent 继承父级前缀，最大化缓存命中。

### 自动维护

`_adapt_context` 每步后三阶段保持 context 在预算内：回收借阅内容 → 压缩旧步骤为摘要 → 截断为磁盘指针。`_reclaim_energy` 能量不足时按优先级弹栈。

### 硬盘缓存（PointerStore）

内容从上下文移出时持久化到本地文件，原地替换为 `摘要:ptr_id` 指针。LLM 通过 `recall` 工具按需取回（类似 page fault）。多级页表四级查找：精确复用 → 同模板参考 → 全局经验 → FTS5 全文。TLB 快表缓存热门指针。

### 缓存分层（L1/L2）

每个 Agent 的私有栈 = L1。同 spawn 组兄弟共享 L2 结果池（`read_peer` 工具），只传指针不传全文。

### 令牌计数

每个 API 调用后累积追踪实际 token 用量。优先用 SDK `count_tokens()` 精确计数，降级到 char/4 估算。所有 token 等权 1:1——能量计的是 context 窗口占用。

## 递归分解（Spawn）

`spawn_agent` 是原生工具。父 Agent 把子任务拆给子 Agent，子独立执行，返回结果被父吸收。

- 子 Agent 和父 Agent 是**同一个类、同一种执行循环**——Spawn 只是工具调用，没有特殊通道
- 子继承父的 system prompt + skills + 失败经验作为消息前缀
- 并行 spawn：检测到多个 spawn_agent → ThreadPoolExecutor 并发
- 父可声明子为叶子（`_child_can_spawn=false`）

## 能量约束

目的是**防止无限递归**。双层计数器：`energy`（流动资金，可预扣返还）+ `total_spent`（硬上限，触顶即停）。

Spawn 能量模型：窗口对齐 `min(父池, 子模型窗口)` → 投资 1% + 冻结 15% → 子获独立池。回收时释放冻结 + 结算投资。吸收预算 = `冻结/3 + 子剩余×0.3`。每层衰减 `×0.8415`，越深池越小，自然停止。

贝叶斯后验辅助决策：Beta 估计任务/子任务/spawn 成功率，Gamma 估计命令耗时。

## 模型路由（ModelRouter）

`models.yaml` 声明模型池（tier：opus/sonnet/haiku/fallback）。按 tier 路由，降级链自动容灾。独立 ProviderAdapter 处理各厂商差异（DeepSeek thinking mode、Kimi reasoning_content、本地无认证）。人类也作为一种模型（stdin 阻塞，干预预算控制）。

## 分形模板

YAML 声明式任务树。三种分叉：natural（LLM 动态拆分）、template（预设结构）、multi_parallel（复制 N 份并行，强制聚合后继续）。模板节点自带指针缓存，相同输入自动复用历史结果。

## 工具

`read_file` `list_dir` `write_file` `view_image` `spawn_agent` `run_command` `use_skill` `recall` `savepoint` `read_peer` `ipython`

Root 有完整描述，子节点去描述（从父继承理解）。叶子节点无 `spawn_agent`。只读工具并行执行。

## 代码文件

```
main.py  config.py  skill.py  agent.py
agent_memory_frame.py  agent_memory_l2.py  agent_memory_report.py
agent_process_executor.py  agent_scheduler_energy.py
agent_kernel_glue.py  agent_kernel_router.py  agent_kernel_verify.py
agent_template.py  pointer_store.py  experience_store.py
plugins/  models.yaml.template  .env.example
```

## 配置

| 参数 | 默认 | 说明 |
|------|------|------|
| `MODEL` | MiniMax-M2.7 | 默认模型 |
| `CONTEXT_BUDGET` | 204800 | 上下文窗口 |
| `SPAWN_INVEST_RATIO` | 0.01 | 投资比例 |
| `SPAWN_RESERVE_RATIO` | 0.15 | 冻结比例 |
| `TOOL_ENERGY_COST` | 200 | 工具调用预扣 |
| `AUTO_PLAN` | smart | always/smart/never |
| `INTERVENTION_BUDGET` | 3 | 人类干预上限 |
| `MAX_SPAWN_DEPTH` | 0 | 深度限制(0=不限制) |

## 限制

- 协议依赖 Anthropic Messages API
- 深层递归下摘要为有损压缩
- 无测试套件
