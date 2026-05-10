# HHSDD

**Hyper-Hierarchical Spawn-Driven Decision** / 超分层派生驱动决策系统

自主 AI Agent，核心设计目标：在有限的 LLM 上下文窗口内，通过 token 预算硬约束管理递归深度，基于贝叶斯后验优化决策。整个设计大量借用操作系统内存管理概念（虚拟内存、页表、swap、TLB、CoW）。

## 设计假设

- **API 协议**：Anthropic Messages API（`tool_use` / `tool_result`）
- **运行环境**：Python 3.10+，依赖 `anthropic` + `pyyaml`
- **核心假设**：Skill 在一次运行中不变；子 Agent 通过消息前缀继承父上下文

## 快速开始

```bash
pip install anthropic pyyaml

cp .env.example .env              # 编辑填入 API key
cp models.yaml.template models.yaml  # 编辑填入你的模型

python main.py --task "你的任务"
python main.py -i --task "..."     # 交互模式
python main.py --resume            # 恢复状态
PLUGINS=example_plugin python main.py --task "..."  # 加载插件
```

## 核心设计 — OS 内存管理类比

| 概念 | OS 类比 | 项目实现 |
|------|--------|---------|
| Context 窗口 | RAM | `CONTEXT_BUDGET` |
| StackFrame | 页表项 | `type/level/content/use_count/last_used_step` |
| PointerStore | Disk swap | `archive/*.md`，stub `摘要:ptr_id` |
| recall 工具 | Page fault | 原地展开 pointer stub（TLB 加速） |
| L2Cache | CPU L2 | 同级 agent 共享指针列表 |
| TLB | 快表 | 5 条 LRU，recall 命中零 I/O |
| 子 Agent 前缀继承 | CoW / 共享页表 | 缓存命中，递归分解可行 |
| Savepoint | PDA 栈快照 | 回退状态，保留推导历史 |
| Energy | 资源配额 | 防止无限递归，token 等权 1:1 |
| Agent | 进程 | 同构递归，Spawn 只是工具之一 |

## 代码文件

```
agent.py                    # Agent 主类
main.py                     # CLI 入口
config.py                   # Config 单例
skill.py                    # Skill + SkillManager

agent_memory_frame.py       # StackFrame, SavepointMeta
agent_memory_l2.py          # L2Cache
agent_memory_report.py      # NodeReport

agent_process_executor.py   # ToolExecutor + TOOL_DEFINITIONS + energy_hooks

agent_scheduler_energy.py   # BayesianEnergyManager + BayesianOptimizer

agent_kernel_glue.py        # APIClient + CacheProvider + FlowchartRecorder +
                            #   EventBus + SavepointManager + PlanContext
agent_kernel_router.py      # ModelRouter + 5 ProviderAdapter
agent_kernel_verify.py      # TaskVerifier

agent_template.py           # 分形模板引擎
pointer_store.py            # 多级页表 + GlobalPointerStore + TLB
experience_store.py         # SQLite + FTS5 经验留存
plugins/                    # importlib 动态插件
```

## 核心机制

### 递归子 Agent（Spawn）

`spawn_agent` 是原生工具。子 Agent 隔离执行，同步阻塞。父级通过 `_absorb_child_result` 吸收结果。

- **上下文继承**：子继承父的 system prompt + skills + shared files + failure experience，最大化缓存命中
- **能量模型**：窗口对齐 `min(父池, 子模型窗口)` → invest 1% + freeze 15% → 子获独立能量池。返回后释放 freeze + 结算 invest
- **吸收预算**：`freeze/3 + 子剩余×0.3`（tokens），预算内完整保留否则压缩
- **深度约束**：能量指数衰减自然约束，`MAX_SPAWN_DEPTH` 硬上限（默认 0=不限）
- **叶子声明**：父可传 `_child_can_spawn=false` 禁止子继续派生；`can_spawn=False` 模型标记为叶子
- **并行 spawn**：检测到多个 spawn_agent → ThreadPoolExecutor 并行执行

### 能量系统

双层计数器：`energy`（软，流动资金，允许透支 -10%）+ `total_spent`（硬，不可逆）。所有 token 等权 1:1。

- **贝叶斯后验**：Beta(α,β) 任务完成/子任务/spawn 成功率；Gamma 命令耗时
- **停止条件**：硬预算耗尽 / 流动资金耗尽 / P(done)>0.9 / 5 轮无进展 / 5 次失败
- **终端奖励**：`base × difficulty × efficiency × step_tier`
- **投资-收益**：explore 成功=本金+ROI，失败=血本无归；exploit 按规则

### 上下文栈

Level 0=constraint(永久) → 1=plan → 2=step_detail/summary/pointer → 3=merge。
`_adapt_context` 三阶段维护：回收 reclaimable → 压缩 step_detail→summary → 截断为 pointer stub。
`_build_messages` 将栈帧组装为 API 消息对。

### 多级页表 + 指针复用

L0(原文) → L1(input_hash 组) → L2(模板索引) → L3(全局)。四级查找：精确复用 → 同模板参考 → 全局经验 → FTS5 全文。
TLB(5 条 LRU) 缓存 recall 结果，原地展开 pointer stub。写完 SHA256 签名防篡改。

### 模型路由（ModelRouter）

`models.yaml` 声明模型，按 tier（opus/sonnet/haiku/fallback）路由。降级链：指定 tier → 同 tier sticky → 同 tier 容灾 → 下一级 → fallback。5 个 ProviderAdapter 处理厂商差异（thinking mode、缓存指标）。

### 分形模板引擎

YAML 声明式模板，三种分叉：natural（LLM 动态）、template（预设结构）、multi_parallel（复制 N 份强制聚合）。四级指针复用，支持在线贝叶斯优化。

### 工具列表（11 个）

`read_file` `list_dir` `write_file` `view_image` `spawn_agent` `run_command` `use_skill` `recall` `savepoint` `read_peer` `ipython`

Root 有完整描述，子节点去描述（从父继承理解）。叶子无 `spawn_agent`。

### 异构厂商支持

| 厂商 | 适配器 | 特有处理 |
|------|--------|---------|
| Anthropic | `ProviderAdapter` | `cache_control: ephemeral` |
| DeepSeek | `DeepSeekAdapter` | thinking mode, `prompt_cache_hit_tokens` |
| Kimi | `KimiAdapter` | max_tokens≥16k, reasoning_content |
| Local | `LocalAdapter` | 无认证 |
| Human | `HumanAdapter` | stdin 阻塞, 干预预算 |

### 人类模型

`model_spec={'id': 'human:default'}` — 展示结构化摘要 + 阻塞 stdin。`intervention_budget` 耗尽自动降级。

## 配置

所有参数通过 `.env` 或环境变量。详见 `.env.example`。

| 参数 | 默认 | 说明 |
|------|------|------|
| `MODEL` | MiniMax-M2.7 | 默认模型 |
| `CONTEXT_BUDGET` | 204800 | 上下文窗口 |
| `CONTEXT_RESERVE` | 50000 | 保留值 |
| `SPAWN_INVEST_RATIO` | 0.01 | 投资比例 |
| `SPAWN_RESERVE_RATIO` | 0.15 | 冻结比例 |
| `TOOL_ENERGY_COST` | 200 | 工具调用预扣 |
| `AUTO_PLAN` | smart | always/smart/never |
| `INTERVENTION_BUDGET` | 3 | 人类干预次数上限 |

## 限制

1. 协议依赖 Anthropic Messages API 格式
2. 深层递归下摘要是有损压缩，可能语义漂移
3. spawn 为同步阻塞（并行 spawn 已支持）
4. 无测试套件

## 相关项目

- [so100-skill](https://github.com/wefio/so100-skill) — SO100 机械臂自主抓取技能
