"""粒子群参数调优 — 纯 stdlib，无外部依赖"""
import json
import random
from pathlib import Path


class Particle:
    __slots__ = ('position', 'velocity', 'best_position', 'best_fitness')

    def __init__(self, position: list, bounds: list):
        self.position = position[:]
        self.velocity = [random.uniform(-0.1, 0.1) * (b[1] - b[0]) for b in bounds]
        self.best_position = position[:]
        self.best_fitness = -float('inf')


class PSOTuner:
    """10 粒子 PSO，优化 BayesianEnergyManager 的 cost 参数。

    粒子 = [cost_step, cost_tool, cost_spawn, explore_roi]
    适应度 = success_rate * 10 + efficiency_bonus * 5
    每 10 个任务跑一次迭代，状态持久化到 pso_state.json。
    """

    BOUNDS = [
        (0.5, 5.0),   # cost_step
        (0.05, 1.0),  # cost_tool
        (0.1, 2.0),   # cost_spawn
        (0.1, 1.5),   # explore_roi
    ]
    N_PARTICLES = 10
    W = 0.7   # 惯性
    C1 = 1.5  # 认知
    C2 = 1.5  # 社会
    ITERATION_INTERVAL = 10

    def __init__(self, state_dir: str = None):
        from agent import Config
        state_dir = Path(state_dir or Config.WORK_DIR)
        state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = state_dir / "pso_state.json"
        self.particles: list = []
        self.global_best_position: list = [2.0, 0.2, 0.5, 0.5]
        self.global_best_fitness: float = -float('inf')
        self.task_buffer: list = []
        self.tasks_since_iteration = 0
        self._load()

    def _load(self):
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding='utf-8'))
                self.global_best_position = data.get("global_best", self.global_best_position)
                self.global_best_fitness = data.get("global_best_fitness", -float('inf'))
                self.tasks_since_iteration = data.get("tasks_since", 0)
                for pd in data.get("particles", []):
                    p = Particle(pd["position"], self.BOUNDS)
                    p.best_position = pd.get("best_position", pd["position"])
                    p.best_fitness = pd.get("best_fitness", -float('inf'))
                    self.particles.append(p)
            except (json.JSONDecodeError, KeyError):
                pass

    def _save(self):
        data = {
            "global_best": self.global_best_position,
            "global_best_fitness": self.global_best_fitness,
            "tasks_since": self.tasks_since_iteration,
            "particles": [
                {"position": p.position, "best_position": p.best_position,
                 "best_fitness": p.best_fitness}
                for p in self.particles
            ]
        }
        self.state_path.write_text(json.dumps(data, indent=2), encoding='utf-8')

    def _init_particles(self):
        self.particles = []
        for _ in range(self.N_PARTICLES):
            pos = [
                max(self.BOUNDS[i][0], min(self.BOUNDS[i][1],
                    self.global_best_position[i] + random.gauss(0, (self.BOUNDS[i][1] - self.BOUNDS[i][0]) * 0.2)))
                for i in range(len(self.BOUNDS))
            ]
            self.particles.append(Particle(pos, self.BOUNDS))

    def get_best_params(self) -> dict:
        b = self.global_best_position
        return {"cost_step": b[0], "cost_tool": b[1], "cost_spawn": b[2], "explore_roi": b[3]}

    def record_task_result(self, success: bool, efficiency: float):
        """任务结束时调用。efficiency = total_spent / total_energy"""
        self.task_buffer.append({"success": success, "efficiency": efficiency})
        self.tasks_since_iteration += 1
        if self.tasks_since_iteration >= self.ITERATION_INTERVAL:
            self._run_iteration()
            self.tasks_since_iteration = 0
            self.task_buffer = []
            self._save()

    def _compute_fitness(self) -> float:
        if not self.task_buffer:
            return 0.0
        success_rate = sum(1 for t in self.task_buffer if t["success"]) / len(self.task_buffer)
        avg_eff = sum(t["efficiency"] for t in self.task_buffer) / len(self.task_buffer)
        return success_rate * 10.0 + (1.0 - avg_eff) * 5.0

    def _run_iteration(self):
        if not self.particles:
            self._init_particles()
        fitness = self._compute_fitness()
        # 只更新最接近实际使用参数的粒子（global best），不是所有粒子
        if fitness > self.global_best_fitness:
            self.global_best_fitness = fitness
        # 找到离 global_best 最近的粒子，更新其 personal best
        best_match = min(self.particles,
                         key=lambda p: sum((p.position[i] - self.global_best_position[i])**2
                                           for i in range(len(self.BOUNDS))))
        if fitness > best_match.best_fitness:
            best_match.best_fitness = fitness
            best_match.best_position = best_match.position[:]
        # 速度/位置更新
        for p in self.particles:
            for i in range(len(self.BOUNDS)):
                r1, r2 = random.random(), random.random()
                p.velocity[i] = (
                    self.W * p.velocity[i]
                    + self.C1 * r1 * (p.best_position[i] - p.position[i])
                    + self.C2 * r2 * (self.global_best_position[i] - p.position[i])
                )
                p.position[i] = max(self.BOUNDS[i][0],
                                    min(self.BOUNDS[i][1], p.position[i] + p.velocity[i]))
