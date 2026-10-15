# -*- coding: utf-8 -*-
"""
决策均匀化消融研究 (Decision Homogenization Ablation Study)
================================================================
同一场景下对比 3 种出口选择策略：
  1. shared         —— 基线：所有智能体共享同一决策函数（v2.7 风格）
  2. random         —— 均匀随机出口选择（下界参照）
  3. counterfactual —— 拥堵感知出口选择（论文 §5.2 提出的干预）

输出：
  - ablation_results.csv  (每个 (seed, strategy) 的原始指标)
  - 终端汇总表 + 置换检验 p 值

运行：
  python ablation_study.py
"""

import numpy as np
import random
import csv
from math import sqrt, cos, sin, pi
from collections import Counter


# ============================================================
# 配置
# ============================================================
N_SEEDS = 20                    # 每个策略的种子数（论文建议 30，此处 20 以加速）
MAX_STEPS = 1500
N_AGENTS = 10

class Cfg:
    WIDTH, HEIGHT = 1200, 800
    DT = 1.0

    # 火灾
    HEAT_DIFFUSION = 0.08
    SMOKE_DIFFUSION = 0.18
    FIRE_SPREAD_BASE = 0.12
    FIRE_DECAY = 0.9995
    FIRE_BLOCK = 0.70
    SMOKE_SOURCE = 0.30
    FIRE_SPREAD_N_TRY_MAX = 40
    FIRE_SPREAD_N_TRY_PER_BURN = 2
    WIND_CHANGE_PROB = 0.02
    WIND_MAX = 2.0

    # 行为参数
    PANIC_MIN, PANIC_MAX = 0.05, 0.95
    HEALTH_INIT = 100.0
    SPEED_MIN, SPEED_MAX = 3.0, 6.0
    RISK_MIN, RISK_MAX = 0.30, 1.20
    COHESION_MIN, COHESION_MAX = 0.0, 1.0

    # 恐慌
    PANIC_GROWTH_PER_DANGER = 0.020
    PANIC_GROWTH_PER_GLOBAL = 0.010
    PANIC_DECAY_RATE = 0.012
    PANIC_GROWTH_LOW_HEALTH = 0.005
    PANIC_FLOOR = 0.05
    PANIC_HEALTH_THRESHOLD = 40.0
    GLOBAL_DANGER_FIRE_COUNT = 600.0

    # 力场
    ALPHA_PANIC = 0.6
    BETA_RISK = 0.5
    W_GOAL = 2.5
    W_FIRE = 3.0
    W_OBS = 1.5
    W_COOP = 1.2
    W_GOAL_BOOST = 4.0
    PANIC_GOAL_GAIN = 1.5

    COOP_RADIUS = 40.0
    TIME_HORIZON = 10
    HEAT_DAMAGE = 2.0
    SMOKE_DAMAGE = 1.0
    STUCK_WINDOW = 40
    STUCK_MIN = 8.0
    STUCK_ESCAPE = 15

    # 出口
    EXITS = [(600, 60), (1150, 700)]
    EXIT_RADIUS = 30.0

    # 出口代价
    EXIT_DIST_PANIC_GAIN = 0.20
    EXIT_RISK_PANIC_DAMP = 0.40
    EXIT_RISK_SCALE = 900.0
    EXIT_RESELECT_INTERVAL = 15
    EXIT_SWITCH_HYSTERESIS = 0.95

    # [策略 3] 反事实
    COUNTERFACTUAL_GAMMA = 300.0
    CONGESTION_WINDOW = 30


# ============================================================
# 火灾环境
# ============================================================
class FireEnvironment:
    def __init__(self, w=Cfg.WIDTH, h=Cfg.HEIGHT):
        self.w, self.h = w, h
        self.heat = np.zeros((h, w), dtype=np.float32)
        self.smoke = np.zeros((h, w), dtype=np.float32)
        self.obstacle = np.zeros((h, w), dtype=np.uint8)
        self.wind = np.array([1.5, 0.5], dtype=float)
        self.ignition = []

    def add_fire_cluster(self, cx, cy, radius=12):
        cx, cy = int(cx), int(cy)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    x, y = cx + dx, cy + dy
                    if 0 <= x < self.w and 0 <= y < self.h \
                            and self.obstacle[y, x] == 0:
                        self.heat[y, x] = 1.0
                        self.ignition.append((x, y, 1.0))

    def add_obstacle(self, x, y, w, h):
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.w, x + w), min(self.h, y + h)
        self.obstacle[y1:y2, x1:x2] = 1

    def is_obstacle(self, x, y):
        ix, iy = int(x), int(y)
        return not (0 <= ix < self.w and 0 <= iy < self.h) \
               or self.obstacle[iy, ix] == 1

    def sample(self, x, y):
        ix, iy = int(x), int(y)
        if 0 <= ix < self.w and 0 <= iy < self.h:
            return float(self.heat[iy, ix]), float(self.smoke[iy, ix])
        return 1.0, 1.0

    def global_danger(self):
        return min(1.0, int(np.sum(self.heat > 0.5)) / Cfg.GLOBAL_DANGER_FIRE_COUNT)

    def step(self):
        # 热扩散
        up = np.roll(self.heat, 1, axis=0)
        down = np.roll(self.heat, -1, axis=0)
        left = np.roll(self.heat, 1, axis=1)
        right = np.roll(self.heat, -1, axis=1)
        avg = (up + down + left + right) * 0.25
        wx, wy = self.wind
        drift = wx * (left - right) * 0.25 + wy * (up - down) * 0.25
        self.heat = (self.heat * (1.0 - Cfg.HEAT_DIFFUSION)
                     + avg * Cfg.HEAT_DIFFUSION + drift * 0.05)
        self.heat *= Cfg.FIRE_DECAY

        # 蔓延
        burning = np.argwhere(self.heat > Cfg.FIRE_BLOCK)
        if len(burning) > 0:
            n_try = min(Cfg.FIRE_SPREAD_N_TRY_MAX,
                        len(burning) * Cfg.FIRE_SPREAD_N_TRY_PER_BURN)
            for _ in range(n_try):
                idx = random.randint(0, len(burning) - 1)
                cy, cx = int(burning[idx][0]), int(burning[idx][1])
                dy = random.choice([-2, -1, 1, 2])
                dx = random.choice([-2, -1, 1, 2])
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < self.h and 0 <= nx < self.w \
                        and self.obstacle[ny, nx] == 0:
                    p = Cfg.FIRE_SPREAD_BASE * (1.0 + (dx * wx + dy * wy) * 0.15)
                    if random.random() < p:
                        self.heat[ny, nx] = min(1.0, self.heat[ny, nx] + 0.45)

        self.heat[self.obstacle == 1] = 0.0
        for x, y, _ in self.ignition:
            self.heat[y, x] = 1.0
        np.clip(self.heat, 0.0, 1.0, out=self.heat)

        # 烟雾
        src = self.heat > Cfg.SMOKE_SOURCE
        self.smoke[src] = np.minimum(1.0, self.smoke[src] + 0.06)
        up = np.roll(self.smoke, 1, axis=0)
        down = np.roll(self.smoke, -1, axis=0)
        left = np.roll(self.smoke, 1, axis=1)
        right = np.roll(self.smoke, -1, axis=1)
        avg = (up + down + left + right) * 0.25
        drift = wx * (right - left) * 0.25 + wy * (down - up) * 0.25
        self.smoke = (self.smoke * (1.0 - Cfg.SMOKE_DIFFUSION)
                      + avg * Cfg.SMOKE_DIFFUSION + drift * 0.10)
        self.smoke *= 0.98
        self.smoke[self.obstacle == 1] = 0.0
        np.clip(self.smoke, 0.0, 1.0, out=self.smoke)

        # 风向
        if random.random() < Cfg.WIND_CHANGE_PROB:
            ang = random.uniform(0, 2 * pi)
            self.wind = np.array([cos(ang) * Cfg.WIND_MAX, sin(ang) * Cfg.WIND_MAX])


# ============================================================
# 行为参数
# ============================================================
class BehaviorParams:
    __slots__ = ("panic", "health", "speed", "risk_aversion", "cohesion")

    def __init__(self):
        self.panic = random.uniform(Cfg.PANIC_MIN, Cfg.PANIC_MAX)
        self.health = Cfg.HEALTH_INIT
        self.speed = random.uniform(Cfg.SPEED_MIN, Cfg.SPEED_MAX)
        self.risk_aversion = random.uniform(Cfg.RISK_MIN, Cfg.RISK_MAX)
        self.cohesion = random.uniform(Cfg.COHESION_MIN, Cfg.COHESION_MAX)

    def update_panic(self, danger, gdanger):
        self.panic += (Cfg.PANIC_GROWTH_PER_GLOBAL * gdanger
                       + Cfg.PANIC_GROWTH_PER_DANGER * danger)
        self.panic -= Cfg.PANIC_DECAY_RATE * self.panic
        if self.health < Cfg.PANIC_HEALTH_THRESHOLD:
            self.panic += Cfg.PANIC_GROWTH_LOW_HEALTH
        self.panic = float(np.clip(self.panic, Cfg.PANIC_FLOOR, 1.0))

    def effective_speed(self):
        h_factor = 0.6 + 0.4 * (self.health / Cfg.HEALTH_INIT)
        return self.speed * h_factor


# ============================================================
# 智能体（可插拔出口选择策略）
# ============================================================
class Agent:
    def __init__(self, aid, x, y, strategy, env=None):
        self.id = aid
        self.pos = np.array([x, y], dtype=float)
        self.exits = [np.array(e, dtype=float) for e in Cfg.EXITS]
        self.strategy = strategy
        self.exit_id = 0
        self.goal = self.exits[0]
        self.behavior = BehaviorParams()
        self.velocity = np.zeros(2, dtype=float)
        self.safe = False
        self.dead = False
        self.trajectory = [(float(x), float(y))]
        self.progress_window = []
        self.stuck_escape_left = 0
        self.escape_dir = np.zeros(2)
        self.exit_history = []  # (step, exit_id) 用于反事实

        # 初始选择
        self.exit_id = self._choose_exit(env, all_agents=None, step=0)
        self.goal = self.exits[self.exit_id]

    # --------------------------------------------------------
    # 出口代价（基线 + 反事实共用）
    # --------------------------------------------------------
    def _cost(self, env, exit_idx, congestion=0.0):
        b = self.behavior
        target = self.exits[exit_idx]
        d = float(np.linalg.norm(self.pos - target))

        # 路径危险（直线采样 + 8px 邻域最大值）
        R_max = 0.0
        if env is not None:
            n = 30
            for k in range(1, n):
                t = k / n
                sx = self.pos[0] + t * (target[0] - self.pos[0])
                sy = self.pos[1] + t * (target[1] - self.pos[1])
                for ddx in (-8, 0, 8):
                    for ddy in (-8, 0, 8):
                        tx, ty = int(sx + ddx), int(sy + ddy)
                        if env.is_obstacle(tx, ty):
                            R_max = max(R_max, 1.0)
                        else:
                            h, s = env.sample(tx, ty)
                            R_max = max(R_max, h + s)

        lam = b.risk_aversion * (1.0 - Cfg.EXIT_RISK_PANIC_DAMP * b.panic)
        lam = max(0.05, lam)
        dist_gain = 1.0 + Cfg.EXIT_DIST_PANIC_GAIN * b.panic
        return d * dist_gain + lam * R_max * Cfg.EXIT_RISK_SCALE + congestion

    # --------------------------------------------------------
    # 出口选择策略
    # --------------------------------------------------------
    def _choose_exit(self, env, all_agents, step):
        if self.strategy == 'shared':
            return self._argmin_cost(env)

        if self.strategy == 'random':
            # 每隔 RESELECT 步随机翻转
            if step > 0 and step % Cfg.EXIT_RESELECT_INTERVAL == 0:
                return random.randint(0, len(self.exits) - 1)
            return self.exit_id

        if self.strategy == 'counterfactual':
            # 拥堵感知：统计最近窗口内其他智能体选了哪个出口
            congestion = [0.0, 0.0]
            if all_agents is not None:
                for other in all_agents:
                    if other.id == self.id:
                        continue
                    for (s_step, eid) in other.exit_history:
                        if step - s_step <= Cfg.CONGESTION_WINDOW:
                            congestion[eid] += Cfg.COUNTERFACTUAL_GAMMA
            costs = [self._cost(env, k, congestion[k])
                     for k in range(len(self.exits))]
            return int(np.argmin(costs))

        return self._argmin_cost(env)

    def _argmin_cost(self, env):
        costs = [self._cost(env, k) for k in range(len(self.exits))]
        best = int(np.argmin(costs))
        # 迟滞
        if best != self.exit_id:
            cur = costs[self.exit_id]
            if costs[best] < cur * Cfg.EXIT_SWITCH_HYSTERESIS:
                return best
            return self.exit_id
        return best

    # --------------------------------------------------------
    def step(self, env, others, step):
        if self.safe or self.dead:
            return

        h_loc, s_loc = env.sample(self.pos[0], self.pos[1])
        danger = min(1.0, h_loc + s_loc)
        dmg = h_loc * Cfg.HEAT_DAMAGE + s_loc * Cfg.SMOKE_DAMAGE
        self.behavior.health = max(0.0, self.behavior.health - dmg)
        self.behavior.update_panic(danger, env.global_danger())
        if self.behavior.health <= 0.0:
            self.dead = True
            return

        # 周期重选
        if step % Cfg.EXIT_RESELECT_INTERVAL == 0:
            new_id = self._choose_exit(env, others, step)
            if new_id != self.exit_id:
                self.exit_id = new_id
                self.goal = self.exits[new_id]
                self.exit_history.append((step, new_id))

        # 力场运动
        F = self._compute_force(env, others)
        if self.stuck_escape_left > 0:
            F = F + self.escape_dir * 3.0
            self.stuck_escape_left -= 1
        f_norm = float(np.linalg.norm(F))
        direction = F / f_norm if f_norm > 1e-6 else np.zeros(2)
        desired_v = direction * self.behavior.effective_speed()
        self.velocity = 0.6 * self.velocity + 0.4 * desired_v

        new_pos = self.pos + self.velocity * Cfg.DT
        new_pos[0] = float(np.clip(new_pos[0], 2, env.w - 3))
        new_pos[1] = float(np.clip(new_pos[1], 2, env.h - 3))
        if env.is_obstacle(new_pos[0], new_pos[1]):
            slide = self._try_slide(env, new_pos)
            new_pos = slide if slide is not None else self.pos
        self.pos = new_pos

        self.trajectory.append((float(self.pos[0]), float(self.pos[1])))
        if len(self.trajectory) > Cfg.TIME_HORIZON:
            self.trajectory.pop(0)

        # 卡住检测
        d_goal = float(np.linalg.norm(self.pos - self.goal))
        self.progress_window.append(d_goal)
        if len(self.progress_window) > Cfg.STUCK_WINDOW:
            self.progress_window.pop(0)
            if self.progress_window[0] - self.progress_window[-1] < Cfg.STUCK_MIN:
                ang = random.uniform(0, 2 * pi)
                self.escape_dir = np.array([cos(ang), sin(ang)])
                self.stuck_escape_left = Cfg.STUCK_ESCAPE

        # 到达出口
        for k, epos in enumerate(self.exits):
            if float(np.linalg.norm(self.pos - epos)) < Cfg.EXIT_RADIUS:
                self.safe = True
                self.exit_id = k
                break

    def _try_slide(self, env, target):
        direction = target - self.pos
        if np.linalg.norm(direction) < 1e-6:
            return None
        d = direction / np.linalg.norm(direction)
        for angle_deg in (30, -30, 60, -60, 90, -90):
            ang = angle_deg * pi / 180
            ca, sa = cos(ang), sin(ang)
            rot = np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]])
            cand = self.pos + rot * np.linalg.norm(direction)
            if not env.is_obstacle(cand[0], cand[1]):
                return cand
        for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            cx, cy = self.pos[0] + ox, self.pos[1] + oy
            if not env.is_obstacle(cx, cy):
                return np.array([cx, cy])
        return None

    def _compute_force(self, env, others):
        b = self.behavior
        d_goal = self.goal - self.pos
        dg = float(np.linalg.norm(d_goal))
        f_goal = d_goal / dg if dg > 1e-6 else np.zeros(2)

        f_fire = np.zeros(2)
        for (ox, oy) in [(-10, 0), (10, 0), (0, -10), (0, 10),
                         (-7, -7), (-7, 7), (7, -7), (7, 7)]:
            tx, ty = int(self.pos[0] + ox), int(self.pos[1] + oy)
            if 0 <= tx < env.w and 0 <= ty < env.h:
                h = env.heat[ty, tx]; s = env.smoke[ty, tx]
                danger = h + s
                if danger > 0.15:
                    dist = sqrt(ox * ox + oy * oy)
                    weight = danger * (1.0 - dist / 15.0)
                    f_fire -= np.array([ox, oy]) / dist * weight

        w_fire = max(0.1, 1.0 + Cfg.BETA_RISK * b.risk_aversion
                     - Cfg.ALPHA_PANIC * b.panic)

        f_obs = np.zeros(2)
        for (ox, oy) in [(-4, 0), (4, 0), (0, -4), (0, 4),
                         (-3, -3), (-3, 3), (3, -3), (3, 3)]:
            tx, ty = int(self.pos[0] + ox), int(self.pos[1] + oy)
            if 0 <= tx < env.w and 0 <= ty < env.h \
                    and env.obstacle[ty, tx] == 1:
                f_obs -= np.array([ox, oy]) / 4.0

        f_coop = np.zeros(2)
        for other in others:
            if other.id == self.id or other.safe or other.dead:
                continue
            d_vec = self.pos - other.pos
            d = float(np.linalg.norm(d_vec))
            if d < 1e-6 or d > Cfg.COOP_RADIUS:
                continue
            repel = d_vec / d * (Cfg.COOP_RADIUS - d) / Cfg.COOP_RADIUS
            time_overlap = 0.0
            for k, tp in enumerate(other.trajectory[-Cfg.TIME_HORIZON:]):
                dt = sqrt((tp[0] - self.pos[0]) ** 2 + (tp[1] - self.pos[1]) ** 2)
                if dt < Cfg.COOP_RADIUS:
                    time_overlap += (Cfg.COOP_RADIUS - dt) / Cfg.COOP_RADIUS / (k + 1)
            f_coop += repel * (1.0 + 0.5 * time_overlap)
            if b.cohesion > 0.5:
                f_coop -= d_vec / d * (b.cohesion - 0.5) * 0.3

        goal_boost = Cfg.W_GOAL_BOOST * (1.0 + Cfg.PANIC_GOAL_GAIN * b.panic)
        w_goal = Cfg.W_GOAL + goal_boost

        return (w_goal * f_goal + Cfg.W_FIRE * w_fire * f_fire
                + Cfg.W_OBS * f_obs + Cfg.W_COOP * f_coop)


# ============================================================
# 运行单个实验
# ============================================================
def run_one(seed, strategy):
    random.seed(seed)
    np.random.seed(seed)

    env = FireEnvironment()
    # 场景（与 v2.7 一致）
    env.add_obstacle(700, 300, 150, 150)
    env.add_obstacle(300, 600, 100, 100)
    for y in (150, 250, 350):
        env.add_fire_cluster(400, y, radius=35)

    # 生成智能体
    agents = []
    tries = 0
    while len(agents) < N_AGENTS and tries < 1000:
        tries += 1
        x = random.randint(50, 200)
        y = random.randint(300, 550)
        if env.is_obstacle(x, y):
            continue
        # 避开火源
        ok = True
        for (fx, fy, _) in env.ignition:
            if sqrt((x - fx) ** 2 + (y - fy) ** 2) < 60:
                ok = False
                break
        if not ok:
            continue
        agents.append(Agent(len(agents), x, y, strategy, env=env))

    # 仿真
    terminated_at = MAX_STEPS
    for step in range(MAX_STEPS):
        env.step()
        for a in agents:
            a.step(env, agents, step)
        if all(a.safe or a.dead for a in agents):
            terminated_at = step
            break

    # 指标
    safe_agents = [a for a in agents if a.safe]
    exit_counts = Counter(a.exit_id for a in safe_agents)
    nA = exit_counts.get(0, 0)
    nB = exit_counts.get(1, 0)
    n_safe = len(safe_agents)
    n_dead = sum(1 for a in agents if a.dead)

    # 归一化香农熵
    if n_safe > 0:
        pA, pB = nA / n_safe, nB / n_safe
        entropy = 0.0
        for p in (pA, pB):
            if p > 0:
                entropy -= p * np.log(p)
        entropy /= np.log(2)  # 归一化到 [0,1]
    else:
        entropy = 0.0

    return {
        'seed': seed,
        'strategy': strategy,
        'n_safe': n_safe,
        'n_dead': n_dead,
        'n_A': nA,
        'n_B': nB,
        'entropy': entropy,
        'terminated_at': terminated_at,
        'safe_rate': n_safe / N_AGENTS,
    }


# ============================================================
# 置换检验（无 scipy 依赖）
# ============================================================
def permutation_test(x, y, n_perm=2000):
    """双样本置换检验：x, y 是两数组，返回双侧 p 值"""
    x, y = np.asarray(x), np.asarray(y)
    obs = abs(np.mean(x) - np.mean(y))
    combined = np.concatenate([x, y])
    n_x = len(x)
    cnt = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        diff = abs(np.mean(combined[:n_x]) - np.mean(combined[n_x:]))
        if diff >= obs:
            cnt += 1
    return cnt / n_perm


# ============================================================
# 主流程
# ============================================================
def main():
    strategies = ['shared', 'random', 'counterfactual']
    all_results = []

    print("=" * 88)
    print(f" 决策均匀化消融研究 —— {N_SEEDS} 种子 × {len(strategies)} 策略")
    print("=" * 88)

    for strategy in strategies:
        print(f"\n[策略: {strategy}]")
        for seed in range(N_SEEDS):
            r = run_one(seed, strategy)
            all_results.append(r)
            if seed % 5 == 0 or seed == N_SEEDS - 1:
                print(f"  seed={seed:>3d}  safe={r['n_safe']:>2d}/{N_AGENTS}  "
                      f"A={r['n_A']:>2d}  B={r['n_B']:>2d}  "
                      f"H={r['entropy']:.3f}  steps={r['terminated_at']}")

    # ---------- 保存原始数据 ----------
    with open("ablation_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)

    # ---------- 汇总表 ----------
    print("\n" + "=" * 88)
    print(" 汇总表")
    print("=" * 88)
    print(f"{'策略':<16s} | {'安全率':>8s} | {'伤亡率':>8s} | "
          f"{'熵 (H)':>8s} | {'A 占比':>8s} | {'B 占比':>8s} | {'平均步数':>10s}")
    print("-" * 88)

    summary = {}
    for strategy in strategies:
        rs = [r for r in all_results if r['strategy'] == strategy]
        safe_rate = np.mean([r['safe_rate'] for r in rs])
        death_rate = np.mean([r['n_dead'] / N_AGENTS for r in rs])
        entropy = np.mean([r['entropy'] for r in rs])
        a_frac = np.mean([r['n_A'] / max(1, r['n_safe']) for r in rs])
        b_frac = np.mean([r['n_B'] / max(1, r['n_safe']) for r in rs])
        steps = np.mean([r['terminated_at'] for r in rs])
        summary[strategy] = {
            'safe_rate': safe_rate, 'death_rate': death_rate,
            'entropy': entropy, 'a_frac': a_frac, 'b_frac': b_frac,
            'steps': steps
        }
        print(f"{strategy:<16s} | {safe_rate:>7.1%} | {death_rate:>7.1%} | "
              f"{entropy:>8.3f} | {a_frac:>7.1%} | {b_frac:>7.1%} | {steps:>10.1f}")

    # ---------- 统计检验 ----------
    print("\n" + "=" * 88)
    print(" 统计检验（熵指标，置换检验双侧 p 值）")
    print("=" * 88)
    ent_shared = [r['entropy'] for r in all_results if r['strategy'] == 'shared']
    ent_rand = [r['entropy'] for r in all_results if r['strategy'] == 'random']
    ent_cf = [r['entropy'] for r in all_results if r['strategy'] == 'counterfactual']

    p1 = permutation_test(ent_shared, ent_cf)
    p2 = permutation_test(ent_cf, ent_rand)
    print(f"  H(shared) = {np.mean(ent_shared):.3f}  ± {np.std(ent_shared):.3f}")
    print(f"  H(counterfactual) = {np.mean(ent_cf):.3f}  ± {np.std(ent_cf):.3f}")
    print(f"  H(random) = {np.mean(ent_rand):.3f}  ± {np.std(ent_rand):.3f}")
    print(f"\n  shared  vs  counterfactual:  p = {p1:.4f}")
    print(f"  counterfactual  vs  random:  p = {p2:.4f}")
    if p1 < 0.05:
        print("  → 反事实干预显著提升熵（p < 0.05）")
    else:
        print("  → 反事实干预效果不显著")

    print(f"\n已保存：ablation_results.csv  ({len(all_results)} 行)")


if __name__ == "__main__":
    main()