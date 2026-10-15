# -*- coding: utf-8 -*-
"""
基于个体行为参数的多智能体协同疏散路径规划方法 v2.2
============================================================
相对 v2.1 的关键升级：
  [S1] 火灾蔓延速率调优至文献量级（不再 100 步爆燃）
  [S2] 双出口场景：出口 A(近/危险) vs 出口 B(远/安全)
  [S3] 行为参数驱动的出口选择（专利核心创新）
        λ_i = risk_aversion_i × (1 - κ · panic_i)
        Cost_i(exit) = D_i(exit) · (1 + ε·panic_i) + λ_i · R_i(exit) · D_scale
  [S4] 出口重评估机制 + 迟滞切换
  [S5] 输出增加出口选择统计
"""

import numpy as np
import random
import csv
from math import sqrt, cos, sin, pi


# =====================================================================
# 全局参数
# =====================================================================
class Params:
    WIDTH, HEIGHT = 1200, 800
    DT = 1.0
    MAX_STEPS = 3000

    # ---------- 火灾动力学（v2.2 调优，接近文献量级） ----------
    HEAT_DIFFUSION = 0.08
    SMOKE_DIFFUSION = 0.18
    FIRE_SPREAD_BASE = 0.12
    FIRE_DECAY = 0.9995
    FIRE_BLOCK = 0.70
    SMOKE_SOURCE = 0.30
    WIND_CHANGE_PROB = 0.02
    WIND_MAX = 2.0

    # 蔓延采样：v2.1 用了 min(120, N*6)，v2.2 降为 min(60, N*3)
    FIRE_SPREAD_N_TRY_MAX = 60
    FIRE_SPREAD_N_TRY_PER_BURN = 3

    # ---------- 行为参数 ----------
    PANIC_MIN, PANIC_MAX = 0.05, 0.95
    HEALTH_INIT = 100.0
    SPEED_MIN, SPEED_MAX = 3.0, 6.0
    RISK_AVERSION_MIN, RISK_AVERSION_MAX = 0.30, 1.20
    COHESION_MIN, COHESION_MAX = 0.0, 1.0

    # 恐慌动力学
    PANIC_GROWTH_PER_DANGER = 0.02
    PANIC_GROWTH_ON_LOW_HEALTH = 0.005
    PANIC_DECAY_RATE = 0.001
    PANIC_FLOOR = 0.05
    PANIC_HEALTH_THRESHOLD = 40.0

    # 行为调制（力场）
    ALPHA_PANIC = 0.6
    BETA_RISK = 0.5

    # 力权重
    W_GOAL = 2.5
    W_FIRE = 3.0
    W_OBS = 1.5
    W_COOP = 1.2

    # 协同
    COOP_RADIUS = 40.0
    TIME_HORIZON = 10

    # 伤害
    HEAT_DAMAGE_RATE = 2.0
    SMOKE_DAMAGE_RATE = 1.0

    # 卡住检测
    STUCK_WINDOW = 40
    STUCK_MIN_PROGRESS = 8.0
    STUCK_ESCAPE_STEPS = 15

    # ---------- 双出口 [S2] ----------
    EXITS = [
        (1150, 80),   # 出口 A：右上，近但危险（火源在前往路径上）
        (1150, 720),  # 出口 B：右下，远但安全
    ]
    EXIT_RADIUS = 30.0

    # ---------- 出口选择参数 [S3] ----------
    EXIT_DIST_PANIC_GAIN = 0.20      # ε: 恐慌放大距离权
    EXIT_RISK_PANIC_DAMP = 0.70      # κ: 恐慌削弱风险规避的强度
    EXIT_RISK_SCALE = 400.0          # D_scale: 使危险项与距离项量级相当
    EXIT_RISK_SAMPLES = 30           # 沿路径采样点数

    # ---------- 出口重评估 [S4] ----------
    EXIT_RESELECT_INTERVAL = 25      # 每 N 步重评估
    EXIT_SWITCH_HYSTERESIS = 0.80    # 新出口代价需 < 当前 × 0.8 才切换


# =====================================================================
# 个体行为参数
# =====================================================================
class BehaviorParams:
    __slots__ = ("panic", "health", "speed", "risk_aversion", "cohesion")

    def __init__(self):
        self.panic = random.uniform(Params.PANIC_MIN, Params.PANIC_MAX)
        self.health = Params.HEALTH_INIT
        self.speed = random.uniform(Params.SPEED_MIN, Params.SPEED_MAX)
        self.risk_aversion = random.uniform(
            Params.RISK_AVERSION_MIN, Params.RISK_AVERSION_MAX)
        self.cohesion = random.uniform(
            Params.COHESION_MIN, Params.COHESION_MAX)

    def vector(self):
        return (self.panic, self.health, self.speed,
                self.risk_aversion, self.cohesion)

    def update_panic(self, danger_level):
        if danger_level > 0.05:
            self.panic += Params.PANIC_GROWTH_PER_DANGER * danger_level
        else:
            self.panic -= Params.PANIC_DECAY_RATE

        if self.health < Params.PANIC_HEALTH_THRESHOLD:
            self.panic += Params.PANIC_GROWTH_ON_LOW_HEALTH

        self.panic = float(np.clip(self.panic, Params.PANIC_FLOOR, 1.0))

    def effective_speed(self):
        h_factor = 0.6 + 0.4 * (self.health / Params.HEALTH_INIT)
        return self.speed * h_factor


# =====================================================================
# 火灾环境
# =====================================================================
class FireEnvironment:
    def __init__(self, w=Params.WIDTH, h=Params.HEIGHT):
        self.w, self.h = w, h
        self.heat = np.zeros((h, w), dtype=np.float32)
        self.smoke = np.zeros((h, w), dtype=np.float32)
        self.obstacle = np.zeros((h, w), dtype=np.uint8)
        self.wind = np.array([1.5, 0.5], dtype=float)
        self.ignition = []

    def add_ignition(self, x, y, intensity=1.0):
        x, y = int(x), int(y)
        if 0 <= x < self.w and 0 <= y < self.h:
            self.heat[y, x] = intensity
            self.ignition.append((x, y, intensity))

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

    # ---------- 步进 ----------
    def step(self):
        self._step_heat()
        self._step_smoke()
        if random.random() < Params.WIND_CHANGE_PROB:
            ang = random.uniform(0, 2 * pi)
            self.wind = np.array([cos(ang) * Params.WIND_MAX,
                                  sin(ang) * Params.WIND_MAX])

    def _step_heat(self):
        up = np.roll(self.heat, 1, axis=0)
        down = np.roll(self.heat, -1, axis=0)
        left = np.roll(self.heat, 1, axis=1)
        right = np.roll(self.heat, -1, axis=1)
        avg = (up + down + left + right) * 0.25

        wx, wy = self.wind
        drift = wx * (left - right) * 0.25 + wy * (up - down) * 0.25

        self.heat = (self.heat * (1.0 - Params.HEAT_DIFFUSION)
                     + avg * Params.HEAT_DIFFUSION
                     + drift * 0.05)
        self.heat *= Params.FIRE_DECAY

        # 从已燃点采样蔓延 [v2.2 采样频率已降低]
        burning = np.argwhere(self.heat > Params.FIRE_BLOCK)
        if len(burning) > 0:
            n_try = min(Params.FIRE_SPREAD_N_TRY_MAX,
                        len(burning) * Params.FIRE_SPREAD_N_TRY_PER_BURN)
            for _ in range(n_try):
                idx = random.randint(0, len(burning) - 1)
                cy, cx = int(burning[idx][0]), int(burning[idx][1])
                dy = random.choice([-2, -1, 1, 2])
                dx = random.choice([-2, -1, 1, 2])
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < self.h and 0 <= nx < self.w \
                        and self.obstacle[ny, nx] == 0:
                    p = Params.FIRE_SPREAD_BASE * \
                        (1.0 + (dx * wx + dy * wy) * 0.15)
                    if random.random() < p:
                        self.heat[ny, nx] = min(1.0, self.heat[ny, nx] + 0.45)

        self.heat[self.obstacle == 1] = 0.0
        for x, y, intensity in self.ignition:
            self.heat[y, x] = max(self.heat[y, x], intensity)
        np.clip(self.heat, 0.0, 1.0, out=self.heat)

    def _step_smoke(self):
        src = self.heat > Params.SMOKE_SOURCE
        self.smoke[src] = np.minimum(1.0, self.smoke[src] + 0.06)

        up = np.roll(self.smoke, 1, axis=0)
        down = np.roll(self.smoke, -1, axis=0)
        left = np.roll(self.smoke, 1, axis=1)
        right = np.roll(self.smoke, -1, axis=1)
        avg = (up + down + left + right) * 0.25

        wx, wy = self.wind
        drift = wx * (right - left) * 0.25 + wy * (down - up) * 0.25
        self.smoke = (self.smoke * (1.0 - Params.SMOKE_DIFFUSION)
                      + avg * Params.SMOKE_DIFFUSION
                      + drift * 0.10)
        self.smoke *= 0.98
        self.smoke[self.obstacle == 1] = 0.0
        np.clip(self.smoke, 0.0, 1.0, out=self.smoke)


# =====================================================================
# 智能体（含行为参数驱动的出口选择）
# =====================================================================
class Agent:
    def __init__(self, aid, x, y, exits):
        self.id = aid
        self.pos = np.array([x, y], dtype=float)
        self.exits = [np.array(e, dtype=float) for e in exits]
        self.exit_id = None                 # 当前选择的出口索引
        self.goal = np.array(self.exits[1], dtype=float)  # 默认选远出口
        self.behavior = BehaviorParams()
        self.velocity = np.zeros(2, dtype=float)
        self.safe = False
        self.dead = False
        self.trajectory = [(float(x), float(y))]

        # 卡住检测
        self.progress_window = []
        self.stuck_escape_left = 0
        self.escape_dir = np.zeros(2)

        # 初始出口选择
        self.exit_id = self._choose_best_exit(None, force=True)
        self.goal = self.exits[self.exit_id]

    # =================================================================
    # [S3] 行为参数驱动的出口选择
    # =================================================================
    def _exit_cost(self, env, exit_idx):
        """
        代价函数（专利核心）：
          C_i(e) = D_i(e) · (1 + ε·panic_i)
                 + λ_i · R_i(e) · D_scale
          其中 λ_i = risk_aversion_i · (1 - κ · panic_i)
        """
        b = self.behavior
        target = self.exits[exit_idx]

        # 几何距离
        d = float(np.linalg.norm(self.pos - target))

        # 路径危险 R_i(e)：沿直线采样平均 heat+smoke
        R = 0.0
        if env is not None:
            n = Params.EXIT_RISK_SAMPLES
            for k in range(1, n):
                t = k / n
                sx = self.pos[0] + t * (target[0] - self.pos[0])
                sy = self.pos[1] + t * (target[1] - self.pos[1])
                if env.is_obstacle(sx, sy):
                    h, s = 1.0, 1.0     # 障碍视作需绕行 → 危险
                else:
                    h, s = env.sample(sx, sy)
                R += (h + s)
            R /= max(1, n - 1)

        # 行为调制风险权重
        lam = b.risk_aversion * (1.0 - Params.EXIT_RISK_PANIC_DAMP * b.panic)
        lam = max(0.05, lam)

        # 恐慌放大距离权重
        dist_gain = 1.0 + Params.EXIT_DIST_PANIC_GAIN * b.panic

        cost = d * dist_gain + lam * R * Params.EXIT_RISK_SCALE
        return cost

    def _choose_best_exit(self, env, force=False):
        if env is None and not force:
            return self.exit_id
        if env is None:
            # 初始无环境信息，默认选远出口（保守）
            return 1

        costs = [self._exit_cost(env, k) for k in range(len(self.exits))]
        best = int(np.argmin(costs))

        # 迟滞：只有当新出口明显更好才切换 [S4]
        if self.exit_id is not None and best != self.exit_id:
            cur_cost = costs[self.exit_id]
            if costs[best] < cur_cost * Params.EXIT_SWITCH_HYSTERESIS:
                return best
            else:
                return self.exit_id
        return best

    # =================================================================
    # 主步进
    # =================================================================
    def step(self, env, others, step_count):
        if self.safe or self.dead:
            return

        # ---- 采样危险、更新健康/恐慌 ----
        h_loc, s_loc = env.sample(self.pos[0], self.pos[1])
        danger = min(1.0, h_loc + s_loc)

        dmg = h_loc * Params.HEAT_DAMAGE_RATE + s_loc * Params.SMOKE_DAMAGE_RATE
        self.behavior.health = max(0.0, self.behavior.health - dmg)
        self.behavior.update_panic(danger)

        if self.behavior.health <= 0.0:
            self.dead = True
            return

        # ---- [S4] 周期性出口重评估 ----
        if step_count % Params.EXIT_RESELECT_INTERVAL == 0:
            new_id = self._choose_best_exit(env)
            if new_id != self.exit_id:
                self.exit_id = new_id
                self.goal = self.exits[new_id]

        # ---- 计算力并运动 ----
        F = self._compute_force(env, others)

        if self.stuck_escape_left > 0:
            F = F + self.escape_dir * 3.0
            self.stuck_escape_left -= 1

        f_norm = float(np.linalg.norm(F))
        direction = F / f_norm if f_norm > 1e-6 else np.zeros(2)

        desired_v = direction * self.behavior.effective_speed()
        self.velocity = 0.6 * self.velocity + 0.4 * desired_v

        new_pos = self.pos + self.velocity * Params.DT
        new_pos[0] = float(np.clip(new_pos[0], 2, env.w - 3))
        new_pos[1] = float(np.clip(new_pos[1], 2, env.h - 3))

        if env.is_obstacle(new_pos[0], new_pos[1]):
            slide = self._try_slide(env, new_pos)
            new_pos = slide if slide is not None else self.pos

        self.pos = new_pos

        self.trajectory.append((float(self.pos[0]), float(self.pos[1])))
        if len(self.trajectory) > Params.TIME_HORIZON:
            self.trajectory.pop(0)

        # ---- 卡住检测 ----
        d_goal = float(np.linalg.norm(self.pos - self.goal))
        self.progress_window.append(d_goal)
        if len(self.progress_window) > Params.STUCK_WINDOW:
            self.progress_window.pop(0)
            if self.progress_window[0] - self.progress_window[-1] \
                    < Params.STUCK_MIN_PROGRESS:
                ang = random.uniform(0, 2 * pi)
                self.escape_dir = np.array([cos(ang), sin(ang)])
                self.stuck_escape_left = Params.STUCK_ESCAPE_STEPS

        # ---- 到达任一出口 ----
        for k, epos in enumerate(self.exits):
            if float(np.linalg.norm(self.pos - epos)) < Params.EXIT_RADIUS:
                self.safe = True
                self.exit_id = k
                break

    # ---------- 切向滑行 ----------
    def _try_slide(self, env, target):
        direction = target - self.pos
        if np.linalg.norm(direction) < 1e-6:
            return None
        d = direction / np.linalg.norm(direction)
        for angle_deg in (30, -30, 60, -60, 90, -90):
            ang = angle_deg * pi / 180
            ca, sa = cos(ang), sin(ang)
            rot = np.array([ca * d[0] - sa * d[1],
                            sa * d[0] + ca * d[1]])
            cand = self.pos + rot * np.linalg.norm(direction)
            if not env.is_obstacle(cand[0], cand[1]):
                return cand
        for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            cx, cy = self.pos[0] + ox, self.pos[1] + oy
            if not env.is_obstacle(cx, cy):
                return np.array([cx, cy])
        return None

    # ---------- 合力 ----------
    def _compute_force(self, env, others):
        b = self.behavior

        # 目标吸引
        d_goal = self.goal - self.pos
        dg = float(np.linalg.norm(d_goal))
        f_goal = d_goal / dg if dg > 1e-6 else np.zeros(2)

        # 火灾排斥（8 邻居采样）
        f_fire = np.zeros(2)
        for (ox, oy) in [(-10, 0), (10, 0), (0, -10), (0, 10),
                         (-7, -7), (-7, 7), (7, -7), (7, 7)]:
            tx, ty = int(self.pos[0] + ox), int(self.pos[1] + oy)
            if 0 <= tx < env.w and 0 <= ty < env.h:
                h = env.heat[ty, tx]
                s = env.smoke[ty, tx]
                danger = h + s
                if danger > 0.15:
                    dist = sqrt(ox * ox + oy * oy)
                    weight = danger * (1.0 - dist / 15.0)
                    f_fire -= np.array([ox, oy]) / dist * weight

        w_fire = max(0.1,
                     1.0 + Params.BETA_RISK * b.risk_aversion
                     - Params.ALPHA_PANIC * b.panic)

        # 障碍排斥
        f_obs = np.zeros(2)
        for (ox, oy) in [(-4, 0), (4, 0), (0, -4), (0, 4),
                         (-3, -3), (-3, 3), (3, -3), (3, 3)]:
            tx, ty = int(self.pos[0] + ox), int(self.pos[1] + oy)
            if 0 <= tx < env.w and 0 <= ty < env.h \
                    and env.obstacle[ty, tx] == 1:
                f_obs -= np.array([ox, oy]) / 4.0

        # 协同力
        f_coop = np.zeros(2)
        for other in others:
            if other.id == self.id or other.safe or other.dead:
                continue
            d_vec = self.pos - other.pos
            d = float(np.linalg.norm(d_vec))
            if d < 1e-6 or d > Params.COOP_RADIUS:
                continue
            repel = d_vec / d * (Params.COOP_RADIUS - d) / Params.COHESION_MAX
            time_overlap = 0.0
            for k, tp in enumerate(other.trajectory[-Params.TIME_HORIZON:]):
                dt = sqrt((tp[0] - self.pos[0]) ** 2
                          + (tp[1] - self.pos[1]) ** 2)
                if dt < Params.COOP_RADIUS:
                    time_overlap += (Params.COOP_RADIUS - dt) \
                                    / Params.COOP_RADIUS / (k + 1)
            f_coop += repel * (1.0 + 0.5 * time_overlap)
            if b.cohesion > 0.5:
                f_coop -= d_vec / d * (b.cohesion - 0.5) * 0.3

        F = (Params.W_GOAL * f_goal
             + Params.W_FIRE * w_fire * f_fire
             + Params.W_OBS * f_obs
             + Params.W_COOP * f_coop)
        return F


# =====================================================================
# 仿真主控
# =====================================================================
class Simulation:
    def __init__(self):
        self.env = FireEnvironment()
        self.agents = []
        self.step_count = 0
        self.history = []

    def build_scene(self):
        # 障碍物布置（不影响双出口路径抉择）
        self.env.add_obstacle(400, 300, 200, 200)   # 中央大障碍
        self.env.add_obstacle(200, 600, 150, 120)   # 左下
        self.env.add_obstacle(900, 500, 120, 150)   # 靠右下（挡B路径但不封死）

    def _is_valid_spawn(self, x, y):
        if self.env.is_obstacle(x, y):
            return False
        for (fx, fy, _) in self.env.ignition:
            if sqrt((x - fx) ** 2 + (y - fy) ** 2) < 100:
                return False
        return True

    def spawn_agents(self, n=10):
        self.agents = []
        tries = 0
        while len(self.agents) < n and tries < 800:
            tries += 1
            x = random.randint(50, 330)
            y = random.randint(50, Params.HEIGHT - 80)
            if not self._is_valid_spawn(x, y):
                continue
            self.agents.append(Agent(len(self.agents), x, y, Params.EXITS))

    def run(self, max_steps=Params.MAX_STEPS, log_every=100):
        # 双火源，位于通往出口 A 的路径上
        self.env.add_ignition(750, 280)
        self.env.add_ignition(870, 180)

        print("=" * 78)
        print(" v2.2 双出口仿真 —— 行为参数驱动的出口选择与协同疏散")
        print("=" * 78)
        print(" 出口 A=(1150,80) 近但危险 | 出口 B=(1150,720) 远但安全")
        print("-" * 78)
        print(f"{'step':>6} | {'safe':>4} | {'dead':>4} | "
              f"{'A':>3} | {'B':>3} | {'avg_h':>7} | {'avg_p':>7} | {'fires':>6}")
        print("-" * 78)

        for step in range(max_steps):
            self.step_count = step
            self.env.step()

            for a in self.agents:
                a.step(self.env, self.agents, step)

            stats = self.collect_stats()
            self.history.append(stats)

            if step % log_every == 0:
                print(f"{step:>6} | {stats['safe']:>4} | {stats['dead']:>4} | "
                      f"{stats['exit_A']:>3} | {stats['exit_B']:>3} | "
                      f"{stats['avg_health']:>7.1f} | "
                      f"{stats['avg_panic']:>7.3f} | "
                      f"{stats['fires']:>6}")

            if stats['safe'] + stats['dead'] >= len(self.agents):
                print(f"\n全员终止于 step = {step}")
                break

        print("-" * 78)
        self.export_csv()

    def collect_stats(self):
        total = len(self.agents)
        safe = sum(1 for a in self.agents if a.safe)
        dead = sum(1 for a in self.agents if a.dead)
        alive = [a for a in self.agents if not a.safe and not a.dead]

        exit_A = sum(1 for a in self.agents if a.safe and a.exit_id == 0)
        exit_B = sum(1 for a in self.agents if a.safe and a.exit_id == 1)

        avg_h = float(np.mean([a.behavior.health for a in alive])) if alive else 0.0
        avg_p = float(np.mean([a.behavior.panic for a in alive])) if alive else 0.0
        fires = int(np.sum(self.env.heat > 0.5))

        return {
            'step': self.step_count,
            'total': total, 'safe': safe, 'dead': dead,
            'exit_A': exit_A, 'exit_B': exit_B,
            'avg_health': avg_h, 'avg_panic': avg_p, 'fires': fires,
        }

    def export_csv(self):
        # 主历史
        with open("sim_history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=['step', 'total', 'safe', 'dead',
                               'exit_A', 'exit_B',
                               'avg_health', 'avg_panic', 'fires'])
            w.writeheader()
            w.writerows(self.history)

        # 个体终态
        with open("agents_final.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(['id', 'panic', 'health', 'speed',
                        'risk_aversion', 'cohesion',
                        'safe', 'dead', 'exit_chosen',
                        'x', 'y', 'dist_to_chosen_exit'])
            for a in self.agents:
                b = a.behavior
                exit_pos = a.exits[a.exit_id] if a.exit_id is not None else a.exits[1]
                d = float(np.linalg.norm(a.pos - exit_pos))
                w.writerow([
                    a.id,
                    f"{b.panic:.3f}", f"{b.health:.2f}",
                    f"{b.speed:.3f}", f"{b.risk_aversion:.3f}",
                    f"{b.cohesion:.3f}",
                    int(a.safe), int(a.dead),
                    a.exit_id if a.exit_id is not None else -1,
                    f"{a.pos[0]:.1f}", f"{a.pos[1]:.1f}",
                    f"{d:.1f}",
                ])

        # 出口选择汇总
        n_A = sum(1 for a in self.agents if a.exit_id == 0)
        n_B = sum(1 for a in self.agents if a.exit_id == 1)
        print(f"\n出口选择汇总：")
        print(f"  出口 A (近/危险)：{n_A} 人")
        print(f"  出口 B (远/安全)：{n_B} 人")
        print(f"\n已导出：sim_history.csv / agents_final.csv")


# =====================================================================
def main():
    random.seed(42)
    np.random.seed(42)
    sim = Simulation()
    sim.build_scene()
    sim.spawn_agents(10)
    sim.run()


if __name__ == "__main__":
    main()