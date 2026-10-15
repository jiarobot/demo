# -*- coding: utf-8 -*-
"""
基于个体行为参数的多智能体协同疏散路径规划方法 v2.5
============================================================
v2.4 -> v2.5 关键修复：
  [P1] 火源改为半径 12 px 的圆形火团（替代单像素）
  [P2] 出口危险采样改为 numpy 切片 ±12 px 最大池化
  [P3] EXIT_RISK_SCALE 900->650, RESELECT 20->15
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

    HEAT_DIFFUSION = 0.08
    SMOKE_DIFFUSION = 0.18
    FIRE_SPREAD_BASE = 0.12
    FIRE_DECAY = 0.9995
    FIRE_BLOCK = 0.70
    SMOKE_SOURCE = 0.30
    WIND_CHANGE_PROB = 0.02
    WIND_MAX = 2.0

    FIRE_SPREAD_N_TRY_MAX = 60
    FIRE_SPREAD_N_TRY_PER_BURN = 3

    PANIC_MIN, PANIC_MAX = 0.05, 0.95
    HEALTH_INIT = 100.0
    SPEED_MIN, SPEED_MAX = 3.0, 6.0
    RISK_AVERSION_MIN, RISK_AVERSION_MAX = 0.30, 1.20
    COHESION_MIN, COHESION_MAX = 0.0, 1.0

    # 恐慌动力学（比例衰减）
    PANIC_GROWTH_PER_DANGER = 0.020
    PANIC_GROWTH_PER_GLOBAL = 0.010
    PANIC_DECAY_RATE = 0.012
    PANIC_GROWTH_ON_LOW_HEALTH = 0.005
    PANIC_FLOOR = 0.05
    PANIC_HEALTH_THRESHOLD = 40.0
    GLOBAL_DANGER_FIRE_COUNT = 400.0

    # 力场
    ALPHA_PANIC = 0.6
    BETA_RISK = 0.5

    W_GOAL = 2.5
    W_FIRE = 3.0
    W_OBS = 1.5
    W_COOP = 1.2

    COOP_RADIUS = 40.0
    TIME_HORIZON = 10

    HEAT_DAMAGE_RATE = 2.0
    SMOKE_DAMAGE_RATE = 1.0

    STUCK_WINDOW = 40
    STUCK_MIN_PROGRESS = 8.0
    STUCK_ESCAPE_STEPS = 15

    # ---------- 出口几何 ----------
    EXITS = [
        (600, 60),      # A: 近/危险
        (1150, 700),    # B: 远/安全
    ]
    EXIT_RADIUS = 30.0

    # ---------- 出口选择（v2.5 重调） ----------
    EXIT_DIST_PANIC_GAIN = 0.20
    EXIT_RISK_PANIC_DAMP = 0.40
    EXIT_RISK_SCALE = 650.0            # [P3] 900 -> 650
    EXIT_RISK_SAMPLES = 40
    EXIT_SAMPLE_HALF_WIN = 12          # [P2] ±12 px 切片

    EXIT_RESELECT_INTERVAL = 15        # [P3] 20 -> 15
    EXIT_SWITCH_HYSTERESIS = 0.95


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

    def update_panic(self, danger_level, global_danger=0.0):
        growth = (Params.PANIC_GROWTH_PER_GLOBAL * global_danger
                  + Params.PANIC_GROWTH_PER_DANGER * danger_level)
        self.panic += growth
        self.panic -= Params.PANIC_DECAY_RATE * self.panic
        if self.health < Params.PANIC_HEALTH_THRESHOLD:
            self.panic += Params.PANIC_GROWTH_ON_LOW_HEALTH
        self.panic = float(np.clip(self.panic, Params.PANIC_FLOOR, 1.0))

    def effective_speed(self):
        h_factor = 0.6 + 0.4 * (self.health / Params.HEALTH_INIT)
        return self.speed * h_factor


# =====================================================================
# 火灾环境（含 [P1] 火团创建）
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
            self.heat[y, x] = max(self.heat[y, x], intensity)
            self.ignition.append((x, y, intensity))

    # ---------- [P1] 新增：圆形火团 ----------
    def add_fire_cluster(self, cx, cy, radius=12, intensity=1.0):
        """在 (cx,cy) 处点燃半径 radius 的圆形火团"""
        cx, cy = int(cx), int(cy)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    x, y = cx + dx, cy + dy
                    if 0 <= x < self.w and 0 <= y < self.h \
                            and self.obstacle[y, x] == 0:
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

    def local_danger_max(self, x, y, half_win=12):
        """[P2] 在 (x,y) 的 ±half_win 邻域内取 heat+smoke 最大值"""
        ix = int(max(half_win, min(self.w - 1 - half_win, x)))
        iy = int(max(half_win, min(self.h - 1 - half_win, y)))
        h_patch = self.heat[iy - half_win:iy + half_win + 1,
                            ix - half_win:ix + half_win + 1]
        s_patch = self.smoke[iy - half_win:iy + half_win + 1,
                             ix - half_win:ix + half_win + 1]
        return float(np.max(h_patch + s_patch))

    def global_danger(self):
        n_burn = int(np.sum(self.heat > 0.5))
        return min(1.0, n_burn / Params.GLOBAL_DANGER_FIRE_COUNT)

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
# 智能体
# =====================================================================
class Agent:
    def __init__(self, aid, x, y, exits, env=None):
        self.id = aid
        self.pos = np.array([x, y], dtype=float)
        self.exits = [np.array(e, dtype=float) for e in exits]
        self.exit_id = None
        self.goal = np.array(self.exits[1], dtype=float)
        self.behavior = BehaviorParams()
        self.velocity = np.zeros(2, dtype=float)
        self.safe = False
        self.dead = False
        self.trajectory = [(float(x), float(y))]

        self.progress_window = []
        self.stuck_escape_left = 0
        self.escape_dir = np.zeros(2)
        self.n_exit_switches = 0

        # 初始选择基于真实火源
        self.exit_id = self._choose_best_exit(env, force=True)
        self.goal = self.exits[self.exit_id]

    # =================================================================
    # [P2] 出口代价：numpy 切片 ±12 px 最大池化
    # =================================================================
    def _exit_cost(self, env, exit_idx):
        b = self.behavior
        target = self.exits[exit_idx]
        d = float(np.linalg.norm(self.pos - target))

        R_max = 0.0
        if env is not None:
            n = Params.EXIT_RISK_SAMPLES
            half_win = Params.EXIT_SAMPLE_HALF_WIN
            for k in range(1, n):
                t = k / n
                sx = self.pos[0] + t * (target[0] - self.pos[0])
                sy = self.pos[1] + t * (target[1] - self.pos[1])
                danger_here = env.local_danger_max(sx, sy, half_win)
                if danger_here > R_max:
                    R_max = danger_here

        lam = b.risk_aversion * (1.0 - Params.EXIT_RISK_PANIC_DAMP * b.panic)
        lam = max(0.05, lam)
        dist_gain = 1.0 + Params.EXIT_DIST_PANIC_GAIN * b.panic
        return d * dist_gain + lam * R_max * Params.EXIT_RISK_SCALE

    def _choose_best_exit(self, env, force=False):
        if env is None and not force:
            return self.exit_id
        if env is None:
            return 1
        costs = [self._exit_cost(env, k) for k in range(len(self.exits))]
        best = int(np.argmin(costs))
        if self.exit_id is not None and best != self.exit_id:
            cur = costs[self.exit_id]
            if costs[best] < cur * Params.EXIT_SWITCH_HYSTERESIS:
                return best
            return self.exit_id
        return best

    # =================================================================
    def step(self, env, others, step_count):
        if self.safe or self.dead:
            return

        h_loc, s_loc = env.sample(self.pos[0], self.pos[1])
        danger = min(1.0, h_loc + s_loc)

        dmg = h_loc * Params.HEAT_DAMAGE_RATE + s_loc * Params.SMOKE_DAMAGE_RATE
        self.behavior.health = max(0.0, self.behavior.health - dmg)

        gdanger = env.global_danger()
        self.behavior.update_panic(danger, gdanger)

        if self.behavior.health <= 0.0:
            self.dead = True
            return

        if step_count % Params.EXIT_RESELECT_INTERVAL == 0:
            new_id = self._choose_best_exit(env)
            if new_id != self.exit_id:
                self.exit_id = new_id
                self.goal = self.exits[new_id]
                self.n_exit_switches += 1

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

        d_goal = float(np.linalg.norm(self.pos - self.goal))
        self.progress_window.append(d_goal)
        if len(self.progress_window) > Params.STUCK_WINDOW:
            self.progress_window.pop(0)
            if self.progress_window[0] - self.progress_window[-1] \
                    < Params.STUCK_MIN_PROGRESS:
                ang = random.uniform(0, 2 * pi)
                self.escape_dir = np.array([cos(ang), sin(ang)])
                self.stuck_escape_left = Params.STUCK_ESCAPE_STEPS

        for k, epos in enumerate(self.exits):
            if float(np.linalg.norm(self.pos - epos)) < Params.EXIT_RADIUS:
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

        return (Params.W_GOAL * f_goal
                + Params.W_FIRE * w_fire * f_fire
                + Params.W_OBS * f_obs
                + Params.W_COOP * f_coop)


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
        self.env.add_obstacle(700, 300, 150, 150)
        self.env.add_obstacle(300, 550, 100, 100)

    def _is_valid_spawn(self, x, y):
        if self.env.is_obstacle(x, y):
            return False
        for (fx, fy, _) in self.env.ignition:
            if sqrt((x - fx) ** 2 + (y - fy) ** 2) < 60:
                return False
        return True

    def spawn_agents(self, n=10):
        self.agents = []
        tries = 0
        while len(self.agents) < n and tries < 1000:
            tries += 1
            x = random.randint(50, 250)
            y = random.randint(50, Params.HEIGHT - 80)
            if not self._is_valid_spawn(x, y):
                continue
            self.agents.append(
                Agent(len(self.agents), x, y, Params.EXITS, env=self.env))

    def run(self, max_steps=Params.MAX_STEPS, log_every=100):
        print("=" * 84)
        print(" v2.5 双出口仿真 —— 火团初始化 + 宽窗危险采样")
        print("=" * 84)
        print(f" 出口 A={Params.EXITS[0]} 近但危险 | "
              f"出口 B={Params.EXITS[1]} 远但安全")
        print(" [P1] 圆形火团  [P2] ±12 px 切片采样  [P3] 系数重调")
        print("-" * 84)
        print(f"{'step':>6} | {'safe':>4} | {'dead':>4} | "
              f"{'A':>3} | {'B':>3} | {'avg_h':>7} | "
              f"{'avg_p':>7} | {'min_p':>6} | {'max_p':>6} | {'fires':>6}")
        print("-" * 84)

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
                      f"{stats['min_panic']:>6.3f} | "
                      f"{stats['max_panic']:>6.3f} | "
                      f"{stats['fires']:>6}")

            if stats['safe'] + stats['dead'] >= len(self.agents):
                print(f"\n全员终止于 step = {step}")
                break

        print("-" * 84)
        self.export_csv()

    def collect_stats(self):
        total = len(self.agents)
        safe = sum(1 for a in self.agents if a.safe)
        dead = sum(1 for a in self.agents if a.dead)
        alive = [a for a in self.agents if not a.safe and not a.dead]
        all_a = self.agents

        exit_A = sum(1 for a in self.agents if a.safe and a.exit_id == 0)
        exit_B = sum(1 for a in self.agents if a.safe and a.exit_id == 1)

        avg_h = float(np.mean([a.behavior.health for a in alive])) if alive else 0.0
        avg_p = float(np.mean([a.behavior.panic for a in all_a]))
        min_p = float(np.min([a.behavior.panic for a in all_a]))
        max_p = float(np.max([a.behavior.panic for a in all_a]))
        fires = int(np.sum(self.env.heat > 0.5))

        return {
            'step': self.step_count,
            'total': total, 'safe': safe, 'dead': dead,
            'exit_A': exit_A, 'exit_B': exit_B,
            'avg_health': avg_h, 'avg_panic': avg_p,
            'min_panic': min_p, 'max_panic': max_p,
            'fires': fires,
        }

    def export_csv(self):
        with open("sim_history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=['step', 'total', 'safe', 'dead',
                               'exit_A', 'exit_B',
                               'avg_health', 'avg_panic',
                               'min_panic', 'max_panic', 'fires'])
            w.writeheader()
            w.writerows(self.history)

        with open("agents_final.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(['id', 'panic', 'health', 'speed',
                        'risk_aversion', 'cohesion',
                        'safe', 'dead', 'exit_chosen', 'n_switches',
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
                    a.n_exit_switches,
                    f"{a.pos[0]:.1f}", f"{a.pos[1]:.1f}",
                    f"{d:.1f}",
                ])

        self._print_correlation_report()
        print(f"\n已导出：sim_history.csv / agents_final.csv")

    def _print_correlation_report(self):
        print("\n" + "=" * 84)
        print(" 行为参数 vs 出口选择 —— 相关性分析")
        print("=" * 84)

        agents = [a for a in self.agents if a.safe]
        if len(agents) < 4:
            print("样本不足，跳过")
            return

        panics = [a.behavior.panic for a in agents]
        ras = [a.behavior.risk_aversion for a in agents]
        p_med = float(np.median(panics))
        r_med = float(np.median(ras))

        print(f"\n[1] 按 panic 中位数 {p_med:.3f} 分组：")
        for label, sel in [("低 panic", lambda a: a.behavior.panic <= p_med),
                           ("高 panic", lambda a: a.behavior.panic >  p_med)]:
            sub = [a for a in agents if sel(a)]
            nA = sum(1 for a in sub if a.exit_id == 0)
            nB = sum(1 for a in sub if a.exit_id == 1)
            print(f"    {label:10s} | 样本={len(sub):2d} | "
                  f"选 A={nA} | 选 B={nB}")

        print(f"\n[2] 按 risk_aversion 中位数 {r_med:.3f} 分组：")
        for label, sel in [("低风险厌恶", lambda a: a.behavior.risk_aversion <= r_med),
                           ("高风险厌恶", lambda a: a.behavior.risk_aversion >  r_med)]:
            sub = [a for a in agents if sel(a)]
            nA = sum(1 for a in sub if a.exit_id == 0)
            nB = sum(1 for a in sub if a.exit_id == 1)
            print(f"    {label:10s} | 样本={len(sub):2d} | "
                  f"选 A={nA} | 选 B={nB}")

        print(f"\n[3] 四象限（panic × risk_aversion）：")
        print(f"    {'panic':>10s} {'risk_av':>10s} | {'样本':>4s} | "
              f"{'A':>4s} | {'B':>4s}")
        for plo in [True, False]:
            for rlo in [True, False]:
                sub = [a for a in agents
                       if (a.behavior.panic <= p_med) == plo
                       and (a.behavior.risk_aversion <= r_med) == rlo]
                nA = sum(1 for a in sub if a.exit_id == 0)
                nB = sum(1 for a in sub if a.exit_id == 1)
                print(f"    {'低' if plo else '高':>10s} "
                      f"{'低' if rlo else '高':>10s} | "
                      f"{len(sub):>4d} | {nA:>4d} | {nB:>4d}")

        switchers = [a for a in agents if a.n_exit_switches > 0]
        print(f"\n[4] 出口切换：{len(switchers)}/{len(agents)} 人")
        for a in switchers:
            b = a.behavior
            print(f"    id={a.id:>2d} panic={b.panic:.3f} "
                  f"RA={b.risk_aversion:.3f} switches={a.n_exit_switches} "
                  f"→ {'A' if a.exit_id == 0 else 'B'}")


# =====================================================================
def main():
    random.seed(42)
    np.random.seed(42)

    sim = Simulation()
    sim.build_scene()
    # [P1] 用圆形火团替代单像素火源
    sim.env.add_fire_cluster(400, 180, radius=12)
    sim.env.add_fire_cluster(500, 100, radius=12)
    sim.spawn_agents(10)
    sim.run()


if __name__ == "__main__":
    main()