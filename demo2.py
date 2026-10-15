# -*- coding: utf-8 -*-
"""
基于个体行为参数的多智能体协同疏散路径规划方法 v2.0
============================================================
设计原则：
  1. 纯数值仿真，仅依赖 numpy + 标准库
  2. 去除公知算法：不使用 A*、不使用 HSV 阈值、不使用卷积核扩散
     —— 规划采用「行为参数调制的连续力场」
     —— 扩散采用「元胞邻居规则」
  3. 输出 CSV，便于专利实验复现

核心创新：
  (1) 个体行为参数向量 B_i = (panic, health, speed, risk_aversion, cohesion)
  (2) 行为调制权重 w_fire(B_i) = 1 + β·risk_aversion - α·panic
      高恐慌 → 更"敢"穿越危险区；高风险厌恶 → 更保守
  (3) 协同排斥力基于邻居预测轨迹的时间-空间重叠度
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
    DT = 0.1                       # 仿真时间步
    MAX_STEPS = 2000

    # ---- 火灾 / 烟雾元胞动力学 ----
    HEAT_DIFFUSION = 0.15
    SMOKE_DIFFUSION = 0.20
    FIRE_SPREAD_BASE = 0.15
    FIRE_DECAY = 0.998
    FIRE_BLOCK = 0.70
    SMOKE_SOURCE = 0.30
    WIND_CHANGE_PROB = 0.02
    WIND_MAX = 2.0

    # ---- 行为参数范围 ----
    PANIC_MIN, PANIC_MAX = 0.05, 0.95
    HEALTH_INIT = 100.0
    SPEED_MIN, SPEED_MAX = 1.5, 3.0
    RISK_AVERSION_MIN, RISK_AVERSION_MAX = 0.30, 1.20
    COHESION_MIN, COHESION_MAX = 0.0, 1.0

    # ---- 恐慌动力学 ----
    PANIC_GROWTH_RATE = 0.01
    PANIC_DECAY_RATE = 0.002
    PANIC_HEALTH_THRESHOLD = 30.0

    # ---- 行为调制系数（核心创新系数）----
    ALPHA_PANIC = 0.6              # 恐慌对火排斥力的削弱系数
    BETA_RISK = 0.5                # 风险厌恶对火排斥力的增强系数

    # ---- 力场权重 ----
    W_GOAL = 1.0
    W_FIRE = 3.0
    W_OBS = 5.0
    W_COOP = 1.5

    # ---- 协同 ----
    COOP_RADIUS = 40.0
    TIME_HORIZON = 10

    # ---- 伤害 ----
    HEAT_DAMAGE_RATE = 2.0
    SMOKE_DAMAGE_RATE = 1.0

    # ---- 目标区 ----
    SAFE_ZONE = (1100, 700)
    SAFE_RADIUS = 25.0


# =====================================================================
# 个体行为参数
# =====================================================================
class BehaviorParams:
    """
    个体行为参数向量 B_i。
    专利核心：本向量驱动所有后续力场权重、伤害响应与协同交互。
    """
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
        """恐慌动态：健康低 + 危险高 → 恐慌增长；否则缓慢衰减"""
        if self.health < Params.PANIC_HEALTH_THRESHOLD:
            self.panic = min(1.0, self.panic + Params.PANIC_GROWTH_RATE)
        self.panic = min(1.0, self.panic
                         + 0.002 * danger_level)
        self.panic = max(0.0, self.panic - Params.PANIC_DECAY_RATE)

    def effective_speed(self):
        """实际移动速度 = 期望速度 × 健康衰减因子"""
        h_factor = 0.5 + 0.5 * (self.health / Params.HEALTH_INIT)
        return self.speed * h_factor


# =====================================================================
# 火灾 / 烟雾环境（元胞规则扩散，不依赖 cv2）
# =====================================================================
class FireEnvironment:
    def __init__(self, w=Params.WIDTH, h=Params.HEIGHT):
        self.w, self.h = w, h
        self.heat = np.zeros((h, w), dtype=np.float32)
        self.smoke = np.zeros((h, w), dtype=np.float32)
        self.obstacle = np.zeros((h, w), dtype=np.uint8)
        self.wind = np.array([1.5, 0.5], dtype=float)
        self.ignition = []

    # ---------- 场景构建 ----------
    def add_ignition(self, x, y, intensity=1.0):
        x, y = int(x), int(y)
        if 0 <= x < self.w and 0 <= y < self.h:
            self.heat[y, x] = intensity
            self.ignition.append((x, y, intensity))

    def add_obstacle(self, x, y, w, h):
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.w, x + w), min(self.h, y + h)
        self.obstacle[y1:y2, x1:x2] = 1

    # ---------- 每步更新 ----------
    def step(self):
        self._step_heat()
        self._step_smoke()
        if random.random() < Params.WIND_CHANGE_PROB:
            ang = random.uniform(0, 2 * pi)
            self.wind = np.array([cos(ang) * Params.WIND_MAX,
                                  sin(ang) * Params.WIND_MAX])

    def _step_heat(self):
        """元胞规则：4 邻域平均 + 风偏漂移 + 随机蔓延 + 火源稳定"""
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

        # 随机蔓延（在已着火点附近产生新火点）
        if np.any(self.heat > Params.FIRE_BLOCK):
            for _ in range(20):
                cy = random.randint(0, self.h - 1)
                cx = random.randint(0, self.w - 1)
                if self.heat[cy, cx] > Params.FIRE_BLOCK:
                    dy = random.choice([-2, -1, 1, 2])
                    dx = random.choice([-2, -1, 1, 2])
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < self.h and 0 <= nx < self.w \
                            and self.obstacle[ny, nx] == 0:
                        p = Params.FIRE_SPREAD_BASE * \
                            (1.0 + (dx * wx + dy * wy) * 0.1)
                        if random.random() < p:
                            self.heat[ny, nx] = min(
                                1.0, self.heat[ny, nx] + 0.4)

        # 障碍物不传播
        self.heat[self.obstacle == 1] = 0.0

        # 持续点燃
        for x, y, intensity in self.ignition:
            self.heat[y, x] = max(self.heat[y, x], intensity)

        np.clip(self.heat, 0.0, 1.0, out=self.heat)

    def _step_smoke(self):
        src = self.heat > Params.SMOKE_SOURCE
        self.smoke[src] = np.minimum(1.0, self.smoke[src] + 0.05)

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

    # ---------- 采样 ----------
    def sample(self, x, y):
        ix, iy = int(x), int(y)
        if 0 <= ix < self.w and 0 <= iy < self.h:
            return float(self.heat[iy, ix]), float(self.smoke[iy, ix])
        return 0.0, 0.0


# =====================================================================
# 智能体
# =====================================================================
class Agent:
    """
    智能体在连续空间中移动：
      1. 目标吸引力 F_goal
      2. 火灾排斥力 F_fire（梯度 + 行为调制）
      3. 障碍物排斥力 F_obs
      4. 协同力 F_coop（基于邻居预测轨迹时间重叠）
    """

    def __init__(self, aid, x, y, goal):
        self.id = aid
        self.pos = np.array([x, y], dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.behavior = BehaviorParams()
        self.velocity = np.zeros(2, dtype=float)
        self.safe = False
        self.dead = False
        self.trajectory = [(float(x), float(y))]

    # ---------- 单步 ----------
    def step(self, env, others):
        if self.safe or self.dead:
            return

        h_loc, s_loc = env.sample(self.pos[0], self.pos[1])
        danger = min(1.0, h_loc + s_loc)

        # 伤害与恐慌更新
        dmg = h_loc * Params.HEAT_DAMAGE_RATE + s_loc * Params.SMOKE_DAMAGE_RATE
        self.behavior.health = max(0.0, self.behavior.health - dmg)
        self.behavior.update_panic(danger)

        if self.behavior.health <= 0.0:
            self.dead = True
            return

        # 计算合力
        F = self._compute_force(env, others)
        f_norm = float(np.linalg.norm(F))
        if f_norm > 1e-6:
            direction = F / f_norm
        else:
            direction = np.zeros(2)

        # 速度平滑（带一阶惯性）
        desired_v = direction * self.behavior.effective_speed()
        self.velocity = 0.5 * self.velocity + 0.5 * desired_v

        # 位置更新
        new_pos = self.pos + self.velocity * Params.DT
        new_pos[0] = float(np.clip(new_pos[0], 1, env.w - 2))
        new_pos[1] = float(np.clip(new_pos[1], 1, env.h - 2))

        # 障碍物硬约束（回退到相邻空格）
        ix, iy = int(new_pos[0]), int(new_pos[1])
        if env.obstacle[iy, ix] == 1:
            fixed = False
            for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                tx, ty = ix + ox, iy + oy
                if 0 <= tx < env.w and 0 <= ty < env.h \
                        and env.obstacle[ty, tx] == 0:
                    new_pos[0] = tx
                    new_pos[1] = ty
                    fixed = True
                    break
            if not fixed and len(self.trajectory) >= 2:
                new_pos = np.array(self.trajectory[-2])

        self.pos = new_pos

        # 记录轨迹
        self.trajectory.append((float(self.pos[0]), float(self.pos[1])))
        if len(self.trajectory) > Params.TIME_HORIZON:
            self.trajectory.pop(0)

        # 到达安全区
        if float(np.linalg.norm(self.pos - self.goal)) < Params.SAFE_RADIUS:
            self.safe = True

    # ---------- 合力计算 ----------
    def _compute_force(self, env, others):
        b = self.behavior

        # --- (1) 目标吸引 ---
        d_goal = self.goal - self.pos
        dg = float(np.linalg.norm(d_goal))
        f_goal = d_goal / dg if dg > 1e-6 else np.zeros(2)

        # --- (2) 火灾梯度排斥（中心差分）---
        eps = 3
        h_l, s_l = env.sample(self.pos[0] - eps, self.pos[1])
        h_r, s_r = env.sample(self.pos[0] + eps, self.pos[1])
        h_u, s_u = env.sample(self.pos[0], self.pos[1] - eps)
        h_d, s_d = env.sample(self.pos[0], self.pos[1] + eps)

        gx = ((h_r + s_r) - (h_l + s_l)) / (2 * eps)
        gy = ((h_d + s_d) - (h_u + s_u)) / (2 * eps)
        f_fire = -np.array([gx, gy])

        # 行为调制：风险厌恶增强，恐慌削弱
        w_fire = max(0.1,
                     1.0 + Params.BETA_RISK * b.risk_aversion
                     - Params.ALPHA_PANIC * b.panic)

        # --- (3) 障碍物排斥（直接邻居检查）---
        f_obs = np.zeros(2)
        for ox, oy in [(-5, 0), (5, 0), (0, -5), (0, 5),
                       (-4, -4), (-4, 4), (4, -4), (4, 4)]:
            tx = int(self.pos[0] + ox)
            ty = int(self.pos[1] + oy)
            if 0 <= tx < env.w and 0 <= ty < env.h \
                    and env.obstacle[ty, tx] == 1:
                f_obs -= np.array([ox, oy]) / 5.0

        # --- (4) 协同力 ---
        f_coop = np.zeros(2)
        for other in others:
            if other.id == self.id or other.safe or other.dead:
                continue
            d_vec = self.pos - other.pos
            d = float(np.linalg.norm(d_vec))
            if d < 1e-6 or d > Params.COOP_RADIUS:
                continue

            # 基础排斥：距离越近越强
            repel = d_vec / d * (Params.COOP_RADIUS - d) / Params.COOP_RADIUS

            # 时间重叠：检查邻居预测轨迹上各点到自身的距离
            time_overlap = 0.0
            for k, tp in enumerate(other.trajectory[-Params.TIME_HORIZON:]):
                dt = sqrt((tp[0] - self.pos[0]) ** 2
                          + (tp[1] - self.pos[1]) ** 2)
                if dt < Params.COOP_RADIUS:
                    time_overlap += (Params.COOP_RADIUS - dt) \
                                    / Params.COOP_RADIUS / (k + 1)

            f_coop += repel * (1.0 + 0.5 * time_overlap)

            # 高凝聚力个体对邻近个体有轻微吸引
            if b.cohesion > 0.5:
                f_coop -= d_vec / d * (b.cohesion - 0.5) * 0.3

        # --- (5) 合成 ---
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

    # ---------- 场景 ----------
    def build_scene(self):
        self.env.add_obstacle(100, 100, 400, 300)
        self.env.add_obstacle(200, 150, 50, 200)
        self.env.add_obstacle(300, 200, 80, 80)
        self.env.add_obstacle(150, 250, 60, 60)

    def spawn_agents(self, n=8):
        self.agents = []
        for i in range(n):
            x = random.randint(50, Params.WIDTH // 2)
            y = random.randint(50, Params.HEIGHT - 150)
            self.agents.append(Agent(i, x, y, Params.SAFE_ZONE))

    # ---------- 主循环 ----------
    def run(self, max_steps=Params.MAX_STEPS, log_every=100):
        self.env.add_ignition(600, 400)
        self.env.add_ignition(700, 350)

        print("=" * 72)
        print(" v2.0 纯数值仿真 —— 行为参数驱动的多智能体协同疏散")
        print("=" * 72)
        header = (f"{'step':>6} | {'safe':>4} | {'dead':>4} | "
                  f"{'avg_h':>7} | {'avg_p':>7} | {'fires':>6}")
        print(header)
        print("-" * 72)

        for step in range(max_steps):
            self.step_count = step

            self.env.step()

            for a in self.agents:
                a.step(self.env, self.agents)

            stats = self.collect_stats()
            self.history.append(stats)

            if step % log_every == 0:
                print(f"{step:>6} | {stats['safe']:>4} | {stats['dead']:>4} | "
                      f"{stats['avg_health']:>7.1f} | "
                      f"{stats['avg_panic']:>7.3f} | "
                      f"{stats['fires']:>6}")

            if stats['safe'] + stats['dead'] >= len(self.agents):
                print(f"\n全员终止于 step = {step}")
                break

        print("-" * 72)
        self.export_csv()

    # ---------- 统计 ----------
    def collect_stats(self):
        total = len(self.agents)
        safe = sum(1 for a in self.agents if a.safe)
        dead = sum(1 for a in self.agents if a.dead)
        alive = [a for a in self.agents if not a.safe and not a.dead]

        avg_h = float(np.mean([a.behavior.health for a in alive])) if alive else 0.0
        avg_p = float(np.mean([a.behavior.panic for a in alive])) if alive else 0.0
        fires = int(np.sum(self.env.heat > 0.5))

        return {
            'step': self.step_count,
            'total': total,
            'safe': safe,
            'dead': dead,
            'avg_health': avg_h,
            'avg_panic': avg_p,
            'fires': fires,
        }

    # ---------- 导出 ----------
    def export_csv(self):
        with open("sim_history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=['step', 'total', 'safe', 'dead',
                               'avg_health', 'avg_panic', 'fires'])
            w.writeheader()
            w.writerows(self.history)

        with open("agents_final.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(['id', 'panic', 'health', 'speed',
                        'risk_aversion', 'cohesion',
                        'safe', 'dead', 'x', 'y'])
            for a in self.agents:
                b = a.behavior
                w.writerow([
                    a.id,
                    f"{b.panic:.3f}",
                    f"{b.health:.2f}",
                    f"{b.speed:.3f}",
                    f"{b.risk_aversion:.3f}",
                    f"{b.cohesion:.3f}",
                    int(a.safe), int(a.dead),
                    f"{a.pos[0]:.1f}", f"{a.pos[1]:.1f}",
                ])

        print("\n已导出：")
        print("  - sim_history.csv   (每帧统计)")
        print("  - agents_final.csv  (最终个体状态)")


# =====================================================================
def main():
    random.seed(42)
    np.random.seed(42)

    sim = Simulation()
    sim.build_scene()
    sim.spawn_agents(8)
    sim.run()


if __name__ == "__main__":
    main()