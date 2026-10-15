# -*- coding: utf-8 -*-
"""
基于个体行为参数的多智能体协同疏散路径规划方法 v2.1
============================================================
v2.0 -> v2.1 修复清单：
  [F1] DT: 0.1 -> 1.0                         (让智能体真的走起来)
  [F2] 恐慌动力学重写                          (不再无条件衰减到 0)
  [F3] 火焰蔓延改为"从已燃点采样"              (火会真的烧起来)
  [F4] 障碍物排斥改为"前方探针 + 切向滑行"      (不再沿墙卡死)
  [F5] 卡住检测与随机逃逸                      (兜底机制)
  [F6] 目标力权重提高，障碍力权重降低           (避免力平衡陷阱)
  [F7] 起始点避开障碍物                        (防止出生即卡)
"""

import numpy as np
import random
import csv
from math import sqrt, cos, sin, pi


# =====================================================================
# 参数
# =====================================================================
class Params:
    WIDTH, HEIGHT = 1200, 800
    DT = 1.0                            # [F1] 关键修复
    MAX_STEPS = 2500

    # 火灾元胞动力学
    HEAT_DIFFUSION = 0.15
    SMOKE_DIFFUSION = 0.20
    FIRE_SPREAD_BASE = 0.35             # [F3] 提高蔓延概率
    FIRE_DECAY = 0.999
    FIRE_BLOCK = 0.70
    SMOKE_SOURCE = 0.30
    WIND_CHANGE_PROB = 0.02
    WIND_MAX = 2.0

    # 行为参数
    PANIC_MIN, PANIC_MAX = 0.05, 0.95
    HEALTH_INIT = 100.0
    SPEED_MIN, SPEED_MAX = 4.0, 8.0      # [F1] 匹配 DT=1.0
    RISK_AVERSION_MIN, RISK_AVERSION_MAX = 0.30, 1.20
    COHESION_MIN, COHESION_MAX = 0.0, 1.0

    # [F2] 恐慌动力学
    PANIC_GROWTH_PER_DANGER = 0.02      # 危险贡献
    PANIC_GROWTH_ON_LOW_HEALTH = 0.005  # 健康低时追加
    PANIC_DECAY_RATE = 0.001            # 无危险时缓慢衰减
    PANIC_FLOOR = 0.05                  # [F2] 恐慌下限，永不归零
    PANIC_HEALTH_THRESHOLD = 40.0

    # 行为调制
    ALPHA_PANIC = 0.6
    BETA_RISK = 0.5

    # [F6] 力权重
    W_GOAL = 2.5                         # [F6] 提高
    W_FIRE = 3.0
    W_OBS = 1.5                          # [F6] 降低
    W_COOP = 1.2

    # 协同
    COOP_RADIUS = 40.0
    TIME_HORIZON = 10

    # 伤害
    HEAT_DAMAGE_RATE = 2.0
    SMOKE_DAMAGE_RATE = 1.0

    # [F5] 卡住检测
    STUCK_WINDOW = 40                   # 观察窗口
    STUCK_MIN_PROGRESS = 8.0            # 窗口内最小进展像素
    STUCK_ESCAPE_STEPS = 15             # 逃逸持续步数

    # 目标区
    SAFE_ZONE = (1100, 700)
    SAFE_RADIUS = 30.0


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
        """
        [F2] 重写：
          - 危险 > 阈值 → 恐慌增长
          - 健康低 → 恐慌额外增长
          - 危险为 0 → 缓慢衰减，但保留下限 PANIC_FLOOR
        """
        if danger_level > 0.05:
            self.panic += Params.PANIC_GROWTH_PER_DANGER * danger_level
        else:
            self.panic -= Params.PANIC_DECAY_RATE

        if self.health < Params.PANIC_HEALTH_THRESHOLD:
            self.panic += Params.PANIC_GROWTH_ON_LOW_HEALTH

        # 限制范围 [FLOOR, 1.0]
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
        return 1.0, 1.0  # 越界视为危险

    # ---------- 步进 ----------
    def step(self):
        self._step_heat()
        self._step_smoke()
        if random.random() < Params.WIND_CHANGE_PROB:
            ang = random.uniform(0, 2 * pi)
            self.wind = np.array([cos(ang) * Params.WIND_MAX,
                                  sin(ang) * Params.WIND_MAX])

    def _step_heat(self):
        # 元胞扩散
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

        # [F3] 从已燃点采样蔓延，而不是盲采样全场
        burning = np.argwhere(self.heat > Params.FIRE_BLOCK)
        if len(burning) > 0:
            n_try = min(120, len(burning) * 6)
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
    def __init__(self, aid, x, y, goal):
        self.id = aid
        self.pos = np.array([x, y], dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.behavior = BehaviorParams()
        self.velocity = np.zeros(2, dtype=float)
        self.safe = False
        self.dead = False
        self.trajectory = [(float(x), float(y))]

        # [F5] 卡住检测
        self.progress_window = []           # 每步的到目标距离
        self.stuck_escape_left = 0
        self.escape_dir = np.zeros(2)

    # ---------- 主步进 ----------
    def step(self, env, others):
        if self.safe or self.dead:
            return

        h_loc, s_loc = env.sample(self.pos[0], self.pos[1])
        danger = min(1.0, h_loc + s_loc)

        dmg = h_loc * Params.HEAT_DAMAGE_RATE + s_loc * Params.SMOKE_DAMAGE_RATE
        self.behavior.health = max(0.0, self.behavior.health - dmg)
        self.behavior.update_panic(danger)

        if self.behavior.health <= 0.0:
            self.dead = True
            return

        # 计算力
        F = self._compute_force(env, others)

        # [F5] 逃逸附加力
        if self.stuck_escape_left > 0:
            F = F + self.escape_dir * 3.0
            self.stuck_escape_left -= 1

        f_norm = float(np.linalg.norm(F))
        direction = F / f_norm if f_norm > 1e-6 else np.zeros(2)

        desired_v = direction * self.behavior.effective_speed()
        # 一阶惯性：更平滑
        self.velocity = 0.6 * self.velocity + 0.4 * desired_v

        new_pos = self.pos + self.velocity * Params.DT
        new_pos[0] = float(np.clip(new_pos[0], 2, env.w - 3))
        new_pos[1] = float(np.clip(new_pos[1], 2, env.h - 3))

        # 硬碰撞：撞到障碍物就沿切向滑行 [F4]
        if env.is_obstacle(new_pos[0], new_pos[1]):
            slide = self._try_slide(env, new_pos)
            if slide is not None:
                new_pos = slide
            else:
                new_pos = self.pos  # 完全无法移动

        self.pos = new_pos

        # 轨迹记录
        self.trajectory.append((float(self.pos[0]), float(self.pos[1])))
        if len(self.trajectory) > Params.TIME_HORIZON:
            self.trajectory.pop(0)

        # [F5] 卡住检测
        d_goal = float(np.linalg.norm(self.pos - self.goal))
        self.progress_window.append(d_goal)
        if len(self.progress_window) > Params.STUCK_WINDOW:
            self.progress_window.pop(0)
            if self.progress_window[0] - self.progress_window[-1] < Params.STUCK_MIN_PROGRESS:
                # 没进展 → 触发逃逸
                ang = random.uniform(0, 2 * pi)
                self.escape_dir = np.array([cos(ang), sin(ang)])
                self.stuck_escape_left = Params.STUCK_ESCAPE_STEPS

        # 到达安全区
        if d_goal < Params.SAFE_RADIUS:
            self.safe = True

    # ---------- 切向滑行 [F4] ----------
    def _try_slide(self, env, target):
        """在障碍物表面找一个能前进的切向位置"""
        direction = target - self.pos
        if np.linalg.norm(direction) < 1e-6:
            return None
        d = direction / np.linalg.norm(direction)

        # 尝试 ±30°、±60°、±90° 旋转
        for angle_deg in (30, -30, 60, -60, 90, -90):
            ang = angle_deg * pi / 180
            ca, sa = cos(ang), sin(ang)
            rot = np.array([ca * d[0] - sa * d[1],
                            sa * d[0] + ca * d[1]])
            candidate = self.pos + rot * np.linalg.norm(direction)
            if not env.is_obstacle(candidate[0], candidate[1]):
                return candidate

        # 尝试 8 邻域单步
        for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            cx = self.pos[0] + ox
            cy = self.pos[1] + oy
            if not env.is_obstacle(cx, cy):
                return np.array([cx, cy])
        return None

    # ---------- 合力 ----------
    def _compute_force(self, env, others):
        b = self.behavior

        # (1) 目标吸引
        d_goal = self.goal - self.pos
        dg = float(np.linalg.norm(d_goal))
        f_goal = d_goal / dg if dg > 1e-6 else np.zeros(2)

        # (2) 火灾排斥 —— 直接采样 8 邻居
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

        # 行为调制
        w_fire = max(0.1,
                     1.0 + Params.BETA_RISK * b.risk_aversion
                     - Params.ALPHA_PANIC * b.panic)

        # (3) 障碍物排斥 —— 前方探针
        f_obs = np.zeros(2)
        for (ox, oy) in [(-4, 0), (4, 0), (0, -4), (0, 4),
                         (-3, -3), (-3, 3), (3, -3), (3, 3)]:
            tx, ty = int(self.pos[0] + ox), int(self.pos[1] + oy)
            if 0 <= tx < env.w and 0 <= ty < env.h \
                    and env.obstacle[ty, tx] == 1:
                f_obs -= np.array([ox, oy]) / 4.0

        # (4) 协同力
        f_coop = np.zeros(2)
        for other in others:
            if other.id == self.id or other.safe or other.dead:
                continue
            d_vec = self.pos - other.pos
            d = float(np.linalg.norm(d_vec))
            if d < 1e-6 or d > Params.COOP_RADIUS:
                continue
            repel = d_vec / d * (Params.COOP_RADIUS - d) / Params.COOP_RADIUS
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
        # [F7] 障碍物精简，避免出生即卡
        self.env.add_obstacle(300, 200, 200, 180)   # 中央一个中等障碍
        self.env.add_obstacle(600, 100, 120, 120)   # 右上小障碍
        self.env.add_obstacle(600, 500, 150, 120)   # 右下小障碍

    def _is_valid_spawn(self, x, y):
        # 出生点不能落在障碍上，也不能离火源太近
        if self.env.is_obstacle(x, y):
            return False
        for (fx, fy, _) in self.env.ignition:
            if sqrt((x - fx) ** 2 + (y - fy) ** 2) < 80:
                return False
        return True

    def spawn_agents(self, n=8):
        self.agents = []
        tries = 0
        while len(self.agents) < n and tries < 500:
            tries += 1
            x = random.randint(50, 350)     # 集中左侧区域
            y = random.randint(50, Params.HEIGHT - 100)
            if not self._is_valid_spawn(x, y):
                continue
            self.agents.append(Agent(len(self.agents), x, y, Params.SAFE_ZONE))

    def run(self, max_steps=Params.MAX_STEPS, log_every=100):
        self.env.add_ignition(700, 400)
        self.env.add_ignition(850, 300)

        print("=" * 76)
        print(" v2.1 纯数值仿真 —— 行为参数驱动的多智能体协同疏散")
        print("=" * 76)
        print(f"{'step':>6} | {'safe':>4} | {'dead':>4} | "
              f"{'avg_h':>7} | {'avg_p':>7} | {'fires':>6}")
        print("-" * 76)

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

        print("-" * 76)
        self.export_csv()

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
            'total': total, 'safe': safe, 'dead': dead,
            'avg_health': avg_h, 'avg_panic': avg_p, 'fires': fires,
        }

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
                        'safe', 'dead', 'x', 'y', 'dist_to_goal'])
            for a in self.agents:
                b = a.behavior
                d = float(np.linalg.norm(a.pos - a.goal))
                w.writerow([
                    a.id,
                    f"{b.panic:.3f}", f"{b.health:.2f}",
                    f"{b.speed:.3f}", f"{b.risk_aversion:.3f}",
                    f"{b.cohesion:.3f}",
                    int(a.safe), int(a.dead),
                    f"{a.pos[0]:.1f}", f"{a.pos[1]:.1f}",
                    f"{d:.1f}",
                ])

        print("\n已导出：sim_history.csv / agents_final.csv")


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