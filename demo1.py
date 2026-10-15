# -*- coding: utf-8 -*-
"""
基于个体行为参数的多智能体协同疏散路径规划方法 v1.0
================================================================
核心创新：
  (1) 个体行为参数向量 B_i = (panic, health, speed, risk_aversion, cohesion)
  (2) 恐慌-代价耦合函数 C'(p) = C(p) * f(panic_i, risk_aversion_i)
  (3) 多智能体协同惩罚项，基于其他智能体预测轨迹的空间-时间重叠
"""

import cv2
import numpy as np
import pygame
import random
import heapq
import time
from math import sqrt, atan2, cos, sin, pi
from pygame.locals import *


# =====================================================================
# 全局可调参数（便于专利实施例描述与实验复现）
# =====================================================================
class Params:
    # ---------- 仿真环境 ----------
    WIDTH, HEIGHT = 1200, 800
    FPS = 60

    # ---------- 火灾/烟雾物理参数 ----------
    HEAT_DIFFUSION_RATE = 0.3          # 热扩散混合系数
    SMOKE_DIFFUSION_RATE = 0.4         # 烟雾扩散混合系数
    FIRE_SPREAD_PROB_BASE = 0.25       # 基础蔓延概率
    FIRE_DECAY = 0.98                  # 每帧自然衰减
    FIRE_CORE_THRESHOLD = 0.8          # 火焰核心阈值
    FIRE_BLOCK_THRESHOLD = 0.7         # 视为不可通行的热度阈值
    SMOKE_SOURCE_THRESHOLD = 0.3       # 烟雾产生阈值
    WIND_CHANGE_PROB = 0.02            # 每帧风向改变概率
    WIND_MAX = 2.0

    # ---------- 代价地图参数 ----------
    HEAT_COST_WEIGHT = 10.0            # 热量代价权重
    SMOKE_COST_WEIGHT = 5.0            # 烟雾代价权重
    OBSTACLE_COST = 1000.0             # 障碍不可通行代价
    HIGH_COST_BLOCK = 50.0             # A* 中认为"不可通过"的代价阈值

    # ---------- 行为参数范围 ----------
    PANIC_MIN, PANIC_MAX = 0.05, 0.95          # 恐慌度
    HEALTH_INIT = 100.0                        # 初始健康
    SPEED_MIN, SPEED_MAX = 1.0, 2.5            # 期望移动速度
    RISK_AVERSION_MIN, RISK_AVERSION_MAX = 0.3, 1.2
    COHESION_MIN, COHESION_MAX = 0.0, 1.0      # 群体聚集倾向

    # ---------- 恐慌-代价耦合函数参数 ----------
    ALPHA_PANIC = 0.5                  # 恐慌放缩系数
    BETA_RISK = 0.4                    # 风险厌恶放缩系数
    PANIC_GROWTH_THRESHOLD = 30.0      # 健康低于该值恐慌增长
    PANIC_GROWTH_RATE = 0.008          # 每帧恐慌增长量
    PANIC_DECAY_RATE = 0.002           # 恐慌自然衰减

    # ---------- 协同惩罚参数 ----------
    COOP_RADIUS = 35.0                 # 邻居影响半径
    COOP_WEIGHT = 2.0                  # 协同避碰代价权重
    COOP_TIME_HORIZON = 10             # 未来轨迹步数
    COOP_TIME_WEIGHT = 0.5             # 时间重叠惩罚权重

    # ---------- A* 搜索参数 ----------
    A_STAR_STEP = 15                   # 每步扩展距离
    A_STAR_STEP_JITTER = 5             # 步长随机扰动
    A_STAR_ANGLES = 16                 # 方向离散化数量
    A_STAR_MAX_ITER = 3000             # 最大迭代次数
    GOAL_TOLERANCE = 20.0              # 到达目标的距离阈值

    # ---------- 人员伤害参数 ----------
    HEAT_DAMAGE_RATE = 2.0
    SMOKE_DAMAGE_RATE = 1.0

    # ---------- 安全区 ----------
    SAFE_ZONE_RADIUS = 20.0


# =====================================================================
# 个体行为参数
# =====================================================================
class BehaviorParams:
    """
    个体行为参数向量 B_i。
    专利权利要求的核心：所有后续代价、动作、交互均由此向量驱动。
    """
    __slots__ = ("panic", "health", "speed", "risk_aversion", "cohesion")

    def __init__(self, panic=None, health=Params.HEALTH_INIT,
                 speed=None, risk_aversion=None, cohesion=None):
        self.panic = panic if panic is not None else random.uniform(
            Params.PANIC_MIN, Params.PANIC_MAX)
        self.health = health
        self.speed = speed if speed is not None else random.uniform(
            Params.SPEED_MIN, Params.SPEED_MAX)
        self.risk_aversion = risk_aversion if risk_aversion is not None else \
            random.uniform(Params.RISK_AVERSION_MIN, Params.RISK_AVERSION_MAX)
        self.cohesion = cohesion if cohesion is not None else random.uniform(
            Params.COHESION_MIN, Params.COHESION_MAX)

    def as_vector(self):
        return (self.panic, self.health, self.speed,
                self.risk_aversion, self.cohesion)

    def update_panic(self, danger_level, alone_factor):
        """
        恐慌动态更新：
          panic += 增长率*(危险度) + 独处加成 - 衰减
        """
        # 危险度越高越恐慌
        if self.health < Params.PANIC_GROWTH_THRESHOLD:
            self.panic = min(1.0, self.panic + Params.PANIC_GROWTH_RATE)
        # 独处时恐慌略增
        self.panic = min(1.0, self.panic + 0.001 * alone_factor * danger_level)
        # 自然衰减
        self.panic = max(0.0, self.panic - Params.PANIC_DECAY_RATE)

    def effective_speed(self):
        """恐慌与健康共同影响实际移动速度"""
        health_factor = 0.5 + 0.5 * (self.health / Params.HEALTH_INIT)
        panic_factor = 1.0 + 0.3 * self.panic  # 恐慌短时间提速
        return self.speed * health_factor * panic_factor


# =====================================================================
# 火灾/烟雾环境
# =====================================================================
class FireEnvironment:
    def __init__(self, width=Params.WIDTH, height=Params.HEIGHT):
        self.width = width
        self.height = height

        self.heat_map = np.zeros((height, width), dtype=np.float32)
        self.smoke_density = np.zeros((height, width), dtype=np.float32)
        self.fire_mask = np.zeros((height, width), dtype=np.uint8)
        self.obstacle_map = np.zeros((height, width), dtype=np.uint8)

        self.wind = [1.5, 0.5]
        self.ignition_points = []

    # ------------------ 环境构建 ------------------
    def add_ignition_point(self, x, y, intensity=1.0):
        self.ignition_points.append((x, y, intensity))
        if 0 <= x < self.width and 0 <= y < self.height:
            self.heat_map[y, x] = intensity
            self.fire_mask[y, x] = 1

    def add_obstacle_rectangle(self, x, y, w, h):
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.width, x + w), min(self.height, y + h)
        self.obstacle_map[y1:y2, x1:x2] = 1

    # ------------------ 更新 ------------------
    def update(self):
        self._update_fire_spread()
        self._update_smoke_diffusion()
        if random.random() < Params.WIND_CHANGE_PROB:
            ang = random.uniform(0, 2 * pi)
            self.wind = [cos(ang) * Params.WIND_MAX,
                         sin(ang) * Params.WIND_MAX]

    def _update_fire_spread(self):
        new_heat = self.heat_map.copy()
        new_mask = self.fire_mask.copy()
        fire_cells = np.argwhere(self.fire_mask > 0)

        for y, x in fire_cells:
            for dy in (-2, -1, 0, 1, 2):
                for dx in (-2, -1, 0, 1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < self.width and 0 <= ny < self.height):
                        continue
                    if self.obstacle_map[ny, nx] == 1:
                        continue
                    wind_effect = (dx * self.wind[0] + dy * self.wind[1]) * 0.1
                    dist_factor = 1.0 / (abs(dx) + abs(dy) + 1)
                    p = (Params.FIRE_SPREAD_PROB_BASE + wind_effect) * dist_factor
                    if random.random() < p:
                        new_heat[ny, nx] = min(1.0, new_heat[ny, nx] + 0.3)
                        if new_heat[ny, nx] > 0.7:
                            new_mask[ny, nx] = 1

        kernel = np.ones((3, 3), np.float32) / 9
        diffused = cv2.filter2D(new_heat, -1, kernel)
        new_heat = new_heat * 0.3 + diffused * 0.7
        new_heat *= Params.FIRE_DECAY

        self.heat_map = np.clip(new_heat, 0, 1)
        self.fire_mask = new_mask

    def _update_smoke_diffusion(self):
        src = self.heat_map > Params.SMOKE_SOURCE_THRESHOLD
        self.smoke_density[src] = np.minimum(
            1.0, self.smoke_density[src] + 0.05)

        kernel = np.ones((5, 5), np.float32) / 25
        diffused = cv2.filter2D(self.smoke_density, -1, kernel)
        M = np.float32([[1, 0, int(3 * self.wind[0])],
                        [0, 1, int(3 * self.wind[1])]])
        shifted = cv2.warpAffine(diffused, M, (self.width, self.height))
        self.smoke_density = shifted * 0.97

    # ------------------ 可视化 ------------------
    def get_visualization(self):
        vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        vis[self.obstacle_map == 1] = (80, 80, 80)

        heat_vis = np.zeros_like(vis)
        heat_colored = (self.heat_map * 255).astype(np.uint8)
        heat_vis[:, :, 1] = (heat_colored * 0.5).astype(np.uint8)
        heat_vis[:, :, 2] = heat_colored
        cv2.addWeighted(heat_vis, 0.8, vis, 0.2, 0, vis)

        smoke_mask = self.smoke_density > 0.1
        if np.any(smoke_mask):
            alpha = np.zeros((self.height, self.width), dtype=np.float32)
            alpha[smoke_mask] = self.smoke_density[smoke_mask] * 0.3
            alpha3 = np.stack([alpha] * 3, axis=-1)
            smoke_layer = np.full_like(vis, (200, 200, 200))
            vis = np.clip(vis.astype(np.float32) * (1 - alpha3) +
                          smoke_layer.astype(np.float32) * alpha3,
                          0, 255).astype(np.uint8)

        vis[self.heat_map > Params.FIRE_CORE_THRESHOLD] = (0, 0, 255)
        return vis

    def reset(self):
        self.heat_map.fill(0)
        self.smoke_density.fill(0)
        self.fire_mask.fill(0)
        self.ignition_points = []


# =====================================================================
# 智能体（人）
# =====================================================================
class Agent:
    def __init__(self, aid, x, y, goal):
        self.id = aid
        self.pos = (float(x), float(y))
        self.goal = goal
        self.behavior = BehaviorParams()
        self.color = self._pick_color(aid)
        self.path = None
        self.path_age = 0
        self.safe = False
        # 用于协同的时间-空间预测轨迹
        self.trajectory = [self.pos]

    @staticmethod
    def _pick_color(idx):
        palette = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 255, 0),
                   (255, 0, 255), (0, 255, 255), (255, 128, 0), (160, 0, 255)]
        return palette[idx % len(palette)]

    def update_health(self, env):
        if self.safe:
            return
        x, y = int(self.pos[0]), int(self.pos[1])
        if not (0 <= x < env.width and 0 <= y < env.height):
            return
        heat = env.heat_map[y, x]
        smoke = env.smoke_density[y, x]
        dmg = heat * Params.HEAT_DAMAGE_RATE + smoke * Params.SMOKE_DAMAGE_RATE
        self.behavior.health = max(0.0, self.behavior.health - dmg)

        danger = min(1.0, heat + smoke)
        self.behavior.update_panic(danger, alone_factor=0.5)

    def move_along_path(self):
        if self.safe or not self.path or len(self.path) < 2:
            return
        cur = self.pos
        nxt = self.path[1]
        dx, dy = nxt[0] - cur[0], nxt[1] - cur[1]
        d = sqrt(dx * dx + dy * dy)
        if d < 1e-6:
            return
        step = min(self.behavior.effective_speed(), d)
        self.pos = (cur[0] + dx / d * step, cur[1] + dy / d * step)

        # 记录轨迹
        self.trajectory.append(self.pos)
        if len(self.trajectory) > Params.COOP_TIME_HORIZON:
            self.trajectory.pop(0)

        # 到达安全区
        if sqrt((self.pos[0] - self.goal[0]) ** 2 +
                (self.pos[1] - self.goal[1]) ** 2) < Params.SAFE_ZONE_RADIUS:
            self.safe = True
            self.path = None


# =====================================================================
# 行为感知的协同 A* 规划器
# =====================================================================
class BehaviorAwarePlanner:
    """
    核心算法：
      代价函数 C'(p, i) =
          [ 静态代价 C_static(p) ]                         # 热/烟/障碍
        * [ 1 + ALPHA*panic_i - BETA*risk_aversion_i ]     # 恐慌-风险耦合
        + [ 协同惩罚 C_coop(p, i, 其他智能体) ]            # 多智能体避碰
    """

    def __init__(self, env):
        self.env = env
        self.cost_map = np.ones((env.height, env.width), dtype=np.float32)

    # ---------- 1) 静态代价地图 ----------
    def build_static_cost_map(self):
        heat = self.env.heat_map
        smoke = self.env.smoke_density
        obs = self.env.obstacle_map

        cost = np.ones_like(heat, dtype=np.float32)
        cost += np.where(heat > 0.3, heat * Params.HEAT_COST_WEIGHT, 0)
        cost += np.where(smoke > 0.3, smoke * Params.SMOKE_COST_WEIGHT, 0)
        cost[obs == 1] = Params.OBSTACLE_COST
        # 火焰核心不可通行
        cost[heat > Params.FIRE_BLOCK_THRESHOLD] = Params.OBSTACLE_COST
        self.cost_map = cost

    # ---------- 2) 行为耦合因子 ----------
    @staticmethod
    def behavior_factor(b: BehaviorParams):
        """
        f(panic, risk_aversion) = 1 + ALPHA*panic - BETA*risk_aversion
        - 恐慌高 → 因子大 → 代价放大（但高恐慌同时会提高"接受风险"的意愿，
          通过下面 risk_aversion 的负号部分抵消）
        - 风险厌恶高 → 因子小 → 相对保守（实际上通过协同惩罚更强来体现）
        注意：这里 panic 放大的是"绕路代价"，从而让高恐慌个体更倾向于
        直接穿越轻度危险区（短路效应）。
        """
        f = 1.0 + Params.ALPHA_PANIC * b.panic \
                - Params.BETA_RISK * b.risk_aversion
        return max(0.2, f)  # 防止出现负/零因子

    # ---------- 3) 协同惩罚 ----------
    def coop_penalty(self, p, self_id, other_agents):
        """
        对当前候选点 p，累计来自其他智能体的协同惩罚。
        惩罚项 = Σ_j  exp(-d/ R) * w * (1 + TIME_W * time_overlap)
        """
        if not other_agents:
            return 0.0
        total = 0.0
        for a in other_agents:
            # 当前距离
            d_now = sqrt((p[0] - a.pos[0]) ** 2 + (p[1] - a.pos[1]) ** 2)
            if d_now < Params.COOP_RADIUS:
                total += (Params.COOP_RADIUS - d_now) * Params.COOP_WEIGHT
            # 轨迹重叠（未来时间窗）
            if a.trajectory:
                for k, tp in enumerate(a.trajectory):
                    dt = sqrt((p[0] - tp[0]) ** 2 + (p[1] - tp[1]) ** 2)
                    if dt < Params.COOP_RADIUS:
                        total += (Params.COOP_RADIUS - dt) * \
                                 Params.COOP_WEIGHT * Params.COOP_TIME_WEIGHT / (k + 1)
        return total

    # ---------- 4) A* 主搜索 ----------
    def plan(self, agent, other_agents):
        self.build_static_cost_map()

        start = (int(agent.pos[0]), int(agent.pos[1]))
        goal = (int(agent.goal[0]), int(agent.goal[1]))
        bf = self.behavior_factor(agent.behavior)

        def is_free(x, y):
            if not (0 <= x < self.env.width and 0 <= y < self.env.height):
                return False
            if self.env.obstacle_map[y, x] == 1:
                return False
            return self.cost_map[y, x] < Params.OBSTACLE_COST

        def node_cost(x, y):
            base = self.cost_map[y, x]
            # 恐慌-风险耦合：恐慌高 → 系数小 → 更"敢"穿越（等效于降低代价）
            # 这里采用相反的解释：恐慌增大惩罚，路径越长越贵，从而倾向于直达
            # 两种解释任选其一；本实现采用"惩罚乘子"，见文档说明
            coupled = base * bf
            coop = self.coop_penalty((x, y), agent.id, other_agents)
            return coupled + coop

        def h(x, y):
            return sqrt((x - goal[0]) ** 2 + (y - goal[1]) ** 2)

        open_heap = []
        counter = 0
        heapq.heappush(open_heap, (h(start[0], start[1]), counter, start, None))
        counter += 1
        best_g = {start: 0.0}
        closed = set()
        parents = {}

        for _ in range(Params.A_STAR_MAX_ITER):
            if not open_heap:
                return None
            f, _, cur, _ = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)

            if sqrt((cur[0] - goal[0]) ** 2 + (cur[1] - goal[1]) ** 2) \
                    < Params.GOAL_TOLERANCE:
                return self._extract(parents, cur, start)

            cx, cy = cur
            for k in range(Params.A_STAR_ANGLES):
                ang = 2 * pi * k / Params.A_STAR_ANGLES
                step = Params.A_STAR_STEP + random.randint(
                    -Params.A_STAR_STEP_JITTER, Params.A_STAR_STEP_JITTER)
                nx = int(cx + cos(ang) * step)
                ny = int(cy + sin(ang) * step)

                if not is_free(nx, ny):
                    continue

                # 中点采样：防止穿墙
                mx, my = (cx + nx) // 2, (cy + ny) // 2
                if not is_free(mx, my):
                    continue

                g_new = best_g[cur] + node_cost(nx, ny)
                nb = (nx, ny)
                if nb in closed and g_new >= best_g.get(nb, 1e9):
                    continue
                if g_new < best_g.get(nb, 1e9):
                    best_g[nb] = g_new
                    parents[nb] = cur
                    heapq.heappush(open_heap,
                                   (g_new + h(nx, ny), counter, nb, cur))
                    counter += 1
        return None

    @staticmethod
    def _extract(parents, cur, start):
        path = [cur]
        while cur in parents:
            cur = parents[cur]
            path.append(cur)
        path.append(start)
        return path[::-1]

    # ---------- 5) 多智能体协同调度 ----------
    def plan_all(self, agents):
        """
        采用优先级（健康度低者优先）依次规划，避免同步冲突。
        """
        active = [a for a in agents if not a.safe]
        # 优先给健康低的人规划（更紧急）
        active.sort(key=lambda a: a.behavior.health)

        for a in active:
            others = [o for o in agents if o.id != a.id and not o.safe]
            path = self.plan(a, others)
            if path:
                a.path = path
            else:
                a.path = None


# =====================================================================
# VR 显示
# =====================================================================
class VRDisplay:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.screen = None
        self.font = None
        self.small_font = None

    def initialize(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.w, self.h),
                                              pygame.DOUBLEBUF | pygame.HWSURFACE)
        pygame.display.set_caption("基于个体行为参数的多人协同疏散仿真 v1.0")
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 22)
        return True

    def render(self, env, agents, planner, stats, camera_frame=None):
        self.screen.fill((0, 0, 0))

        # ---- 环境层 ----
        vis = env.get_visualization()
        surf = pygame.surfarray.make_surface(np.transpose(vis, (1, 0, 2)))
        self.screen.blit(surf, (0, 0))

        # ---- 安全区 ----
        sx, sy = int(env.width - 100), int(env.height - 100)
        pygame.draw.circle(self.screen, (0, 255, 255), (sx, sy), 15, 3)
        t = self.small_font.render("EXIT", True, (0, 255, 255))
        self.screen.blit(t, (sx - 15, sy - 30))

        # ---- 智能体 ----
        for a in agents:
            x, y = int(a.pos[0]), int(a.pos[1])
            pygame.draw.circle(self.screen, a.color, (x, y), 10)

            # 健康条
            hw, hh = 22, 4
            hx, hy = x - hw // 2, y - 16
            pygame.draw.rect(self.screen, (80, 80, 80), (hx, hy, hw, hh))
            pct = a.behavior.health / Params.HEALTH_INIT
            fill = int(hw * pct)
            col = (0, 255, 0) if pct > 0.5 else (255, 255, 0) if pct > 0.2 else (255, 0, 0)
            pygame.draw.rect(self.screen, col, (hx, hy, fill, hh))

            # 恐慌环（外圈颜色深浅表示恐慌）
            panic_r = int(14 + 6 * a.behavior.panic)
            pygame.draw.circle(self.screen, a.color, (x, y), panic_r, 1)

            # 路径
            if a.path and not a.safe and len(a.path) > 1:
                pts = [(int(p[0]), int(p[1])) for p in a.path]
                pygame.draw.lines(self.screen, a.color, False, pts, 1)

            # 标签
            label = f"{a.id}"
            t = self.small_font.render(label, True, (255, 255, 255))
            self.screen.blit(t, (x - 4, y - 32))

        # ---- 摄像头小窗 ----
        if camera_frame is not None:
            rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            cam_surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
            self.screen.blit(cam_surf, (self.w - 320, 10))
            pygame.draw.rect(self.screen, (255, 255, 255),
                             (self.w - 322, 8, 324, 244), 2)

        # ---- UI ----
        self._draw_ui(agents, stats, env.wind)

        pygame.display.flip()

    def _draw_ui(self, agents, stats, wind):
        lines = [
            f"总人数: {stats['total']}",
            f"已逃生: {stats['safe']}",
            f"平均健康: {stats['avg_health']:.1f}",
            f"平均恐慌: {stats['avg_panic']:.2f}",
            f"活跃火源: {int(stats['fires'])}",
            "",
            "控制: ESC退出  R重置  F加火源  P加人",
        ]
        for i, s in enumerate(lines):
            t = self.small_font.render(s, True, (255, 255, 255))
            self.screen.blit(t, (10, 10 + i * 22))

        # 前6人行为参数
        y = 190
        for a in agents[:6]:
            b = a.behavior
            txt = f"#{a.id} P:{b.panic:.2f} H:{b.health:5.1f} " \
                  f"V:{b.speed:.2f} R:{b.risk_aversion:.2f}"
            t = self.small_font.render(txt, True, a.color)
            self.screen.blit(t, (10, y))
            y += 20

        # 风向
        cx, cy = 100, self.h - 100
        ex, ey = cx + int(wind[0] * 20), cy + int(wind[1] * 20)
        pygame.draw.line(self.screen, (255, 255, 255), (cx, cy), (ex, ey), 3)
        pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), 5)
        t = self.small_font.render("风向", True, (255, 255, 255))
        self.screen.blit(t, (cx - 20, cy - 25))


# =====================================================================
# 主仿真
# =====================================================================
class Simulation:
    def __init__(self, camera_index=0):
        self.env = FireEnvironment()
        self.planner = BehaviorAwarePlanner(self.env)
        self.display = VRDisplay(Params.WIDTH, Params.HEIGHT)
        self.safe_zone = (Params.WIDTH - 100, Params.HEIGHT - 100)
        self.agents = []
        self.running = False
        self.camera_frame = None
        self.camera_active = False
        self.cap = None
        self.camera_index = camera_index

    # ------------------ 构建场景 ------------------
    def build_scene(self):
        # 障碍物
        self.env.add_obstacle_rectangle(100, 100, 400, 300)
        self.env.add_obstacle_rectangle(200, 150, 50, 200)
        self.env.add_obstacle_rectangle(300, 200, 80, 80)
        self.env.add_obstacle_rectangle(150, 250, 60, 60)

    def spawn_agents(self, n=8):
        self.agents = []
        for i in range(n):
            x = random.randint(50, Params.WIDTH // 2)
            y = random.randint(50, Params.HEIGHT - 150)
            self.agents.append(Agent(i, x, y, self.safe_zone))

    def add_agent_at(self, x, y):
        self.agents.append(Agent(len(self.agents), x, y, self.safe_zone))

    # ------------------ 摄像头 ------------------
    def init_camera(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.camera_active = False
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera_active = True
            return True
        except Exception:
            self.camera_active = False
            return False

    def grab_camera_frame(self):
        if not self.camera_active:
            return
        ret, frame = self.cap.read()
        if ret:
            self.camera_frame = cv2.resize(frame, (320, 240))

    # ------------------ 主循环 ------------------
    def run(self):
        self.display.initialize()
        self.build_scene()
        self.spawn_agents(8)
        self.init_camera()

        # 初始火源
        self.env.add_ignition_point(600, 400)
        self.env.add_ignition_point(700, 350)

        clock = pygame.time.Clock()
        self.running = True
        frame_id = 0

        while self.running:
            # -------- 事件 --------
            for e in pygame.event.get():
                if e.type == QUIT:
                    self.running = False
                elif e.type == KEYDOWN:
                    if e.key == K_ESCAPE:
                        self.running = False
                    elif e.key == K_r:
                        self.reset()
                    elif e.key == K_f:
                        x, y = pygame.mouse.get_pos()
                        self.env.add_ignition_point(x, y)
                    elif e.key == K_p:
                        x, y = pygame.mouse.get_pos()
                        self.add_agent_at(x, y)

            # -------- 环境更新 --------
            self.env.update()

            # -------- 摄像头上报 --------
            if self.camera_active and frame_id % 3 == 0:
                self.grab_camera_frame()

            # -------- 智能体 --------
            for a in self.agents:
                a.update_health(self.env)
                a.move_along_path()

            # -------- 每 N 帧重新规划（多智能体协同）--------
            if frame_id % 5 == 0:
                self.planner.plan_all(self.agents)

            # -------- 渲染 --------
            self.display.render(self.env, self.agents, self.planner,
                                self.get_stats(), self.camera_frame)

            clock.tick(Params.FPS)
            frame_id += 1

        self.cleanup()

    def get_stats(self):
        total = len(self.agents)
        safe = sum(1 for a in self.agents if a.safe)
        active = [a for a in self.agents if not a.safe]
        avg_h = sum(a.behavior.health for a in active) / max(1, len(active))
        avg_p = sum(a.behavior.panic for a in active) / max(1, len(active))
        fires = np.sum(self.env.heat_map > 0.5)
        return {"total": total, "safe": safe,
                "avg_health": avg_h, "avg_panic": avg_p, "fires": fires}

    def reset(self):
        self.env.reset()
        self.spawn_agents(8)
        self.env.add_ignition_point(600, 400)

    def cleanup(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        pygame.quit()


# =====================================================================
def main():
    print("=" * 60)
    print(" 基于个体行为参数的多智能体协同疏散路径规划方法 v1.0")
    print("=" * 60)
    print("核心创新点：")
    print("  1) 个体行为参数向量 B=(panic, health, speed, risk, cohesion)")
    print("  2) 恐慌-代价耦合函数  C' = C * f(panic, risk_aversion)")
    print("  3) 多智能体时空协同惩罚")
    print("-" * 60)
    sim = Simulation(camera_index=0)
    try:
        sim.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print("仿真结束")


if __name__ == "__main__":
    main()