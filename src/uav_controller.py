# ==========================================
# uav_controller.py - 无人机群编队与人工势场
# ==========================================
import numpy as np
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE_DIR, 'config') not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, 'config'))
import config

class UAVFleet:
    def __init__(self, num_uavs):
        self.num_uavs = num_uavs
        
        # 状态数组
        self.xs = np.zeros(num_uavs, dtype=np.float32)
        self.ys = np.zeros(num_uavs, dtype=np.float32)
        self.vxs = np.zeros(num_uavs, dtype=np.float32)
        self.vys = np.zeros(num_uavs, dtype=np.float32)
        
        # 记录历史轨迹
        self.history = [[] for _ in range(num_uavs)]

        # 计算边缘引力衰减蒙版
        margin = max(1, int(config.RADAR_RADIUS_GRIDS) * 2)
        penalty = config.EDGE_PENALTY_FACTOR

        weight_x = np.ones(config.GRID_W, dtype=np.float32)
        weight_x[:margin] = np.linspace(penalty, 1.0, margin)
        weight_x[-margin:] = np.linspace(1.0, penalty, margin)

        weight_y = np.ones(config.GRID_H, dtype=np.float32)
        weight_y[:margin] = np.linspace(penalty, 1.0, margin)
        weight_y[-margin:] = np.linspace(1.0, penalty, margin)

        # 生成2D蒙版
        self.edge_penalty_mask = np.outer(weight_y, weight_x)
        self._init_positions()

    def _init_positions(self):
        """根据 config 切入点初始化多机位置"""
        entry_pts = config.UAV_ENTRY_POINTS
        r_limit = config.RADAR_RADIUS_GRIDS
        for i in range(self.num_uavs):
            # 如果无人机多于切入点，循环使用切入点，并加上一点微小偏移防重叠
            pt = entry_pts[i % len(entry_pts)]
            self.xs[i] = pt[0] + np.random.uniform(-1, 1)
            self.ys[i] = pt[1] + np.random.uniform(-1, 1)
            self.xs[i] = np.clip(self.xs[i], r_limit, config.GRID_W - 1 - r_limit)
            self.ys[i] = np.clip(self.ys[i], r_limit, config.GRID_H - 1 - r_limit)
            self.history[i].append((self.xs[i], self.ys[i]))

    def calculate_apf_and_move(self, prob_map):
        """核心：基于人工势场法计算航向并移动"""
        speed_grids = config.UAV_STEP_DIST_KM / config.GRID_RES_KM
        prob_map_2d = prob_map.reshape((config.GRID_H, config.GRID_W))
        perceived_prob_map = prob_map_2d * self.edge_penalty_mask

        # 创建全图坐标网格
        Y, X = np.mgrid[0:config.GRID_H, 0:config.GRID_W]
        # 生成 Voronoi 映射图
        distances = np.zeros((self.num_uavs, config.GRID_H, config.GRID_W), dtype=np.float32)
        for i in range(self.num_uavs):
            distances[i] = (X - self.xs[i]) ** 2 + (Y - self.ys[i]) ** 2
        voronoi_map = np.argmin(distances, axis=0)

        max_idx = np.argmax(prob_map)
        target_y = max_idx // config.GRID_W
        target_x = max_idx % config.GRID_W

        for i in range(self.num_uavs):
            # --- 提取当前无人机的专属目标 ---
            my_mask = (voronoi_map == i)
            my_prob_map = np.where(my_mask, perceived_prob_map, -1.0)
            max_idx = np.argmax(my_prob_map)
            target_y = max_idx // config.GRID_W
            target_x = max_idx % config.GRID_W

            # --- 计算引力 ---
            dx_att = target_x - self.xs[i]
            dy_att = target_y - self.ys[i]
            dist_att = math.hypot(dx_att, dy_att) + 1e-5

            # 如果已经到达高概率区中心，可以随机盘旋
            if dist_att < config.RADAR_RADIUS_GRIDS:
                vec_att_x = np.random.uniform(-1, 1)
                vec_att_y = np.random.uniform(-1, 1)
            else:
                vec_att_x = dx_att / dist_att
                vec_att_y = dy_att / dist_att

            # --- 计算斥力 (多机防碰撞与阵型展开) ---
            vec_rep_x = 0.0
            vec_rep_y = 0.0
            for j in range(self.num_uavs):
                if i == j: continue

                dx_rep = self.xs[i] - self.xs[j]
                dy_rep = self.ys[i] - self.ys[j]
                dist_rep = math.hypot(dx_rep, dy_rep) + 1e-5

                if dist_rep < config.REPEL_DISTANCE_GRIDS:
                    force = (1.0 / dist_rep - 1.0 / config.REPEL_DISTANCE_GRIDS) ** 2
                    vec_rep_x += (dx_rep / dist_rep) * force
                    vec_rep_y += (dy_rep / dist_rep) * force

            # ---------------- 计算惯性与合力 ----------------
            # 合力 = 惯性 + 引力*系数 + 斥力*系数
            new_vx = (config.W_INERTIA * self.vxs[i] +
                      config.C_ATTRACT * vec_att_x +
                      config.C_REPEL * vec_rep_x)
            new_vy = (config.W_INERTIA * self.vys[i] +
                      config.C_ATTRACT * vec_att_y +
                      config.C_REPEL * vec_rep_y)

            # 归一化为巡航速度
            mag = math.hypot(new_vx, new_vy) + 1e-5
            self.vxs[i] = (new_vx / mag) * speed_grids
            self.vys[i] = (new_vy / mag) * speed_grids

            # ---------------- 移动与边界限制 ----------------
            self.xs[i] += self.vxs[i]
            self.ys[i] += self.vys[i]

            # 保留一个雷达半径的缓冲距离
            r_limit = config.RADAR_RADIUS_GRIDS
            self.xs[i] = np.clip(self.xs[i], r_limit, config.GRID_W - 1 - r_limit)
            self.ys[i] = np.clip(self.ys[i], r_limit, config.GRID_H - 1 - r_limit)

            self.history[i].append((self.xs[i], self.ys[i]))

    def get_positions(self):
        """返回所有无人机的当前坐标集"""
        return self.xs, self.ys