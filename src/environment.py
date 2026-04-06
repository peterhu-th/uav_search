# ==========================================
# environment.py - 环境状态与 C++ 交互
# ==========================================
import ctypes
import numpy as np
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE_DIR, 'config') not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, 'config'))
import config

class Environment:
    def __init__(self):
        # 初始化 1D 概率网格
        self.prob_map = np.zeros(config.TOTAL_GRIDS, dtype=np.float32)
        
        # 加载 C++ DLL
        current_file_path = os.path.abspath(__file__)
        src_dir = os.path.dirname(current_file_path)
        project_root = os.path.dirname(src_dir)
        dll_name = 'bayes_core.dll' if os.name == 'nt' else 'bayes_core.so'
        dll_path = os.path.join(project_root, 'lib', dll_name)

        try:
            if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(os.path.join(project_root, 'lib'))
            self.lib = ctypes.CDLL(dll_path)
        except OSError as e:
            raise RuntimeError(f"无法加载 DLL: {dll_path}\n>>> 操作系统报错: {e} <<<\n")

        # 定义 C++ 函数的参数类型
        float_array_1d = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
        
        self.lib.time_update.argtypes = [
            float_array_1d, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float
        ]
        
        self.lib.measurement_update.argtypes = [
            float_array_1d, ctypes.c_int, ctypes.c_int, 
            float_array_1d, float_array_1d, ctypes.c_int, 
            ctypes.c_float, ctypes.c_float
        ]

        # 初始化目标真实位置和热力图
        self.true_target_x = 0.0
        self.true_target_y = 0.0
        self.target_vx = 0.0
        self.target_vy = 0.0
        self._init_target()

    def _init_target(self):
        """初始化目标状态与贝叶斯先验概率"""
        if config.TARGET_INIT_MODE == 'gaussian':
            cx, cy = config.TARGET_INIT_CENTER
            self.true_target_x, self.true_target_y = float(cx), float(cy)
            # 在中心生成一个高斯分布的初始热力云
            for y in range(config.GRID_H):
                for x in range(config.GRID_W):
                    dist_sq = (x - cx)**2 + (y - cy)**2
                    idx = y * config.GRID_W + x
                    # 初始方差
                    self.prob_map[idx] = math.exp(-dist_sq / (2.0 * (config.GAUSSIAN_SIGMA * 6)**2))
        else:
            # Uniform 完全未知
            self.true_target_x = np.random.uniform(0, config.GRID_W)
            self.true_target_y = np.random.uniform(0, config.GRID_H)
            self.prob_map.fill(1.0)
            
        # 归一化
        self.prob_map /= np.sum(self.prob_map)

    def move_true_target(self, uav_xs=None, uav_ys=None):
        """控制上帝视角的真实目标机动"""
        speed = config.TARGET_STEP_DIST_KM / config.GRID_RES_KM # 每步移动的网格数
        
        if config.TARGET_TRUE_MOTION == 'random':
            angle = np.random.uniform(0, 2 * math.pi)
            self.target_vx = speed * math.cos(angle)
            self.target_vy = speed * math.sin(angle)
            
        elif config.TARGET_TRUE_MOTION == 'straight':
            # 保持初始的随机方向
            if self.target_vx == 0 and self.target_vy == 0:
                angle = np.random.uniform(0, 2 * math.pi)
                self.target_vx = speed * math.cos(angle)
                self.target_vy = speed * math.sin(angle)
                
        elif config.TARGET_TRUE_MOTION == 'evasive' and uav_xs is not None and len(uav_xs) > 0:
            # 躲避最近的无人机
            min_dist = float('inf')
            nearest_uav_idx = -1
            for i in range(len(uav_xs)):
                dist = math.hypot(self.true_target_x - uav_xs[i], self.true_target_y - uav_ys[i])
                if dist < min_dist:
                    min_dist = dist
                    nearest_uav_idx = i
            
            # 如果无人机在 100 km 内 (33个网格)，开始逃逸
            if min_dist < (100.0 / config.GRID_RES_KM):
                dx = self.true_target_x - uav_xs[nearest_uav_idx]
                dy = self.true_target_y - uav_ys[nearest_uav_idx]
                # 归一化并赋予速度
                length = math.hypot(dx, dy) + 1e-5
                self.target_vx = (dx / length) * speed
                self.target_vy = (dy / length) * speed
            else:
                # 安全时随机游走
                angle = np.random.uniform(0, 2 * math.pi)
                self.target_vx = speed * math.cos(angle)
                self.target_vy = speed * math.sin(angle)

        # 更新坐标并限制在边界内 (遇到边界反弹速度)
        self.true_target_x += self.target_vx
        self.true_target_y += self.target_vy

        if self.true_target_x < 0:
            self.true_target_x = -self.true_target_x
            self.target_vx *= -1
        elif self.true_target_x >= config.GRID_W:
            self.true_target_x = 2 * (config.GRID_W - 1) - self.true_target_x
            self.target_vx *= -1

        if self.true_target_y < 0:
            self.true_target_y = -self.true_target_y
            self.target_vy *= -1
        elif self.true_target_y >= config.GRID_H:
            self.true_target_y = 2 * (config.GRID_H - 1) - self.true_target_y
            self.target_vy *= -1

    def time_update_bayes(self):
        """调用 C++ 进行马尔可夫高斯扩散"""
        self.lib.time_update(
            self.prob_map, config.GRID_W, config.GRID_H, 
            ctypes.c_float(config.GAUSSIAN_SIGMA), ctypes.c_float(config.PRUNE_THRESHOLD)
        )

    def measurement_update_bayes(self, uav_xs, uav_ys):
        """调用 C++ 进行雷达扫描区域的概率降温"""
        num_uavs = len(uav_xs)
        if num_uavs == 0:
            return
        
        ux_arr = np.array(uav_xs, dtype=np.float32)
        uy_arr = np.array(uav_ys, dtype=np.float32)
        
        self.lib.measurement_update(
            self.prob_map, config.GRID_W, config.GRID_H,
            ux_arr, uy_arr, ctypes.c_int(num_uavs), 
            ctypes.c_float(config.RADAR_RADIUS_GRIDS), ctypes.c_float(config.PROB_DETECT)
        )

    def check_capture(self, uav_xs, uav_ys):
        """判定真实目标是否被任何一架无人机捕获"""
        for ux, uy in zip(uav_xs, uav_ys):
            dist = math.hypot(self.true_target_x - ux, self.true_target_y - uy)
            if dist <= config.RADAR_RADIUS_GRIDS:
                if np.random.rand() <= config.PROB_DETECT:
                    return True
        return False