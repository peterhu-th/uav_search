# ==========================================
# config.py - 全局参数配置中心
# ==========================================
import numpy as np

# ----------------- 环境与网格参数 -----------------
AREA_WIDTH_KM = 306.0       # 海域宽度（km）
AREA_HEIGHT_KM = 444.0      # 海域高度（km）
GRID_RES_KM = 3.0           # 网格分辨率（km/格）


GRID_W = int(AREA_WIDTH_KM / GRID_RES_KM)
GRID_H = int(AREA_HEIGHT_KM / GRID_RES_KM)
TOTAL_GRIDS = GRID_W * GRID_H

# ----------------- 时间与仿真参数 -----------------
DT_MINUTES = 6.0               # 仿真时间步长 (min)
DT_HOURS = DT_MINUTES / 60.0    # 转换为小时，用于速度计算

TARGET_SUCCESS_RATE = 0.95      # 判定搜索成功的概率阈值
PRUNE_THRESHOLD = 1e-8          # 极小概率阈值

# ----------------- 敌方目标参数 -----------------
TARGET_SPEED_KPH = 30.0                                     # 目标最大机动速度 (km/h)
TARGET_STEP_DIST_KM = TARGET_SPEED_KPH * DT_HOURS 
GAUSSIAN_SIGMA = TARGET_STEP_DIST_KM / GRID_RES_KM * 2.0    # 转化为网格数作为高斯扩散的 sigma

TARGET_INIT_MODE = 'gaussian'                               # 初始分布模式: 'uniform' (完全未知) 或 'gaussian' (先验中心)
TARGET_TRUE_MOTION = 'straight'                             # 目标的机动策略: 'random', 'straight', 'inertia', 'evasive'


# ----------------- BZK-005 无人机参数 -----------------
UAV_SPEED_KPH = 180.0                                       # 巡航速度 (km/h)
UAV_STEP_DIST_KM = UAV_SPEED_KPH * DT_HOURS
UAV_STEP_GRIDS = UAV_STEP_DIST_KM / GRID_RES_KM             # 无人机每步能飞几格

RADAR_WIDTH_KM = 30                                         # 雷达扫描幅宽 (km)
RADAR_RADIUS_GRIDS = (RADAR_WIDTH_KM / 2.0) / GRID_RES_KM   # 雷达半径(网格数)
PROB_DETECT = 0.95                                          # 探测概率 p_d

UAV_ENTRY_POINTS = [                                        # 无人机切入点坐标列表 (可配多个，代表不同方向入场)
    (0, GRID_H - 1)                                         # 默认从西北角入场
]


# ----------------- 人工势场决策参数 (APF) -----------------
W_INERTIA = 0.4                 # 惯性权重
C_ATTRACT = 1.0                 # 概率引力系数
C_REPEL = 10                    # 无人机间斥力系数
REPEL_DISTANCE_GRIDS = 15       # 斥力生效距离(网格数)
EDGE_PENALTY_FACTOR = 0.1

# ----------------- 蒙特卡洛仿真参数 -----------------
MC_SIMULATIONS = 20             # 每次测试运行的蒙特卡洛仿真次数
MAX_SIMULATION_HOURS = 100.0    # 任务二:最大允许搜索时间
TASK2_UAV_COUNT = 2             # 任务二:测试的无人机数量
TASK3_TIME_LIMIT_HOURS = 7.8   # 任务三:时间约束
TASK3_START_UAV_COUNT = 8       # 任务三:递减测试的起始无人机数量