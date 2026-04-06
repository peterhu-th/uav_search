# ==========================================
# config.py - 全局参数配置中心
# ==========================================

import numpy as np
import math


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

TARGET_INIT_MODE = 'uniform'                                # 初始分布模式: 'uniform' (完全未知) 或 'gaussian' (先验中心)
TARGET_TRUE_MOTION = 'straight'                             # 目标的机动策略: 'random', 'straight', 'inertia', 'evasive'


# ----------------- BZK-005 无人机参数 -----------------
UAV_SPEED_KPH = 180.0                                       # 巡航速度 (km/h)
UAV_STEP_DIST_KM = UAV_SPEED_KPH * DT_HOURS
UAV_STEP_GRIDS = UAV_STEP_DIST_KM / GRID_RES_KM

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


# ----------------- 信任度崩塌策略 -----------------
COLLAPSE_TRIGGER_DIST = 3.0     # 判定到达中心点的距离阈值 (网格)
COLLAPSE_DELAY_STEPS = 0        # 延迟步数
COLLAPSE_FACTOR = 0.3           # 崩塌系数


# ----------------- 蒙特卡洛仿真参数 -----------------
MC_SIMULATIONS = 20             # 每次测试仿真次数
MAX_SIMULATION_HOURS = 100.0    # 最大允许搜索时间
UAV_COUNT = 2                   # search_time 无人机数量
START_UAV_COUNT = 10            # min_uavs 递减起始无人机数量


# ----------------- 计算无人机就位用时 -----------------
def _haversine(lon1, lat1, lon2, lat2):
    """利用半正矢公式计算地球两点间的大圆距离 (单位: km)"""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


LAUNCH_LON, LAUNCH_LAT = 120.85, 27.912
AREA_TOP_LEFT_LON, AREA_TOP_LEFT_LAT = 124.0, 25.0
TOTAL_MISSION_TIME_HOURS = 10.0

# 计算实际切入点
offset_lat = -RADAR_WIDTH_KM / 111.32
offset_lon = RADAR_WIDTH_KM / (111.32 * math.cos(math.radians(AREA_TOP_LEFT_LAT)))

ENTRY_LON = AREA_TOP_LEFT_LON + offset_lon
ENTRY_LAT = AREA_TOP_LEFT_LAT + offset_lat

UAV_ARRIVAL_DIST_KM = _haversine(LAUNCH_LON, LAUNCH_LAT, ENTRY_LON, ENTRY_LAT)
ARRIVAL_TIME_HOURS = round(UAV_ARRIVAL_DIST_KM / UAV_SPEED_KPH, 1)
TIME_LIMIT_HOURS = round(TOTAL_MISSION_TIME_HOURS - ARRIVAL_TIME_HOURS, 1)
