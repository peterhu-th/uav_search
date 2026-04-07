# ==========================================
# demo.py - 基础贝叶斯热力图与协同搜索演示
# ==========================================

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'config'))
sys.path.append(os.path.join(BASE_DIR, 'src'))
import config
from environment import Environment
from uav_controller import UAVFleet

def main():

    print("\n" + "=" * 40)
    print(">>> 启动任务一仿真演示 <<<")
    print(f"目标初始分布模式: {config.TARGET_INIT_MODE}")
    print(f"信任度崩塌机制: 启用 (仅在 gaussian 模式下生效)")
    print("=" * 40 + "\n")

    env = Environment()
    fleet = UAVFleet(num_uavs=5)
    target_history = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))

    step = 0
    max_steps = 200

    while step < max_steps:
        # 概率热力图随时间扩散
        env.time_update_bayes()

        # 真实目标机动
        uxs, uys = fleet.get_positions()
        env.move_true_target(uxs, uys)

        # 无人机根据热力图计算势场并移动
        fleet.calculate_apf_and_move(env.prob_map)
        env.apply_confidence_collapse(uxs, uys)
        uxs, uys = fleet.get_positions()
        env.measurement_update_bayes(uxs, uys)

        # 抓捕判断
        if env.check_capture(uxs, uys):
            print(f"\n目标在第 {step} 步被成功捕获！")
            break

        # 可视化渲染
        ax.clear()
        prob_map_2d = env.prob_map.reshape((config.GRID_H, config.GRID_W))

        # 绘制热力底图
        im = ax.imshow(prob_map_2d, cmap='jet', origin='lower',
                       extent=[0, config.GRID_W, 0, config.GRID_H],
                       vmin=0, vmax=np.max(prob_map_2d) + 1e-9)

        # 绘制真实目标
        if len(target_history) > 1:
            tx, ty = zip(*target_history)
            ax.plot(tx, ty, color='red', linestyle=':', linewidth=2, label='Target Trajectory', zorder=4)
        ax.scatter(env.true_target_x, env.true_target_y, c='red', marker='*', s=150, label='True Target', zorder=5)

        # 绘制无人机及其雷达覆盖范围
        colors = ['cyan', 'magenta', 'lime']
        for i in range(fleet.num_uavs):
            c = colors[i % len(colors)]
            ax.scatter(uxs[i], uys[i], c=c, marker='^', s=100, edgecolors='black', zorder=4)

            # 历史轨迹线
            if len(fleet.history[i]) > 1:
                hx, hy = zip(*fleet.history[i])
                ax.plot(hx, hy, color=c, linestyle='-', linewidth=1.5, alpha=0.8, zorder=3)
            # 雷达覆盖圆圈
            circle = plt.Circle((uxs[i], uys[i]), config.RADAR_RADIUS_GRIDS,
                                color=c, fill=False, linestyle='--', linewidth=1.5, zorder=4)
            ax.add_patch(circle)

        ax.set_title(f"Bayesian Heatmap Search - Step {step}")
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        ax.legend(loc='upper right')

        plt.pause(0.01)
        step += 1

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()