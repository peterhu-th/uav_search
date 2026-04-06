# ==========================================
# min_uavs.py - 时间约束下求最少无人机数量
# ==========================================
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'config'))
sys.path.append(os.path.join(BASE_DIR, 'src'))

import config
from environment import Environment
from uav_controller import UAVFleet


def test_uav_count(num_uavs, save_plots=False):
    """测试特定无人机数量，返回 (成功率, 失败的轨迹记录)"""
    print(f"\n>>> 正在评估 N = {num_uavs} 架无人机 ...")

    success_count = 0
    max_steps = int(config.TASK3_TIME_LIMIT_HOURS / config.DT_HOURS)
    failed_record = None

    for i in range(config.MC_SIMULATIONS):
        env = Environment()
        fleet = UAVFleet(num_uavs=num_uavs)

        step = 0
        captured = False
        target_history = []

        while step < max_steps:
            target_history.append((env.true_target_x, env.true_target_y))

            env.time_update_bayes()

            uxs, uys = fleet.get_positions()
            env.move_true_target(uxs, uys)
            fleet.calculate_apf_and_move(env.prob_map)

            uxs, uys = fleet.get_positions()
            env.measurement_update_bayes(uxs, uys)

            if env.check_capture(uxs, uys):
                success_count += 1
                captured = True
                break

            step += 1

        status = "成功" if captured else "超时失败"
        print(f"[{i + 1}/{config.MC_SIMULATIONS}] 结果: {status}")

        # 记录失败的轨迹
        if not captured and failed_record is None:
            target_history.append((env.true_target_x, env.true_target_y))
            failed_record = (fleet.history, target_history, env.prob_map.copy())

        if save_plots:
            save_dir = os.path.join(BASE_DIR, "data", "min_uavs", f"N_{num_uavs}", str(i))
            save_path = os.path.join(save_dir, "plot.png")
            plot_title = f"Task 3: N={num_uavs} Sim={i} [{status}]"
            plot_trajectory(fleet.history, target_history, env.prob_map.copy(), plot_title, save_path=save_path)

    current_success_rate = success_count / config.MC_SIMULATIONS
    print(f"N = {num_uavs} 架限时成功率: {current_success_rate * 100:.1f}%")

    return current_success_rate, failed_record


def plot_trajectory(fleet_history, target_history, prob_map, title, save_path=None):
    """绘制全局静态轨迹汇总图 (复用逻辑)"""
    fig, ax = plt.subplots(figsize=(10, 8))
    prob_map_2d = prob_map.reshape((config.GRID_H, config.GRID_W))
    im = ax.imshow(prob_map_2d, cmap='jet', origin='lower', extent=[0, config.GRID_W, 0, config.GRID_H], vmin=0,
                   vmax=np.max(prob_map_2d) + 1e-9)

    tx, ty = zip(*target_history)
    ax.plot(tx, ty, color='red', linestyle=':', linewidth=2, label='Target Trajectory', zorder=4)
    ax.scatter(tx[-1], ty[-1], c='red', marker='X', s=200, label='Target Escaped', zorder=5)

    colors = ['cyan', 'magenta', 'lime', 'yellow', 'orange', 'pink', 'white', 'lightgreen']
    for i, h in enumerate(fleet_history):
        c = colors[i % len(colors)]
        hx, hy = zip(*h)
        ax.plot(hx, hy, color=c, linestyle='-', linewidth=1.5, alpha=0.8)
        ax.scatter(hx[-1], hy[-1], c=c, marker='^', s=100, edgecolors='black', zorder=4)

    ax.set_title(title)
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.legend(loc='upper right')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='false', choices=['true', 'false'])
    args = parser.parse_args()
    save_plots = (args.save.lower() == 'true')

    print(f"求解最少无人机数量")

    current_uavs = config.TASK3_START_UAV_COUNT
    min_required_uavs = current_uavs

    while current_uavs > 0:
        success_rate, failed_record = test_uav_count(current_uavs, save_plots)

        # 失败则停止递减
        if success_rate < config.TARGET_SUCCESS_RATE:
            print("\n" + "=" * 40)
            print(
                f"测试在 N = {current_uavs} 架时未达标 ({success_rate * 100:.1f}% < {config.TARGET_SUCCESS_RATE * 100}%)")
            min_required_uavs = current_uavs + 1
            print(f"结论:最少需要部署的无人机数量为: {min_required_uavs} 架")
            print("=" * 40 + "\n")

            if failed_record is not None:
                print("渲染导致不合格的失败轨迹图")
                plot_trajectory(failed_record[0], failed_record[1], failed_record[2],
                                f"Failure Case with N={current_uavs} UAVs")
            break
        print(f"N = {current_uavs} 达标，尝试 N = {current_uavs - 1} ...")
        current_uavs -= 1


if __name__ == "__main__":
    main()