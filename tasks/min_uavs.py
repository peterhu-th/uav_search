# ==========================================
# min_uavs.py - 时间约束下求最少无人机数量
# ==========================================
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

# 【已删除导致命名冲突的 from more_itertools.more import collapse】

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'config'))
sys.path.append(os.path.join(BASE_DIR, 'src'))

import config
from environment import Environment
from uav_controller import UAVFleet


def test_uav_count(num_uavs, save_plots, use_collapse, exp_dir):
    """测试特定无人机数量，返回 (成功率, 失败的轨迹记录)"""
    print(f"\n>>> 正在评估 N = {num_uavs} 架无人机 ...")

    success_count = 0
    successful_times = []
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

            if use_collapse:
                env.apply_confidence_collapse(uxs, uys)

            uxs, uys = fleet.get_positions()
            env.measurement_update_bayes(uxs, uys)

            if env.check_capture(uxs, uys):
                success_count += 1
                captured = True
                time_spent = step * config.DT_HOURS
                successful_times.append(time_spent)

                break

            step += 1

        status = "成功" if captured else "超时失败"
        print(f"[{i + 1}/{config.MC_SIMULATIONS}] 结果: {status}")

        if not captured and failed_record is None:
            target_history.append((env.true_target_x, env.true_target_y))
            failed_record = (fleet.history, target_history, env.prob_map.copy())

        if save_plots and exp_dir:
            save_path = os.path.join(exp_dir, f"N_{num_uavs}_sim_{i}.png")
            plot_title = f"Task 3: N={num_uavs} Sim={i} [{status}]"
            plot_trajectory(fleet.history, target_history, env.prob_map.copy(), plot_title, save_path=save_path)

    current_success_rate = success_count / config.MC_SIMULATIONS
    print(f"N = {num_uavs} 架限时成功率: {current_success_rate * 100:.1f}%")
    avg_time = np.mean(successful_times) if successful_times else 0.0
    std_time = np.std(successful_times) if successful_times else 0.0
    if successful_times:
        print(f"成功捕获平均耗时: {avg_time:.2f} h (标准差: {std_time:.2f} h)")
    return current_success_rate, failed_record, avg_time, std_time


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
    parser.add_argument('--collapse', type=str, default='true', choices=['true', 'false'])
    args = parser.parse_args()
    save_plots = (args.save.lower() == 'true')
    use_collapse = (args.collapse.lower() == 'true')

    print("\n" + "=" * 40)
    print(f"求解最少无人机数量")
    print(f"目标分布: {config.TARGET_INIT_MODE} | 信任度崩塌: {'开启' if use_collapse else '关闭'}")
    print("=" * 40 + "\n")

    exp_dir = None

    if save_plots:
        mode_str = config.TARGET_INIT_MODE
        collapse_str = "collapse_on" if use_collapse else "collapse_off"
        sub_path = f"{mode_str}_{collapse_str}"

        base_dir = os.path.join(BASE_DIR, "data", "min_uavs", sub_path)
        os.makedirs(base_dir, exist_ok=True)

        exp_id = 0
        while os.path.exists(os.path.join(base_dir, str(exp_id))):
            exp_id += 1
        exp_dir = os.path.join(base_dir, str(exp_id))
        os.makedirs(exp_dir)
        print(f"数据将保存至: {exp_dir}")

    current_uavs = config.TASK3_START_UAV_COUNT
    best_avg, best_std = 0.0, 0.0

    while current_uavs > 0:
        success_rate, failed_record, avg_time, std_time = test_uav_count(current_uavs, save_plots, use_collapse, exp_dir)
        if success_rate < config.TARGET_SUCCESS_RATE:
            print("\n" + "=" * 40)
            print(
                f"测试在 N = {current_uavs} 架时未达标 ({success_rate * 100:.1f}% < {config.TARGET_SUCCESS_RATE * 100}%)")
            min_required_uavs = current_uavs + 1
            if min_required_uavs <= config.TASK3_START_UAV_COUNT:
                print(f"该配置 ({min_required_uavs}架) 的平均耗时: {best_avg:.2f} h (标准差: {best_std:.2f} h)")

            print("=" * 40 + "\n")

            if failed_record is not None:
                print("渲染导致不合格的失败轨迹图")
                plot_trajectory(failed_record[0], failed_record[1], failed_record[2],
                                f"Failure Case with N={current_uavs} UAVs")
            break

        best_avg = avg_time
        best_std = std_time

        print(f"N = {current_uavs} 达标，尝试 N = {current_uavs - 1} ...")
        current_uavs -= 1


if __name__ == "__main__":
    main()