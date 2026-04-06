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
from plot_utils import plot_trajectory

def test_uav_count(num_uavs, save_plots, use_collapse, exp_dir):
    """测试特定无人机数量，返回 (成功率, 失败的轨迹记录)"""
    print(f"\n>>> 正在评估 N = {num_uavs} 架无人机 ...")

    success_count = 0
    successful_times = []
    max_steps = int(config.TIME_LIMIT_HOURS / config.DT_HOURS)
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
            save_path = os.path.join(exp_dir, f"N_{num_uavs}", f"sim_{i}.png")
            plot_title = f"N={num_uavs} Sim={i} [{status}]"
            plot_trajectory(fleet.history, target_history, env.prob_map.copy(),
                            plot_title, save_path=save_path, is_caught=captured)

    current_success_rate = success_count / config.MC_SIMULATIONS
    print(f"N = {num_uavs} 架限时成功率: {current_success_rate * 100:.1f}%")
    avg_time = np.mean(successful_times) if successful_times else 0.0
    std_time = np.std(successful_times) if successful_times else 0.0
    if successful_times:
        print(f"成功捕获平均耗时: {avg_time:.2f} h (标准差: {std_time:.2f} h)")
    return current_success_rate, failed_record, avg_time, std_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--collapse', type=str, default='false', choices=['true', 'false'])
    args = parser.parse_args()
    save_plots = (args.save.lower() == 'true')
    use_collapse = (args.collapse.lower() == 'true')

    print("\n" + "=" * 40)
    print(f"求最少无人机数量")
    print(f"目标运动：{config.TARGET_TRUE_MOTION} | 目标分布: {config.TARGET_INIT_MODE} | 信任度崩塌: {'开启' if use_collapse else '禁用'}")
    print("=" * 40 + "\n")

    exp_dir = None

    if save_plots:
        mode_str = config.TARGET_INIT_MODE
        motion_str = config.TARGET_TRUE_MOTION
        collapse_str = "collapse" if use_collapse else ""
        entropy_str = "entropy" if config.ENTROPY_INJECTION_RATE != 0 else ""
        if mode_str == 'uniform': collapse_str = ""
        sub_path = f"{mode_str}_{motion_str}_{collapse_str}_{entropy_str}"

        base_dir = os.path.join(BASE_DIR, "data", "min_uavs", sub_path)
        os.makedirs(base_dir, exist_ok=True)

        exp_id = 0
        while os.path.exists(os.path.join(base_dir, str(exp_id))):
            exp_id += 1
        exp_dir = os.path.join(base_dir, str(exp_id))
        os.makedirs(exp_dir)
        print(f"数据将保存至: {exp_dir}")

    current_uavs = config.START_UAV_COUNT
    best_avg, best_std = 0.0, 0.0

    while current_uavs > 0:
        success_rate, failed_record, avg_time, std_time = test_uav_count(current_uavs, save_plots, use_collapse, exp_dir)
        if success_rate < config.TARGET_SUCCESS_RATE:
            print("\n" + "=" * 40)
            print(
                f"测试在 N = {current_uavs} 架时未达标 ({success_rate * 100:.1f}% < {config.TARGET_SUCCESS_RATE * 100}%)")
            min_required_uavs = current_uavs + 1
            if min_required_uavs <= config.START_UAV_COUNT:
                print(f"该配置 ({min_required_uavs}架) 的平均耗时: {best_avg:.2f} h (标准差: {best_std:.2f} h)")

            print("=" * 40 + "\n")

            if failed_record is not None:
                print("渲染导致不合格的失败轨迹图")
                plot_trajectory(failed_record[0], failed_record[1], failed_record[2],
                                f"Failure Case with N={current_uavs} UAVs", is_caught=False)
            break

        best_avg = avg_time
        best_std = std_time

        print(f"N = {current_uavs} 达标，尝试 N = {current_uavs - 1} ...")
        current_uavs -= 1


if __name__ == "__main__":
    main()