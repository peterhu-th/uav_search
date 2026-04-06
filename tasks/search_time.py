# ==========================================
# search_time.py - 计算特定无人机数量（2架）的平均搜索时间
# ==========================================
import os
import sys
import numpy as np
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'config'))
sys.path.append(os.path.join(BASE_DIR, 'src'))
import config
from environment import Environment
from uav_controller import UAVFleet
from plot_utils import plot_trajectory


def run_single_simulation(sim_id, num_uavs, use_collapse):
    """运行单次仿真，返回(耗时, UAV轨迹, 目标轨迹, 最终热力图, 是否成功)"""
    env = Environment()
    fleet = UAVFleet(num_uavs=num_uavs)

    step = 0
    max_steps = int(config.MAX_SIMULATION_HOURS / config.DT_HOURS)
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
            target_history.append((env.true_target_x, env.true_target_y))
            time_spent = step * config.DT_HOURS
            print(f"[{sim_id + 1}/{config.MC_SIMULATIONS}] 耗时: {time_spent:.2f} h")
            return time_spent, fleet.history, target_history, env.prob_map.copy(), True

        step += 1

    print(f"[{sim_id + 1}/{config.MC_SIMULATIONS}] 超过最大时间 {config.MAX_SIMULATION_HOURS} h")
    return max_steps * config.DT_HOURS, fleet.history, target_history, env.prob_map.copy(), False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--collapse', type=str, default='true', choices=['true', 'false'])
    args = parser.parse_args()
    save_plots = (args.save.lower() == 'true')
    use_collapse = (args.collapse.lower() == 'true')

    print("\n" + "=" * 40)
    print(f"计算 {config.UAV_COUNT} 架无人机的平均捕获时间")
    print(f"步长 {config.DT_MINUTES} min, 蒙特卡洛次数 {config.MC_SIMULATIONS}")
    print(f"目标运动：{config.TARGET_TRUE_MOTION} | 目标分布: {config.TARGET_INIT_MODE} | 信任度崩塌: {'开启' if use_collapse else '禁用'} | 底噪: {config.ENTROPY_INJECTION_RATE}")
    print("=" * 40 + "\n")

    exp_dir = None

    if save_plots:
        mode_str = config.TARGET_INIT_MODE
        motion_str = config.TARGET_TRUE_MOTION
        collapse_str = "collapse" if use_collapse else ""
        entropy_str = "entropy" if config.ENTROPY_INJECTION_RATE != 0 else ""
        if mode_str == 'uniform': collapse_str = ""
        sub_path = f"{mode_str}_{motion_str}_{collapse_str}_{entropy_str}"

        base_dir = os.path.join(BASE_DIR, "data", "search_time", sub_path)
        os.makedirs(base_dir, exist_ok=True)

        exp_id = 0
        while os.path.exists(os.path.join(base_dir, str(exp_id))):
            exp_id += 1
        exp_dir = os.path.join(base_dir, str(exp_id))
        os.makedirs(exp_dir)
        print(f"数据将保存至: {exp_dir}")

    successful_times = []
    longest_time = -1
    longest_record = None

    for i in range(config.MC_SIMULATIONS):
        time_spent, f_hist, t_hist, p_map, success = run_single_simulation(i, config.UAV_COUNT, use_collapse)

        if save_plots and exp_dir:
            save_path = os.path.join(exp_dir, f"sim_{i}.png")
            status_str = "成功" if success else "超时"
            plot_title = f"Sim={i} [{status_str}]"
            plot_trajectory(f_hist, t_hist, p_map, plot_title, save_path=save_path, is_caught=success)

        if success:
            successful_times.append(time_spent)
            if time_spent > longest_time:
                longest_time = time_spent
                longest_record = (f_hist, t_hist, p_map)

    if len(successful_times) > 0:
        avg_time = np.mean(successful_times)
        std_time = np.std(successful_times)
        success_rate = len(successful_times) / config.MC_SIMULATIONS * 100
        print("\n" + "=" * 40)
        print(f"最终结果:")
        print(f"成功率:   {success_rate:.1f}%")
        print(f"平均耗时: {avg_time:.2f} h (标准差: {std_time:.2f} h)")
        print(f"平均总耗时: {avg_time + config.ARRIVAL_TIME_HOURS:.2f} h")
        print("=" * 40 + "\n")

        if longest_record is not None:
            print(f"渲染耗时最长 ({longest_time:.2f} h) 的轨迹...")
            plot_trajectory(longest_record[0], longest_record[1], longest_record[2],
                            f"Longest Successful Search ({longest_time:.2f} h)", is_caught=True)
    else:
        print("\n所有仿真均超时，未能计算出平均时间")


if __name__ == "__main__":
    main()