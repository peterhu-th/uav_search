# ==========================================
# search_time.py - 计算特定无人机数量（2架）的平均搜索时间
# ==========================================
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'config'))
sys.path.append(os.path.join(BASE_DIR, 'src'))
import config
from environment import Environment
from uav_controller import UAVFleet


def run_single_simulation(sim_id, num_uavs):
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

        uxs, uys = fleet.get_positions()
        env.measurement_update_bayes(uxs, uys)

        if env.check_capture(uxs, uys):
            target_history.append((env.true_target_x, env.true_target_y))
            time_spent = step * config.DT_HOURS
            print(f"[ {sim_id + 1}/{config.MC_SIMULATIONS}] 耗时: {time_spent:.2f} h")
            return time_spent, fleet.history, target_history, env.prob_map.copy(), True

        step += 1

    print(f"[{sim_id + 1}/{config.MC_SIMULATIONS}] 超过最大时间 {config.MAX_SIMULATION_HOURS} h")
    return max_steps * config.DT_HOURS, fleet.history, target_history, env.prob_map.copy(), False


def plot_trajectory(fleet_history, target_history, prob_map, title):
    """绘制全局静态轨迹汇总图"""
    fig, ax = plt.subplots(figsize=(10, 8))

    prob_map_2d = prob_map.reshape((config.GRID_H, config.GRID_W))
    im = ax.imshow(prob_map_2d, cmap='jet', origin='lower',
                   extent=[0, config.GRID_W, 0, config.GRID_H],
                   vmin=0, vmax=np.max(prob_map_2d) + 1e-9)

    # 画目标轨迹
    tx, ty = zip(*target_history)
    ax.plot(tx, ty, color='red', linestyle=':', linewidth=2, label='Target Trajectory', zorder=4)
    ax.scatter(tx[0], ty[0], c='white', marker='o', s=80, edgecolors='red', label='Target Start')
    ax.scatter(tx[-1], ty[-1], c='red', marker='*', s=200, label='Target Caught', zorder=5)

    # 画无人机轨迹
    colors = ['cyan', 'magenta', 'lime', 'yellow', 'orange', 'pink', 'white', 'lightgreen']
    for i, h in enumerate(fleet_history):
        c = colors[i % len(colors)]
        hx, hy = zip(*h)
        ax.plot(hx, hy, color=c, linestyle='-', linewidth=1.5, alpha=0.8)
        ax.scatter(hx[0], hy[0], c=c, marker='s', s=50, edgecolors='black')
        ax.scatter(hx[-1], hy[-1], c=c, marker='^', s=100, edgecolors='black', zorder=4)

    ax.set_title(title)
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.legend(loc='upper right')
    plt.show()

def main():
    print(f"计算 {config.TASK2_UAV_COUNT} 架无人机的平均捕获时间")
    print(f"步长 {config.DT_MINUTES} min, 蒙特卡洛次数 {config.MC_SIMULATIONS}")

    successful_times = []

    # 记录耗时最长的一次仿真数据
    longest_time = -1
    longest_record = None

    for i in range(config.MC_SIMULATIONS):
        time_spent, f_hist, t_hist, p_map, success = run_single_simulation(i, config.TASK2_UAV_COUNT)

        if success:
            successful_times.append(time_spent)
            # 记录最长耗时的轨迹
            if time_spent > longest_time:
                longest_time = time_spent
                longest_record = (f_hist, t_hist, p_map)

    if len(successful_times) > 0:
        avg_time = np.mean(successful_times)
        success_rate = len(successful_times) / config.MC_SIMULATIONS * 100
        print("\n" + "=" * 40)
        print(f"任务二 最终结果报告:")
        print(f"测试数量: {config.TASK2_UAV_COUNT} 架无人机")
        print(f"成功率:   {success_rate:.1f}%")
        print(f"平均耗时: {avg_time:.2f} h")
        print(f"平均总耗时: {avg_time + 2.3:.2f} h")
        print("=" * 40 + "\n")

        if longest_record is not None:
            print(f"渲染耗时最长 ({longest_time:.2f} h) 的轨迹...")
            plot_trajectory(longest_record[0], longest_record[1], longest_record[2],
                            f"Task 2: Longest Successful Search ({longest_time:.2f} h)")
    else:
        print("\n所有仿真均超时，未能计算出平均时间")


if __name__ == "__main__":
    main()