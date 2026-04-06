# ==========================================
# plot_utils.py - 轨迹与热力图渲染工具
# ==========================================
import os
import numpy as np
import matplotlib.pyplot as plt
import config

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_trajectory(fleet_history, target_history, prob_map, title, save_path=None, is_caught=True):
    """绘制全局静态轨迹汇总图"""
    fig, ax = plt.subplots(figsize=(10, 8))

    prob_map_2d = prob_map.reshape((config.GRID_H, config.GRID_W))
    im = ax.imshow(prob_map_2d, cmap='jet', origin='lower',
                   extent=[0, config.GRID_W, 0, config.GRID_H],
                   vmin=0, vmax=np.max(prob_map_2d) + 1e-9)

    # 目标轨迹
    tx, ty = zip(*target_history)
    ax.plot(tx, ty, color='red', linestyle=':', linewidth=2, label='Target Trajectory', zorder=4)
    ax.scatter(tx[0], ty[0], c='white', marker='o', s=80, edgecolors='red', label='Target Start')

    # 根据是否抓捕成功渲染不同终点图标
    if is_caught:
        ax.scatter(tx[-1], ty[-1], c='red', marker='*', s=200, label='Target Caught', zorder=5)
    else:
        ax.scatter(tx[-1], ty[-1], c='red', marker='X', s=200, label='Target Escaped', zorder=5)

    # 无人机轨迹
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

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()