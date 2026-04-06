// ==========================================
// bayes_core.cpp - 贝叶斯热力图底层计算引擎
// ==========================================
#include <vector>
#include <cmath>
#include <algorithm>

extern "C" {

    #if defined(_WIN32)
        #define EXPORT __declspec(dllexport)
    #else
        #define EXPORT
    #endif

    // 计算全反射边界后的坐标
    inline int get_reflected_coord(int coord, int max_val) {
        if (coord < 0) {
            return -coord; 
        } else if (coord >= max_val) {
            return 2 * (max_val - 1) - coord;
        }
        return coord;
    }

    // ==========================================
    // 功能1：时间更新（高斯核概率扩散与全反射）
    // ==========================================
    EXPORT void time_update(float* prob_map, int width, int height, float sigma, float prune_threshold) {
        int total_grids = width * height;
        // 1. 开辟一个干净的临时缓冲区，防止污染马尔可夫链
        std::vector<float> temp_map(total_grids, 0.0f);
        
        // 预计算高斯核窗口大小 (取 3 倍 sigma 即可覆盖 99% 的概率)
        int kernel_radius = std::ceil(3.0f * sigma);
        if (kernel_radius < 1) kernel_radius = 1;

        // 2. 遍历全图
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                float current_prob = prob_map[idx];

                // 极小值剪枝
                if (current_prob < prune_threshold) {
                    continue; 
                }

                // 高斯扩散
                float sum_weights = 0.0f;
                std::vector<float> local_weights;
                std::vector<std::pair<int, int>> local_coords;

                // 计算局部高斯权重
                for (int dy = -kernel_radius; dy <= kernel_radius; ++dy) {
                    for (int dx = -kernel_radius; dx <= kernel_radius; ++dx) {
                        float dist_sq = dx * dx + dy * dy;
                        // 高斯函数 exp(-d^2 / (2 * sigma^2))
                        float weight = std::exp(-dist_sq / (2.0f * sigma * sigma));
                        local_weights.push_back(weight);
                        sum_weights += weight;

                        // 全反射边界判定
                        int nx = get_reflected_coord(x + dx, width);
                        int ny = get_reflected_coord(y + dy, height);
                        local_coords.push_back({nx, ny});
                    }
                }

                // 将当前网格的概率按照权重比例分配给周围网格
                for (size_t i = 0; i < local_weights.size(); ++i) {
                    float distributed_prob = current_prob * (local_weights[i] / sum_weights);
                    int target_idx = local_coords[i].second * width + local_coords[i].first;
                    temp_map[target_idx] += distributed_prob;
                }
            }
        }

        // 将缓冲区数据写回原数组
        float total_prob = 0.0f;
        for (int i = 0; i < total_grids; ++i) {
            prob_map[i] = temp_map[i];
            total_prob += prob_map[i];
        }

        // 全局概率归一化
        if (total_prob > 0.0f) {
            for (int i = 0; i < total_grids; ++i) {
                prob_map[i] /= total_prob;
            }
        }
    }


    // ==========================================
    // 功能2：观测更新（雷达区域冷却）
    // ==========================================
    EXPORT void measurement_update(float* prob_map, int width, int height, 
                                   float* uav_x, float* uav_y, int num_uavs, 
                                   float radar_radius, float p_d) {
        
        float radar_radius_sq = radar_radius * radar_radius;

        // 遍历所有无人机，执行概率惩罚
        for (int k = 0; k < num_uavs; ++k) {
            int cx = static_cast<int>(std::round(uav_x[k]));
            int cy = static_cast<int>(std::round(uav_y[k]));
            int r = static_cast<int>(std::ceil(radar_radius));

            // 只遍历无人机附近的包围盒
            int min_y = std::max(0, cy - r);
            int max_y = std::min(height - 1, cy + r);
            int min_x = std::max(0, cx - r);
            int max_x = std::min(width - 1, cx + r);

            for (int y = min_y; y <= max_y; ++y) {
                for (int x = min_x; x <= max_x; ++x) {
                    float dist_sq = (x - uav_x[k]) * (x - uav_x[k]) + (y - uav_y[k]) * (y - uav_y[k]);
                    
                    // 如果在雷达无盲区范围内，未发现目标，概率锐减 (乘以 1 - p_d)
                    if (dist_sq <= radar_radius_sq) {
                        int idx = y * width + x;
                        prob_map[idx] *= (1.0f - p_d);
                    }
                }
            }
        }

        // 全图归一化
        float sum_prob = 0.0f;
        int total_grids = width * height;
        for (int i = 0; i < total_grids; ++i) {
            sum_prob += prob_map[i];
        }

        if (sum_prob > 0.0f) {
            for (int i = 0; i < total_grids; ++i) {
                prob_map[i] /= sum_prob;
            }
        }
    }
}