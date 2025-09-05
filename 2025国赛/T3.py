import numpy as np
from scipy.optimize import differential_evolution
import pandas as pd
import time
import os

# =============================================================================
# 1. 全局常量与几何计算函数 (基础模块)
# =============================================================================

# 物理和环境参数
P_M1_0 = np.array([20000.0, 0.0, 2000.0])  # 导弹初始位置
v_M = 300.0  # 导弹速度
P_FY1_0 = np.array([17800.0, 0.0, 1800.0])  # 无人机1初始位置
R_C = 10.0  # 烟幕云团半径
v_sink = 3.0  # 烟雾下降速度
g = 9.8  # 重力加速度

# 真目标关键点列表 (8点外接盒模型)
key_points = [
    np.array([7, 207.0, 0.0]),
    np.array([7, 193.0, 0.0]),
    np.array([-7, 207.0, 0.0]),
    np.array([-7, 193.0, 0.0]),
    np.array([7, 207.0, 10.0]),
    np.array([7, 193.0, 10.0]),
    np.array([-7, 207.0, 10.0]),
    np.array([-7, 193.0, 10.0]),
]

# 导弹轨迹预计算
d_M1 = (np.array([0.0, 0.0, 0.0]) - P_M1_0) / np.linalg.norm(
    np.array([0.0, 0.0, 0.0]) - P_M1_0
)


def P_M1(t):
    """计算t时刻导弹M1的位置"""
    return P_M1_0 + v_M * t * d_M1


# 几何计算函数
def dist_point_to_segment(p, a, b):
    """计算点p到线段ab的最短距离"""
    v = b - a
    w = p - a
    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(p - a)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(p - b)
    b_proj = c1 / c2
    pb = a + b_proj * v
    return np.linalg.norm(p - pb)


# =============================================================================
# 2. 计算单枚烟幕弹遮蔽区间的函数
# =============================================================================


def get_shielding_intervals(v_FY1, theta_rad, t_drop, t_fuse, P_FY1_0, dt=1e-2):
    """计算单枚烟幕弹产生的有效遮蔽时间区间列表"""
    # --- 轨迹与起爆点计算 ---
    d_FY1 = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])
    P_drop = P_FY1_0 + v_FY1 * t_drop * d_FY1
    v_drop = v_FY1 * d_FY1
    P_detonate = P_drop + v_drop * t_fuse + np.array([0, 0, -0.5 * g * t_fuse**2])
    t_detonate = t_drop + t_fuse

    def P_C(t):
        return P_detonate + np.array([0, 0, -v_sink * (t - t_detonate)])

    # --- 模拟并记录遮蔽时刻 ---
    shielded_times = []
    t_start = t_detonate
    t_end = t_detonate + 20.0

    for t in np.arange(t_start, t_end, dt):
        p_m1_t = P_M1(t)
        p_c_t = P_C(t)
        is_fully_shielded = True
        for p_key in key_points:
            if dist_point_to_segment(p_c_t, p_m1_t, p_key) >= R_C:
                is_fully_shielded = False
                break
        if is_fully_shielded:
            shielded_times.append(t)

    # --- 将离散的遮蔽时刻转换为连续的时间区间 ---
    if not shielded_times:
        return []

    intervals = []
    start_of_interval = shielded_times[0]
    for i in range(1, len(shielded_times)):
        if shielded_times[i] - shielded_times[i - 1] > dt * 1.5:
            intervals.append((start_of_interval, shielded_times[i - 1]))
            start_of_interval = shielded_times[i]
    intervals.append((start_of_interval, shielded_times[-1]))

    return intervals


# =============================================================================
# 3. 合并区间并计算总长度的函数
# =============================================================================


def calculate_union_length(intervals):
    """计算区间列表并集的总长度"""
    if not intervals:
        return 0

    intervals.sort(key=lambda x: x[0])

    merged = []
    if not intervals:
        return 0

    current_start, current_end = intervals[0]

    for next_start, next_end in intervals[1:]:
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    merged.append((current_start, current_end))

    total_length = sum(end - start for start, end in merged)
    return total_length


# =============================================================================
# 4. 问题三的目标函数与优化过程
# =============================================================================


def objective_function_Q3(params):
    """问题三的目标函数，使用变量变换法处理约束"""
    v_FY1, theta_rad, t_drop1, t_fuse1, dt_drop2, t_fuse2, dt_drop3, t_fuse3 = params

    t_drop2 = t_drop1 + dt_drop2
    t_drop3 = t_drop2 + dt_drop3

    all_intervals = []

    # 计算三枚弹药各自的遮蔽区间
    all_intervals.extend(
        get_shielding_intervals(v_FY1, theta_rad, t_drop1, t_fuse1, P_FY1_0)
    )
    all_intervals.extend(
        get_shielding_intervals(v_FY1, theta_rad, t_drop2, t_fuse2, P_FY1_0)
    )
    all_intervals.extend(
        get_shielding_intervals(v_FY1, theta_rad, t_drop3, t_fuse3, P_FY1_0)
    )

    total_length = calculate_union_length(all_intervals)

    # 优化器默认求最小值，所以返回总时长的负数
    return -total_length


def run_optimization_Q3():
    """执行问题三的完整优化流程"""
    print("--- 问题三：单无人机三弹药优化 ---")

    # 1. 定义变换后变量的搜索边界
    # [v, θ, t_drop1, t_fuse1, Δt_drop2, t_fuse2, Δt_drop3, t_fuse3]
    bounds = [
        (70, 140),  # v_FY1 (m/s)
        (np.pi * 170 / 180, np.pi * 190 / 180),
        (0, 5),  # t_drop1 (s)
        (0, 5),  # t_fuse1 (s)
        (1, 5),  # Δt_drop2 (s)
        (0, 5),  # t_fuse2 (s)
        (1, 5),  # Δt_drop3 (s)
        (0, 5),  # t_fuse3 (s)
    ]

    # 2. 调用差分进化算法
    print("开始优化 (8维空间，计算量较大，请耐心等待)...")
    start_time = time.time()

    result = differential_evolution(
        objective_function_Q3,
        bounds,
        strategy="best1bin",
        maxiter=300,
        popsize=25,
        tol=0.01,
        disp=True,
        workers=-1,  # 使用所有CPU核心并行计算
    )

    end_time = time.time()
    print(f"\n优化完成，耗时: {end_time - start_time:.2f} 秒")

    # 3. 处理并保存结果
    if result.success:
        p = result.x
        max_time = -result.fun

        # 还原真实的投放时间
        t_drop1 = p[2]
        t_drop2 = p[2] + p[4]
        t_drop3 = t_drop2 + p[6]

        # 提取最优参数
        v_opt, theta_opt = p[0], p[1]
        params_list = [(t_drop1, p[3]), (t_drop2, p[5]), (t_drop3, p[7])]

        print("\n--- 最优策略 (问题三) ---")
        print(f"无人机飞行速度: {v_opt:.2f} m/s")
        print(f"无人机飞行方向: {np.rad2deg(theta_opt):.2f}°")
        print("------------------------------------")
        for i, (td, tf) in enumerate(params_list):
            print(f"弹药{i+1}: 投放时间={td:.2f}s, 引信={tf:.2f}s")
        print("------------------------------------")
        print(f"最大总有效遮蔽时长: {max_time:.4f} s")

        # 4. 生成并保存 result1.xlsx
        save_results_to_excel(v_opt, theta_opt, params_list)

    else:
        print("\n优化未成功收敛。可以尝试增加 maxiter 或 popsize。")


def save_results_to_excel(v, theta_rad, params_list):
    """将最优策略的详细信息保存到 result1.xlsx"""
    print("\n正在生成 result1.xlsx 文件...")

    results_data = []
    d_FY1 = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])

    for i, (t_drop, t_fuse) in enumerate(params_list):
        # 计算投放点
        p_drop = P_FY1_0 + v * t_drop * d_FY1
        # 计算起爆点
        v_drop = v * d_FY1
        p_detonate = p_drop + v_drop * t_fuse + np.array([0, 0, -0.5 * g * t_fuse**2])

        row = {
            "无人机编号": "FY1",
            "烟幕弹编号": f"G{i+1}",
            "飞行方向（度）": np.rad2deg(theta_rad),
            "飞行速度（m/s）": v,
            "投放点x": p_drop[0],
            "投放点y": p_drop[1],
            "投放点z": p_drop[2],
            "起爆点x": p_detonate[0],
            "起爆点y": p_detonate[1],
            "起爆点z": p_detonate[2],
        }
        results_data.append(row)

    df = pd.DataFrame(results_data)

    # 确保列的顺序与模板文件一致
    column_order = [
        "无人机编号",
        "烟幕弹编号",
        "飞行方向（度）",
        "飞行速度（m/s）",
        "投放点x",
        "投放点y",
        "投放点z",
        "起爆点x",
        "起爆点y",
        "起爆点z",
    ]
    df = df[column_order]

    # 保存到Excel文件
    filename = "result1.xlsx"
    df.to_excel(filename, index=False, float_format="%.4f")
    print(f"结果已成功保存到 '{os.path.abspath(filename)}'")


# --- 主程序入口 ---
if __name__ == "__main__":
    run_optimization_Q3()
