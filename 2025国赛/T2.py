import numpy as np
from scipy.optimize import differential_evolution
import time

# =============================================================================
# 1. 全局常量与几何计算函数 (在多次调用中不变的部分)
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
# 用于辅助目标的真目标中心点
P_TARGET_CENTER = np.array([0.0, 200.0, 5.0])

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
# 2. 核心函数：计算遮蔽时长并返回辅助信息
#    (为了避免在辅助函数中重复计算，我们让主函数返回额外信息)
# =============================================================================


def calculate_shielding_info(params, dt=1e-3):
    """
    计算给定策略下的遮蔽信息。

    参数:
    params (list-like): [v_FY1, theta_FY1_rad, t_drop, t_fuse]
    dt (float): 数值模拟的时间步长

    返回:
    tuple: (total_shielding_time, P_detonate, t_detonate)
           - total_shielding_time: 总有效遮蔽时长 (s)
           - P_detonate: 烟幕起爆点的三维坐标
           - t_detonate: 烟幕起爆的时刻
    """
    v_FY1, theta_FY1_rad, t_drop, t_fuse = params

    # --- 轨迹与起爆点计算 ---
    d_FY1 = np.array([np.cos(theta_FY1_rad), np.sin(theta_FY1_rad), 0.0])
    P_drop = P_FY1_0 + v_FY1 * t_drop * d_FY1
    v_drop = v_FY1 * d_FY1
    P_detonate = P_drop + v_drop * t_fuse + np.array([0, 0, -0.5 * g * t_fuse**2])
    t_detonate = t_drop + t_fuse

    def P_C(t):
        return P_detonate + np.array([0, 0, -v_sink * (t - t_detonate)])

    # --- 模拟计算有效遮蔽时长 ---
    total_shielding_time = 0.0
    t_start = t_detonate
    t_end = t_detonate + 20.0

    for t in np.arange(t_start, t_end, dt):
        p_m1_t = P_M1(t)
        p_c_t = P_C(t)
        is_fully_shielded = True
        for p_key in key_points:
            distance = dist_point_to_segment(p_c_t, p_m1_t, p_key)
            if distance >= R_C:
                is_fully_shielded = False
                break
        if is_fully_shielded:
            total_shielding_time += dt

    return total_shielding_time, P_detonate, t_detonate


# =============================================================================
# 3. 方案C：融合后的目标函数与优化过程
# =============================================================================


def objective_function_C(params):
    """
    方案C的目标函数：
    - 如果遮蔽时长 > 0，返回 -时长 (最小化负值=最大化原值)
    - 如果遮蔽时长 = 0，返回辅助目标 (起爆点到视线的距离)
    """
    shielding_time, P_detonate, t_detonate = calculate_shielding_info(params)

    if shielding_time > 0:
        return -shielding_time
    else:
        # 计算辅助目标：起爆点到起爆时刻视线的距离
        p_m1_at_detonation = P_M1(t_detonate)
        proxy_distance = dist_point_to_segment(
            P_detonate, p_m1_at_detonation, P_TARGET_CENTER
        )
        return proxy_distance


def run_optimization_C():
    """执行方案C的完整优化流程"""
    print("--- 方案C：融合方案 (启发式边界 + 辅助目标函数) ---")

    # 1. 定义缩减后的搜索边界 (来自方案A)
    bounds = [
        (70, 140),  # v_FY1: 速度
        (np.pi * 170 / 180, np.pi * 190 / 180),  # θ_FY1: 角度 [90°, 270°]
        (0, 5),  # t_drop: 投放时间
        (0, 5),  # t_fuse: 引信时间
    ]

    # 2. 调用差分进化算法进行优化
    print("开始优化... (这可能需要几分钟，具体时间取决于您的CPU核心数)")
    start_time = time.time()

    result = differential_evolution(
        objective_function_C,
        bounds,
        strategy="best1bin",
        maxiter=200,
        popsize=20,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True,
        workers=-1,  # 利用所有CPU核心并行计算
    )

    end_time = time.time()
    print(f"\n优化完成，耗时: {end_time - start_time:.2f} 秒")

    # 3. 输出最终结果
    if result.success:
        optimal_params = result.x
        # 使用最优参数，最后精确计算一次真实的遮蔽时长
        final_shielding_time, _, _ = calculate_shielding_info(optimal_params)

        print("\n--- 最优策略 (方案C) ---")
        print(f"无人机飞行速度 (v_FY1): {optimal_params[0]:.2f} m/s")
        print(f"无人机飞行方向 (θ_FY1): {np.rad2deg(optimal_params[1]):.2f}°")
        print(f"烟幕弹投放时间 (t_drop): {optimal_params[2]:.2f} s")
        print(f"烟幕弹引信时间 (t_fuse): {optimal_params[3]:.2f} s")
        print("------------------------------------")
        print(f"最大有效遮蔽时长: {final_shielding_time:.4f} s")
        print("------------------------------------")
    else:
        print("\n优化未成功收敛。可以尝试增加 maxiter 或 popsize。")


# --- 主程序入口 ---
if __name__ == "__main__":
    run_optimization_C()
