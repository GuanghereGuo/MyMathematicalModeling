import numpy as np
from scipy.optimize import differential_evolution
import time

# =============================================================================
# 1. 全局常量与几何计算函数 (基础模块)
# =============================================================================

# 物理和环境参数
R_C = 10.0  # 烟幕云团半径
v_sink = 3.0  # 烟雾下降速度
g = 9.8  # 重力加速度

# 初始位置定义
P_FY1_0 = np.array([17800.0, 0.0, 1800.0])
P_FY2_0 = np.array([17800.0, 200.0, 1800.0])
P_FY3_0 = np.array([17800.0, -200.0, 1800.0])
DRONE_INITIAL_POS = [P_FY1_0, P_FY2_0, P_FY3_0]

P_M1_0 = np.array([20000.0, 0.0, 2000.0])
P_M2_0 = np.array([20000.0, 200.0, 2000.0])
P_M3_0 = np.array([20000.0, -200.0, 2000.0])
v_M = 300.0

# 真目标关键点列表 (8点外接盒模型)
key_points = [
    np.array([7, 207.0, 0.0]), np.array([7, 193.0, 0.0]),
    np.array([-7, 207.0, 0.0]), np.array([-7, 193.0, 0.0]),
    np.array([7, 207.0, 10.0]), np.array([7, 193.0, 10.0]),
    np.array([-7, 207.0, 10.0]), np.array([-7, 193.0, 10.0]),
]


# 导弹轨迹函数
def get_missile_trajectory_func(p_m_0):
    """根据导弹初始位置，返回其轨迹函数"""
    d_m = (np.array([0.0, 0.0, 0.0]) - p_m_0) / np.linalg.norm(p_m_0)

    def P_M(t):
        return p_m_0 + v_M * t * d_m

    return P_M


P_M1_func = get_missile_trajectory_func(P_M1_0)
P_M2_func = get_missile_trajectory_func(P_M2_0)
P_M3_func = get_missile_trajectory_func(P_M3_0)
MISSILE_TRAJECTORY_FUNCS = [P_M1_func, P_M2_func, P_M3_func]


# 几何计算函数
def dist_point_to_segment(p, a, b):
    v = b - a;
    w = p - a
    c1 = np.dot(w, v)
    if c1 <= 0: return np.linalg.norm(p - a)
    c2 = np.dot(v, v)
    if c2 <= c1: return np.linalg.norm(p - b)
    b_proj = c1 / c2
    pb = a + b_proj * v
    return np.linalg.norm(p - pb)


# =============================================================================
# 2. 核心计算函数 (泛化版本)
# =============================================================================

def get_shielding_intervals(drone_params, p_fy_0, p_m_func, dt=1e-2):
    """计算单枚烟幕弹对单枚导弹的遮蔽区间列表"""
    v_fy, theta_rad, t_drop, t_fuse = drone_params

    d_fy = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])
    p_drop = p_fy_0 + v_fy * t_drop * d_fy
    v_drop = v_fy * d_fy
    p_detonate = p_drop + v_drop * t_fuse + np.array([0, 0, -0.5 * g * t_fuse ** 2])
    t_detonate = t_drop + t_fuse

    def p_c(t):
        return p_detonate + np.array([0, 0, -v_sink * (t - t_detonate)])

    shielded_times = []
    t_start, t_end = t_detonate, t_detonate + 20.0

    for t in np.arange(t_start, t_end, dt):
        p_m_t = p_m_func(t)
        p_c_t = p_c(t)
        is_fully_shielded = all(dist_point_to_segment(p_c_t, p_m_t, p_key) < R_C for p_key in key_points)
        if is_fully_shielded:
            shielded_times.append(t)

    if not shielded_times: return []

    intervals = []
    start_of_interval = shielded_times[0]
    for i in range(1, len(shielded_times)):
        if shielded_times[i] - shielded_times[i - 1] > dt * 1.5:
            intervals.append((start_of_interval, shielded_times[i - 1]))
            start_of_interval = shielded_times[i]
    intervals.append((start_of_interval, shielded_times[-1]))

    return intervals


def calculate_union_length(intervals):
    """计算区间列表并集的总长度"""
    if not intervals: return 0
    intervals.sort(key=lambda x: x[0])
    merged = [];
    current_start, current_end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged.append((current_start, current_end))
    return sum(end - start for start, end in merged)


# =============================================================================
# 3. 问题四的目标函数与优化过程
# =============================================================================

def objective_function_Q4(params):
    """问题四的目标函数"""
    # 将12维向量拆分为3组无人机参数
    drone1_params = params[0:4]
    drone2_params = params[4:8]
    drone3_params = params[8:12]
    all_drone_params = [drone1_params, drone2_params, drone3_params]

    total_shielding_sum = 0.0

    # 对每一枚导弹计算其总遮蔽时长
    for p_m_func in MISSILE_TRAJECTORY_FUNCS:
        intervals_for_this_missile = []
        # 累加来自三架无人机的遮蔽效果
        for i in range(3):
            intervals_for_this_missile.extend(
                get_shielding_intervals(all_drone_params[i], DRONE_INITIAL_POS[i], p_m_func)
            )

        # 计算该导弹的总遮蔽时长（并集长度）
        length_for_this_missile = calculate_union_length(intervals_for_this_missile)
        total_shielding_sum += length_for_this_missile

    return -total_shielding_sum


def run_optimization_Q4():
    """执行问题四的完整优化流程"""
    print("--- 问题四：三无人机 vs 三导弹协同优化 ---")

    # 定义12个决策变量的搜索边界
    # 每4个为一组: [v, θ, t_drop, t_fuse]
    bounds = [
        # FY1
        (70, 140), (np.pi * 165 / 180, np.pi * 190 / 180), (0, 5), (0, 5),
        # FY2
        (70, 140), (np.pi * 165 / 180, np.pi * 190 / 180), (0, 5), (0, 5),
        # FY3
        (70, 140), (np.pi * 165 / 180, np.pi * 190 / 180), (0, 5), (0, 5),
    ]

    print("开始优化 (12维空间，计算量极大，请耐心等待)...")
    start_time = time.time()

    result = differential_evolution(
        objective_function_Q4,
        bounds,
        strategy='best1bin',
        maxiter=400,  # 增加迭代次数
        popsize=30,  # 增加种群规模
        tol=0.01,
        disp=True,
        workers=-1  # 使用所有CPU核心并行计算
    )

    end_time = time.time()
    print(f"\n优化完成，耗时: {end_time - start_time:.2f} 秒")

    if result.success:
        print_results_Q4(result.x, -result.fun)
    else:
        print("\n优化未成功收敛。这是一个非常复杂的问题，可以尝试进一步增加 maxiter 或 popsize。")


def print_results_Q4(params, total_max_time):
    """在控制台按格式打印问题四的最优结果"""
    drone1_params = params[0:4]
    drone2_params = params[4:8]
    drone3_params = params[8:12]
    all_drone_params = [drone1_params, drone2_params, drone3_params]

    print("\n\n" + "=" * 60)
    print(" " * 15 + "问题四 最优协同策略")
    print("=" * 60)
    print(f"总有效遮蔽时长 (三枚导弹合计): {total_max_time:.4f} s")
    print("=" * 60)

    for i in range(3):
        v, theta_rad, t_drop, t_fuse = all_drone_params[i]
        direction_deg = np.rad2deg(theta_rad) % 360

        print(f"\n--- 无人机 FY{i + 1} 策略 ---")
        print(f"  - 飞行速度: {v:.4f} m/s")
        print(f"  - 飞行方向: {direction_deg:.4f}°")
        print(f"  - 投放时间: {t_drop:.4f} s")
        print(f"  - 引信时间: {t_fuse:.4f} s")

    print("\n" + "=" * 60)
    print('注：飞行方向以x轴为正向，逆时针为正，取值0~360（度）。')
    print("=" * 60 + "\n")


# --- 主程序入口 ---
if __name__ == '__main__':
    run_optimization_Q4()
