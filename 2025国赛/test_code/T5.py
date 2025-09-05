import numpy as np
from scipy.optimize import differential_evolution
import time

# =============================================================================
# 1. 全局常量与几何计算函数 (基础模块)
# =============================================================================

# 物理和环境参数
R_C = 10.0;
v_sink = 3.0;
g = 9.8

# 初始位置定义
P_FY_0_ALL = [
    np.array([17800.0, 0.0, 1800.0]),  # FY1
    np.array([17800.0, 200.0, 1800.0]),  # FY2
    np.array([17800.0, -200.0, 1800.0]),  # FY3
    np.array([17800.0, 400.0, 1800.0]),  # FY4
    np.array([17800.0, -400.0, 1800.0])  # FY5
]

P_M_0_ALL = [
    np.array([20000.0, 0.0, 2000.0]),  # M1
    np.array([20000.0, 200.0, 2000.0]),  # M2
    np.array([20000.0, -200.0, 2000.0])  # M3
]
v_M = 300.0

# 真目标关键点列表
key_points = [
    np.array([7, 207., 0.]), np.array([7, 193., 0.]), np.array([-7, 207., 0.]), np.array([-7, 193., 0.]),
    np.array([7, 207., 10.]), np.array([7, 193., 10.]), np.array([-7, 207., 10.]), np.array([-7, 193., 10.]),
]


# 预计算导弹轨迹函数
def get_missile_trajectory_func(p_m_0):
    d_m = (np.array([0., 0., 0.]) - p_m_0) / np.linalg.norm(p_m_0)
    return lambda t: p_m_0 + v_M * t * d_m


MISSILE_TRAJECTORY_FUNCS = [get_missile_trajectory_func(p) for p in P_M_0_ALL]


# 几何与区间计算函数 (与之前相同)
def dist_point_to_segment(p, a, b):
    v = b - a;
    w = p - a;
    c1 = np.dot(w, v)
    if c1 <= 0: return np.linalg.norm(p - a)
    c2 = np.dot(v, v)
    if c2 <= c1: return np.linalg.norm(p - b)
    return np.linalg.norm(p - (a + (c1 / c2) * v))


def calculate_union_length(intervals):
    if not intervals: return 0
    intervals.sort(key=lambda x: x[0])
    merged = [];
    current_start, current_end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end)); current_start, current_end = next_start, next_end
    merged.append((current_start, current_end))
    return sum(end - start for start, end in merged)


# =============================================================================
# 2. 核心计算函数 (泛化版本)
# =============================================================================

def get_shielding_intervals(grenade_full_params, p_m_func, dt=0.1):
    """计算单枚烟幕弹对单枚导弹的遮蔽区间列表"""
    p_fy_0, v_fy, theta_rad, t_drop, t_fuse = grenade_full_params

    d_fy = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])
    p_drop = p_fy_0 + v_fy * t_drop * d_fy
    p_detonate = p_drop + v_fy * d_fy * t_fuse + np.array([0, 0, -0.5 * g * t_fuse ** 2])
    t_detonate = t_drop + t_fuse

    def p_c(t):
        return p_detonate + np.array([0, 0, -v_sink * (t - t_detonate)])

    shielded_times = []
    for t in np.arange(t_detonate, t_detonate + 20.0, dt):
        if all(dist_point_to_segment(p_c(t), p_m_func(t), p_key) < R_C for p_key in key_points):
            shielded_times.append(t)

    if not shielded_times: return []

    intervals = []
    start = shielded_times[0]
    for i in range(1, len(shielded_times)):
        if shielded_times[i] - shielded_times[i - 1] > dt * 1.5:
            intervals.append((start, shielded_times[i - 1]));
            start = shielded_times[i]
    intervals.append((start, shielded_times[-1]))
    return intervals


# =============================================================================
# 3. 问题五的目标函数与优化过程
# =============================================================================

def objective_function_Q5(params, dt):
    """问题五的目标函数，dt为模拟精度"""
    total_shielding_sum = 0.0

    all_grenade_params = []
    for i in range(5):
        drone_params = params[i * 8: (i + 1) * 8]
        p_fy_0 = P_FY_0_ALL[i]
        v_fy, theta_rad, t_d1, t_f1, dt_d2, t_f2, dt_d3, t_f3 = drone_params

        t_d2 = t_d1 + dt_d2
        t_d3 = t_d2 + dt_d3

        all_grenade_params.append((p_fy_0, v_fy, theta_rad, t_d1, t_f1))
        all_grenade_params.append((p_fy_0, v_fy, theta_rad, t_d2, t_f2))
        all_grenade_params.append((p_fy_0, v_fy, theta_rad, t_d3, t_f3))

    for p_m_func in MISSILE_TRAJECTORY_FUNCS:
        intervals_for_this_missile = []
        for grenade_params in all_grenade_params:
            intervals_for_this_missile.extend(
                get_shielding_intervals(grenade_params, p_m_func, dt)
            )
        total_shielding_sum += calculate_union_length(intervals_for_this_missile)

    return -total_shielding_sum


def run_optimization_Q5():
    """执行问题五的完整优化流程"""
    print("--- 问题五：五无人机(15弹药) vs 三导弹协同优化 ---")

    drone_bounds = [
        (70, 140), (np.pi * 165 / 180, np.pi * 190 / 180),  # v, theta
        (0, 5), (0, 5), # t_d1, t_f1
        (1, 5), (0, 5), # dt_d2, t_f2
        (1, 5), (0, 5) # dt_d3, t_f3
    ]
    bounds = drone_bounds * 5

    print("开始优化 (40维空间，计算量极大，将采用两阶段精度策略)...")
    start_time = time.time()

    # 第一阶段：使用低精度(dt=0.1)快速探索
    dt_fast = 0.1
    print(f"\n[阶段一] 使用低精度(dt={dt_fast})进行快速探索...")

    # --- 这是修改的核心部分 ---
    # 之前错误的写法:
    # result = differential_evolution(lambda p: objective_function_Q5(p, dt=dt_fast), ...)
    #
    # 正确的写法:
    # 直接传递目标函数，并使用 args 参数传递额外的 dt_fast
    # 注意 args 必须是一个元组，所以是 (dt_fast,)
    result = differential_evolution(
        objective_function_Q5,
        bounds,
        args=(dt_fast,),  # <--- 关键修复
        strategy='best1bin',
        maxiter=10000,
        popsize=60,
        tol=0.01,
        disp=True,
        workers=-1
    )

    end_time = time.time()
    print(f"\n优化完成，总耗时: {end_time - start_time:.2f} 秒")

    if result.success:
        optimal_params = result.x

        # 第二阶段：使用高精度(dt=0.01)验证最终结果
        dt_accurate = 0.01
        print(f"\n[阶段二] 使用高精度(dt={dt_accurate})验证最优解...")
        final_max_time = -objective_function_Q5(optimal_params, dt=dt_accurate)

        print_results_Q5(optimal_params, final_max_time)
    else:
        print("\n优化未成功收敛。这是一个极具挑战性的问题，可尝试继续增加 maxiter 或 popsize。")


def print_results_Q5(params, total_max_time):
    """在控制台打印问题五的最优结果"""
    print("\n\n" + "=" * 70)
    print(" " * 20 + "问题五 最优协同策略")
    print("=" * 70)
    print(f"总有效遮蔽时长 (三枚导弹合计): {total_max_time:.4f} s")
    print("=" * 70)

    for i in range(5):
        drone_params = params[i * 8: (i + 1) * 8]
        v, theta_rad, t_d1, t_f1, dt_d2, t_f2, dt_d3, t_f3 = drone_params
        direction_deg = np.rad2deg(theta_rad) % 360
        t_d2 = t_d1 + dt_d2
        t_d3 = t_d2 + dt_d3

        print(f"\n--- 无人机 FY{i + 1} 策略 ---")
        print(f"  - 共享参数: 速度={v:.2f} m/s, 方向={direction_deg:.2f}°")
        print(f"  - 弹药1: 投放时间={t_d1:.2f}s, 引信={t_f1:.2f}s")
        print(f"  - 弹药2: 投放时间={t_d2:.2f}s, 引信={t_f2:.2f}s")
        print(f"  - 弹药3: 投放时间={t_d3:.2f}s, 引信={t_f3:.2f}s")

    print("\n" + "=" * 70)
    print('注：飞行方向以x轴为正向，逆时针为正，取值0~360（度）。')
    print("=" * 70 + "\n")


# --- 主程序入口 ---
if __name__ == '__main__':
    run_optimization_Q5()
