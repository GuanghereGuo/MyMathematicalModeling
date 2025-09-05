import numpy as np
from scipy.optimize import differential_evolution
import time

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
    np.array([7, 207.0, 0.0]), np.array([7, 193.0, 0.0]),
    np.array([-7, 207.0, 0.0]), np.array([-7, 193.0, 0.0]),
    np.array([7, 207.0, 10.0]), np.array([7, 193.0, 10.0]),
    np.array([-7, 207.0, 10.0]), np.array([-7, 193.0, 10.0]),
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
    if c1 <= 0: return np.linalg.norm(p - a)
    c2 = np.dot(v, v)
    if c2 <= c1: return np.linalg.norm(p - b)
    b_proj = c1 / c2
    pb = a + b_proj * v
    return np.linalg.norm(p - pb)


# =============================================================================
# 2. 计算单枚烟幕弹遮蔽区间的函数
# =============================================================================

def get_shielding_intervals(v_FY1, theta_rad, t_drop, t_fuse, P_FY1_0, dt=1e-2):
    """计算单枚烟幕弹产生的有效遮蔽时间区间列表"""
    d_FY1 = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])
    P_drop = P_FY1_0 + v_FY1 * t_drop * d_FY1
    v_drop = v_FY1 * d_FY1
    P_detonate = P_drop + v_drop * t_fuse + np.array([0, 0, -0.5 * g * t_fuse ** 2])
    t_detonate = t_drop + t_fuse

    def P_C(t):
        return P_detonate + np.array([0, 0, -v_sink * (t - t_detonate)])

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

    all_intervals.extend(get_shielding_intervals(v_FY1, theta_rad, t_drop1, t_fuse1, P_FY1_0))
    all_intervals.extend(get_shielding_intervals(v_FY1, theta_rad, t_drop2, t_fuse2, P_FY1_0))
    all_intervals.extend(get_shielding_intervals(v_FY1, theta_rad, t_drop3, t_fuse3, P_FY1_0))

    total_length = calculate_union_length(all_intervals)

    return -total_length


def run_optimization_Q3():
    """执行问题三的完整优化流程"""
    print("--- 问题三：单无人机三弹药优化 ---")

    bounds = [
        (70, 140), # v_FY1: 速度
        (np.pi * 170 / 180, np.pi * 190 / 180), # theta: 方向
        (0, 5), # t_drop1: 投放时间
        (0, 5), # t_fuse1: 引信时间
        (1, 5), # dt_drop2: 第二枚相对第一枚的投放时间差
        (0, 5), # t_fuse2: 第二枚引信时间
        (1, 5), # dt_drop3: 第三枚相对第二枚的投放时间差
        (0, 5) # t_fuse3: 第三枚引信时间
    ]

    print("开始优化 (8维空间，计算量较大，请耐心等待)...")
    start_time = time.time()

    result = differential_evolution(
        objective_function_Q3, bounds, strategy='best1bin', maxiter=300,
        popsize=25, tol=0.01, disp=True, workers=-1
    )

    end_time = time.time()
    print(f"\n优化完成，耗时: {end_time - start_time:.2f} 秒")

    if result.success:
        p = result.x
        max_time = -result.fun

        # --- 结果处理与计算 ---
        v_opt, theta_opt_rad = p[0], p[1]
        direction_deg = np.rad2deg(theta_opt_rad) % 360
        d_FY1 = np.array([np.cos(theta_opt_rad), np.sin(theta_opt_rad), 0.0])

        t_drop1 = p[2]
        t_drop2 = p[2] + p[4]
        t_drop3 = t_drop2 + p[6]

        params_list = [(t_drop1, p[3]), (t_drop2, p[5]), (t_drop3, p[7])]

        grenade_data = []
        for i, (t_drop, t_fuse) in enumerate(params_list):
            p_drop = P_FY1_0 + v_opt * t_drop * d_FY1
            v_drop = v_opt * d_FY1
            p_detonate = p_drop + v_drop * t_fuse + np.array([0, 0, -0.5 * g * t_fuse ** 2])
            grenade_data.append({
                "id": f"G{i + 1}",
                "p_drop": p_drop,
                "p_detonate": p_detonate
            })

        # --- 按指定格式在控制台输出结果 ---
        print("\n\n" + "=" * 80)
        print(" " * 25 + "问题三 最优策略详细结果")
        print("=" * 80)

        # 定义列标题和宽度
        header_fmt = "{:<35} {:>15} {:>15} {:>15}"
        row_fmt = "{:<35} {:>15.4f} {:>15.4f} {:>15.4f}"
        id_fmt = "{:<35} {:>15} {:>15} {:>15}"

        print(header_fmt.format('', '1', '2', '3'))
        print("-" * 80)

        print(row_fmt.format('无人机运动方向 (度)', direction_deg, direction_deg, direction_deg))
        print(row_fmt.format('无人机运动速度 (m/s)', v_opt, v_opt, v_opt))
        print(id_fmt.format('烟幕干扰弹编号', grenade_data[0]['id'], grenade_data[1]['id'], grenade_data[2]['id']))
        print(row_fmt.format('烟幕干扰弹投放点的x坐标 (m)', grenade_data[0]['p_drop'][0], grenade_data[1]['p_drop'][0],
                             grenade_data[2]['p_drop'][0]))
        print(row_fmt.format('烟幕干扰弹投放点的y坐标 (m)', grenade_data[0]['p_drop'][1], grenade_data[1]['p_drop'][1],
                             grenade_data[2]['p_drop'][1]))
        print(row_fmt.format('烟幕干扰弹投放点的z坐标 (m)', grenade_data[0]['p_drop'][2], grenade_data[1]['p_drop'][2],
                             grenade_data[2]['p_drop'][2]))
        print(row_fmt.format('烟幕干扰弹起爆点的x坐标 (m)', grenade_data[0]['p_detonate'][0],
                             grenade_data[1]['p_detonate'][0], grenade_data[2]['p_detonate'][0]))
        print(row_fmt.format('烟幕干扰弹起爆点的y坐标 (m)', grenade_data[0]['p_detonate'][1],
                             grenade_data[1]['p_detonate'][1], grenade_data[2]['p_detonate'][1]))
        print(row_fmt.format('烟幕干扰弹起爆点的z坐标 (m)', grenade_data[0]['p_detonate'][2],
                             grenade_data[1]['p_detonate'][2], grenade_data[2]['p_detonate'][2]))
        print("{:<35} {:>15.4f}".format('有效干扰时长 (s)', max_time))

        print("-" * 80)
        print('注：以x轴为正向，逆时针方向为正，取值0~360（度）。')
        print("=" * 80 + "\n")

    else:
        print("\n优化未成功收敛。可以尝试增加 maxiter 或 popsize。")


# --- 主程序入口 ---
if __name__ == '__main__':
    run_optimization_Q3()
