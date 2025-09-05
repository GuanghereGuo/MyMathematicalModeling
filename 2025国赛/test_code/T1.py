import numpy as np


def T_shielding(v_FY1, theta_FY1_rad, t_drop, t_fuse, dt=1e-4):
    P_M1_0 = np.array([20000.0, 0.0, 2000.0])  # 导弹初始位置
    v_M = 300.0  # 导弹速度
    P_FY1_0 = np.array([17800.0, 0.0, 1800.0])  # 无人机1初始位置
    R_C = 10.0  # 烟雾弹半径
    v_sink = 3.0  # 烟雾下降速度
    g = 9.8  # 重力加速度

    # 定义关键点列表
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

    # 导弹单位方向向量
    d_M1 = (np.array([0.0, 0.0, 0.0]) - P_M1_0) / np.linalg.norm(
        np.array([0.0, 0.0, 0.0]) - P_M1_0
    )

    def P_M1(t):
        return P_M1_0 + v_M * t * d_M1

    # 无人机1单位方向向量
    d_FY1 = np.array([np.cos(theta_FY1_rad), np.sin(theta_FY1_rad), 0.0])

    def P_FY1(t):
        return P_FY1_0 + v_FY1 * t * d_FY1

    # 引爆时间点
    t_detonate = t_drop + t_fuse

    # 计算投放点位置
    P_drop = P_FY1(t_drop)
    v_drop = v_FY1 * d_FY1
    P_detonate = P_drop + v_drop * t_fuse + np.array([0, 0, -0.5 * g * t_fuse**2])

    def P_C(t):
        if t < t_detonate:
            return None
        return P_detonate + np.array([0, 0, -v_sink * (t - t_detonate)])

    # 计算几何魅力时刻
    def dist_point_to_segment(p, a, b):
        v = b - a  # 向量 AB
        w = p - a  # 向量 AP
        c1 = np.dot(w, v)  # AP 在 AB 上的投影长度
        if c1 <= 0:  # P 在 A 点前方
            return np.linalg.norm(p - a)
        c2 = np.dot(v, v)
        if c2 <= c1:  # P 在 B 点后方
            return np.linalg.norm(p - b)
        b_proj = c1 / c2
        pb = a + b_proj * v  # P 在 AB 上的投影点
        return np.linalg.norm(p - pb)

    # 计算有效遮蔽时长
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

    return total_shielding_time


if __name__ == "__main__":

    # --- 测试案例1：使用问题一的参数 ---
    print("--- 测试问题一的参数 ---")
    v_q1 = 120
    # "朝向假目标方向" 在xy平面上是沿x轴负方向，即180度或pi弧度
    theta_q1_rad = np.pi
    tdrop_q1 = 1.5
    tfuse_q1 = 3.6

    shielding_time_q1 = T_shielding(v_q1, theta_q1_rad, tdrop_q1, tfuse_q1)
    print(f"使用问题一的参数，计算得到的有效遮蔽时长: {shielding_time_q1:.4f} s\n")

    print("--- 测试问题二的参数 ---")
    # v_q1 = 86.82
    # # "朝向假目标方向" 在xy平面上是沿x轴负方向，即180度或pi弧度
    # theta_q1_rad = 178.70 / 180 * np.pi
    # tdrop_q1 = 1.01
    # tfuse_q1 = 3.15
    v_q1 = 70
    theta_q1_rad = 176.60 / 180 * np.pi
    tdrop_q1 = 0
    tfuse_q1 = 2.45

    shielding_time_q1 = T_shielding(v_q1, theta_q1_rad, tdrop_q1, tfuse_q1)
    print(f"使用问题二的参数，计算得到的有效遮蔽时长: {shielding_time_q1:.4f} s\n")
