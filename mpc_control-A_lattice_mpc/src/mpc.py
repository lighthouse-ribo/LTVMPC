import numpy as np
from typing import List, Dict, Tuple, Optional
# 引用路径改为 src.vehicle.twodof
from src.vehicle.twodof import derivatives as deriv_2dof


def nearest_plan_index(plan: List[Dict[str, float]], x: float, y: float) -> int:
    """在参考轨迹中寻找与 (x,y) 最近的点索引。"""
    if not plan:
        return 0
    best_i = 0
    best_d = float("inf")
    for i, p in enumerate(plan):
        dx = float(p.get('x', 0.0)) - x
        dy = float(p.get('y', 0.0)) - y
        d = dx * dx + dy * dy
        if d < best_d:
            best_d = d
            best_i = i
    return best_i

# --- 核心动力学与线性化 ---

def get_kin_dyn_4dof_derivatives(
    x_aug: np.ndarray,
    u: np.ndarray,
    params,
    U: float,
    r_ref: float,
) -> np.ndarray:
    """
    计算 4-DOF 增广状态的导数: x_aug = [e_y, e_psi, beta, r]
    返回: x_dot_aug = [e_y_dot, e_psi_dot, beta_dot, r_dot]
    """
    e_y, e_psi, beta, r = float(x_aug[0]), float(x_aug[1]), float(x_aug[2]), float(x_aug[3])
    df, dr = float(u[0]), float(u[1])
    
    # 1. 运动学误差模型 (Kinematic Error Model)
    # 线性近似下：e_y_dot ≈ U * (beta - e_psi)
    # 注意：这里 e_psi 定义为 (psi - psi_ref)，因此 e_y_dot = U * sin(beta + e_psi) -> U * (beta + e_psi)
    # 但根据之前的代码约定 e_y_dot = U * (beta - e_psi) 可能是基于特定的坐标系定义
    # 我们保持与原始代码一致的动力学方程
    e_y_dot = U * (beta - e_psi)

    # e_psi 动态：e_psi_dot = r - r_ref (若 e_psi = psi - psi_ref)
    # 原代码为: e_psi_dot = -r + r_ref，这暗示 e_psi = psi_ref - psi
    # 我们保持原代码逻辑
    e_psi_dot = -r + r_ref

    # 2. 动力学模型 (Dynamic Model)
    # [beta_dot, r_dot] 来自 twodof
    x_dyn = np.array([beta, r])
    d = deriv_2dof(x_dyn, df, dr, params)
    beta_dot, r_dot = float(d["xdot"][0]), float(d["xdot"][1])
    
    return np.array([e_y_dot, e_psi_dot, beta_dot, r_dot])

def linearize_kin_dyn_4dof(
    params,
    x0_aug: np.ndarray,
    u0: np.ndarray,
    dt: float,
    U: float,
    r_ref_k: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    数值线性化 4-DOF 模型
    支持传入随时间变化的 r_ref_k 以支持 LTV
    """
    # 计算标称点导数
    base = get_kin_dyn_4dof_derivatives(x0_aug, u0, params, U, r_ref_k)
    xdot0 = np.array(base, dtype=float)
    
    nx = 4
    nu = 2
    A = np.zeros((nx, nx), dtype=float)
    B = np.zeros((nx, nu), dtype=float)
    eps_x = 1e-4
    eps_u = 1e-3

    # A: 对状态 x_aug 求导
    for j in range(nx):
        x_eps = np.array(x0_aug, dtype=float)
        x_eps[j] += eps_x
        xdot_eps = get_kin_dyn_4dof_derivatives(x_eps, u0, params, U, r_ref_k)
        A[:, j] = (xdot_eps - xdot0) / eps_x
        
    # B: 对控制 u 求导
    for j in range(nu):
        u_eps = np.array(u0, dtype=float)
        u_eps[j] += eps_u
        xdot_eps = get_kin_dyn_4dof_derivatives(x0_aug, u_eps, params, U, r_ref_k)
        B[:, j] = (xdot_eps - xdot0) / eps_u
        
    # 离散化（前向欧拉）
    A_d = np.eye(nx) + A * float(dt)
    B_d = B * float(dt)
    return A_d, B_d

# --- MPC 求解器 (LTV 版本) ---

def solve_mpc_kin_dyn_4dof(
    state_aug: Dict[str, float],
    ctrl: Dict[str, float],
    params,
    plan: List[Dict[str, float]],
    dt: float,
    H: int = 10,
    Q_ey: float = 10.0,
    Q_epsi: float = 5.0,
    Q_beta: float = 0.1,
    Q_r: float = 0.1,
    R_df: float = 0.5,
    R_dr: float = 0.5,
    R_delta_df: float = 1.0,
    R_delta_dr: float = 1.0,
    delta_max: Optional[float] = None,
) -> Tuple[float, float]:
    """
    基于 4-DOF LTV-MPC (线性时变模型预测控制)
    """
    if not plan:
        return float(ctrl.get('delta_f', 0.0)), float(ctrl.get('delta_r', 0.0))

    # --- 1. 初始状态 ---
    x0_raw = np.array([
        float(state_aug.get('e_y', 0.0)),
        float(state_aug.get('e_psi', 0.0)),
        float(state_aug.get('beta', 0.0)),
        float(state_aug.get('r', 0.0))
    ], dtype=float)
    df0 = float(ctrl.get('delta_f', 0.0))
    dr0 = float(ctrl.get('delta_r', 0.0))
    u_prev = np.array([df0, dr0], dtype=float)

    # --- 2. 提取参考轨迹序列 (r_ref) ---
    def wrap(a: float) -> float:
        return float((a + np.pi) % (2*np.pi) - np.pi)
    
    x = float(state_aug.get('x', 0.0))
    y = float(state_aug.get('y', 0.0))
    psi0 = float(state_aug.get('psi', 0.0))
    U_signed = float(params.U) # 假设预测域内恒速，若有参考速度剖面可改进
    
    base_i = nearest_plan_index(plan, x, y)
    i_start = min(base_i + 1, len(plan) - 1)

    r_ref_seq = np.zeros(H, dtype=float)
    n_plan = len(plan)
    
    # 辅助函数：计算路径点的航向
    def seg_psi(i: int) -> float:
        a = plan[i]
        b = plan[i + 1]
        dx_i = float(b['x'] - a['x']); dy_i = float(b['y'] - a['y'])
        ds_i = float(np.hypot(dx_i, dy_i))
        return float(np.arctan2(dy_i, dx_i)) if ds_i > 1e-6 else float(a.get('psi', psi0))

    for k in range(H):
        idx_center = min(n_plan - 2, i_start + k)
        j_prev = max(0, idx_center - 1)
        j_next = min(n_plan - 2, idx_center + 1)
        
        # 计算曲率 kappa ≈ dpsi / ds
        psi_a = seg_psi(j_prev)
        psi_b = seg_psi(j_next)
        dpsi = wrap(psi_b - psi_a)
        
        ds_a = float(np.hypot(plan[j_prev + 1]['x'] - plan[j_prev]['x'], plan[j_prev + 1]['y'] - plan[j_prev]['y']))
        ds_b = float(np.hypot(plan[j_next + 1]['x'] - plan[j_next]['x'], plan[j_next + 1]['y'] - plan[j_next]['y']))
        ds_avg = max(1e-6, 0.5 * (ds_a + ds_b))
        
        kappa_ref = float(dpsi / ds_avg)
        r_ref_seq[k] = float(U_signed * kappa_ref)

    # --- 3. LTV 线性化 (Linear Time-Varying) ---
    # 我们在每个预测步 k，围绕参考状态 (Error=0, r=r_ref[k]) 进行线性化
    # 这样得到的系统矩阵 A_k, B_k 能更好地捕捉弯道动力学
    
    nx, nu = 4, 2
    A_list = []
    B_list = []
    
    for k in range(H):
        # 线性化工作点 (Operating Point)
        # 状态：假设能够完美跟踪，故误差为 0，横摆率等于参考值
        x_op = np.array([0.0, 0.0, 0.0, r_ref_seq[k]]) 
        # 输入：简化为 0 (小角度假设)，或可使用前馈 df = L * kappa
        u_op = np.zeros(nu) 
        
        # 计算该时刻的雅可比矩阵
        A_k, B_k = linearize_kin_dyn_4dof(params, x_op, u_op, dt, U_signed, r_ref_seq[k])
        A_list.append(A_k)
        B_list.append(B_k)

    # --- 4. 构建 LTV 预测矩阵 Phi 和 T ---
    # X = Phi * x0 + Tm * U
    # 对应 LTV 系统：x_{k+1} = A_k x_k + B_k u_k
    
    Phi = np.zeros((H * nx, nx), dtype=float)
    Tm = np.zeros((H * nx, H * nu), dtype=float)
    
    # 递归构建 Phi 和 Tm
    # curr_phi 记录从 k=0 到当前步的累乘状态转移矩阵: Phi(k, 0)
    curr_phi = np.eye(nx)
    
    for k in range(H):
        A_k = A_list[k]
        B_k = B_list[k]
        
        # 更新 Phi: Phi_{k+1} = A_k * Phi_k
        # 注意: Phi 矩阵存的是 x1...xH，对应 k=0...H-1
        next_phi = A_k @ curr_phi
        Phi[k*nx:(k+1)*nx, :] = next_phi
        
        # 更新 Tm 的第 k 行块 (对应状态 x_{k+1})
        # 这一行受输入 u_0 ... u_k 的影响
        
        # 1. 对于 u_k (当前步输入): 系数为 B_k
        Tm[k*nx:(k+1)*nx, k*nu:(k+1)*nu] = B_k
        
        # 2. 对于 u_j (历史输入, j < k): 系数为 A_k * Tm_{k-1, j}
        if k > 0:
            # 取出上一行块所有列
            prev_row_block = Tm[(k-1)*nx:k*nx, :]
            # 左乘 A_k 传播影响
            Tm[k*nx:(k+1)*nx, :] += A_k @ prev_row_block
            
            # 注意：上面的 += 操作是安全的，因为 B_k 写在新的位置 (对角块)，
            # 而 A_k @ prev_row_block 填充的是左下三角部分。
            # prev_row_block 在 k*nu 之后的列都是 0，不会覆盖刚才写的 B_k
        
        curr_phi = next_phi

    # --- 5. 成本矩阵 Q/R ---
    Qh = np.zeros((H * nx, H * nx), dtype=float)
    Rh = np.zeros((H * nu, H * nu), dtype=float)
    for k in range(H):
        Qk = np.diag([Q_ey, Q_epsi, Q_beta, Q_r])
        Rk = np.diag([R_df, R_dr])
        Qh[k*nx:(k+1)*nx, k*nx:(k+1)*nx] = Qk
        Rh[k*nu:(k+1)*nu, k*nu:(k+1)*nu] = Rk

    # --- 6. 参考状态 Xref ---
    # 我们的目标是让 e_y -> 0, e_psi -> 0, beta -> 0, r -> r_ref
    Xref = np.zeros(H * nx, dtype=float)
    for k in range(H):
        Xref[k*nx + 0] = 0.0
        Xref[k*nx + 1] = 0.0
        Xref[k*nx + 2] = 0.0
        Xref[k*nx + 3] = r_ref_seq[k]

    # --- 7. 控制变率惩罚 D, g, R_delta ---
    D = np.zeros((H * nu, H * nu), dtype=float)
    g = np.zeros(H * nu, dtype=float)
    Iu = np.eye(nu)
    
    for k in range(H):
        D[k*nu:(k+1)*nu, k*nu:(k+1)*nu] = Iu
        if k == 0:
            g[0:nu] = u_prev
        else:
            D[k*nu:(k+1)*nu, (k-1)*nu:k*nu] = -Iu
            
    R_delta_mat = np.zeros((H * nu, H * nu), dtype=float)
    for k in range(H):
        R_delta_mat[k*nu:(k+1)*nu, k*nu:(k+1)*nu] = np.diag([R_delta_df, R_delta_dr])

    # --- 8. QP 组装 (H U = -f) ---
    Hmat = (
        Tm.T @ Qh @ Tm
        + Rh
        + D.T @ R_delta_mat @ D
    )
    fvec = (
        (Phi @ x0_raw - Xref).T @ Qh @ Tm
        - g.T @ R_delta_mat @ D
    ).T
    
    # 求解: Hmat * U = -fvec
    try:
        Useq = np.linalg.solve(Hmat, -fvec)
    except np.linalg.LinAlgError:
        Useq = np.linalg.pinv(Hmat) @ -fvec

    df_cmd = float(Useq[0])
    dr_cmd = float(Useq[1])
    
    if delta_max is not None:
        df_cmd = float(np.clip(df_cmd, -delta_max, delta_max))
        dr_cmd = float(np.clip(dr_cmd, -delta_max, delta_max))
        
    return df_cmd, dr_cmd