import numpy as np
import math
import copy
from .curves import CubicSpline2D
from .lattice_config import LatticeConfig
# [新增] 导入后端优化器 (带容错处理)
try:
    from .backend_optimizer import CfsOptimizer
except ImportError:
    print("Warning: CfsOptimizer not found. Backend optimization will be disabled.")
    CfsOptimizer = None

# ================= 向量化数学工具 (保留高效实现) =================

def calc_quintic_coeffs_vectorized(xs, vxs, axs, xe, vxe, axe, T):
    """五次多项式系数计算 (Vectorized)"""
    T = np.atleast_1d(T)
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    
    a0 = np.full_like(T, xs)
    a1 = np.full_like(T, vxs)
    a2 = np.full_like(T, axs / 2.0)
    
    h = xe - a0 - a1*T - a2*T2
    v = vxe - a1 - 2*a2*T
    a = axe - 2*a2
    
    inv_T3 = 1.0 / (T3 + 1e-6)
    inv_T4 = 1.0 / (T4 + 1e-6)
    inv_T5 = 1.0 / (T5 + 1e-6)

    a3 = (10 * h - 4 * v * T + 0.5 * a * T2) * inv_T3
    a4 = (-15 * h + 7 * v * T - a * T2) * inv_T4
    a5 = (6 * h - 3 * v * T + 0.5 * a * T2) * inv_T5
    
    return a0, a1, a2, a3, a4, a5

def calc_quartic_coeffs_vectorized(xs, vxs, axs, vxe, axe, T):
    """四次多项式系数计算 (Vectorized)"""
    T = np.atleast_1d(T)
    T2, T3, T4 = T**2, T**3, T**4
    
    a0 = np.full_like(T, xs)
    a1 = np.full_like(T, vxs)
    a2 = np.full_like(T, axs / 2.0)
    
    b1 = vxe - a1 - 2*a2*T
    b2 = axe - 2*a2
    
    a3 = (b1 * 12 * T2 - b2 * 4 * T3) / (12 * T4 + 1e-6)
    a4 = (-b1 * 6 * T + b2 * 3 * T2) / (12 * T4 + 1e-6)
    
    return a0, a1, a2, a3, a4

class Trajectory:
    def __init__(self):
        # 基础 Frenet 状态
        self.t = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        
        # 笛卡尔状态 (MPC 需要)
        self.x = []
        self.y = []
        self.yaw = [] # psi
        self.k = []   # 曲率 kappa
        self.v = []   # 线性速度
        self.a = []   # 线性加速度
        
        self.cost = 0.0

class LatticePlanner:
    def __init__(self):
        self.cfg = LatticeConfig()
        self.optimizer = CfsOptimizer(self.cfg) if CfsOptimizer else None
        
        # [新增] 轨迹记忆模块
        # 用于存储上一帧的最终轨迹，实现“时空一致性”约束
        self.last_optimized_path = None
        self._debug_logs_left = 0

    def plan(self, grid_map, obs_dist, start_state, ref_path_points, resolution=0.5, prev_frenet_state=None):
        """
        核心规划接口 (集成 CFS 后端优化版)
        :param grid_map: 0/1 栅格地图
        :param obs_dist: 障碍物距离场 (EDT)
        :param start_state: 车辆状态
        :param ref_path_points: A* 全局路径点
        :param resolution: 地图分辨率
        """
        if len(ref_path_points) < 3:
            return None
        
        # ==========================================
        # 1. 参考线预处理 (新增优化：重采样与去噪)
        # ==========================================
        # 即使 A* 输出已经平滑，重采样能保证 Frenet 坐标系的数值稳定性
        # 这一步将解决“锯齿”问题，提供均匀分布的参考点
        rx, ry = self._preprocess_reference_line(ref_path_points)
        
        if not rx:
             print("[Lattice] Preprocessing failed to generate valid points.")
             return None

        # 2. 构建参考线 & 状态转换 (Cartesian -> Frenet)
        try:
            # 使用处理后的均匀点构建样条
            csp = CubicSpline2D(rx, ry)
            
            # 计算起始点的 Frenet 状态
            self._debug_logs_left = getattr(self.cfg, "debug_max_logs", 0)
            s0, d0, d_d0, d_dd0, s_d0, s_dd0 = self._cartesian_to_frenet(start_state, csp, prev_frenet_state)
            
        except Exception as e:
            # 捕获样条构建或投影计算中的数学错误 (如除零)
            print(f"[Lattice] Init failed (Spline/Transform): {e}")
            return None

        # 3. 采样空间生成
        d_samples = np.arange(-self.cfg.d_road_w, self.cfg.d_road_w + 1e-5, self.cfg.sample_d_step)
        t_samples = np.arange(self.cfg.time_min, self.cfg.time_max + 1e-5, self.cfg.time_step)
        v_samples = np.array([self.cfg.target_speed, self.cfg.target_speed * 0.6, self.cfg.target_speed * 0.3])
        
        T_grid, V_grid, D_grid = np.meshgrid(t_samples, v_samples, d_samples, indexing='ij')
        T_flat, V_flat, D_flat = T_grid.ravel(), V_grid.ravel(), D_grid.ravel()
        
        # 4. 批量生成多项式系数
        sa = calc_quartic_coeffs_vectorized(s0, s_d0, s_dd0, V_flat, 0.0, T_flat)
        da = calc_quintic_coeffs_vectorized(d0, d_d0, d_dd0, D_flat, 0.0, 0.0, T_flat)
        
        # 5. Cost 预筛选
        costs = self.cfg.w_lat_diff * D_flat**2 + \
                self.cfg.w_efficiency * (self.cfg.target_speed - V_flat)**2 + \
                self.cfg.w_time * (1.0 / (T_flat + 0.1))
        
        # 6. 轨迹生成与筛选
        valid_paths = []
        sorted_idx = np.argsort(costs)
        
        # 限制候选数量以保证实时性
        for idx in sorted_idx[:self.cfg.max_candidates]:
            c_sa = [arr[idx] for arr in sa]
            c_da = [arr[idx] for arr in da]
            T_end = T_flat[idx]
            
            t = np.arange(0.0, T_end, 0.1)
            if len(t) < 2: continue
            
            # 纵向加速度检查
            s_dd = 2*c_sa[2] + 6*c_sa[3]*t + 12*c_sa[4]*t**2
            if np.max(np.abs(s_dd)) > self.cfg.max_accel: continue
            
            p = Trajectory()
            p.t = t
            p.s = c_sa[0] + c_sa[1]*t + c_sa[2]*t**2 + c_sa[3]*t**3 + c_sa[4]*t**4
            p.s_d = c_sa[1] + 2*c_sa[2]*t + 3*c_sa[3]*t**2 + 4*c_sa[4]*t**3
            p.s_dd = s_dd
            self._log_sd_issue("traj_s_d", p.s_d)
            
            p.d = c_da[0] + c_da[1]*t + c_da[2]*t**2 + c_da[3]*t**3 + c_da[4]*t**4 + c_da[5]*t**5
            p.d_d = c_da[1] + 2*c_da[2]*t + 3*c_da[3]*t**2 + 4*c_da[4]*t**3 + 5*c_da[5]*t**4
            p.d_dd = 2*c_da[2] + 6*c_da[3]*t + 12*c_da[4]*t**2 + 20*c_da[5]*t**3
            p.d_ddd = 6*c_da[3] + 24*c_da[4]*t + 60*c_da[5]*t**2
            
            self._f2c(p, csp)
            
            if np.max(np.abs(p.k)) > self.cfg.max_curvature: continue
            
            is_coll, risk = self._check_collision(p, grid_map, obs_dist, resolution)
            if not is_coll:
                p.v = p.s_d
                p.a = p.s_dd
                
                j_smooth = np.sum(p.d_ddd**2) + np.sum(p.s_dd**2)
                j_eff = np.sum(np.abs(self.cfg.target_speed - p.s_d))
                j_lat = np.sum(p.d**2)
                
                p.cost = self.cfg.w_collision * risk + \
                         self.cfg.w_smoothness * j_smooth + \
                         self.cfg.w_efficiency * j_eff + \
                         self.cfg.w_lat_diff * j_lat
                
                valid_paths.append(p)
        
        # 7. 最佳路径选择与后端优化集成 (核心修改)
        if valid_paths:
            # Step 1: 选出 Lattice 原始最佳路径 (粗糙解)
            best_traj = min(valid_paths, key=lambda x: x.cost)
            raw_path = self._format_output(best_traj)
            
            # 默认返回原始路径 (保底)
            final_path = raw_path
            
            # Step 2: 调用 CFS 后端优化
            if self.optimizer is not None:
                try:
                    # 执行优化
                    # 注意：必须传入 obs_dist (距离场) 和 resolution
                    opt_res = self.optimizer.optimize(
                        initial_path=raw_path, 
                        dist_map=obs_dist, 
                        resolution=resolution, 
                        prev_path=self.last_optimized_path
                    )
                    
                    # 验证结果
                    if opt_res is not None and len(opt_res) > 0:
                        final_path = opt_res
                        
                        # [关键] 更新记忆 (Deep Copy 防止外部篡改)
                        self.last_optimized_path = copy.deepcopy(final_path)
                    else:
                        # 优化器返回空，重置记忆
                        self.last_optimized_path = None
                        
                except Exception as e:
                    # 容错处理：打印错误但不中断程序，自动降级为 Lattice 原始路径
                    print(f"[Lattice] Optimization Exception: {e}")
                    self.last_optimized_path = None
                    final_path = raw_path
            
            return final_path
            
        return None

    def _preprocess_reference_line(self, raw_points):

        if not raw_points:
            return [], []

        # 提取坐标
        x = np.array([p['x'] for p in raw_points])
        y = np.array([p['y'] for p in raw_points])

        # 1. 简单的距离过滤 (防止点重合导致样条计算除零错误)
        if len(x) > 1:
            diff = np.hypot(np.diff(x), np.diff(y))
            # 保留第一个点，以及距离前一个点大于 1mm 的点
            valid_idx = np.concatenate(([True], diff > 1e-3))
            x = x[valid_idx]
            y = y[valid_idx]

        if len(x) < 3:
            # 点太少无法构建三次样条，直接返回
            return x.tolist(), y.tolist()

        # 2. 构建临时样条进行重采样
        # 间距设为 0.5m 比较合适，既保留几何特征又平滑
        try:
            temp_csp = CubicSpline2D(x, y)
            total_length = temp_csp.s[-1]
            
            # 均匀采样
            ds = 0.5  # [Config] 重采样步长
            s_new = np.arange(0, total_length, ds)
            
            # 如果最后一段太短，忽略；如果恰好到终点，包含终点
            # 确保终点被包含
            if s_new[-1] < total_length:
                s_new = np.append(s_new, total_length)

            rx, ry = [], []
            for s in s_new:
                ix, iy = temp_csp.calc_position(s)
                rx.append(ix)
                ry.append(iy)
            
            return rx, ry
            
        except Exception as e:
            # 如果样条构建失败（极其罕见），降级返回原始点
            print(f"[Lattice] Preprocess warning: {e}")
            return x.tolist(), y.tolist()

    def _cartesian_to_frenet(self, state, csp, prev):
        """将车辆状态投影到 Frenet 坐标系"""
        # 1. 如果有上一帧的 Frenet 状态，优先使用 (闭环连续性)
        if prev is not None:
            # 简单的预测更新，防止重规划跳变
            # 这里简化处理，直接返回，实际项目中可能需要基于 dt 递推一步
            # 为了稳健，我们暂时每次都重算投影，但可以用 prev 里的 s 作为搜索初值
            pass

        # 2. 寻找匹配点 s
        s_guess = prev['s'] if (prev and 's' in prev) else 0.0
        # 如果是第一次，全范围搜索；否则在 s_guess 附近搜索
        if prev is None:
            s = csp.find_projection(state['x'], state['y'])
        else:
            # 局部搜索，提高效率
            # 注意: CubicSpline2D.find_projection 内部目前实现了全搜索
            # 如果性能有瓶颈，可以修改 curves.py 增加 start_s 参数
            s = csp.find_projection(state['x'], state['y'])

        # 3. 计算参考点状态
        rx, ry = csp.calc_position(s)
        ryaw = csp.calc_yaw(s)
        rk = csp.calc_curvature(s)
        
        # 4. 计算 Frenet 状态
        dx = state['x'] - rx
        dy = state['y'] - ry
        
        # d (横向偏差) = 向量(dx, dy) 叉乘 参考方向向量
        # 叉乘: dx*sin(-ryaw) + dy*cos(-ryaw) ... 简化公式如下:
        d = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
        
        # 速度分解
        v = state['v']
        yaw_diff = state['yaw'] - ryaw
        # 归一化角度
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
        
        s_d = v * math.cos(yaw_diff) / (1 - rk * d)
        d_d = v * math.sin(yaw_diff)
        self._log_sd_issue("init_s_d", np.array([s_d]), yaw_diff=yaw_diff)
        
        # 加速度分解 (简化: 假设主要加速度在切向)
        # 实际应包含向心加速度项，这里简化处理 s_dd ~= a, d_dd ~= 0
        s_dd = state.get('a', 0.0)
        d_dd = 0.0 
        
        return s, d, d_d, d_dd, s_d, s_dd

    def _log_sd_issue(self, tag, s_d_values, yaw_diff=None):
        if not getattr(self.cfg, "debug", False):
            return
        if self._debug_logs_left <= 0:
            return
        eps = getattr(self.cfg, "debug_speed_epsilon", 0.05)
        s_d_values = np.asarray(s_d_values)
        neg_idx = np.where(s_d_values < -eps)[0]
        near_idx = np.where(np.abs(s_d_values) <= eps)[0]
        if len(neg_idx) == 0 and len(near_idx) == 0:
            return
        msg = f"[Lattice][debug] {tag}: s_d min={s_d_values.min():.3f}, max={s_d_values.max():.3f}"
        if len(neg_idx) > 0:
            msg += f", first_neg_idx={int(neg_idx[0])}"
        if len(near_idx) > 0:
            msg += f", first_near0_idx={int(near_idx[0])}"
        if yaw_diff is not None:
            msg += f", yaw_diff={yaw_diff:.3f}"
        print(msg)
        self._debug_logs_left -= 1

    def _f2c(self, p, csp):
        """Frenet 转 Cartesian (修复：增加终点线性外推)"""
        # 获取参考线的最大 s 值
        max_s = csp.s[-1]
        
        # 批量计算参考线状态
        for i in range(len(p.t)):
            s_val = p.s[i]
            d_val = p.d[i]
            
            if s_val <= max_s:
                # 正常情况：在参考线范围内
                rx, ry = csp.calc_position(s_val)
                ryaw = csp.calc_yaw(s_val)
                # rk = csp.calc_curvature(s_val) # 暂时不用，下面用数值微分算
            else:
                # [关键修复] 溢出情况：线性外推 (Linear Extrapolation)
                # 避免 curves.py 内部 clamp 导致坐标重叠、梯度为0、曲率爆炸
                ds = s_val - max_s
                rx_end, ry_end = csp.calc_position(max_s)
                ryaw_end = csp.calc_yaw(max_s)
                
                # 沿切线方向延伸
                rx = rx_end + ds * math.cos(ryaw_end)
                ry = ry_end + ds * math.sin(ryaw_end)
                ryaw = ryaw_end # 航向角保持不变
            
            # 坐标变换 (Frenet -> Global)
            x = rx - d_val * math.sin(ryaw)
            y = ry + d_val * math.cos(ryaw)
            
            p.x.append(x)
            p.y.append(y)
            
            # 航向角变换 (近似: yaw = ref_yaw + atan(d_d / s_d))
            # 注意：如果 s_d 为 0 (虽然极少见)，math.atan2 会处理
            yaw = ryaw + math.atan2(p.d_d[i], p.s_d[i])
            p.yaw.append(yaw)
            
        # 使用数值微分补全 k (比复杂解析公式更稳健)
        # 现在有了线性外推，x 和 y 不会重叠，gradient 计算将恢复正常
        dx = np.gradient(p.x)
        dy = np.gradient(p.y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # 增加极小值防止除零
        den = np.hypot(dx, dy)**3 + 1e-6
        k = (dx * ddy - dy * ddx) / den
        p.k = k.tolist()

    def _get_vehicle_circles(self):
        """
        [新增] 根据车辆尺寸自动生成覆盖圆参数
        返回: list of {'offset': float, 'radius': float}
        """
        # 1. 基础参数
        length = self.cfg.length
        width = self.cfg.width
        
        # 2. 策略：沿着长轴，每隔一定距离放置一个圆
        # 间距设为车宽的一半，可以保证较好的覆盖且圆数量适中
        spacing = width / 2.0  
        
        # 计算需要多少个圆
        num_circles = math.ceil(length / spacing)
        
        # 重新调整间距，使其均匀分布
        if num_circles < 1: num_circles = 1
        actual_spacing = length / num_circles
        
        # 3. 计算覆盖半径 (勾股定理：覆盖矩形对角线)
        # 覆盖矩形块大小：(width) x (actual_spacing)
        radius = math.hypot(width / 2.0, actual_spacing / 2.0) + 0.1 # +0.1为额外余量
        
        # 4. 生成偏移量 (假设车辆原点在几何中心)
        # 如果车辆原点是后轴，需要根据后悬长度调整 range 的起始点
        # 这里默认 path.x, path.y 是车辆几何中心
        circles = []
        start_offset = -length / 2.0 + actual_spacing / 2.0
        
        for i in range(num_circles):
            offset = start_offset + i * actual_spacing
            circles.append({'offset': offset, 'radius': radius})
            
        return circles

    def _check_collision(self, p, grid, dist_m, res):
        """
        基于距离场的多圆覆盖碰撞检测 (修复版：自动适应大车尺寸)
        """
        rows, cols = grid.shape
        origin_x, origin_y = 0.0, -10.0
        
        # [修改] 动态生成覆盖圆，不再硬编码
        # 为了性能，这个 collision_circles 可以在 __init__ 里只算一次
        # 但为了演示清晰，放在这里
        collision_circles = self._get_vehicle_circles()
        
        # 遍历轨迹上的每一个点 (步长可调整)
        step = 2
        for i in range(0, len(p.x), step): 
            cx, cy, cyaw = p.x[i], p.y[i], p.yaw[i]
            
            # 预计算三角函数
            cos_yaw = math.cos(cyaw)
            sin_yaw = math.sin(cyaw)
            
            # 检查车身上的每一个覆盖圆
            for circle in collision_circles:
                # 坐标变换：从车身坐标系 -> 世界坐标系
                # x' = center_x + offset * cos(yaw)
                # y' = center_y + offset * sin(yaw)
                circle_x = cx + circle['offset'] * cos_yaw
                circle_y = cy + circle['offset'] * sin_yaw
                
                # 转换到栅格索引
                c = int((circle_x - origin_x) / res)
                r = int((circle_y - origin_y) / res)
                
                # 1. 越界检查 (出地图视为碰撞)
                if not (0 <= r < rows and 0 <= c < cols):
                    return True, 100.0
                
                # 2. 查表获取障碍物距离
                obs_dist = dist_m[r, c]
                
                # 3. 碰撞判断
                if obs_dist < circle['radius']:
                    return True, 100.0
                
        return False, 0.0

    def _format_output(self, p):
        """转换为字典列表输出"""
        traj = []
        for i in range(len(p.t)):
            traj.append({
                't': p.t[i],
                'x': p.x[i],
                'y': p.y[i],
                'psi': p.yaw[i],
                'v': p.v[i],
                'a': p.a[i],
                'k': p.k[i]
            })

        return traj
