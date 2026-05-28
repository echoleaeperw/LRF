import torch
import numpy as np
import json
import os
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# NuScenes 状态向量格式说明:
# past_gt:    [NA, PT, 6]  →  [x, y, cos_h, sin_h, speed, hdot]
# future_pred: [NA, FT, 4]  →  [x, y, cos_h, sin_h]  (无速度，需差分估计)
_STATE_IDX_X = 0
_STATE_IDX_Y = 1
_STATE_IDX_COS_H = 2
_STATE_IDX_SIN_H = 3
_STATE_IDX_SPEED = 4     # 标量速度 (m/s)，不是 vx！
_STATE_IDX_HDOT = 5      # 航向变化率 (rad/s)，不是 vy！


class ScenarioExtractor:
    """
    从场景数据中提取结构化文本描述，包括：
    - 车辆位置、速度、方向等信息
    - 场景道路布局和环境
    - 车辆间的相对关系
    - 识别的潜在风险点
    """
    
    def __init__(self, model=None, normalizer=None, att_normalizer=None, llm_provider: str = "deepseek"):
        """
        初始化场景提取器
        
        参数:
            model: 交通模型，用于获取归一化器等
            normalizer: 状态归一化器，用于还原实际物理量
            att_normalizer: 属性归一化器，用于还原实际物理量
            llm_provider: LLM提供商，用于场景描述生成
        """
        self.model = model
        self.normalizer = normalizer
        self.att_normalizer = att_normalizer
        self.llm_provider = llm_provider
        
        self.vehicle_types = {
            (1, 0, 0, 0, 0): "car",
            (0, 1, 0, 0, 0): "truck",
            (0, 0, 1, 0, 0): "bus",
            (0, 0, 0, 1, 0): "motorcycle", 
            (0, 0, 0, 0, 1): "bicycle",

        }
        
    @staticmethod
    def _distance_point_to_path(point: np.ndarray, path: np.ndarray) -> float:
        """Calculate the minimum distance from a point to a piecewise path defined by a sequence of points."""
        # 确保路径至少有两个点
        if len(path) < 2:
            return np.linalg.norm(point - path[0]) if len(path) == 1 else np.inf

        min_dist = np.inf
        p = np.array(point)
        for i in range(len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i+1])
            
            # 计算线段方向向量
            line_vec = p2 - p1
            # 计算点到线段起点的向量
            point_vec = p - p1
            
            # 计算点在线段上的投影比例
            # np.dot(line_vec, line_vec) 等于 line_len_sq
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0: # 避免除以零
                t = 0
            else:
                t = np.dot(point_vec, line_vec) / line_len_sq
            
            # 将投影比例限制在线段范围内 [0, 1]
            t = np.clip(t, 0, 1)
            
            # 计算线段上离点最近的点
            closest_point_on_segment = p1 + t * line_vec
            
            # 更新最小距离
            dist = np.linalg.norm(p - closest_point_on_segment)
            if dist < min_dist:
                min_dist = dist
        
        return min_dist

    def extract_structured_scenario(self, 
                                    scene_graph, 
                                    map_env=None, 
                                    map_idx=None, 
                                    past_traj=None,
                                    future_pred=None,
                                    latent_z=None,
                                    output_path=None) -> Dict:
        """
        核心方法：从 scene_graph 提取结构化场景数据字典。
        
        输出 JSON 包含完整的车辆轨迹、语义信息、物理属性、运动趋势分析等，
        供 LLM 分析和权重生成使用。
        
        参数:
            scene_graph: 场景图对象，包含车辆信息
            map_env: 地图环境对象
            map_idx: 地图索引
            past_traj: 过去轨迹 [NA, PT, 6]
            future_pred: 预测的未来轨迹 [NA, FT, 4]
            latent_z: 潜变量（可选，仅导出用）
            output_path: 输出 JSON 文件路径（可选）
            
        返回:
            结构化场景数据字典
        """



        # 确保有正确的归一化器
        if self.normalizer is None and self.model is not None:
            self.normalizer = self.model.get_normalizer()
        if self.att_normalizer is None and self.model is not None:
            self.att_normalizer = self.model.get_att_normalizer()
        
        # 创建场景数据字典
        scenario_data = {
            "format_version": "1.0",
            "description": "Generated scenario from STRIVE model",
            "map": self._extract_map_name(map_env, map_idx) if map_env is not None else "unknown",
            "dt": self.model.dt if self.model else 0.1,
            "vehicles": [],
            "latent_z": self._encode_latent_z(latent_z) if latent_z is not None else None
        }
        
        # 提取完整轨迹
        trajectories = self._extract_complete_trajectories(scene_graph, past_traj, future_pred)
        
        # 提取车辆语义信息
        semantic_info = self._extract_semantic_info(scene_graph)
        
        # 提取车辆物理属性
        physical_props = self._extract_physical_properties(scene_graph)
        
        # 提取每辆车的地图上下文（车道、交叉口等）
        map_context = self._extract_map_context(scene_graph, map_env, map_idx)
        
        # 检查并创建batch信息如果不存在
        if not hasattr(scene_graph, 'batch') or scene_graph.batch is None:
            # 为单场景图创建默认的batch索引
            num_vehicles = scene_graph.past_gt.size(0)
            scene_graph.batch = torch.zeros(num_vehicles, dtype=torch.long, device=scene_graph.past_gt.device)
        
        # 检查并创建ptr信息如果不存在
        if not hasattr(scene_graph, 'ptr') or scene_graph.ptr is None:
            # 为单场景图创建默认的ptr索引（第一个车辆为自车）
            num_vehicles = scene_graph.past_gt.size(0)
            scene_graph.ptr = torch.tensor([0, num_vehicles], dtype=torch.long, device=scene_graph.past_gt.device)
        
        # 整合车辆信息并添加时间序列分析
        batch_indices = scene_graph.batch
        unique_batches = torch.unique(batch_indices)
        dt = self.model.dt if self.model else 0.5
        
        for b in unique_batches:
            batch_mask = batch_indices == b
            batch_vehicles = torch.where(batch_mask)[0]
            
            # 收集当前批次的轨迹信息用于相对运动分析
            ego_trajectory = None
            other_trajectories = []
            ego_idx = None
            
            for i, v_idx in enumerate(batch_vehicles):
                # 判断是否为自车
                is_ego = v_idx == scene_graph.ptr[b].item()
                
                # 提取时间序列轨迹信息进行分析
                vehicle_trajectory = trajectories[v_idx.item()]
                motion_analysis = self._analyze_motion_trends(vehicle_trajectory)
                
                if is_ego:
                    ego_trajectory = vehicle_trajectory
                    ego_idx = v_idx
                else:
                    other_trajectories.append(vehicle_trajectory)
                
                vid = v_idx.item()
                vehicle_data = {
                    "id": vid,
                    "is_ego": bool(is_ego.item()) if isinstance(is_ego, torch.Tensor) else bool(is_ego),
                    "type": semantic_info[vid],
                    "length": physical_props[vid]["length"],
                    "width": physical_props[vid]["width"],
                    "trajectory": vehicle_trajectory,
                    "motion_analysis": motion_analysis,
                    "map_context": map_context.get(vid),
                }
                
                scenario_data["vehicles"].append(vehicle_data)
            
            # 添加相对运动分析和场景动态分析
            if ego_trajectory and other_trajectories:
                relative_motion = self._compute_relative_motion(ego_trajectory, other_trajectories, dt)
                scenario_data["relative_motion_analysis"] = relative_motion
            
            # 添加整体场景动态分析
            if past_traj is not None:
                past_states = past_traj
                if self.normalizer is not None:
                    past_states = self.normalizer.unnormalize(past_states)
                
                batch_vehicles_states = past_states[batch_vehicles]
                traffic_flow = self._analyze_traffic_flow(batch_vehicles_states, dt)
                complexity = self._analyze_scenario_complexity(batch_vehicles_states, dt)
                
                scenario_data["dynamic_analysis"] = {
                    "traffic_flow": traffic_flow,
                    "complexity": complexity,
                }
        
        # 如果提供了输出路径，保存为文件
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(scenario_data, f, indent=2)
        
        return scenario_data
    
    def _analyze_traffic_flow(self, vehicles_states, dt):
        """分析交通流特征。vehicles_states: [N, T, 6] 已反归一化的 past_gt。"""
        if vehicles_states.size(0) < 2:
            return "车辆数量不足，无法分析交通流"
        
        final_states = vehicles_states[:, -1, :]  # 最后一帧
        velocities = []
        for i in range(vehicles_states.size(0)):
            if final_states.size(1) > 5:
                speed = float(final_states[i, _STATE_IDX_SPEED].cpu().item())
                velocities.append(speed)
        
        if not velocities:
            return "无速度数据"
        
        avg_velocity = np.mean(velocities)
        velocity_std = np.std(velocities)
        
        if velocity_std < 2.0:
            flow_type = "均匀流动"
        elif velocity_std < 5.0:
            flow_type = "轻微拥堵"
        else:
            flow_type = "严重拥堵或混乱"
        
        return f"{flow_type}, 平均速度{avg_velocity:.1f}m/s, 速度标准差{velocity_std:.1f}m/s"
    
    def _analyze_scenario_complexity(self, vehicles_states, dt):
        """分析场景复杂度"""
        num_vehicles = vehicles_states.size(0)
        
        # 基于车辆数量的复杂度
        if num_vehicles <= 2:
            base_complexity = "简单"
        elif num_vehicles <= 5:
            base_complexity = "中等"
        else:
            base_complexity = "复杂"
        
        # 基于运动模式的复杂度
        motion_complexity = 0
        for i in range(num_vehicles):
            trajectory = self._extract_temporal_trajectories(vehicles_states[i], dt)
            if len(trajectory) > 1:
                # 计算轨迹曲率
                positions = [(p["x"], p["y"]) for p in trajectory]
                if len(positions) > 2:
                    curvature = self._calculate_trajectory_curvature(positions)
                    if curvature > 0.1:
                        motion_complexity += 1
        
        if motion_complexity > num_vehicles * 0.5:
            motion_desc = "高运动复杂度"
        elif motion_complexity > 0:
            motion_desc = "中运动复杂度"
        else:
            motion_desc = "低运动复杂度"
        
        return f"{base_complexity}场景, {motion_desc}"
    
    def _calculate_trajectory_curvature(self, positions):
        """计算轨迹曲率"""
        if len(positions) < 3:
            return 0.0
        
        curvatures = []
        for i in range(1, len(positions) - 1):
            p1, p2, p3 = positions[i-1], positions[i], positions[i+1]
            
            # 计算三点形成的角度变化
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # 计算角度
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                curvatures.append(angle)
        
        return np.mean(curvatures) if curvatures else 0.0
    
    
    def _extract_temporal_trajectories(self, vehicle_trajectory, dt):
        """
        提取单个车辆的时间序列轨迹信息（用于 past_gt 数据，6维状态）
        
        参数:
            vehicle_trajectory: 车辆轨迹 (PT, 6) - [x, y, cos_h, sin_h, speed, hdot]
            dt: 时间步长
            
        返回:
            轨迹点列表，每个点为 {"t", "x", "y", "heading", "velocity"}
        """
        trajectory_data = []
        PT = vehicle_trajectory.size(0)
        
        for t in range(PT):
            state = vehicle_trajectory[t].cpu().numpy()
            time_val = -dt * (PT - t - 1)
            
            x, y = state[_STATE_IDX_X], state[_STATE_IDX_Y]
            cos_h, sin_h = state[_STATE_IDX_COS_H], state[_STATE_IDX_SIN_H]
            heading = np.arctan2(sin_h, cos_h) * 180 / np.pi
            
            # state[4] 是标量速度 speed，直接使用
            velocity = float(state[_STATE_IDX_SPEED]) if state.shape[0] > 5 else 0.0
            
            trajectory_data.append({
                "t": float(time_val),
                "x": float(x),
                "y": float(y),
                "heading": float(heading),
                "velocity": float(velocity)
            })
        
        return trajectory_data
    
    def _analyze_motion_trends(self, trajectory_data):
        """
        分析车辆运动趋势
        
        参数:
            trajectory_data: 轨迹点列表，每个点包含 {"t", "x", "y", "heading", "velocity"}
            
        返回:
            运动趋势描述字符串
        """
        if len(trajectory_data) < 2:
            return "数据不足"
        
        velocities = [point["velocity"] for point in trajectory_data]
        initial_velocity = velocities[0]
        final_velocity = velocities[-1]
        velocity_change = final_velocity - initial_velocity
        
        # 计算朝向变化
        headings = [point["heading"] for point in trajectory_data]
        heading_change = abs(headings[-1] - headings[0])
        if heading_change > 180:
            heading_change = 360 - heading_change
        
        trend_parts = []
        
        if abs(velocity_change) > 1.0:
            if velocity_change > 0:
                trend_parts.append(f"加速({velocity_change:.1f}m/s)")
            else:
                trend_parts.append(f"减速({abs(velocity_change):.1f}m/s)")
        else:
            trend_parts.append("匀速")
        
        if heading_change > 10:
            trend_parts.append(f"转向({heading_change:.1f}度)")
        else:
            trend_parts.append("直行")
        
        if final_velocity < 0.5:
            trend_parts.append("近乎静止")
        elif final_velocity > 15:
            trend_parts.append("高速行驶")
        
        return ", ".join(trend_parts)
    
    def _compute_relative_motion(self, ego_trajectory, other_trajectories, dt):
        """
        计算车辆间的相对运动关系
        
        参数:
            ego_trajectory: 自车轨迹数据
            other_trajectories: 其他车辆轨迹数据列表
            dt: 时间步长
            
        返回:
            相对运动分析字符串
        """
        if not ego_trajectory or not other_trajectories:
            return "无相对运动数据"
        
        relative_analysis = []
        
        for i, other_traj in enumerate(other_trajectories):
            if not other_traj:
                continue
                
            # 计算最终时刻的相对位置
            ego_final = ego_trajectory[-1]
            other_final = other_traj[-1]
            
            dx = other_final["x"] - ego_final["x"]
            dy = other_final["y"] - ego_final["y"]
            distance = np.sqrt(dx*dx + dy*dy)
            
            # 计算相对速度
            ego_vel = ego_final["velocity"]
            other_vel = other_final["velocity"]
            relative_speed = other_vel - ego_vel
            
            # 计算相对位置方向
            angle = np.arctan2(dy, dx) * 180 / np.pi
            if angle < 0:
                angle += 360
            
            # 确定相对位置描述
            if angle < 45 or angle >= 315:
                position_desc = "前方"
            elif 45 <= angle < 135:
                position_desc = "左侧"
            elif 135 <= angle < 225:
                position_desc = "后方"
            else:
                position_desc = "右侧"
            
            # 分析运动趋势
            if len(ego_trajectory) > 1 and len(other_traj) > 1:
                # 计算距离变化趋势
                initial_dx = other_traj[0]["x"] - ego_trajectory[0]["x"]
                initial_dy = other_traj[0]["y"] - ego_trajectory[0]["y"]
                initial_distance = np.sqrt(initial_dx*initial_dx + initial_dy*initial_dy)
                
                distance_change = distance - initial_distance
                
                if distance_change < -2.0:
                    trend_desc = "快速接近"
                elif distance_change < -0.5:
                    trend_desc = "缓慢接近"
                elif distance_change > 2.0:
                    trend_desc = "快速远离"
                elif distance_change > 0.5:
                    trend_desc = "缓慢远离"
                else:
                    trend_desc = "保持距离"
            else:
                trend_desc = "趋势不明"
            
            relative_analysis.append(
                f"车辆{i+1}: {position_desc}{distance:.1f}m, {trend_desc}, 相对速度{relative_speed:.1f}m/s"
            )
        
        return "; ".join(relative_analysis)
    
    
    
    def _unflip_coords(self, x: float, y: float, map_name: str, map_env) -> Tuple[float, float]:
        """
        将 STRIVE 内部的翻转坐标还原为 nuScenes 地图 API 的原始坐标。

        新加坡地图在数据预处理时做了 y = mheight - y 翻转，
        查询地图 API 前需要反转回来。
        """
        from datasets.map_env import NUSC_MAP_SIZES
        is_singapore = map_name.startswith("singapore")
        if is_singapore and getattr(map_env, "flip_singapore", True):
            mheight = NUSC_MAP_SIZES[map_name][0]
            y = mheight - y
        return x, y

    def _get_single_vehicle_map_context(self, x: float, y: float, nusc_map) -> Dict:
        """
        查询 nuScenes 地图 API，返回单个车辆所在位置的结构化地图上下文。

        参数:
            x, y: 已还原为 nuScenes 原始坐标系的世界坐标。
            nusc_map: NuScenesMap 实例。

        返回 JSON-serializable 字典。
        """
        from nuscenes.map_expansion.arcline_path_utils import discretize_lane

        context: Dict[str, Any] = {
            "on_road": False,
            "lane": None,
            "in_intersection": False,
        }

        try:
            closest_lane_token = nusc_map.get_closest_lane(x, y, radius=5.0)
            if not closest_lane_token:
                return context

            context["on_road"] = True

            arcline_path = nusc_map.get_arcline_path(closest_lane_token)
            lane_pts = np.array(discretize_lane(arcline_path, resolution_meters=0.5))[:, :2]
            dist_to_cl = float(self._distance_point_to_path(np.array([x, y]), lane_pts))

            in_intersection = False
            try:
                layers_at_pt = nusc_map.layers_on_point(x, y)
                if layers_at_pt.get("lane_connector"):
                    in_intersection = True
            except Exception:
                pass
            context["in_intersection"] = in_intersection

            direction = "straight"
            try:
                rec = nusc_map.get("lane", closest_lane_token)
                td = rec.get("turn_direction", "NONE")
                if td and td != "NONE":
                    direction = td.lower()
            except Exception:
                pass
            if in_intersection:
                direction = "turning"

            context["lane"] = {
                "token": closest_lane_token,
                "type": "lane_connector" if in_intersection else "lane",
                "direction": direction,
                "distance_to_centerline_m": round(dist_to_cl, 3),
            }

        except Exception as e:
            logger.warning(f"地图上下文查询失败 ({x:.1f}, {y:.1f}): {e}")

        return context

    def _extract_map_context(self,
                             scene_graph,
                             map_env,
                             map_idx) -> Dict[int, Dict]:
        """
        批量提取所有车辆的地图上下文。

        使用每辆车在 past_gt 最后一帧（即 t=0）的反归一化世界坐标，
        先还原 Singapore 翻转，再查询 nuScenes 地图 API。

        返回:
            {vehicle_idx: map_context_dict, ...}
        """
        result: Dict[int, Dict] = {}

        if map_env is None or map_idx is None:
            return result

        if self.normalizer is None:
            logger.warning("缺少 normalizer，无法查询地图上下文。")
            return result

        idx_val = map_idx.item() if isinstance(map_idx, torch.Tensor) else int(map_idx)
        map_name = map_env.map_list[idx_val]
        nusc_map = map_env.nusc_maps[map_name]

        last_states = self.normalizer.unnormalize(scene_graph.past_gt[:, -1, :])

        for v_idx in range(last_states.size(0)):
            state = last_states[v_idx].cpu().numpy()
            x_raw, y_raw = float(state[0]), float(state[1])
            x_map, y_map = self._unflip_coords(x_raw, y_raw, map_name, map_env)
            result[v_idx] = self._get_single_vehicle_map_context(x_map, y_map, nusc_map)

        return result

    def _extract_complete_trajectories(self, scene_graph, past_traj=None, future_pred=None) -> Dict[int, List]:
        """
        提取完整的车辆轨迹（过去 + 未来），统一为:
          {"t": 秒, "x": m, "y": m, "heading": 度, "velocity": m/s}
        
        past_gt: [NA, PT, 6]  →  state[4] = speed (标量)
        future_pred: [NA, FT, 4]  →  无速度字段，需差分估计
        """
        if past_traj is None:
            past_traj = scene_graph.past_gt
        
        if self.normalizer is not None:
            if past_traj is not None:
                past_traj = self.normalizer.unnormalize(past_traj)
            if future_pred is not None:
                future_pred = self.normalizer.unnormalize(future_pred)
        
        trajectories = {}
        dt = self.model.dt if self.model else 0.5
        
        for v_idx in range(scene_graph.past_gt.size(0)):
            traj = []
            
            # ====== 过去轨迹 (6维: x, y, cos_h, sin_h, speed, hdot) ======
            if past_traj is not None:
                for t in range(past_traj.size(1)):
                    state = past_traj[v_idx, t].cpu().numpy()
                    time_val = -dt * (past_traj.size(1) - t)
                    
                    x, y = state[_STATE_IDX_X], state[_STATE_IDX_Y]
                    cos_h, sin_h = state[_STATE_IDX_COS_H], state[_STATE_IDX_SIN_H]
                    heading = np.arctan2(sin_h, cos_h) * 180 / np.pi
                    
                    velocity = float(state[_STATE_IDX_SPEED]) if state.shape[0] > 5 else 0.0
                    
                    traj.append({
                        "t": float(time_val),
                        "x": float(x),
                        "y": float(y),
                        "heading": float(heading),
                        "velocity": float(velocity)
                    })
            
            # ====== 未来轨迹 (4维: x, y, cos_h, sin_h → 用差分估计速度) ======
            if future_pred is not None:
                if len(future_pred.shape) == 3:
                    future = future_pred[v_idx]
                elif len(future_pred.shape) == 4:
                    future = future_pred[v_idx, 0]
                else:
                    logger.warning(f"未知的future_pred形状: {future_pred.shape}")
                    trajectories[v_idx] = traj
                    continue
                
                for t in range(future.size(0)):
                    state = future[t].cpu().numpy()
                    time_val = dt * (t + 1)
                    
                    x, y = state[_STATE_IDX_X], state[_STATE_IDX_Y]
                    cos_h, sin_h = state[_STATE_IDX_COS_H], state[_STATE_IDX_SIN_H]
                    heading = np.arctan2(sin_h, cos_h) * 180 / np.pi
                    
                    # future_pred 通常只有4维，需从位置差分估计速度
                    velocity = 0.0
                    if state.shape[0] > 5:
                        velocity = float(state[_STATE_IDX_SPEED])
                    elif t == 0 and traj:
                        prev = traj[-1]
                        dx = x - prev["x"]
                        dy = y - prev["y"]
                        velocity = np.sqrt(dx * dx + dy * dy) / dt
                    elif t > 0:
                        prev_state = future[t - 1].cpu().numpy()
                        dx = x - prev_state[_STATE_IDX_X]
                        dy = y - prev_state[_STATE_IDX_Y]
                        velocity = np.sqrt(dx * dx + dy * dy) / dt
                    
                    traj.append({
                        "t": float(time_val),
                        "x": float(x),
                        "y": float(y),
                        "heading": float(heading),
                        "velocity": float(velocity)
                    })
            
            trajectories[v_idx] = traj
        
        return trajectories
    
    def _extract_semantic_info(self, scene_graph) -> Dict[int, str]:
        """
        提取车辆语义信息
        
        参数:
            scene_graph: 场景图对象
            
        返回:
            字典，键为车辆ID，值为车辆类型（如"car", "truck"等）
        """
        semantic_info = {}
        
        # 检查并创建batch信息如果不存在
        if not hasattr(scene_graph, 'batch') or scene_graph.batch is None:
            # 为单场景图创建默认的batch索引
            num_vehicles = scene_graph.sem.size(0) if hasattr(scene_graph, 'sem') and scene_graph.sem is not None else scene_graph.past_gt.size(0)
            scene_graph.batch = torch.zeros(num_vehicles, dtype=torch.long, device=scene_graph.past_gt.device)
        
        # 检查并创建ptr信息如果不存在
        if not hasattr(scene_graph, 'ptr') or scene_graph.ptr is None:
            # 为单场景图创建默认的ptr索引（第一个车辆为自车）
            num_vehicles = scene_graph.sem.size(0) if hasattr(scene_graph, 'sem') and scene_graph.sem is not None else scene_graph.past_gt.size(0)
            scene_graph.ptr = torch.tensor([0, num_vehicles], dtype=torch.long, device=scene_graph.past_gt.device)
        
        # 获取语义信息
        if hasattr(scene_graph, 'sem') and scene_graph.sem is not None:
            for v_idx in range(scene_graph.sem.size(0)):
                sem_vec = tuple(scene_graph.sem[v_idx].cpu().numpy().astype(int).tolist())
                
                # 判断是否为自车
                batch_idx = scene_graph.batch[v_idx]

                is_ego = v_idx == scene_graph.ptr[batch_idx].item()
                
                if is_ego:
                    semantic_info[v_idx] = "ego_vehicle"
                elif sem_vec in self.vehicle_types:
                    semantic_info[v_idx] = self.vehicle_types[sem_vec]
                else:
                    # 默认为小汽车
                    semantic_info[v_idx] = "unknown_vehicle"
        else:
            # 如果没有语义信息，根据是否为自车分配类型
            for v_idx in range(scene_graph.past_gt.size(0)):
                batch_idx = scene_graph.batch[v_idx]
                is_ego = v_idx == scene_graph.ptr[batch_idx].item()
                semantic_info[v_idx] = "ego_vehicle" if is_ego else "vehicle"
        
        return semantic_info
    
    def _extract_physical_properties(self, scene_graph) -> Dict[int, Dict]:
        """
        提取车辆物理属性
        
        参数:
            scene_graph: 场景图对象
            
        返回:
            字典，键为车辆ID，值为包含物理属性的字典
        """
        physical_props = {}
        
        # 获取车辆属性（归一化的）
        vehicle_attr = scene_graph.lw  # 长宽信息
        
        # 如果有归一化器，还原实际物理量
        if self.att_normalizer is not None:
            vehicle_attr = self.att_normalizer.unnormalize(vehicle_attr)
        
        # 处理每个车辆
        for v_idx in range(vehicle_attr.size(0)):
            length, width = vehicle_attr[v_idx].cpu().numpy()
            
            physical_props[v_idx] = {
                "length": float(length),
                "width": float(width),
                # 可以添加更多属性，如质量、最大加速度等
                "mass": 1500.0,  # 默认质量（kg）
                "max_acceleration": 3.0,  # 默认最大加速度（m/s^2）
                "max_deceleration": 8.0,  # 默认最大减速度（m/s^2）
                "max_steering_angle": 70.0  # 默认最大转向角（度）
            }
        
        return physical_props
    
    def _extract_map_name(self, map_env, map_idx) -> str:
        """
        提取地图名称
        
        参数:
            map_env: 地图环境对象
            map_idx: 地图索引
            
        返回:
            地图名称
        """
        if hasattr(map_env, 'map_list') and map_idx is not None:
            idx = map_idx.item() if isinstance(map_idx, torch.Tensor) else map_idx
            try:
                return map_env.map_list[idx]
            except:
                pass
        
        return "unknown_map"
    
    def _encode_latent_z(self, latent_z):
        """
        编码潜变量为可序列化格式
        
        参数:
            latent_z: 潜变量张量
            
        返回:
            可序列化的潜变量列表
        """
        if latent_z is None:
            return None
        
        if isinstance(latent_z, torch.Tensor):
            return latent_z.detach().cpu().numpy().tolist()
        
        return latent_z
    