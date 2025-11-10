import torch
import numpy as np
import json
import os
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from longterm.core.description_generator import LLMGenerateDescription


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

    def extract_scenario_description(self, 
                                    scene_graph, 
                                   map_env=None, 
                                   map_idx=None, 
                                   future_pred=None,
                                   identify_risks=True,
                                   auto_save=True,
                                   save_dir="scenario_descriptions") -> str:
        """
        从场景图中提取场景描述
        
        参数:
            scene_graph: 场景图对象，包含车辆信息
            map_env: 地图环境对象
            map_idx: 地图索引
            future_pred: 预测的未来轨迹
            identify_risks: 是否识别潜在风险点
            auto_save: 是否自动保存到文件
            save_dir: 保存目录
            
        返回:
            场景的文本描述
        """
        # 确保有正确的归一化器
        if self.normalizer is None and self.model is not None:
            self.normalizer = self.model.get_normalizer()
        if self.att_normalizer is None and self.model is not None:
            self.att_normalizer = self.model.get_att_normalizer()
            
        # 提取基本场景信息
        description = self._extract_basic_info(scene_graph)
        
        # 提取车辆信息
        vehicles_desc = self._extract_vehicles_info(scene_graph)
        description += "\n\n" + vehicles_desc
        
        # 提取地图和环境信息
        if map_env is not None and map_idx is not None:
            map_desc = self._extract_map_info(scene_graph, map_env, map_idx)
            description += "\n\n" + map_desc
            
        # 提取车辆间关系
        # relations_desc = self._extract_vehicle_relations(scene_graph)
        # description += "\n\n" + relations_desc
        
        # 识别潜在风险点
        if identify_risks and future_pred is not None:
            risks_desc = self._identify_risks(scene_graph, future_pred)
            description += "\n\n" + risks_desc
        
        # 自动保存到文件
        if auto_save:
            logger.info(f"开始自动保存场景描述，保存目录: {save_dir}")
            logger.info(f"场景描述长度: {len(description)} 字符")
            try:
                saved_path = self.save_scenario_description(description, save_dir)
                if saved_path:
                    logger.info(f"场景描述自动保存成功: {saved_path}")
                else:
                    logger.error("场景描述自动保存失败")
            except Exception as e:
                logger.error(f"自动保存过程中发生异常: {e}")
                import traceback
                logger.error(f"异常详情: {traceback.format_exc()}")
        else:
            logger.info("auto_save=False，跳过自动保存")
            
            
        return description

    def extract_scenario_description_for_longterm(self, 
                                                  scene_graph, 
                                                  map_env, 
                                                  map_idx,
                                                  future_pred
                                                  ) -> str:
        """
        实现一个完整的、逻辑正确的数据处理管道，用于生成高质量的场景描述。
        1. 使用 extract_carla_scenario 提取最全面的结构化数据。
        2. 使用 LLMGenerateDescription 将该结构化数据转换为生动的自然语言。
        """
        # 1. 确保初始化了必要的组件
        if self.normalizer is None and self.model is not None:
            self.normalizer = self.model.get_normalizer()
        if self.att_normalizer is None and self.model is not None:
            self.att_normalizer = self.model.get_att_normalizer()

        # 2. 调用 extract_carla_scenario 来生成最完整的结构化数据
        # 注意：这里我们只在内部使用数据，所以不保存文件 (output_path=None)
        scenario_data = self.extract_carla_scenario(
            scene_graph=scene_graph,
            map_env=map_env,
            map_idx=map_idx,
            past_traj=scene_graph.past_gt, # 使用真实的过去轨迹
            future_pred=future_pred,       # 使用传入的未来预测
            latent_z=None,                 # 在此阶段不需要
            output_path=None
        )
        # import pdb; pdb.set_trace()

        # 3. 初始化LLM描述生成器，使用指定的provider
        llm_generator = LLMGenerateDescription(provider=self.llm_provider)

        # 4. 调用 llmdescriptiongenerate 方法，传入完整的JSON数据
        llm_description = llm_generator.llmdescriptiongenerate(
            scenario_data=scenario_data
        )

        print("--- Generated Long-Term Description ---")
        print(llm_description)
        print("---------------------------------------")
        return llm_description
        
    def extract_carla_scenario(self, 
                              scene_graph, 
                              map_env=None, 
                              map_idx=None, 
                              past_traj=None,
                              future_pred=None,
                              latent_z=None,
                              output_path=None) -> Dict:
        """
        从场景图中提取CARLA可用的场景数据ke yi
        
        参数:
            scene_graph: 场景图对象，包含车辆信息
            map_env: 地图环境对象
            map_idx: 地图索引
            past_traj: 过去轨迹
            future_pred: 预测的未来轨迹
            latent_z: 潜变量
            output_path: 输出文件路径，如果提供则保存为文件
            
        返回:
            包含场景信息的字典，可用于CARLA模拟
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
        
        # 如果有地图信息，执行轨迹与地图对齐
        if map_env is not None and map_idx is not None:
            trajectories = self._align_trajectories_with_map(trajectories, map_env, map_idx)
        
        # 检查并创建batch信息如果不存在（在这里也需要检查）
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
                
                vehicle_data = {
                    "id": v_idx.item(),
                    "is_ego": bool(is_ego.item()) if isinstance(is_ego, torch.Tensor) else bool(is_ego),
                    "type": semantic_info[v_idx.item()],
                    "length": physical_props[v_idx.item()]["length"],
                    "width": physical_props[v_idx.item()]["width"],
                    "trajectory": vehicle_trajectory,
                    "motion_analysis": motion_analysis  # 新增运动趋势分析
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
                risk_assessment = self._assess_scenario_risks(batch_vehicles_states, dt)
                
                scenario_data["dynamic_analysis"] = {
                    "traffic_flow": traffic_flow,
                    "complexity": complexity,
                    "risk_assessment": risk_assessment
                }
        
        # 如果提供了输出路径，保存为文件
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(scenario_data, f, indent=2)
        
        return scenario_data
    
    def _extract_basic_info(self, scene_graph) -> str:
        """提取基本场景信息"""
        num_vehicles = scene_graph.past.size(0)
        
        # 检查并创建batch信息如果不存在
        if not hasattr(scene_graph, 'batch') or scene_graph.batch is None:
            # 为单场景图创建默认的batch索引
            scene_graph.batch = torch.zeros(num_vehicles, dtype=torch.long, device=scene_graph.past.device)
        
        batch_size = scene_graph.batch.max().item() + 1
        
        return f"场景包含{batch_size}个场景图，共有{num_vehicles}个车辆。时间步长为{self.model.dt if self.model else 0.5}秒。"
    
    def generate_temporal_scenario_description(self, scene_graph, map_env=None, map_idx=None) -> str:
        """
        生成包含完整时间序列信息的场景描述
        
        参数:
            scene_graph: 场景图对象
            map_env: 地图环境对象
            map_idx: 地图索引
            
        返回:
            详细的时间序列场景描述字符串
        """
        # 确保有正确的归一化器
        if self.normalizer is None and self.model is not None:
            self.normalizer = self.model.get_normalizer()
        if self.att_normalizer is None and self.model is not None:
            self.att_normalizer = self.model.get_att_normalizer()
        
        description_parts = []
        
        # 1. 基本场景信息
        basic_info = self._extract_basic_info(scene_graph)
        description_parts.append(f"## 场景基本信息\n{basic_info}")
        
        # 2. 详细的车辆时间序列信息
        vehicles_info = self._extract_vehicles_info(scene_graph)
        description_parts.append(f"## 车辆时间序列分析\n{vehicles_info}")
        
        # 3. 地图上下文信息（如果可用）
        if map_env is not None:
            map_info = self._extract_map_context(scene_graph, map_env, map_idx)
            description_parts.append(f"## 地图上下文\n{map_info}")
        
        # 4. 整体场景动态分析
        dynamic_analysis = self._generate_dynamic_scenario_analysis(scene_graph)
        description_parts.append(f"## 场景动态分析\n{dynamic_analysis}")
        
        return "\n\n".join(description_parts)
    
    def _generate_dynamic_scenario_analysis(self, scene_graph) -> str:
        """
        生成整体场景的动态分析
        
        参数:
            scene_graph: 场景图对象
            
        返回:
            动态分析字符串
        """
        # 获取完整的过去轨迹
        past_states = scene_graph.past_gt
        if self.normalizer is not None:
            past_states = self.normalizer.unnormalize(past_states)
        
        batch_indices = scene_graph.batch
        unique_batches = torch.unique(batch_indices)
        dt = self.model.dt if self.model else 0.5
        
        analysis_parts = []
        
        for b in unique_batches:
            batch_mask = batch_indices == b
            batch_vehicles = torch.where(batch_mask)[0]
            
            # 分析整体交通流
            traffic_flow = self._analyze_traffic_flow(past_states[batch_vehicles], dt)
            analysis_parts.append(f"场景 {b.item()} 交通流分析: {traffic_flow}")
            
            # 分析场景复杂度
            complexity = self._analyze_scenario_complexity(past_states[batch_vehicles], dt)
            analysis_parts.append(f"场景 {b.item()} 复杂度评估: {complexity}")
            
            # 分析潜在风险点
            risk_assessment = self._assess_scenario_risks(past_states[batch_vehicles], dt)
            analysis_parts.append(f"场景 {b.item()} 风险评估: {risk_assessment}")
        
        return "\n".join(analysis_parts)
    
    def _analyze_traffic_flow(self, vehicles_states, dt):
        """分析交通流特征"""
        if vehicles_states.size(0) < 2:
            return "车辆数量不足，无法分析交通流"
        
        # 计算平均速度
        final_states = vehicles_states[:, -1, :]
        velocities = []
        for i in range(vehicles_states.size(0)):
            if final_states.size(1) > 5:
                vx, vy = final_states[i, 4:6].cpu().numpy()
                velocity = np.sqrt(vx*vx + vy*vy)
                velocities.append(velocity)
        
        if not velocities:
            return "无速度数据"
        
        avg_velocity = np.mean(velocities)
        velocity_std = np.std(velocities)
        
        # 分析速度分布
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
    
    def _assess_scenario_risks(self, vehicles_states, dt):
        """评估场景风险"""
        num_vehicles = vehicles_states.size(0)
        if num_vehicles < 2:
            return "单车场景，无碰撞风险"
        
        risk_factors = []
        
        # 计算车辆间最小距离
        final_positions = []
        for i in range(num_vehicles):
            state = vehicles_states[i, -1, :].cpu().numpy()
            x, y = state[0], state[1]
            final_positions.append((x, y))
        
        min_distance = float('inf')
        for i in range(len(final_positions)):
            for j in range(i+1, len(final_positions)):
                dist = np.sqrt(
                    (final_positions[i][0] - final_positions[j][0])**2 +
                    (final_positions[i][1] - final_positions[j][1])**2
                )
                min_distance = min(min_distance, dist)
        
        if min_distance < 3.0:
            risk_factors.append("极近距离接触")
        elif min_distance < 8.0:
            risk_factors.append("近距离行驶")
        
        # 分析速度差异
        velocities = []
        for i in range(num_vehicles):
            state = vehicles_states[i, -1, :].cpu().numpy()
            if len(state) > 5:
                vx, vy = state[4:6]
                velocity = np.sqrt(vx*vx + vy*vy)
                velocities.append(velocity)
        
        if velocities:
            velocity_range = max(velocities) - min(velocities)
            if velocity_range > 10.0:
                risk_factors.append("速度差异过大")
        
        if not risk_factors:
            return "低风险场景"
        else:
            return f"风险因素: {', '.join(risk_factors)}"
    
    def _extract_vehicles_info(self, scene_graph) -> str:
        """提取车辆信息（包含时间序列）"""
        result = "车辆信息:\n"
        
        # 获取车辆属性（归一化的）
        vehicle_attr = scene_graph.lw  # 长宽信息
        
        # 获取完整的过去轨迹（归一化的）
        past_states = scene_graph.past_gt  # (NA, PT, 6)
        
        # 如果有归一化器，还原实际物理量
        if self.normalizer is not None:
            past_states = self.normalizer.unnormalize(past_states)
        if self.att_normalizer is not None:
            vehicle_attr = self.att_normalizer.unnormalize(vehicle_attr)
            
        # 获取批次信息
        batch_indices = scene_graph.batch
        unique_batches = torch.unique(batch_indices)
        
        # 时间步长
        dt = self.model.dt if self.model else 0.5
        
        for b in unique_batches:
            batch_mask = batch_indices == b
            batch_vehicles = torch.where(batch_mask)[0]
            
            result += f"\n场景 {b.item()} 中的车辆:\n"
            
            # 收集所有车辆的轨迹信息
            ego_trajectory = None
            other_trajectories = []
            ego_idx = None
            
            non_ego_counter = 1  # 非ego车辆的编号从1开始
            for i, v_idx in enumerate(batch_vehicles):
                # 判断是否为自车
                is_ego = v_idx == scene_graph.ptr[b].item()
                
                # 提取时间序列轨迹信息
                temporal_info = self._extract_temporal_trajectories(past_states[v_idx], dt)
                motion_analysis = self._analyze_motion_trends(temporal_info)
                
                # 获取当前状态（最后一帧）
                current_state = past_states[v_idx, -1, :].cpu().numpy()
                x, y, cos_h, sin_h = current_state[:4]
                vx, vy = current_state[4:6] if current_state.shape[0] > 5 else (0, 0)
                
                # 计算当前速度和朝向
                speed = np.sqrt(vx*vx + vy*vy)
                heading = np.arctan2(sin_h, cos_h) * 180 / np.pi
                
                # 获取车辆尺寸
                length, width = vehicle_attr[v_idx].cpu().numpy()
                
                if is_ego:
                    ego_trajectory = temporal_info
                    ego_idx = v_idx
                    result += f"  自车: 当前位置=({x:.2f}, {y:.2f}), " \
                             f"当前速度={speed:.2f}m/s, 朝向={heading:.1f}度, 尺寸={length:.1f}x{width:.1f}m\n"
                    result += f"    运动趋势: {motion_analysis}\n"
                else:
                    other_trajectories.append(temporal_info)
                    result += f"  车辆 {non_ego_counter}: 当前位置=({x:.2f}, {y:.2f}), " \
                             f"当前速度={speed:.2f}m/s, 朝向={heading:.1f}度, 尺寸={length:.1f}x{width:.1f}m\n"
                    result += f"    运动趋势: {motion_analysis}\n"
                    non_ego_counter += 1
            
            # 添加相对运动分析
            if ego_trajectory and other_trajectories:
                relative_motion = self._compute_relative_motion(ego_trajectory, other_trajectories, dt)
                result += f"\n  相对运动分析: {relative_motion}\n"
                
        return result
    
    def _extract_temporal_trajectories(self, vehicle_trajectory, dt):
        """
        提取单个车辆的时间序列轨迹信息
        
        参数:
            vehicle_trajectory: 车辆轨迹 (PT, 6) - [x, y, cos_h, sin_h, vx, vy]
            dt: 时间步长
            
        返回:
            包含轨迹信息的字典
        """
        trajectory_data = []
        PT = vehicle_trajectory.size(0)
        
        for t in range(PT):
            state = vehicle_trajectory[t].cpu().numpy()
            time = -dt * (PT - t - 1)  # 负时间表示过去
            
            # 提取位置和朝向
            x, y, cos_h, sin_h = state[:4]
            heading = np.arctan2(sin_h, cos_h) * 180 / np.pi
            
            # 提取速度
            if state.shape[0] > 5:
                vx, vy = state[4:6]
                velocity = np.sqrt(vx*vx + vy*vy)
            else:
                velocity = 0.0
            
            trajectory_data.append({
                "time": float(time),
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
            trajectory_data: 轨迹数据列表
            
        返回:
            运动趋势描述字符串
        """
        if len(trajectory_data) < 2:
            return "数据不足"
        
        # 计算速度变化
        velocities = [point["velocity"] for point in trajectory_data]
        initial_velocity = velocities[0]
        final_velocity = velocities[-1]
        velocity_change = final_velocity - initial_velocity
        
        # 计算加速度
        if len(velocities) > 1:
            dt = trajectory_data[1]["t"] - trajectory_data[0]["t"]
            acceleration = velocity_change / (dt * (len(velocities) - 1))
        else:
            acceleration = 0.0
        
        # 计算位移
        positions = [(point["x"], point["y"]) for point in trajectory_data]
        total_distance = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += np.sqrt(dx*dx + dy*dy)
        
        # 计算朝向变化
        headings = [point["heading"] for point in trajectory_data]
        heading_change = abs(headings[-1] - headings[0])
        if heading_change > 180:
            heading_change = 360 - heading_change
        
        # 生成运动趋势描述
        trend_parts = []
        
        # 速度趋势
        if abs(velocity_change) > 1.0:  # 速度变化超过1m/s
            if velocity_change > 0:
                trend_parts.append(f"加速({velocity_change:.1f}m/s)")
            else:
                trend_parts.append(f"减速({abs(velocity_change):.1f}m/s)")
        else:
            trend_parts.append("匀速")
        
        # 转向趋势
        if heading_change > 10:  # 朝向变化超过10度
            trend_parts.append(f"转向({heading_change:.1f}度)")
        else:
            trend_parts.append("直行")
        
        # 运动状态
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
    
    
    
    def _get_single_vehicle_map_context(self, vehicle_state_unnorm, map_env, map_idx) -> Dict:
        """
        使用nuScenes地图API，精确分析单个车辆在地图上的上下文信息。

        参数:
            vehicle_state_unnorm: 单个车辆的已反归一化的状态张量 (包含世界坐标)。
            map_env: NuScenesMapEnv 实例。
            map_idx: 当前地图的整数索引。

        返回:
            一个包含高层语义标签的字典。
        """
        context = {
            'is_on_lane': False,
            'lane_direction': 'unknown',
            'is_in_intersection': False,
            'distance_to_centerline': -1.0,
            'lane_type': 'lane' # 默认为普通车道
        }
        
        try:
            # 1. 获取正确的地图对象
            map_name = map_env.map_list[map_idx]
            nusc_map = map_env.nusc_maps[map_name]
            coords = tuple(vehicle_state_unnorm[:2].cpu().numpy())

            # 2. 精确查找最近的车道
            # 使用 get_closest_lane, 指定一个合理的搜索半径
            closest_lane_id = nusc_map.get_closest_lane(coords[0], coords[1], radius=5)

            if not closest_lane_id:
                # 如果在5米内找不到车道，则认为不在车道上
                return context

            context['is_on_lane'] = True
            
            # 3. 获取车道记录和中心线
            lane_record = nusc_map.get('lane', closest_lane_id)
            arcline_path = nusc_map.get_arcline_path(closest_lane_id)

            # 4. 精确计算到中心线的距离
            if arcline_path:
                # 仅使用x, y坐标进行距离计算
                path_xy = [node[:2] for node in arcline_path]
                context['distance_to_centerline'] = self._distance_point_to_path(coords, path_xy)

            # 5. 判断车道方向和类型
            context['is_in_intersection'] = nusc_map.is_in_intersection(coords)
            
            if lane_record.get('lane_type'): # nuScenes-plus 属性
                 context['lane_type'] = lane_record['lane_type']

            if context['is_in_intersection']:
                 context['lane_direction'] = 'turning'
                 context['lane_type'] = 'lane_connector'
            elif 'turn_direction' in lane_record and lane_record['turn_direction'] != 'NONE':
                context['lane_direction'] = lane_record['turn_direction'].lower()
            else:
                context['lane_direction'] = 'straight'
            
            # 6. 判断是否在交叉路口 (已在步骤5集成)

        except Exception as e:
            logger.warning(f"分析车辆地图上下文时出错: {e}")
        
        return context

    def _map_env_description(self, scene_graph, map_env, map_idx) -> str:
        """
        为整个场景生成一个详细的、包含空间和语义信息的地图描述。

        参数:
            scene_graph: 场景图对象。
            map_env: NuScenesMapEnv 实例。
            map_idx: 当前地图的索引。

        返回:
            一个供LLM使用的、信息密集的地图描述字符串。
        """
        if not map_env or map_idx is None:
            return "环境信息: 无可用的地图数据。"

        # 确保有反归一化的状态
        if self.normalizer is None:
            logger.warning("无法生成地图描述，因为缺少归一化器。")
            return "环境信息: 归一化器未初始化。"
            
        last_states_unnorm = self.normalizer.unnormalize(scene_graph.past_gt[:, -1, :])
        
        # 1. 获取总体地图信息和自车状态
        map_name = "未知地图"
        try:
            map_name = map_env.map_list[map_idx.item()]
        except Exception:
            pass
        
        result = f"环境信息: 场景发生在地图 '{map_name}'。\n"

        # 提取自车状态用于相对位置计算
        # 假设单场景或批处理中的第一个为自车
        ego_idx = scene_graph.ptr[0].item() 
        ego_state = last_states_unnorm[ego_idx, :].cpu().numpy()
        ego_pos = ego_state[:2]
        ego_heading = np.arctan2(ego_state[3], ego_state[2])

        # 2. 分析每个车辆的地图上下文和相对关系
        vehicle_contexts = []
        for i in range(last_states_unnorm.size(0)):
            vehicle_state = last_states_unnorm[i]
            context = self._get_single_vehicle_map_context(vehicle_state, map_env, map_idx.item())
            
            is_ego = (i == ego_idx)
            vehicle_id_str = "自车" if is_ego else f"车辆 {i}"

            # 3. 将上下文翻译成自然语言
            desc = f" - {vehicle_id_str}: "
            
            # 添加与自车的相对关系 (非自车)
            if not is_ego:
                v_pos = vehicle_state[:2].cpu().numpy()
                rel_pos = v_pos - ego_pos
                
                # 转换到自车坐标系
                rel_x = rel_pos[0] * np.cos(-ego_heading) - rel_pos[1] * np.sin(-ego_heading)
                rel_y = rel_pos[0] * np.sin(-ego_heading) + rel_pos[1] * np.cos(-ego_heading)
                
                direction = self._get_direction(rel_x, rel_y)
                distance = np.sqrt(rel_x*rel_x + rel_y*rel_y)
                
                desc += f"位于自车{direction}，距离{distance:.1f}米。 "

            if not context['is_on_lane']:
                desc += "当前不在任何已知车道上，可能处于停车场或偏离道路区域。"
            else:
                if context['is_in_intersection']:
                    desc += "正位于一个交叉路口区域内，"
                
                dist_str = f"距中心线{context['distance_to_centerline']:.2f}米"

                if context['lane_direction'] == 'straight':
                    desc += f"行驶在一条直行车道上 ({dist_str})。"
                elif context['lane_direction'] == 'left':
                    desc += f"行驶在一条左转车道上 ({dist_str})。"
                elif context['lane_direction'] == 'right':
                    desc += f"行驶在一条右转车道上 ({dist_str})。"
                elif context['lane_direction'] == 'turning':
                    desc += f"正在通过一个路口连接处进行转弯 ({dist_str})。"
                else:
                    desc += f"所在车道方向未知 ({dist_str})。"
            
            vehicle_contexts.append(desc)

        return result + "\n".join(vehicle_contexts)

    def _extract_map_info(self, scene_graph, map_env, map_idx) -> str:
        """
        提取地图和环境信息。
        此方法现在调用新的核心描述函数 _map_env_description。
        """
        return self._map_env_description(scene_graph, map_env, map_idx)
    
    def _get_direction(self, rel_x: float, rel_y: float) -> str:
        """根据相对坐标获取方向描述"""
        angle = np.arctan2(rel_y, rel_x) * 180 / np.pi
        
        if -22.5 <= angle < 22.5:
            return "前方"
        elif 22.5 <= angle < 67.5:
            return "右前方"
        elif 67.5 <= angle < 112.5:
            return "右侧"
        elif 112.5 <= angle < 157.5:
            return "右后方"
        elif 157.5 <= angle <= 180 or -180 <= angle < -157.5:
            return "后方"
        elif -157.5 <= angle < -112.5:
            return "左后方"
        elif -112.5 <= angle < -67.5:
            return "左侧"
        elif -67.5 <= angle < -22.5:
            return "左前方"
        else:
            return "未知方向"

    def _identify_risks(self, scene_graph, future_pred) -> str:
        """识别场景中的潜在风险点"""
        result = "潜在风险点:\n"
        
        try:
            # 计算TTC (Time-to-Collision)
            ttc_risks = self._compute_ttc_risks(scene_graph, future_pred)
            if ttc_risks:
                result += ttc_risks
                
            # 计算横向最小距离风险
            lat_risks = self._compute_lateral_distance_risks(scene_graph, future_pred)
            if lat_risks:
                result += lat_risks
                
            # 计算偏航速率风险
            yaw_risks = self._compute_yaw_rate_risks(scene_graph, future_pred)
            if yaw_risks:
                result += yaw_risks
                
        except Exception as e:
            result += f"识别风险时出错: {e}\n"
            
        if result == "潜在风险点:\n":
            result += "未检测到明显风险点。\n"
            
        return result
    
    def _compute_ttc_risks(self, scene_graph, future_pred) -> str:
        """计算TTC风险"""
        return ""  # 此处需要根据实际场景实现
    
    def _compute_lateral_distance_risks(self, scene_graph, future_pred) -> str:
        """计算横向最小距离风险"""
        return ""  # 此处需要根据实际场景实现
    
    def _compute_yaw_rate_risks(self, scene_graph, future_pred) -> str:
        """计算偏航速率风险"""
        return ""  # 此处需要根据实际场景实现

    def _extract_complete_trajectories(self, scene_graph, past_traj=None, future_pred=None) -> Dict[int, List]:
        """
        提取完整的车辆轨迹（过去和未来）
        
        参数:
            scene_graph: 场景图对象
            past_traj: 过去轨迹，如果为None则使用scene_graph.past_gt
            future_pred: 未来轨迹，预测或真实
            
        返回:
            字典，键为车辆ID，值为轨迹列表 [{"t": 时间, "x": x坐标, "y": y坐标, "heading": 朝向角度, "velocity": 速度}, ...]
        """
        # 获取过去轨迹
        if past_traj is None:
            past_traj = scene_graph.past_gt
        
        # 如果有归一化器，还原实际物理量
        if self.normalizer is not None:
            if past_traj is not None:
                past_traj = self.normalizer.unnormalize(past_traj)
            if future_pred is not None:
                future_pred = self.normalizer.unnormalize(future_pred)
        
        trajectories = {}
        dt = self.model.dt if self.model else 0.1
        
        # 处理每个车辆
        for v_idx in range(scene_graph.past_gt.size(0)):
            trajectories[v_idx] = []
            
            # 添加过去轨迹
            if past_traj is not None:
                for t in range(past_traj.size(1)):
                    state = past_traj[v_idx, t].cpu().numpy()
                    time = -dt * (past_traj.size(1) - t)
                    
                    # 提取位置和朝向
                    x, y, cos_h, sin_h = state[:4]
                    heading = np.arctan2(sin_h, cos_h) * 180 / np.pi
                    
                    # 提取速度（如果有）
                    velocity = 0.0
                    if state.shape[0] > 5:
                        vx, vy = state[4:6]
                        velocity = np.sqrt(vx*vx + vy*vy)
                    
                    trajectories[v_idx].append({
                        "t": float(time),
                        "x": float(x),
                        "y": float(y),
                        "heading": float(heading),
                        "velocity": float(velocity)
                    })
            
            # 添加未来轨迹
            if future_pred is not None:
                # 处理不同形状的future_pred
                if len(future_pred.shape) == 3:  # (NA, FT, 4+)
                    future = future_pred[v_idx]
                elif len(future_pred.shape) == 4:  # (NA, NS, FT, 4+)
                    # 使用第一个样本，通常是均值轨迹
                    future = future_pred[v_idx, 0]
                else:
                    logger.warning(f"未知的future_pred形状: {future_pred.shape}")
                    continue
                
                for t in range(future.size(0)):
                    state = future[t].cpu().numpy()
                    time = dt * (t + 1)
                    
                    # 提取位置和朝向
                    x, y, cos_h, sin_h = state[:4]
                    heading = np.arctan2(sin_h, cos_h) * 180 / np.pi
                    
                    # 提取速度（如果有）
                    velocity = 0.0
                    if state.shape[0] > 5:
                        vx, vy = state[4:6]
                        velocity = np.sqrt(vx*vx + vy*vy)
                    elif t > 0 and future.size(0) > 1:
                        # 用位置差估计速度
                        prev_state = future[t-1].cpu().numpy()
                        vx = (x - prev_state[0]) / dt
                        vy = (y - prev_state[1]) / dt
                        velocity = np.sqrt(vx*vx + vy*vy)
                    
                    trajectories[v_idx].append({
                        "t": float(time),
                        "x": float(x),
                        "y": float(y),
                        "heading": float(heading),
                        "velocity": float(velocity)
                    })
        
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
    
    def _align_trajectories_with_map(self, trajectories, map_env, map_idx):
        """
        确保轨迹与地图对齐
        
        参数:
            trajectories: 轨迹字典
            map_env: 地图环境对象
            map_idx: 地图索引
            
        返回:
            对齐后的轨迹字典
        """
        # 这个函数在真实场景中需要考虑:
        # 1. 检查轨迹点是否在可行驶区域内
        # 2. 如果不在可行驶区域，则投影到最近的可行驶区域
        # 3. 平滑调整后的轨迹
        
        # 此处简化实现，只添加一个标志表示轨迹已对齐
        aligned_trajectories = trajectories.copy()
        
        for v_id, traj in aligned_trajectories.items():
            for point in traj:
                point["map_aligned"] = True
        
        return aligned_trajectories
    
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
    
    def generate_carla_scenario_script(self, scenario_data, output_path):
        """
        生成CARLA兼容的Python脚本
        
        参数:
            scenario_data: 场景数据字典
            output_path: 输出脚本路径
            
        返回:
            生成的脚本内容
        """
        # 生成CARLA脚本模板
        script = """#!/usr/bin/env python

import carla
import math
import random
import time
import numpy as np
import argparse

def main():
    argparser = argparse.ArgumentParser(description='CARLA Scenario Runner')
    argparser.add_argument('--host', default='localhost', help='IP of the CARLA server')
    argparser.add_argument('--port', default=2000, type=int, help='TCP port of the CARLA server')
    argparser.add_argument('--sync', action='store_true', help='Enable synchronous mode')
    args = argparser.parse_args()

    # 连接到CARLA服务器
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    
    # 获取世界和地图
    world = client.get_world()
    
    # 如果需要，切换到指定地图
    if world.get_map().name != '{map_name}':
        try:
            world = client.load_world('{map_name}')
        except:
            print("无法加载地图 {map_name}, 使用当前地图")
    
    # 设置同步模式
    if args.sync:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = {dt}
        world.apply_settings(settings)
    
    # 创建所有车辆
    vehicles = []
    try:
        blueprint_library = world.get_blueprint_library()
        
        # 生成车辆
{spawn_vehicles}
        
        # 等待一下让场景初始化
        time.sleep(2)
        
        # 开始场景回放
        print("开始场景回放...")
        
        # 将场景数据加载到字典中
        scenario_data = SCENARIO_DATA_PLACEHOLDER
        
        # 执行场景
        frame = 0
        max_frames = {max_frames}
        
        while frame < max_frames:
            if args.sync:
                world.tick()
            else:
                time.sleep({dt})
            
            # 更新每个车辆的位置
{update_vehicles}
            
            frame += 1
        
        print("场景回放完成")
        
    finally:
        # 清理车辆
        print("清理场景...")
        client.apply_batch([carla.command.DestroyActor(v) for v in vehicles])
        
        # 恢复异步模式
        if args.sync:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("用户中断")
"""
        
        # 生成车辆生成代码
        spawn_vehicles_code = ""
        for i, vehicle in enumerate(scenario_data["vehicles"]):
            vehicle_type = vehicle["type"]
            # 根据类型选择合适的车辆蓝图
            if vehicle_type == "ego_vehicle" or vehicle_type == "car":
                bp_filter = "vehicle.tesla.model3"
            elif vehicle_type == "truck":
                bp_filter = "vehicle.carlamotors.carlacola"
            elif vehicle_type == "bus":
                bp_filter = "vehicle.volkswagen.t2"
            elif vehicle_type == "motorcycle":
                bp_filter = "vehicle.yamaha.yzf"
            elif vehicle_type == "bicycle":
                bp_filter = "vehicle.diamondback.century"
            else:
                bp_filter = "vehicle.audi.a2"
            
            # 获取初始位置和朝向
            init_point = vehicle["trajectory"][0]
            x, y = init_point["x"], init_point["y"]
            heading = init_point["heading"]
            
            # CARLA中朝向是以度为单位的，需要转换
            yaw = heading
            
            spawn_code = """
        # 生成车辆 {} ({})
        blueprint = blueprint_library.filter('{}')[0]
        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', '0, 0, 0')  # 黑色
        transform = carla.Transform(
            carla.Location(x={}, y={}, z=0.5),
            carla.Rotation(yaw={})
        )
        vehicle_{} = world.spawn_actor(blueprint, transform)
        vehicles.append(vehicle_{})
        {}
        {}
""".format(i, vehicle_type, bp_filter, x, y, yaw, i, i, 
          '# 设置为自动驾驶模式（仅用于调试）' if vehicle["is_ego"] else '# 非自车',
          'vehicle_' + str(i) + '.set_autopilot(True)' if vehicle["is_ego"] else '')
            spawn_vehicles_code += spawn_code
        
        # 生成车辆更新代码
        update_vehicles_code = ""
        for i, vehicle in enumerate(scenario_data["vehicles"]):
            update_code = """
            # 更新车辆 {}
            if {} < len(scenario_data["vehicles"]) and frame < len(scenario_data["vehicles"][{}]["trajectory"]):
                traj_point = scenario_data["vehicles"][{}]["trajectory"][frame]
                x, y = traj_point["x"], traj_point["y"]
                heading = traj_point["heading"]
                
                # 设置车辆位置和朝向
                transform = carla.Transform(
                    carla.Location(x=x, y=y, z=0.5),
                    carla.Rotation(yaw=heading)
                )
                vehicles[{}].set_transform(transform)
                
                # 如果有速度信息，也设置速度
                if "velocity" in traj_point:
                    velocity = traj_point["velocity"]
                    direction = carla.Vector3D(
                        x=math.cos(math.radians(heading)),
                        y=math.sin(math.radians(heading)),
                        z=0
                    )
                    vehicles[{}].set_target_velocity(direction * velocity)
""".format(i, i, i, i, i, i)
            update_vehicles_code += update_code
        
        # 计算最大帧数
        max_frames = max([len(v["trajectory"]) for v in scenario_data["vehicles"]]) if scenario_data["vehicles"] else 0
        
        # 替换模板中的占位符
        import json
        scenario_dict_str = json.dumps(scenario_data, indent=4)
        
        # 字符串替换
        formatted_script = script.replace('{map_name}', scenario_data.get("map", "Town04"))
        formatted_script = formatted_script.replace('{dt}', str(scenario_data.get("dt", 0.1)))
        formatted_script = formatted_script.replace('{spawn_vehicles}', spawn_vehicles_code)
        formatted_script = formatted_script.replace('{update_vehicles}', update_vehicles_code)
        formatted_script = formatted_script.replace('{max_frames}', str(max_frames))
        formatted_script = formatted_script.replace('SCENARIO_DATA_PLACEHOLDER', scenario_dict_str)
        
        # 如果提供了输出路径，保存脚本
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(formatted_script)
        
        return formatted_script
    
    def save_scenario_description(self, description: str, save_dir: str = "scenario_descriptions") -> str:
        """
        保存场景描述到固定文件
        
        参数:
            description: 场景描述文本
            save_dir: 保存目录
            
        返回:
            保存的文件路径
        """
        import time
        
        logger.info(f"=== 开始保存场景描述 ===")
        logger.info(f"保存目录: {save_dir}")
        logger.info(f"当前工作目录: {os.getcwd()}")
        logger.info(f"描述长度: {len(description)} 字符")
        
        # 确保保存目录存在
        try:
            # 创建绝对路径
            abs_save_dir = os.path.abspath(save_dir)
            logger.info(f"绝对保存路径: {abs_save_dir}")
            
            os.makedirs(abs_save_dir, exist_ok=True)
            logger.info(f"保存目录创建成功: {abs_save_dir}")
            
            # 检查目录权限
            if not os.access(abs_save_dir, os.W_OK):
                logger.error(f"目录无写入权限: {abs_save_dir}")
                return ""
                
        except Exception as e:
            logger.error(f"创建保存目录失败: {e}")
            return ""
        
        # 使用固定的文件名，确保每次都保存到同一个文件
        file_path = os.path.join(abs_save_dir, "current_scenario_description.txt")
        metadata_path = os.path.join(abs_save_dir, "scenario_metadata.json")
        
        logger.info(f"场景描述文件路径: {file_path}")
        logger.info(f"元数据文件路径: {metadata_path}")
        
        try:
            # 保存场景描述
            logger.info("开始写入场景描述文件...")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(description)
            logger.info(f"场景描述文件写入成功，大小: {os.path.getsize(file_path)} 字节")
            
            # 同时保存元数据（时间戳等）
            logger.info("开始写入元数据文件...")
            metadata = {
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "description_file": "current_scenario_description.txt",
                "description_length": len(description),
                "file_size_bytes": os.path.getsize(file_path),
                "absolute_path": file_path,
                "status": "saved"
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"元数据文件写入成功，大小: {os.path.getsize(metadata_path)} 字节")
            
            logger.info(f"=== 场景描述保存完成: {file_path} ===")
            return file_path
            
        except Exception as e:
            logger.error(f"保存场景描述失败: {e}")
            import traceback
            logger.error(f"保存异常详情: {traceback.format_exc()}")
            return ""
    
    @staticmethod
    def load_scenario_description(save_dir: str = "scenario_descriptions") -> str:
        """
        从固定文件加载场景描述
        
        参数:
            save_dir: 保存目录
            
        返回:
            场景描述文本，如果加载失败则返回空字符串
        """
        file_path = "./STRIVE/scenario_descriptions/current_scenario_description.txt"
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    description = f.read()
                logger.info(f"成功加载场景描述: {file_path}")
                return description
            else:
                logger.warning(f"场景描述文件不存在: {file_path}")
                return ""
                
        except Exception as e:
            logger.error(f"加载场景描述失败: {e}")
            return "" 