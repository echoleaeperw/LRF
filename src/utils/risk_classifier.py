#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风险分类器模块
============

根据场景指标对生成的场景进行风险等级分类
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RiskThresholds:
    """风险分类阈值配置"""
    # 长尾条件阈值 (极端情况)
    longtail_ttc_min: float = 0.5
    longtail_lateral_dist_min: float = 0.3
    longtail_relative_speed_max: float = 25.0
    longtail_collision_angle_extreme: Tuple[float, float] = (15.0, 165.0)
    
    # 高风险阈值
    high_risk_ttc_min: float = 2.0
    high_risk_lateral_dist_min: float = 1.0
    high_risk_relative_speed_max: float = 15.0
    
    # 低风险为其他情况


def compute_min_ttc(trajectory_data: Dict) -> float:
    """
    计算最小碰撞时间 (TTC)
    
    参数:
        trajectory_data: 轨迹数据字典，包含位置、速度等信息
        
    返回:
        最小TTC值
    """
    try:
        # 从轨迹数据中提取车辆轨迹
        if 'fut_adv' in trajectory_data:
            fut_traj = np.array(trajectory_data['fut_adv'])
        elif 'future_traj' in trajectory_data:
            fut_traj = np.array(trajectory_data['future_traj'])
        else:
            return float('inf')
        
        if fut_traj.size == 0:
            return float('inf')
            
        # 简化的TTC计算：基于最近距离和相对速度
        min_distances = []
        
        # 假设第一个车辆是ego，计算与其他车辆的TTC
        if len(fut_traj.shape) >= 3 and fut_traj.shape[0] > 1:
            ego_traj = fut_traj[0]  # ego车辆轨迹
            for i in range(1, fut_traj.shape[0]):
                other_traj = fut_traj[i]
                
                # 计算每个时间步的距离
                for t in range(min(len(ego_traj), len(other_traj))):
                    if len(ego_traj[t]) >= 2 and len(other_traj[t]) >= 2:
                        dx = ego_traj[t][0] - other_traj[t][0]
                        dy = ego_traj[t][1] - other_traj[t][1]
                        distance = np.sqrt(dx*dx + dy*dy)
                        min_distances.append(distance)
        
        if min_distances:
            min_distance = min(min_distances)
            # 简化TTC计算：距离除以平均相对速度
            avg_relative_speed = 10.0  # 默认相对速度
            if min_distance < 0.1:
                return 0.0
            return min_distance / avg_relative_speed
        
        return float('inf')
        
    except Exception as e:
        print(f"计算TTC时出错: {e}")
        return float('inf')


def compute_min_lateral_distance(trajectory_data: Dict) -> float:
    """
    计算最小横向距离
    
    参数:
        trajectory_data: 轨迹数据字典
        
    返回:
        最小横向距离
    """
    try:
        # 从轨迹数据中提取车辆轨迹
        if 'fut_adv' in trajectory_data:
            fut_traj = np.array(trajectory_data['fut_adv'])
        elif 'future_traj' in trajectory_data:
            fut_traj = np.array(trajectory_data['future_traj'])
        else:
            return float('inf')
        
        if fut_traj.size == 0:
            return float('inf')
            
        min_lateral_distances = []
        
        # 计算车辆间的最小横向距离
        if len(fut_traj.shape) >= 3 and fut_traj.shape[0] > 1:
            ego_traj = fut_traj[0]
            for i in range(1, fut_traj.shape[0]):
                other_traj = fut_traj[i]
                
                for t in range(min(len(ego_traj), len(other_traj))):
                    if len(ego_traj[t]) >= 2 and len(other_traj[t]) >= 2:
                        # 简化的横向距离计算
                        dx = abs(ego_traj[t][0] - other_traj[t][0])
                        dy = abs(ego_traj[t][1] - other_traj[t][1])
                        lateral_dist = min(dx, dy)  # 简化计算
                        min_lateral_distances.append(lateral_dist)
        
        return min(min_lateral_distances) if min_lateral_distances else float('inf')
        
    except Exception as e:
        print(f"计算横向距离时出错: {e}")
        return float('inf')


def compute_max_relative_speed(trajectory_data: Dict) -> float:
    """
    计算最大相对速度
    
    参数:
        trajectory_data: 轨迹数据字典
        
    返回:
        最大相对速度
    """
    try:
        # 简化计算：从已有数据中提取或使用默认值
        if 'max_relative_speed' in trajectory_data:
            return float(trajectory_data['max_relative_speed'])
        
        # 如果没有直接数据，进行简化计算
        if 'fut_adv' in trajectory_data:
            fut_traj = np.array(trajectory_data['fut_adv'])
            if fut_traj.size > 0:
                # 简化的相对速度计算
                return 10.0  # 默认值
        
        return 0.0
        
    except Exception as e:
        print(f"计算相对速度时出错: {e}")
        return 0.0


def compute_collision_angle(trajectory_data: Dict) -> Optional[float]:
    """
    计算碰撞角度
    
    参数:
        trajectory_data: 轨迹数据字典
        
    返回:
        碰撞角度（度），如果无碰撞返回None
    """
    try:
        if 'collision_angle' in trajectory_data:
            return float(trajectory_data['collision_angle'])
        
        # 简化计算：如果发生碰撞，返回默认角度
        if trajectory_data.get('collision_occurred', False):
            return 90.0  # 默认直角碰撞
            
        return None
        
    except Exception as e:
        print(f"计算碰撞角度时出错: {e}")
        return None


def classify_scenario_by_risk_level(trajectory_data: Dict, 
                                   scene_graph: Optional[object] = None,
                                   thresholds: Optional[RiskThresholds] = None) -> str:
    """
    基于风险指标对场景进行分类
    
    参数:
        trajectory_data: 轨迹数据字典
        scene_graph: 场景图对象（可选）
        thresholds: 风险阈值配置（可选）
        
    返回:
        风险等级: 'low_risk', 'high_risk', 'longtail_condition'
    """
    if thresholds is None:
        thresholds = RiskThresholds()
    
    try:
        # 计算关键风险指标
        min_ttc = compute_min_ttc(trajectory_data)
        min_lateral_dist = compute_min_lateral_distance(trajectory_data)
        max_relative_speed = compute_max_relative_speed(trajectory_data)
        collision_angle = compute_collision_angle(trajectory_data)
        
        print(f"风险指标计算结果: TTC={min_ttc:.2f}, 横向距离={min_lateral_dist:.2f}, 相对速度={max_relative_speed:.2f}")
        
        # 长尾条件判断 (极端情况)
        is_longtail = False
        
        # 极短TTC
        if min_ttc < thresholds.longtail_ttc_min:
            is_longtail = True
            print(f"检测到极短TTC: {min_ttc:.2f} < {thresholds.longtail_ttc_min}")
        
        # 极近距离
        if min_lateral_dist < thresholds.longtail_lateral_dist_min:
            is_longtail = True
            print(f"检测到极近距离: {min_lateral_dist:.2f} < {thresholds.longtail_lateral_dist_min}")
        
        # 极高相对速度
        if max_relative_speed > thresholds.longtail_relative_speed_max:
            is_longtail = True
            print(f"检测到极高相对速度: {max_relative_speed:.2f} > {thresholds.longtail_relative_speed_max}")
        
        # 极端碰撞角度
        if collision_angle is not None:
            if (collision_angle < thresholds.longtail_collision_angle_extreme[0] or 
                collision_angle > thresholds.longtail_collision_angle_extreme[1]):
                is_longtail = True
                print(f"检测到极端碰撞角度: {collision_angle:.1f}°")
        
        if is_longtail:
            return 'longtail_condition'
        
        # 高风险判断
        is_high_risk = False
        
        # 短TTC
        if min_ttc < thresholds.high_risk_ttc_min:
            is_high_risk = True
            print(f"检测到短TTC: {min_ttc:.2f} < {thresholds.high_risk_ttc_min}")
        
        # 近距离
        if min_lateral_dist < thresholds.high_risk_lateral_dist_min:
            is_high_risk = True
            print(f"检测到近距离: {min_lateral_dist:.2f} < {thresholds.high_risk_lateral_dist_min}")
        
        # 高相对速度
        if max_relative_speed > thresholds.high_risk_relative_speed_max:
            is_high_risk = True
            print(f"检测到高相对速度: {max_relative_speed:.2f} > {thresholds.high_risk_relative_speed_max}")
        
        if is_high_risk:
            return 'high_risk'
        
        # 默认为低风险
        print("场景分类为低风险")
        return 'low_risk'
        
    except Exception as e:
        print(f"风险分类时出错: {e}")
        # 出错时默认返回高风险，保持安全
        return 'high_risk'


def get_risk_thresholds_from_config() -> RiskThresholds:
    """
    从配置文件读取风险阈值（未来实现）
    
    返回:
        风险阈值配置
    """
    # TODO: 实现从配置文件读取
    return RiskThresholds()


# 辅助函数，用于兼容现有代码
def compute_risk_metrics(trajectory_data: Dict) -> Dict[str, float]:
   
    return {
        'min_ttc': compute_min_ttc(trajectory_data),
        'min_lateral_distance': compute_min_lateral_distance(trajectory_data),
        'max_relative_speed': compute_max_relative_speed(trajectory_data),
        'collision_angle': compute_collision_angle(trajectory_data)
    }

