#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对抗性轨迹评估模块
=================

实现四个关键评估指标：
1. 轨迹真实性（Trajectory Realism）
2. 交互合理性（Interaction Consistency）  
3. 长尾事件覆盖率（Long-Tail Event Coverage）
4. Sim-to-Real差距（Simulation-to-Real Gap）

基于潜变量Z的深度分析和轨迹质量评估
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from losses.adv_gen_nusc import compute_velocities_from_traj, compute_longitudinal_metrics, check_single_veh_coll
from utils.logger import Logger
from longtail_coverage_analyzer import LongtailCoverageAnalyzer


@dataclass
class EvaluationResults:
    """评估结果数据结构"""
    trajectory_realism: Dict[str, float]
    interaction_consistency: Dict[str, float] 
    longtail_coverage: Dict[str, float]
    sim_to_real_gap: Dict[str, float]
    detailed_metrics: Dict[str, Any]
    

class AdversarialTrajectoryEvaluator:
    """对抗性轨迹评估器"""
    
    def __init__(self, device='cuda', verbose=True):
        self.device = device
        self.verbose = verbose
        self.real_data_stats = {}  # 存储真实数据统计信息
        self.evaluation_cache = {}  # 缓存评估结果
        
        # 物理约束参数
        self.physics_constraints = {
            'max_speed': 50.0,      # m/s (180 km/h)
            'max_accel': 8.0,       # m/s²
            'max_decel': 10.0,      # m/s²
            'max_yaw_rate': 30.0,   # deg/s
            'max_lat_accel': 8.0,   # m/s²
            'min_turning_radius': 3.0  # m
        }
        
        # 交互行为模式定义
        self.interaction_patterns = {
            'following': {'min_distance': 2.0, 'max_distance': 50.0},
            'lane_change': {'lateral_threshold': 1.5, 'duration_threshold': 3.0},
            'overtaking': {'speed_diff_threshold': 2.0},
            'merging': {'angle_threshold': 15.0},
            'yielding': {'decel_threshold': 2.0}
        }
        
        Logger.log("AdversarialTrajectoryEvaluator 初始化完成")
    
    def evaluate_comprehensive(self, 
                             z_original: torch.Tensor,
                             z_adversarial: torch.Tensor, 
                             traj_original: torch.Tensor,
                             traj_adversarial: torch.Tensor,
                             traj_real_dataset: torch.Tensor,
                             scene_graph: Any,
                             model: Any) -> EvaluationResults:
        """
        综合评估对抗性轨迹
        
        参数:
            z_original: 原始潜变量 [NA, z_dim]
            z_adversarial: 对抗性潜变量 [NA, z_dim]
            traj_original: 原始轨迹 [NA, T, 4]
            traj_adversarial: 对抗性轨迹 [NA, T, 4]
            traj_real_dataset: 真实数据集轨迹 [N_real, T, 4]
            scene_graph: 场景图
            model: 交通模型
        """
        if self.verbose:
            Logger.log("开始综合评估对抗性轨迹...")
        
        # 1. 轨迹真实性评估
        trajectory_realism = self._evaluate_trajectory_realism(
            z_original, z_adversarial, traj_original, traj_adversarial, model
        )
        
        # 2. 交互合理性评估
        interaction_consistency = self._evaluate_interaction_consistency(
            traj_adversarial, scene_graph
        )
        
        # 3. 长尾事件覆盖率评估
        longtail_coverage = self._evaluate_longtail_coverage(
            z_adversarial, traj_adversarial, traj_real_dataset
        )
        
        # 4. Sim-to-Real差距评估
        sim_to_real_gap = self._evaluate_sim_to_real_gap(
            traj_adversarial, traj_real_dataset
        )
        
        # 5. 详细指标计算
        detailed_metrics = self._compute_detailed_metrics(
            z_original, z_adversarial, traj_original, traj_adversarial
        )
        
        results = EvaluationResults(
            trajectory_realism=trajectory_realism,
            interaction_consistency=interaction_consistency,
            longtail_coverage=longtail_coverage,
            sim_to_real_gap=sim_to_real_gap,
            detailed_metrics=detailed_metrics
        )
        
        if self.verbose:
            self._print_evaluation_summary(results)
        
        return results
    
    def _evaluate_trajectory_realism(self, 
                                   z_original: torch.Tensor,
                                   z_adversarial: torch.Tensor,
                                   traj_original: torch.Tensor, 
                                   traj_adversarial: torch.Tensor,
                                   model: Any) -> Dict[str, float]:
        """评估轨迹真实性"""
        
        # 1. Z偏离程度分析
        z_deviation = torch.norm(z_adversarial - z_original, dim=-1).mean().item()
        z_deviation_std = torch.norm(z_adversarial - z_original, dim=-1).std().item()
        
        # 2. Z在先验分布中的合理性
        with torch.no_grad():
            # 假设我们有先验分布的均值和方差
            z_mean = z_original.mean(dim=0)
            z_var = z_original.var(dim=0)
            
            # 计算对抗性z的对数似然
            z_logprob = -0.5 * torch.sum((z_adversarial - z_mean)**2 / (z_var + 1e-6), dim=-1)
            z_likelihood_score = z_logprob.mean().item()
        
        # 3. 物理约束检查
        physics_realism = self._check_physics_constraints(traj_adversarial)
        
        # 4. 运动平滑性检查  
        motion_smoothness = self._check_motion_smoothness(traj_adversarial)
        
        # 5. 轨迹连续性检查
        trajectory_continuity = self._check_trajectory_continuity(traj_adversarial)
        
        return {
            'z_deviation_mean': z_deviation,
            'z_deviation_std': z_deviation_std,
            'z_likelihood_score': z_likelihood_score,
            'physics_realism_score': physics_realism,
            'motion_smoothness_score': motion_smoothness,
            'trajectory_continuity_score': trajectory_continuity,
            'overall_realism_score': np.mean([
                1.0 / (1.0 + z_deviation),  # Z偏离越小越好
                max(0, z_likelihood_score / 10.0),  # 似然得分
                physics_realism,
                motion_smoothness,
                trajectory_continuity
            ])
        }
    
    def _evaluate_interaction_consistency(self, 
                                        traj_adversarial: torch.Tensor,
                                        scene_graph: Any) -> Dict[str, float]:
        """评估交互合理性"""
        
        NA, T, _ = traj_adversarial.shape
        
        # 1. 车辆间距离合理性
        inter_vehicle_distances = self._compute_inter_vehicle_distances(traj_adversarial)
        distance_realism = self._evaluate_distance_realism(inter_vehicle_distances)
        
        # 2. 相对速度合理性
        relative_velocities = self._compute_relative_velocities(traj_adversarial)
        velocity_consistency = self._evaluate_velocity_consistency(relative_velocities)
        
        # 3. 交互行为模式识别
        interaction_patterns = self._identify_interaction_patterns(traj_adversarial)
        pattern_realism = self._evaluate_pattern_realism(interaction_patterns)
        
        # 4. 社会性驾驶规范检查
        social_compliance = self._check_social_compliance(traj_adversarial)
        
        # 5. 碰撞合理性（如果发生碰撞）
        collision_realism = self._evaluate_collision_realism(traj_adversarial)
        
        return {
            'distance_realism_score': distance_realism,
            'velocity_consistency_score': velocity_consistency,
            'pattern_realism_score': pattern_realism,
            'social_compliance_score': social_compliance,
            'collision_realism_score': collision_realism,
            'overall_interaction_score': np.mean([
                distance_realism,
                velocity_consistency, 
                pattern_realism,
                social_compliance,
                collision_realism
            ])
        }
    
    def _evaluate_longtail_coverage(self, 
                                  z_adversarial: torch.Tensor,
                                  traj_adversarial: torch.Tensor,
                                  traj_real_dataset: torch.Tensor) -> Dict[str, float]:
        """评估长尾事件覆盖率"""
        
        # 1. Z空间探索程度 (原有方法)
        z_space_coverage = self._compute_z_space_coverage(z_adversarial)
        
        # 2. 稀有事件识别 (原有方法)
        rare_events = self._identify_rare_events(traj_adversarial, traj_real_dataset)
        
        # 3. 事件多样性评估 (原有方法)
        event_diversity = self._compute_event_diversity(traj_adversarial)
        
        # 4. 极端行为检测 (原有方法)
        extreme_behaviors = self._detect_extreme_behaviors(traj_adversarial)
        
        # 5. 覆盖率统计 (原有方法)
        coverage_statistics = self._compute_coverage_statistics(
            traj_adversarial, traj_real_dataset
        )

        # 6. 新增：基于物理特征的覆盖率分析
        try:
            physical_analyzer = LongtailCoverageAnalyzer(
                real_world_trajectories=traj_real_dataset, 
                device=self.device
            )
            physical_coverage_results = physical_analyzer.calculate_coverage(traj_adversarial)
        except Exception as e:
            Logger.log(f"物理长尾覆盖率分析失败: {e}")
            physical_coverage_results = {
                'physical_longtail_hit_rate': 0.0,
                'physical_num_longtail_scenarios': 0,
                'physical_triggered_conditions': ['analysis_failed']
            }
        
        return {
            # 保留原有指标
            'z_space_coverage_score': z_space_coverage,
            'rare_event_count': len(rare_events),
            'event_diversity_score': event_diversity,
            'extreme_behavior_ratio': extreme_behaviors,
            'coverage_completeness': coverage_statistics,
            
            # 新增物理覆盖率指标
            'physical_longtail_hit_rate': physical_coverage_results.get('longtail_hit_rate', 0.0),
            'physical_num_longtail_scenarios': physical_coverage_results.get('num_longtail_scenarios', 0),
            'physical_triggered_conditions': physical_coverage_results.get('triggered_conditions', []),

            # 更新总体分数计算
            'overall_longtail_score': np.mean([
                z_space_coverage,
                min(1.0, len(rare_events) / 10.0),  # 归一化稀有事件数量
                event_diversity,
                extreme_behaviors,
                physical_coverage_results.get('longtail_hit_rate', 0.0) # 将新的命中率加入平均
            ])
        }
    
    def _evaluate_sim_to_real_gap(self, 
                                traj_adversarial: torch.Tensor,
                                traj_real_dataset: torch.Tensor) -> Dict[str, float]:
        """
        评估Sim-to-Real差距
        =======================

        该指标旨在量化生成轨迹与真实世界数据之间的差异，确保生成场景的真实性。
        我们从四个维度评估Sim-to-Real差距：统计分布、运动特征、频域特性和高层语义行为。
        
        **Sim-to-Real Gap (SRG)** 定义为四个子指标差距的平均值：
        
        .. math::
            \mathrm{SRG} = \frac{1}{4} \left( G_{\text{dist}} + G_{\text{motion}} + G_{\text{freq}} + G_{\text{sem}} \right)

        最终得分为 (1 - SRG)，分数越高表示与真实数据越接近。

        其中:
            - **\(G_{\text{dist}}\) (Distribution Gap):** 
              量化模拟轨迹 (\( \mathcal{T}_{\text{sim}} \)) 与真实轨迹 (\( \mathcal{T}_{\text{real}} \)) 在关键运动学特征分布上的差异。
              该差距通过计算两者速度分布的Kullback-Leibler (KL)散度来衡量。

            - **\(G_{\text{motion}}\) (Motion Feature Gap):** 
              衡量基础运动学统计数据的差异，如速度和加速度的均值与标准差。
              它是这些统计量在模拟与真实数据间归一化差异的聚合。

            - **\(G_{\text{freq}}\) (Frequency Domain Gap):** 
              通过比较轨迹在频域上的特性来评估运动的平滑性和周期性。
              该差距通过计算位置序列的功率谱密度（Power Spectral Density, PSD）的平均差异来确定。

            - **\(G_{\text{sem}}\) (Semantic Gap):** 
              评估高层驾驶行为模式的频率差异。我们首先从轨迹中识别出如“加速”、“减速”、“高速巡航”等语义行为，
              然后计算这些行为在模拟与真实数据集中出现的频率差异。
        """
        
        # 1. 统计分布差异
        distribution_gap = self._compute_distribution_gap(traj_adversarial, traj_real_dataset)
        
        # 2. 运动特征对比
        motion_feature_gap = self._compute_motion_feature_gap(traj_adversarial, traj_real_dataset)
        
        # 3. 频谱分析差异
        frequency_domain_gap = self._compute_frequency_domain_gap(traj_adversarial, traj_real_dataset)
        
        # 4. 高层语义差异
        semantic_gap = self._compute_semantic_gap(traj_adversarial, traj_real_dataset)
        
        return {
            'distribution_gap_score': distribution_gap,
            'motion_feature_gap_score': motion_feature_gap,
            'frequency_domain_gap_score': frequency_domain_gap,
            'semantic_gap_score': semantic_gap,
            'overall_sim_to_real_score': 1.0 - np.mean([
                distribution_gap,
                motion_feature_gap,
                frequency_domain_gap,
                semantic_gap
            ])  # 差距越小，得分越高
        }
    
    # ===== 物理约束检查 =====
    def _check_physics_constraints(self, trajectories: torch.Tensor) -> float:
        """检查物理约束"""
        NA, T, _ = trajectories.shape
        violations = 0
        total_checks = 0
        
        # 计算速度和加速度
        velocities, speeds = compute_velocities_from_traj(trajectories, dt=0.1)
        
        # 1. 速度约束检查
        speed_violations = (speeds > self.physics_constraints['max_speed']).sum()
        violations += speed_violations
        total_checks += speeds.numel()
        
        # 2. 加速度约束检查
        if T > 2:
            accelerations = (speeds[:, 1:] - speeds[:, :-1]) / 0.1
            accel_violations = (torch.abs(accelerations) > self.physics_constraints['max_accel']).sum()
            violations += accel_violations
            total_checks += accelerations.numel()
        
        # 3. 偏航率约束检查
        if T > 1:
            yaw_angles = torch.atan2(trajectories[:, :-1, 3], trajectories[:, :-1, 2])
            yaw_rates = torch.abs(yaw_angles[:, 1:] - yaw_angles[:, :-1]) / 0.1
            yaw_rate_violations = (yaw_rates > np.radians(self.physics_constraints['max_yaw_rate'])).sum()
            violations += yaw_rate_violations
            total_checks += yaw_rates.numel()
        
        physics_score = 1.0 - (violations.float() / total_checks).item()
        return max(0.0, physics_score)
    
    def _check_motion_smoothness(self, trajectories: torch.Tensor) -> float:
        """检查运动平滑性"""
        NA, T, _ = trajectories.shape
        
        if T < 3:
            return 1.0
        
        # 计算二阶导数（加加速度）
        positions = trajectories[:, :, :2]
        vel = positions[:, 1:] - positions[:, :-1]  # [NA, T-1, 2]
        accel = vel[:, 1:] - vel[:, :-1]  # [NA, T-2, 2]
        jerk = accel[:, 1:] - accel[:, :-1]  # [NA, T-3, 2]
        
        # 计算加加速度的平均值作为平滑性指标
        jerk_magnitude = torch.norm(jerk, dim=-1)
        smoothness_score = 1.0 / (1.0 + jerk_magnitude.mean().item())
        
        return smoothness_score
    
    def _check_trajectory_continuity(self, trajectories: torch.Tensor) -> float:
        """检查轨迹连续性"""
        NA, T, _ = trajectories.shape
        
        if T < 2:
            return 1.0
        
        # 检查位置跳跃
        position_diffs = torch.norm(trajectories[:, 1:, :2] - trajectories[:, :-1, :2], dim=-1)
        max_allowed_jump = 5.0  # 5米的最大跳跃
        jumps = (position_diffs > max_allowed_jump).float()
        
        continuity_score = 1.0 - jumps.mean().item()
        return continuity_score
    
    # ===== 交互分析函数 =====
    def _compute_inter_vehicle_distances(self, trajectories: torch.Tensor) -> torch.Tensor:
        """计算车辆间距离"""
        NA, T, _ = trajectories.shape
        distances = torch.zeros(NA, NA, T)
        
        for t in range(T):
            positions = trajectories[:, t, :2]  # [NA, 2]
            for i in range(NA):
                for j in range(i+1, NA):
                    dist = torch.norm(positions[i] - positions[j])
                    distances[i, j, t] = dist
                    distances[j, i, t] = dist
        
        return distances
    
    def _evaluate_distance_realism(self, distances: torch.Tensor) -> float:
        """评估距离合理性"""
        # 检查最小安全距离
        min_safe_distance = 2.0  # 2米
        too_close_ratio = (distances < min_safe_distance).float().mean()
        
        # 检查合理的交互距离范围
        reasonable_distances = ((distances >= 2.0) & (distances <= 50.0)).float().mean()
        
        distance_score = reasonable_distances.item() * (1.0 - too_close_ratio.item())
        return distance_score
    
    def _compute_relative_velocities(self, trajectories: torch.Tensor) -> torch.Tensor:
        """计算相对速度"""
        velocities, _ = compute_velocities_from_traj(trajectories, dt=0.1)
        NA, T, _ = velocities.shape
        
        relative_vels = torch.zeros(NA, NA, T, 2)
        for i in range(NA):
            for j in range(i+1, NA):
                rel_vel = velocities[i] - velocities[j]
                relative_vels[i, j] = rel_vel
                relative_vels[j, i] = -rel_vel
        
        return relative_vels
    
    def _evaluate_velocity_consistency(self, relative_velocities: torch.Tensor) -> float:
        """评估速度一致性"""
        # 计算相对速度的合理性
        rel_speeds = torch.norm(relative_velocities, dim=-1)
        
        # 极端相对速度检查
        extreme_rel_speed_threshold = 20.0  # m/s
        extreme_ratio = (rel_speeds > extreme_rel_speed_threshold).float().mean()
        
        consistency_score = 1.0 - extreme_ratio.item()
        return consistency_score
    
    # ===== 长尾事件分析 =====
    def _compute_z_space_coverage(self, z_adversarial: torch.Tensor) -> float:
        """计算Z空间覆盖程度"""
        # 使用主成分分析评估Z空间的探索程度
        z_np = z_adversarial.detach().cpu().numpy()
        
        # 计算协方差矩阵的特征值
        cov_matrix = np.cov(z_np.T)
        eigenvalues = np.linalg.eigvals(cov_matrix).real
        
        # 使用有效秩来衡量覆盖程度
        total_variance = np.sum(eigenvalues)
        effective_rank = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
        
        coverage_score = effective_rank / len(eigenvalues)
        return min(1.0, coverage_score)
    
    def _identify_rare_events(self, traj_adversarial: torch.Tensor, 
                            traj_real_dataset: torch.Tensor) -> List[Dict]:
        """识别稀有事件"""
        rare_events = []
        
        # 1. 极端加速度事件
        velocities, speeds = compute_velocities_from_traj(traj_adversarial, dt=0.1)
        accelerations = (speeds[:, 1:] - speeds[:, :-1]) / 0.1
        
        extreme_accel_threshold = 6.0  # m/s²
        extreme_accel_events = torch.abs(accelerations) > extreme_accel_threshold
        
        if extreme_accel_events.any():
            rare_events.append({
                'type': 'extreme_acceleration',
                'count': extreme_accel_events.sum().item(),
                'severity': 'high'
            })
        
        # 2. 急转弯事件
        yaw_angles = torch.atan2(traj_adversarial[:, :-1, 3], traj_adversarial[:, :-1, 2])
        yaw_rates = torch.abs(yaw_angles[:, 1:] - yaw_angles[:, :-1]) / 0.1
        
        extreme_yaw_threshold = np.radians(25.0)  # 25度/秒
        extreme_yaw_events = yaw_rates > extreme_yaw_threshold
        
        if extreme_yaw_events.any():
            rare_events.append({
                'type': 'extreme_turning',
                'count': extreme_yaw_events.sum().item(),
                'severity': 'medium'
            })
        
        # 3. 近距离交互事件
        distances = self._compute_inter_vehicle_distances(traj_adversarial)
        close_interaction_threshold = 3.0  # 3米
        close_interactions = distances < close_interaction_threshold
        close_interactions = close_interactions & (distances > 0)  # 排除自身
        
        if close_interactions.any():
            rare_events.append({
                'type': 'close_interaction',
                'count': close_interactions.sum().item(),
                'severity': 'high'
            })
        
        return rare_events
    
    def _compute_event_diversity(self, trajectories: torch.Tensor) -> float:
        """计算事件多样性"""
        # 使用轨迹特征的熵来衡量多样性
        NA, T, _ = trajectories.shape
        
        # 提取关键特征
        velocities, speeds = compute_velocities_from_traj(trajectories, dt=0.1)
        
        # 将特征离散化并计算熵
        speed_bins = torch.histc(speeds.flatten(), bins=10, min=0, max=20)
        speed_probs = speed_bins / speed_bins.sum()
        speed_entropy = -torch.sum(speed_probs * torch.log(speed_probs + 1e-8))
        
        # 归一化熵值
        max_entropy = np.log(10)  # 10个bins的最大熵
        diversity_score = (speed_entropy / max_entropy).item()
        
        return diversity_score
    
    def _detect_extreme_behaviors(self, trajectories: torch.Tensor) -> float:
        """检测极端行为比例"""
        total_behaviors = 0
        extreme_behaviors = 0
        
        # 1. 极端速度
        velocities, speeds = compute_velocities_from_traj(trajectories, dt=0.1)
        high_speed_threshold = 25.0  # m/s (90 km/h)
        extreme_speeds = speeds > high_speed_threshold
        extreme_behaviors += extreme_speeds.sum().item()
        total_behaviors += speeds.numel()
        
        # 2. 急刹车
        if speeds.shape[1] > 1:
            decelerations = -(speeds[:, 1:] - speeds[:, :-1]) / 0.1
            hard_braking_threshold = 5.0  # m/s²
            hard_braking = decelerations > hard_braking_threshold
            extreme_behaviors += hard_braking.sum().item()
            total_behaviors += decelerations.numel()
        
        extreme_ratio = extreme_behaviors / total_behaviors if total_behaviors > 0 else 0.0
        return extreme_ratio
    
    # ===== Sim-to-Real差距分析 =====
    def _compute_distribution_gap(self, traj_sim: torch.Tensor, 
                                traj_real: torch.Tensor) -> float:
        """计算分布差异"""
        # 提取速度分布
        _, speeds_sim = compute_velocities_from_traj(traj_sim, dt=0.1)
        _, speeds_real = compute_velocities_from_traj(traj_real, dt=0.1)
        
        # 使用KL散度衡量分布差异
        speeds_sim_flat = speeds_sim.flatten().detach().cpu().numpy()
        speeds_real_flat = speeds_real.flatten().detach().cpu().numpy()
        
        # 计算直方图
        bins = np.linspace(0, 30, 30)
        hist_sim, _ = np.histogram(speeds_sim_flat, bins=bins, density=True)
        hist_real, _ = np.histogram(speeds_real_flat, bins=bins, density=True)
        
        # 添加小值避免零值
        hist_sim = hist_sim + 1e-8
        hist_real = hist_real + 1e-8
        
        # 计算KL散度
        kl_divergence = stats.entropy(hist_sim, hist_real)
        
        # 转换为0-1分数（KL散度越小越好）
        distribution_gap = min(1.0, kl_divergence / 5.0)
        return distribution_gap
    
    def _compute_motion_feature_gap(self, traj_sim: torch.Tensor, 
                                  traj_real: torch.Tensor) -> float:
        """计算运动特征差异"""
        # 计算各种运动特征的统计差异
        
        # 1. 速度特征
        _, speeds_sim = compute_velocities_from_traj(traj_sim, dt=0.1)
        _, speeds_real = compute_velocities_from_traj(traj_real, dt=0.1)
        
        speed_mean_gap = abs(speeds_sim.mean() - speeds_real.mean()).item()
        speed_std_gap = abs(speeds_sim.std() - speeds_real.std()).item()
        
        # 2. 加速度特征  
        if speeds_sim.shape[1] > 1:
            accel_sim = (speeds_sim[:, 1:] - speeds_sim[:, :-1]) / 0.1
            accel_real = (speeds_real[:, 1:] - speeds_real[:, :-1]) / 0.1
            
            accel_mean_gap = abs(accel_sim.mean() - accel_real.mean()).item()
            accel_std_gap = abs(accel_sim.std() - accel_real.std()).item()
        else:
            accel_mean_gap = accel_std_gap = 0.0
        
        # 归一化并组合
        feature_gaps = [
            speed_mean_gap / 20.0,  # 归一化到0-1
            speed_std_gap / 10.0,
            accel_mean_gap / 5.0,
            accel_std_gap / 3.0
        ]
        
        motion_gap = np.mean(feature_gaps)
        return min(1.0, motion_gap)
    
    def _compute_frequency_domain_gap(self, traj_sim: torch.Tensor, 
                                    traj_real: torch.Tensor) -> float:
        """计算频域差异"""
        # 简化的频域分析（基于位置序列的频谱）
        
        # 提取位置序列
        pos_sim = traj_sim[:, :, :2].detach().cpu().numpy()
        pos_real = traj_real[:, :, :2].detach().cpu().numpy()
        
        # 计算平均功率谱密度
        def compute_psd(positions):
            psds = []
            for traj in positions:
                if len(traj) > 4:  # 确保有足够的点
                    x_fft = np.abs(np.fft.fft(traj[:, 0]))[:len(traj)//2]
                    y_fft = np.abs(np.fft.fft(traj[:, 1]))[:len(traj)//2]
                    psd = (x_fft + y_fft) / 2
                    psds.append(psd)
            return np.mean(psds, axis=0) if psds else np.array([0])
        
        psd_sim = compute_psd(pos_sim)
        psd_real = compute_psd(pos_real)
        
        # 计算频谱差异
        if len(psd_sim) > 0 and len(psd_real) > 0:
            min_len = min(len(psd_sim), len(psd_real))
            psd_sim = psd_sim[:min_len]
            psd_real = psd_real[:min_len]
            
            frequency_gap = np.mean(np.abs(psd_sim - psd_real)) / (np.mean(psd_real) + 1e-8)
        else:
            frequency_gap = 0.0
        
        return min(1.0, frequency_gap)
    
    def _compute_semantic_gap(self, traj_sim: torch.Tensor, 
                            traj_real: torch.Tensor) -> float:
        """计算语义差异"""
        # 基于高层行为模式的差异
        
        # 1. 行为模式统计
        patterns_sim = self._extract_behavior_patterns(traj_sim)
        patterns_real = self._extract_behavior_patterns(traj_real)
        
        # 2. 计算模式分布差异
        pattern_types = set(patterns_sim.keys()) | set(patterns_real.keys())
        
        semantic_gap = 0.0
        for pattern in pattern_types:
            freq_sim = patterns_sim.get(pattern, 0) / len(traj_sim)
            freq_real = patterns_real.get(pattern, 0) / len(traj_real)
            semantic_gap += abs(freq_sim - freq_real)
        
        semantic_gap /= len(pattern_types) if pattern_types else 1
        return semantic_gap
    
    def _extract_behavior_patterns(self, trajectories: torch.Tensor) -> Dict[str, int]:
        """提取行为模式"""
        patterns = defaultdict(int)
        
        velocities, speeds = compute_velocities_from_traj(trajectories, dt=0.1)
        NA, T = speeds.shape
        
        for i in range(NA):
            traj_speeds = speeds[i]
            
            # 识别不同的行为模式
            if T > 1:
                speed_changes = traj_speeds[1:] - traj_speeds[:-1]
                
                # 加速行为
                if torch.any(speed_changes > 2.0):
                    patterns['acceleration'] += 1
                
                # 减速行为
                if torch.any(speed_changes < -2.0):
                    patterns['deceleration'] += 1
                
                # 匀速行为
                if torch.all(torch.abs(speed_changes) < 0.5):
                    patterns['constant_speed'] += 1
                
                # 高速行为
                if torch.any(traj_speeds > 15.0):
                    patterns['high_speed'] += 1
                
                # 低速行为
                if torch.any(traj_speeds < 3.0):
                    patterns['low_speed'] += 1
        
        return dict(patterns)
    
    # ===== 辅助函数 =====
    def _compute_coverage_statistics(self, traj_sim: torch.Tensor, 
                                   traj_real: torch.Tensor) -> float:
        """计算覆盖率统计"""
        # 简化的覆盖率计算
        # 基于速度范围的覆盖
        
        _, speeds_sim = compute_velocities_from_traj(traj_sim, dt=0.1)
        _, speeds_real = compute_velocities_from_traj(traj_real, dt=0.1)
        
        speed_range_real = (speeds_real.min().item(), speeds_real.max().item())
        speed_range_sim = (speeds_sim.min().item(), speeds_sim.max().item())
        
        # 计算覆盖重叠
        overlap_min = max(speed_range_real[0], speed_range_sim[0])
        overlap_max = min(speed_range_real[1], speed_range_sim[1])
        
        if overlap_max > overlap_min:
            overlap = overlap_max - overlap_min
            real_range = speed_range_real[1] - speed_range_real[0]
            coverage = overlap / (real_range + 1e-8)
        else:
            coverage = 0.0
        
        return min(1.0, coverage)
    
    def _identify_interaction_patterns(self, trajectories: torch.Tensor) -> Dict[str, int]:
        """识别交互模式"""
        patterns = defaultdict(int)
        NA, T, _ = trajectories.shape
        
        if NA < 2:
            return dict(patterns)
        
        # 计算车辆间距离和相对速度
        distances = self._compute_inter_vehicle_distances(trajectories)
        relative_vels = self._compute_relative_velocities(trajectories)
        
        # 识别跟驰行为
        for i in range(NA):
            for j in range(i+1, NA):
                min_dist = distances[i, j].min().item()
                avg_dist = distances[i, j].mean().item()
                
                if 2.0 < avg_dist < 20.0 and min_dist > 1.0:
                    patterns['following'] += 1
                
                # 识别超车行为
                rel_vel_norm = torch.norm(relative_vels[i, j], dim=-1)
                if torch.any(rel_vel_norm > 5.0) and avg_dist < 30.0:
                    patterns['overtaking'] += 1
        
        return dict(patterns)
    
    def _evaluate_pattern_realism(self, patterns: Dict[str, int]) -> float:
        """评估模式真实性"""
        # 基于预定义的合理模式分布
        expected_patterns = {
            'following': 0.3,
            'overtaking': 0.1,
            'lane_changing': 0.2,
            'merging': 0.1
        }
        
        total_patterns = sum(patterns.values())
        if total_patterns == 0:
            return 0.5  # 中性分数
        
        realism_score = 0.0
        for pattern, expected_freq in expected_patterns.items():
            actual_freq = patterns.get(pattern, 0) / total_patterns
            realism_score += 1.0 - abs(actual_freq - expected_freq)
        
        return realism_score / len(expected_patterns)
    
    def _check_social_compliance(self, trajectories: torch.Tensor) -> float:
        """检查社会性驾驶规范"""
        # 简化的社会性规范检查
        violations = 0
        total_checks = 0
        
        # 1. 安全距离检查
        distances = self._compute_inter_vehicle_distances(trajectories)
        unsafe_distances = (distances < 2.0) & (distances > 0)
        violations += unsafe_distances.sum().item()
        total_checks += (distances > 0).sum().item()
        
        # 2. 速度合理性检查
        velocities, speeds = compute_velocities_from_traj(trajectories, dt=0.1)
        excessive_speeds = speeds > 20.0  # 72 km/h
        violations += excessive_speeds.sum().item()
        total_checks += speeds.numel()
        
        compliance_score = 1.0 - (violations / total_checks) if total_checks > 0 else 1.0
        return max(0.0, compliance_score)
    
    def _evaluate_collision_realism(self, trajectories: torch.Tensor) -> float:
        """评估碰撞合理性"""
        # 检查碰撞是否以合理的方式发生
        NA, T, _ = trajectories.shape
        
        if NA < 2:
            return 1.0
        
        collision_score = 1.0
        
        # 检查是否有碰撞
        has_collision = False
        
        for i in range(NA):
            for j in range(i+1, NA):
                traj_i = trajectories[i:i+1]  # [1, T, 4] 
                traj_j = trajectories[j:j+1]  # [1, T, 4]
                
                # 简化的碰撞检测
                distances = torch.norm(traj_i[:, :, :2] - traj_j[:, :, :2], dim=-1)
                min_distance = distances.min().item()
                
                if min_distance < 1.5:  # 认为发生了碰撞
                    has_collision = True
                    
                    # 检查碰撞前的行为是否合理
                    min_dist_idx = distances.argmin().item()
                    if min_dist_idx > 0:
                        # 检查接近过程是否渐进
                        approach_distances = distances[0, max(0, min_dist_idx-3):min_dist_idx+1]
                        if len(approach_distances) > 1:
                            # 距离应该逐渐减小
                            is_gradual = torch.all(approach_distances[1:] <= approach_distances[:-1] + 0.5)
                            if not is_gradual:
                                collision_score *= 0.7  # 突然碰撞降低分数
        
        return collision_score
    
    def _compute_detailed_metrics(self, 
                                z_original: torch.Tensor,
                                z_adversarial: torch.Tensor,
                                traj_original: torch.Tensor,
                                traj_adversarial: torch.Tensor) -> Dict[str, Any]:
        """计算详细指标"""
        
        detailed = {}
        
        # Z分析
        detailed['z_analysis'] = {
            'original_mean': z_original.mean(dim=0).tolist(),
            'adversarial_mean': z_adversarial.mean(dim=0).tolist(),
            'z_dimension': z_original.shape[-1],
            'z_norm_original': torch.norm(z_original, dim=-1).mean().item(),
            'z_norm_adversarial': torch.norm(z_adversarial, dim=-1).mean().item()
        }
        
        # 轨迹分析
        velocities_orig, speeds_orig = compute_velocities_from_traj(traj_original, dt=0.1)
        velocities_adv, speeds_adv = compute_velocities_from_traj(traj_adversarial, dt=0.1)
        
        detailed['trajectory_analysis'] = {
            'speed_stats_original': {
                'mean': speeds_orig.mean().item(),
                'std': speeds_orig.std().item(),
                'max': speeds_orig.max().item(),
                'min': speeds_orig.min().item()
            },
            'speed_stats_adversarial': {
                'mean': speeds_adv.mean().item(), 
                'std': speeds_adv.std().item(),
                'max': speeds_adv.max().item(),
                'min': speeds_adv.min().item()
            }
        }
        
        return detailed
    
    def _print_evaluation_summary(self, results: EvaluationResults):
        """打印评估摘要"""
        Logger.log("=" * 60)
        Logger.log("对抗性轨迹评估结果摘要")
        Logger.log("=" * 60)
        
        Logger.log(f"1. 轨迹真实性评估:")
        for key, value in results.trajectory_realism.items():
            Logger.log(f"   {key}: {value:.4f}")
        
        Logger.log(f"\n2. 交互合理性评估:")
        for key, value in results.interaction_consistency.items():
            Logger.log(f"   {key}: {value:.4f}")
        
        Logger.log(f"\n3. 长尾事件覆盖率评估:")
        for key, value in results.longtail_coverage.items():
            if isinstance(value, list):
                Logger.log(f"   {key}: {value}")
            else:
                Logger.log(f"   {key}: {value:.4f}")
        
        Logger.log(f"\n4. Sim-to-Real差距评估:")
        for key, value in results.sim_to_real_gap.items():
            Logger.log(f"   {key}: {value:.4f}")
        
        # 计算总体评分
        overall_score = np.mean([
            results.trajectory_realism['overall_realism_score'],
            results.interaction_consistency['overall_interaction_score'],
            results.longtail_coverage['overall_longtail_score'],
            results.sim_to_real_gap['overall_sim_to_real_score']
        ])
        
        Logger.log(f"\n总体评估分数: {overall_score:.4f}")
        Logger.log("=" * 60)
    
    def save_evaluation_results(self, results: EvaluationResults, output_path: str):
        """保存评估结果"""
        output_data = {
            'trajectory_realism': results.trajectory_realism,
            'interaction_consistency': results.interaction_consistency,
            'longtail_coverage': results.longtail_coverage,
            'sim_to_real_gap': results.sim_to_real_gap,
            'detailed_metrics': results.detailed_metrics
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        Logger.log(f"评估结果已保存到: {output_path}")

    def analyze_longtail_events_8s(self, 
                                 trajectories: torch.Tensor, 
                                 vehicle_lengths: torch.Tensor = None,
                                 vehicle_widths: torch.Tensor = None,
                                 dt: float = 0.5) -> Dict[str, Any]:
        """
        分析生成后的轨迹，在8s的整体时间内检测长尾事件
        
        检测以下长尾事件：
        1. TTC小于0.5s的事件
        2. 两车之间的距离很小（<2.0m）
        3. 发生碰撞
        
        参数:
            trajectories: 车辆轨迹 [NA, T, 4] (x, y, heading, speed)
            vehicle_lengths: 车辆长度 [NA] (如果为None，使用默认值4.5m)
            vehicle_widths: 车辆宽度 [NA] (如果为None，使用默认值2.0m)
            dt: 时间步长，默认0.1s
            
        返回:
            Dict包含：
            - has_longtail_event: bool，是否发生长尾事件
            - longtail_event_types: List[str]，发生的长尾事件类型
            - event_details: Dict，详细的事件信息
            - vehicle_event_flags: torch.Tensor [NA]，每个车辆是否参与长尾事件
        """
        NA, T, _ = trajectories.shape
        device = trajectories.device
        
        # 默认车辆尺寸
        if vehicle_lengths is None:
            vehicle_lengths = torch.full((NA,), 4.5, device=device)
        if vehicle_widths is None:
            vehicle_widths = torch.full((NA,), 2.0, device=device)
        
        # 初始化结果
        longtail_event_types = []
        event_details = {
            'ttc_events': [],
            'distance_events': [],
            'collision_events': []
        }
        vehicle_event_flags = torch.zeros(NA, dtype=torch.bool, device=device)
        
        # 1. 检测TTC小于0.5s的事件
        ttc_threshold = 0.5  # 秒
        min_distance_threshold = 2.0  # 米
        
        for i in range(NA):
            for j in range(i+1, NA):
                # 计算车辆i和j之间的交互
                traj_i = trajectories[i]  # [T, 4]
                traj_j = trajectories[j]  # [T, 4]
                
                # 计算每个时间步的距离
                distances = torch.norm(traj_i[:, :2] - traj_j[:, :2], dim=1)  # [T]
                min_distance = torch.min(distances).item()
                
                # 检查最小距离事件
                if min_distance < min_distance_threshold:
                    longtail_event_types.append('small_distance')
                    event_details['distance_events'].append({
                        'vehicle_pair': (i, j),
                        'min_distance': min_distance,
                        'time_step': torch.argmin(distances).item()
                    })
                    vehicle_event_flags[i] = True
                    vehicle_event_flags[j] = True
                
                # 计算TTC
                for t in range(T-1):
                    # 当前位置和速度
                    pos_i = traj_i[t, :2]
                    pos_j = traj_j[t, :2]
                    vel_i = traj_i[t, 3] * torch.tensor([torch.cos(traj_i[t, 2]), 
                                                        torch.sin(traj_i[t, 2])], device=device)
                    vel_j = traj_j[t, 3] * torch.tensor([torch.cos(traj_j[t, 2]), 
                                                        torch.sin(traj_j[t, 2])], device=device)
                    
                    # 相对位置和速度
                    rel_pos = pos_j - pos_i
                    rel_vel = vel_j - vel_i
                    
                    # 计算TTC：只有当车辆在相互接近时才有意义
                    rel_speed_squared = torch.sum(rel_vel * rel_vel)
                    if rel_speed_squared > 1e-6:  # 避免除零
                        # 计算相对位置在相对速度方向上的投影
                        dot_product = torch.sum(rel_pos * rel_vel)
                        if dot_product < 0:  # 车辆在相互接近
                            distance = torch.norm(rel_pos)
                            rel_speed = torch.sqrt(rel_speed_squared)
                            ttc = distance / rel_speed
                            
                            if ttc < ttc_threshold:
                                longtail_event_types.append('low_ttc')
                                event_details['ttc_events'].append({
                                    'vehicle_pair': (i, j),
                                    'ttc': ttc.item(),
                                    'time_step': t,
                                    'distance': distance.item()
                                })
                                vehicle_event_flags[i] = True
                                vehicle_event_flags[j] = True
                
                # 3. 检测碰撞事件（使用简化的圆形碰撞检测）
                vehicle_radius_i = (vehicle_lengths[i] + vehicle_widths[i]) / 4.0  # 简化为圆形
                vehicle_radius_j = (vehicle_lengths[j] + vehicle_widths[j]) / 4.0
                collision_threshold = vehicle_radius_i + vehicle_radius_j
                
                collision_mask = distances < collision_threshold
                if torch.any(collision_mask):
                    longtail_event_types.append('collision')
                    collision_times = torch.where(collision_mask)[0]
                    event_details['collision_events'].append({
                        'vehicle_pair': (i, j),
                        'collision_times': collision_times.tolist(),
                        'min_distance': min_distance,
                        'collision_threshold': collision_threshold.item()
                    })
                    vehicle_event_flags[i] = True
                    vehicle_event_flags[j] = True
        
        # 去除重复的事件类型
        longtail_event_types = list(set(longtail_event_types))
        
        return {
            'has_longtail_event': len(longtail_event_types) > 0,
            'longtail_event_types': longtail_event_types,
            'event_details': event_details,
            'vehicle_event_flags': vehicle_event_flags,
            'num_vehicles_involved': torch.sum(vehicle_event_flags).item(),
            'total_events': len(event_details['ttc_events']) + 
                           len(event_details['distance_events']) + 
                           len(event_details['collision_events'])
        }

    def analyze_longtail_events_8s_enhanced(self,
                                          trajectories: torch.Tensor,
                                          vehicle_lengths: torch.Tensor = None,
                                          vehicle_widths: torch.Tensor = None,
                                          dt: float = 0.5) -> Dict[str, Any]:
        """
        增强版长尾事件分析函数，使用更精确的TTC计算
        
        使用现有的compute_longitudinal_metrics函数来计算更准确的TTC值
        
        参数:
            trajectories: 车辆轨迹 [NA, T, 4] (x, y, heading, speed)
            vehicle_lengths: 车辆长度 [NA]
            vehicle_widths: 车辆宽度 [NA]  
            dt: 时间步长
            
        返回:
            Dict包含长尾事件分析结果
        """
        NA, T, _ = trajectories.shape
        device = trajectories.device
        
        # 默认车辆尺寸
        if vehicle_lengths is None:
            vehicle_lengths = torch.full((NA,), 4.5, device=device)
        if vehicle_widths is None:
            vehicle_widths = torch.full((NA,), 2.0, device=device)
        
        # 初始化结果
        longtail_event_types = []
        event_details = {
            'ttc_events': [],
            'distance_events': [],
            'collision_events': []
        }
        vehicle_event_flags = torch.zeros(NA, dtype=torch.bool, device=device)
        
        # 阈值设置
        ttc_threshold = 0.5  # 秒
        min_distance_threshold = 2.0  # 米
        
        # 检查所有车辆对
        for i in range(NA):
            for j in range(i+1, NA):
                traj_i = trajectories[i:i+1]  # [1, T, 4]
                traj_j = trajectories[j:j+1]  # [1, T, 4]
                
                # 1. 使用精确的TTC计算
                try:
                    d_rel_long, v_rel_long, ttc = compute_longitudinal_metrics(
                        traj_i, traj_j, dt=dt, epsilon=1e-6
                    )
                    
                    # 检查返回的张量形状
                    if ttc.numel() == 0:
                        continue  # 跳过空的TTC结果
                    

                    # 检查TTC事件
                    valid_ttc_mask = ~torch.isinf(ttc) & (ttc > 0) & (ttc < ttc_threshold)
                    if torch.any(valid_ttc_mask):
                        valid_ttc_values = ttc[valid_ttc_mask]
                        if len(valid_ttc_values) > 0:
                            min_ttc = torch.min(valid_ttc_values).item()
                            # 找到最小TTC的索引，但要确保在有效范围内
                            min_ttc_indices = torch.where(ttc == torch.min(ttc[valid_ttc_mask]))[0]
                            if len(min_ttc_indices) > 0:
                                min_ttc_time = min_ttc_indices[0].item()
                                
                                # 确保索引不越界
                                if (min_ttc_time < d_rel_long.shape[0] and 
                                    min_ttc_time < v_rel_long.shape[0] and
                                    min_ttc_time >= 0):
                                    
                                    longtail_event_types.append('low_ttc')
                                    event_details['ttc_events'].append({
                                        'vehicle_pair': (i, j),
                                        'min_ttc': min_ttc,
                                        'time_step': min_ttc_time,
                                        'longitudinal_distance': d_rel_long[min_ttc_time].item(),
                                        'relative_velocity': v_rel_long[min_ttc_time].item()
                                    })
                                    vehicle_event_flags[i] = True
                                    vehicle_event_flags[j] = True
                        
                except Exception as e:
                    Logger.log(f"TTC计算失败 (车辆 {i}, {j}): {e}")
                
                # 2. 距离检查
                distances = torch.norm(traj_i[0, :, :2] - traj_j[0, :, :2], dim=1)  # [T]
                min_distance = torch.min(distances).item()
                
                if min_distance < min_distance_threshold:
                    longtail_event_types.append('small_distance')
                    event_details['distance_events'].append({
                        'vehicle_pair': (i, j),
                        'min_distance': min_distance,
                        'time_step': torch.argmin(distances).item()
                    })
                    vehicle_event_flags[i] = True
                    vehicle_event_flags[j] = True
                
                # 3. 碰撞检测（使用现有的碰撞检测函数）
                try:
                    lw_i = torch.stack([vehicle_lengths[i], vehicle_widths[i]])
                    lw_j = torch.stack([vehicle_lengths[j], vehicle_widths[j]])
                    
                    veh_coll, coll_time = check_single_veh_coll(
                        traj_i[0], lw_i, traj_j, lw_j.unsqueeze(0)
                    )
                    
                    # check_single_veh_coll 返回 numpy 数组，需要转换
                    if isinstance(veh_coll, np.ndarray):
                        veh_coll = torch.from_numpy(veh_coll)
                    if isinstance(coll_time, np.ndarray):
                        coll_time = torch.from_numpy(coll_time)
                    
                    # 确保返回的数据有效
                    if veh_coll.numel() > 0 and torch.any(veh_coll):
                        longtail_event_types.append('collision')
                        # 安全地提取碰撞时间
                        if coll_time.numel() > 0:
                            collision_time_value = coll_time.flatten()[0].item()
                        else:
                            collision_time_value = 0
                            
                        event_details['collision_events'].append({
                            'vehicle_pair': (i, j),
                            'collision_time': collision_time_value,
                            'min_distance': min_distance
                        })
                        vehicle_event_flags[i] = True
                        vehicle_event_flags[j] = True
                        
                except Exception as e:
                    Logger.log(f"碰撞检测失败 (车辆 {i}, {j}): {e}")
        
        # 去除重复的事件类型
        longtail_event_types = list(set(longtail_event_types))
        
        return {
            'has_longtail_event': len(longtail_event_types) > 0,
            'longtail_event_types': longtail_event_types,
            'event_details': event_details,
            'vehicle_event_flags': vehicle_event_flags,
            'num_vehicles_involved': torch.sum(vehicle_event_flags).item(),
            'total_events': len(event_details['ttc_events']) + 
                           len(event_details['distance_events']) + 
                           len(event_details['collision_events']),
            'analysis_summary': {
                'ttc_events_count': len(event_details['ttc_events']),
                'distance_events_count': len(event_details['distance_events']),
                'collision_events_count': len(event_details['collision_events']),
                'vehicles_with_events_ratio': torch.sum(vehicle_event_flags).item() / NA
            }
        }


# ===== 示例使用函数 =====
def example_evaluation():
    """示例评估函数"""
    
    # 创建示例数据
    NA, T, z_dim = 6, 25, 64
    
    # 模拟潜变量和轨迹数据
    z_original = torch.randn(NA, z_dim)
    z_adversarial = z_original + 0.1 * torch.randn(NA, z_dim)  # 添加对抗性扰动
    
    traj_original = torch.randn(NA, T, 4)
    traj_adversarial = traj_original + 0.05 * torch.randn(NA, T, 4)  # 添加轨迹扰动
    
    # 使用基于真实数据集结构的数据
    traj_real_dataset = torch.zeros(100, T, 4)  # 基于真实数据集结构
    
    # 创建评估器
    evaluator = AdversarialTrajectoryEvaluator(device='cpu', verbose=True)
    
    # 执行评估（需要模拟scene_graph和model）
    class MockSceneGraph:
        def __init__(self):
            self.ptr = torch.tensor([0, 2, 4, 6])  # 模拟batch指针
    
    class MockModel:
        pass
    
    scene_graph = MockSceneGraph()
    model = MockModel()
    
    # 运行评估
    results = evaluator.evaluate_comprehensive(
        z_original=z_original,
        z_adversarial=z_adversarial,
        traj_original=traj_original,
        traj_adversarial=traj_adversarial,
        traj_real_dataset=traj_real_dataset,
        scene_graph=scene_graph,
        model=model
    )
    
    # 保存结果
    evaluator.save_evaluation_results(results, "evaluation_results.json")
    
    return results


def example_longtail_analysis():
    """
    长尾事件分析函数的使用示例
    """
    print("=" * 50)
    print("长尾事件分析函数使用示例")
    print("=" * 50)
    
    # 创建评估器
    evaluator = AdversarialTrajectoryEvaluator(device='cpu', verbose=True)
    
    # 创建示例轨迹数据 (2辆车，80个时间步，对应8秒)
    NA, T = 2, 80  # 2辆车，80个时间步 (8s @ 0.1s/step)
    device = 'cpu'
    
    # 车辆1：直线行驶
    traj1 = torch.zeros(T, 4)
    traj1[:, 0] = torch.linspace(0, 40, T)  # x坐标从0到40m
    traj1[:, 1] = 0  # y坐标保持0
    traj1[:, 2] = 0  # 朝向角0度 (向右)
    traj1[:, 3] = 5.0  # 速度5m/s
    
    # 车辆2：从侧面接近，会发生近距离接触
    traj2 = torch.zeros(T, 4)
    traj2[:, 0] = torch.linspace(20, 20, T)  # x坐标保持20m
    traj2[:, 1] = torch.linspace(10, -1, T)  # y坐标从10m到-1m，会接近车辆1
    traj2[:, 2] = -np.pi/2  # 朝向角-90度 (向下)
    traj2[:, 3] = 2.0  # 速度2m/s
    
    # 组合轨迹
    trajectories = torch.stack([traj1, traj2])  # [2, 80, 4]
    
    print(f"轨迹数据形状: {trajectories.shape}")
    print(f"时间长度: {T * 0.1:.1f}秒")
    print(f"车辆数量: {NA}")
    
    # 基础版本分析
    print("\n--- 基础版长尾事件分析 ---")
    basic_results = evaluator.analyze_longtail_events_8s(trajectories)
    
    print(f"是否发生长尾事件: {basic_results['has_longtail_event']}")
    print(f"长尾事件类型: {basic_results['longtail_event_types']}")
    print(f"参与事件的车辆数量: {basic_results['num_vehicles_involved']}")
    print(f"总事件数: {basic_results['total_events']}")
    
    if basic_results['event_details']['distance_events']:
        print("距离事件详情:")
        for event in basic_results['event_details']['distance_events']:
            print(f"  车辆对 {event['vehicle_pair']}: 最小距离 {event['min_distance']:.2f}m")
    
    # 增强版本分析
    print("\n--- 增强版长尾事件分析 ---")
    enhanced_results = evaluator.analyze_longtail_events_8s_enhanced(trajectories)
    
    print(f"是否发生长尾事件: {enhanced_results['has_longtail_event']}")
    print(f"长尾事件类型: {enhanced_results['longtail_event_types']}")
    print(f"分析摘要:")
    for key, value in enhanced_results['analysis_summary'].items():
        print(f"  {key}: {value}")
    
    if enhanced_results['event_details']['ttc_events']:
        print("TTC事件详情:")
        for event in enhanced_results['event_details']['ttc_events']:
            print(f"  车辆对 {event['vehicle_pair']}: 最小TTC {event['min_ttc']:.3f}s")
    
    print("\n长尾事件分析完成!")
    return basic_results, enhanced_results


if __name__ == "__main__":
    print("对抗性轨迹评估模块")
    print("运行示例评估...")
    
    # 运行原有的示例评估
    results = example_evaluation()
    
    print("\n" + "="*60)
    
    # 运行长尾事件分析示例
    basic_results, enhanced_results = example_longtail_analysis()
    print("评估完成！") 