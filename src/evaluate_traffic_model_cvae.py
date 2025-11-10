#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STRIVE CVAE 质量评估脚本 - 直接使用 traffic_model.pth
使用 traffic_model.pth 在 nuScenes 数据集上生成预测轨迹，并与真实未来轨迹对比
验证两个关键指标：
1. 轨迹重建 MSE < 0.05 m²
2. 加速度和曲率分布 KL 散度 < 0.5

同时计算标准轨迹评估指标：
- ADE (Average Displacement Error): 平均位移误差
- FDE (Final Displacement Error): 最终位移误差
- MR@2m (Miss Rate): 未命中率
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from torch_geometric.data import DataLoader as GraphDataLoader
import time
from typing import Optional, Dict, List, Tuple

# 添加项目路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

from models.traffic_model import TrafficModel
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from utils.torch import get_device, load_state, count_params
from utils.common import dict2obj
from utils.logger import Logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import track, Progress
from rich.table import Table

console = Console()


class CVAEModelEvaluator:
    """直接评估 CVAE 模型在数据集上的重建质量"""
    
    def __init__(self, model, test_dataset, map_env, device, 
                 max_mse_threshold=0.05, max_kl_divergence=0.5, 
                 filter_outliers=True, outlier_iqr_factor=1.5):
        self.model = model
        self.test_dataset = test_dataset
        self.map_env = map_env
        self.device = device
        self.max_mse_threshold = max_mse_threshold
        self.max_kl_divergence = max_kl_divergence
        self.filter_outliers = filter_outliers
        self.outlier_iqr_factor = outlier_iqr_factor
        
        self.model.eval()
    
    def remove_outliers_iqr(self, data, factor=1.5):
        """使用 IQR 方法移除异常值"""
        if len(data) < 4:  # 数据太少无法计算四分位数
            return data, []
        
        data_array = np.array(data)
        Q1 = np.percentile(data_array, 25)
        Q3 = np.percentile(data_array, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # 标记异常值
        outlier_mask = (data_array < lower_bound) | (data_array > upper_bound)
        outliers = data_array[outlier_mask]
        filtered_data = data_array[~outlier_mask]
        
        return filtered_data.tolist(), outliers.tolist()
        
    def calculate_trajectory_mse(self, pred_traj, gt_traj, valid_mask=None):
        """计算轨迹均方误差，只计算有效点"""
        # pred_traj: [NA, FT, 4+]
        # gt_traj: [NA, FT, 6]
        # valid_mask: [NA, FT] 可见性掩码
        
        # 只比较位置 (x, y)
        pred_pos = pred_traj[:, :, :2]  # [NA, FT, 2]
        gt_pos = gt_traj[:, :, :2]      # [NA, FT, 2]
        
        # 如果有 valid_mask，只计算可见点
        if valid_mask is not None:
            # 扩展 mask 到位置维度 [NA, FT, 2]
            mask_expanded = valid_mask.unsqueeze(-1).expand_as(pred_pos)
            pred_pos_valid = pred_pos[mask_expanded].view(-1, 2)
            gt_pos_valid = gt_pos[mask_expanded].view(-1, 2)
            
            if pred_pos_valid.numel() == 0:
                return float('nan')
            
            # 计算欧氏距离平方
            mse = torch.mean((pred_pos_valid - gt_pos_valid) ** 2).item()
        else:
            # 过滤掉 NaN 值
            valid = ~torch.isnan(gt_pos).any(dim=-1)  # [NA, FT]
            if valid.sum() == 0:
                return float('nan')
            
            mask_expanded = valid.unsqueeze(-1).expand_as(pred_pos)
            pred_pos_valid = pred_pos[mask_expanded].view(-1, 2)
            gt_pos_valid = gt_pos[mask_expanded].view(-1, 2)
            
            mse = torch.mean((pred_pos_valid - gt_pos_valid) ** 2).item()
        
        return mse
    
    def calculate_ade(self, pred_traj: torch.Tensor, gt_traj: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> float:
        """计算平均位移误差 (Average Displacement Error, ADE)"""
        # pred_traj: [NA, FT, 4+]
        # gt_traj: [NA, FT, 6]
        # valid_mask: [NA, FT] 可见性掩码
        
        # 只比较位置 (x, y)
        pred_pos = pred_traj[:, :, :2]  # [NA, FT, 2]
        gt_pos = gt_traj[:, :, :2]      # [NA, FT, 2]
        
        # 计算欧氏距离
        displacement = torch.norm(pred_pos - gt_pos, dim=-1)  # [NA, FT]
        
        # 应用有效掩码
        if valid_mask is not None:
            displacement = displacement * valid_mask
            num_valid = valid_mask.sum()
            if num_valid == 0:
                return float('nan')
        else:
            # 过滤 NaN 值
            valid = ~torch.isnan(displacement)
            displacement = displacement[valid]
            num_valid = valid.sum()
            if num_valid == 0:
                return float('nan')
        
        # 计算平均值
        ade = displacement.sum() / num_valid
        return ade.item()
    
    def calculate_fde(self, pred_traj: torch.Tensor, gt_traj: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> float:
        """计算最终位移误差 (Final Displacement Error, FDE)"""
        # pred_traj: [NA, FT, 4+]
        # gt_traj: [NA, FT, 6]
        # valid_mask: [NA, FT] 可见性掩码
        
        # 只比较位置 (x, y)
        pred_pos = pred_traj[:, :, :2]  # [NA, FT, 2]
        gt_pos = gt_traj[:, :, :2]      # [NA, FT, 2]
        
        batch_size = pred_pos.size(0)
        fde_list = []
        
        for i in range(batch_size):
            # 找到每个轨迹的最后一个有效时间步
            if valid_mask is not None:
                valid_indices = torch.where(valid_mask[i])[0]
                if len(valid_indices) == 0:
                    continue
                last_idx = valid_indices[-1]
            else:
                # 找到非 NaN 的最后一个点
                valid_indices = torch.where(~torch.isnan(gt_pos[i, :, 0]))[0]
                if len(valid_indices) == 0:
                    continue
                last_idx = valid_indices[-1]
            
            # 计算最后一个点的欧氏距离
            final_displacement = torch.norm(pred_pos[i, last_idx] - gt_pos[i, last_idx])
            fde_list.append(final_displacement.item())
        
        if len(fde_list) == 0:
            return float('nan')
        
        return np.mean(fde_list)
    
    def calculate_miss_rate(self, pred_traj: torch.Tensor, gt_traj: torch.Tensor, 
                           miss_threshold: float = 2.0, valid_mask: Optional[torch.Tensor] = None) -> float:
        """计算未命中率 (Miss Rate)"""
        # pred_traj: [NA, FT, 4+]
        # gt_traj: [NA, FT, 6]
        # valid_mask: [NA, FT] 可见性掩码
        
        # 只比较位置 (x, y)
        pred_pos = pred_traj[:, :, :2]  # [NA, FT, 2]
        gt_pos = gt_traj[:, :, :2]      # [NA, FT, 2]
        
        batch_size = pred_pos.size(0)
        miss_count = 0
        
        for i in range(batch_size):
            # 计算每个时间步的位移
            displacement = torch.norm(pred_pos[i] - gt_pos[i], dim=-1)
            
            # 应用有效掩码
            if valid_mask is not None:
                displacement = displacement[valid_mask[i]]
            else:
                valid = ~torch.isnan(displacement)
                displacement = displacement[valid]
            
            if len(displacement) == 0:
                continue
            
            # 检查是否有任何时间步超过阈值
            if (displacement > miss_threshold).any():
                miss_count += 1
        
        return miss_count / batch_size if batch_size > 0 else 0.0
    
    def calculate_acceleration(self, trajectory, dt=0.5):
        """计算加速度序列，过滤 NaN 值"""
        # trajectory: [NA, FT, 4+]
        positions = trajectory[:, :, :2]  # [NA, FT, 2]
        
        # 过滤掉包含 NaN 的轨迹
        valid_mask = ~torch.isnan(positions).any(dim=-1)  # [NA, FT]
        
        accelerations_list = []
        for agent_idx in range(positions.size(0)):
            agent_pos = positions[agent_idx][valid_mask[agent_idx]]  # [T_valid, 2]
            
            if len(agent_pos) < 3:  # 需要至少3个点才能计算加速度
                continue
            
            # 计算速度
            velocities = torch.diff(agent_pos, dim=0) / dt  # [T-1, 2]
            
            # 计算加速度
            accelerations = torch.diff(velocities, dim=0) / dt  # [T-2, 2]
            
            # 加速度大小
            acc_magnitude = torch.norm(accelerations, dim=-1)  # [T-2]
            accelerations_list.append(acc_magnitude.cpu().numpy())
        
        if len(accelerations_list) == 0:
            return np.array([])
        
        return np.concatenate(accelerations_list)
    
    def calculate_curvature(self, trajectory, dt=0.5):
        """计算曲率序列，过滤 NaN 值"""
        # trajectory: [NA, FT, 4+]
        positions = trajectory[:, :, :2].cpu().numpy()  # [NA, FT, 2]
        
        curvatures = []
        for agent_traj in positions:
            # 过滤掉 NaN 值
            valid_idx = ~np.isnan(agent_traj).any(axis=1)
            agent_traj_valid = agent_traj[valid_idx]
            
            # 对每个智能体的轨迹计算曲率
            if len(agent_traj_valid) < 3:
                continue
                
            dx = np.gradient(agent_traj_valid[:, 0])
            dy = np.gradient(agent_traj_valid[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            
            numerator = np.abs(dx * ddy - dy * ddx)
            denominator = np.power(dx**2 + dy**2, 1.5)
            denominator = np.where(denominator < 1e-8, 1e-8, denominator)
            
            curvature = numerator / denominator
            
            # 过滤掉无效曲率值
            valid_curvature = curvature[1:-1]  # 去除边界点
            valid_curvature = valid_curvature[~np.isnan(valid_curvature)]
            valid_curvature = valid_curvature[~np.isinf(valid_curvature)]
            
            curvatures.extend(valid_curvature)
        
        return np.array(curvatures)
    
    def calculate_kl_divergence(self, dist1, dist2, bins=50):
        """计算 KL 散度，处理 NaN 和 Inf"""
        # 过滤掉无效值
        dist1 = dist1[~np.isnan(dist1)]
        dist1 = dist1[~np.isinf(dist1)]
        dist2 = dist2[~np.isnan(dist2)]
        dist2 = dist2[~np.isinf(dist2)]
        
        if len(dist1) == 0 or len(dist2) == 0:
            return float('nan')
        
        if len(dist1) < 10 or len(dist2) < 10:  # 样本太少
            return float('nan')
        
        min_val = min(np.min(dist1), np.min(dist2))
        max_val = max(np.max(dist1), np.max(dist2))
        
        if max_val <= min_val or np.isnan(min_val) or np.isnan(max_val):
            return float('nan')
        
        hist1, _ = np.histogram(dist1, bins=bins, range=(min_val, max_val), density=True)
        hist2, _ = np.histogram(dist2, bins=bins, range=(min_val, max_val), density=True)
        
        # 归一化
        sum1 = np.sum(hist1)
        sum2 = np.sum(hist2)
        
        if sum1 <= 1e-10 or sum2 <= 1e-10:
            return float('nan')
        
        hist1 = hist1 / sum1
        hist2 = hist2 / sum2
        
        # 添加平滑项避免 log(0)
        epsilon = 1e-10
        hist1 = hist1 + epsilon
        hist2 = hist2 + epsilon
        
        # 重新归一化
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        # 计算 KL 散度
        kl_div = np.sum(hist1 * np.log(hist1 / hist2))
        
        # 检查结果是否有效
        if np.isnan(kl_div) or np.isinf(kl_div):
            return float('nan')
        
        return kl_div
    
    def evaluate_batch(self, scene_graph, map_idx):
        """评估一个批次"""
        with torch.no_grad():
            # 使用后验均值进行重建（这是CVAE的标准评估方式）
            pred = self.model(scene_graph, map_idx, self.map_env, use_post_mean=True)
            future_pred = pred['future_pred']  # [NA, FT, 4]
            future_gt = scene_graph.future_gt  # [NA, FT, 6]
            future_vis = scene_graph.future_vis  # [NA, FT]
            
            # 反归一化
            normalizer = self.model.get_normalizer()
            future_pred_unnorm = normalizer.unnormalize(future_pred)
            future_gt_unnorm = normalizer.unnormalize(future_gt)
            
            # 只评估可见的时间步
            valid_mask = future_vis == 1.0
            if valid_mask.sum() == 0:
                return None
            
            # 计算 MSE（使用 valid_mask）
            mse = self.calculate_trajectory_mse(future_pred_unnorm, future_gt_unnorm, valid_mask)
            
            # 计算标准轨迹评估指标
            ade = self.calculate_ade(future_pred_unnorm, future_gt_unnorm, valid_mask)
            fde = self.calculate_fde(future_pred_unnorm, future_gt_unnorm, valid_mask)
            mr_2m = self.calculate_miss_rate(future_pred_unnorm, future_gt_unnorm, miss_threshold=2.0, valid_mask=valid_mask)
            
            # 计算加速度
            pred_acc = self.calculate_acceleration(future_pred_unnorm)
            gt_acc = self.calculate_acceleration(future_gt_unnorm)
            
            # 计算曲率
            pred_curv = self.calculate_curvature(future_pred_unnorm)
            gt_curv = self.calculate_curvature(future_gt_unnorm)
            
            # 计算 KL 散度
            acc_kl = self.calculate_kl_divergence(pred_acc, gt_acc) if len(pred_acc) > 0 and len(gt_acc) > 0 else float('nan')
            curv_kl = self.calculate_kl_divergence(pred_curv, gt_curv) if len(pred_curv) > 0 and len(gt_curv) > 0 else float('nan')
            
            # 检查是否有有效指标
            if np.isnan(mse) and np.isnan(acc_kl) and np.isnan(curv_kl) and np.isnan(ade):
                return None
            
            return {
                'mse': mse,
                'ade': ade,
                'fde': fde,
                'mr_2m': mr_2m,
                'acc_kl': acc_kl,
                'curv_kl': curv_kl,
                'num_agents': future_pred.size(0),
                'num_timesteps': future_pred.size(1)
            }
    
    def evaluate_dataset(self, num_samples=None):
        """在整个数据集上评估"""
        # 创建数据加载器
        test_loader = GraphDataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        all_metrics = {
            'mse': [],
            'ade': [],
            'fde': [],
            'mr_2m': [],
            'acc_kl': [],
            'curv_kl': []
        }
        
        # 存储每个场景的详细结果
        scene_details = []
        
        total_scenes = num_samples if num_samples else len(test_loader)
        console.print(f"\n[cyan]开始评估 CVAE 模型在 nuScenes 数据集上的重建质量...[/cyan]")
        console.print(f"[dim]评估场景数: {total_scenes}[/dim]\n")
        
        with Progress() as progress:
            task = progress.add_task("[cyan]评估中...", total=total_scenes)
            
            for i, data in enumerate(test_loader):
                if num_samples and i >= num_samples:
                    break
                
                scene_graph, map_idx = data
                scene_graph = scene_graph.to(self.device)
                map_idx = map_idx.to(self.device)
                
                try:
                    batch_result = self.evaluate_batch(scene_graph, map_idx)
                    
                    if batch_result is not None:
                        # 只添加有效的指标
                        if not np.isnan(batch_result['mse']):
                            all_metrics['mse'].append(batch_result['mse'])
                        if not np.isnan(batch_result['ade']):
                            all_metrics['ade'].append(batch_result['ade'])
                        if not np.isnan(batch_result['fde']):
                            all_metrics['fde'].append(batch_result['fde'])
                        if not np.isnan(batch_result['mr_2m']):
                            all_metrics['mr_2m'].append(batch_result['mr_2m'])
                        if not np.isnan(batch_result['acc_kl']):
                            all_metrics['acc_kl'].append(batch_result['acc_kl'])
                        if not np.isnan(batch_result['curv_kl']):
                            all_metrics['curv_kl'].append(batch_result['curv_kl'])
                        
                        # 记录详细信息
                        scene_details.append({
                            'scene_id': i,
                            'num_agents': batch_result['num_agents'],
                            'num_timesteps': batch_result['num_timesteps'],
                            'mse': batch_result['mse'] if not np.isnan(batch_result['mse']) else None,
                            'ade': batch_result['ade'] if not np.isnan(batch_result['ade']) else None,
                            'fde': batch_result['fde'] if not np.isnan(batch_result['fde']) else None,
                            'mr_2m': batch_result['mr_2m'] if not np.isnan(batch_result['mr_2m']) else None,
                            'acc_kl': batch_result['acc_kl'] if not np.isnan(batch_result['acc_kl']) else None,
                            'curv_kl': batch_result['curv_kl'] if not np.isnan(batch_result['curv_kl']) else None,
                            'mse_pass': batch_result['mse'] <= self.max_mse_threshold if not np.isnan(batch_result['mse']) else False,
                            'kl_pass': (batch_result['acc_kl'] <= self.max_kl_divergence and 
                                       batch_result['curv_kl'] <= self.max_kl_divergence) if (
                                not np.isnan(batch_result['acc_kl']) and not np.isnan(batch_result['curv_kl'])) else False
                        })
                        
                except Exception as e:
                    console.print(f"[yellow]场景 {i} 评估失败: {e}[/yellow]")
                    scene_details.append({
                        'scene_id': i,
                        'num_agents': 0,
                        'num_timesteps': 0,
                        'mse': None,
                        'acc_kl': None,
                        'curv_kl': None,
                        'mse_pass': False,
                        'kl_pass': False,
                        'error': str(e)
                    })
                    continue
                
                progress.update(task, advance=1)
        
        # 计算统计指标
        if len(all_metrics['mse']) == 0:
            console.print("[red]错误: 没有有效的评估结果[/red]")
            return {
                'avg_mse': float('nan'),
                'std_mse': float('nan'),
                'avg_acc_kl': float('nan'),
                'std_acc_kl': float('nan'),
                'avg_curv_kl': float('nan'),
                'std_curv_kl': float('nan'),
                'mse_pass_rate': 0.0,
                'kl_pass_rate': 0.0,
                'total_scenes': 0
            }
        
        # 过滤异常值
        filtered_metrics = {
            'mse': all_metrics['mse'].copy(),
            'ade': all_metrics['ade'].copy(),
            'fde': all_metrics['fde'].copy(),
            'mr_2m': all_metrics['mr_2m'].copy(),
            'acc_kl': all_metrics['acc_kl'].copy(),
            'curv_kl': all_metrics['curv_kl'].copy()
        }
        
        outlier_info = {}
        
        if self.filter_outliers:
            console.print(f"\n[cyan]检测并过滤异常值 (IQR 因子: {self.outlier_iqr_factor})...[/cyan]")
            
            # 过滤 MSE 异常值
            if len(all_metrics['mse']) > 0:
                filtered_mse, mse_outliers = self.remove_outliers_iqr(
                    all_metrics['mse'], self.outlier_iqr_factor
                )
                filtered_metrics['mse'] = filtered_mse
                outlier_info['mse_outliers'] = mse_outliers
                outlier_info['mse_outlier_count'] = len(mse_outliers)
                console.print(f"  MSE: 移除 {len(mse_outliers)} 个异常值 "
                            f"({len(mse_outliers)}/{len(all_metrics['mse'])} = "
                            f"{100*len(mse_outliers)/len(all_metrics['mse']):.1f}%)")
            
            # 过滤 ADE 异常值
            if len(all_metrics['ade']) > 0:
                filtered_ade, ade_outliers = self.remove_outliers_iqr(
                    all_metrics['ade'], self.outlier_iqr_factor
                )
                filtered_metrics['ade'] = filtered_ade
                outlier_info['ade_outliers'] = ade_outliers
                outlier_info['ade_outlier_count'] = len(ade_outliers)
                console.print(f"  ADE: 移除 {len(ade_outliers)} 个异常值 "
                            f"({len(ade_outliers)}/{len(all_metrics['ade'])} = "
                            f"{100*len(ade_outliers)/len(all_metrics['ade']):.1f}%)")
            
            # 过滤 FDE 异常值
            if len(all_metrics['fde']) > 0:
                filtered_fde, fde_outliers = self.remove_outliers_iqr(
                    all_metrics['fde'], self.outlier_iqr_factor
                )
                filtered_metrics['fde'] = filtered_fde
                outlier_info['fde_outliers'] = fde_outliers
                outlier_info['fde_outlier_count'] = len(fde_outliers)
                console.print(f"  FDE: 移除 {len(fde_outliers)} 个异常值 "
                            f"({len(fde_outliers)}/{len(all_metrics['fde'])} = "
                            f"{100*len(fde_outliers)/len(all_metrics['fde']):.1f}%)")
            
            # 过滤加速度 KL 异常值
            if len(all_metrics['acc_kl']) > 0:
                filtered_acc_kl, acc_kl_outliers = self.remove_outliers_iqr(
                    all_metrics['acc_kl'], self.outlier_iqr_factor
                )
                filtered_metrics['acc_kl'] = filtered_acc_kl
                outlier_info['acc_kl_outliers'] = acc_kl_outliers
                outlier_info['acc_kl_outlier_count'] = len(acc_kl_outliers)
                console.print(f"  加速度 KL: 移除 {len(acc_kl_outliers)} 个异常值 "
                            f"({len(acc_kl_outliers)}/{len(all_metrics['acc_kl'])} = "
                            f"{100*len(acc_kl_outliers)/len(all_metrics['acc_kl']):.1f}%)")
            
            # 过滤曲率 KL 异常值
            if len(all_metrics['curv_kl']) > 0:
                filtered_curv_kl, curv_kl_outliers = self.remove_outliers_iqr(
                    all_metrics['curv_kl'], self.outlier_iqr_factor
                )
                filtered_metrics['curv_kl'] = filtered_curv_kl
                outlier_info['curv_kl_outliers'] = curv_kl_outliers
                outlier_info['curv_kl_outlier_count'] = len(curv_kl_outliers)
                console.print(f"  曲率 KL: 移除 {len(curv_kl_outliers)} 个异常值 "
                            f"({len(curv_kl_outliers)}/{len(all_metrics['curv_kl'])} = "
                            f"{100*len(curv_kl_outliers)/len(all_metrics['curv_kl']):.1f}%)")
        
        results = {
            # 原始数据统计（包含异常值）
            'raw_avg_mse': np.mean(all_metrics['mse']) if len(all_metrics['mse']) > 0 else float('nan'),
            'raw_avg_ade': np.mean(all_metrics['ade']) if len(all_metrics['ade']) > 0 else float('nan'),
            'raw_avg_acc_kl': np.mean(all_metrics['acc_kl']) if len(all_metrics['acc_kl']) > 0 else float('nan'),
            'raw_avg_curv_kl': np.mean(all_metrics['curv_kl']) if len(all_metrics['curv_kl']) > 0 else float('nan'),
            
            # 过滤后的统计（主要指标）
            'avg_mse': np.mean(filtered_metrics['mse']) if len(filtered_metrics['mse']) > 0 else float('nan'),
            'std_mse': np.std(filtered_metrics['mse']) if len(filtered_metrics['mse']) > 0 else float('nan'),
            'min_mse': np.min(filtered_metrics['mse']) if len(filtered_metrics['mse']) > 0 else float('nan'),
            'max_mse': np.max(filtered_metrics['mse']) if len(filtered_metrics['mse']) > 0 else float('nan'),
            
            'avg_ade': np.mean(filtered_metrics['ade']) if len(filtered_metrics['ade']) > 0 else float('nan'),
            'std_ade': np.std(filtered_metrics['ade']) if len(filtered_metrics['ade']) > 0 else float('nan'),
            'min_ade': np.min(filtered_metrics['ade']) if len(filtered_metrics['ade']) > 0 else float('nan'),
            'max_ade': np.max(filtered_metrics['ade']) if len(filtered_metrics['ade']) > 0 else float('nan'),
            
            'avg_fde': np.mean(filtered_metrics['fde']) if len(filtered_metrics['fde']) > 0 else float('nan'),
            'std_fde': np.std(filtered_metrics['fde']) if len(filtered_metrics['fde']) > 0 else float('nan'),
            'min_fde': np.min(filtered_metrics['fde']) if len(filtered_metrics['fde']) > 0 else float('nan'),
            'max_fde': np.max(filtered_metrics['fde']) if len(filtered_metrics['fde']) > 0 else float('nan'),
            
            'avg_mr_2m': np.mean(filtered_metrics['mr_2m']) if len(filtered_metrics['mr_2m']) > 0 else float('nan'),
            'std_mr_2m': np.std(filtered_metrics['mr_2m']) if len(filtered_metrics['mr_2m']) > 0 else float('nan'),
            
            'avg_acc_kl': np.mean(filtered_metrics['acc_kl']) if len(filtered_metrics['acc_kl']) > 0 else float('nan'),
            'std_acc_kl': np.std(filtered_metrics['acc_kl']) if len(filtered_metrics['acc_kl']) > 0 else float('nan'),
            'min_acc_kl': np.min(filtered_metrics['acc_kl']) if len(filtered_metrics['acc_kl']) > 0 else float('nan'),
            'max_acc_kl': np.max(filtered_metrics['acc_kl']) if len(filtered_metrics['acc_kl']) > 0 else float('nan'),
            'avg_curv_kl': np.mean(filtered_metrics['curv_kl']) if len(filtered_metrics['curv_kl']) > 0 else float('nan'),
            'std_curv_kl': np.std(filtered_metrics['curv_kl']) if len(filtered_metrics['curv_kl']) > 0 else float('nan'),
            'min_curv_kl': np.min(filtered_metrics['curv_kl']) if len(filtered_metrics['curv_kl']) > 0 else float('nan'),
            'max_curv_kl': np.max(filtered_metrics['curv_kl']) if len(filtered_metrics['curv_kl']) > 0 else float('nan'),
            
            'mse_pass_rate': np.mean(np.array(filtered_metrics['mse']) <= self.max_mse_threshold) if len(filtered_metrics['mse']) > 0 else 0.0,
            'ade_pass_rate': np.mean(np.array(filtered_metrics['ade']) <= 0.5) if len(filtered_metrics['ade']) > 0 else 0.0,  # 使用 0.5m 作为 ADE 阈值
            'acc_kl_pass_rate': np.mean(np.array(filtered_metrics['acc_kl']) <= self.max_kl_divergence) if len(filtered_metrics['acc_kl']) > 0 else 0.0,
            'curv_kl_pass_rate': np.mean(np.array(filtered_metrics['curv_kl']) <= self.max_kl_divergence) if len(filtered_metrics['curv_kl']) > 0 else 0.0,
            'kl_pass_rate': np.mean([
                np.mean(np.array(filtered_metrics['acc_kl']) <= self.max_kl_divergence) if len(filtered_metrics['acc_kl']) > 0 else 0.0,
                np.mean(np.array(filtered_metrics['curv_kl']) <= self.max_kl_divergence) if len(filtered_metrics['curv_kl']) > 0 else 0.0
            ]),
            'total_scenes': total_scenes,
            'valid_mse_scenes': len(all_metrics['mse']),
            'filtered_mse_scenes': len(filtered_metrics['mse']),
            'valid_kl_scenes': min(len(all_metrics['acc_kl']), len(all_metrics['curv_kl'])),
            'filtered_kl_scenes': min(len(filtered_metrics['acc_kl']), len(filtered_metrics['curv_kl'])),
            'scene_details': scene_details,
            'outlier_info': outlier_info,
            'filtering_enabled': self.filter_outliers
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="评估 STRIVE CVAE 模型 (traffic_model.pth) 的轨迹生成质量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本评估（使用验证集）
  python evaluate_traffic_model_cvae.py \\
      --ckpt model_ckpt/traffic_model.pth \\
      --data_dir data/nuscenes
  
  # 评估指定数量的场景
  python evaluate_traffic_model_cvae.py \\
      --ckpt model_ckpt/traffic_model.pth \\
      --data_dir data/nuscenes \\
      --num_samples 500
  
  # 使用自定义阈值
  python evaluate_traffic_model_cvae.py \\
      --ckpt model_ckpt/traffic_model.pth \\
      --data_dir data/nuscenes \\
      --max_mse 0.05 \\
      --max_kl 0.5
        """
    )
    
    parser.add_argument(
        "--ckpt",
        required=True,
        help="训练好的 traffic_model.pth 路径"
    )
    
    parser.add_argument(
        "--data_dir",
        required=True,
        help="nuScenes 数据集目录路径"
    )
    
    parser.add_argument(
        "--data_version",
        default="trainval",
        help="数据集版本（默认：trainval）"
    )
    
    parser.add_argument(
        "--split",
        default="val",
        choices=['train', 'val', 'test'],
        help="使用哪个数据集分割（默认：val）"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="评估的场景数量（默认：全部）"
    )
    
    parser.add_argument(
        "--max_mse",
        type=float,
        default=0.05,
        help="MSE 阈值 (m²)，默认: 0.05"
    )
    
    parser.add_argument(
        "--max_kl",
        type=float,
        default=0.5,
        help="KL 散度阈值，默认: 0.5"
    )
    
    parser.add_argument(
        "--out",
        default="./out/cvae_evaluation",
        help="输出目录"
    )
    
    parser.add_argument(
        "--no_filter_outliers",
        action="store_true",
        help="不过滤异常值（默认会过滤）"
    )
    
    parser.add_argument(
        "--outlier_iqr_factor",
        type=float,
        default=1.5,
        help="IQR 异常值检测因子（默认：1.5，更大=更宽松）"
    )
    
    args = parser.parse_args()
    
    # 显示评估配置
    console.print(Panel.fit(
        f"[bold cyan]STRIVE CVAE 模型质量评估[/bold cyan]\n\n"
        f"[yellow]模型路径:[/yellow] {args.ckpt}\n"
        f"[yellow]数据集:[/yellow] {args.data_dir} ({args.split} split)\n"
        f"[yellow]评估方法:[/yellow] 使用后验均值重建 (model.reconstruct)\n\n"
        f"[green]质量阈值:[/green]\n"
        f"  • 轨迹 MSE: ≤ {args.max_mse} m²\n"
        f"  • 加速度 KL 散度: ≤ {args.max_kl}\n"
        f"  • 曲率 KL 散度: ≤ {args.max_kl}",
        title="配置信息",
        border_style="cyan"
    ))
    
    # 设备设置
    device = get_device()
    console.print(f"[dim]使用设备: {device}[/dim]\n")
    
    # 创建输出目录
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, 'evaluation_log.txt')
    Logger.init(log_path)
    Logger.log(f'开始 CVAE 模型评估: {args.ckpt}')
    
    # 加载数据集和地图
    console.print("[cyan]加载数据集和地图环境...[/cyan]")
    
    data_path = os.path.join(args.data_dir, args.data_version)
    
    # 默认配置（与训练时一致）
    cfg = dict2obj({
        'past_len': 4,
        'future_len': 12,
        'map_obs_size_pix': 256,
        'map_obs_bounds': [-17.0, -38.5, 60.0, 38.5],
        'map_layers': ['drivable_area', 'carpark_area', 'road_divider', 'lane_divider'],
        'agent_types': ['car', 'truck'],
        'reduce_cats': False,
        'map_feat_size': 64,
        'past_feat_size': 64,
        'future_feat_size': 64,
        'latent_size': 32,
        'model_output_bicycle': True,
        'conv_kernel_list': [7, 5, 5, 3, 3, 3],
        'conv_stride_list': [2, 2, 2, 2, 2, 2],
        'conv_filter_list': [16, 32, 64, 64, 128, 128]
    })
    
    map_env = NuScenesMapEnv(
        data_path,
        bounds=cfg.map_obs_bounds,
        L=cfg.map_obs_size_pix,
        W=cfg.map_obs_size_pix,
        layers=cfg.map_layers,
        device=device
    )
    
    test_dataset = NuScenesDataset(
        data_path,
        map_env,
        version=args.data_version,
        split=args.split,
        categories=cfg.agent_types,
        npast=cfg.past_len,
        nfuture=cfg.future_len,
        reduce_cats=cfg.reduce_cats
    )
    
    console.print(f"[green]✓ 数据集加载完成: {len(test_dataset)} 个场景[/green]")
    
    # 创建模型
    console.print("[cyan]创建模型...[/cyan]")
    model = TrafficModel(
        cfg.past_len, cfg.future_len, cfg.map_obs_size_pix, len(test_dataset.categories),
        map_feat_size=cfg.map_feat_size,
        past_feat_size=cfg.past_feat_size,
        future_feat_size=cfg.future_feat_size,
        latent_size=cfg.latent_size,
        output_bicycle=cfg.model_output_bicycle,
        conv_channel_in=map_env.num_layers,
        conv_kernel_list=cfg.conv_kernel_list,
        conv_stride_list=cfg.conv_stride_list,
        conv_filter_list=cfg.conv_filter_list
    ).to(device)
    
    # 加载模型权重
    ckpt_epoch, _ = load_state(args.ckpt, model, map_location=device)
    Logger.log(f'加载模型权重 (epoch {ckpt_epoch})')
    console.print(f"[green]✓ 模型加载完成 (epoch {ckpt_epoch})[/green]")
    console.print(f"[dim]模型参数数量: {count_params(model):,}[/dim]\n")
    
    # 设置归一化器
    model.set_normalizer(test_dataset.get_state_normalizer())
    model.set_att_normalizer(test_dataset.get_att_normalizer())
    if cfg.model_output_bicycle:
        from datasets.utils import NUSC_BIKE_PARAMS
        model.set_bicycle_params(NUSC_BIKE_PARAMS)
    
    # 创建评估器
    evaluator = CVAEModelEvaluator(
        model=model,
        test_dataset=test_dataset,
        map_env=map_env,
        device=device,
        max_mse_threshold=args.max_mse,
        max_kl_divergence=args.max_kl,
        filter_outliers=not args.no_filter_outliers,
        outlier_iqr_factor=args.outlier_iqr_factor
    )
    
    # 执行评估
    start_time = time.time()
    results = evaluator.evaluate_dataset(num_samples=args.num_samples)
    eval_time = time.time() - start_time
    
    # 打印结果
    console.print("\n" + "="*70)
    
    # 检查是否有有效结果
    has_valid_mse = not np.isnan(results['avg_mse'])
    has_valid_ade = not np.isnan(results['avg_ade'])
    has_valid_fde = not np.isnan(results['avg_fde'])
    has_valid_mr = not np.isnan(results['avg_mr_2m'])
    has_valid_acc_kl = not np.isnan(results['avg_acc_kl'])
    has_valid_curv_kl = not np.isnan(results['avg_curv_kl'])
    
    # 判断是否通过（只需要满足阈值）
    mse_pass = has_valid_mse and results['avg_mse'] <= args.max_mse
    ade_pass = has_valid_ade and results['avg_ade'] <= 0.5  # ADE 阈值 0.5m
    acc_kl_pass = has_valid_acc_kl and results['avg_acc_kl'] <= args.max_kl
    curv_kl_pass = has_valid_curv_kl and results['avg_curv_kl'] <= args.max_kl
    
    # 格式化输出
    if has_valid_mse:
        mse_avg_str = f"{results['avg_mse']:.6f} m²"
        mse_min_str = f"{results['min_mse']:.6f} m²"
        mse_max_str = f"{results['max_mse']:.6f} m²"
        mse_status = "✅ 通过" if mse_pass else "❌ 未通过"
    else:
        mse_avg_str = "N/A"
        mse_min_str = "N/A"
        mse_max_str = "N/A"
        mse_status = "N/A"
    
    if has_valid_acc_kl:
        acc_kl_avg_str = f"{results['avg_acc_kl']:.6f}"
        acc_kl_min_str = f"{results['min_acc_kl']:.6f}"
        acc_kl_max_str = f"{results['max_acc_kl']:.6f}"
        acc_kl_status = "✅ 通过" if acc_kl_pass else "❌ 未通过"
    else:
        acc_kl_avg_str = "N/A"
        acc_kl_min_str = "N/A"
        acc_kl_max_str = "N/A"
        acc_kl_status = "N/A"
    
    if has_valid_curv_kl:
        curv_kl_avg_str = f"{results['avg_curv_kl']:.6f}"
        curv_kl_min_str = f"{results['min_curv_kl']:.6f}"
        curv_kl_max_str = f"{results['max_curv_kl']:.6f}"
        curv_kl_status = "✅ 通过" if curv_kl_pass else "❌ 未通过"
    else:
        curv_kl_avg_str = "N/A"
        curv_kl_min_str = "N/A"
        curv_kl_max_str = "N/A"
        curv_kl_status = "N/A"
    
    # 准备统计信息
    filtering_info = ""
    if results.get('filtering_enabled', False):
        filtering_info = (
            f"\n[dim]异常值过滤: "
            f"MSE 移除 {results['outlier_info'].get('mse_outlier_count', 0)} 个, "
            f"加速度 KL 移除 {results['outlier_info'].get('acc_kl_outlier_count', 0)} 个, "
            f"曲率 KL 移除 {results['outlier_info'].get('curv_kl_outlier_count', 0)} 个[/dim]"
        )
        
        # 显示原始平均值对比
        if has_valid_mse and not np.isnan(results.get('raw_avg_mse', float('nan'))):
            filtering_info += f"\n[dim]原始 MSE 平均值（含异常值）: {results['raw_avg_mse']:.6f} m²[/dim]"
    
    console.print(Panel.fit(
        f"[bold yellow]CVAE 模型质量评估结果[/bold yellow]"
        f"{' [过滤异常值后]' if results.get('filtering_enabled', False) else ''}\n\n"
        f"[cyan]1. 轨迹重建 MSE:[/cyan]\n"
        f"   平均值: [bold]{mse_avg_str}[/bold]  |  最小值: [bold green]{mse_min_str}[/bold green]  |  最大值: {mse_max_str}\n"
        f"   阈值: ≤ {args.max_mse} m²  |  结果: [bold]{mse_status}[/bold]\n\n"
        f"[cyan]2. 加速度分布 KL 散度:[/cyan]\n"
        f"   平均值: [bold]{acc_kl_avg_str}[/bold]  |  最小值: [bold green]{acc_kl_min_str}[/bold green]  |  最大值: {acc_kl_max_str}\n"
        f"   阈值: ≤ {args.max_kl}  |  结果: [bold]{acc_kl_status}[/bold]\n\n"
        f"[cyan]3. 曲率分布 KL 散度:[/cyan]\n"
        f"   平均值: [bold]{curv_kl_avg_str}[/bold]  |  最小值: [bold green]{curv_kl_min_str}[/bold green]  |  最大值: {curv_kl_max_str}\n"
        f"   阈值: ≤ {args.max_kl}  |  结果: [bold]{curv_kl_status}[/bold]\n\n"
        f"[cyan]评估统计:[/cyan]\n"
        f"   总场景数: {results['total_scenes']} | "
        f"有效 MSE: {results.get('valid_mse_scenes', 0)} → 过滤后: {results.get('filtered_mse_scenes', 0)} | "
        f"有效 KL: {results.get('valid_kl_scenes', 0)} → 过滤后: {results.get('filtered_kl_scenes', 0)}\n"
        f"   评估用时: {eval_time:.2f} 秒"
        f"{filtering_info}",
        title="📊 评估结果",
        border_style="green" if (mse_pass and acc_kl_pass and curv_kl_pass) else "yellow"
    ))
    
    # 保存结果
    results_file = os.path.join(args.out, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"CVAE Model Evaluation Results\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Model: {args.ckpt}\n")
        f.write(f"Dataset: {args.data_dir} ({args.split})\n")
        f.write(f"Total scenes: {results['total_scenes']}\n")
        f.write(f"Valid MSE scenes: {results.get('valid_mse_scenes', 0)}\n")
        f.write(f"Filtered MSE scenes: {results.get('filtered_mse_scenes', 0)}\n")
        f.write(f"Valid KL scenes: {results.get('valid_kl_scenes', 0)}\n")
        f.write(f"Filtered KL scenes: {results.get('filtered_kl_scenes', 0)}\n")
        f.write(f"Outlier filtering: {'Enabled' if results.get('filtering_enabled', False) else 'Disabled'}\n\n")
        
        if results.get('filtering_enabled', False) and 'outlier_info' in results:
            f.write(f"Outliers Removed:\n")
            f.write(f"  MSE: {results['outlier_info'].get('mse_outlier_count', 0)}\n")
            f.write(f"  Acceleration KL: {results['outlier_info'].get('acc_kl_outlier_count', 0)}\n")
            f.write(f"  Curvature KL: {results['outlier_info'].get('curv_kl_outlier_count', 0)}\n\n")
            
            if not np.isnan(results.get('raw_avg_mse', float('nan'))):
                f.write(f"Original MSE (with outliers): {results['raw_avg_mse']:.6f} m²\n")
                f.write(f"Filtered MSE (without outliers): {results['avg_mse']:.6f} m²\n\n")
        
        f.write(f"Results:\n")
        f.write(f"-" * 50 + "\n")
        if has_valid_mse:
            f.write(f"1. Trajectory MSE (threshold: ≤ {args.max_mse} m²)\n")
            f.write(f"   Average: {results['avg_mse']:.6f} m²\n")
            f.write(f"   Minimum: {results['min_mse']:.6f} m² (best case)\n")
            f.write(f"   Maximum: {results['max_mse']:.6f} m²\n")
            f.write(f"   Status: {'PASS' if mse_pass else 'FAIL'}\n\n")
        else:
            f.write(f"1. Trajectory MSE: N/A\n\n")
        
        if has_valid_acc_kl:
            f.write(f"2. Acceleration KL Divergence (threshold: ≤ {args.max_kl})\n")
            f.write(f"   Average: {results['avg_acc_kl']:.6f}\n")
            f.write(f"   Minimum: {results['min_acc_kl']:.6f} (best case)\n")
            f.write(f"   Maximum: {results['max_acc_kl']:.6f}\n")
            f.write(f"   Status: {'PASS' if acc_kl_pass else 'FAIL'}\n\n")
        else:
            f.write(f"2. Acceleration KL Divergence: N/A\n\n")
        
        if has_valid_curv_kl:
            f.write(f"3. Curvature KL Divergence (threshold: ≤ {args.max_kl})\n")
            f.write(f"   Average: {results['avg_curv_kl']:.6f}\n")
            f.write(f"   Minimum: {results['min_curv_kl']:.6f} (best case)\n")
            f.write(f"   Maximum: {results['max_curv_kl']:.6f}\n")
            f.write(f"   Status: {'PASS' if curv_kl_pass else 'FAIL'}\n\n")
        else:
            f.write(f"3. Curvature KL Divergence: N/A\n\n")
        
        f.write(f"-" * 50 + "\n")
        if mse_pass and acc_kl_pass and curv_kl_pass:
            f.write(f"Overall: ALL METRICS PASS\n")
        else:
            f.write(f"Overall: SOME METRICS FAIL\n")
    
    console.print(f"\n[green]✓ 结果已保存到: {results_file}[/green]")
    
    # 保存详细场景结果到CSV
    if 'scene_details' in results and len(results['scene_details']) > 0:
        import pandas as pd
        csv_file = os.path.join(args.out, 'scene_details.csv')
        
        # 创建DataFrame
        df = pd.DataFrame(results['scene_details'])
        
        # 按MSE排序（方便查看最差的场景）
        if 'mse' in df.columns:
            df = df.sort_values('mse', ascending=False, na_position='last')
        
        # 保存CSV
        df.to_csv(csv_file, index=False, float_format='%.6f')
        console.print(f"[green]✓ 详细场景结果已保存到: {csv_file}[/green]")
        
        # 显示统计信息
        if has_valid_mse:
            console.print(f"\n[cyan]场景统计分析:[/cyan]")
            console.print(f"  • MSE > 1.0 m² 的场景数: {len(df[df['mse'] > 1.0])} / {len(df[df['mse'].notna()])}")
            console.print(f"  • MSE > 0.5 m² 的场景数: {len(df[df['mse'] > 0.5])} / {len(df[df['mse'].notna()])}")
            console.print(f"  • MSE > 0.1 m² 的场景数: {len(df[df['mse'] > 0.1])} / {len(df[df['mse'].notna()])}")
            console.print(f"  • MSE ≤ 0.05 m² (通过) 的场景数: {len(df[df['mse'] <= 0.05])} / {len(df[df['mse'].notna()])}")
            
            # 显示最差的5个场景
            worst_scenes = df[df['mse'].notna()].head(5)
            if len(worst_scenes) > 0:
                console.print(f"\n[yellow]MSE 最高的 5 个场景:[/yellow]")
                from rich.table import Table
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("场景ID", justify="right", style="cyan")
                table.add_column("MSE (m²)", justify="right", style="red")
                table.add_column("加速度KL", justify="right")
                table.add_column("曲率KL", justify="right")
                table.add_column("车辆数", justify="right")
                
                for _, row in worst_scenes.iterrows():
                    table.add_row(
                        str(row['scene_id']),
                        f"{row['mse']:.6f}" if pd.notna(row['mse']) else "N/A",
                        f"{row['acc_kl']:.6f}" if pd.notna(row['acc_kl']) else "N/A",
                        f"{row['curv_kl']:.6f}" if pd.notna(row['curv_kl']) else "N/A",
                        str(row['num_agents'])
                    )
                console.print(table)
    
    Logger.log(f'评估完成，用时 {eval_time:.2f} 秒')
    
    # 返回状态码
    if mse_pass and acc_kl_pass and curv_kl_pass:
        console.print("\n[bold green]🎉 CVAE 模型满足所有质量指标要求！[/bold green]\n")
        return 0
    else:
        console.print("\n[bold yellow]⚠️ CVAE 模型未能满足所有质量指标[/bold yellow]\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

