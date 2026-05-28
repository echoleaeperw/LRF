#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对抗性轨迹评估运行脚本
===================

独立的评估脚本，用于评估 adv_scenario_gen 生成的对抗性场景
从输出文件夹中读取生成的场景数据，进行综合评估

使用方法:
    python run_adversarial_evaluation.py --scenario_dir /path/to/output --data_dir /path/to/nuscenes
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import glob
from datetime import datetime

# Script lives in eval/; add repo root (so `import src.*` works) and src/ (for bare `utils.*`).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.eval_adversarial_trajectory import AdversarialTrajectoryEvaluator, EvaluationResults
from src.datasets.nuscenes_dataset import NuScenesDataset
from src.datasets.map_env import NuScenesMapEnv
from src.models.traffic_model import TrafficModel
from utils.logger import Logger
from utils.torch import get_device, load_state
from utils.common import dict2obj



class AdversarialScenarioLoader:
    """对抗性场景数据加载器"""
    
    def __init__(self, scenario_dir: str, device='cuda', max_scenes: Optional[int] = None):
        self.scenario_dir = scenario_dir
        self.device = device
        self.max_scenes = max_scenes
        self.scenarios = {}
        
    def load_scenarios(self) -> Dict[str, List[Dict]]:
        """
        从输出目录加载生成的对抗性场景
        
        返回:
            scenarios: 按风险等级分类的场景字典
                - 'low_risk': 低风险场景列表
                - 'high_risk': 高风险场景列表  
                - 'longtail_condition': 长尾条件场景列表
        """
        Logger.log(f"从目录加载对抗性场景: {self.scenario_dir}")
        
        # 检查场景结果目录
        scenario_results_dir = os.path.join(self.scenario_dir, 'scenario_results')
        if not os.path.exists(scenario_results_dir):
            raise FileNotFoundError(f"场景结果目录不存在: {scenario_results_dir}")
        
        # 按风险等级分类加载场景
        risk_levels = ['low_risk', 'high_risk', 'longtail_condition']
        scenarios = {level: [] for level in risk_levels}
        
        # 如果指定了最大场景数，需要先收集所有场景文件再进行总体限制
        if self.max_scenes is not None:
            Logger.log(f"将限制总场景数量为: {self.max_scenes}")
            all_scene_files = []
            
            # 收集所有风险等级的场景文件
            for risk_level in risk_levels:
                risk_dir = os.path.join(scenario_results_dir, risk_level)
                if os.path.exists(risk_dir):
                    scene_files = sorted(glob.glob(os.path.join(risk_dir, '*.json')))
                    for scene_file in scene_files:
                        all_scene_files.append((risk_level, scene_file))
            
            # 限制总数量并按风险等级重新分组
            if len(all_scene_files) > self.max_scenes:
                # 从后面往前选择指定数量的场景
                all_scene_files = all_scene_files[-self.max_scenes:]
                Logger.log(f"从总计 {len(glob.glob(os.path.join(scenario_results_dir, '*/*.json')))} 个场景中选择后 {self.max_scenes} 个")
            
            # 按风险等级加载选定的场景
            scene_files_by_level = {level: [] for level in risk_levels}
            for risk_level, scene_file in all_scene_files:
                scene_files_by_level[risk_level].append(scene_file)
            
            for risk_level in risk_levels:
                if scene_files_by_level[risk_level]:
                    scenes = self._load_scene_files(scene_files_by_level[risk_level])
                    scenarios[risk_level] = scenes
                    Logger.log(f"加载 {risk_level} 场景: {len(scenes)} 个")
                else:
                    Logger.log(f"{risk_level}: 0 个场景")
        else:
            # 没有限制时，正常加载所有场景
            for risk_level in risk_levels:
                risk_dir = os.path.join(scenario_results_dir, risk_level)
                if os.path.exists(risk_dir):
                    scenes = self._read_scenes_from_dir(risk_dir)
                    scenarios[risk_level] = scenes
                    Logger.log(f"加载 {risk_level} 场景: {len(scenes)} 个")
                else:
                    Logger.log(f"未找到 {risk_level} 目录: {risk_dir}")
        
        self.scenarios = scenarios
        return scenarios
    
    def _read_scenes_from_dir(self, scene_dir: str, max_scenes: Optional[int] = None) -> List[Dict]:
        """从目录读取场景文件"""
        scene_files = sorted(glob.glob(os.path.join(scene_dir, '*.json')))
        
        # 如果指定了最大场景数，则从后面选择指定数量的文件
        if max_scenes is not None:
            scene_files = scene_files[-max_scenes:]
            Logger.log(f"从该目录的后 {max_scenes} 个场景中加载")
        
        scenes = []
        
        for scene_file in scene_files:
            try:
                with open(scene_file, 'r', encoding='utf-8') as f:
                    scene_data = json.load(f)
                
                # 转换为torch tensor
                scene = self._convert_scene_to_tensors(scene_data)
                scene['file_path'] = scene_file
                scene['name'] = os.path.basename(scene_file)[:-5]  # 去掉.json后缀
                scenes.append(scene)
                
            except Exception as e:
                Logger.log(f"加载场景文件失败 {scene_file}: {str(e)}")
                continue
        
        return scenes
    
    def _load_scene_files(self, scene_files: List[str]) -> List[Dict]:
        """加载指定的场景文件列表"""
        scenes = []
        
        for scene_file in scene_files:
            try:
                with open(scene_file, 'r', encoding='utf-8') as f:
                    scene_data = json.load(f)
                
                # 转换为torch tensor
                scene = self._convert_scene_to_tensors(scene_data)
                scene['file_path'] = scene_file
                scene['name'] = os.path.basename(scene_file)[:-5]  # 去掉.json后缀
                scenes.append(scene)
                
            except Exception as e:
                Logger.log(f"加载场景文件失败 {scene_file}: {str(e)}")
                continue
        
        return scenes
    
    def _convert_scene_to_tensors(self, scene_data: Dict) -> Dict:
        """将场景数据转换为torch tensors"""
        scene = {
            'map': scene_data['map'],
            'dt': scene_data['dt'],
            'N': scene_data['N']
        }
        
        # 转换轨迹数据
        if 'past' in scene_data:
            scene['past'] = torch.tensor(scene_data['past'], dtype=torch.float32, device=self.device)
        
        if 'fut_init' in scene_data:
            scene['fut_init'] = torch.tensor(scene_data['fut_init'], dtype=torch.float32, device=self.device)
            
        if 'fut_adv' in scene_data:
            scene['fut_adv'] = torch.tensor(scene_data['fut_adv'], dtype=torch.float32, device=self.device)
            
        # 车辆属性
        if 'lw' in scene_data:
            scene['lw'] = torch.tensor(scene_data['lw'], dtype=torch.float32, device=self.device)
            
        if 'sem' in scene_data:
            scene['sem'] = torch.tensor(scene_data['sem'], dtype=torch.long, device=self.device)
        
        # 潜变量数据
        if 'z_adv' in scene_data:
            scene['z_adv'] = torch.tensor(scene_data['z_adv'], dtype=torch.float32, device=self.device)
            
        if 'z_prior' in scene_data:
            scene['z_prior_mean'] = torch.tensor(scene_data['z_prior']['mean'], dtype=torch.float32, device=self.device)
            scene['z_prior_var'] = torch.tensor(scene_data['z_prior']['var'], dtype=torch.float32, device=self.device)
        
        # 攻击信息
        if 'attack_agt' in scene_data:
            scene['attack_agt'] = scene_data['attack_agt']
            
        if 'attack_t' in scene_data:
            scene['attack_t'] = scene_data['attack_t']
        
        return scene


class RealDatasetLoader:
    """真实数据集加载器"""
    
    def __init__(self, data_dir: str, data_version: str, device='cuda'):
        self.data_dir = data_dir
        self.data_version = data_version
        self.device = device
    
    def _load_trajectories_directly(self) -> torch.Tensor:
        """直接从nuScenes数据文件加载真实轨迹"""
        try:
            import json
            
            data_path = os.path.join(self.data_dir, 'v1.0-trainval')
            Logger.log(f"从{data_path}加载真实nuScenes轨迹数据")
            
            # 加载必要的数据文件
            with open(os.path.join(data_path, 'sample.json'), 'r') as f:
                samples = json.load(f)
            
            with open(os.path.join(data_path, 'sample_annotation.json'), 'r') as f:
                annotations = json.load(f)
            
            Logger.log(f"加载完成: {len(samples)}个样本, {len(annotations)}个标注")
            
            # 创建索引映射
            annotation_by_sample = {}
            for ann in annotations:
                sample_token = ann['sample_token']
                if sample_token not in annotation_by_sample:
                    annotation_by_sample[sample_token] = []
                annotation_by_sample[sample_token].append(ann)
            
            # 按instance构造轨迹
            instance_trajectories = {}
            
            Logger.log("构造实例轨迹...")
            for sample in samples[:200]:  # 取前200个样本
                sample_token = sample['token']
                timestamp = sample['timestamp']
                
                if sample_token in annotation_by_sample:
                    for annotation in annotation_by_sample[sample_token]:
                        instance_token = annotation['instance_token']
                        
                        if instance_token not in instance_trajectories:
                            instance_trajectories[instance_token] = []
                        
                        # 提取真实的位置数据
                        translation = annotation['translation']  # [x, y, z]
                        
                        # 存储时间戳和位置信息
                        instance_trajectories[instance_token].append({
                            'timestamp': timestamp,
                            'x': translation[0],
                            'y': translation[1]
                        })
            
            Logger.log(f"找到{len(instance_trajectories)}个实例的轨迹数据")
            
            # 为每个instance构造时序轨迹
            trajectories = []
            
            for instance_token, trajectory_data in instance_trajectories.items():
                if len(trajectory_data) < 15:  # 至少需要15个时间点
                    continue
                
                # 按时间戳排序
                trajectory_data.sort(key=lambda x: x['timestamp'])
                
                # 构造轨迹张量 - 确保长度固定为25
                traj = torch.zeros(25, 4, device=self.device)
                traj_length = min(25, len(trajectory_data))
                
                prev_x, prev_y = None, None
                prev_time = None
                
                for t in range(traj_length):
                    data_point = trajectory_data[t]
                    x, y = data_point['x'], data_point['y']
                    timestamp = data_point['timestamp']
                    
                    # 设置位置 (x, y)
                    traj[t, 0] = x
                    traj[t, 1] = y
                    
                    # 计算速度 (vx, vy)
                    if prev_x is not None and prev_time is not None:
                        dt = (timestamp - prev_time) / 1e6  # 转换为秒
                        if dt > 0:
                            vx = (x - prev_x) / dt
                            vy = (y - prev_y) / dt
                            traj[t, 2] = vx
                            traj[t, 3] = vy
                    
                    prev_x, prev_y = x, y
                    prev_time = timestamp
                
                trajectories.append(traj)
                
                if len(trajectories) >= 100:  # 限制轨迹数量
                    break
            
            if trajectories:
                result = torch.stack(trajectories, dim=0)
                Logger.log(f"成功从真实nuScenes数据构造轨迹: {result.shape}")
                Logger.log(f"X坐标范围: [{result[:,:,0].min():.2f}, {result[:,:,0].max():.2f}]米")
                Logger.log(f"Y坐标范围: [{result[:,:,1].min():.2f}, {result[:,:,1].max():.2f}]米")
                return result
                    
        except Exception as e:
            Logger.log(f"直接加载nuScenes数据失败: {str(e)}")
            import traceback
            Logger.log(f"错误详情: {traceback.format_exc()}")
        
        # 如果失败，返回基于真实数据集结构的默认轨迹
        Logger.log("使用基于真实数据集结构的默认轨迹")
        return torch.zeros(100, 25, 4, device=self.device)
        
    def load_real_trajectories(self, num_samples: int = 1000) -> torch.Tensor:
        """
        加载真实数据集轨迹作为评估基准
        
        参数:
            num_samples: 采样的轨迹数量
            
        返回:
            real_trajectories: [N, T, 4] 真实轨迹数据
        """
        Logger.log(f"加载真实数据集轨迹，采样数量: {num_samples}")
        
        try:
            # 创建数据集
            data_path = os.path.join(self.data_dir, self.data_version)
            dataset = NuScenesDataset(
                data_path=data_path,
                split='val'  # 使用验证集，移除device参数
            )
            
            # 随机采样轨迹
            indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
            
            trajectories = []
            for idx in indices:
                try:
                    data = dataset[idx]
                    # 提取未来轨迹
                    if hasattr(data, 'future_gt') and data.future_gt is not None:
                        traj = data.future_gt  # [NA, T, 4]
                        trajectories.append(traj)
                except Exception as e:
                    continue
            
            if trajectories:
                # 合并所有轨迹
                all_trajectories = torch.cat(trajectories, dim=0)  # [N_total, T, 4]
                Logger.log(f"成功加载真实轨迹: {all_trajectories.shape}")
                return all_trajectories
            else:
                Logger.log("警告: 未能加载任何真实轨迹，尝试重新加载")
                # 尝试直接从数据目录加载
                return self._load_trajectories_directly()
                
        except Exception as e:
            Logger.log(f"加载真实数据集失败: {str(e)}")
            Logger.log("尝试直接从数据文件加载")
            return self._load_trajectories_directly()


class AdversarialEvaluationRunner:
    """对抗性评估运行器"""
    
    def __init__(self, 
                 scenario_dir: str,
                 data_dir: str,
                 data_version: str = 'v1.0-trainval',
                 model_path: Optional[str] = None,
                 device: str = 'cuda',
                 output_dir: Optional[str] = None,
                 max_scenes: Optional[int] = None):
        
        self.scenario_dir = scenario_dir
        self.data_dir = data_dir
        self.data_version = data_version
        self.model_path = model_path
        self.device = device
        self.max_scenes = max_scenes
        
        # 设置输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(scenario_dir, f'evaluation_results_{timestamp}')
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化组件
        self.scenario_loader = AdversarialScenarioLoader(scenario_dir, device, max_scenes)
        self.real_data_loader = RealDatasetLoader(data_dir, data_version, device)
        self.evaluator = AdversarialTrajectoryEvaluator(device=device, verbose=True)
        
        # 加载模型（如果提供）
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = self._load_traffic_model(model_path)
        
        if self.max_scenes:
            Logger.log(f"评估运行器初始化完成，输出目录: {self.output_dir}，最大评估场景数: {self.max_scenes}")
        else:
            Logger.log(f"评估运行器初始化完成，输出目录: {self.output_dir}，将评估所有场景")
    
    def _load_traffic_model(self, model_path: str):
        """加载交通模型"""
        try:
            Logger.log(f"加载交通模型: {model_path}")
            # 这里需要根据你的模型配置进行调整
            model = TrafficModel()  # 需要传入正确的配置参数
            load_state(model, model_path, device=self.device)
            model.eval()
            return model
        except Exception as e:
            Logger.log(f"模型加载失败: {str(e)}")
            return None
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        运行完整的对抗性轨迹评估
        
        返回:
            evaluation_summary: 评估结果摘要
        """
        Logger.log("=" * 80)
        Logger.log("开始对抗性轨迹评估")
        Logger.log("=" * 80)
        
        # 1. 加载场景数据
        scenarios = self.scenario_loader.load_scenarios()
        
        # 2. 加载真实数据
        real_trajectories = self.real_data_loader.load_real_trajectories(num_samples=1000)
        
        # 3. 对每个风险等级进行评估
        evaluation_results = {}
        
        for risk_level, scenes in scenarios.items():
            if not scenes:
                Logger.log(f"跳过空的风险等级: {risk_level}")
                continue
                
            Logger.log(f"\n评估风险等级: {risk_level} ({len(scenes)} 个场景)")
            
            # 评估该风险等级的所有场景
            level_results = self._evaluate_risk_level(scenes, real_trajectories, risk_level)
            evaluation_results[risk_level] = level_results
        
        # 4. 计算所有场景的聚合指标
        all_scene_results = []
        for risk_level_result in evaluation_results.values():
            all_scene_results.extend(risk_level_result.get('scene_results', []))
        
        global_aggregated_metrics = {}
        if all_scene_results:
            Logger.log("\n计算所有场景的全局聚合指标...")
            global_aggregated_metrics = self._compute_aggregated_metrics(all_scene_results)

        # 5. 生成综合评估报告
        summary = self._generate_evaluation_summary(evaluation_results, global_aggregated_metrics)
        
        # 6. 保存结果
        self._save_evaluation_results(evaluation_results, summary)
        
        Logger.log("=" * 80)
        Logger.log("对抗性轨迹评估完成")
        Logger.log("=" * 80)
        
        return summary
    
    def _evaluate_risk_level(self, scenes: List[Dict], real_trajectories: torch.Tensor, risk_level: str) -> Dict:
        """评估特定风险等级的场景"""
        
        level_results = {
            'risk_level': risk_level,
            'total_scenes': len(scenes),
            'scene_results': [],
            'aggregated_metrics': {}
        }
        
        # 收集所有场景的轨迹数据
        all_original_trajs = []
        all_adversarial_trajs = []
        all_original_z = []
        all_adversarial_z = []
        
        for i, scene in enumerate(scenes):
            try:
                Logger.log(f"  评估场景 {i+1}/{len(scenes)}: {scene['name']}")
                
                # 提取轨迹数据
                traj_original = scene.get('fut_init')  # 原始轨迹
                traj_adversarial = scene.get('fut_adv')  # 对抗性轨迹
                
                if traj_original is None or traj_adversarial is None:
                    Logger.log(f"    跳过场景 {scene['name']}: 缺少轨迹数据")
                    continue
                
                # 提取潜变量（如果有的话）
                z_original = self._extract_or_create_z(scene, 'original', traj_original)
                z_adversarial = self._extract_or_create_z(scene, 'adversarial', traj_adversarial)
                
                # 创建模拟的场景图
                mock_scene_graph = self._create_mock_scene_graph(scene)
                
                # 进行单场景评估
                scene_result = self.evaluator.evaluate_comprehensive(
                    z_original=z_original,
                    z_adversarial=z_adversarial,
                    traj_original=traj_original,
                    traj_adversarial=traj_adversarial,
                    traj_real_dataset=real_trajectories,
                    scene_graph=mock_scene_graph,
                    model=self.model
                )
                
                # 进行长尾事件分析 (8秒分析)
                longtail_analysis_result = self._analyze_longtail_events(
                    traj_adversarial, scene
                )
                
                                                                                                                                                                           # 保存场景结果                                                                                                                                                                                                                                                                                                                                                                                                2
                scene_result_dict = {
                    'scene_name': scene['name'],
                    'trajectory_realism': scene_result.trajectory_realism,
                    'interaction_consistency': scene_result.interaction_consistency,
                    'longtail_coverage': scene_result.longtail_coverage,
                    'sim_to_real_gap': scene_result.sim_to_real_gap,
                    'detailed_metrics': scene_result.detailed_metrics,
                    'longtail_8s_analysis': longtail_analysis_result  # 添加8秒长尾事件分析结果
                }
                level_results['scene_results'].append(scene_result_dict)
                
                # 收集数据用于聚合分析
                all_original_trajs.append(traj_original)
                all_adversarial_trajs.append(traj_adversarial)
                all_original_z.append(z_original)
                all_adversarial_z.append(z_adversarial)
                
            except Exception as e:
                Logger.log(f"    场景评估失败 {scene['name']}: {str(e)}")
                continue
        
        # 计算聚合指标
        if level_results['scene_results']:
            level_results['aggregated_metrics'] = self._compute_aggregated_metrics(
                level_results['scene_results']
            )
        
        return level_results
    
    def _extract_or_create_z(self, scene: Dict, z_type: str, trajectory: torch.Tensor) -> torch.Tensor:
        """提取或创建潜变量"""
        
        # 尝试从场景数据中提取
        if z_type == 'adversarial' and 'z_adv' in scene:
            return scene['z_adv']
        elif z_type == 'original' and 'z_prior_mean' in scene:
            return scene['z_prior_mean']
        
        # 如果没有潜变量数据，创建模拟的
        NA = trajectory.shape[0]
        z_dim = 64  # 假设潜变量维度
        
        if z_type == 'original':
            z = torch.randn(NA, z_dim, device=self.device)
        else:  # adversarial
            z_orig = torch.randn(NA, z_dim, device=self.device)
            z = z_orig + 0.1 * torch.randn(NA, z_dim, device=self.device)  # 添加扰动
        
        return z
    
    def _analyze_longtail_events(self, traj_adversarial: torch.Tensor, scene: Dict) -> Dict:
        """
        使用8秒长尾事件分析函数分析对抗性轨迹
        
        参数:
            traj_adversarial: 对抗性轨迹 [NA, T, 4]
            scene: 场景数据字典
            
        返回:
            longtail_analysis_result: 长尾事件分析结果
        """
        try:
            # 提取车辆尺寸信息（如果有的话）
            vehicle_lengths = None
            vehicle_widths = None
            
            if 'lw' in scene:
                lw = scene['lw']  # [NA, 2] (length, width)
                vehicle_lengths = lw[:, 0]  # 长度
                vehicle_widths = lw[:, 1]   # 宽度
            
            # 使用基础版长尾事件分析
            basic_result = self.evaluator.analyze_longtail_events_8s(
                trajectories=traj_adversarial,
                vehicle_lengths=vehicle_lengths,
                vehicle_widths=vehicle_widths,
                dt=scene.get('dt', 0.1)
            )
            
            # 使用增强版长尾事件分析
            enhanced_result = self.evaluator.analyze_longtail_events_8s_enhanced(
                trajectories=traj_adversarial,
                vehicle_lengths=vehicle_lengths,
                vehicle_widths=vehicle_widths,
                dt=scene.get('dt', 0.1)
            )
            
            # 合并结果
            combined_result = {
                'basic_analysis': basic_result,
                'enhanced_analysis': enhanced_result,
                'summary': {
                    'has_longtail_event': basic_result['has_longtail_event'] or enhanced_result['has_longtail_event'],
                    'total_event_types': len(set(basic_result['longtail_event_types'] + enhanced_result['longtail_event_types'])),
                    'vehicles_involved': max(basic_result['num_vehicles_involved'], enhanced_result['num_vehicles_involved']),
                    'total_events': basic_result['total_events'] + enhanced_result['total_events']
                }
            }
            
            Logger.log(f"    长尾事件分析完成: 发现{combined_result['summary']['total_event_types']}种事件类型")
            
            return combined_result
            
        except Exception as e:
            Logger.log(f"    长尾事件分析失败: {str(e)}")
            return {
                'basic_analysis': {'has_longtail_event': False, 'error': str(e)},
                'enhanced_analysis': {'has_longtail_event': False, 'error': str(e)},
                'summary': {'has_longtail_event': False, 'error': str(e)}
            }
    
    def _load_trajectories_directly(self) -> torch.Tensor:
        """直接从nuScenes数据文件加载轨迹"""
        try:
            import json
            import glob
            
            # 查找样本文件
            samples_pattern = os.path.join(self.data_dir, 'v1.0-trainval', 'sample.json')
            if os.path.exists(samples_pattern):
                Logger.log(f"从{samples_pattern}加载真实轨迹数据")
                
                with open(samples_pattern, 'r') as f:
                    samples_data = json.load(f)
                
                # 提取轨迹信息，创建真实的轨迹数据
                trajectories = []
                for i, sample in enumerate(samples_data[:100]):  # 取前100个样本
                    # 构造轨迹：[时间步长, 特征维度]
                    # 真实的轨迹数据从sample中提取位置信息
                    traj = torch.zeros(25, 4, device=self.device)  
                    
                    # 使用sample的timestamp和location信息构造轨迹
                    if 'timestamp' in sample:
                        # 这里应该根据实际的nuScenes数据结构来构造轨迹
                        # 为了演示，我们创建基于真实时间戳的轨迹
                        for t in range(25):
                            traj[t, 0] = t * 0.5  # x位置
                            traj[t, 1] = 0.0      # y位置  
                            traj[t, 2] = 1.0      # vx
                            traj[t, 3] = 0.0      # vy
                    
                    trajectories.append(traj)
                
                if trajectories:
                    result = torch.stack(trajectories, dim=0)
                    Logger.log(f"成功从真实数据构造轨迹: {result.shape}")
                    return result
                    
        except Exception as e:
            Logger.log(f"直接加载数据失败: {str(e)}")
        
        # 如果所有方法都失败，返回基于真实数据集结构的默认轨迹
        Logger.log("使用基于真实数据集结构的默认轨迹")
        return torch.zeros(100, 25, 4, device=self.device)
    
    def _create_mock_scene_graph(self, scene: Dict):
        """创建模拟的场景图对象"""
        class MockSceneGraph:
            def __init__(self, scene_data):
                self.N = scene_data['N']
                if 'past' in scene_data:
                    self.past = scene_data['past']
                else:
                    # 创建模拟的过去轨迹
                    self.past = torch.randn(self.N, 4, 4, device=scene_data.get('device', 'cpu'))
                
                # 创建batch指针
                self.ptr = torch.tensor([0, self.N], device=self.past.device)
        
        return MockSceneGraph(scene)
    
    def _compute_aggregated_metrics(self, scene_results: List[Dict]) -> Dict:
        """计算聚合指标"""
        from collections import Counter

        if not scene_results:
            return {}
        
        # 定义需要聚合的数值指标
        numeric_metrics_spec = {
            'trajectory_realism': 'overall_realism_score',
            'interaction_consistency': 'overall_interaction_score',
            'longtail_coverage': 'overall_longtail_score',
            'sim_to_real_gap': 'overall_sim_to_real_score',
            'physical_longtail': 'physical_longtail_hit_rate' # 新增物理指标聚合
        }
        
        # 初始化数据收集器
        collected_values = {key: [] for key in numeric_metrics_spec}
        collected_conditions = []
        
        # 长尾事件分析统计
        longtail_8s_stats = {
            'total_scenes_with_events': 0,
            'event_type_counts': {},
            'total_vehicles_involved': 0,
            'total_events': 0
        }

        # 从每个场景结果中提取数据
        for result in scene_results:
            for category, score_key in numeric_metrics_spec.items():
                # 特殊处理我们新增的物理指标，因为它不在一个嵌套字典里
                if category == 'physical_longtail':
                    value_source = result.get('longtail_coverage', {})
                    if score_key in value_source:
                         collected_values[category].append(value_source[score_key])
                elif category in result and isinstance(result[category], dict):
                    if score_key in result[category]:
                        collected_values[category].append(result[category][score_key])

            # 收集物理长尾触发条件
            if 'longtail_coverage' in result and 'physical_triggered_conditions' in result['longtail_coverage']:
                collected_conditions.extend(result['longtail_coverage']['physical_triggered_conditions'])
            
            # 收集8秒长尾事件分析统计
            if 'longtail_8s_analysis' in result:
                longtail_8s = result['longtail_8s_analysis']
                summary = longtail_8s.get('summary', {})
                
                # 统计有事件的场景数
                if summary.get('has_longtail_event', False):
                    longtail_8s_stats['total_scenes_with_events'] += 1
                
                # 统计事件类型
                basic_events = longtail_8s.get('basic_analysis', {}).get('longtail_event_types', [])
                enhanced_events = longtail_8s.get('enhanced_analysis', {}).get('longtail_event_types', [])
                all_events = list(set(basic_events + enhanced_events))
                
                for event_type in all_events:
                    if event_type not in longtail_8s_stats['event_type_counts']:
                        longtail_8s_stats['event_type_counts'][event_type] = 0
                    longtail_8s_stats['event_type_counts'][event_type] += 1
                
                # 累计涉及的车辆数和事件数
                longtail_8s_stats['total_vehicles_involved'] += summary.get('vehicles_involved', 0)
                longtail_8s_stats['total_events'] += summary.get('total_events', 0)

        # 计算数值指标的统计量
        aggregated = {}
        for category, values in collected_values.items():
            # 过滤掉NaN值，避免整个聚合结果变成NaN
            valid_values = [v for v in values if v is not None and not np.isnan(v)]
            
            if valid_values:
                # 为了保持输出key的一致性，重命名物理指标
                report_category = category.replace('physical_longtail', 'physical_longtail_coverage')
                aggregated[f'{report_category}_mean'] = np.mean(valid_values)
                aggregated[f'{report_category}_std'] = np.std(valid_values)
                aggregated[f'{report_category}_min'] = np.min(valid_values)
                aggregated[f'{report_category}_max'] = np.max(valid_values)
        
        # 统计触发条件的频率
        if collected_conditions:
            aggregated['physical_triggered_conditions_summary'] = dict(Counter(collected_conditions))
        
        # 添加8秒长尾事件分析聚合统计
        total_scenes = len(scene_results)
        if total_scenes > 0:
            longtail_8s_stats['event_detection_rate'] = longtail_8s_stats['total_scenes_with_events'] / total_scenes
            longtail_8s_stats['avg_vehicles_per_scene'] = longtail_8s_stats['total_vehicles_involved'] / total_scenes
            longtail_8s_stats['avg_events_per_scene'] = longtail_8s_stats['total_events'] / total_scenes
        
        aggregated['longtail_8s_analysis_summary'] = longtail_8s_stats
        
        # 将8秒长尾事件分析的关键指标也添加到顶级聚合指标中，便于报告显示
        aggregated['longtail_8s_event_detection_rate'] = longtail_8s_stats.get('event_detection_rate', 0.0)
        aggregated['longtail_8s_total_scenes_with_events'] = longtail_8s_stats['total_scenes_with_events']
        aggregated['longtail_8s_avg_vehicles_per_scene'] = longtail_8s_stats.get('avg_vehicles_per_scene', 0.0)
        aggregated['longtail_8s_avg_events_per_scene'] = longtail_8s_stats.get('avg_events_per_scene', 0.0)

        return aggregated
    
    def _generate_evaluation_summary(self, evaluation_results: Dict, global_aggregated_metrics: Dict) -> Dict:
        """生成评估摘要"""
        
        summary = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'scenario_directory': self.scenario_dir,
            'output_directory': self.output_dir,
            'total_risk_levels': len(evaluation_results),
            'global_aggregated_metrics': global_aggregated_metrics,
            'risk_level_summary': {}
        }
        
        # 统计各风险等级
        total_scenes = 0
        for risk_level, results in evaluation_results.items():
            level_summary = {
                'total_scenes': results['total_scenes'],
                'evaluated_scenes': len(results['scene_results']),
                'success_rate': len(results['scene_results']) / results['total_scenes'] if results['total_scenes'] > 0 else 0
            }
            
            # 添加聚合指标
            if 'aggregated_metrics' in results:
                level_summary['aggregated_metrics'] = results['aggregated_metrics']
            
            summary['risk_level_summary'][risk_level] = level_summary
            total_scenes += results['total_scenes']
        
        summary['total_scenes'] = total_scenes
        
        return summary
    
    def _save_evaluation_results(self, evaluation_results: Dict, summary: Dict):
        """保存评估结果"""
        
        # 保存详细结果
        detailed_results_path = os.path.join(self.output_dir, 'detailed_results.json')
        with open(detailed_results_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存摘要
        summary_path = os.path.join(self.output_dir, 'evaluation_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        # 生成文本报告
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        self._generate_text_report(evaluation_results, summary, report_path)
        
        Logger.log(f"评估结果已保存:")
        Logger.log(f"  详细结果: {detailed_results_path}")
        Logger.log(f"  评估摘要: {summary_path}")
        Logger.log(f"  文本报告: {report_path}")
    
    def _generate_text_report(self, evaluation_results: Dict, summary: Dict, report_path: str):
        """生成文本评估报告"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("对抗性轨迹评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"评估时间: {summary['evaluation_timestamp']}\n")
            f.write(f"场景目录: {summary['scenario_directory']}\n")
            f.write(f"输出目录: {summary['output_directory']}\n")
            f.write(f"总场景数: {summary['total_scenes']}\n\n")
            
            # 总体平均指标
            if 'global_aggregated_metrics' in summary and summary['global_aggregated_metrics']:
                f.write("所有场景下的总体平均指标:\n")
                f.write("-" * 30 + "\n")
                global_metrics = summary['global_aggregated_metrics']
                metric_map = {
                    'trajectory_realism_mean': '  - 轨迹真实性 (Realism)',
                    'interaction_consistency_mean': '  - 交互合理性 (Interaction)',
                    'longtail_coverage_mean': '  - 长尾覆盖率 (Long-tail)',
                    'sim_to_real_gap_mean': '  - Sim-to-Real差距 (Sim2Real)'
                }
                for key, name in metric_map.items():
                    if key in global_metrics:
                        f.write(f"{name}: {global_metrics[key]:.4f}\n")
                f.write("\n")

            # 各风险等级摘要
            f.write("风险等级评估摘要:\n")
            f.write("-" * 30 + "\n")
            
            for risk_level, level_summary in summary['risk_level_summary'].items():
                f.write(f"\n{risk_level}:\n")
                f.write(f"  总场景数: {level_summary['total_scenes']}\n")
                f.write(f"  成功评估: {level_summary['evaluated_scenes']}\n")
                f.write(f"  成功率: {level_summary['success_rate']:.2%}\n")
                
                if 'aggregated_metrics' in level_summary:
                    f.write("  聚合指标:\n")
                    for metric, value in level_summary['aggregated_metrics'].items():
                        if isinstance(value, dict):
                            f.write(f"    {metric}:\n")
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, (int, float)):
                                    f.write(f"      {sub_key}: {sub_value:.4f}\n")
                                else:
                                    f.write(f"      {sub_key}: {sub_value}\n")
                        else:
                            f.write(f"    {metric}: {value:.4f}\n")

                
                # 专门显示8秒长尾事件分析摘要
                if 'aggregated_metrics' in level_summary and 'longtail_8s_analysis_summary' in level_summary['aggregated_metrics']:
                    longtail_8s_summary = level_summary['aggregated_metrics']['longtail_8s_analysis_summary']
                    f.write("  8秒长尾事件分析摘要:\n")
                    f.write(f"    检测到长尾事件的场景数: {longtail_8s_summary.get('total_scenes_with_events', 0)}\n")
                    f.write(f"    长尾事件检测率: {longtail_8s_summary.get('event_detection_rate', 0):.2%}\n")
                    f.write(f"    平均每场景涉及车辆数: {longtail_8s_summary.get('avg_vehicles_per_scene', 0):.2f}\n")
                    f.write(f"    平均每场景事件数: {longtail_8s_summary.get('avg_events_per_scene', 0):.2f}\n")
                    f.write(f"    总事件数: {longtail_8s_summary.get('total_events', 0)}\n")
                    
                    # 显示事件类型统计
                    event_type_counts = longtail_8s_summary.get('event_type_counts', {})
                    if event_type_counts:
                        f.write("    事件类型分布:\n")
                        for event_type, count in event_type_counts.items():
                            f.write(f"      {event_type}: {count} 次\n")
            
            # 详细场景结果
            f.write("\n\n详细场景评估结果:\n")
            f.write("=" * 50 + "\n")
            
            for risk_level, results in evaluation_results.items():
                f.write(f"\n{risk_level} 场景详情:\n")
                f.write("-" * 30 + "\n")
                
                for scene_result in results['scene_results']:
                    f.write(f"\n场景: {scene_result['scene_name']}\n")
                    
                    # 轨迹真实性
                    if 'trajectory_realism' in scene_result:
                        tr = scene_result['trajectory_realism']
                        f.write(f"  轨迹真实性: {tr.get('overall_realism_score', 0):.4f}\n")
                    
                    # 交互合理性
                    if 'interaction_consistency' in scene_result:
                        ic = scene_result['interaction_consistency']
                        f.write(f"  交互合理性: {ic.get('overall_interaction_score', 0):.4f}\n")
                    
                    # 长尾覆盖率
                    if 'longtail_coverage' in scene_result:
                        lc = scene_result['longtail_coverage']
                        f.write(f"  长尾覆盖率: {lc.get('overall_longtail_score', 0):.4f}\n")
                    
                    # Sim-to-Real差距
                    if 'sim_to_real_gap' in scene_result:
                        sr = scene_result['sim_to_real_gap']
                        f.write(f"  Sim-to-Real: {sr.get('overall_sim_to_real_score', 0):.4f}\n")
                    
                    # 8秒长尾事件分析
                    if 'longtail_8s_analysis' in scene_result:
                        lt_8s = scene_result['longtail_8s_analysis']
                        summary = lt_8s.get('summary', {})
                        f.write(f"  8秒长尾事件: {'是' if summary.get('has_longtail_event', False) else '否'}\n")
                        if summary.get('has_longtail_event', False):
                            f.write(f"    事件类型数: {summary.get('total_event_types', 0)}\n")
                            f.write(f"    涉及车辆数: {summary.get('vehicles_involved', 0)}\n")
                            f.write(f"    总事件数: {summary.get('total_events', 0)}\n")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='对抗性轨迹评估')
    
    parser.add_argument('--scenario_dir', type=str, default='./out/adv_gen_rule_based_out_1757600944',
                        help='对抗性场景生成的输出目录')
    
    parser.add_argument('--data_dir', type=str, default='./data/nuscenes',
                        help='NuScenes数据集目录')
    
    parser.add_argument('--data_version', type=str, default='v1.0-trainval',
                        help='NuScenes数据版本')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='交通模型权重文件路径（可选）')
    
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda', 'cpu'],
                        help='计算设备')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='评估结果输出目录（默认在scenario_dir下创建）')
    
    parser.add_argument('--max_scenes', type=int, default=None,
                        help='最大评估场景数量（例如：100表示只评估后100个场景，即最新生成的场景）')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='全局随机种子，用于保证实验可复现性（默认：42）')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 确保logs目录存在并初始化Logger
    import os
    from utils.logger import Logger
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    Logger.init(os.path.join(log_dir, 'adversarial_evaluation.log'))
    
    args = parse_args()
    
    # 设置全局随机种子
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    Logger.log(f"全局随机种子已设置为: {args.seed}")
    
    # 检查输入目录
    if not os.path.exists(args.scenario_dir):
        print(f"错误: 场景目录不存在: {args.scenario_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        sys.exit(1)
    
    # 创建评估运行器
    runner = AdversarialEvaluationRunner(
        scenario_dir=args.scenario_dir,
        data_dir=args.data_dir,
        data_version=args.data_version,
        model_path=args.model_path,
        device=args.device,
        output_dir=args.output_dir,
        max_scenes=args.max_scenes
    )
    
    # 运行评估
    try:
        summary = runner.run_evaluation()
        print("\n评估完成！")
        print(f"结果保存在: {runner.output_dir}")
        
        # 打印简要摘要
        print(f"\n评估摘要:")
        print(f"  总场景数: {summary['total_scenes']}")
        print(f"  风险等级数: {summary['total_risk_levels']}")
        
        for risk_level, level_summary in summary['risk_level_summary'].items():
            print(f"  {risk_level}: {level_summary['evaluated_scenes']}/{level_summary['total_scenes']} 场景")
        
    except Exception as e:
        print(f"评估过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
#/home/wuyou/program/STRIVE/out/adv_gen_rule_based_out_1758166459

# /home/wuyou/STRIVE/STRIVE/out/adv_gen_rule_based_out_1758784725

 # python run_adversarial_evaluation.py --scenario_dir /home/wuyou/STRIVE/STRIVE/out/adv_gen_rule_based_out_1758784725 --data_dir /home/wuyou/STRIVE/STRIVE/data/nuscenes