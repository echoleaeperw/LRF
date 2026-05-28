# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os, time
import gc
import tqdm
import torch
import torch.optim as optim
from torch import nn
import numpy as np

from torch_geometric.data import DataLoader as GraphDataLoader
from torch_geometric.data import Batch as GraphBatch

from datasets import nuscenes_utils as nutils
from models.traffic_model import TrafficModel
from losses.traffic_model import compute_coll_rate_env, compute_coll_rate_veh
from longterm.agents.analysis import AnalysisAgent, LLM_analysis_results

from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from utils.common import dict2obj, mkdir
from utils.logger import Logger, throw_err
from utils.torch import get_device, load_state
from utils.scenario_gen import determine_feasibility_nusc, detach_embed_info
from utils.scenario_gen import viz_optim_results, prepare_output_dict
from utils.scenario_gen import activate_stationary_vehicles, perturb_stationary_z
from utils.adv_gen_optim import run_adv_gen_optim, compute_adv_gen_success
from utils.init_optim import run_init_optim
from utils.risk_classifier import classify_scenario_by_risk_level
from utils.feasibility_reporter import FeasibilityReporter
from planners.planner import PlannerConfig
from utils.config import get_parser, add_base_args
import pdb

# 导入LLM相关模块
import logging
from src.llm.config_loader import ConfigLoader
from src.llm.weight_manager import WeightManager
from src.llm.scenario_extractor import ScenarioExtractor
from src.llm.carla_export import generate_carla_scenario_script

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("adv_scenario_gen")

def parse_cfg():
    parser = get_parser('Adversarial scenario generation')
    parser = add_base_args(parser)

    # data
    parser.add_argument('--split', type=str, default='val',
                        choices=['test', 'val', 'train'],
                        help='Which split of the dataset to find scnarios in')
    parser.add_argument('--val_size', type=int, default=400, help='The size of the validation set used to split the trainval version of the dataset.')
    parser.add_argument('--seq_interval', type=int, default=10, help='skips ahead this many steps from start of current sequence to get the next sequence.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help="Shuffle data")
    parser.set_defaults(shuffle=False)

    parser.add_argument('--adv_attack_with', type=str, default=None,
                        choices=['pedestrian', 'cyclist', 'motorcycle', 'car', 'truck'],
                        help='what to attack with (optional - by default will use any kind of agent)')

    # which planner to attack
    parser.add_argument('--planner', type=str, default='ego',
                        choices=['ego', 'hardcode'],
                        help='Which planner to attack. ego is will use ego motion from nuscenes dataset (i.e. the replay planner).')
    parser.add_argument('--planner_cfg', type=str, default='default',
                        help='hyperparameter configuration to use for the planner (if relevant)')

    # determining feasibility
    parser.add_argument('--feasibility_thresh', type=float, default=10.0, help='Future samples for target must be within this many meters from another agent for the initialization scenario to be feasible.')
    parser.add_argument('--feasibility_time', type=int, default=4, help='For feasibility, only consider timesteps >= feasibility_time, i.e., do not try to crash at timestep 0.')
    parser.add_argument('--feasibility_vel', type=float, default=0.5, help='maximum velocity (delta position of one timestep) of sampled trajectory for an agent must be >= this thresh to be considered feasible')
    parser.add_argument('--feasibility_infront_min', type=float, default=0.0, help='threshold for how in-front-of the ego vehicle the attacker is (measured by cosine similarity).')
    parser.add_argument('--feasibility_check_sep', dest='feasibility_check_sep', action='store_true',
                        help="If given, ensures attacker and target on not separated by non-drivable area.")
    parser.set_defaults(feasibility_check_sep=False)

    # optimizer & losses
    # initialization optimization
    parser.add_argument('--init_loss_match_ext', type=float, default=10.0, help='Match initial trajectory from nuScenes data.')
    parser.add_argument('--init_loss_motion_prior_ext', type=float, default=0.1, help='Keep latent z likely under the traffic model prior.')
    # adversarial optimization
    parser.add_argument('--loss_coll_veh', type=float, default=20.0, help='Loss to avoid vehicle-vehicle collisions between non-planner agents.')
    parser.add_argument('--loss_coll_veh_plan', type=float, default=20.0, help='Loss to avoid collisions between the planner and unlikely adversaries.')
    parser.add_argument('--loss_coll_env', type=float, default=20.0, help='Loss to avoid vehicle-environment collisions for non-planner agents.')
    parser.add_argument('--loss_init_z', type=float, default=0.5, help='Loss to keep latent z near init for unlikely adversaries (i.e. the MAX weight of init loss).')
    parser.add_argument('--loss_init_z_atk', type=float, default=0.05, help='Loss to keep latent z near init for likely adversaries (i.e. the MIN weight of init loss).')
    parser.add_argument('--loss_motion_prior', type=float, default=1.0, help='Loss to keep latent z likely under motion prior for unlikely adversaries (i.e. the MAX weight of prior loss).')
    parser.add_argument('--loss_motion_prior_atk', type=float, default=0.005, help='Loss to keep latent z likely under motion prior for likely adversaries (i.e. the MIN weight of prior loss).')
    parser.add_argument('--loss_motion_prior_ext', type=float, default=0.0001, help='Loss to keep latent z likely under motion prior for the planner.')
    parser.add_argument('--loss_match_ext', type=float, default=10.0, help='Match predicted planner trajectory to true planner rollout.')
    parser.add_argument('--loss_adv_crash', type=float, default=2.0, help='Minimize distance between planner and adversaries.')

    parser.add_argument('--loss_ttc', type=float, default=1.5, help='Time-to-Collision损失权重，惩罚与前车碰撞时间过短的行为。')
    parser.add_argument('--loss_thw', type=float, default=0.5,
                        help='Time Headway 损失权重，与 TTC 互补：速度相近时 TTC 无梯度，THW 仍有效。'
                             '纵向急刹场景建议 2.0–3.5，横向/路口场景建议 0.5–1.0。0 表示禁用。')
    parser.add_argument('--loss_min_dist_lat', type=float, default=1.0, help='横向最小距离损失权重，惩罚与其他车辆横向距离过近的行为。')
    parser.add_argument('--loss_yaw_rate', type=float, default=0.8, help='横摆角速度损失权重，惩罚剧烈、不稳定的转向动作。')
    parser.add_argument('--loss_yaw_rate_ego', type=float, default=0.5, help='目标车辆的横摆角速度损失权重。')
    parser.add_argument('--loss_yaw_rate_non_ego', type=float, default=1.0, help='非目标车辆的横摆角速度损失权重。')
    # 运动学可行性软约束（防止极端加速度/速度/Jerk 产生"捷径解"）
    parser.add_argument('--loss_kine_veh', type=float, default=0.5,
                        help='背景车辆运动学可行性损失权重（速度≤20m/s, 加速度≤5m/s², Jerk≤10m/s³）。0表示禁用。')
    parser.add_argument('--loss_kine_atk', type=float, default=0.1,
                        help='攻击车辆运动学可行性损失权重（放宽上限：速度≤30m/s, 加速度≤9m/s², Jerk≤18m/s³）。0表示禁用。')
    parser.add_argument('--num_iters', type=int, default=300, help='Number of optimization iterations.')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate for adam.')

    parser.add_argument('--viz', dest='viz', action='store_true',
                        help="If given, saves low-quality visualization before and after optimization.")
    parser.set_defaults(viz=False)
    parser.add_argument('--save', dest='save', action='store_true',
                        help="If given, saves the scenarios as json so they can be used later.")
    parser.set_defaults(save=False)
    
    # LLM权重相关配置
    parser.add_argument('--use_llm', dest='use_llm', type=lambda x: str(x).lower() != 'false',
                        default=True,
                        help="是否使用LLM生成的动态权重 (True/False，可在cfg文件中覆盖)")
    parser.add_argument('--llm_config_path', type=str, default='configs/llm_weights_config.yaml',
                        help='LLM权重配置文件路径')
    parser.add_argument('--llm_cache_dir', type=str, default=None,
                        help='LLM权重缓存目录（None=禁用缓存，每次场景都调用LLM）')
    parser.add_argument('--llm_model', type=str, default='Pro/zai-org/GLM-5',
                        help='使用的LLM模型名称 (gpt-4o, gpt-3.5-turbo, deepseek-chat, deepseek-reasoner)')

    # Ablation / comparison: control which LLM components are used
    parser.add_argument('--ablation_mode', type=str, default='full',
                        choices=['full', 'no_llm', 'no_attacker_sel', 'no_weight_adapt',
                                 'aggressive_fixed', 'grid_search'],
                        help='Ablation: full=Ours; no_llm=STRIVE fixed weights; '
                             'no_attacker_sel=LLM weights but no attacker selection; '
                             'no_weight_adapt=LLM attacker but fixed weights; '
                             'aggressive_fixed=max-aggressive hand-tuned weights (upper-bound fixed-weight baseline); '
                             'grid_search=best weights from exhaustive grid search without LLM')

    parser.add_argument('--viz_timestamps', type=str, default="1.0,3.0,5.0",
                        help='需要保存可视化帧的特定时间戳（以逗号分隔，例如 "1.0,2.5,4.0"）')

    #这都消融实验需要做的参数
    parser.add_argument('--include_field_visualization', dest='include_field_visualization', action='store_true',
                        help='是否生成势场可视化并传递给LLM分析')

    # 添加CARLA场景生成相关参数
    parser.add_argument('--extract_latent', dest='extract_latent', action='store_true',
                        help="如果指定，将提取潜变量并解码为完整场景")
    parser.set_defaults(extract_latent=False)
    parser.add_argument('--generate_carla', dest='generate_carla', action='store_true',
                        help="如果指定，将生成可供CARLA使用的场景脚本")
    parser.set_defaults(generate_carla=False)
    parser.add_argument('--carla_output_dir', type=str, default='carla_scenarios',
                        help='CARLA场景输出目录')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Global random seed for reproducibility.')

    # Early stopping
    parser.add_argument('--max_scenes', type=int, default=0,
                        help='若>0，当已保存成功场景数量达到此值时提前终止（用于可重复性实验批量运行）。0=不限制。')

    # 添加风险等级参数
    parser.add_argument('--risk_level', type=str, default='longtail_condition',
                        choices=['low_risk', 'high_risk', 'longtail_condition'],
                        help='目标风险等级：low_risk(低风险), high_risk(高风险), longtail_condition(长尾条件)')

    # 添加可视化帧保存参数
    parser.add_argument('--keep_viz_frames', dest='keep_viz_frames', action='store_true',
                        help="如果指定，将保存用于创建可视化视频的单个帧")
    parser.set_defaults(keep_viz_frames=False)
    
    # 添加静止车辆显示控制参数
    parser.add_argument('--mark_stationary', dest='mark_stationary', action='store_true',
                        help="如果指定，将用灰色和低透明度标记静止/停放的车辆")
    parser.set_defaults(mark_stationary=True)
    
    # 添加动态车辆数量筛选参数
    parser.add_argument('--min_moving_vehicles', type=int, default=0,
                        help="场景中最少需要的移动车辆数量（不包括ego）。设为0禁用筛选。默认为0。")
    parser.add_argument('--stationary_velocity_threshold', type=float, default=0.5,
                        help="用于判断车辆是否静止的速度阈值(m/s)。速度低于此值的车辆被视为静止。")
    
    # 添加激活静止车辆参数
    parser.add_argument('--no_activate_stationary', dest='activate_stationary', action='store_false',
                        help="如果指定，将禁用静止车辆激活功能（默认开启）")
    parser.set_defaults(activate_stationary=True)
    parser.add_argument('--activate_min_speed', type=float, default=3.0,
                        help="激活静止车辆的最小速度 (m/s)")
    parser.add_argument('--activate_max_speed', type=float, default=3.0,
                        help="激活静止车辆的最大速度 (m/s)")

    args = parser.parse_args()
    config_dict = vars(args)
    # Config dict to object
    config = dict2obj(config_dict)
    
    return config, config_dict

def run_one_epoch(data_loader, batch_size, model, map_env, device, out_path, loss_weights,
                  planner_name=None,
                  planner_cfg='default',
                  feasibility_thresh=10.0,
                  feasibility_time=4,
                  feasibility_vel=0.5,
                  feasibility_infront_min=0.0,
                  feasibility_check_sep=True,
                  num_iters=300,
                  lr=0.05,
                  viz=True,
                  save=True,
                  adv_attack_with=None,
                  weight_manager=None,
                  config=None,
                  ablation_mode='full',
                  scenario_description: str = "",
                  feasibility_reporter: "FeasibilityReporter | None" = None,
                  max_scenes: int = 0,
                  ):
    '''
    Run through dataset and find possible scenarios.
    
    参数:
        weight_manager: 权重管理器，如果不为None，则使用动态权重
    '''
    pbar_data = tqdm.tqdm(data_loader)

    gen_out_path = out_path
    mkdir(gen_out_path)
    if viz:
        gen_out_path_viz = os.path.join(gen_out_path, 'viz_results')
        mkdir(gen_out_path_viz)
    if save:
        gen_out_path_scenes = os.path.join(gen_out_path, 'scenario_results')
        mkdir(gen_out_path_scenes)

    # 可行性报告输出目录（与 scenario_results 同级）
    gen_out_path_feasibility = os.path.join(gen_out_path, 'feasibility_reports')
    if feasibility_reporter is not None and save:
        mkdir(gen_out_path_feasibility)

    # 如果需要生成CARLA场景，创建输出目录
    if config and hasattr(config, 'generate_carla') and config.generate_carla:
        carla_output_dir = os.path.join(gen_out_path, config.carla_output_dir)
        mkdir(carla_output_dir)
        Logger.log(f'创建CARLA场景输出目录: {carla_output_dir}')

    data_idx = 0
    empty_cache = False
    batch_i = []
    batch_scene_graph = []
    batch_map_idx = []
    batch_total_NA = 0
    batch_attacker_ids = []
    _scenes_saved = 0  # 已保存场景计数，用于 max_scenes 早停

    # 计时统计累计量
    _time_stats = {
        'scene_count':    0,   # 已处理场景数
        'total':          0.0, # 总耗时（含跳过场景）
        'llm':            0.0, # LLM 调用耗时（纯 LLM pipeline，不含 model.sample 和可视化）
        'llm_prep':       0.0, # LLM 前置准备耗时（model.sample_batched + 势场可视化）
        'init_optim':     0.0, # init_optim 耗时
        'adv_optim':      0.0, # run_adv_gen_optim 耗时
        'feasibility':    0.0, # 可行性检测耗时
    }

    for i, data in enumerate(pbar_data):
        _t_scene_start = time.time()
        sample_pred = None
        scene_graph, map_idx = data
        if empty_cache:
            empty_cache = False
            gc.collect()
            torch.cuda.empty_cache()
        try:
            scene_graph = scene_graph.to(device)
            map_idx = map_idx.to(device)
            is_last_batch = i == (len(data_loader)-1)
            _llm_elapsed = _llm_prep_elapsed = _init_elapsed = _adv_elapsed = 0.0

            _maps = [map_env.map_list[map_idx[b]] for b in range(map_idx.size(0))]
            _na   = scene_graph.past.size(0)
            Logger.log(f'┌─ Scene {i:04d}  agents={_na}  map={_maps[0]}')

            # 如果使用动态权重，根据场景更新权重
            current_weights = loss_weights
            if weight_manager is not None:
                try:
                    _t_llm_prep_start = time.time()
                    Logger.log('│  [1/3] LLM 权重生成...')

                    # 采样一次以获得未来轨迹预测，用于场景描述
                    with torch.no_grad():
                        future_pred_sample = model.sample_batched(scene_graph, map_idx, map_env, 1, include_mean=True)

                    # 生成势场可视化（如果配置启用）
                    field_info = None
                    if getattr(config, 'include_field_visualization', False):
                        try:
                            import base64
                            import sys
                            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                            from field_model_direct_prediction import DirectPredictionQuadrantFieldModel

                            quadrant_field_model = DirectPredictionQuadrantFieldModel(
                                model=model, scene_graph=scene_graph,
                                map_idx=map_idx, map_env=map_env,
                                dt=data_loader.dataset.dt
                            )
                            max_time = (quadrant_field_model.future_pred.shape[1] - 1) * data_loader.dataset.dt
                            t_key = min(2.0, max_time)
                            field_out_dir = os.path.join(gen_out_path, f'scene_{i}', 'field_analysis')
                            os.makedirs(field_out_dir, exist_ok=True)

                            quadrant_img_path = os.path.join(field_out_dir, f'quadrant2_vehicles_t_{t_key:.1f}s.png')
                            quadrant_field_model.visualize_vehicles_in_quadrant2(time_point=t_key, save_path=quadrant_img_path, figsize=(12, 10))
                            quadrant_field_path = os.path.join(field_out_dir, f'quadrant2_field_t_{t_key:.1f}s.png')
                            quadrant_field_model.visualize_quadrant2_field_at_time(time_point=t_key, save_path=quadrant_field_path, figsize=(14, 10))
                            quadrant_traj_path = os.path.join(field_out_dir, f'quadrant2_trajectory.png')
                            quadrant_field_model.visualize_trajectory_in_quadrant2(save_path=quadrant_traj_path, figsize=(14, 10))

                            try:
                                with open(quadrant_img_path, 'rb') as img_file:
                                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                                field_info = {
                                    'has_image': True, 'image_base64': img_base64,
                                    'image_format': 'png', 'time_point': t_key,
                                    'visualization_type': 'quadrant2_vehicles',
                                    'statistics': {
                                        'num_vehicles': quadrant_field_model.num_agents,
                                        'coordinate_system': 'quadrant2 (x<0, y>0)',
                                        'time_steps': quadrant_field_model.future_pred.shape[1],
                                    }
                                }
                                Logger.log(f'│      势场可视化已生成 (t={t_key:.1f}s)')
                            except Exception as e:
                                field_info = {'has_image': False, 'visualization_type': 'quadrant2_vehicles', 'error': str(e)}
                                Logger.log(f'│      势场图片编码失败: {e}')
                        except Exception as e:
                            Logger.log(f'│      势场可视化生成失败: {e}')
                            field_info = None

                    _llm_prep_elapsed = time.time() - _t_llm_prep_start
                    _time_stats['llm_prep'] += _llm_prep_elapsed

                    # 更新权重（这里才是真正的 LLM 调用）
                    _t_llm_start = time.time()
                    current_weights = weight_manager.update_from_scenario(
                        scene_graph=scene_graph, map_env=map_env, map_idx=map_idx,
                        future_pred=future_pred_sample.get('future_pred', None),
                        past_traj=scene_graph.past,
                        driving_objectives=f"场景{i}的对抗性驾驶行为分析",
                        extra_context=field_info,
                        risk_level=config.risk_level
                    )

                    # Ablation: no_weight_adapt = use LLM for attacker only, keep fixed weights
                    if ablation_mode == 'no_weight_adapt':
                        current_weights = loss_weights

                    # 记录关键权重变化
                    important_keys = ['adv_crash', 'ttc', 'min_dist_lat', 'yaw_rate', 'motion_prior_atk', 'init_z_atk']
                    changed = [(k, loss_weights.get(k), current_weights.get(k))
                               for k in important_keys
                               if loss_weights.get(k) != current_weights.get(k)]
                    if changed:
                        changes_str = '  '.join(f'{k}: {o}→{n}' for k, o, n in changed)
                        Logger.log(f'│      权重更新: {changes_str}')

                    _llm_elapsed = time.time() - _t_llm_start
                    _time_stats['llm'] += _llm_elapsed
                    Logger.log(f'│      完成 (prep={_llm_prep_elapsed:.1f}s  llm={_llm_elapsed:.1f}s)')

                except Exception as e:
                    try:
                        _llm_elapsed = time.time() - _t_llm_start
                        _time_stats['llm'] += _llm_elapsed
                    except NameError:
                        pass
                    import traceback
                    Logger.log(f'│      LLM失败 ({type(e).__name__}: {e})，回退静态权重')
                    Logger.log(f'│      {traceback.format_exc().splitlines()[-1]}')
                    current_weights = loss_weights

            # First sample prior to get possible futures
            with torch.no_grad():
                sample_pred = model.sample_batched(scene_graph, map_idx, map_env, 20, include_mean=True)
                # sample_pred = model.sample(scene_graph, map_idx, map_env, 20, include_mean=True)

            empty_cache = True
            # determine if this sequence is feasible for scenario generation
            _t_feas_start = time.time()
            feasible, feasible_time, feasible_dist = determine_feasibility_nusc(sample_pred['future_pred'],
                                                                                model.get_normalizer(),
                                                                                feasibility_thresh,
                                                                                feasibility_time,
                                                                                0.0,
                                                                                feasibility_infront_min=feasibility_infront_min,
                                                                                check_non_drivable_separation=feasibility_check_sep,
                                                                                map_env=map_env,
                                                                                map_idx=map_idx)

            if planner_name == 'ego':
                ego_gt = model.get_normalizer().unnormalize(scene_graph.future_gt[0])
                ego_vels = torch.norm(ego_gt[1:, :2] - ego_gt[:-1, :2], dim=-1)
                max_vel = torch.max(ego_vels).cpu().item()
                if max_vel < feasibility_vel:
                    Logger.log('│  跳过: ego速度过低')
                    if not is_last_batch:
                        continue
            elif planner_name == 'hardcode':
                ego_samps = model.get_normalizer().unnormalize(sample_pred['future_pred'][0].detach())
                ego_vels = torch.norm(ego_samps[:, 1:, :2] - ego_samps[:, :-1, :2], dim=-1)
                max_vel = torch.max(ego_vels).cpu().item()
                if max_vel < feasibility_vel:
                    Logger.log('│  跳过: ego采样速度过低')
                    if not is_last_batch:
                        continue

            _time_stats['feasibility'] += time.time() - _t_feas_start

            if feasible is None:
                Logger.log('│  跳过: 场景中只有ego')
                if not is_last_batch:
                    continue
            elif torch.sum(feasible).item() == 0:
                Logger.log('│  跳过: 附近无可行攻击者')
                if not is_last_batch:
                    continue

            # 检查场景中移动车辆的数量（可选筛选）
            min_moving = getattr(config, 'min_moving_vehicles', 0)
            stat_vel_thresh = getattr(config, 'stationary_velocity_threshold', 0.5)
            if min_moving > 0:
                dt = data_loader.dataset.dt
                future_pred_mean = sample_pred['future_pred'][:, 0, :, :]
                future_pred_unnorm = model.get_normalizer().unnormalize(future_pred_mean)
                stationary_vehicles = nutils.detect_stationary_vehicles(
                    future_pred_unnorm, velocity_threshold=stat_vel_thresh, dt=dt
                )
                NA_scene = future_pred_unnorm.size(0)
                moving_count = NA_scene - len(stationary_vehicles) - 1
                if moving_count < min_moving:
                    Logger.log(f'│  跳过: 仅 {moving_count} 辆移动车辆 < {min_moving}')
                    if not is_last_batch:
                        continue

            is_feas = False
            if feasible is not None and torch.sum(feasible).item() > 0:
                is_feas = True
                if adv_attack_with is not None:
                    feas_sem = scene_graph.sem[1:]
                    veclist = [tuple(feas_sem[aidx].to(int).cpu().numpy().tolist()) for aidx in range(feas_sem.size(0))]
                    is_adv_atk = [data_loader.dataset.vec2cat[curvec] == adv_attack_with for curvec in veclist]
                    adv_atk_feas = torch.zeros_like(feasible)
                    adv_atk_feas[is_adv_atk] = True
                    feasible = torch.logical_and(feasible, adv_atk_feas)
                    if torch.sum(feasible) == 0:
                        Logger.log('│  跳过: 无指定类别的可行攻击者')
                        is_feas = False

                if is_feas:
                    feasible_dist[~feasible] = float('inf')
                    temp_attack_agt = torch.min(feasible_dist, dim=0)[1] + 1

            # This is a feasible seed, add it to the batch
            if is_feas:
                batch_scene_graph += scene_graph.to_data_list()
                batch_map_idx.append(map_idx)
                batch_i.append(i)
                batch_total_NA += scene_graph.future_gt.size(0)
                # 记录当前场景 LLM 攻击者局部索引
                _cur_atk_id = None
                if weight_manager is not None and hasattr(weight_manager, 'attacker_vehicle_id'):
                    _cur_atk_id = weight_manager.attacker_vehicle_id
                batch_attacker_ids.append(_cur_atk_id)
                Logger.log(f'│  可行 (batch_NA={batch_total_NA}，候选攻击者=agt{int(temp_attack_agt)})')

            if batch_total_NA < batch_size and not is_last_batch:
                # collect more before performing optim
                continue
            else:
                if len(batch_scene_graph) == 0:
                    # this is the last seq in dataset, and we have no other seqs queueued
                    continue
                # create the batch
                scene_graph = GraphBatch.from_data_list(batch_scene_graph)
                map_idx = torch.cat(batch_map_idx, dim=0)
                cur_batch_i = batch_i
                cur_batch_attacker_ids = batch_attacker_ids  # 每个场景各自的攻击者 ID

                Logger.log(f'│  批次已构建 (B={len(cur_batch_i)}, NA={scene_graph.past.size(0)})')

                # 将dt添加到scene_graph
                scene_graph.dt = data_loader.dataset.dt

                # reset
                batch_scene_graph = []
                batch_map_idx = []
                batch_i = []
                batch_total_NA = 0
                batch_attacker_ids = []

            B = map_idx.size(0)
            NA = scene_graph.past.size(0)
            ego_inds = scene_graph.ptr[:-1]
            ego_mask = torch.zeros((NA), dtype=torch.bool)
            ego_mask[ego_inds] = True
            
            #
            # Initialize optimization
            #
            # embed past and map to get inputs to decoder used during optim
            with torch.no_grad():
                embed_info_attached = model.embed(scene_graph, map_idx, map_env)
            # need to detach all the encoder outputs from current comp graph to be used in optimization
            embed_info = detach_embed_info(embed_info_attached)

            init_future_pred = init_traj = z_init = init_coll_env = None

            planner = plan_out_path = None
            if planner_name == 'hardcode':
                from planners.hardcode_goalcond_nusc import HardcodeNuscPlanner, CONFIG_DICT
                assert(planner_cfg in CONFIG_DICT)
                planner = HardcodeNuscPlanner(map_env, PlannerConfig(**CONFIG_DICT[planner_cfg]))

            # start from GT scene future (reconstructed with motion model)
            z_init = embed_info_attached['posterior_out'][0].detach()
            init_traj = scene_graph.future_gt[:, :, :4].clone().detach()
            Logger.log('│  [2/3] 初始化优化...')
            
            # 激活静止车辆（如果启用）
            activate_stationary = getattr(config, 'activate_stationary', False)
            if activate_stationary:
                dt = data_loader.dataset.dt
                stat_vel_thresh = getattr(config, 'stationary_velocity_threshold', 0.5)
                activate_min_speed = getattr(config, 'activate_min_speed', 2.0)
                activate_max_speed = getattr(config, 'activate_max_speed', 5.0)
                
                init_traj, activated_vehicles = activate_stationary_vehicles(
                    init_traj, scene_graph, model, 
                    dt=dt,
                    velocity_threshold=stat_vel_thresh,
                    min_speed=activate_min_speed,
                    max_speed=activate_max_speed,
                    ego_mask=ego_mask
                )
                
                # 同时给静止车辆的 z 添加扰动
                if activated_vehicles:
                    z_init = perturb_stationary_z(z_init, activated_vehicles, perturbation_scale=0.5)

            # run initial optimization to closely fit nuscenes scene
            _t_init_start = time.time()
            z_init, init_fit_traj, _ = run_init_optim(z_init, init_traj, scene_graph.future_vis, 0.1, loss_weights, model,
                                                      scene_graph, map_env, map_idx, 75, embed_info, embed_info['prior_out'])
            # if we're using a specific planner, replace ego with planner rollout
            if planner_name == 'hardcode':
                # reset planner
                all_init_state = model.get_normalizer().unnormalize(scene_graph.past_gt[:, -1, :])
                all_init_veh_att = model.get_att_normalizer().unnormalize(scene_graph.lw)
                planner.reset(all_init_state, all_init_veh_att, scene_graph.batch, B, map_idx)
                # rollout
                init_non_ego = model.normalizer.unnormalize(init_fit_traj[~ego_mask]).cpu().numpy()
                plan_t = np.linspace(model.dt, model.dt*model.FT, model.FT)
                init_agt_ptr = scene_graph.ptr - torch.arange(B+1, device=scene_graph.ptr.device)
                planner_init = planner.rollout(init_non_ego, plan_t, init_agt_ptr.cpu().numpy(), plan_t,
                                                control_all=False).to(scene_graph.future_gt)
                planner_init = model.get_normalizer().normalize(planner_init)
                # replace init traj ego's with planner traj
                init_traj[ego_mask] = planner_init

                # and optim a bit more, now to match the planner traj
                z_init, init_fit_traj, _ = run_init_optim(z_init, init_traj, scene_graph.future_vis, lr, loss_weights, model,
                                                            scene_graph, map_env, map_idx, 100, embed_info, embed_info['prior_out'])

                # check if planner collides with scene trajectories already. if so, not worth continuing
                from losses.adv_gen_nusc import check_single_veh_coll
                bvalid = []
                for b in range(B):
                    init_hardcode_coll, _ = check_single_veh_coll(model.get_normalizer().unnormalize(init_fit_traj[scene_graph.ptr[b]]),
                                                                    model.get_att_normalizer().unnormalize(scene_graph.lw[scene_graph.ptr[b]]),
                                                                    model.get_normalizer().unnormalize(init_fit_traj[(scene_graph.ptr[b]+1):scene_graph.ptr[b+1]]),
                                                                    model.get_att_normalizer().unnormalize(scene_graph.lw[(scene_graph.ptr[b]+1):scene_graph.ptr[b+1]])
                                                                    )
                    bvalid.append(np.sum(init_hardcode_coll) == 0)

                bvalid = np.array(bvalid, dtype=bool)
                if np.sum(bvalid) < B:
                    Logger.log(f'│      Planner碰撞过滤: {B-np.sum(bvalid)}/{B} 场景无效')
                    if np.sum(bvalid) == 0:
                        Logger.log('│      跳过: 批次中无有效场景')
                        continue
                    # need to remove invalid scenarios from batch
                    # rebuild and reset all necessary variables
                    map_idx = map_idx[bvalid]
                    cur_batch_i = [bi for b, bi in enumerate(cur_batch_i) if bvalid[b]]
                    cur_batch_attacker_ids = [aid for b, aid in enumerate(cur_batch_attacker_ids) if bvalid[b]]

                    avalid = np.zeros((NA), dtype=bool) # which agents are part of new graphs
                    for b in range(B):
                        if bvalid[b]:
                            avalid[scene_graph.ptr[b]:scene_graph.ptr[b+1]] = True

                    z_init = z_init[avalid]
                    init_traj = init_traj[avalid]

                    init_batch_data_list = scene_graph.to_data_list()
                    init_batch_data_list = [g for b, g, in enumerate(init_batch_data_list) if bvalid[b]]
                    scene_graph = GraphBatch.from_data_list(init_batch_data_list)

                    B = map_idx.size(0)
                    NA = scene_graph.past.size(0)
                    ego_inds = scene_graph.ptr[:-1]
                    ego_mask = torch.zeros((NA), dtype=torch.bool)
                    ego_mask[ego_inds] = True

                    with torch.no_grad():
                        embed_info_attached = model.embed(scene_graph, map_idx, map_env)
                    embed_info = detach_embed_info(embed_info_attached)

            with torch.no_grad():
                init_future_pred = model.decode_embedding(z_init, embed_info_attached, scene_graph, map_idx, map_env)['future_pred'].detach()
                init_coll_env_dict = compute_coll_rate_env(scene_graph, map_idx, init_future_pred.unsqueeze(1).contiguous(),
                                                    map_env, model.get_normalizer(), model.get_att_normalizer(),
                                                    ego_only=False)
                init_coll_env = init_coll_env_dict['did_collide'].cpu().numpy()[:, 0] # NA

                # make sure ego is actual data or planner rollout - not our initial fitting
                init_future_pred[ego_mask] = init_traj[ego_mask]

            # 计算初始轨迹的势场信息（用于后续优化参考）
            field_guidance = None
            if getattr(config, 'use_field_guidance', False):
                try:
                    pass  # 接口预留，实际势场引导待实现
                except Exception as e:
                    field_guidance = None

            if planner_name == 'hardcode':
                plan_out_path = None
                if viz:
                    plan_out_path = os.path.join(gen_out_path_viz, 'planner_out')
                    cur_seq_str = 'sample_' + '_'.join(['%03d' for b in range(len(cur_batch_i))]) % tuple([cur_batch_i[b] for b in range(len(cur_batch_i))])
                    plan_out_path = os.path.join(plan_out_path, cur_seq_str)
                    mkdir(plan_out_path)

            _init_elapsed = time.time() - _t_init_start
            _time_stats['init_optim'] += _init_elapsed
            Logger.log(f'│      完成 ({_init_elapsed:.1f}s)')

            # adversarial optimization
            cur_z = z_init.clone().detach()
            tgt_prior_distrib = (embed_info['prior_out'][0][ego_mask], embed_info['prior_out'][1][ego_mask])
            other_prior_distrib = (embed_info['prior_out'][0][~ego_mask], embed_info['prior_out'][1][~ego_mask])

            # 从LLM分析结果中获取每个场景各自的攻击车辆ID
            attack_agt_idx = None
            if weight_manager is not None:
                try:
                    scene_sizes = (scene_graph.ptr[1:] - scene_graph.ptr[:-1]).tolist()
                    per_scene_ids = []
                    all_valid = True
                    for b in range(B):
                        vid = cur_batch_attacker_ids[b] if b < len(cur_batch_attacker_ids) else None
                        if vid is None or vid <= 0:
                            all_valid = False
                            break
                        if vid >= scene_sizes[b]:
                            Logger.log(f'│  WARNING: 场景{b}攻击者索引 vehicle_{vid} >= 场景大小 {scene_sizes[b]}，回退soft-min')
                            all_valid = False
                            break
                        per_scene_ids.append(vid)
                    if all_valid and len(per_scene_ids) == B:
                        attack_agt_idx = per_scene_ids
                except Exception:
                    attack_agt_idx = None
            # Ablation: no_attacker_sel = use LLM weights but do not fix attacker (soft-min over all)
            if ablation_mode == 'no_attacker_sel':
                attack_agt_idx = None

            _atk_str = f'agt{attack_agt_idx}' if attack_agt_idx else '自动'
            Logger.log(f'│  [3/3] 对抗优化  (攻击者={_atk_str})...')
            _t_adv_start = time.time()
            adv_gen_out = run_adv_gen_optim(cur_z, lr, current_weights, model, scene_graph, map_env, map_idx,
                                            num_iters, embed_info, 
                                            planner_name, tgt_prior_distrib, other_prior_distrib,
                                            feasibility_time, feasibility_infront_min,
                                            planner=planner,
                                            planner_viz_out=plan_out_path,
                                            attack_agt_idx=attack_agt_idx,
                                            # 添加新的参数
                                            ttc_epsilon=1e-6,
                                            dt=data_loader.dataset.dt,  # 使用数据集的实际时间步长
                                            ttc_safe=3.0,
                                            min_dist_lat_k=2.0,
                                            min_dist_lat_gap=0.5,
                                            yaw_rate_threshold=15.0)
            cur_z, final_result_traj, final_decoder_out, cur_min_agt, cur_min_t = adv_gen_out
            _adv_elapsed = time.time() - _t_adv_start
            _time_stats['adv_optim'] += _adv_elapsed
            attack_agt = cur_min_agt
            attack_t = cur_min_t

            adv_succeeded = []
            if attack_agt is not None:
                other_ptr = scene_graph.ptr - torch.arange(len(scene_graph.ptr), device=scene_graph.ptr.device)
                for b in range(B):
                    cur_adv_succeeded = compute_adv_gen_success(final_result_traj[scene_graph.ptr[b]:scene_graph.ptr[b+1]],
                                                        model,
                                                        GraphBatch.from_data_list([scene_graph.to_data_list()[b]]),
                                                        attack_agt[b] - scene_graph.ptr[b].item())
                    adv_succeeded.append(cur_adv_succeeded)
            else:
                adv_succeeded = [False] * B

            # ── 后验可行性报告（每个 batch 元素单独报告）───────────────────────
            if feasibility_reporter is not None:
                dt_val = data_loader.dataset.dt if hasattr(data_loader.dataset, 'dt') else 0.5
                for b in range(B):
                    try:
                        b_start = scene_graph.ptr[b].item()
                        b_end   = scene_graph.ptr[b + 1].item()
                        # unnormalize trajectory to real-world coordinates before feasibility check
                        b_traj_norm = final_result_traj[b_start:b_end, 0]  # [NA_b, T, 4] normalized
                        b_traj = model.get_normalizer().unnormalize(b_traj_norm).detach().cpu()
                        b_ego_mask = torch.zeros(b_end - b_start, dtype=torch.bool)
                        b_ego_mask[0] = True  # ego 总是第一个
                        b_atk_idx = (attack_agt[b] - scene_graph.ptr[b].item()
                                     if attack_agt is not None and attack_agt[b] >= 0 else None)
                        # unnormalize vehicle length/width to real-world meters
                        b_lw = model.get_att_normalizer().unnormalize(
                            scene_graph.x[b_start:b_end, :2]
                        ).detach().cpu()
                        scene_id = f"batch{cur_batch_i[b]:04d}"
                        rpt = feasibility_reporter.report(
                            scene_id=scene_id,
                            traj=b_traj,
                            dt=dt_val,
                            ego_mask=b_ego_mask,
                            attacker_idx=b_atk_idx,
                            lw=b_lw,
                        )
                        Logger.log(
                            f'  [feasibility] {rpt.summary}'
                        )
                        if save:
                            rpt_path = os.path.join(
                                gen_out_path_feasibility,
                                f'feasibility_{scene_id}.json'
                            )
                            feasibility_reporter.save_report(rpt, rpt_path)
                    except Exception as _fe:
                        logger.warning(f'FeasibilityReporter failed for batch {b}: {_fe}')

            _adv_succ_cnt = sum(adv_succeeded)
            Logger.log(f'│      完成 ({_adv_elapsed:.1f}s)  碰撞: {_adv_succ_cnt}/{B}')

            _scene_elapsed = time.time() - _t_scene_start
            _time_stats['total'] += _scene_elapsed
            _time_stats['scene_count'] += 1
            n = _time_stats['scene_count']
            Logger.log(f'└─ 耗时 {_scene_elapsed:.1f}s  '
                       f'[init={_init_elapsed:.0f}s  adv={_adv_elapsed:.0f}s'
                       f'{f"  llm_prep={_llm_prep_elapsed:.0f}s  llm={_llm_elapsed:.0f}s" if weight_manager is not None else ""}]')

            # 每10个场景打印一次汇总
            if n % 10 == 0:
                _tot = _time_stats['total']
                Logger.log(f'{"─"*55}')
                Logger.log(f'  [汇总 {n} 场景]  均值: 总={_tot/n:.1f}s  '
                           f'LLM_prep={_time_stats["llm_prep"]/n:.1f}s  '
                           f'LLM={_time_stats["llm"]/n:.1f}s({100*_time_stats["llm"]/_tot:.0f}%)  '
                           f'init={_time_stats["init_optim"]/n:.1f}s  '
                           f'adv={_time_stats["adv_optim"]/n:.1f}s')
                Logger.log(f'{"─"*55}')

            # output scenario and viz
            scene_graph_list = scene_graph.to_data_list()
            for b in range(B):
                # 使用基于风险等级的分类替代原有的攻击成功/失败分类
                result_dir = None

                cur_attack_agt = attack_agt[b] - scene_graph.ptr[b].item() if attack_agt is not None else 0 # make index local to each batch idx
                cur_attack_t = attack_t[b] if attack_t is not None else 0

                if save:
                    import json
                    # save scenario
                    scene_out_dict = prepare_output_dict(scene_graph_list[b], map_idx[b].item(), map_env, data_loader.dataset.dt, model,
                                                          init_future_pred[scene_graph.ptr[b]:scene_graph.ptr[b+1]],
                                                          final_result_traj[scene_graph.ptr[b]:scene_graph.ptr[b+1]][:,0],
                                                          cur_attack_agt,
                                                          cur_attack_t,
                                                          cur_z[scene_graph.ptr[b]:scene_graph.ptr[b+1]],
                                                          (embed_info['prior_out'][0][scene_graph.ptr[b]:scene_graph.ptr[b+1]], embed_info['prior_out'][1][scene_graph.ptr[b]:scene_graph.ptr[b+1]]),
                                                          internal_ego_traj=final_decoder_out['future_pred'][scene_graph.ptr[b]].detach()
                                                          )

                    # 使用风险分类器对场景进行分类
                    try:
                        result_dir = classify_scenario_by_risk_level(scene_out_dict, scene_graph_list[b])
                    except Exception as e:
                        result_dir = 'high_risk'

                    cur_scene_out_path = os.path.join(gen_out_path_scenes, result_dir)
                    mkdir(cur_scene_out_path)

                    fout_path = os.path.join(cur_scene_out_path, 'scene_%04d.json' % cur_batch_i[b])
                    Logger.log(f'  → 保存 [{result_dir}] {os.path.basename(fout_path)}')
                    with open(fout_path, 'w') as writer:
                        json.dump(scene_out_dict, writer)
                    _scenes_saved += 1

                    # 生成基于初始轨迹的势场可视化
                    """
                    if viz and scene_out_dict.get('fut_init') is not None:
                        try:
                            import sys
                            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                            from field_model_init_trajectory import InitialTrajectoryFieldModel
                            from field_model_direct_prediction import DirectPredictionQuadrantFieldModel
                            
                            # 创建势场可视化目录
                            field_viz_dir = os.path.join(cur_scene_out_path, 'field_analysis')
                            mkdir(field_viz_dir)
                            
                            # 创建初始轨迹的势场模型
                            init_field_model = InitialTrajectoryFieldModel(scene_out_dict, trajectory_type='initial')
                            
                            # 创建对抗轨迹的势场模型
                            adv_field_model = InitialTrajectoryFieldModel(scene_out_dict, trajectory_type='adversarial')
                            
                            # 创建第二象限势场模型（直接从模型生成当前场景）
                            try:
                                quadrant_field_model = DirectPredictionQuadrantFieldModel(
                                    model=model,
                                    scene_graph=scene_graph_list[b],
                                    map_idx=map_idx[b],
                                    map_env=map_env,
                                    dt=data_loader.dataset.dt
                                )
                                Logger.log('第二象限Field Model创建成功')
                            except Exception as e:
                                Logger.log(f'第二象限Field Model创建失败: {e}')
                                quadrant_field_model = None
                            
                            # 选择关键时间点进行可视化
                            num_timesteps = len(scene_out_dict['fut_init'][0])
                            dt = scene_out_dict['dt']
                            max_time = (num_timesteps - 1) * dt
                            time_points = [t for t in [0.5, 2.0, 4.0, 6.0] if t <= max_time]
                            
                            # 生成可视化
                            for t in time_points:
                                # 初始轨迹势场
                                init_field_model.visualize_field_at_time(
                                    t, save_path=os.path.join(field_viz_dir, f'field_initial_t_{t:.1f}s.png')
                                )
                                
                                # 对抗轨迹势场
                                adv_field_model.visualize_field_at_time(
                                    t, save_path=os.path.join(field_viz_dir, f'field_adversarial_t_{t:.1f}s.png')
                                )
                                
                                # 对比图
                                init_field_model.visualize_comparison(
                                    t, adv_field_model, 
                                    save_path=os.path.join(field_viz_dir, f'field_comparison_t_{t:.1f}s.png')
                                )
                                
                                # 第二象限车辆位置图（如果模型创建成功）
                                if quadrant_field_model is not None:
                                    try:
                                        quadrant_field_model.visualize_vehicles_in_quadrant2(
                                            time_point=t,
                                            save_path=os.path.join(field_viz_dir, f'quadrant2_vehicles_t_{t:.1f}s.png'),
                                            figsize=(12, 10)
                                        )
                                    except Exception as e:
                                        Logger.log(f'第二象限车辆位置图生成失败 t={t:.1f}s: {e}')
                                    
                                    # 第二象限势场图
                                    try:
                                        quadrant_field_model.visualize_quadrant2_field_at_time(
                                            time_point=t,
                                            save_path=os.path.join(field_viz_dir, f'quadrant2_field_t_{t:.1f}s.png'),
                                            figsize=(14, 10)
                                        )
                                    except Exception as e:
                                        Logger.log(f'第二象限势场图生成失败 t={t:.1f}s: {e}')
                            
                            # 生成第二象限完整轨迹图和势场对比图
                            if quadrant_field_model is not None:
                                try:
                                    quadrant_field_model.visualize_trajectory_in_quadrant2(
                                        save_path=os.path.join(field_viz_dir, f'quadrant2_full_trajectory.png'),
                                        figsize=(14, 10)
                                    )
                                    Logger.log('第二象限完整轨迹图生成成功')
                                except Exception as e:
                                    Logger.log(f'第二象限完整轨迹图生成失败: {e}')
                                
                                # 生成势场时间演化对比图
                                try:
                                    quadrant_field_model.visualize_quadrant2_field_comparison(
                                        time_points=time_points[:3],  # 选择前3个时间点
                                        save_path=os.path.join(field_viz_dir, f'quadrant2_field_comparison.png'),
                                        figsize=(18, 6)
                                    )
                                    Logger.log('第二象限势场时间演化对比图生成成功')
                                except Exception as e:
                                    Logger.log(f'第二象限势场时间演化对比图生成失败: {e}')
                            
                            Logger.log(f'Field visualizations (including quadrant2) saved to {field_viz_dir}')
                            
                        except ImportError as e:
                            Logger.log(f'Warning: field model module not found, skipping field visualization: {e}')
                        except Exception as e:
                            Logger.log(f'Error generating field visualization: {str(e)}')
                            import traceback
                            Logger.log(f'Traceback: {traceback.format_exc()}')
                    """
                    # 如果需要提取潜变量和生成CARLA场景
                    if hasattr(config, 'extract_latent') and config.extract_latent:
                        # 创建场景提取器
                        scenario_extractor = ScenarioExtractor(model, model.get_normalizer(), model.get_att_normalizer())
                        
                        # 创建CARLA场景输出目录
                        carla_out_dir = os.path.join(gen_out_path_scenes, result_dir, 'carla')
                        mkdir(carla_out_dir)
                        
                        # 将单个图包装成一个批处理对象以匹配extractor的API
                        single_scene_batch = GraphBatch.from_data_list([scene_graph_list[b]])
                        # 提取场景数据
                        scenario_data = scenario_extractor.extract_structured_scenario(
                            #scene_graph=scene_graph_list[b],
                            scene_graph=single_scene_batch,
                            map_env=map_env,
                            #map_idx=map_idx[b].item(),
                            map_idx=map_idx[b:b+1],
                            past_traj=scene_graph.past_gt[scene_graph.ptr[b]:scene_graph.ptr[b+1]],
                            future_pred=final_result_traj[scene_graph.ptr[b]:scene_graph.ptr[b+1]][:,0],
                            latent_z=cur_z[scene_graph.ptr[b]:scene_graph.ptr[b+1]],
                            output_path=os.path.join(carla_out_dir, f'scene_{cur_batch_i[b]:04d}_data.json')
                        )
                        
                        # 如果需要生成CARLA脚本
                        if hasattr(config, 'generate_carla') and config.generate_carla:
                            # 生成CARLA场景脚本
                            generate_carla_scenario_script(
                                scenario_data=scenario_data,
                                output_path=os.path.join(carla_out_dir, f'scene_{cur_batch_i[b]:04d}_carla.py')
                            )
                            Logger.log(f'生成CARLA场景脚本: scene_{cur_batch_i[b]:04d}_carla.py')

                if viz:
                    cur_viz_out_path = os.path.join(gen_out_path_viz, result_dir)
                    mkdir(cur_viz_out_path)

                    # 解析时间戳
                    save_timestamps_list = None
                    if config.viz_timestamps is not None:
                        try:
                            save_timestamps_list = [float(t.strip()) for t in config.viz_timestamps.split(',')]
                        except ValueError:
                            Logger.log(f"警告: 无法解析 viz_timestamps '{config.viz_timestamps}'。请使用逗号分隔的数字。")


                    # save before viz
                    cur_crop_t = attack_t[b] if attack_t is not None else 0
                    pred_prefix = 'test_sample_%d_before' % (cur_batch_i[b])
                    pred_out_path = os.path.join(cur_viz_out_path, pred_prefix)
                    mark_stationary = getattr(config, 'mark_stationary', True)
                    viz_optim_results(pred_out_path, scene_graph, map_idx, map_env, model,
                                        init_future_pred, planner_name, cur_attack_agt,
                                        cur_crop_t,
                                        bidx=b,
                                        show_gt=True, # show entire nuscenes scene
                                        ow_gt=init_traj,
                                        keep_frames=config.keep_viz_frames,
                                        hide_stationary=mark_stationary)

                    # save after optimization viz
                    pred_prefix = 'test_sample_%d_after' % (cur_batch_i[b])
                    pred_out_path = os.path.join(cur_viz_out_path, pred_prefix)
                    viz_optim_results(pred_out_path, scene_graph, map_idx, map_env, model,
                                        final_result_traj, planner_name, cur_attack_agt, cur_crop_t,
                                        bidx=b,
                                        show_gt_idx=0,
                                        ow_gt=final_decoder_out['future_pred'].clone().detach(), # show our internal pred of planner as "gt" since final_result_traj is actual planner traj
                                        keep_frames=config.keep_viz_frames,
                                        hide_stationary=mark_stationary)

        except RuntimeError as e:
            Logger.log(f'│  RuntimeError: {e}  — 跳过本场景')
            for p in model.parameters():
                if p.grad is not None:
                    del p.grad
            empty_cache = True
            if 'CUDA' in str(e) or 'device-side assert' in str(e):
                Logger.log('│  CUDA 错误检测到，尝试重置 CUDA 状态...')
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            continue

        # 早停：已保存场景数达到 max_scenes 则终止
        if max_scenes > 0 and _scenes_saved >= max_scenes:
            Logger.log(f'[早停] 已生成 {_scenes_saved} 个场景，达到 max_scenes={max_scenes} 限制，提前终止。')
            break

    # ── 最终时间汇总 ──────────────────────────────────────────────────
    n   = max(_time_stats['scene_count'], 1)
    tot = _time_stats['total']
    Logger.log('═' * 62)
    Logger.log(f'  全部 {n} 个场景完成   总耗时 {tot:.0f}s ({tot/60:.1f} min)   均值 {tot/n:.1f}s/场景')
    Logger.log(f'  ┌ LLM_prep   {_time_stats["llm_prep"]/n:6.1f}s/场景  ({100*_time_stats["llm_prep"]/tot:4.1f}%)  (model.sample+可视化)')
    Logger.log(f'  ├ LLM        {_time_stats["llm"]/n:6.1f}s/场景  ({100*_time_stats["llm"]/tot:4.1f}%)  (纯LLM API调用)')
    Logger.log(f'  ├ init_optim {_time_stats["init_optim"]/n:6.1f}s/场景  ({100*_time_stats["init_optim"]/tot:4.1f}%)')
    Logger.log(f'  └ adv_optim  {_time_stats["adv_optim"]/n:6.1f}s/场景  ({100*_time_stats["adv_optim"]/tot:4.1f}%)')
    Logger.log('═' * 62)


def main():
    cfg, cfg_dict = parse_cfg()

    # create output directory and logging
    cfg.out = cfg.out + "_" + str(int(time.time()))
    mkdir(cfg.out)
    log_path = os.path.join(cfg.out, 'adv_gen_log.txt')
    Logger.init(log_path)
    # save arguments used
    Logger.log('Args: ' + str(cfg_dict))

    # set global random seed for reproducibility
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    Logger.log('Global seed set to %d' % (cfg.seed))

    # device setup
    device = get_device()
    Logger.log('Using device %s...' % (str(device)))

    # load dataset
    # first create map environment
    data_path = os.path.join(cfg.data_dir, cfg.data_version)
    map_env = NuScenesMapEnv(data_path,
                            bounds=cfg.map_obs_bounds,
                            L=cfg.map_obs_size_pix,
                            W=cfg.map_obs_size_pix,
                            layers=cfg.map_layers,
                            device=device,
                            load_lanegraph=(cfg.planner=='hardcode'),
                            lanegraph_res_meters=1.0
                            )
    test_dataset = NuScenesDataset(data_path, map_env,
                            version=cfg.data_version,
                            split=cfg.split,
                            categories=cfg.agent_types,
                            npast=cfg.past_len,
                            nfuture=cfg.future_len,
                            seq_interval=cfg.seq_interval,
                            randomize_val=True,
                            val_size=cfg.val_size,
                            reduce_cats=cfg.reduce_cats
                            )

    def worker_init_fn(worker_id):
        np.random.seed(cfg.seed + worker_id)

    # create loaders    
    test_loader = GraphDataLoader(test_dataset,
                                    batch_size=1, # will collect batches on the fly after determining feasibility
                                    shuffle=cfg.shuffle,
                                    num_workers=cfg.num_workers,
                                    pin_memory=False,
                                    worker_init_fn=worker_init_fn)

    # create model
    model = TrafficModel(cfg.past_len, cfg.future_len, cfg.map_obs_size_pix, len(test_dataset.categories),
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

    # load model weights
    if cfg.ckpt is not None:
        ckpt_epoch, _ = load_state(cfg.ckpt, model, map_location=device)
        Logger.log('Loaded checkpoint from epoch %d...' % (ckpt_epoch))
    else:
        throw_err('Must pass in model weights to do scenario generation!')

    # so can unnormalize as needed
    model.set_normalizer(test_dataset.get_state_normalizer())
    model.set_att_normalizer(test_dataset.get_att_normalizer())
    if cfg.model_output_bicycle:
        from datasets.utils import NUSC_BIKE_PARAMS
        model.set_bicycle_params(NUSC_BIKE_PARAMS)

    # 从配置构建基础损失权重字典
    loss_weights = {
        'coll_veh'             : cfg.loss_coll_veh,
        'coll_veh_plan'        : cfg.loss_coll_veh_plan,
        'coll_env'             : cfg.loss_coll_env,
        'motion_prior'         : cfg.loss_motion_prior,
        'motion_prior_atk'     : cfg.loss_motion_prior_atk,
        'init_z'               : cfg.loss_init_z,
        'init_z_atk'           : cfg.loss_init_z_atk,
        'motion_prior_ext'     : cfg.loss_motion_prior_ext,
        'match_ext'            : cfg.loss_match_ext,
        'adv_crash'            : cfg.loss_adv_crash,
        'init_match_ext'       : cfg.init_loss_match_ext,
        'init_motion_prior_ext': cfg.init_loss_motion_prior_ext,
        'ttc'                  : cfg.loss_ttc,
        'thw'                  : getattr(cfg, 'loss_thw', 0.5),
        'min_dist_lat'         : cfg.loss_min_dist_lat,
        'yaw_rate'             : cfg.loss_yaw_rate,
        'yaw_rate_ego'         : cfg.loss_yaw_rate_ego,
        'yaw_rate_non_ego'     : cfg.loss_yaw_rate_non_ego,
        'kine_veh'             : getattr(cfg, 'loss_kine_veh', 0.5),
        'kine_atk'             : getattr(cfg, 'loss_kine_atk', 0.1),
    }

    # --- 第一步：消融实验的静态权重覆盖（与LLM无关，先于LLM初始化执行）---
    ablation_mode = getattr(cfg, 'ablation_mode', 'full')

    if ablation_mode == 'aggressive_fixed':
        # 最激进的手工调参静态权重基线：所有风险相关项推到上界，不涉及LLM
        Logger.log('[Ablation aggressive_fixed] 使用最激进手工调参静态权重 (消融实验基线)')
        loss_weights['adv_crash']        = 10.0
        loss_weights['ttc']              = 6.0
        loss_weights['min_dist_lat']     = 5.0
        loss_weights['yaw_rate']         = 3.0
        loss_weights['yaw_rate_non_ego'] = 3.0
        loss_weights['coll_veh']         = 30.0
        loss_weights['coll_veh_plan']    = 30.0
        loss_weights['coll_env']         = 10.0
        loss_weights['motion_prior_atk'] = 0.001
        loss_weights['init_z_atk']       = 0.01
        for k, v in loss_weights.items():
            Logger.log(f'  [aggressive_fixed] {k}: {v}')

    elif ablation_mode == 'grid_search':
        # 使用离线网格搜索找到的最优权重，需先运行 ablation_weight_search.py
        import json
        grid_best_path = getattr(cfg, 'grid_search_weights_path',
                                  os.path.join(os.path.dirname(cfg.out), 'grid_search_best_weights.json'))
        if os.path.exists(grid_best_path):
            with open(grid_best_path) as _f:
                grid_best = json.load(_f)
            loss_weights.update(grid_best)
            Logger.log(f'[Ablation grid_search] 加载网格搜索最优权重: {grid_best_path}')
            for k, v in loss_weights.items():
                Logger.log(f'  [grid_search] {k}: {v}')
        else:
            Logger.log(f'[Ablation grid_search] 未找到预计算权重文件: {grid_best_path}，退回默认权重')
            Logger.log('  请先运行: python src/ablation_weight_search.py 生成最优权重文件')

    # --- 第二步：初始化 LLM 权重管理器（仅完整方法和 no_weight_adapt 模式启用）---
    _use_llm_modes = ('full', 'no_attacker_sel', 'no_weight_adapt')
    weight_manager = None
    if getattr(cfg, 'use_llm', False) and ablation_mode in _use_llm_modes:
        try:
            if not os.path.exists(cfg.llm_config_path):
                Logger.log(f'警告: LLM配置文件 {cfg.llm_config_path} 不存在，使用默认权重')
            else:
                llm_config = ConfigLoader.load_config(cfg.llm_config_path)
                ConfigLoader.setup_logging(llm_config)
                if ConfigLoader.is_llm_enabled(llm_config):
                    weight_manager = WeightManager(
                        static_weights=loss_weights,
                        use_llm=True,
                        model_name=cfg.llm_model,
                        cache_dir=cfg.llm_cache_dir,
                        traffic_model=model
                    )
                    Logger.log('LLM权重管理器初始化成功')
                else:
                    Logger.log('LLM在配置中被禁用，使用默认权重')
        except Exception as e:
            Logger.log(f'初始化LLM权重管理器失败: {e}，回退使用静态权重')
    else:
        Logger.log(f'[{ablation_mode}] 使用固定权重，不启用LLM')

    # 初始化可行性报告器（跨所有场景聚合）
    feasibility_reporter = FeasibilityReporter()

    # run through dataset once and generate possible scenarios
    model.train()
    run_one_epoch(test_loader, cfg.batch_size, model, map_env, device, cfg.out, loss_weights,
                  planner_name=cfg.planner, 
                  planner_cfg=cfg.planner_cfg,
                  feasibility_thresh=cfg.feasibility_thresh,
                  feasibility_time=cfg.feasibility_time,
                  feasibility_vel=cfg.feasibility_vel,
                  feasibility_infront_min=cfg.feasibility_infront_min,
                  feasibility_check_sep=cfg.feasibility_check_sep,
                  num_iters=cfg.num_iters,
                  lr=cfg.lr,
                  viz=cfg.viz,
                  save=cfg.save,
                  adv_attack_with=cfg.adv_attack_with,
                  weight_manager=weight_manager,
                  config=cfg,
                  ablation_mode=ablation_mode,
                  feasibility_reporter=feasibility_reporter,
                  max_scenes=getattr(cfg, 'max_scenes', 0))

    # ── 所有场景处理完后：打印可行性汇总统计并保存 ──────────────────────────
    aggregate_stats = feasibility_reporter.print_aggregate()
    if cfg.save and aggregate_stats:
        import json as _json
        agg_path = os.path.join(cfg.out, 'feasibility_aggregate.json')
        with open(agg_path, 'w') as _af:
            _json.dump(aggregate_stats, _af, indent=2)
        Logger.log(f'可行性汇总统计已保存至: {agg_path}')

    # 所有场景处理完后，输出跨场景LLM权重方差统计（用于分析LLM适应性）
    if weight_manager is not None \
            and hasattr(weight_manager, 'llm_weight_log') \
            and weight_manager.llm_weight_log:
        import json as _json
        variance_summary = weight_manager.summarize_llm_weight_variance()
        variance_path = os.path.join(cfg.out, 'llm_weight_variance_summary.json')
        with open(variance_path, 'w') as _vf:
            _json.dump(variance_summary, _vf, indent=2)
        Logger.log(f'LLM weight variance summary saved to {variance_path}')


if __name__ == "__main__":
    main()
