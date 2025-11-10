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
from utils.adv_gen_optim import run_adv_gen_optim, compute_adv_gen_success
from utils.sol_optim import run_find_solution_optim, compute_sol_success
from utils.init_optim import run_init_optim
from utils.risk_classifier import classify_scenario_by_risk_level
from planners.planner import PlannerConfig
from utils.config import get_parser, add_base_args
import pdb

# 导入LLM相关模块
import logging
from src.llm.config_loader import ConfigLoader
from src.llm.weight_manager import WeightManager
from src.llm.scenario_extractor import ScenarioExtractor

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
    parser.add_argument('--loss_min_dist_lat', type=float, default=1.0, help='横向最小距离损失权重，惩罚与其他车辆横向距离过近的行为。')
    parser.add_argument('--loss_yaw_rate', type=float, default=0.8, help='横摆角速度损失权重，惩罚剧烈、不稳定的转向动作。')
    parser.add_argument('--loss_yaw_rate_ego', type=float, default=0.5, help='目标车辆的横摆角速度损失权重。')
    parser.add_argument('--loss_yaw_rate_non_ego', type=float, default=1.0, help='非目标车辆的横摆角速度损失权重。')
    # solution optimization
    parser.add_argument('--sol_future_len', type=int, default=16, help='The number of timesteps to roll out to compute collision losses for solution. If > model/data future len, will avoid irrecoverable final states for solution.')
    parser.add_argument('--sol_loss_coll_veh', type=float, default=10.0, help='Loss to avoid planner-vehicle collisions.')
    parser.add_argument('--sol_loss_coll_env', type=float, default=10.0, help='Loss to avoid planner-environment collisions.')
    parser.add_argument('--sol_loss_motion_prior', type=float, default=0.005, help='Loss to keep planner z likely under the motion prior.')
    parser.add_argument('--sol_loss_init_z', type=float, default=0.0, help='Loss to keep planner z near output of adv optim.')
    parser.add_argument('--sol_loss_motion_prior_ext', type=float, default=0.001, help='Loss to keep non-planner z near output of adv optim.')
    parser.add_argument('--sol_loss_match_ext', type=float, default=10.0, help='Match trajectories from output of adv optim for non-planner agents.')

    parser.add_argument('--num_iters', type=int, default=300, help='Number of optimization iterations.')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate for adam.')

    parser.add_argument('--viz', dest='viz', action='store_true',
                        help="If given, saves low-quality visualization before and after optimization.")
    parser.set_defaults(viz=False)
    parser.add_argument('--save', dest='save', action='store_true',
                        help="If given, saves the scenarios as json so they can be used later.")
    parser.set_defaults(save=False)
    
    # LLM权重相关配置
    parser.add_argument('--use_llm', dest='use_llm', action='store_true',
                        help="如果指定，则使用LLM生成的动态权重")
    parser.set_defaults(use_llm=True)  # 改为默认启用LLM
    parser.add_argument('--llm_config_path', type=str, default='configs/llm_weights_config.yaml',
                        help='LLM权重配置文件路径')
    parser.add_argument('--llm_cache_dir', type=str, default='./llm_cache',
                        help='LLM权重缓存目录')
    parser.add_argument('--llm_model', type=str, default='deepseek-reasoner',
                        help='使用的LLM模型名称 (gpt-4o, gpt-3.5-turbo, deepseek-chat, deepseek-reasoner)')

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
    
    # 添加风险等级参数
    parser.add_argument('--risk_level', type=str, default='high_risk',
                        choices=['low_risk', 'high_risk', 'longtail_condition'],
                        help='目标风险等级：low_risk(低风险), high_risk(高风险), longtail_condition(长尾条件)')

    # 添加可视化帧保存参数
    parser.add_argument('--keep_viz_frames', dest='keep_viz_frames', action='store_true',
                        help="如果指定，将保存用于创建可视化视频的单个帧")
    parser.set_defaults(keep_viz_frames=False)

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
                  sol_future_len=16,
                  num_iters=300,
                  lr=0.05,
                  viz=True,
                  save=True,
                  adv_attack_with=None,
                  weight_manager=None,
                  config=None,
                  scenario_description: str = ""
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
    for i, data in enumerate(pbar_data):
        start_t = time.time()
        sample_pred = None
        scene_graph, map_idx = data
        if empty_cache:
            empty_cache = False
            gc.collect()
            torch.cuda.empty_cache()
        try:
            scene_graph = scene_graph.to(device)
            map_idx = map_idx.to(device)
            print('scene_%d' % (i))
            print(scene_graph)
            print([map_env.map_list[map_idx[b]] for b in range(map_idx.size(0))])
            is_last_batch = i == (len(data_loader)-1)
            
            # 如果使用动态权重，根据场景更新权重
            current_weights = loss_weights
            if weight_manager is not None:
                try:
                    Logger.log(f'=== 场景 {i}: 开始动态权重生成 ===')
                    
                    # 采样一次以获得未来轨迹预测，用于场景描述
                    with torch.no_grad():
                        future_pred_sample = model.sample_batched(scene_graph, map_idx, map_env, 1, include_mean=True)
                    
                    Logger.log('获取场景预测轨迹成功，开始longterm分析...')
                    
                    # 生成势场可视化（如果配置启用）
                    field_info = None
                    if getattr(config, 'include_field_visualization', False):
                        try:
                            Logger.log('生成势场可视化图片...')
                            import tempfile
                            import base64
                            import sys
                            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                           
                            from field_model_direct_prediction import DirectPredictionQuadrantFieldModel
                            
                            # 创建第二象限势场模型（直接从模型生成）
                            quadrant_field_model = DirectPredictionQuadrantFieldModel(
                                model=model,
                                scene_graph=scene_graph,
                                map_idx=map_idx,
                                map_env=map_env,
                                dt=data_loader.dataset.dt
                            )
                            
                            # 选择关键时间点（基于模型预测的轨迹长度）
                            max_time = (quadrant_field_model.future_pred.shape[1] - 1) * data_loader.dataset.dt
                            t_key = min(2.0, max_time)
                            
                            # 创建势场图片保存目录
                            field_out_dir = os.path.join(gen_out_path, f'scene_{i}', 'field_analysis')
                            os.makedirs(field_out_dir, exist_ok=True)
                        
                            
                            # 生成第二象限车辆位置图
                            quadrant_img_path = os.path.join(field_out_dir, f'quadrant2_vehicles_t_{t_key:.1f}s.png')
                            quadrant_field_model.visualize_vehicles_in_quadrant2(
                                time_point=t_key, 
                                save_path=quadrant_img_path, 
                                figsize=(12, 10)
                            )
                            Logger.log(f'第二象限车辆位置图已保存到: {quadrant_img_path}')
                            
                            # 生成第二象限势场图
                            quadrant_field_path = os.path.join(field_out_dir, f'quadrant2_field_t_{t_key:.1f}s.png')
                            quadrant_field_model.visualize_quadrant2_field_at_time(
                                time_point=t_key,
                                save_path=quadrant_field_path,
                                figsize=(14, 10)
                            )
                            Logger.log(f'第二象限势场图已保存到: {quadrant_field_path}')
                            
                            # 生成第二象限轨迹图
                            quadrant_traj_path = os.path.join(field_out_dir, f'quadrant2_trajectory.png')
                            quadrant_field_model.visualize_trajectory_in_quadrant2(
                                save_path=quadrant_traj_path,
                                figsize=(14, 10)
                            )
                            Logger.log(f'第二象限轨迹图已保存到: {quadrant_traj_path}')
                            
                            # 读取第二象限图片并编码（用于LLM分析）
                            try:
                                with open(quadrant_img_path, 'rb') as img_file:
                                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                                
                                # 设置第二象限势场信息
                                field_info = {
                                    'has_image': True,
                                    'image_base64': img_base64,
                                    'image_format': 'png',
                                    'time_point': t_key,
                                    'visualization_type': 'quadrant2_vehicles',
                                    'statistics': {
                                        'num_vehicles': quadrant_field_model.num_agents,
                                        'coordinate_system': 'quadrant2 (x<0, y>0)',
                                        'time_steps': quadrant_field_model.future_pred.shape[1],
                                        'visualization_description': f'第二象限车辆位置图，显示{quadrant_field_model.num_agents}辆车在t={t_key:.1f}s的位置和运动状态'
                                    }
                                }
                                
                                Logger.log(f'第二象限势场可视化生成成功，图片大小: {len(img_base64)} bytes')
                                
                            except Exception as e:
                                Logger.log(f'第二象限图片编码失败: {e}')
                                field_info = {
                                    'has_image': False,
                                    'visualization_type': 'quadrant2_vehicles',
                                    'error': str(e)
                                }
                            
                        except Exception as e:
                            Logger.log(f'Warning: 势场可视化生成失败: {str(e)}')
                            import traceback
                            Logger.log(f'详细错误: {traceback.format_exc()}')
                            field_info = None
                    
                    # 更新权重 - 传递完整的参数集合（包括势场信息和风险等级）
                    current_weights = weight_manager.update_from_scenario(
                        scene_graph=scene_graph,
                        map_env=map_env,
                        map_idx=map_idx,
                        future_pred=future_pred_sample.get('future_pred', None),
                        past_traj=scene_graph.past,  # 传递过去轨迹
                        driving_objectives=f"场景{i}的对抗性驾驶行为分析",  # 添加目标描述
                        extra_context=field_info,  # 传递势场图片信息
                        risk_level=config.risk_level  # 传递目标风险等级
                    )
                    
                    Logger.log(f'=== 场景 {i}: 动态权重生成完成 ===')
                    
                    # 验证和记录权重更新
                    if current_weights != loss_weights:
                        Logger.log('检测到权重更新，记录关键变化:')
                        
                        # 记录关键权重的变化
                        important_weights = ['coll_veh', 'adv_crash', 'ttc', 'tlc', 'thw', 'min_dist_lat', 'yaw_rate', 'coll_env', 'delta_v', 'path_adherence', 'motion_prior_atk', 'init_z_atk']
                        for key in important_weights:
                            old_val = loss_weights.get(key, 'N/A')
                            new_val = current_weights.get(key, 'N/A')
                            if old_val != new_val:
                                Logger.log(f'  {key}: {old_val} → {new_val}')
                        
                        Logger.log('权重更新应用成功')
                    else:
                        Logger.log('权重未发生变化，可能使用了缓存或回退机制')
                    
                except Exception as e:
                    Logger.log(f'=== 场景 {i}: 动态权重生成失败 ===')
                    Logger.log(f'错误类型: {type(e).__name__}')
                    Logger.log(f'错误信息: {str(e)}')
                    import traceback
                    Logger.log(f'详细错误:\n{traceback.format_exc()}')
                    Logger.log('回退使用静态权重继续生成...')
                    current_weights = loss_weights

            # First sample prior to get possible futures
            with torch.no_grad():
                sample_pred = model.sample_batched(scene_graph, map_idx, map_env, 20, include_mean=True)
                # sample_pred = model.sample(scene_graph, map_idx, map_env, 20, include_mean=True)

            empty_cache = True
            # determine if this sequence is feasible for scenario generation
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
                # make sure the ego "planner" has max velocity above some thresh for
                # it to be considered an interesting scenario
                ego_gt = model.get_normalizer().unnormalize(scene_graph.future_gt[0])
                ego_vels = torch.norm(ego_gt[1:, :2] - ego_gt[:-1, :2], dim=-1)
                max_vel = torch.max(ego_vels).cpu().item()
                if max_vel < feasibility_vel:
                    Logger.log('Ego vehicle not moving more than velocity threshold, skipping...')
                    if not is_last_batch:
                        continue
            elif planner_name == 'hardcode':
                # make sure some sample of the ego went over the velocity thresh
                #   so with some confidence it will be an interesting scenario
                ego_samps = model.get_normalizer().unnormalize(sample_pred['future_pred'][0].detach()) # NS x FT x 4
                ego_vels = torch.norm(ego_samps[:, 1:, :2] - ego_samps[:, :-1, :2], dim=-1) # NS x FT-1
                max_vel = torch.max(ego_vels).cpu().item()
                if max_vel < feasibility_vel:
                    Logger.log('Ego samples not moving more than velocity threshold, skipping...')
                    if not is_last_batch:
                        continue

            if feasible is None:
                Logger.log('Only ego vehicle in scene, skipping...')
                if not is_last_batch:
                    continue
            elif torch.sum(feasible).item() == 0:
                Logger.log('Infeasible, no vehicles near ego, skipping...')
                if not is_last_batch:
                    continue

            is_feas = False
            if feasible is not None and torch.sum(feasible).item() > 0:
                is_feas = True
                if adv_attack_with is not None:
                    # only attack with a specific category
                    feas_sem = scene_graph.sem[1:]
                    veclist = [tuple(feas_sem[aidx].to(int).cpu().numpy().tolist()) for aidx in range(feas_sem.size(0))]
                    is_adv_atk = [data_loader.dataset.vec2cat[curvec] == adv_attack_with for curvec in veclist]
                    adv_atk_feas = torch.zeros_like(feasible)
                    adv_atk_feas[is_adv_atk] = True
                    feasible = torch.logical_and(feasible, adv_atk_feas)
                    if torch.sum(feasible) == 0:
                        Logger.log('No feasible attackers of requested category, skipping...')
                        is_feas = False

                if is_feas:
                    # print which vehicle is the best candidate (for info/debug purposes)
                    feasible_dist[~feasible] = float('inf')
                    temp_attack_agt = torch.min(feasible_dist, dim=0)[1] + 1
                    print('Heuristic attack agt is %d' % (temp_attack_agt))
                    print('Heuristic attack time is %d' % (feasible_time[temp_attack_agt-1]))  

            # This is a feasible seed, add it to the batch
            if is_feas:
                Logger.log('Feasible. Adding to batch...')
                batch_scene_graph += scene_graph.to_data_list()
                batch_map_idx.append(map_idx)
                batch_i.append(i)
                batch_total_NA += scene_graph.future_gt.size(0)
                Logger.log('Current batch NA: %d' % (batch_total_NA))

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

                Logger.log('Formed batch! Starting optimization...')
                Logger.log(scene_graph)

                # 将dt添加到scene_graph
                scene_graph.dt = data_loader.dataset.dt

                # reset
                batch_scene_graph = []
                batch_map_idx = []
                batch_i = []
                batch_total_NA = 0

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
                Logger.log('Using planner config:')
                Logger.log(CONFIG_DICT[planner_cfg])
                planner = HardcodeNuscPlanner(map_env, PlannerConfig(**CONFIG_DICT[planner_cfg])) 

            # start from GT scene future (reconstructed with motion model)
            z_init = embed_info_attached['posterior_out'][0].detach()
            init_traj = scene_graph.future_gt[:, :, :4].clone().detach()
            Logger.log('Running initialization optimization...')

            # run initial optimization to closely fit nuscenes scene
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
                Logger.log('Fine-tune init with planner rollout...')
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
                    Logger.log('Planner already caused collision after init, removing from batch...')
                    if np.sum(bvalid) == 0:
                        Logger.log('No valid sequences left in batch! Skipping...')
                        continue
                    # need to remove invalid scenarios from batch
                    # rebuild and reset all necessary variables
                    map_idx = map_idx[bvalid]
                    cur_batch_i = [bi for b, bi in enumerate(cur_batch_i) if bvalid[b]]

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

                    Logger.log(scene_graph)

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
                    Logger.log('Computing initial trajectory field for optimization guidance...')
                    # 注意：这里我们暂时只提供接口，实际的势场引导将在后续实现
                    # field_guidance将被传递给优化函数使用
                    Logger.log('Field guidance interface prepared (implementation pending)')
                except Exception as e:
                    Logger.log(f'Warning: Failed to prepare field guidance: {str(e)}')
                    field_guidance = None

            if planner_name == 'hardcode':
                plan_out_path = None
                if viz:
                    plan_out_path = os.path.join(gen_out_path_viz, 'planner_out')
                    cur_seq_str = 'sample_' + '_'.join(['%03d' for b in range(len(cur_batch_i))]) % tuple([cur_batch_i[b] for b in range(len(cur_batch_i))])
                    plan_out_path = os.path.join(plan_out_path, cur_seq_str)
                    mkdir(plan_out_path)

            # adversarial optimization
            cur_z = z_init.clone().detach()
            tgt_prior_distrib = (embed_info['prior_out'][0][ego_mask], embed_info['prior_out'][1][ego_mask])
            other_prior_distrib = (embed_info['prior_out'][0][~ego_mask], embed_info['prior_out'][1][~ego_mask])
            
            # 从LLM分析结果中获取攻击车辆ID
            attack_agt_idx = None
            Logger.log('=== 开始提取LLM识别的攻击车辆ID ===')
            Logger.log(f'[DEBUG] weight_manager is None? {weight_manager is None}')
            if weight_manager is not None:
                Logger.log(f'[DEBUG] weight_manager type: {type(weight_manager)}')
                Logger.log(f'[DEBUG] Has attacker_vehicle_id? {hasattr(weight_manager, "attacker_vehicle_id")}')
                if hasattr(weight_manager, 'attacker_vehicle_id'):
                    Logger.log(f'[DEBUG] attacker_vehicle_id value: {weight_manager.attacker_vehicle_id}')
                
            if weight_manager is not None and hasattr(weight_manager, 'attacker_vehicle_id'):
                try:
                    Logger.log('=== 从weight_manager获取LLM识别的攻击车辆ID ===')
                    vehicle_id = weight_manager.attacker_vehicle_id
                    Logger.log(f'[DEBUG] Extracted vehicle_id: {vehicle_id}')
                    if vehicle_id is not None:
                        if vehicle_id > 0:  # vehicle_id=0是ego，不应该作为攻击者
                            attack_agt_idx = [vehicle_id] * B  # 直接使用vehicle_id作为scene local索引
                            Logger.log(f'✓ LLM识别的攻击车辆: 场景JSON id={vehicle_id}, 作为scene_local_idx={vehicle_id}传递给优化器')
                        else:
                            Logger.log('[WARNING] vehicle_id=0是ego车辆，不能作为攻击者，将使用自动检测')
                            attack_agt_idx = None
                    else:
                        Logger.log('[INFO] attacker_vehicle_id is None，将使用自动检测')
                except Exception as e:
                    Logger.log(f'✗ 获取攻击车辆ID失败: {e}')
                    import traceback
                    Logger.log(f'详细错误:\n{traceback.format_exc()}')
            else:
                Logger.log('[INFO] weight_manager 或 attacker_vehicle_id 不可用，将使用自动检测')
                
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

            # build the solution optimization batch (only indices that succeeded)
            print(adv_succeeded)
            batch_graph_list = scene_graph.to_data_list()
            sol_graph_list = [batch_graph_list[b] for b in range(B) if adv_succeeded[b]]
            sol_succeeded = []
            if len(sol_graph_list) > 0:
                Logger.log('Batch adv optim successes:')
                Logger.log(adv_succeeded)

                sol_scene_graph = GraphBatch.from_data_list(sol_graph_list)
                Logger.log('Solution scene graph:')
                Logger.log(sol_scene_graph)

                sol_amask = torch.zeros((NA), dtype=torch.bool) # which agents are part of solution graphs
                sol_bmask = torch.zeros((B), dtype=torch.bool) # which batch indices need a solution
                for b in range(B):
                    if adv_succeeded[b]:
                        sol_amask[scene_graph.ptr[b]:scene_graph.ptr[b+1]] = True
                        sol_bmask[b] = True

                Logger.log('Adv gen succeeded! Finding solution...')
                # collect info for just the batch indices that need solution
                sol_in_final_result_traj = final_result_traj[sol_amask]
                sol_NA = sol_in_final_result_traj.size(0)
                sol_ego_inds = sol_scene_graph.ptr[:-1]
                sol_ego_mask = torch.zeros((sol_NA), dtype=torch.bool)
                sol_ego_mask[sol_ego_inds] = True
                sol_in_cur_z = cur_z.clone().detach()[sol_amask]
                sol_map_idx = map_idx[sol_bmask]
                sol_embed_info = dict()
                for k, v in embed_info.items():
                    if isinstance(v, torch.Tensor):
                        # everything else
                        sol_embed_info[k] = v[sol_amask]
                    elif isinstance(v, tuple):
                        # posterior or prior output
                        sol_embed_info[k] = (v[0][sol_amask], v[1][sol_amask])
                sol_tgt_prior_distrib = (tgt_prior_distrib[0][sol_bmask], tgt_prior_distrib[1][sol_bmask])
                sol_other_prior_distrib = (sol_embed_info['prior_out'][0][~sol_ego_mask], sol_embed_info['prior_out'][1][~sol_ego_mask])

                sol_z, sol_result_traj, sol_decoder_out = run_find_solution_optim(sol_in_cur_z, sol_in_final_result_traj, sol_future_len, 
                                                                                    lr, loss_weights, model, sol_scene_graph, map_env, sol_map_idx,
                                                                                    num_iters, sol_embed_info,
                                                                                    sol_tgt_prior_distrib, sol_other_prior_distrib)

                for b in range(sol_map_idx.size(0)):
                    cur_sol_succeeded = compute_sol_success(sol_result_traj[sol_scene_graph.ptr[b]:sol_scene_graph.ptr[b+1]][:,0:1],
                                                        model,
                                                        GraphBatch.from_data_list([sol_scene_graph.to_data_list()[b]]),
                                                        map_env,
                                                        sol_map_idx[b:b+1])
                    sol_succeeded.append(cur_sol_succeeded)

                cur_sidx = 0
                final_sol_suceeded = [False]*B
                for b in range(B):
                    if adv_succeeded[b]:
                        final_sol_suceeded[b] = sol_succeeded[cur_sidx]
                        cur_sidx += 1
                sol_succeeded = final_sol_suceeded

            print(sol_succeeded)

            Logger.log('Optimized sequence in %f sec!' % (time.time() - start_t))

            # output scenario and viz
            cur_sidx = 0
            scene_graph_list = scene_graph.to_data_list()
            for b in range(B):
                # 使用基于风险等级的分类替代原有的攻击成功/失败分类
                result_dir = None

                cur_attack_agt = attack_agt[b] - scene_graph.ptr[b].item() if attack_agt is not None else 0 # make index local to each batch idx
                cur_attack_t = attack_t[b] if attack_t is not None else 0

                if save:
                    import json
                    # save scenario
                    out_sol_traj = sol_result_traj[sol_scene_graph.ptr[cur_sidx]:sol_scene_graph.ptr[cur_sidx+1]][:,0] if adv_succeeded[b] else None
                    out_sol_z = sol_z[sol_scene_graph.ptr[cur_sidx]:sol_scene_graph.ptr[cur_sidx+1]] if out_sol_traj is not None else None
                    scene_out_dict = prepare_output_dict(scene_graph_list[b], map_idx[b].item(), map_env, data_loader.dataset.dt, model,
                                                          init_future_pred[scene_graph.ptr[b]:scene_graph.ptr[b+1]],
                                                          final_result_traj[scene_graph.ptr[b]:scene_graph.ptr[b+1]][:,0],
                                                          out_sol_traj,
                                                          cur_attack_agt,
                                                          cur_attack_t,
                                                          cur_z[scene_graph.ptr[b]:scene_graph.ptr[b+1]],
                                                          out_sol_z,
                                                          (embed_info['prior_out'][0][scene_graph.ptr[b]:scene_graph.ptr[b+1]], embed_info['prior_out'][1][scene_graph.ptr[b]:scene_graph.ptr[b+1]]),
                                                          internal_ego_traj=final_decoder_out['future_pred'][scene_graph.ptr[b]].detach()
                                                          )

                    # 使用风险分类器对场景进行分类
                    try:
                        result_dir = classify_scenario_by_risk_level(scene_out_dict, scene_graph_list[b])
                        Logger.log(f'场景 {cur_batch_i[b]} 分类结果: {result_dir}')
                    except Exception as e:
                        Logger.log(f'风险分类出错: {e}，使用默认分类high_risk')
                        result_dir = 'high_risk'
                    
                    # 更新输出路径
                    cur_scene_out_path = os.path.join(gen_out_path_scenes, result_dir)
                    mkdir(cur_scene_out_path)

                    fout_path = os.path.join(cur_scene_out_path, 'scene_%04d.json' % cur_batch_i[b])
                    Logger.log('Saving scene to %s' % (fout_path))
                    with open(fout_path, 'w') as writer:
                        json.dump(scene_out_dict, writer)

                    # 生成基于初始轨迹的势场可视化
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
                        scenario_data = scenario_extractor.extract_carla_scenario(
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
                            scenario_extractor.generate_carla_scenario_script(
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
                    viz_optim_results(pred_out_path, scene_graph, map_idx, map_env, model,
                                        init_future_pred, planner_name, cur_attack_agt,
                                        cur_crop_t,
                                        bidx=b,
                                        show_gt=True, # show entire nuscenes scene
                                        ow_gt=init_traj,
                                        keep_frames=config.keep_viz_frames)

                    # save after optimization viz
                    pred_prefix = 'test_sample_%d_after' % (cur_batch_i[b])
                    pred_out_path = os.path.join(cur_viz_out_path, pred_prefix)
                    viz_optim_results(pred_out_path, scene_graph, map_idx, map_env, model,
                                        final_result_traj, planner_name, cur_attack_agt, cur_crop_t,
                                        bidx=b,
                                        show_gt_idx=0,
                                        ow_gt=final_decoder_out['future_pred'].clone().detach(), # show our internal pred of planner as "gt" since final_result_traj is actual planner traj
                                        keep_frames=config.keep_viz_frames)

                    if adv_succeeded[b]:
                        pred_prefix = 'test_sample_%d_sol' % (cur_batch_i[b])
                        pred_out_path = os.path.join(cur_viz_out_path, pred_prefix)
                        viz_optim_results(pred_out_path, sol_scene_graph, sol_map_idx, map_env, model,
                                        sol_result_traj, planner_name, cur_attack_agt, cur_crop_t,
                                        bidx=cur_sidx,
                                        show_gt_idx=0,
                                        ow_gt=final_result_traj[sol_amask][:, 0], # show the failed planner path pre-sol as "GT"
                                        keep_frames=config.keep_viz_frames)
                
                if adv_succeeded[b]:
                    cur_sidx += 1

        except RuntimeError as e:
            Logger.log('Caught error in optim batch %s!' % (str(e)))
            Logger.log('Skipping')
            raise e
            for p in model.parameters():
                if p.grad is not None:
                    del p.grad  # free some memory
            empty_cache = True
            continue

def main():
    cfg, cfg_dict = parse_cfg()

    # create output directory and logging
    cfg.out = cfg.out + "_" + str(int(time.time()))
    mkdir(cfg.out)
    log_path = os.path.join(cfg.out, 'adv_gen_log.txt')
    Logger.init(log_path)
    # save arguments used
    Logger.log('Args: ' + str(cfg_dict))

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

    # create loaders    
    test_loader = GraphDataLoader(test_dataset,
                                    batch_size=1, # will collect batches on the fly after determining feasibility
                                    shuffle=cfg.shuffle,
                                    num_workers=cfg.num_workers,
                                    pin_memory=False,
                                    worker_init_fn=lambda _: np.random.seed()) # get around numpy RNG seed bug

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

    # 创建默认损失权重字典
    loss_weights = {
        'coll_veh' : cfg.loss_coll_veh,
        'coll_veh_plan' : cfg.loss_coll_veh_plan,
        'coll_env' : cfg.loss_coll_env,
        'motion_prior' : cfg.loss_motion_prior,
        'motion_prior_atk' : cfg.loss_motion_prior_atk,
        'init_z' : cfg.loss_init_z,
        'init_z_atk': cfg.loss_init_z_atk,
        'motion_prior_ext' : cfg.loss_motion_prior_ext,
        'match_ext' : cfg.loss_match_ext,
        'adv_crash' : cfg.loss_adv_crash,
        'sol_coll_veh' : cfg.sol_loss_coll_veh,
        'sol_coll_env' : cfg.sol_loss_coll_env,
        'sol_motion_prior' : cfg.sol_loss_motion_prior,
        'sol_init_z' : cfg.sol_loss_init_z,
        'sol_motion_prior_ext' : cfg.sol_loss_motion_prior_ext,
        'sol_match_ext' : cfg.sol_loss_match_ext,
        'init_match_ext' : cfg.init_loss_match_ext,
        'init_motion_prior_ext' : cfg.init_loss_motion_prior_ext,
        # 添加新的损失权重
        'ttc': cfg.loss_ttc if hasattr(cfg, 'loss_ttc') else 1.5,
        'min_dist_lat': cfg.loss_min_dist_lat if hasattr(cfg, 'loss_min_dist_lat') else 1.0,
        'yaw_rate': cfg.loss_yaw_rate if hasattr(cfg, 'loss_yaw_rate') else 0.8,
        'yaw_rate_ego': cfg.loss_yaw_rate_ego if hasattr(cfg, 'loss_yaw_rate_ego') else 0.5,
        'yaw_rate_non_ego': cfg.loss_yaw_rate_non_ego if hasattr(cfg, 'loss_yaw_rate_non_ego') else 1.0
    }

    # 如果启用了LLM，初始化权重管理器
    weight_manager = None
    if hasattr(cfg, 'use_llm') and cfg.use_llm:
        try:
            Logger.log('正在使用LLM动态生成权重...')
            
            # 加载LLM配置
            if not os.path.exists(cfg.llm_config_path):
                Logger.log(f'警告: LLM配置文件 {cfg.llm_config_path} 不存在，使用默认权重')
            else:
                # 加载配置
                llm_config = ConfigLoader.load_config(cfg.llm_config_path)
                ConfigLoader.setup_logging(llm_config)
                
                # 检查LLM是否启用
                if ConfigLoader.is_llm_enabled(llm_config):
                    # 初始化权重管理器
                    weight_manager = WeightManager(
                        static_weights=loss_weights,
                        use_llm=True,
                        model_name=cfg.llm_model,
                        cache_dir=cfg.llm_cache_dir,
                        traffic_model=model
                    )
                    #调试一下weightmanager
                    #pdb.set_trace()
                    
                    Logger.log('LLM权重管理器初始化成功')
                else:
                    Logger.log('LLM在配置中被禁用，使用默认权重')
        except Exception as e:
            Logger.log(f'初始化LLM权重管理器失败: {e}')
            Logger.log('使用默认权重继续...')
            
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
                  sol_future_len=cfg.sol_future_len,
                  num_iters=cfg.num_iters,
                  lr=cfg.lr,
                  viz=cfg.viz,
                  save=cfg.save,
                  adv_attack_with=cfg.adv_attack_with,
                  weight_manager=weight_manager,
                  config=cfg)


if __name__ == "__main__":
    main()
