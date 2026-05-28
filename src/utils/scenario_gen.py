# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os

import numpy as np
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt

from datasets import nuscenes_utils as nutils
from utils.common import dict2obj, mkdir
from utils.logger import Logger, throw_err

import datasets.nuscenes_utils as nutils


def activate_stationary_vehicles(init_traj, scene_graph, model, dt=0.5, 
                                  velocity_threshold=0.5, min_speed=2.0, max_speed=5.0,
                                  ego_mask=None):
    """
    激活静止车辆，给它们注入沿着当前朝向的运动轨迹。
    
    :param init_traj: (NA, FT, 4) 初始轨迹 (x, y, cos_h, sin_h)，已归一化
    :param scene_graph: 场景图，包含过去轨迹信息
    :param model: 交通模型，用于获取归一化器
    :param dt: 时间步长 (秒)
    :param velocity_threshold: 判断静止的速度阈值 (m/s)
    :param min_speed: 激活后的最小速度 (m/s)
    :param max_speed: 激活后的最大速度 (m/s)
    :param ego_mask: ego车辆mask，这些车辆不会被激活
    :return: 激活后的轨迹，激活的车辆索引列表
    """
    device = init_traj.device
    
    # 反归一化以进行检测
    init_traj_unnorm = model.get_normalizer().unnormalize(init_traj.clone())
    
    NA, FT, _ = init_traj_unnorm.shape
    
    # 检测静止车辆
    stationary_vehicles = nutils.detect_stationary_vehicles(
        init_traj_unnorm, velocity_threshold=velocity_threshold, dt=dt
    )
    
    if not stationary_vehicles:
        Logger.log('[激活静止车辆] 没有检测到静止车辆')
        return init_traj, []
    
    # 排除ego车辆
    if ego_mask is not None:
        ego_indices = set(torch.where(ego_mask)[0].cpu().numpy().tolist())
        stationary_vehicles = [v for v in stationary_vehicles if v not in ego_indices]
    
    if not stationary_vehicles:
        Logger.log('[激活静止车辆] 没有需要激活的静止车辆（排除ego后）')
        return init_traj, []
    
    Logger.log(f'[激活静止车辆] 检测到 {len(stationary_vehicles)} 辆静止车辆: {stationary_vehicles}')
    
    activated_traj = init_traj_unnorm.clone()
    activated_vehicles = []
    
    for v_idx in stationary_vehicles:
        # 获取车辆的初始状态（从过去轨迹的最后一帧）
        past_state = model.get_normalizer().unnormalize(scene_graph.past_gt[v_idx, -1, :4])
        
        # 获取朝向 (cos_h, sin_h)
        cos_h = past_state[2].item()
        sin_h = past_state[3].item()
        
        # 确保朝向向量是单位向量
        h_norm = np.sqrt(cos_h**2 + sin_h**2)
        if h_norm > 0:
            cos_h = cos_h / h_norm
            sin_h = sin_h / h_norm
        else:
            # 如果朝向为零，给一个默认方向
            cos_h = 1.0
            sin_h = 0.0
        
        # 随机生成一个合理的速度
        speed = np.random.uniform(min_speed, max_speed)
        
        # 计算位移增量
        dx = cos_h * speed * dt
        dy = sin_h * speed * dt
        
        # 生成沿着朝向前进的轨迹
        start_x = past_state[0].item()
        start_y = past_state[1].item()
        
        for t in range(FT):
            # 添加少量随机扰动让轨迹更自然
            noise_factor = 0.05
            noise_x = np.random.uniform(-noise_factor, noise_factor) * speed * dt
            noise_y = np.random.uniform(-noise_factor, noise_factor) * speed * dt
            
            new_x = start_x + (t + 1) * dx + noise_x
            new_y = start_y + (t + 1) * dy + noise_y
            
            activated_traj[v_idx, t, 0] = new_x
            activated_traj[v_idx, t, 1] = new_y
            # 保持朝向不变
            activated_traj[v_idx, t, 2] = cos_h
            activated_traj[v_idx, t, 3] = sin_h
        
        activated_vehicles.append(v_idx)
        Logger.log(f'  车辆 {v_idx}: 激活速度 {speed:.1f} m/s, 方向 ({cos_h:.2f}, {sin_h:.2f})')
    
    # 重新归一化，并确保在正确的设备上
    activated_traj = model.get_normalizer().normalize(activated_traj).to(device)
    
    Logger.log(f'[激活静止车辆] 成功激活 {len(activated_vehicles)} 辆车辆')
    
    return activated_traj, activated_vehicles


def perturb_stationary_z(z, stationary_vehicles, perturbation_scale=1.0):
    """
    给静止车辆的潜变量 z 添加扰动，以鼓励模型生成运动轨迹。
    
    :param z: (NA, z_dim) 潜变量
    :param stationary_vehicles: 静止车辆索引列表
    :param perturbation_scale: 扰动幅度
    :return: 扰动后的 z
    """
    if not stationary_vehicles:
        return z
    
    z_perturbed = z.clone()
    
    for v_idx in stationary_vehicles:
        # 添加高斯噪声
        noise = torch.randn_like(z[v_idx]) * perturbation_scale
        z_perturbed[v_idx] = z[v_idx] + noise
    
    return z_perturbed

def detach_embed_info(embed_info_attached):
    embed_info = dict()
    for k, v in embed_info_attached.items():
        if isinstance(v, torch.Tensor):
            # everything else
            embed_info[k] = v.detach()
        elif isinstance(v, tuple):
            # posterior or prior output
            embed_info[k] = (v[0].detach(), v[1].detach())
    return embed_info

def determine_feasibility_nusc(samples, normalizer, feasibility_thresh, 
                                feasibility_time=0, feasibility_vel=0.0,
                                feasibility_infront_min=None,
                                check_non_drivable_separation=True,
                                map_env=None,
                                map_idx=None):
    '''
    Determine whether the given sequences are plausible to seed scenario generation
    by measuring distance between ego and other agents at each step of each sample. 

    NOTE: This assumes the samples are of a SINGLE scene graph, i.e. idx 0 is the ego trajectory
    and all other indices are other agents in the same scene.

    :param samples: NORMALIZED samples from the model for the future (NA x NS x FT x 4)
    :param feasibility_thresh: if agents are ever less than this distance (meters) a part in
                                any sampled timestep, it's considered to be a feasible "seed" past.
    :param feasibility_time: must be >= feasibility time to be considered feasible. 
    :param feasibility_vel: maximum velocity of sampled trajectory for an agent must be >= this thresh
                            to be considered feasible.
    :param feasibility_infront_min: in [-1, 1]. if not None, dot product between ego heading and vector from ego to
                                    potential attacker must be >= this threshold in order to be condiered
                                    feasible. i.e encourage attacker to be in front of ego.
    :param check_non_drivable_separation: if true, only feasible if samples of ego/others that result in
                                            minimum distance are not separated by non-drivable area.
    :param map_env: env
    :param map_idx: (1, )

    :return feasible: (NA-1, ) True if the agent comes within the feasibility thresh, False otherwise
    :return feasible_time_step: (NA-1, ) the timestep index at which each agent comes closest to the ego.
    :return feasible: (NA-1, ) how far agent is from planner.
    '''
    if samples.size(0) == 1:
        # we only have ego, so it's not feasible
        return None, None, None
    samples = normalizer.unnormalize(samples)
    ego_samples = samples[0:1, :, :, :]
    agent_samples = samples[1:, :, :, :]
    NA, NS, FT, _ = agent_samples.size()
    ego_agent_dists = torch.norm(ego_samples[:,:,:,:2] - agent_samples[:,:,:,:2], dim=-1) # (NA-1, NS, FT)
    ego_agent_dists = ego_agent_dists[:, :, feasibility_time:]

    if feasibility_infront_min is not None:
        # mask out steps where agents are behind ego up to some threshold
        assert(feasibility_infront_min >= -1)
        assert(feasibility_infront_min <= 1)
        ego_h = ego_samples[:, :, feasibility_time:, 2:4]
        ego_pos = ego_samples[:, :, feasibility_time:, :2]
        agent_pos = agent_samples[:, :, feasibility_time:, :2]

        ego2agent = agent_pos - ego_pos
        ego2agent = ego2agent / torch.norm(ego2agent, dim=-1, keepdim=True)
        cossim = torch.sum(ego2agent * ego_h, dim=-1) # (NA-1, NS, T')
        infront = cossim >= feasibility_infront_min
        ego_agent_dists[~infront] = float('inf') # make sure they are filtered out

    min_samp_dists, min_samp_inds = torch.min(ego_agent_dists, dim=1) # (NA-1, FT)
    feasible_dist, feasible_time_step = torch.min(min_samp_dists, dim=1) # (NA-1, )
    feasible_time_step = feasible_time_step + feasibility_time
    feasible = (ego_agent_dists < feasibility_thresh).sum(dim=[1, 2]) > 0

    if check_non_drivable_separation:
        min_samp_inds = min_samp_inds[torch.arange(NA), feasible_time_step-feasibility_time]
        # the trajectories corresponding to minimum distances when passing ego
        min_samp_agent_trajs = agent_samples[torch.arange(NA), min_samp_inds]
        min_samp_ego_trajs = ego_samples.expand(NA, NS, FT, 4)[torch.arange(NA), min_samp_inds]
        min_samp_agent_state = min_samp_agent_trajs[torch.arange(NA), feasible_time_step][:, :2]
        min_samp_ego_state = min_samp_ego_trajs[torch.arange(NA), feasible_time_step][:, :2]
        
        intersect_feasible = nutils.check_line_layer(map_env.nusc_raster[:, 0], map_env.nusc_dx, 
                                                        min_samp_agent_state, min_samp_ego_state, 
                                                        map_idx.expand(NA))
        feasible = torch.logical_and(feasible, ~intersect_feasible)

    agent_vels = torch.norm(agent_samples[:, :, 1:, :2] - agent_samples[:, :, :-1, :2], dim=-1) # (NA-1, NS, FT-1)
    max_vels = torch.max(torch.max(agent_vels, dim=1)[0], dim=1)[0] # (NA-1, )
    feasible = torch.logical_and(feasible, max_vels > feasibility_vel)

    return feasible, feasible_time_step, feasible_dist


METRIC_NAMES = ['planner_coll_atk', 'planner_coll_others', 'adv_success',
                'planner_coll_h', 'planner_coll_ang', 'planner_coll_env',
                'veh_coll_rate', 'env_coll_atk', 'env_coll_others',
                'match_ext_pos', 'match_ext_ang',
                'z_ll_atk', 'z_ll_internal', 'z_ll_planner',
                'init_pos_diff_atk', 'init_pos_diff_others']

def log_metric(metric_dict, stat_str, metric_np):
    if stat_str not in metric_dict:
        metric_dict[stat_str] = []
    metric_dict[stat_str] += metric_np.tolist()
    return metric_dict

def log_freq_stat(freq_dict_cnt, freq_dict_total, stat_str, cnt_add, tot_add):
    if stat_str not in freq_dict_cnt:
        freq_dict_cnt[stat_str] = 0
        freq_dict_total[stat_str] = 0
    freq_dict_cnt[stat_str] += cnt_add
    freq_dict_total[stat_str] += tot_add
    return freq_dict_cnt, freq_dict_total

def print_metrics(metrics, freq_metrics_cnt, freq_metrics_total):
    for k, v in metrics.items():
        Logger.log('%s = %f' % (k, np.mean(v)))
    for k, v in freq_metrics_cnt.items():
        Logger.log('%s = %f' % (k, float(v) / freq_metrics_total[k]))

def wandb_log_metrics(metrics, freq_metrics_cnt, freq_metrics_total):
    import wandb
    wandb_metrics = {}
    for k, v in metrics.items():
        wandb_metrics[k] = np.mean(v)
    for k, v in freq_metrics_cnt.items():
        wandb_metrics[k] = float(v) / freq_metrics_total[k]
    wandb.log(wandb_metrics)


def detect_all_collision_vehicles(traj, lw, collision_thresh=0.02):
    """
    检测所有涉及碰撞的车辆。
    
    :param traj: (N x T x 4) 未归一化的轨迹
    :param lw: (N x 2) 未归一化的车辆长宽
    :param collision_thresh: IoU 阈值
    :return: list of vehicle indices that are involved in any collision
    """
    from shapely.geometry import Polygon
    
    if traj is None or lw is None:
        return []
    
    NA, FT, _ = traj.shape
    traj_np = traj.cpu().numpy() if torch.is_tensor(traj) else traj
    lw_np = lw.cpu().numpy() if torch.is_tensor(lw) else lw
    
    collision_vehicles = set()
    poly_cache = dict()
    
    for ai in range(NA):
        for aj in range(ai + 1, NA):
            for t in range(FT):
                # 获取或计算 ai 的多边形
                if (ai, t) not in poly_cache:
                    ai_state = traj_np[ai, t, :]
                    if np.any(np.isnan(ai_state)):
                        poly_cache[(ai, t)] = None
                        continue
                    ai_corners = nutils.get_corners(ai_state, lw_np[ai])
                    ai_poly = Polygon(ai_corners)
                    poly_cache[(ai, t)] = ai_poly
                else:
                    ai_poly = poly_cache[(ai, t)]
                    if ai_poly is None:
                        continue
                
                # 获取或计算 aj 的多边形
                if (aj, t) not in poly_cache:
                    aj_state = traj_np[aj, t, :]
                    if np.any(np.isnan(aj_state)):
                        poly_cache[(aj, t)] = None
                        continue
                    aj_corners = nutils.get_corners(aj_state, lw_np[aj])
                    aj_poly = Polygon(aj_corners)
                    poly_cache[(aj, t)] = aj_poly
                else:
                    aj_poly = poly_cache[(aj, t)]
                    if aj_poly is None:
                        continue
                
                # 计算 IoU
                try:
                    cur_iou = ai_poly.intersection(aj_poly).area / ai_poly.union(aj_poly).area
                    if cur_iou > collision_thresh:
                        # 两个车辆都标记为碰撞
                        collision_vehicles.add(ai)
                        collision_vehicles.add(aj)
                        break  # 找到碰撞后跳出时间循环
                except:
                    continue
    
    return list(collision_vehicles)


def viz_optim_results(out_path, scene_graph, map_idx, map_env,
                        model, future_pred, planner_name, attack_agt, crop_t,
                        viz_bounds=[-60.0, -60.0, 60.0, 60.0],
                        bidx=0,
                        ow_gt=None,
                        show_gt=False,
                        show_gt_idx=None,
                        save_timestamps=None,
                        keep_frames=False,
                        hide_stationary=False,
                        bev_mode=False):
    '''
    attack_agt is LOCAL - i.e. with respect to the subgraph at batch index bidx
    
    :param hide_stationary: 如果为True，静止车辆将使用灰色和低透明度显示
    '''
    NA = scene_graph.ptr[bidx+1] - scene_graph.ptr[bidx]
    
    # 获取时间步长
    dt = scene_graph.dt if hasattr(scene_graph, 'dt') else 0.5
    
    # 检测所有碰撞车辆和静止车辆
    collision_vehicles = None
    stationary_vehicles = None
    if future_pred is not None:
        try:
            # 获取当前batch的轨迹和lw
            start_idx = scene_graph.ptr[bidx]
            end_idx = scene_graph.ptr[bidx + 1]
            
            # 处理 future_pred 的形状
            if len(future_pred.shape) == 4:
                # (NA, NS, T, 4) -> 取第一个样本
                local_traj = future_pred[start_idx:end_idx, 0, :, :]
            else:
                # (NA, T, 4)
                local_traj = future_pred[start_idx:end_idx, :, :]
            
            local_lw = scene_graph.lw[start_idx:end_idx]
            
            # 反归一化
            local_traj_unnorm = model.get_normalizer().unnormalize(local_traj)
            local_lw_unnorm = model.get_att_normalizer().unnormalize(local_lw)
            
            # 检测碰撞车辆
            collision_vehicles = detect_all_collision_vehicles(local_traj_unnorm, local_lw_unnorm)
            if collision_vehicles:
                Logger.log(f'[VIZ] 检测到碰撞车辆: {collision_vehicles}')
            
            # 检测静止车辆
            stationary_vehicles = nutils.detect_stationary_vehicles(local_traj_unnorm, velocity_threshold=0.5, dt=dt)
            if stationary_vehicles:
                Logger.log(f'[VIZ] 检测到静止/停放车辆: {stationary_vehicles} (共{len(stationary_vehicles)}辆)')
            
        except Exception as e:
            Logger.log(f'[VIZ] 车辆检测失败: {e}')
            collision_vehicles = None
            stationary_vehicles = None
    
    # 生成颜色方案（包含碰撞车辆的红色标注，静止车辆的灰色标注）
    car_colors = nutils.get_adv_coloring(
        NA, attack_agt, 0, 
        collision_vehicles=collision_vehicles,
        stationary_vehicles=stationary_vehicles if hide_stationary else None
    )
    
    nutils.viz_scene_graph(scene_graph, map_idx, map_env, bidx, out_path,
                                model.get_normalizer(), model.get_att_normalizer(),
                                future_pred=future_pred,
                                viz_traj=True,
                                make_video=True,
                                show_gt=show_gt,
                                show_gt_idx=show_gt_idx,
                                viz_bounds=viz_bounds,
                                crop_t=crop_t,
                                center_viz=crop_t is None,
                                car_colors=car_colors,
                                ow_gt=ow_gt,
                                save_timestamps=save_timestamps,
                                bev_mode=bev_mode
                                )
    nutils.viz_scene_graph(scene_graph, map_idx, map_env, bidx, out_path + '_vid',
                                model.get_normalizer(), model.get_att_normalizer(),
                                future_pred=future_pred,
                                viz_traj=False,
                                make_video=True,
                                show_gt=show_gt,
                                show_gt_idx=show_gt_idx,
                                viz_bounds=viz_bounds,
                                crop_t=crop_t,
                                center_viz=crop_t is None,
                                car_colors=car_colors,
                                ow_gt=ow_gt,
                                save_timestamps=save_timestamps,
                                keep_frames=keep_frames,
                                bev_mode=bev_mode
                                )

def prepare_output_dict(scene_graph, map_idx, map_env, dt, model,
                        init_fut_traj,
                        adv_fut_traj,
                        attack_agt=None,
                        attack_t=None,
                        adv_z=None,
                        prior_distrib=None,
                        attack_bike_params=None,
                        internal_ego_traj=None):
    out_dict = {'N' :  int(init_fut_traj.size(0)), 'dt' : dt}
    map_name = map_env.map_list[map_idx]
    out_dict['map'] = map_name

    # unnormalize trajectories and lw
    normalizer = model.get_normalizer()
    past = normalizer.unnormalize(scene_graph.past_gt)
    init_fut_traj = normalizer.unnormalize(init_fut_traj)
    adv_fut_traj = normalizer.unnormalize(adv_fut_traj)
    lw = model.get_att_normalizer().unnormalize(scene_graph.lw)

    # vehicle attributes
    lw_out = lw.cpu().numpy()
    out_dict['lw'] = lw_out.tolist()
    sem_out = scene_graph.sem.cpu().numpy()
    out_dict['sem'] = sem_out.tolist()

    # past motion (shared among all trajectories)
    past_out = past.cpu().numpy()
    out_dict['past'] = past_out.tolist()
    # initialization
    init_out = init_fut_traj.cpu().numpy()
    out_dict['fut_init'] = init_out.tolist()
    # adversarial scene
    adv_out = adv_fut_traj.cpu().numpy()
    out_dict['fut_adv'] = adv_out.tolist()
    if internal_ego_traj is not None:
        internal_ego_traj = normalizer.unnormalize(internal_ego_traj)
        internal_ego_out = internal_ego_traj.cpu().numpy()
        out_dict['fut_internal_ego'] = internal_ego_out.tolist()
    # attackers and t
    if attack_agt is not None:
        out_dict['attack_agt'] = int(attack_agt)
    if attack_t is not None:
        out_dict['attack_t'] = int(attack_t)
    # latents
    if adv_z is not None:
        out_dict['z_adv'] = adv_z.detach().cpu().numpy().tolist()
    if prior_distrib is not None:
        prior_mean = prior_distrib[0].cpu().numpy()
        prior_var = prior_distrib[1].cpu().numpy()
        out_dict['z_prior'] = {'mean' : prior_mean.tolist(), 'var' : prior_var.tolist()}
    
    # bicycle acceleration profile for baselines
    if attack_bike_params is not None:
        out_dict['attack_bike_prof'] = attack_bike_params.cpu().numpy().tolist()

    return out_dict
