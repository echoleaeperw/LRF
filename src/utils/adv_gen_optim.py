# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch import nn

from losses.traffic_model import compute_coll_rate_env
from losses.adv_gen_nusc import interp_traj

from utils.transforms import transform2frame
from utils.logger import Logger, throw_err
from utils.scenario_gen import log_metric, log_freq_stat, viz_optim_results

def collate_tgt_other_z(scene_graph, tgt_z, other_z):
    '''
    Combines latents into the correct full scene graph structure with tgt at the 0 index of
    each scene graph.
    :param tgt_z: (B, NS, D) or (B, D)
    :param other_z: (NA-B, NS, D) or (NA-B, D)
    '''
    B = tgt_z.size(0)
    if len(tgt_z.size()) == 3:
        cur_z = torch.empty((0, other_z.size(1), other_z.size(2))).to(other_z)
    elif len(tgt_z.size()) == 2:
        cur_z = torch.empty((0, other_z.size(1))).to(other_z)
    prev_idx = 0
    for bidx in range(B):
        cur_size = scene_graph.ptr[bidx+1] - scene_graph.ptr[bidx] - 1
        cur_z = torch.cat([cur_z, tgt_z[bidx:bidx+1], other_z[prev_idx:prev_idx+cur_size]], dim=0)
        prev_idx += cur_size
    return cur_z


def run_adv_gen_optim(cur_z, lr, loss_weights, model, scene_graph, map_env, map_idx,
                    num_iters, embed_info, planner_name, tgt_prior_distrib, other_prior_distrib,
                    feasibility_time, feasibility_infront_min,
                    planner=None, planner_viz_out=None,
                    attack_agt_idx=None,
                    future_len=None,
                    veh_coll_buffer=10,
                    ttc_epsilon=1e-6,
                    dt=0.1,
                    ttc_safe=3.0,
                    min_dist_lat_k=2.0,
                    min_dist_lat_gap=0.5,
                    yaw_rate_threshold=15.0):
    '''
    :param attack_agt_idx: list of attackers LOCAL to each scene in the batched graph.
    :param future_len: if given, rolls out scenario future this many steps rather than the default of the model
    :param ttc_epsilon: Prevent division by zero in TTC calculation
    :param dt: Time step for calculating velocity and yaw rate
    :param ttc_safe: TTC safe threshold (seconds)
    :param min_dist_lat_k: Control the severity of lateral distance loss penalty
    :param min_dist_lat_gap: Lateral minimum safe gap (meters)
    :param yaw_rate_threshold: Comfort threshold for yaw rate (degrees/second)
    '''
    B = map_idx.size(0)
    NA = cur_z.size(0)
    ego_inds = scene_graph.ptr[:-1]
    ego_mask = torch.zeros((NA), dtype=torch.bool)
    ego_mask[ego_inds] = True

    if attack_agt_idx is not None:
        attack_agt_idx = torch.tensor(attack_agt_idx).to(scene_graph.ptr)
        scene_sizes = scene_graph.ptr[1:] - scene_graph.ptr[:-1]
        # 局部索引必须 > 0（排除 ego）且 < 场景内 agent 数
        lower_bad = attack_agt_idx <= 0
        upper_bad = attack_agt_idx >= scene_sizes
        if lower_bad.any() or upper_bad.any():
            Logger.log(f'│  攻击者局部索引越界 (local={attack_agt_idx.tolist()}, '
                       f'sizes={scene_sizes.tolist()})，回退soft-min')
            attack_agt_idx = None
        else:
            attack_agt_idx = attack_agt_idx + scene_graph.ptr[:-1]  # local → global

    if future_len is None:
        future_len = model.FT

    # ── 优化变量设置 ────────────────────────────────────────────────────────
    # ego/planner z: 始终可训练（match loss）
    tgt_z = cur_z[ego_mask].clone().detach()
    tgt_z.requires_grad = True

    # 非ego车辆 z 的初始值（用于冻结背景车）
    other_z_all_init = cur_z[~ego_mask].clone().detach()

    if attack_agt_idx is not None:
        # 只优化 LLM 识别的攻击者；背景车 z 硬冻结（不进入 optimizer）
        atk_mask_global = torch.zeros(NA, dtype=torch.bool, device=cur_z.device)
        atk_mask_global[attack_agt_idx] = True
        atk_mask_nonego = atk_mask_global[~ego_mask]   # (NA-B,) 攻击者在非ego空间的mask
        atk_pos = atk_mask_nonego.nonzero(as_tuple=False).flatten()  # 攻击者行索引

        atk_z = other_z_all_init[atk_mask_nonego].clone()
        atk_z.requires_grad = True

        optim_z = [tgt_z, atk_z]
        atk_only_mode = True
        Logger.log(f'│  优化模式: 仅攻击者z可训练  '
                   f'(攻击者={atk_mask_nonego.sum().item()}, 冻结={int((~atk_mask_nonego).sum())})' )
    else:
        # 未指定攻击者：回退至全部非ego优化（原始行为）
        other_z_all_init.requires_grad = True
        atk_z = other_z_all_init
        atk_mask_nonego = None
        atk_pos = None
        optim_z = [tgt_z, atk_z]
        atk_only_mode = False

    cur_z = collate_tgt_other_z(scene_graph, tgt_z,
                                 other_z_all_init if not atk_only_mode else
                                 other_z_all_init.detach())

    adv_optim = optim.Adam(optim_z, lr=lr)

    # create loss functions
    from losses.adv_gen_nusc import TgtMatchingLoss, AdvGenLoss       
    tgt_loss = TgtMatchingLoss(loss_weights)
    adv_loss = AdvGenLoss(loss_weights,
                            model.get_att_normalizer().unnormalize(scene_graph.lw),
                            map_idx[scene_graph.batch],
                            map_env,
                            cur_z[~ego_mask].clone().detach(),
                            scene_graph.ptr,
                            veh_coll_buffer=veh_coll_buffer,
                            crash_loss_min_time=feasibility_time,
                            crash_loss_min_infront=feasibility_infront_min,
                            ttc_epsilon=ttc_epsilon,
                            dt=dt,
                            ttc_safe=ttc_safe,
                            min_dist_lat_k=min_dist_lat_k,
                            min_dist_lat_gap=min_dist_lat_gap,
                            yaw_rate_threshold=yaw_rate_threshold)

    if planner_name != 'ego':
        # for real planners need to set initial state for rollouts
        all_init_state = model.get_normalizer().unnormalize(scene_graph.past_gt[:, -1, :])
        all_init_veh_att = model.get_att_normalizer().unnormalize(scene_graph.lw)
        planner.reset(all_init_state, all_init_veh_att, scene_graph.batch, B, map_idx)

    # default to open-loop
    planner_inject_traj = True # whether to use observed planner future rather than internal prediction during decoder rollout
    adv_use_own_pred = False   # whether to use our differentiable approx of planner rather than real observations to compute loss
    # if using hardcode, is closed-loop
    if planner_name == 'hardcode':
        Logger.log('NOTE: Operating in closed-loop for adv gen optimization!')
        planner_inject_traj = False
        adv_use_own_pred = True

        cur_agt_ptr = scene_graph.ptr - torch.arange(B+1, device=scene_graph.ptr.device)
        plan_t = np.linspace(model.dt, model.dt*future_len, future_len)

    # run optim
    pbar_optim = tqdm.tqdm(range(num_iters))
    for oidx in pbar_optim:
        def closure():
            adv_optim.zero_grad()

            # decode to get current future and compute loss
            loss = loss_dict = planner_fut = None
            if planner_name == 'ego':
                # in open-loop operation: already rolled out planner a single time, just try to attack.
                # for ego planner, take GT future traj
                planner_fut = scene_graph.future_gt[ego_mask][:, :, :4]
                # if we're injecting external, must be the same length as rollout
                assert planner_inject_traj is False or planner_fut.size(1) == future_len

            # ── 重建完整 other_z ─────────────────────────────────────────────
            if atk_only_mode:
                # 背景车保持冻结初始值；攻击者使用可训练 atk_z
                # index_put (out-of-place) 对 atk_z 位置可微，对冻结背景不产生梯度
                other_z = other_z_all_init.detach().index_put((atk_pos,), atk_z)
            else:
                other_z = atk_z  # == other_z_all_init (requires_grad=True)

            tgt_loss_input_z = collate_tgt_other_z(scene_graph, tgt_z, other_z.clone().detach())
            other_loss_input_z = collate_tgt_other_z(scene_graph, tgt_z.clone().detach(), other_z)

            # forward pass to compute target-specific loss
            tgt_decoder_out = model.decode_embedding(tgt_loss_input_z, embed_info, scene_graph, map_idx, map_env,
                                                ext_future=planner_fut if planner_inject_traj else None,
                                                nfuture=future_len)
            # forward pass to compute controlled agent loss
            other_decoder_out = model.decode_embedding(other_loss_input_z, embed_info, scene_graph, map_idx, map_env,
                                                ext_future=planner_fut if planner_inject_traj else None,
                                                nfuture=future_len)

            # rollout planner allowing it to react to the current model rollout
            if planner_name == 'hardcode':
                cur_agt_pred = tgt_decoder_out['future_pred'][~ego_mask]
                cur_agt_pred = model.normalizer.unnormalize(cur_agt_pred).detach().cpu().numpy()
                planner_fut = planner.rollout(cur_agt_pred, plan_t, cur_agt_ptr.cpu().numpy(), plan_t,
                                                # viz=True,
                                                control_all=False).to(scene_graph.future_gt)
                planner_fut = model.get_normalizer().normalize(planner_fut)

            # compute each loss
            loss_dict = dict()
            tgt_match_loss_dict = tgt_loss(model.get_normalizer().unnormalize(tgt_decoder_out['future_pred'][ego_mask]),
                                            model.get_normalizer().unnormalize(planner_fut),
                                            tgt_z,
                                            tgt_prior_distrib)
            loss_dict = {'tgt_match_' + k : v for k, v in tgt_match_loss_dict.items()}

            tgt_traj = planner_fut if not adv_use_own_pred else other_decoder_out['future_pred'][ego_mask]
            adv_loss_dict = adv_loss(model.get_normalizer().unnormalize(other_decoder_out['future_pred']),
                                        model.get_normalizer().unnormalize(tgt_traj),
                                        other_z,
                                        other_prior_distrib,
                                        attack_agt_idx=attack_agt_idx)
            adv_loss_dict = {'adv_' + k : v for k, v in adv_loss_dict.items()}
            loss_dict = {**loss_dict, **adv_loss_dict}

            # add together to get single loss
            loss = loss_dict['tgt_match_loss'] + loss_dict['adv_loss']

            progress_bar_metrics = {}
            for k, v in loss_dict.items():
                if v is None:
                    continue
                progress_bar_metrics[k] = torch.mean(v).item()
            pbar_optim.set_postfix(progress_bar_metrics)

            # backprop
            loss.backward()
            return loss

        # update
        closure()
        adv_optim.step()

    # get final results: rebuild other_z one last time (no grad needed)
    with torch.no_grad():
        if atk_only_mode:
            other_z_final = other_z_all_init.detach().index_put((atk_pos,), atk_z.detach())
        else:
            other_z_final = atk_z.detach()
    cur_z = collate_tgt_other_z(scene_graph, tgt_z, other_z_final)
    with torch.no_grad():
        final_decoder_out = model.decode_embedding(cur_z, embed_info, scene_graph, map_idx, map_env, nfuture=future_len)
    final_result_traj = final_decoder_out['future_pred'].unsqueeze(1).clone().detach()
    if planner_name == 'ego':
        # replace the ego (planner) traj with GT
        final_result_traj[ego_inds, torch.zeros_like(ego_inds)] = scene_graph.future_gt[ego_mask][:, :, :4]
    elif planner_name == 'hardcode':
        # replace ego (planner) traj with actual output of planner
        cur_agt_pred = final_decoder_out['future_pred'][~ego_mask]
        cur_agt_pred = model.normalizer.unnormalize(cur_agt_pred).detach().cpu().numpy()
        planner_fut = planner.rollout(cur_agt_pred, plan_t, cur_agt_ptr.cpu().numpy(), plan_t,
                                        viz=planner_viz_out,
                                        control_all=False).to(scene_graph.future_gt)
        planner_fut = model.get_normalizer().normalize(planner_fut)
        final_result_traj[ego_inds, torch.zeros_like(ego_inds)] = planner_fut

    # compute loss one more time to get final min agt/t
    tgt_traj = final_result_traj[ego_inds, torch.zeros_like(ego_inds)] # the true planner rollout
    adv_loss_dict = adv_loss(model.get_normalizer().unnormalize(final_decoder_out['future_pred']),
                                model.get_normalizer().unnormalize(tgt_traj),
                                cur_z[~ego_mask].clone().detach(),
                                other_prior_distrib,
                                return_mins=True,
                                attack_agt_idx=attack_agt_idx)
    cur_min_agt = cur_min_t = None
    if 'min_agt' in adv_loss_dict:
        cur_min_agt = adv_loss_dict['min_agt'] + scene_graph.ptr[:-1].cpu().numpy()
    if 'min_t' in adv_loss_dict:
        cur_min_t = adv_loss_dict['min_t']

    return cur_z, final_result_traj, final_decoder_out, cur_min_agt, cur_min_t


def compute_adv_gen_success(final_result_traj, model, scene_graph, attack_agt):
    '''
    Computes whether scenario is successful in colliding w/ planner
    All inputs assumed NORMALIZED.
    :param final_result_traj: (NA, 1, FT, 4) the final scenario where the agent at idx=0 is the true planner
                                            reaction to the scenario (NOT the model's prediction of the planner)
    '''
    from losses.adv_gen_nusc import check_single_veh_coll

    # final result trajectories
    planner_gt_fut = model.get_normalizer().unnormalize(final_result_traj[0, 0])
    other_fut = model.get_normalizer().unnormalize(final_result_traj[1:, 0])
    # agent attribs
    planner_lw = model.get_att_normalizer().unnormalize(scene_graph.lw[0])
    other_lw = model.get_att_normalizer().unnormalize(scene_graph.lw[1:])

    # Collision with planner target
    planner_coll_all, planner_coll_time = check_single_veh_coll(planner_gt_fut, planner_lw, other_fut, other_lw)
    attack_coll = planner_coll_all[attack_agt-1]
    adv_success = bool(attack_coll)

    return adv_success