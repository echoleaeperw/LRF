# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
from torch import nn

import numpy as np
import math

from losses.common import log_normal
from utils.transforms import transform2frame
import datasets.nuscenes_utils as nutils

class TgtMatchingLoss(nn.Module):
    '''
    Loss to encourage future pred to be close to some target while staying
    likely under the motion prior.
    '''
    def __init__(self, loss_weights):
        '''
        :param loss_weights: dict of weightings for loss terms
        '''
        super(TgtMatchingLoss, self).__init__()
        self.loss_weights = loss_weights
        self.motion_prior_loss = MotionPriorLoss()

    def forward(self, future_pred, tgt_traj, z, prior_out):
        '''
        :param future_pred: NA x T x 4 UNNORMALIZED
        :param tgt_traj: NA x T x 4 UNNORMALIZED
        :param z: NA x D
        :param prior_out: tuple of (mean, var) each of size (NA x D)
        '''
        loss_out = {}
        loss = 0.0

        if self.loss_weights['match_ext'] > 0.0:
            # matching error
            tgt_loss = torch.sum((future_pred - tgt_traj)**2, dim=-1)
            loss = loss + self.loss_weights['match_ext']*tgt_loss.mean()
            loss_out['match_ext_loss'] = tgt_loss

        if self.loss_weights['motion_prior_ext'] > 0.0:
            # motion prior
            motion_prior_loss = self.motion_prior_loss(z, prior_out)
            loss = loss + self.loss_weights['motion_prior_ext']*motion_prior_loss.mean()
            loss_out['motion_prior_ext_loss'] = motion_prior_loss

        loss_out['loss'] = loss

        return loss_out

class AdvGenLoss(nn.Module):
    '''
    Loss to encourage agents to create adversarial scenario for a given target.
    '''
    def __init__(self, loss_weights, veh_att, mapixes, map_env, init_z, ptr,
                    veh_coll_buffer=0.0,
                    crash_loss_min_time=0,
                    crash_loss_min_infront=None,
                    ttc_epsilon=1e-6,
                    dt=0.1,
                    ttc_safe=3.0,
                    min_dist_lat_k=2.0,
                    min_dist_lat_gap=0.5,
                    yaw_rate_threshold=15.0
                    ):
        '''
        :param loss_weights: dict of weightings for loss terms
        :param veh_att: UNNORMALIZED lw for the vehicles that will be computing loss for (NA x 2)
        :param mapixes: map index corresponding to ALL agents (NA,)
        :param map_env: the map environment holding map info
        :param init_z: (NA-B) x D initial latent z of all non-target agents
        :param crash_loss_min_time: only computes loss using times past this threshold
        :param crash_loss_min_infront: if not None [-1, 1], any attacker with cosine similarity < this threshold will
                                        be ignored (i.e. provide no loss to cause a crash) if attacker is being optimized
        :param ttc_epsilon: 防止TTC计算中除零的小常数
        :param dt: 时间步长，用于计算速度和角速度
        :param ttc_safe: TTC安全阈值(秒)
        :param min_dist_lat_k: 控制横向距离攻击性损失的系数
        :param min_dist_lat_gap: 横向最小安全间隙(米) - 在此攻击性版本中未使用
        :param yaw_rate_threshold: 横摆角速度攻击阈值(度/秒)，攻击旨在超过此阈值
        '''
        super(AdvGenLoss, self).__init__()
        self.loss_weights = loss_weights
        self.init_z = init_z
        self.motion_prior_loss = MotionPriorLoss()
        # these are only computed on non-planner vehicles
        self.ptr = ptr
        self.graph_sizes = self.ptr[1:] - self.ptr[:-1]
        self.ego_mask = torch.zeros((veh_att.size(0)), dtype=torch.bool)
        self.ego_mask[self.ptr[:-1]] = True
        # vehicle collision will only be on non-ego, so need to adjust ptr accordingly
        self.nonego_ptr = self.ptr - torch.arange(len(self.ptr), device=self.ptr.device)
        self.veh_coll_loss = VehCollLoss(veh_att,
                                          buffer_dist=veh_coll_buffer,
                                          ptr=self.ptr)
        self.env_coll_loss = EnvCollLoss(veh_att[~self.ego_mask], mapixes[~self.ego_mask], map_env)
        self.crash_min_t = crash_loss_min_time
        self.crash_min_infront = crash_loss_min_infront
        if self.crash_min_infront is not None:
            assert(self.crash_min_infront >= -1)
            assert(self.crash_min_infront <= 1)
            
        # 初始化新的损失函数
        # 对抗型TTC（鼓励更小TTC）在本类中使用；AvoidCollLoss仍使用安全型TTC
        self.ttc_loss_atk = TTCLossAtk(epsilon=ttc_epsilon, dt=dt)
        self.min_dist_lat_loss = MinDistLatLoss(k=min_dist_lat_k)
        self.yaw_rate_loss = YawRateLoss(dt=dt, comfort_threshold_deg=yaw_rate_threshold)
        self.veh_att = veh_att  # 存储车辆尺寸，用于计算横向距离

    def forward(self, future_pred, tgt_traj, z, prior_out,
                    return_mins=False,
                    attack_agt_idx=None):
        ''' 
        :param future_pred: NA x T x 4 UNNORMALIZED output of the motion model where idx=0 is modeling the planner
        :param tgt_traj: B x T x 4 UNNORMALIZED planner trajectory to attack.
        :param z: (NA-B) x D latents for all non-planner agents (potential attackers)
        :param prior_out: tuple of (mean, var) each of size (NA-B) x D
        :param return_mins: returns indices of the current "most likely" of these
        :param attack_agt_idx: list of indices within each scene graph to use as the attacker. These should be global
                                    to the entire batched scene graph.
        '''
        NA = future_pred.size(0)
        B = tgt_traj.size(0)
        adv_crash_loss = min_dist = dist_traj = None
        cur_min_agt = cur_min_t = None
        if 'adv_crash' in self.loss_weights and self.loss_weights['adv_crash'] > 0.0:
            # minimize POSITIONAL distance
            attacker_pred = future_pred[~self.ego_mask][:, self.crash_min_t:, :] # (NA-B, T, 4)
            tgt_pred = tgt_traj[:, self.crash_min_t:, :4]
            tgt_expanded = torch.cat([tgt_pred[b:b+1, :, :].expand(self.graph_sizes[b]-1, -1, -1) for b in range(B)], dim=0)                
            dist_traj = torch.norm(attacker_pred[:,:,:2] - tgt_expanded[:,:,:2], dim=-1) # (NA-B, T)
            min_dist_in = dist_traj
            if self.crash_min_infront is not None:
                behind_steps = check_behind(attacker_pred.detach(), tgt_pred.detach(), self.ptr, self.crash_min_infront)
                behind_traj = torch.sum(behind_steps, dim=1, keepdim=True) == behind_steps.size(1)
                behind_traj = behind_traj.expand_as(behind_steps)
                if torch.sum(behind_traj.reshape((-1))) == behind_steps.size(0)*behind_traj.size(1):
                    # every agent is behind... just optim normally for now
                    behind_traj = torch.zeros_like(behind_traj).to(torch.bool)
                min_dist_in = torch.where(behind_traj, float('inf')*torch.ones_like(min_dist_in), min_dist_in) # set to 0 weight for agent behind the target

            if attack_agt_idx is not None:
                print(f"[DEBUG] Using LLM-specified attack_agt_idx: {attack_agt_idx.cpu().numpy() if torch.is_tensor(attack_agt_idx) else attack_agt_idx}")
                attack_agt_mask = torch.zeros(future_pred.size(0), dtype=torch.bool).to(attack_agt_idx.device)
                attack_agt_mask[attack_agt_idx] = True
                attack_agt_mask = attack_agt_mask[~self.ego_mask].unsqueeze(1).expand_as(min_dist_in)
                print(f"[DEBUG] attack_agt_mask applied: {torch.sum(attack_agt_mask).item()} elements selected")
                min_dist_in = torch.where(~attack_agt_mask, float('inf')*torch.ones_like(min_dist_in), min_dist_in) # set to 0 weight for agent behind the target                

            # soft min over all timesteps and agents
            NT = future_pred.size(1) - self.crash_min_t
            min_dist = [nn.functional.softmin(min_dist_in[self.nonego_ptr[b]:self.nonego_ptr[b+1]].view(-1), dim=0) for b in range(B)]
            # handle case where all frames for all agents are behind and therefore softmin is nan. (set to all 0 prob)
            min_dist = [bdist if torch.isnan(bdist[0]).item() == False else torch.zeros_like(bdist) for bdist in min_dist]

            cur_min_agt = [(torch.max(min_dist_b, dim=0)[1].item() // NT) + 1 for min_dist_b in min_dist]
            cur_min_t = [(torch.max(min_dist_b, dim=0)[1].item() % NT) + self.crash_min_t for min_dist_b in min_dist]

            min_dist = torch.cat(min_dist, dim=0)
            dist_traj = dist_traj.view(-1)**2 # (NA-B*T)
            weighted_adv_crash = min_dist * dist_traj # (NA-B*T)
            adv_crash_loss = [torch.sum(weighted_adv_crash[(self.nonego_ptr[b]*NT):(self.nonego_ptr[b+1]*NT)]) for b in range(B)]
            adv_crash_loss = torch.stack(adv_crash_loss)
            print(adv_crash_loss)

            print('Cur min agt: ' + str(cur_min_agt))
            print('Cur min t: ' + str(cur_min_t))

        # high weight for non attackers
        if min_dist is not None:
            prior_reweight = min_dist.detach().reshape((future_pred.size(0)-B,  -1)) # NA-B x T
            prior_reweight = 1.0 - torch.sum(prior_reweight, dim=1)
        else:
            prior_reweight = torch.ones(future_pred.size(0) - B, device=future_pred.device)

        #
        # motion prior
        #

        motion_prior_loss = None
        if 'motion_prior' in self.loss_weights and self.loss_weights['motion_prior'] > 0.0:
            motion_prior_loss = self.motion_prior_loss(z, prior_out)
            prior_coeff = prior_reweight*self.loss_weights['motion_prior'] + \
                                (1.0 - prior_reweight)*self.loss_weights['motion_prior_atk']
            motion_prior_loss = motion_prior_loss * prior_coeff

        #
        # Regularizers
        #

        # interpolate trajectory to avoid cheating collision tests
        future_pred_interp = interp_traj(future_pred, scale_factor=3)

        # vehicle collision penalty
        veh_coll_loss_val = veh_coll_plan_loss_val = None
        if 'coll_veh' in self.loss_weights or 'coll_veh_plan' in self.loss_weights:
            if self.loss_weights['coll_veh'] > 0.0 or self.loss_weights['coll_veh_plan'] > 0.0:
                veh_coll_pens, veh_coll_mask = self.veh_coll_loss(future_pred_interp, return_raw=True) # only on non-target trajectories

            if self.loss_weights['coll_veh'] > 0.0:
                # need to mask out colls with ego
                non_ego_coll_mask = torch.ones((1, NA, NA), dtype=torch.bool).to(veh_coll_mask.device) # NA x NA
                for b in range(B):
                    non_ego_coll_mask[0, self.ptr[b], :] = False
                    non_ego_coll_mask[0, :, self.ptr[b]] = False
                non_ego_coll_mask = torch.logical_and(veh_coll_mask, non_ego_coll_mask.expand_as(veh_coll_mask))
                if torch.sum(non_ego_coll_mask) == 0:
                    veh_coll_loss_val = torch.Tensor([0.0]).to(veh_coll_pens.device)
                else:
                    veh_coll_loss_val = veh_coll_pens[non_ego_coll_mask]

            if self.loss_weights['coll_veh_plan'] > 0.0:
                # prior_reweight is size (NA-B, )
                # use prior reweight to down-weight possible attackers
                ego_coll_weight = torch.ones((NA)).to(veh_coll_pens.device)
                ego_coll_weight[~self.ego_mask] = prior_reweight
                ego_pen_mat = torch.ones((1, NA, NA)).to(veh_coll_pens.device)
                ego_coll_mask = torch.zeros((1, NA, NA), dtype=torch.bool).to(veh_coll_mask.device) # NA x NA
                for b in range(B):
                    # note this assumes diagonal will be thrown out.
                    ego_pen_mat[0, self.ptr[b], :] = ego_coll_weight
                    ego_pen_mat[0, :, self.ptr[b]] = ego_coll_weight
                    ego_coll_mask[0, self.ptr[b], :] = True
                    ego_coll_mask[0, :, self.ptr[b]] = True
                ego_coll_mask = torch.logical_and(veh_coll_mask, ego_coll_mask.expand_as(veh_coll_mask))
                # directly weight penalties
                veh_plan_weighted_pens = veh_coll_pens * ego_pen_mat.expand_as(veh_coll_pens)
                if torch.sum(ego_coll_mask) == 0:
                    veh_coll_plan_loss_val = torch.Tensor([0.0]).to(veh_coll_pens.device)
                else:
                    veh_coll_plan_loss_val = veh_plan_weighted_pens[ego_coll_mask]

        # env collision penalty
        env_coll_loss_val = None
        if 'coll_env' in self.loss_weights and self.loss_weights['coll_env'] > 0.0: 
            env_coll_loss_val = self.env_coll_loss(future_pred_interp[~self.ego_mask])
        # init loss
        init_loss = None
        if 'init_z' in self.loss_weights and self.loss_weights['init_z'] > 0.0:
            init_loss = torch.sum((self.init_z - z)**2, dim=1)
            # stay close to init for non-attacking vehicles
            init_z_coeff = prior_reweight*self.loss_weights['init_z'] + \
                                (1.0 - prior_reweight)*self.loss_weights['init_z_atk']
            # print(init_z_coeff)
            init_loss = torch.sum(init_loss * init_z_coeff)

        loss = 0.0
        loss_out = {}

        if init_loss is not None:
            # already applied weighting
            loss = loss + init_loss.mean()
            loss_out['init_loss'] = init_loss

        if motion_prior_loss is not None:
            # already applied weighting
            loss = loss + motion_prior_loss.mean() 
            loss_out['motion_prior_loss'] = motion_prior_loss

        if veh_coll_loss_val is not None:
            loss = loss + self.loss_weights['coll_veh']*veh_coll_loss_val.mean()
            loss_out['coll_veh_loss'] = veh_coll_loss_val

        if veh_coll_plan_loss_val is not None:
            loss = loss + self.loss_weights['coll_veh_plan']*veh_coll_plan_loss_val.mean()
            loss_out['coll_veh_plan_loss'] = veh_coll_plan_loss_val

        if env_coll_loss_val is not None:
            loss = loss + self.loss_weights['coll_env']*env_coll_loss_val.mean()
            loss_out['coll_env_loss'] = env_coll_loss_val

        if adv_crash_loss is not None:
            loss = loss + self.loss_weights['adv_crash']*adv_crash_loss.mean()
            loss_out['adv_crash_loss'] = adv_crash_loss

        # 添加新的损失函数计算
        # 先构造对齐后的攻击者与目标配对（按图）
        attacker_pred = future_pred[~self.ego_mask]
        B_local = tgt_traj.size(0)
        tgt_expanded_for_pairs = torch.cat([
            tgt_traj[b:b+1, :, :].expand(self.graph_sizes[b]-1, -1, -1) for b in range(B_local)
        ], dim=0) if B_local > 0 else tgt_traj
        
        # 如果指定了攻击车辆，创建mask来筛选
        attacker_mask_for_loss = None
        if attack_agt_idx is not None:
            print(f"[DEBUG-NEW-LOSS] attack_agt_idx: {attack_agt_idx.cpu().numpy() if torch.is_tensor(attack_agt_idx) else attack_agt_idx}")
            # 创建全局mask
            full_attack_mask = torch.zeros(future_pred.size(0), dtype=torch.bool).to(future_pred.device)
            full_attack_mask[attack_agt_idx] = True
            # 只保留非ego车辆的mask部分
            attacker_mask_for_loss = full_attack_mask[~self.ego_mask]
            print(f"[DEBUG-NEW-LOSS] Total attackers: {attacker_pred.size(0)}, Selected by mask: {attacker_mask_for_loss.sum().item()}")

        # TTC（对抗型）损失：越小越好（以负的 inverse TTC 形式加入）
        ttc_loss_val = None
        if 'ttc' in self.loss_weights and self.loss_weights['ttc'] > 0.0 and attacker_pred.numel() > 0:
            ttc_loss_val = self.ttc_loss_atk(attacker_pred, tgt_expanded_for_pairs)
            if torch.isnan(ttc_loss_val).any():
                ttc_loss_val = torch.where(torch.isnan(ttc_loss_val), torch.zeros_like(ttc_loss_val), ttc_loss_val)
            
            # 如果指定了攻击车辆，只对这些车辆计算损失
            if attacker_mask_for_loss is not None:
                # 将mask扩展到时间维度以匹配ttc_loss_val的形状 [NA-B, T-1]
                mask_expanded = attacker_mask_for_loss.unsqueeze(1).expand_as(ttc_loss_val)
                # 将非攻击车辆的损失置零
                ttc_loss_val_masked = ttc_loss_val * mask_expanded.float()
                # 只对选中的车辆求平均
                if attacker_mask_for_loss.sum() > 0:
                    loss = loss + self.loss_weights['ttc'] * ttc_loss_val_masked.sum() / mask_expanded.sum()
                else:
                    loss = loss + self.loss_weights['ttc'] * ttc_loss_val.mean()
            else:
                loss = loss + self.loss_weights['ttc'] * ttc_loss_val.mean()
            loss_out['ttc_loss'] = ttc_loss_val

        # 横向最小距离损失：只惩罚正向净距（鼓励重叠/贴靠）
        min_dist_lat_loss_val = None
        if 'min_dist_lat' in self.loss_weights and self.loss_weights['min_dist_lat'] > 0.0 and attacker_pred.numel() > 0:
            min_dist_lat_loss_val = self.min_dist_lat_loss(
                attacker_pred, tgt_expanded_for_pairs, self.veh_att, self.ego_mask, self.graph_sizes
            )
            
            # 如果指定了攻击车辆，只对这些车辆计算损失
            if attacker_mask_for_loss is not None:
                # min_dist_lat_loss_val 形状可能是 [NA-B] 或 [NA-B, T]，需要匹配
                if len(min_dist_lat_loss_val.shape) == 1:
                    # 如果是 [NA-B]，直接相乘
                    min_dist_lat_loss_val_masked = min_dist_lat_loss_val * attacker_mask_for_loss.float()
                else:
                    # 如果是 [NA-B, T]，扩展mask
                    mask_expanded = attacker_mask_for_loss.unsqueeze(1).expand_as(min_dist_lat_loss_val)
                    min_dist_lat_loss_val_masked = min_dist_lat_loss_val * mask_expanded.float()
                
                # 只对选中的车辆求平均
                if attacker_mask_for_loss.sum() > 0:
                    num_selected = mask_expanded.sum() if len(min_dist_lat_loss_val.shape) > 1 else attacker_mask_for_loss.sum()
                    loss = loss + self.loss_weights['min_dist_lat'] * min_dist_lat_loss_val_masked.sum() / num_selected
                else:
                    loss = loss + self.loss_weights['min_dist_lat'] * min_dist_lat_loss_val.mean()
            else:
                loss = loss + self.loss_weights['min_dist_lat'] * min_dist_lat_loss_val.mean()
            loss_out['min_dist_lat_loss'] = min_dist_lat_loss_val

        # 横摆角速度损失
        yaw_rate_loss_val = None
        if 'yaw_rate' in self.loss_weights and self.loss_weights['yaw_rate'] > 0.0:
            # 计算所有车辆的横摆角速度损失
            yaw_rate_loss_val = self.yaw_rate_loss(future_pred)
            # 对非目标车辆应用权重
            if not self.ego_mask.all():  # 确保有非目标车辆
                # 分开处理目标车辆和非目标车辆
                ego_yaw_rate = yaw_rate_loss_val[self.ego_mask].mean() if torch.any(self.ego_mask) else 0.0
                
                # 对非目标车辆，如果指定了攻击车辆，只对这些车辆计算
                non_ego_yaw_rate_vals = yaw_rate_loss_val[~self.ego_mask]  # [NA-B, T-1]
                if attacker_mask_for_loss is not None and attacker_mask_for_loss.sum() > 0:
                    # 只计算指定攻击车辆的横摆角速度损失
                    # 扩展mask到时间维度
                    mask_expanded = attacker_mask_for_loss.unsqueeze(1).expand_as(non_ego_yaw_rate_vals)
                    non_ego_yaw_rate = (non_ego_yaw_rate_vals * mask_expanded.float()).sum() / mask_expanded.sum()
                    print(f"[DEBUG-YAW-RATE] Using only {attacker_mask_for_loss.sum().item()} specified attackers for yaw rate loss")
                else:
                    # 所有非目标车辆的平均
                    non_ego_yaw_rate = non_ego_yaw_rate_vals.mean() if torch.any(~self.ego_mask) else 0.0
                
                # 使用不同权重
                if 'yaw_rate_ego' in self.loss_weights:
                    ego_weight = self.loss_weights['yaw_rate_ego']
                else:
                    ego_weight = self.loss_weights['yaw_rate']
                    
                if 'yaw_rate_non_ego' in self.loss_weights:
                    non_ego_weight = self.loss_weights['yaw_rate_non_ego']
                else:
                    non_ego_weight = self.loss_weights['yaw_rate']
                
                # 合并损失
                yaw_rate_weighted = ego_weight * ego_yaw_rate + non_ego_weight * non_ego_yaw_rate
                loss = loss + yaw_rate_weighted
            else:
                # 如果只有目标车辆
                loss = loss + self.loss_weights['yaw_rate'] * yaw_rate_loss_val.mean()
                
            loss_out['yaw_rate_loss'] = yaw_rate_loss_val

        loss_out['loss'] = loss         

        # output attacking agent and time if not set ahead of time
        if return_mins:
            if cur_min_agt is not None:
                loss_out['min_agt'] = np.array(cur_min_agt, dtype=np.int64)
            if cur_min_t is not None:
                loss_out['min_t'] = np.array(cur_min_t, dtype=np.int64)

        return loss_out

class AvoidCollLoss(nn.Module):
    '''
    Loss to discourage vehicle/environment collisions with high likelihood under prior.
    '''
    def __init__(self, loss_weights, veh_att, mapixes, map_env, init_z,
                    veh_coll_buffer=0.0,
                    single_veh_idx=None,
                    ptr=None,
                    ttc_epsilon=1e-6,
                    dt=0.1,
                    ttc_safe=3.0,
                    min_dist_lat_k=2.0,
                    min_dist_lat_gap=0.5,
                    yaw_rate_threshold=15.0,
                    ):
        '''
        :param loss_weights: dict of weightings for loss terms
        :param veh_att: UNNORMALIZED lw for the vehicles that will be computing loss for (NA x 2)
        :param mapixes: map index corresponding to ALL agents (NA,)
        :param map_env: the map environment holding map info
        :param init_z:
        :param veh_coll_buffer: adds extra buffer around vehicles to penalize
        :param single_veh_idx: if not None, computes all losses w.r.t to ONE agent index in each batched scene graph.
                                i.e. if single_veh_idx = 0, only collisions involve agent 0
                                will be included in the computed loss. To use this, MUST also pass in ptr from the scene graph.
        :param ttc_epsilon: 防止TTC计算中除零的小常数
        :param dt: 时间步长，用于计算速度和角速度
        :param ttc_safe: TTC安全阈值(秒)
        :param min_dist_lat_k: 控制横向距离攻击性损失的系数
        :param min_dist_lat_gap: 横向最小安全间隙(米) - 在此攻击性版本中未使用
        :param yaw_rate_threshold: 横摆角速度攻击阈值(度/秒)，攻击旨在超过此阈值
        '''
        super(AvoidCollLoss, self).__init__()
        self.loss_weights = loss_weights
        self.init_z = init_z
        self.single_veh_idx = single_veh_idx
        self.ptr = ptr            
        self.use_single_agt = self.single_veh_idx is not None
        self.motion_prior_loss = MotionPriorLoss()
        self.veh_coll_loss = VehCollLoss(veh_att,
                                          buffer_dist=veh_coll_buffer,
                                          single_veh_idx=self.single_veh_idx,
                                          ptr=self.ptr)
        if self.use_single_agt:
            assert(self.ptr is not None)
            self.single_mask = torch.zeros((veh_att.size(0)), dtype=torch.bool).to(veh_att.device)
            single_inds = self.ptr[:-1] + self.single_veh_idx
            self.single_mask[single_inds] = True
            veh_att = veh_att[self.single_mask]
            mapixes = mapixes[self.single_mask]
        self.env_coll_loss = EnvCollLoss(veh_att, mapixes, map_env)
        self.veh_att = veh_att  # 存储车辆尺寸
        
        # 初始化新的损失函数
        self.ttc_loss = TTCLoss(epsilon=ttc_epsilon, dt=dt, ttc_safe=ttc_safe)
        self.min_dist_lat_loss = MinDistLatLoss(k=min_dist_lat_k)
        self.yaw_rate_loss = YawRateLoss(dt=dt, comfort_threshold_deg=yaw_rate_threshold)

    def forward(self, future_pred, z, prior_out, tgt_traj=None):
        ''' 
        IF single_veh_idx is not None, z and prior_out should be (B, D) rather than (NA, D)

        :param future_pred: NA x T x 4
        :param tgt_traj: 如果提供，将用于计算与目标的交互损失，例如TTC和横向距离
        '''        
        loss = 0.0
        loss_out = {}

        future_pred_interp = interp_traj(future_pred, scale_factor=3)

        if self.loss_weights['coll_veh'] > 0.0:
            # vehicle collision penalty
            veh_coll_loss_val = self.veh_coll_loss(future_pred_interp) # num colliding pairs
            loss = loss + self.loss_weights['coll_veh']*veh_coll_loss_val.mean()
            loss_out['coll_veh_loss'] = veh_coll_loss_val

        if self.loss_weights['coll_env'] > 0.0:
            # env collision penalty
            env_coll_input = future_pred_interp if not self.use_single_agt else future_pred_interp[self.single_mask]
            env_coll_loss_val = self.env_coll_loss(env_coll_input)
            loss = loss + self.loss_weights['coll_env']*env_coll_loss_val.mean()
            loss_out['coll_env_loss'] = env_coll_loss_val

        if self.loss_weights['motion_prior'] > 0.0:
            # motion prior
            motion_prior_loss = self.motion_prior_loss(z, prior_out)
            loss = loss + self.loss_weights['motion_prior']*motion_prior_loss.mean()
            loss_out['motion_prior_loss'] = motion_prior_loss

        if self.loss_weights['init_z'] > 0.0:
            # init loss
            init_loss = torch.sum((self.init_z - z)**2, dim=1)
            loss = loss + self.loss_weights['init_z']*init_loss.mean()
            loss_out['init_loss'] = init_loss
        
        # 添加新的损失函数计算，仅当提供了目标轨迹时才计算交互损失
        if tgt_traj is not None:
            # TTC损失
            if 'ttc' in self.loss_weights and self.loss_weights['ttc'] > 0.0:
                # 如果使用单个代理，可能需要选择正确的车辆
                pred_input = future_pred if not self.use_single_agt else future_pred[self.single_mask]
                ttc_loss_val = self.ttc_loss(pred_input, tgt_traj)
                # 处理可能的NaN值
                if torch.isnan(ttc_loss_val).any():
                    ttc_loss_val = torch.where(torch.isnan(ttc_loss_val), torch.zeros_like(ttc_loss_val), ttc_loss_val)
                loss = loss + self.loss_weights['ttc'] * ttc_loss_val.mean()
                loss_out['ttc_loss'] = ttc_loss_val

            # 横向最小距离损失
            if 'min_dist_lat' in self.loss_weights and self.loss_weights['min_dist_lat'] > 0.0:
                pred_input = future_pred if not self.use_single_agt else future_pred[self.single_mask]
                min_dist_lat_loss_val = self.min_dist_lat_loss(pred_input, tgt_traj, self.veh_att)
                loss = loss + self.loss_weights['min_dist_lat'] * min_dist_lat_loss_val.mean()
                loss_out['min_dist_lat_loss'] = min_dist_lat_loss_val

        # 横摆角速度损失 (不需要目标轨迹)
        if 'yaw_rate' in self.loss_weights and self.loss_weights['yaw_rate'] > 0.0:
            pred_input = future_pred if not self.use_single_agt else future_pred[self.single_mask]
            yaw_rate_loss_val = self.yaw_rate_loss(pred_input)
            loss = loss + self.loss_weights['yaw_rate'] * yaw_rate_loss_val.mean()
            loss_out['yaw_rate_loss'] = yaw_rate_loss_val

        loss_out['loss'] = loss

        return loss_out

class MotionPriorLoss(nn.Module):
    '''
    Measures negative log-likelihood of latent z under the given prior.
    '''
    def __init__(self):
        super(MotionPriorLoss, self).__init__()

    def forward(self, z, prior_out):
        '''
        :param z: current latent vec (NA x z_dim) or (NA x NS x z_dim)
        :param prior_out: output of prior which contains:
            :prior_mean: mean of the prior output (NA x z_dim)
            :prior_var: variance of the prior output (NA x z_dim)
        :return: log-likelihood of cur_z under prior for all agents NA
        '''
        prior_mean = prior_out[0]
        prior_var = prior_out[1]
        if len(z.size()) == 3:
            # then it's NA x NS x z_dim
            prior_mean = prior_mean.unsqueeze(1)
            prior_var = prior_var.unsqueeze(1)
        return -log_normal(z, prior_mean, prior_var)

class EnvCollLoss(nn.Module):
    def __init__(self, veh_att, mapixes, map_env):
        super(EnvCollLoss, self).__init__()
        self.map_env = map_env
        self.mapixes = mapixes
        self.penalty_dists =  torch.sqrt((veh_att[:, 0]**2 / 4.0) + (veh_att[:, 1]**2 / 4.0))
        self.veh_att = veh_att

    def forward(self, traj):
        '''
        :param traj: (NA x T x 4) trajectories (x,y,hx,hy) for each agent to determine collision penalty.
                                should be UNNORMALIZED.
        :return: loss
        '''
        NA, T, _ = traj.size()
        traj = traj.view(NA*T, 4)
        cur_att = self.veh_att.view(NA, 1, 2).expand(NA, T, 2).reshape(NA*T, 2)

        # get collisions w/ non-drivable (first layer)
        drivable_raster = self.map_env.nusc_raster[:, 0]
        cur_mapixes = self.mapixes.view(NA, 1).expand(NA, T).reshape(NA*T)
        coll_pt = nutils.get_coll_point(drivable_raster,
                                        self.map_env.nusc_dx,
                                        traj.detach(),
                                        cur_att,
                                        cur_mapixes)

        if torch.sum(torch.isnan(coll_pt)) == NA*T*2:
            return torch.Tensor([0.0]).to(traj.device)

        # compute penalties
        valid = ~torch.isnan(torch.sum(coll_pt, axis=1))
        traj_cent = traj[:,:2][valid]
        cur_dists = torch.norm(traj_cent - coll_pt[valid], dim=1)
        cur_pen_dists = self.penalty_dists.view(NA, 1).expand(NA, T).reshape(NA*T)[valid]
        cur_penalties = 1.0 - (cur_dists / cur_pen_dists)

        return cur_penalties

class VehCollLoss(nn.Module):
    '''
    Penalizes collision between vehicles with circle approximation.
    '''
    def __init__(self, veh_att,
                       num_circ=5,
                       buffer_dist=0.0,
                       single_veh_idx=None,
                       ptr=None
                       ):
        '''
        :param veh_att: UNNORMALIZED lw for the vehicles that will be computing loss for (NA x 2)
        :param num_circ: number of circles used to approximate each vehicle.
        :param buffer: extra buffer distance that circles must be apart to avoid being penalized
        :param single_veh_idx: only computes loss w.r.t a single vehicle at the given index within each scene graph.
                                If given must also include.ptr from scene graph as input even if only a single batch.
        :param ptr: if given, treats the traj input as a batch of scene graphs and only computes collisions between
                    vehicles in the same batch. if None, assumes a single batch
        '''
        super(VehCollLoss, self).__init__()
        self.veh_att = veh_att
        self.buffer_dist = buffer_dist
        self.single_veh_idx = single_veh_idx
        self.ptr = ptr

        NA = self.veh_att.size(0)
        # construct centroids circles of each agent
        self.veh_rad = self.veh_att[:, 1] / 2. # radius of the discs for each vehicle assuming length > width
        cent_min = -(self.veh_att[:, 0] / 2.) + self.veh_rad
        cent_max = (self.veh_att[:, 0] / 2.) - self.veh_rad
        cent_x = torch.stack([torch.linspace(cent_min[vidx].item(), cent_max[vidx].item(), num_circ) for vidx in range(NA)], dim=0).to(veh_att.device)
        # create dummy states for centroids with y=0 and hx,hy=1,0 so can transform later
        self.centroids = torch.stack([cent_x, torch.zeros_like(cent_x), torch.ones_like(cent_x), torch.zeros_like(cent_x)], dim=2)
        self.num_circ = num_circ
        # minimum distance that two vehicle circle centers can be apart without collision
        self.penalty_dists = self.veh_rad.view(NA, 1).expand(NA, NA) + self.veh_rad.view(1, NA).expand(NA, NA) + self.buffer_dist
        # need a mask to ignore self-collisions when computing
        self.off_diag_mask = ~torch.eye(NA, dtype=torch.bool).to(self.veh_att.device)
        if self.ptr is None:
            # assume single batch
            self.ptr = torch.Tensor([0, NA]).to(self.veh_att.device).to(torch.long)
        # ignore collisiosn with other scene graphs in the batch
        batch_mask = torch.zeros((NA, NA), dtype=torch.bool).to(self.veh_att.device)
        for b in range(1, len(self.ptr)):
            # only the block corresponding to pairs of vehicles in the same scene graph matter
            batch_mask[self.ptr[b-1]:self.ptr[b], self.ptr[b-1]:self.ptr[b]] = True
        self.off_diag_mask = torch.logical_and(self.off_diag_mask, batch_mask)

        if self.single_veh_idx is not None:
            # then only want to use penalties associated with a single agent
            single_mask = torch.zeros((NA), dtype=torch.bool).to(self.off_diag_mask.device)
            single_inds = self.ptr[:-1] + self.single_veh_idx
            single_mask[single_inds] = True # target is always at index 0 of each scene graph
            single_mask_row = single_mask.view(1, NA).expand(NA, NA)
            single_mask_col = single_mask.view(NA, 1).expand(NA, NA)
            single_mask = torch.logical_or(single_mask_row, single_mask_col)
            self.off_diag_mask = torch.logical_and(self.off_diag_mask, single_mask)


    def forward(self, traj, att_inds=None, return_raw=False):
        '''
        :param traj: (N x T x 4) trajectories (x,y,hx,hy) for each agent to determine collision penalty.
                                should be UNNORMALIZED.
        :param att_inds: list [N] if not using the full list of agents that was used to init the loss, which indices are you using
        :param return_raw: if True, returns the (T x N x N) penalty cost matrix and (T x N x N) mask where each True entry is a
                            colliding pair that is valid (according to settings)
        :return: loss
        '''
        NA, T, _ = traj.size()

        traj = traj[:, :, :4].view(NA*T, 4)
        cur_centroids = self.centroids
        if att_inds is not None:
            cur_centroids = cur_centroids[att_inds]
        cur_cent = cur_centroids.view(NA, 1, self.num_circ, 4).expand(NA, T, self.num_circ, 4).reshape(NA*T, self.num_circ, 4)
        # centroids are in local, need to transform to global based on current traj
        world_cent = transform2frame(traj, cur_cent, inverse=True).view(NA, T, self.num_circ, 4)[:, :, :, :2] # only need centers
        
        world_cent = world_cent.transpose(0, 1) # T x NA X C x 2
        # distances between all pairs of circles between all pairs of agents
        cur_cent1 = world_cent.view(T, NA, 1, self.num_circ, 2).expand(T, NA, NA, self.num_circ, 2).reshape(T*NA*NA, self.num_circ, 2)
        cur_cent2 = world_cent.view(T, 1, NA, self.num_circ, 2).expand(T, NA, NA, self.num_circ, 2).reshape(T*NA*NA, self.num_circ, 2)
        pair_dists = torch.cdist(cur_cent1, cur_cent2).view(T*NA*NA, self.num_circ*self.num_circ)

        # get minimum distance overall all circle pairs between each pair
        min_pair_dists = torch.min(pair_dists, 1)[0].view(T, NA, NA)   
        cur_penalty_dists = self.penalty_dists
        if att_inds is not None:
            cur_penalty_dists = cur_penalty_dists[att_inds][:, att_inds]
        cur_penalty_dists = cur_penalty_dists.view(1, NA, NA)
        is_colliding_mask = min_pair_dists <= cur_penalty_dists
        # diagonals are self collisions so ignore them
        cur_off_diag_mask = self.off_diag_mask
        if att_inds is not None:
            cur_off_diag_mask = cur_off_diag_mask[att_inds][:, att_inds]
            # print(cur_off_diag_mask)
        is_colliding_mask = torch.logical_and(is_colliding_mask, cur_off_diag_mask.view(1, NA, NA))
        if not return_raw and torch.sum(is_colliding_mask) == 0:
            return torch.Tensor([0.0]).to(traj.device)
        # compute penalties
        # penalty is inverse normalized distance apart for those already colliding
        cur_penalties = 1.0 - (min_pair_dists / cur_penalty_dists)

        if return_raw:
            return cur_penalties, is_colliding_mask
        else:
            cur_penalties = cur_penalties[is_colliding_mask]
            return cur_penalties    

ENV_COLL_THRESH = 0.05 # up to 5% of vehicle can be off the road
VEH_COLL_THRESH = 0.02 # IoU must be over this to count as a collision for metric (not loss)

def check_single_veh_coll(traj_tgt, lw_tgt, traj_others, lw_others):
    '''
    Checks if the target trajectory collides with each of the given other trajectories.

    Assumes all trajectories and attributes are UNNORMALIZED. Handles nan frames in traj_others by simply skipping.

    :param traj_tgt: (T x 4)
    :param lw_tgt: (2, )
    :param traj_others: (N x T x 4)
    :param lw_others: (N x 2)

    :returns veh_coll: (N)
    :returns coll_time: (N)
    '''
    import datasets.nuscenes_utils as nutils
    from shapely.geometry import Polygon

    NA, FT, _ = traj_others.size()
    traj_tgt = traj_tgt.cpu().numpy()
    lw_tgt = lw_tgt.cpu().numpy()
    traj_others = traj_others.cpu().numpy()
    lw_others = lw_others.cpu().numpy()

    veh_coll = np.zeros((NA), dtype=bool)
    coll_time = np.ones((NA), dtype=np.int64)*FT
    poly_cache = dict() # for the tgt polygons since used many times
    for aj in range(NA):
        for t in range(FT):
            # compute iou
            if t not in poly_cache:
                ai_state = traj_tgt[t, :]
                ai_corners = nutils.get_corners(ai_state, lw_tgt)
                ai_poly = Polygon(ai_corners)
                poly_cache[t] = ai_poly
            else:
                ai_poly = poly_cache[t]

            aj_state = traj_others[aj, t, :]
            if np.sum(np.isnan(aj_state)) > 0:
                continue
            aj_corners = nutils.get_corners(aj_state, lw_others[aj])
            aj_poly = Polygon(aj_corners)
            cur_iou = ai_poly.intersection(aj_poly).area / ai_poly.union(aj_poly).area
            if cur_iou > VEH_COLL_THRESH:
                veh_coll[aj] = True
                coll_time[aj] = t
                break # don't need to check rest of sequence

    return veh_coll, coll_time

def check_pairwise_veh_coll(traj, lw):
    '''
    Computes collision rate for all pairs of given trajectories.

    Assumes all trajectories and attributes are UNNORMALIZED.

    :param traj: (N x T x 4)
    :param lw: (N x 2)

    returns: NA x NS with a 1 if collided
    '''
    import datasets.nuscenes_utils as nutils
    from shapely.geometry import Polygon

    NA, FT, _ = traj.size()
    traj = traj.cpu().numpy()
    lw = lw.cpu().numpy()

    veh_coll = np.zeros((NA), dtype=bool)
    poly_cache = dict()    
    # loop over every timestep in every sample for this combination
    coll_count = 0
    for ai in range(NA):
        for aj in range(ai+1, NA): # don't double count
            if veh_coll[ai]:
                break # already found a collision, move on.
            for t in range(FT):
                # compute iou
                if (ai, t) not in poly_cache:
                    ai_state = traj[ai, t, :]
                    ai_corners = nutils.get_corners(ai_state, lw[ai])
                    ai_poly = Polygon(ai_corners)
                    poly_cache[(ai, t)] = ai_poly
                else:
                    ai_poly = poly_cache[(ai, t)]

                if (aj, t) not in poly_cache:
                    aj_state = traj[aj, t, :]
                    aj_corners = nutils.get_corners(aj_state, lw[aj])
                    aj_poly = Polygon(aj_corners)
                    poly_cache[(aj, t)] = aj_poly
                else:
                    aj_poly = poly_cache[(aj, t)]

                cur_iou = ai_poly.intersection(aj_poly).area / ai_poly.union(aj_poly).area
                if cur_iou > VEH_COLL_THRESH:
                    coll_count += 1
                    veh_coll[ai] = True
                    break # don't need to check rest of sequence

    coll_dict = {
        'num_coll_veh' : float(coll_count),
        'num_traj_veh' : float(NA),
        'did_collide' : veh_coll
    }

    return coll_dict

def interp_traj(future_pred, scale_factor=3):
    '''
    :param future_pred (NA x T x 4) or (NA x NS x T x 4)
    '''
    mult_samp = False
    if len(future_pred.size()) == 4:
        mult_samp = True
        NA, NS, T, _ = future_pred.size()
        future_pred = future_pred.reshape(NA*NS, T, 4)
    future_pred_interp = nn.functional.interpolate(future_pred.transpose(1,2),
                                                        scale_factor=scale_factor,
                                                        mode='linear').transpose(1,2)
    # normalize heading (NOTE: interp hx and hy is not exactly correct, but close)
    interp_pos = future_pred_interp[:, :, :2]
    interp_h = future_pred_interp[:, :, 2:4]
    interp_h = interp_h / torch.norm(interp_h, dim=-1, keepdim=True)
    future_pred_interp = torch.cat([interp_pos, interp_h], dim=-1)
    if mult_samp:
        future_pred_interp = future_pred_interp.reshape(NA, NS, future_pred_interp.size(1), 4)
    return future_pred_interp

def check_behind(attacker_fut, tgt_fut, ptr, crash_min_infront):
    '''
    checks if each attacker is behind the target at each time step.

    :param attacker_fut: future for all attackers (NA-B, T, 4)
    :param tgt_fut: future for targets (B, T, 4)
    :param ptr: graph ptr to start of each batch
    :param crash_min_infront: threshold to determine if behind

    :return behind_steps: (NA-B, T) True if attacker currently behind tgt
    '''
    graph_sizes = ptr[1:] - ptr[:-1]
    B = graph_sizes.size(0)
    attacker_pred = attacker_fut
    tgt_expanded = torch.cat([tgt_fut[b:b+1, :, :].expand(graph_sizes[b]-1, -1, -1) for b in range(B)], dim=0)

    tgt_h = tgt_expanded[:, :, 2:4]
    tgt_pos = tgt_expanded[:, :, :2]
    atk_pos = attacker_pred[:, :, :2]

    tgt2atk = atk_pos - tgt_pos
    tgt2atk = tgt2atk / torch.norm(tgt2atk, dim=-1, keepdim=True)
    cossim = torch.sum(tgt2atk * tgt_h, dim=-1) # (NA-B, T)

    # determine for each agent if:
    #   - behind and stays behind
    behind_steps = cossim < crash_min_infront
    return behind_steps

def compute_velocities_from_traj(traj, dt=0.1):
    """
    从轨迹计算速度
    
    参数:
        traj: 轨迹数据 [NA, T, 4] (x, y, hx, hy)
        dt: 时间步长
        
    返回:
        velocities: 速度向量 [NA, T-1, 2]
        speeds: 速度大小 [NA, T-1]
    """
    # 计算位置差分
    pos_diff = traj[:, 1:, :2] - traj[:, :-1, :2]
    
    # 除以时间步长得到速度
    velocities = pos_diff / dt
    
    # 计算速度大小
    speeds = torch.norm(velocities, dim=2)
    
    return velocities, speeds

def compute_yaw_rate_from_traj(traj, dt=0.1):
    """
    从轨迹计算横摆角速度
    
    参数:
        traj: 轨迹数据 [NA, T, 4] (x, y, hx, hy)
        dt: 时间步长
        
    返回:
        yaw_rates: 横摆角速度 [NA, T-1]
    """
    # 从hx, hy计算角度
    headings = torch.atan2(traj[:, :, 3], traj[:, :, 2])
    
    # 计算角度差分
    heading_diff = headings[:, 1:] - headings[:, :-1]
    
    # 处理角度跨越±π的情况
    heading_diff = torch.atan2(torch.sin(heading_diff), torch.cos(heading_diff))
    
    # 除以时间步长得到角速度
    yaw_rates = heading_diff / dt
    
    return yaw_rates

def compute_longitudinal_metrics(ego_traj, target_traj, dt=0.1, epsilon=1e-6):
    """
    计算纵向交互指标（相对距离、相对速度、TTC）
    
    参数:
        ego_traj: 自车轨迹 [B, T, 4]
        target_traj: 目标车轨迹 [B, T, 4]
        dt: 时间步长
        epsilon: 数值稳定性常数
        
    返回:
        d_rel_long: 纵向相对距离 [B, T-1]
        v_rel_long: 纵向相对速度 [B, T-1]
        ttc: 碰撞时间 [B, T-1]
    """
    # 计算速度
    ego_vel, ego_speed = compute_velocities_from_traj(ego_traj, dt)
    target_vel, target_speed = compute_velocities_from_traj(target_traj, dt)
    
    # 提取朝向向量
    ego_heading = ego_traj[:, :-1, 2:4]  # 使用与速度对应的时间步
    
    # 计算相对位置
    rel_pos = target_traj[:, :-1, :2] - ego_traj[:, :-1, :2]  # [B, T-1, 2]
    
    # 计算相对距离
    rel_dist = torch.norm(rel_pos, dim=2)  # [B, T-1]
    
    # 计算朝向单位向量
    ego_heading_norm = ego_heading / (torch.norm(ego_heading, dim=2, keepdim=True) + epsilon)
    
    # 计算纵向相对距离（投影到自车朝向上）
    d_rel_long = torch.sum(rel_pos * ego_heading_norm, dim=2)  # [B, T-1]
    
    # 计算相对速度
    rel_vel = target_vel - ego_vel  # [B, T-1, 2]
    
    # 计算纵向相对速度（投影到自车朝向上）
    v_rel_long = torch.sum(rel_vel * ego_heading_norm, dim=2)  # [B, T-1]
    
    # 计算TTC
    # TTC只在目标车(target)在自车(ego)前方(d_rel_long > 0)，
    # 且自车比目标车快(v_rel_long < 0)时才有意义
    ttc = torch.ones_like(v_rel_long) * float('inf')
    
    is_approaching = v_rel_long < 0
    is_target_in_front = d_rel_long > 0
    valid_ttc = is_approaching & is_target_in_front

    if torch.any(valid_ttc):
        # v_rel_long是负值，所以需要取反
        ttc[valid_ttc] = d_rel_long[valid_ttc] / (-v_rel_long[valid_ttc] + epsilon)
    
    return d_rel_long, v_rel_long, ttc

def compute_lateral_metrics(ego_traj, target_traj, ego_width, target_width):
    """
    计算横向交互指标（横向距离等）
    
    参数:
        ego_traj: 自车轨迹 [B, T, 4]
        target_traj: 目标车轨迹 [B, T, 4]
        ego_width: 自车宽度 [B] 或标量
        target_width: 目标车宽度 [B] 或标量
        
    返回:
        d_lat: 横向距离 [B, T]
        d_lat_gap: 车辆边缘之间的横向净距 [B, T]
    """
    # 提取朝向向量和位置
    ego_pos = ego_traj[:, :, :2]  # [B, T, 2]
    target_pos = target_traj[:, :, :2]  # [B, T, 2]
    ego_heading = ego_traj[:, :, 2:4]  # [B, T, 2]
    
    # 计算纵向单位向量
    ego_heading_norm = ego_heading / torch.norm(ego_heading, dim=2, keepdim=True)
    
    # 计算横向单位向量（逆时针旋转90度）
    ego_lateral = torch.stack([-ego_heading_norm[:, :, 1], ego_heading_norm[:, :, 0]], dim=2)
    
    # 计算相对位置
    rel_pos = target_pos - ego_pos  # [B, T, 2]
    
    # 计算横向距离（投影到横向单位向量上）
    d_lat = torch.abs(torch.sum(rel_pos * ego_lateral, dim=2))  # [B, T]
    
    # 确保宽度是正确的形状
    if not isinstance(ego_width, torch.Tensor):
        ego_width = torch.tensor(ego_width).to(ego_traj.device)
    if not isinstance(target_width, torch.Tensor):
        target_width = torch.tensor(target_width).to(target_traj.device)
    
    if len(ego_width.shape) == 0:
        ego_width = ego_width.unsqueeze(0).expand(ego_traj.shape[0])
    if len(target_width.shape) == 0:
        target_width = target_width.unsqueeze(0).expand(target_traj.shape[0])
    
    # 计算车辆边缘之间的横向净距
    half_width_sum = (ego_width.unsqueeze(1) / 2) + (target_width.unsqueeze(1) / 2)
    d_lat_gap = d_lat - half_width_sum  # [B, T]
    
    return d_lat, d_lat_gap

class TTCLoss(nn.Module):
    """
    计算Time-to-Collision损失
    
    当自车比前方车辆速度更快（正在接近）时，通过TTC衡量碰撞的紧急程度
    TTC越小，表示碰撞风险越高，损失值越大
    """
    def __init__(self, epsilon=1e-6, dt=0.1, ttc_safe=3.0):
        """
        初始化TTC损失函数
        
        参数:
            epsilon: 防止除零的小常数
            dt: 时间步长
            ttc_safe: 安全TTC阈值(秒)
        """
        super(TTCLoss, self).__init__()
        self.epsilon = epsilon
        self.dt = dt
        self.ttc_safe = ttc_safe
    
    def forward(self, future_pred, tgt_traj):
        """
        计算TTC损失
        
        参数:
            future_pred: 攻击车辆预测轨迹 [NA-B, T, 4]
            tgt_traj: 目标车辆预测轨迹 [B, T, 4]
            
        返回:
            ttc_loss: TTC损失值 [NA-B, T-1]
        """
        # 获取批次大小
        NA_B = future_pred.size(0)
        B = tgt_traj.size(0)
        T = future_pred.size(1)
        
        # 为每个攻击车辆匹配目标车辆
        graph_sizes = [NA_B // B] * B  # 每个目标匹配相同数量的攻击车辆
        tgt_expanded = torch.cat([tgt_traj[b:b+1, :, :].expand(graph_sizes[b], -1, -1) for b in range(B)], dim=0)
        
        # 计算纵向交互指标
        d_rel_long, v_rel_long, ttc = compute_longitudinal_metrics(future_pred, tgt_expanded, dt=self.dt, epsilon=self.epsilon)
        
        # 设计损失函数：当TTC小于安全阈值时，损失为它们差值的平方，这种方式更具鲁棒性
        ttc_loss = torch.clamp(self.ttc_safe - ttc, min=0.0).pow(2)
        
        return ttc_loss

class TTCLossAtk(nn.Module):
    """
    对抗型TTC：返回 inverse TTC（inf 记为0），用于鼓励更短TTC（更危险）。
    上层以正权重直接加和即可。
    """
    def __init__(self, epsilon=1e-6, dt=0.1):
        super(TTCLossAtk, self).__init__()
        self.epsilon = epsilon
        self.dt = dt

    def forward(self, future_pred, tgt_traj):
        _, _, ttc = compute_longitudinal_metrics(
            future_pred, tgt_traj, dt=self.dt, epsilon=self.epsilon
        )
        inv_ttc = torch.where(torch.isinf(ttc), torch.zeros_like(ttc), 1.0 / (ttc + self.epsilon))
        return inv_ttc

class MinDistLatLoss(nn.Module):
    """
    横向最小距离攻击损失：只惩罚正向净距（gap>0），鼓励更贴靠/重叠。
    要求传入与攻击者对齐的目标扩展，或提供 `ego_mask` 和 `graph_sizes` 以便内部配对。
    """
    def __init__(self, k=1.0):
        super(MinDistLatLoss, self).__init__()
        self.k = k

    def forward(self, attacker_pred, tgt_or_expanded, veh_att_full, ego_mask=None, graph_sizes=None):
        # 对齐目标轨迹
        if ego_mask is not None and graph_sizes is not None:
            # 构造按图展开的目标轨迹
            B = ego_mask.sum().item()
            tgt_traj_full = tgt_or_expanded  # [B, T, 4]
            tgt_expanded = torch.cat([
                tgt_traj_full[b:b+1, :, :].expand(graph_sizes[b]-1, -1, -1) for b in range(B)
            ], dim=0)
            # 宽度对齐：攻击者宽度（非ego）与按图扩展的目标宽度
            atk_width = veh_att_full[~ego_mask, 1]
            tgt_width = veh_att_full[ego_mask, 1]
            tgt_width_expanded = torch.cat([
                tgt_width[b:b+1].expand(graph_sizes[b]-1) for b in range(B)
            ], dim=0)
        else:
            # 假定已对齐
            tgt_expanded = tgt_or_expanded
            # 退化做法：使用同长度的常量宽度，防止出错（不建议常用）
            atk_width = veh_att_full[:attacker_pred.size(0), 1]
            tgt_width_expanded = atk_width.clone()

        # 计算横向净距
        _, d_lat_gap = compute_lateral_metrics(attacker_pred, tgt_expanded, atk_width, tgt_width_expanded)
        # 只惩罚正 gap（距离过宽），负 gap(重叠)不惩罚
        pos_gap = torch.relu(d_lat_gap)
        return self.k * pos_gap.pow(2)

class YawRateLoss(nn.Module):
    """
    计算横摆角速度损失 (攻击型)
    
    鼓励产生剧烈、不稳定的转向动作，以生成失控或极具攻击性的驾驶行为场景。
    损失函数鼓励模型的横摆角速度超过一个给定的阈值。
    """
    def __init__(self, dt=0.1, comfort_threshold_deg=15.0, epsilon=1e-6):
        """
        初始化攻击型横摆角速度损失函数
        
        参数:
            dt: 时间步长
            comfort_threshold_deg: 舒适阈值，单位为度/秒。攻击的目标是超过此阈值。
            epsilon: 防止除零的小常数
        """
        super(YawRateLoss, self).__init__()
        self.dt = dt
        # 将阈值从度转换为弧度
        self.comfort_threshold = comfort_threshold_deg * (math.pi / 180.0)
        self.epsilon = epsilon
    
    def forward(self, future_pred):
        """
        计算攻击型横摆角速度损失
        
        参数:
            future_pred: 车辆预测轨迹 [NA, T, 4]
            
        返回:
            yaw_rate_loss: 横摆角速度损失 [NA, T-1]
        """
        # 计算横摆角速度
        yaw_rates = compute_yaw_rate_from_traj(future_pred, dt=self.dt)
        
        # 计算角速度绝对值
        abs_yaw_rate = torch.abs(yaw_rates)
        
        # 攻击性损失：鼓励角速度超过阈值。
        # 当角速度超过阈值时，我们希望损失变小（因为目标达成）。
        # 因此，损失是 1 / (超过阈值的部分)。
        excess_yaw_rate = torch.nn.functional.relu(abs_yaw_rate - self.comfort_threshold)
        yaw_rate_loss = 1.0 / (excess_yaw_rate + self.epsilon)
        
        return yaw_rate_loss
    

    