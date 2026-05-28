# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Baseline: Random perturbation of latent z, no optimization.
# Generates scenarios by z_adv = z_prior_mean + scale * randn * sqrt(prior_var), then decode.
# Output format is compatible with eval_adv_gen.py.

import os
import sys
import json
import argparse
import tqdm
import torch
import numpy as np

# Script lives in eval/; resolve repo root one level up, then add src/ to PYTHONPATH.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from torch_geometric.data import DataLoader as GraphDataLoader
from utils.common import dict2obj, mkdir
from utils.logger import Logger
from utils.torch import get_device, load_state
from utils.scenario_gen import prepare_output_dict, detach_embed_info
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from models.traffic_model import TrafficModel


def parse_cfg():
    parser = argparse.ArgumentParser(description='Random perturbation baseline: z_adv = z_prior + noise, no optimization.')
    parser.add_argument('-c', '--config', required=True, help='Path to YAML config (same format as adv_scenario_gen).')
    parser.add_argument('--out', type=str, default='./out/random_baseline_out', help='Output directory.')
    parser.add_argument('--perturbation_scale', type=float, default=1.0,
                        help='Scale for Gaussian noise: z_adv = z_prior + scale * N(0, prior_var).')
    parser.add_argument('--max_scenes', type=int, default=None, help='Max number of scenes to generate (default: all).')

    args = parser.parse_args()
    # Load YAML config
    import yaml
    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    if cfg_dict is None:
        cfg_dict = {}
    # Defaults for map/model (from add_base_args) if not in config
    defaults = {
        'map_obs_bounds': [-17.0, -38.5, 60.0, 38.5],
        'map_obs_size_pix': 256,
        'map_layers': ['drivable_area', 'carpark_area', 'road_divider', 'lane_divider'],
        'agent_types': ['car', 'truck'],
        'past_len': 4,
        'future_len': 12,
        'model_output_bicycle': True,
        'map_feat_size': 64,
        'past_feat_size': 64,
        'future_feat_size': 64,
        'latent_size': 32,
        'conv_kernel_list': [7, 5, 5, 3, 3, 3],
        'conv_stride_list': [2, 2, 2, 2, 2, 2],
        'conv_filter_list': [16, 32, 64, 64, 128, 128],
        'reduce_cats': False,
    }
    for k, v in defaults.items():
        if k not in cfg_dict:
            cfg_dict[k] = v
    cfg_dict['out'] = args.out
    cfg_dict['perturbation_scale'] = args.perturbation_scale
    cfg_dict['max_scenes'] = args.max_scenes
    config = dict2obj(cfg_dict)
    return config


def main():
    cfg = parse_cfg()
    device = get_device()
    mkdir(cfg.out)
    scene_out_dir = os.path.join(cfg.out, 'scenario_results', 'high_risk')
    mkdir(scene_out_dir)

    data_path = os.path.join(cfg.data_dir, cfg.data_version)
    map_env = NuScenesMapEnv(
        data_path,
        bounds=cfg.map_obs_bounds,
        L=cfg.map_obs_size_pix,
        W=cfg.map_obs_size_pix,
        layers=cfg.map_layers,
        device=device,
        load_lanegraph=False,
    )
    test_dataset = NuScenesDataset(
        data_path, map_env,
        version=cfg.data_version,
        split=cfg.split,
        categories=cfg.agent_types,
        npast=cfg.past_len,
        nfuture=cfg.future_len,
        seq_interval=cfg.seq_interval,
        randomize_val=True,
        val_size=cfg.val_size,
        reduce_cats=cfg.reduce_cats,
    )
    test_loader = GraphDataLoader(
        test_dataset,
        batch_size=1,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers or 0,
        pin_memory=False,
    )

    model = TrafficModel(
        cfg.past_len, cfg.future_len, cfg.map_obs_size_pix, len(test_dataset.categories),
        map_feat_size=cfg.map_feat_size,
        past_feat_size=cfg.past_feat_size,
        future_feat_size=cfg.future_feat_size,
        latent_size=cfg.latent_size,
        output_bicycle=getattr(cfg, 'model_output_bicycle', True),
        conv_channel_in=map_env.num_layers,
        conv_kernel_list=cfg.conv_kernel_list,
        conv_stride_list=cfg.conv_stride_list,
        conv_filter_list=cfg.conv_filter_list,
    ).to(device)

    if cfg.ckpt:
        load_state(cfg.ckpt, model, map_location=device)
    model.set_normalizer(test_dataset.get_state_normalizer())
    model.set_att_normalizer(test_dataset.get_att_normalizer())
    if getattr(cfg, 'model_output_bicycle', True):
        from datasets.utils import NUSC_BIKE_PARAMS
        model.set_bicycle_params(NUSC_BIKE_PARAMS)

    model.eval()
    dt = test_dataset.dt
    max_scenes = getattr(cfg, 'max_scenes', None)
    scale = getattr(cfg, 'perturbation_scale', 1.0)

    Logger.log('Random baseline: perturbation_scale=%.2f' % scale)
    data_idx = 0
    for i, (scene_graph, map_idx) in enumerate(tqdm.tqdm(test_loader, desc='Random baseline')):
        if max_scenes is not None and data_idx >= max_scenes:
            break
        scene_graph = scene_graph.to(device)
        map_idx = map_idx.to(device)
        B = map_idx.size(0)
        NA = scene_graph.past.size(0)
        ego_inds = scene_graph.ptr[:-1]
        ego_mask = torch.zeros(NA, dtype=torch.bool, device=device)
        ego_mask[ego_inds] = True

        with torch.no_grad():
            embed_info_attached = model.embed(scene_graph, map_idx, map_env)
            embed_info = detach_embed_info(embed_info_attached)
            prior_mean = embed_info['prior_out'][0]
            prior_var = embed_info['prior_out'][1]
            z_adv = prior_mean + scale * torch.randn_like(prior_mean, device=device) * torch.sqrt(prior_var.clamp(min=1e-6))
            init_fut = model.decode_embedding(prior_mean, embed_info_attached, scene_graph, map_idx, map_env)['future_pred'].detach()
            adv_fut = model.decode_embedding(z_adv, embed_info_attached, scene_graph, map_idx, map_env)['future_pred'].detach()

        for b in range(B):
            ptr_b = scene_graph.ptr[b].item()
            ptr_b1 = scene_graph.ptr[b + 1].item()
            scene_graph_b = scene_graph.to_data_list()[b]
            init_b = init_fut[ptr_b:ptr_b1]
            adv_b = adv_fut[ptr_b:ptr_b1]
            prior_b = (prior_mean[ptr_b:ptr_b1].detach(), prior_var[ptr_b:ptr_b1].detach())
            # attack_agt=1 (first non-ego), attack_t=0 for eval compatibility
            scene_out_dict = prepare_output_dict(
                scene_graph_b, map_idx[b].item(), map_env, dt, model,
                init_b, adv_b,
                attack_agt=1,
                attack_t=0,
                adv_z=z_adv[ptr_b:ptr_b1].detach(),
                prior_distrib=prior_b,
            )
            fout = os.path.join(scene_out_dir, 'scene_%04d.json' % data_idx)
            with open(fout, 'w') as f:
                json.dump(scene_out_dict, f)
            data_idx += 1
            if max_scenes is not None and data_idx >= max_scenes:
                break

    Logger.log('Saved %d scenarios to %s' % (data_idx, scene_out_dir))


if __name__ == '__main__':
    main()
