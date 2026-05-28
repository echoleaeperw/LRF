#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对基于规则的对抗生成结果进行四维评估
========================================

使用 AdversarialTrajectoryEvaluator 对
  out/adv_gen_rule_based_out_1773900166-best/scenario_results/
下的 high_risk / longtail_condition / low_risk 三类场景进行批量评估。

JSON 字段说明（来自 adv_gen_rule_based）：
  - N        : int，智能体数量
  - dt       : float，时间步长（0.5s）
  - lw       : [NA, 2]，车辆长宽
  - past     : [NA, 4, 6]，过去轨迹（x,y,cos_h,sin_h,speed,?）
  - fut_init : [NA, T, 4]，初始（非对抗）未来轨迹 (x,y,cos_h,sin_h)
  - fut_adv  : [NA, T, 4]，对抗性未来轨迹   (x,y,cos_h,sin_h)
  - z_adv    : [NA, z_dim]，对抗性潜变量
  - z_prior  : {'mean': [NA,z_dim], 'var': [NA,z_dim]}，先验参数

运行方法：
  cd /home/wuyou/STRIVE/STRIVE
  python run_eval_rule_based.py
  python run_eval_rule_based.py --subset high_risk
  python run_eval_rule_based.py --out_dir out/my_custom_out
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any

# ──────────────────────────────────────────────────────────────
# 路径设置：把 src/ 加入 Python 路径，以便找到 utils/eval_adversarial_trajectory
# ──────────────────────────────────────────────────────────────
_script_dir = os.path.abspath(os.path.dirname(__file__))
_repo_root  = os.path.abspath(os.path.join(_script_dir, '..'))
_src_root   = os.path.join(_repo_root, 'src')
_md_root    = os.path.join(_repo_root, '..', 'MD')

for _p in [_src_root, _md_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.eval_adversarial_trajectory import AdversarialTrajectoryEvaluator
from utils.logger import Logger


# ──────────────────────────────────────────────────────────────
# 数据加载工具
# ──────────────────────────────────────────────────────────────

def load_scene_json(json_path: str) -> Dict[str, Any]:
    """加载单个场景 JSON 文件，转为 Tensor 格式"""
    with open(json_path, 'r') as f:
        d = json.load(f)

    dt = float(d['dt'])

    # 轨迹：(x, y, cos_h, sin_h)  [NA, T, 4]
    fut_init = torch.tensor(d['fut_init'], dtype=torch.float32)   # [NA, T, 4]
    fut_adv  = torch.tensor(d['fut_adv'],  dtype=torch.float32)   # [NA, T, 4]

    # 潜变量
    z_adv    = torch.tensor(d['z_adv'],         dtype=torch.float32)  # [NA, z_dim]
    z_mean   = torch.tensor(d['z_prior']['mean'], dtype=torch.float32) # [NA, z_dim]

    # 车辆尺寸
    lw = torch.tensor(d['lw'], dtype=torch.float32)  # [NA, 2]
    vehicle_lengths = lw[:, 0]
    vehicle_widths  = lw[:, 1]

    return {
        'dt':             dt,
        'traj_init':      fut_init,
        'traj_adv':       fut_adv,
        'z_adv':          z_adv,
        'z_prior_mean':   z_mean,
        'vehicle_lengths': vehicle_lengths,
        'vehicle_widths':  vehicle_widths,
        'attack_agt':     int(d.get('attack_agt', -1)),
        'attack_t':       int(d.get('attack_t', -1)),
        'scene_name':     Path(json_path).stem,
        'category':       Path(json_path).parent.name,
    }


def collect_all_scenes(result_dir: str, subsets: List[str]) -> List[str]:
    """收集指定子集下所有 JSON 文件路径"""
    paths = []
    for subset in subsets:
        subset_dir = os.path.join(result_dir, subset)
        if not os.path.isdir(subset_dir):
            print(f"[警告] 子集目录不存在，跳过: {subset_dir}")
            continue
        for fname in sorted(os.listdir(subset_dir)):
            if fname.endswith('.json'):
                paths.append(os.path.join(subset_dir, fname))
    return paths


def build_real_dataset_tensor(all_scene_paths: List[str]) -> torch.Tensor:
    """
    用所有场景的 fut_init（原始/非对抗轨迹）拼接为 real_dataset 基准。
    形状：[N_total, T, 4]
    """
    all_trajs = []
    for p in all_scene_paths:
        try:
            with open(p, 'r') as f:
                d = json.load(f)
            traj = torch.tensor(d['fut_init'], dtype=torch.float32)  # [NA, T, 4]
            all_trajs.append(traj)
        except Exception as e:
            print(f"[警告] 无法加载 {p}: {e}")
    if not all_trajs:
        raise RuntimeError("无法加载任何场景来构建 real_dataset")
    return torch.cat(all_trajs, dim=0)   # [N_total, T, 4]


# ──────────────────────────────────────────────────────────────
# 单场景评估
# ──────────────────────────────────────────────────────────────

def evaluate_single_scene(evaluator: AdversarialTrajectoryEvaluator,
                           scene: Dict[str, Any],
                           real_dataset: torch.Tensor) -> Dict[str, Any]:
    """对单个场景执行全套四维评估 + 长尾事件分析"""

    traj_init = scene['traj_init']
    traj_adv  = scene['traj_adv']
    z_adv     = scene['z_adv']
    z_prior   = scene['z_prior_mean']

    # T 对齐：real_dataset 可能来自其他场景，时间步数一致时才能比较
    T_adv  = traj_adv.shape[1]
    T_real = real_dataset.shape[1]
    if T_real != T_adv:
        min_T = min(T_adv, T_real)
        traj_init     = traj_init[:, :min_T, :]
        traj_adv      = traj_adv[:,  :min_T, :]
        real_ds_trunc = real_dataset[:, :min_T, :]
    else:
        real_ds_trunc = real_dataset

    # ① 综合四维评估
    comp_results = evaluator.evaluate_comprehensive(
        z_original=z_prior,
        z_adversarial=z_adv,
        traj_original=traj_init,
        traj_adversarial=traj_adv,
        traj_real_dataset=real_ds_trunc,
        scene_graph=None,
        model=None,
    )

    # ② 增强版长尾事件分析
    longtail_results = evaluator.analyze_longtail_events_8s_enhanced(
        trajectories=traj_adv,
        vehicle_lengths=scene['vehicle_lengths'],
        vehicle_widths=scene['vehicle_widths'],
        dt=scene['dt'],
    )

    return {
        'scene_name':           scene['scene_name'],
        'category':             scene['category'],
        'trajectory_realism':   comp_results.trajectory_realism,
        'interaction_consistency': comp_results.interaction_consistency,
        'longtail_coverage':    comp_results.longtail_coverage,
        'sim_to_real_gap':      comp_results.sim_to_real_gap,
        'longtail_event_analysis': {
            'has_longtail_event':        longtail_results['has_longtail_event'],
            'longtail_event_types':      longtail_results['longtail_event_types'],
            'num_vehicles_involved':     longtail_results['num_vehicles_involved'],
            'total_events':              longtail_results['total_events'],
            'analysis_summary':          longtail_results.get('analysis_summary', {}),
        },
    }


# ──────────────────────────────────────────────────────────────
# 聚合统计
# ──────────────────────────────────────────────────────────────

def aggregate_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按类别分组并计算均值"""

    def mean_dict(dicts):
        if not dicts:
            return {}
        keys = [k for k, v in dicts[0].items() if isinstance(v, (int, float))]
        return {k: float(np.mean([d[k] for d in dicts if isinstance(d.get(k), (int, float))]))
                for k in keys}

    categories = {}
    for r in all_results:
        cat = r['category']
        categories.setdefault(cat, []).append(r)

    summary = {}
    for cat, items in categories.items():
        summary[cat] = {
            'count': len(items),
            'trajectory_realism':      mean_dict([i['trajectory_realism'] for i in items]),
            'interaction_consistency': mean_dict([i['interaction_consistency'] for i in items]),
            'longtail_coverage':       mean_dict([i['longtail_coverage'] for i in items]),
            'sim_to_real_gap':         mean_dict([i['sim_to_real_gap'] for i in items]),
            'longtail_hit_rate':       float(np.mean([
                1 if i['longtail_event_analysis']['has_longtail_event'] else 0
                for i in items
            ])),
        }

    # 全局汇总
    summary['ALL'] = {
        'count': len(all_results),
        'trajectory_realism':      mean_dict([r['trajectory_realism'] for r in all_results]),
        'interaction_consistency': mean_dict([r['interaction_consistency'] for r in all_results]),
        'longtail_coverage':       mean_dict([r['longtail_coverage'] for r in all_results]),
        'sim_to_real_gap':         mean_dict([r['sim_to_real_gap'] for r in all_results]),
        'longtail_hit_rate':       float(np.mean([
            1 if r['longtail_event_analysis']['has_longtail_event'] else 0
            for r in all_results
        ])),
    }

    return summary


# ──────────────────────────────────────────────────────────────
# 打印摘要
# ──────────────────────────────────────────────────────────────

def print_summary(summary: Dict[str, Any]):
    print("\n" + "=" * 70)
    print("  批量评估摘要")
    print("=" * 70)

    for cat, stats in summary.items():
        print(f"\n【{cat}】  共 {stats['count']} 个场景")
        print(f"  长尾事件触发率          : {stats['longtail_hit_rate']:.2%}")

        tr = stats['trajectory_realism']
        if tr:
            print(f"  轨迹真实性总分           : {tr.get('overall_realism_score', float('nan')):.4f}")

        ic = stats['interaction_consistency']
        if ic:
            print(f"  交互合理性总分           : {ic.get('overall_interaction_score', float('nan')):.4f}")

        lc = stats['longtail_coverage']
        if lc:
            print(f"  长尾覆盖率总分           : {lc.get('overall_longtail_score', float('nan')):.4f}")
            print(f"  物理长尾命中率           : {lc.get('physical_longtail_hit_rate', float('nan')):.4f}")

        sg = stats['sim_to_real_gap']
        if sg:
            print(f"  Sim-to-Real 总分         : {sg.get('overall_sim_to_real_score', float('nan')):.4f}")

    print("=" * 70)


# ──────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="对规则对抗生成结果进行四维评估")
    parser.add_argument(
        '--out_dir',
        default='out/adv_gen_rule_based_out_1773900166-best',
        help='生成结果根目录（包含 scenario_results/）'
    )
    parser.add_argument(
        '--subset',
        nargs='+',
        default=['high_risk', 'longtail_condition', 'low_risk'],
        help='要评估的子集，可选: high_risk longtail_condition low_risk'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='计算设备'
    )
    parser.add_argument(
        '--save_json',
        default='out/eval_rule_based_results.json',
        help='保存详细结果的 JSON 路径'
    )
    parser.add_argument(
        '--save_summary',
        default='out/eval_rule_based_summary.json',
        help='保存聚合摘要的 JSON 路径'
    )
    parser.add_argument(
        '--max_scenes',
        type=int,
        default=None,
        help='调试用：最多评估前 N 个场景'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 初始化 Logger
    os.makedirs(os.path.join(_script_dir, 'out'), exist_ok=True)
    Logger.init(os.path.join(_script_dir, 'out', 'eval_rule_based.log'))

    result_dir = os.path.join(_script_dir, args.out_dir, 'scenario_results')
    if not os.path.isdir(result_dir):
        print(f"[错误] 场景目录不存在: {result_dir}")
        sys.exit(1)

    # ① 收集所有 JSON 路径
    all_paths = collect_all_scenes(result_dir, args.subset)
    if args.max_scenes is not None:
        all_paths = all_paths[:args.max_scenes]
    print(f"共找到 {len(all_paths)} 个场景文件，子集: {args.subset}")

    # ② 构建 real_dataset（用所有场景的 fut_init 聚合）
    print("构建 real_dataset 基准（fut_init 聚合）...")
    real_dataset = build_real_dataset_tensor(all_paths)
    print(f"real_dataset shape: {real_dataset.shape}")

    # ③ 初始化评估器（dt 从第一个文件读取，所有文件 dt 应相同）
    with open(all_paths[0], 'r') as f:
        sample_dt = float(json.load(f)['dt'])
    print(f"检测到 dt = {sample_dt}s，以此初始化评估器")

    evaluator = AdversarialTrajectoryEvaluator(
        device=args.device,
        verbose=False,   # 批量模式关闭逐场景打印
        dt=sample_dt,
    )

    # ④ 批量评估
    all_results = []
    failed = 0
    for idx, path in enumerate(all_paths):
        scene_name = Path(path).stem
        try:
            scene = load_scene_json(path)
            result = evaluate_single_scene(evaluator, scene, real_dataset)
            all_results.append(result)
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  [{idx+1}/{len(all_paths)}] {scene_name} ✓")
        except Exception as e:
            print(f"  [{idx+1}/{len(all_paths)}] {scene_name} ✗  错误: {e}")
            failed += 1

    print(f"\n评估完成：{len(all_results)} 成功 / {failed} 失败")

    # ⑤ 聚合统计
    summary = aggregate_results(all_results)
    print_summary(summary)

    # ⑥ 保存结果
    os.makedirs(os.path.dirname(os.path.join(_script_dir, args.save_json)), exist_ok=True)

    detail_path  = os.path.join(_script_dir, args.save_json)
    summary_path = os.path.join(_script_dir, args.save_summary)

    # 详细结果中 Tensor 转 list
    def to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump(to_serializable(all_results), f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {detail_path}")

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(to_serializable(summary), f, ensure_ascii=False, indent=2)
    print(f"聚合摘要已保存: {summary_path}")


if __name__ == '__main__':
    main()
