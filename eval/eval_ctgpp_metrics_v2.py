#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 feasibility_reporter.py 和 offroad_analysis.py 的已有逻辑，
按 CTG++ Table 1 四个指标评估 STRIVE 生成的对抗场景。

指标定义（严格对齐 CTG++ 论文）：
  fail     — (碰撞 Agent 数 + 出路 Agent 数) / 总 Agent 数  ↓越低越好
  rule     — 规则违反度：STRIVE 规则 = 造成碰撞，violation = 未达到碰撞目标的场景比例  ↓
  real     — Agent 级运动学 Wasserstein 距离均值（a_lon / a_lat / jerk）  ↓
  rel real — 场景级相对运动学 Wasserstein 距离均值（agent-pair 相对 a_lon / a_lat / jerk）  ↓

GT 参考：fut_init（模型名义预测，即无对抗引导的正常生成）
用法：
  cd <仓库根目录>
  conda activate strive   # 或你的环境
  python eval/eval_ctgpp_metrics_v2.py \
      --run_dir out/adv_gen_rule_based_out_1773900166-best \
      --map_dir ./data/nuscenes/maps
"""
import os, sys, glob, json, argparse
import numpy as np
from scipy.stats import wasserstein_distance

# ── 把项目 src 加入路径（本文件位于仓库根下的 eval/） ───────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from utils.feasibility_reporter import FeasibilityReporter, _get_corners   # noqa: E402

# ── 地图加载（复用 offroad_analysis.py 逻辑） ─────────────────────────────────
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely import prepared

NUSC_MAP_SIZES = {
    'singapore-onenorth':      [2025.0, 1585.6],
    'singapore-hollandvillage':[2922.9, 2808.3],
    'singapore-queenstown':    [3687.1, 3228.6],
    'boston-seaport':          [2118.1, 2979.5],
}
_MAP_CACHE = {}


def load_drivable_area(map_dir, map_name):
    if map_name in _MAP_CACHE:
        return _MAP_CACHE[map_name]
    mjson = os.path.join(map_dir, f"{map_name}.json")
    if not os.path.exists(mjson):
        _MAP_CACHE[map_name] = (None, 0.0)
        return None, 0.0
    with open(mjson) as f:
        mdata = json.load(f)
    node_lut = {n["token"]: (n["x"], n["y"]) for n in mdata["node"]}
    poly_lut = {}
    for p in mdata["polygon"]:
        coords = [node_lut[t] for t in p["exterior_node_tokens"] if t in node_lut]
        if len(coords) >= 3:
            poly_lut[p["token"]] = Polygon(coords)
    polys = []
    for da in mdata.get("drivable_area", []):
        for pt in da["polygon_tokens"]:
            if pt in poly_lut and poly_lut[pt].is_valid:
                polys.append(poly_lut[pt])
    if not polys:
        _MAP_CACHE[map_name] = (None, 0.0)
        return None, 0.0
    merged   = unary_union(polys).buffer(2.0)
    flip_h   = NUSC_MAP_SIZES[map_name][0] if map_name.startswith("singapore") else 0.0
    prep_poly = prepared.prep(merged)
    _MAP_CACHE[map_name] = (prep_poly, flip_h)
    return prep_poly, flip_h


def agent_offroad(traj_i, map_dir, map_name):
    """单辆车：轨迹中是否有任意一帧在可行驶区域外。"""
    poly, flip_h = load_drivable_area(map_dir, map_name)
    if poly is None:
        return False
    for t in range(traj_i.shape[0]):
        x, y = traj_i[t, 0], traj_i[t, 1]
        if np.isnan(x) or np.isnan(y):
            continue
        y_c = (flip_h - y) if flip_h > 0 else y
        if not poly.contains(Point(x, y_c)):
            return True
    return False


# ── 运动学提取（与 feasibility_reporter._check_vehicle_kinematics 同逻辑） ─────
def extract_kinematics(traj_i, dt):
    """
    返回 (a_lon, a_lat, jerk) 各为 1-D numpy 数组。
    traj_i: (T, 4)  (x, y, hx, hy)
    """
    T = traj_i.shape[0]
    if T < 3:
        return np.array([]), np.array([]), np.array([])

    dpos  = np.diff(traj_i[:, :2], axis=0) / dt       # (T-1, 2)
    speeds = np.linalg.norm(dpos, axis=1)              # (T-1)

    hx = traj_i[:-1, 2];  hy = traj_i[:-1, 3]
    hn = np.hypot(hx, hy); hn = np.where(hn < 1e-6, 1.0, hn)
    fwd = np.stack([hx/hn, hy/hn], axis=1)            # (T-1, 2)
    lat = np.stack([-hy/hn, hx/hn], axis=1)           # (T-1, 2)

    dvel  = np.diff(dpos, axis=0) / dt                 # (T-2, 2)
    a_lon = np.abs(np.sum(dvel * fwd[:-1], axis=1))    # (T-2)
    a_lat = np.abs(np.sum(dvel * lat[:-1], axis=1))    # (T-2)

    jerk = np.array([])
    if T >= 4:
        djrk = np.diff(dvel, axis=0) / dt             # (T-3, 2)
        jerk = np.linalg.norm(djrk, axis=1)           # (T-3)

    return a_lon, a_lat, jerk


# ── Wasserstein（归一化直方图，对齐 CTG++ 方式） ──────────────────────────────
def wasserstein_hist(gen_vals, gt_vals, n_bins=50, clip_pct=99):
    gen_vals = gen_vals[np.isfinite(gen_vals)]
    gt_vals  = gt_vals[np.isfinite(gt_vals)]
    if len(gen_vals) < 5 or len(gt_vals) < 5:
        return float('nan')
    hi   = np.percentile(np.concatenate([gen_vals, gt_vals]), clip_pct)
    bins = np.linspace(0.0, hi + 1e-8, n_bins + 1)
    gh, _ = np.histogram(gen_vals.clip(0, hi), bins=bins, density=True)
    rh, _ = np.histogram(gt_vals.clip(0, hi),  bins=bins, density=True)
    gh /= (gh.sum() + 1e-12);  rh /= (rh.sum() + 1e-12)
    bc = 0.5 * (bins[:-1] + bins[1:])
    return wasserstein_distance(bc, bc, gh, rh)


# ── 碰撞检测：attacker vs ego（用 _get_corners + Shapely IoU） ────────────────
def check_attacker_ego_collision(traj_adv, lw, atk_idx, ego_idx=0):
    """返回 (collision_detected:bool, iou:float, approach_speed:float)"""
    try:
        from shapely.geometry import Polygon as SPoly
    except ImportError:
        return False, 0.0, 0.0
    T = traj_adv.shape[0]
    peak_iou = 0.0; coll_t = -1
    for t in range(T):
        try:
            ac = _get_corners(traj_adv[atk_idx, t], lw[atk_idx])
            ec = _get_corners(traj_adv[ego_idx, t], lw[ego_idx])
            pa = SPoly(ac); pe = SPoly(ec)
            if not pa.is_valid or not pe.is_valid: continue
            u = pa.union(pe).area
            if u < 1e-8: continue
            iou = pa.intersection(pe).area / u
            if iou > 0.02 and coll_t < 0: coll_t = t
            if iou > peak_iou: peak_iou = iou
        except Exception:
            continue
    if coll_t < 0:
        return False, peak_iou, 0.0
    # approach speed at collision
    dt_approx = 0.5
    approach_v = 0.0
    if coll_t >= 1:
        av = (traj_adv[atk_idx, coll_t, :2] - traj_adv[atk_idx, coll_t-1, :2]) / dt_approx
        ev = (traj_adv[ego_idx, coll_t, :2] - traj_adv[ego_idx, coll_t-1, :2]) / dt_approx
        rp = traj_adv[atk_idx, coll_t, :2] - traj_adv[ego_idx, coll_t, :2]
        rd = np.linalg.norm(rp) + 1e-8
        approach_v = float(-np.dot(av - ev, rp / rd))
    return True, peak_iou, approach_v


# ── 主评估 ────────────────────────────────────────────────────────────────────
def evaluate(run_dir, map_dir, subdirs=None):
    if subdirs is None:
        subdirs = ["longtail_condition", "high_risk", "low_risk"]

    scene_dir = os.path.join(run_dir, "scenario_results")
    files = []
    for sub in subdirs:
        files += sorted(glob.glob(os.path.join(scene_dir, sub, "scene_*.json")))
    print(f"找到 {len(files)} 个场景文件 (子目录: {subdirs})")

    # 收集统计量
    total_agents = 0
    fail_agents  = 0      # collision OR offroad

    # rule: 场景级 — 未成功触发碰撞（attacker-ego）的场景数
    n_no_collision  = 0   # IoU ≤ 0.02（完全未碰撞）
    n_total_scenes  = 0

    # real: fut_adv 与 fut_init 的运动学分布
    adv_alon, adv_alat, adv_jerk = [], [], []
    ref_alon, ref_alat, ref_jerk = [], [], []

    # rel real: agent-pair 相对运动学
    adv_rel_alon, adv_rel_alat, adv_rel_jerk = [], [], []
    ref_rel_alon, ref_rel_alat, ref_rel_jerk = [], [], []

    reporter = FeasibilityReporter()
    skipped = 0

    for fpath in files:
        try:
            with open(fpath) as f:
                d = json.load(f)
        except Exception as e:
            skipped += 1; continue

        traj_adv  = np.array(d["fut_adv"],  dtype=np.float32)   # (NA, T, 4)
        traj_init = np.array(d["fut_init"], dtype=np.float32)   # (NA, T, 4)
        lw        = np.array(d["lw"],       dtype=np.float32)   # (NA, 2)
        dt        = float(d.get("dt", 0.5))
        map_name  = d.get("map", "boston-seaport")
        NA, T, _  = traj_adv.shape
        atk_idx   = int(d.get("attack_agt", 1))
        atk_idx   = max(1, min(atk_idx, NA - 1))
        n_total_scenes += 1

        # ── 1. fail: 每辆车是否碰撞 OR 出路 ──────────────────────────────
        # a) 出路检测（每辆车独立）
        offroad_mask = np.array([
            agent_offroad(traj_adv[i], map_dir, map_name) for i in range(NA)
        ])

        # b) 碰撞检测：attacker vs ego（精确 OBB IoU）
        coll_detected, peak_iou, approach_v = check_attacker_ego_collision(
            traj_adv, lw, atk_idx, ego_idx=0
        )
        coll_mask = np.zeros(NA, dtype=bool)
        if coll_detected:
            coll_mask[0]       = True   # ego
            coll_mask[atk_idx] = True   # attacker

        # c) background-background 碰撞（中心距离 < 阈值，快速检测）
        bg_mask = np.ones(NA, dtype=bool)
        bg_mask[0] = False; bg_mask[atk_idx] = False
        bg_indices = np.where(bg_mask)[0]
        for ii_pos, i in enumerate(bg_indices):
            for j in bg_indices[ii_pos + 1:]:
                for t in range(T):
                    dist = np.linalg.norm(traj_adv[i, t, :2] - traj_adv[j, t, :2])
                    min_r = (np.mean(lw[i]) + np.mean(lw[j])) / 2
                    if dist < min_r * 0.8:
                        coll_mask[i] = True; coll_mask[j] = True; break

        fail_mask     = coll_mask | offroad_mask
        total_agents += NA
        fail_agents  += int(fail_mask.sum())

        # ── 2. rule: attacker-ego 是否达到碰撞（IoU > 0.02） ─────────────
        if not coll_detected:
            n_no_collision += 1

        # ── 3. real: 单车运动学分布 ───────────────────────────────────────
        for i in range(NA):
            al, alt, jk = extract_kinematics(traj_adv[i],  dt)
            al0, alt0, jk0 = extract_kinematics(traj_init[i], dt)
            if len(al):  adv_alon.extend(al.tolist())
            if len(alt): adv_alat.extend(alt.tolist())
            if len(jk):  adv_jerk.extend(jk.tolist())
            if len(al0): ref_alon.extend(al0.tolist())
            if len(alt0):ref_alat.extend(alt0.tolist())
            if len(jk0): ref_jerk.extend(jk0.tolist())

        # ── 4. rel real: agent-pair 相对运动学 ───────────────────────────
        for i in range(NA):
            for j in range(i + 1, NA):
                for traj_pair, col_alon, col_alat, col_jerk in [
                    (traj_adv,  adv_rel_alon, adv_rel_alat, adv_rel_jerk),
                    (traj_init, ref_rel_alon, ref_rel_alat, ref_rel_jerk),
                ]:
                    rp = traj_pair[i, :, :2] - traj_pair[j, :, :2]   # (T, 2)
                    if T < 3: continue
                    rv   = np.diff(rp,  axis=0) / dt                   # (T-1, 2)
                    if len(rv) < 2: continue
                    ra   = np.diff(rv,  axis=0) / dt                   # (T-2, 2)
                    # 投影到 agent i 的朝向
                    hx = traj_pair[i, :-2, 2]; hy = traj_pair[i, :-2, 3]
                    hn = np.hypot(hx, hy); hn = np.where(hn < 1e-6, 1.0, hn)
                    fwd = np.stack([hx/hn, hy/hn], axis=1)
                    lft = np.stack([-hy/hn, hx/hn], axis=1)
                    col_alon.extend(np.abs(np.sum(ra * fwd, axis=1)).tolist())
                    col_alat.extend(np.abs(np.sum(ra * lft, axis=1)).tolist())
                    if len(ra) >= 2:
                        rj = np.linalg.norm(np.diff(ra, axis=0) / dt, axis=1)
                        col_jerk.extend(rj.tolist())

    print(f"处理完成: {n_total_scenes} 个场景，跳过 {skipped} 个\n")

    # ── 计算最终指标 ──────────────────────────────────────────────────────────
    fail = fail_agents / max(total_agents, 1)

    # rule = 未达成对抗目标（no collision）的场景比例
    rule = n_no_collision / max(n_total_scenes, 1)

    # real
    w_al  = wasserstein_hist(np.array(adv_alon), np.array(ref_alon))
    w_alt = wasserstein_hist(np.array(adv_alat), np.array(ref_alat))
    w_jk  = wasserstein_hist(np.array(adv_jerk), np.array(ref_jerk))
    real  = float(np.nanmean([w_al, w_alt, w_jk]))

    # rel real
    w_ral  = wasserstein_hist(np.array(adv_rel_alon), np.array(ref_rel_alon))
    w_ralt = wasserstein_hist(np.array(adv_rel_alat), np.array(ref_rel_alat))
    w_rjk  = wasserstein_hist(np.array(adv_rel_jerk), np.array(ref_rel_jerk))
    rel_real = float(np.nanmean([w_ral, w_ralt, w_rjk]))

    # ── 输出对比表 ────────────────────────────────────────────────────────────
    print("=" * 72)
    print("CTG++ Table 1 指标对比（对应规则：GPT collision / 对抗碰撞任务）")
    print("=" * 72)
    print(f"{'Method':<28} {'fail↓':>7} {'rule↓':>7} {'real↓':>7} {'rel real↓':>10}")
    print("-" * 72)
    # CTG++ 原文 GPT collision 规则数字
    baselines = [
        ("BITS",          0.176, 0.660, 0.107, 0.359),
        ("BITS+opt",      0.277, 0.130, 0.068, 0.362),
        ("CTG",           0.356, 0.000, 0.074, 0.349),
        ("CTG++ (best)",  0.264, 0.000, 0.085, 0.331),
    ]
    for name, f_, r_, re_, rr_ in baselines:
        print(f"{name:<28} {f_:>7.3f} {r_:>7.3f} {re_:>7.3f} {rr_:>10.3f}")
    print("-" * 72)
    print(f"{'STRIVE Ours (LLM)':<28} {fail:>7.3f} {rule:>7.3f} {real:>7.3f} {rel_real:>10.3f}")
    print("=" * 72)
    print()
    print("详细分解：")
    print(f"  fail    = {fail:.4f}  | "
          f"碰撞/出路 Agent: {fail_agents}/{total_agents} "
          f"(场景 {n_total_scenes} 个，avg {total_agents/n_total_scenes:.1f} agents/scene)")
    print(f"  rule    = {rule:.4f}  | "
          f"未成功碰撞（IoU≤0.02）: {n_no_collision}/{n_total_scenes} 场景")
    print(f"  real    = {real:.4f}  | "
          f"a_lon={w_al:.4f}  a_lat={w_alt:.4f}  jerk={w_jk:.4f}")
    print(f"           GT参考: fut_init (模型名义预测，{len(ref_alon)} 样本)")
    print(f"  rel real= {rel_real:.4f}  | "
          f"a_lon={w_ral:.4f}  a_lat={w_ralt:.4f}  jerk={w_rjk:.4f}")
    print()
    print("注：rule 对应 CTG++ 'GPT collision' 规则（双方目标均为发生碰撞）。")
    print("    real/rel real GT 为 fut_init（无引导的模型名义输出），")
    print("    CTG++ 使用 nuScenes GT，二者来源不同，Wasserstein 绝对值仅供参考对比。")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    result = {
        "n_scenes":    n_total_scenes,
        "n_agents":    total_agents,
        "subdirs":     subdirs,
        "fail":        round(fail, 4),
        "rule":        round(rule, 4),
        "real":        round(real, 4),
        "rel_real":    round(rel_real, 4),
        "detail": {
            "fail_agents":     fail_agents,
            "no_collision_scenes": n_no_collision,
            "real_alon":  round(w_al,  4),
            "real_alat":  round(w_alt, 4),
            "real_jerk":  round(w_jk,  4),
            "relreal_alon": round(w_ral,  4),
            "relreal_alat": round(w_ralt, 4),
            "relreal_jerk": round(w_rjk,  4),
        },
    }
    out_path = os.path.join(run_dir, "ctgpp_table1_metrics.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n结果已保存: {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",  required=True)
    parser.add_argument("--map_dir",  default=None)
    parser.add_argument("--subdirs",  nargs="+",
                        default=["longtail_condition", "high_risk", "low_risk"])
    args = parser.parse_args()

    map_dir = args.map_dir
    if map_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(args.run_dir)))
        map_dir = os.path.join(base, "data", "nuscenes", "maps")

    print(f"Run dir : {args.run_dir}")
    print(f"Map dir : {map_dir}")
    evaluate(
        run_dir = os.path.abspath(args.run_dir),
        map_dir = map_dir,
        subdirs = args.subdirs,
    )
