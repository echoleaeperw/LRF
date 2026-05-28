"""
使用 NuScenes 地图多边形精确检测轨迹是否离开可行驶区域。
对每辆车的 future 轨迹点检查是否落在 drivable_area 多边形内。
"""

import os, sys, json, glob, argparse
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import prepared


def load_drivable_area(map_json_path, map_name):
    """加载地图的 drivable_area 多边形，新加坡地图需要 y 轴翻转"""
    with open(map_json_path) as f:
        mdata = json.load(f)

    node_lut = {n["token"]: (n["x"], n["y"]) for n in mdata["node"]}
    poly_lut = {}
    for p in mdata["polygon"]:
        coords = [node_lut[t] for t in p["exterior_node_tokens"] if t in node_lut]
        if len(coords) >= 3:
            poly_lut[p["token"]] = Polygon(coords)

    polys = []
    for da in mdata["drivable_area"]:
        for pt in da["polygon_tokens"]:
            if pt in poly_lut and poly_lut[pt].is_valid:
                polys.append(poly_lut[pt])

    if len(polys) == 0:
        return None, 0.0
    merged = unary_union(polys).buffer(2.0)

    NUSC_MAP_SIZES = {
        'singapore-onenorth': [2025.0, 1585.6],
        'singapore-hollandvillage': [2922.9, 2808.3],
        'singapore-queenstown': [3687.1, 3228.6],
        'boston-seaport': [2118.1, 2979.5],
    }
    is_singapore = map_name.startswith("singapore")
    flip_h = NUSC_MAP_SIZES[map_name][0] if is_singapore else 0.0
    return prepared.prep(merged), flip_h


MAP_NAMES = [
    "singapore-onenorth", "singapore-hollandvillage",
    "singapore-queenstown", "boston-seaport"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--map_dir", default=None, help="NuScenes maps directory")
    args = parser.parse_args()

    scenario_dir = os.path.join(args.run_dir, "scenario_results")
    map_dir = args.map_dir or os.path.join(os.path.dirname(args.run_dir), "data/nuscenes/maps")
    if not os.path.isdir(map_dir):
        map_dir = os.path.join(os.path.dirname(os.path.dirname(args.run_dir)), "data/nuscenes/maps")

    print(f"地图目录: {map_dir}")
    drivable_cache = {}
    for mname in MAP_NAMES:
        mjson = os.path.join(map_dir, f"{mname}.json")
        if os.path.exists(mjson):
            print(f"  加载 {mname}...")
            drivable_cache[mname] = load_drivable_area(mjson, mname)
        else:
            print(f"  [WARN] 未找到 {mjson}")

    scene_files = sorted(glob.glob(os.path.join(scenario_dir, "**", "scene_*.json"), recursive=True))
    print(f"找到 {len(scene_files)} 个场景文件\n")

    role_total = {"ego": 0, "attacker": 0, "background": 0}
    role_offroad = {"ego": 0, "attacker": 0, "background": 0}
    scene_count = 0
    scene_offroad_count = 0

    for sf in scene_files:
        with open(sf) as f:
            data = json.load(f)

        map_name = data.get("map", "")
        if map_name not in drivable_cache or drivable_cache[map_name][0] is None:
            continue

        prep_geom, flip_h = drivable_cache[map_name]
        fut = np.array(data["fut_adv"])
        N, T, _ = fut.shape
        atk_idx = data.get("attack_agt", -1)
        scene_count += 1
        scene_has_offroad = False

        for i in range(N):
            role = "ego" if i == 0 else ("attacker" if i == atk_idx else "background")
            role_total[role] += 1
            offroad = False
            for t in range(T):
                xy = fut[i, t, :2]
                if np.any(np.isnan(xy)):
                    continue
                y_check = (flip_h - xy[1]) if flip_h > 0 else xy[1]
                if not prep_geom.contains(Point(xy[0], y_check)):
                    offroad = True
                    break
            if offroad:
                role_offroad[role] += 1
                scene_has_offroad = True

        if scene_has_offroad:
            scene_offroad_count += 1

    print(f"{'='*60}")
    print(f"Off-Road Analysis (NuScenes Map, {scene_count} scenes)")
    print(f"{'='*60}")
    print(f"{'Role':<15} {'Total':>8} {'OffRoad':>8} {'Rate':>8}")
    print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    for role in ["ego", "attacker", "background"]:
        t = role_total[role]
        o = role_offroad[role]
        pct = (o / t * 100) if t > 0 else 0
        print(f"{role:<15} {t:>8} {o:>8} {pct:>7.2f}%")
    total_v = sum(role_total.values())
    total_o = sum(role_offroad.values())
    print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    print(f"{'All':<15} {total_v:>8} {total_o:>8} {total_o/max(total_v,1)*100:>7.2f}%")
    print(f"\nScene-level: {scene_offroad_count}/{scene_count} "
          f"({scene_offroad_count/max(scene_count,1)*100:.1f}%)")

    out_path = os.path.join(args.run_dir, "dynamics_analysis", "offroad_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "total_scenes": scene_count,
            "role_total": role_total,
            "role_offroad": role_offroad,
            "role_rate_pct": {r: round(role_offroad[r] / max(role_total[r], 1) * 100, 2)
                              for r in role_total},
            "scene_offroad_count": scene_offroad_count,
        }, f, indent=2)
    print(f"保存至: {out_path}")


if __name__ == "__main__":
    main()
