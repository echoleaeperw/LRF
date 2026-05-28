"""
CARLA 场景脚本生成模块。

从 ScenarioExtractor 产出的结构化 scenario_data (dict) 生成可在 CARLA 中回放的 Python 脚本。
"""

import os
import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_VEHICLE_BLUEPRINTS = {
    "ego_vehicle": "vehicle.tesla.model3",
    "car": "vehicle.tesla.model3",
    "truck": "vehicle.carlamotors.carlacola",
    "bus": "vehicle.volkswagen.t2",
    "motorcycle": "vehicle.yamaha.yzf",
    "bicycle": "vehicle.diamondback.century",
}
_DEFAULT_BLUEPRINT = "vehicle.audi.a2"

_CARLA_TEMPLATE = r'''#!/usr/bin/env python
"""Auto-generated CARLA scenario replay script."""

import carla
import math
import time
import json
import argparse


SCENARIO_DATA = __SCENARIO_DATA_PLACEHOLDER__


def main():
    parser = argparse.ArgumentParser(description='CARLA Scenario Replay')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=2000, type=int)
    parser.add_argument('--sync', action='store_true')
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    target_map = SCENARIO_DATA.get("map", "")
    if target_map and world.get_map().name != target_map:
        try:
            world = client.load_world(target_map)
        except Exception:
            print(f"Cannot load map {target_map}, using current map")

    dt = SCENARIO_DATA.get("dt", 0.5)
    if args.sync:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = dt
        world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicles = []

    try:
        for v_info in SCENARIO_DATA["vehicles"]:
            traj = v_info["trajectory"]
            if not traj:
                continue
            bp_name = __BLUEPRINT_MAP__.get(v_info.get("type", ""), __DEFAULT_BP__)
            bp = blueprint_library.filter(bp_name)[0]
            init = traj[0]
            transform = carla.Transform(
                carla.Location(x=init["x"], y=init["y"], z=0.5),
                carla.Rotation(yaw=init["heading"]),
            )
            actor = world.spawn_actor(bp, transform)
            vehicles.append(actor)

        max_frames = max(len(v["trajectory"]) for v in SCENARIO_DATA["vehicles"])
        print(f"Replaying {len(vehicles)} vehicles for {max_frames} frames ...")

        for frame in range(max_frames):
            if args.sync:
                world.tick()
            else:
                time.sleep(dt)
            for idx, v_info in enumerate(SCENARIO_DATA["vehicles"]):
                traj = v_info["trajectory"]
                if frame >= len(traj):
                    continue
                pt = traj[frame]
                vehicles[idx].set_transform(
                    carla.Transform(
                        carla.Location(x=pt["x"], y=pt["y"], z=0.5),
                        carla.Rotation(yaw=pt["heading"]),
                    )
                )
                vel = pt.get("velocity", 0)
                if vel > 0:
                    heading_rad = math.radians(pt["heading"])
                    vehicles[idx].set_target_velocity(
                        carla.Vector3D(
                            x=math.cos(heading_rad) * vel,
                            y=math.sin(heading_rad) * vel,
                            z=0,
                        )
                    )
        print("Replay finished.")
    finally:
        client.apply_batch([carla.command.DestroyActor(v) for v in vehicles])
        if args.sync:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)


if __name__ == "__main__":
    main()
'''


def generate_carla_scenario_script(
    scenario_data: Dict,
    output_path: Optional[str] = None,
) -> str:
    """
    从结构化 scenario_data 生成 CARLA 回放脚本。

    参数:
        scenario_data: ScenarioExtractor.extract_structured_scenario 的输出字典。
        output_path: 若提供则将脚本写入该路径。

    返回:
        生成的 Python 脚本字符串。
    """
    data_str = json.dumps(scenario_data, indent=2, ensure_ascii=False)
    bp_map_str = json.dumps(_VEHICLE_BLUEPRINTS, indent=2, ensure_ascii=False)

    script = _CARLA_TEMPLATE
    script = script.replace("__SCENARIO_DATA_PLACEHOLDER__", data_str)
    script = script.replace("__BLUEPRINT_MAP__", bp_map_str)
    script = script.replace("__DEFAULT_BP__", f'"{_DEFAULT_BLUEPRINT}"')

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(script)
        logger.info(f"CARLA script saved: {output_path}")

    return script
