# STRIVE Loss Function Knowledge Base

This document defines every loss term used in **`AdvGenLoss`** (the adversarial scenario generation
optimizer in `src/losses/adv_gen_nusc.py`).  
The AnalysisAgent MUST reference this KB when formulating `loss_priority_order` and `constraints_to_relax`.

---

## Critical Decision Table — What the LLM Controls vs. What is Auto-Computed

| Loss Term | Who Controls It | LLM's Role | Auto-Mechanism |
|-----------|----------------|-----------|----------------|
| `adv_crash` | **LLM + Auto** | Specify `attacker_vehicle_id` (→ `attack_agt_idx`). Set relative weight HIGH always. | Soft-min over all timesteps + agents; automatically focuses gradient on closest moment |
| `ttc` | **LLM** | Set HIGH for timing-critical attacks (cut-in, intersection rush, sudden braking) | None — only active if LLM sets weight > 0 |
| `min_dist_lat` | **LLM** | Set HIGH for lateral squeeze attacks (AggressiveCutIn, LaneDeparture, Pincer) | None — only active if LLM sets weight > 0 |
| `yaw_rate` | **LLM** | Set HIGH when sharp turns / lane changes are required (total yaw rate switch) | None — only active if LLM sets weight > 0 |
| `yaw_rate_ego` | **LLM** | Set HIGH when ego evasive steering matters (LaneDeparture, IntersectionRush); LOW when attacker drives the action | Overrides `yaw_rate` for ego vehicle only |
| `yaw_rate_non_ego` | **LLM** | Set HIGH when attacker needs aggressive turns (AggressiveCutIn, Pincer); LOW for speed-based attacks | Overrides `yaw_rate` for non-ego vehicles only |
| `motion_prior` / `motion_prior_atk` | **Auto (interpolated)** | Advise whether to RELAX for attacker | Automatically interpolated: non-attackers get high weight; attacker gets low `motion_prior_atk` weight based on `prior_reweight` from adv_crash |
| `init_z` / `init_z_atk` | **Auto** | Same as motion_prior | Same interpolation mechanism |
| `coll_veh` | **Rule** | Keep active always (LLM may slightly reduce for multi-threat scenarios) | Prevents unrealistic pile-ups |
| `coll_veh_plan` | **Rule** | Keep active — penalizes ego (planner) collisions with non-target vehicles | Prevents ego from crashing into bystanders |
| `coll_env` | **Rule** | Keep active (LLM may reduce only for LaneDeparture) | Prevents off-road driving |

**Key insight**: The LLM's two most important decisions are:
1. **Which vehicle is the attacker** (`attacker_vehicle_id` → `attack_agt_idx`) — this focuses ALL losses on one vehicle
2. **The ratio of `ttc : min_dist_lat : yaw_rate`** — this determines the *type* of attack

`adv_crash` should always be highest; `motion_prior_atk` should always be relaxed for the attacker.

---

## Overview: Two Categories of Loss Terms

| Category | Terms | Purpose |
|----------|-------|---------|
| **Adversarial** (maximize risk) | `adv_crash`, `ttc`, `min_dist_lat`, `yaw_rate`, `yaw_rate_ego`, `yaw_rate_non_ego` | Drive the attacker into dangerous proximity with ego |
| **Regularizer** (maintain realism) | `motion_prior`, `coll_veh`, `coll_veh_plan`, `coll_env`, `init_z` | Keep non-attacking vehicles realistic; prevent off-road driving |

---

## Adversarial Loss Terms

### `adv_crash` — Adversarial Crash Loss

**Key**: `adv_crash`  
**Direction**: Minimize (lower value = closer collision)  
**Effect**: Directly minimizes the **positional distance** between the attacker vehicle and ego over
all future timesteps, using a soft-min weighting to focus on the most dangerous moment.  

- The loss applies only to non-ego vehicles.  
- A `crash_loss_min_infront` threshold can optionally filter out attackers that are already behind ego.  
- When `attack_agt_idx` is specified (from LLM), only that specific vehicle receives the gradient.  

**When to prioritize HIGH**: All scenarios where you want a direct collision trajectory.  
**Typical weight range**: 1.0–5.0 (relative to other terms)

---

### `ttc` — Time-To-Collision Adversarial Loss (TTCLossAtk)

**Key**: `ttc`  
**Direction**: Minimize (lower TTC = more dangerous)  
**Effect**: Reduces the **time remaining before a collision** between attacker and ego. Implemented
via `TTCLossAtk` — an adversarial version that rewards approaches that reduce TTC below the safe
threshold (default `ttc_safe = 3.0s`).  

**Physical link**: TTC = Relative Distance / Relative Approach Speed  
**When to prioritize HIGH**: Cut-in, sudden braking, intersection rush — any scenario where timing is critical.  
**Risk threshold**: TTC < 2.0s → high risk; TTC < 0.5s → extreme risk  
**Typical weight range**: 0.5–2.0

---

### `min_dist_lat` — Minimum Lateral Distance Loss (MinDistLatLoss)

**Key**: `min_dist_lat`  
**Direction**: Minimize (lower = more lateral squeeze)  
**Effect**: Minimizes the **lateral (side-by-side) distance** between attacker and ego, accounting for
vehicle dimensions. Controlled by coefficient `k` (default 2.0) — higher k = stronger forcing.  

**Physical link**: MinDist_lat = perpendicular distance between vehicle edges  
**When to prioritize HIGH**: AggressiveCutIn, LaneDeparture, parallel merge conflicts.  
**Risk threshold**: < 1.0m → high risk; < 0.3m → extreme risk  
**Typical weight range**: 0.5–3.0

---

### `yaw_rate` — Yaw Rate Loss (YawRateLoss)

**Key**: `yaw_rate`  
**Direction**: Maximize (higher heading rate = more aggressive steering)  
**Effect**: Encourages the attacker (or ego) to make **sharp, sudden turns** by maximizing heading
angle change rate. Applies when heading rate exceeds `yaw_rate_threshold` (default 15 deg/s).  

**Physical link**: YawRate = |Δheading| / dt  
**When to prioritize HIGH**: Scenarios requiring sharp lane changes, intersection cuts, or spiral paths.  
**Risk threshold**: > 15 deg/s → sharp; > 30 deg/s → extreme  
**Typical weight range**: 0.1–1.0 (usually lower; amplifies the steering component)

---

### `yaw_rate_ego` — Ego Yaw Rate Weight

**Key**: `yaw_rate_ego`  
**Direction**: Maximize (for ego vehicle specifically)  
**Effect**: Controls how much the ego vehicle's yaw rate contributes to the yaw rate loss.
When set HIGH, it encourages the ego to make sharp evasive maneuvers (swerving).
When set LOW, the ego maintains a steady heading while the attacker does the aggressive steering.

**When to set HIGH**: LaneDeparture, IntersectionRush — scenarios where ego's evasive reaction matters.  
**When to set LOW**: AggressiveCutIn, SuddenBraking — the attacker drives the action, ego reacts passively.  
**Typical weight range**: 0.5–2.0

---

### `yaw_rate_non_ego` — Non-Ego (Attacker) Yaw Rate Weight

**Key**: `yaw_rate_non_ego`  
**Direction**: Maximize (for attacker/non-ego vehicles)  
**Effect**: Controls how much the attacker's yaw rate contributes to the yaw rate loss.
When set HIGH, it encourages the attacker to make sharp, aggressive lane changes or turns.
This is the primary steering aggressiveness control for the attacking vehicle.

**When to set HIGH**: AggressiveCutIn, MultiVehiclePincer, IntersectionRush — attacker needs sharp turns.  
**When to set LOW**: SuddenBraking, SuddenAcceleration — attacker mainly changes speed, not heading.  
**Typical weight range**: 0.5–2.0

---

## Regularizer / Constraint Terms

### `motion_prior` — Motion Prior Loss (CVAE Prior Regularization)

**Key**: `motion_prior` / `motion_prior_atk`  
**Direction**: Minimize (keeps trajectory on the learned distribution)  
**Effect**: Penalizes trajectories that deviate from the CVAE model's learned prior distribution.
Non-attackers use `motion_prior` (high weight = realistic). Attackers use `motion_prior_atk`
(low weight = allow aggressive deviations). The weight is interpolated based on attacker probability.  

**IMPORTANT**: Reducing `motion_prior` for the attacker is essential for ANY adversarial behavior.
Without relaxing this constraint, the model reverts to average/safe trajectories.  
**When to RELAX (set low)**: When attacker needs to deviate significantly from typical behavior.  
**When to KEEP HIGH**: For non-attacking background vehicles to maintain scene realism.

---

### `init_z` — Initial Latent Code Regularization

**Key**: `init_z` / `init_z_atk`  
**Direction**: Minimize (keeps latent z close to the initial sample)  
**Effect**: Penalizes the optimized latent code `z` from drifting too far from its initial value.
Works alongside `motion_prior` as a stability regularizer. Like `motion_prior`, attackers have a
separate `init_z_atk` weight.  

**When to RELAX**: Same as `motion_prior_atk` — relax for the attacker to enable aggressive z shifts.  
**When to KEEP HIGH**: For background vehicles.

---

### `coll_veh` — Vehicle-to-Vehicle Collision Avoidance

**Key**: `coll_veh` / `coll_veh_plan`  
**Direction**: Minimize (penalizes inter-vehicle collisions among non-target pairs)  
**Effect**: Prevents non-attacking vehicles from colliding with each other or with ego (except for
the intended attacker→ego collision). `coll_veh_plan` specifically penalizes planner (ego) collisions.  

**When to RELAX**: Generally keep this active to avoid unrealistic pile-ups.  
**Exception**: In extreme long-tail multi-threat scenarios, slightly reduce to allow secondary
attacker proximity.

---

### `coll_env` — Environmental Collision Avoidance

**Key**: `coll_env`  
**Direction**: Minimize (penalizes off-road driving)  
**Effect**: Penalizes vehicles from leaving the drivable area using the map environment.  

**When to RELAX**: Rarely — only for LaneDeparture scenarios where the attacker intentionally
leaves the road. Even then, keep non-zero to prevent full off-road trajectories.

---

## Loss Priority Templates by Scenario Type

Use these as starting points when generating `loss_priority_order` and `constraints_to_relax`:

### AggressiveCutIn
```
loss_priority_order:  adv_crash > min_dist_lat > yaw_rate > ttc
yaw_rate_ego:         LOW  (ego reacts passively)
yaw_rate_non_ego:     HIGH (attacker makes aggressive lane change)
constraints_to_relax: motion_prior_atk, init_z_atk
keep_active:          coll_veh (background), coll_env, motion_prior (background)
```

### LaneDeparture
```
loss_priority_order:  adv_crash > min_dist_lat > yaw_rate > ttc
yaw_rate_ego:         HIGH (ego evasive swerving matters)
yaw_rate_non_ego:     HIGH (attacker also steering aggressively)
constraints_to_relax: motion_prior_atk, init_z_atk, coll_env (slightly)
keep_active:          coll_veh (background), motion_prior (background)
```

### IntersectionRush / T-Bone
```
loss_priority_order:  adv_crash > ttc > min_dist_lat > yaw_rate
yaw_rate_ego:         MEDIUM (ego may need to swerve at intersection)
yaw_rate_non_ego:     HIGH   (attacker rushes through with sharp turn)
constraints_to_relax: motion_prior_atk, init_z_atk
keep_active:          coll_env, coll_veh (background)
```

### SuddenBraking / RearEnd
```
loss_priority_order:  adv_crash > ttc > yaw_rate
yaw_rate_ego:         LOW (ego brakes, doesn't swerve)
yaw_rate_non_ego:     LOW (attacker brakes, heading stays constant)
constraints_to_relax: motion_prior_atk, init_z_atk
keep_active:          coll_env, coll_veh (background)
```

### StationaryObstacleActivation
```
loss_priority_order:  adv_crash > min_dist_lat > ttc
yaw_rate_ego:         LOW    (ego reacts passively)
yaw_rate_non_ego:     MEDIUM (attacker turns into ego's path)
constraints_to_relax: motion_prior_atk, init_z_atk, coll_veh (attacker only)
keep_active:          coll_env, motion_prior (background)
```

### Multi-threat / Pincer
```
loss_priority_order:  adv_crash > ttc > min_dist_lat > yaw_rate
yaw_rate_ego:         MEDIUM (ego trapped, may swerve)
yaw_rate_non_ego:     HIGH   (multiple attackers converging with sharp turns)
constraints_to_relax: motion_prior_atk (for both attackers), init_z_atk
keep_active:          coll_env
```

---

## Loss Weight Naming Convention

The LLM should use these EXACT key names in `loss_priority_order` and `constraints_to_relax`:

| LLM Reference | Actual Loss Key | Notes |
|---------------|----------------|-------|
| `L_AdversarialCrash` | `adv_crash` | Primary crash driver — always set HIGH |
| `L_TTC` | `ttc` | Time-to-collision adversarial |
| `L_MinDist_lat` | `min_dist_lat` | Lateral distance squeeze |
| `L_YawRate` | `yaw_rate` | Steering aggressiveness (total switch) |
| `L_YawRate_Ego` | `yaw_rate_ego` | Ego yaw rate — HIGH for evasive scenarios |
| `L_YawRate_NonEgo` | `yaw_rate_non_ego` | Attacker yaw rate — HIGH for aggressive turns |
| `L_VehicleCollision` | `coll_veh` | Inter-vehicle collision avoidance |
| `L_VehicleCollision_Planner` | `coll_veh_plan` | Ego (planner) collision avoidance |
| `L_EnvironmentCollision` | `coll_env` | Environmental collision avoidance |
| `L_MotionBehavior` | `motion_prior_atk` | Attacker trajectory realism (RELAX for attacks) |
| `L_SceneSimilarity` | `init_z_atk` | Attacker latent code stability (RELAX for attacks) |
