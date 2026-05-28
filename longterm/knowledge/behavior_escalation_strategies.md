# Adversarial Behavior Escalation Strategies



This knowledge base describes **transformation pathways** rather than static behavior patterns. For each scenario type:

1. **Identify current state** - Where is the scenario now? (safe, low_risk, high_risk)
2. **Determine target risk level** - Where do we want it to be? (high_risk, longtail_condition)
3. **Apply transformation strategy** - What specific changes are needed?
4. **Calculate parameter deltas** - By how much should each variable change?

---

## I. Lateral Interaction Scenarios

### 1. AggressiveCutIn - Lateral Intrusion Attack

**Scenario Archetype**: Vehicle performing lateral lane change/merge that invades ego vehicle's space
**Primary Metrics**: MinDist_lat, YawRate, TTC
**Collision Type**: cut_in, side-swipe

---

#### Pathway 1A: Safe/Low Risk → High Risk

**Starting Conditions (Typical Safe Scenario)**:
- Lateral separation: 2.5-4.0m between vehicles
- Lateral velocity: < 0.2 m/s (minimal lateral motion)
- Yaw rate: < 5 deg/s (normal lane keeping)
- Both vehicles traveling in adjacent lanes
- TTC: > 5s or infinite (no projected collision)

**Target Conditions (High Risk)**:
- MinDist_lat: < 1.0m
- YawRate: > 15 deg/s
- TTC: < 2.0s

**Transformation Strategy**:

| Variable | Current Value | Target Value | Required Change | Change Type |
|----------|--------------|--------------|-----------------|-------------|
| **Lateral Distance** | 2.5-4.0m | < 1.0m | **-1.5 to -3.0m** | Compress |
| **Lateral Velocity** | 0.1-0.2 m/s | 0.5-1.0 m/s | **+0.3-0.8 m/s** | Increase |
| **Yaw Rate** | 2-5 deg/s | 15-20 deg/s | **+10-15 deg/s** | Sharp increase |
| **Longitudinal Speed** | Similar to ego | +10-20 km/h | **+10-20 km/h** | Accelerate |
| **Approach Angle** | Parallel | 15-30° | **Angled approach** | Change trajectory |

**Critical Vehicle**: Adjacent vehicle (typically in left lane cutting right, or right lane cutting left)

**Attack Execution**:
- **Timing**: Execute when ego vehicle is vulnerable (mid-turn, accelerating, distracted)
- **Method**: Sharp steering input → rapid lateral displacement → compress distance
- **Duration**: Complete maneuver in 1.0-1.5 seconds (minimal warning time)

**Loss Function Configuration**:
```
L_AdversarialCrash: 0.90-1.00  (primary objective)
L_MinDist_lat:      0.80-1.00  (critical for cut-in)
L_YawRate:          0.60-0.80  (enable sharp maneuver)
L_TTC:              0.30-0.50  (secondary consideration)
L_MotionPrior:      0.05-0.10  (break normal patterns)
```

**Physical Plausibility Checks**:
- Maximum lateral acceleration: < 8 m/s² (typical vehicle limit)
- Steering angle feasible: < 35° at given speed
- No teleportation: smooth continuous motion

---

#### Pathway 1B: High Risk → Longtail Extreme

**Starting Conditions (Already High Risk)**:
- MinDist_lat: 0.5-1.0m
- YawRate: 15-20 deg/s
- TTC: 1.0-2.0s
- Cut-in maneuver already in progress

**Target Conditions (Longtail Extreme)**:
- MinDist_lat: < 0.3m (collision imminent)
- YawRate: > 25 deg/s (extremely aggressive)
- TTC: < 0.5s

**Additional Transformation Required**:

| Variable | Current (High Risk) | Target (Longtail) | Additional Change |
|----------|---------------------|-------------------|-------------------|
| **Lateral Distance** | 0.5-1.0m | < 0.3m | **-0.2-0.7m further** |
| **Yaw Rate** | 15-20 deg/s | > 25 deg/s | **+5-10 deg/s** |
| **TTC** | 1.0-2.0s | < 0.5s | **Halve time window** |
| **Execution Speed** | Rapid | Near-instantaneous | **Max acceleration** |

**Intensification Tactics**:
- Remove any gradual transition → make it sudden
- Maximize steering angle and rate
- Execute at point of no return for ego vehicle
- Combine with speed increase if needed

**Loss Function Adjustment** (from high_risk baseline):
```
L_AdversarialCrash: 1.00       (maximum)
L_MinDist_lat:      1.00       (absolute priority)
L_YawRate:          0.80-0.90  (push to physical limits)
L_MotionPrior:      0.02-0.05  (completely ignore normal behavior)
```

---

## II. Longitudinal Interaction Scenarios

### 2. SuddenBraking - Emergency Deceleration Attack

**Scenario Archetype**: Lead vehicle performs sudden deceleration causing rear-end collision risk
**Primary Metrics**: TTC, DeltaV, Deceleration
**Collision Type**: rear_end

---

#### Pathway 2A: Safe/Low Risk → High Risk

**Starting Conditions (Typical Safe Following)**:
- Following distance: 20-40m
- Relative velocity: near zero (±2 m/s)
- Lead vehicle speed: steady (50-80 km/h)
- TTC: > 5s
- Both vehicles maintaining constant speeds

**Target Conditions (High Risk)**:
- TTC: < 2.0s
- Deceleration: > 4.5 m/s²
- DeltaV: > 15 m/s (54 km/h)

**Transformation Strategy**:

| Variable | Current Value | Target Value | Required Change |
|----------|--------------|--------------|-----------------|
| **Lead Vehicle Speed** | 60 km/h (16.7 m/s) | 20-30 km/h | **Drop by 30-40 km/h** |
| **Deceleration Rate** | 0-1 m/s² (normal) | > 4.5 m/s² | **Emergency braking level** |
| **Speed Drop Time** | N/A | < 3 seconds | **Sudden execution** |
| **TTC** | > 5s | < 2.0s | **Compress by 3+ seconds** |
| **Following Distance** | 20-40m | Becomes critical | **Insufficient for reaction** |

**Critical Vehicle**: Lead vehicle (directly in front of ego)

**Attack Execution**:
- **Timing**: When ego vehicle has minimal warning (around curves, after crest, in dense traffic)
- **Method**: Apply maximum braking force instantly without gradual deceleration
- **Key**: Large initial separation creates false sense of safety
- **Duration**: Reach final speed in 2-3 seconds

**Loss Function Configuration**:
```
L_AdversarialCrash: 0.90-1.00  (primary objective)
L_TTC:              0.80-0.90  (critical for rear-end)
L_DeltaV:           0.60-0.80  (maximize speed differential)
L_THW:              0.40-0.60  (time headway pressure)
L_MotionPrior:      0.05-0.10  (unnatural sudden stop)
```

**Physical Plausibility Checks**:
- Maximum deceleration: < 9 m/s² (typical ABS limit, dry road)
- Brake lights activate immediately
- No reverse motion or instant teleportation

---

#### Pathway 2B: High Risk → Longtail Extreme

**Starting Conditions (Already High Risk)**:
- TTC: 1.0-2.0s
- Deceleration: 4.5-6.0 m/s²
- Lead vehicle already braking hard

**Target Conditions (Longtail Extreme)**:
- TTC: < 0.5s
- Deceleration: > 6.0 m/s² (near-maximum)
- DeltaV: > 25 m/s (~90 km/h)

**Additional Transformation Required**:

| Variable | Current (High Risk) | Target (Longtail) | Additional Change |
|----------|---------------------|-------------------|-------------------|
| **Deceleration** | 4.5-6.0 m/s² | > 6.0 m/s² | **Push to ABS limits** |
| **Final Speed** | 30-40 km/h | < 20 km/h | **Even lower endpoint** |
| **TTC** | 1.0-2.0s | < 0.5s | **Halve again** |

**Intensification Tactics**:
- Maximum possible deceleration (ABS activation)
- Drop speed to near-complete stop if possible
- Consider combined scenario: braking + slight swerve (ego must brake AND steer)

---

## III. Time-Proximity Scenarios

### 3. AggressiveTailgating - Persistent Close Following

**Scenario Archetype**: Following vehicle maintains dangerously short headway
**Primary Metrics**: THW, TTC, Longitudinal_Distance
**Collision Type**: rear_end

---

#### Pathway 3A: Safe → High Risk

**Starting Conditions (Safe Following)**:
- Time headway (THW): > 2.0s
- Following distance: > 30m at highway speeds, > 10m at city speeds
- Following vehicle maintains safe separation

**Target Conditions (High Risk)**:
- THW: < 1.0s
- Longitudinal distance: < 5m
- TTC: < 2.0s (any minor speed change triggers)

**Transformation Strategy**:

| Variable | Current Value | Target Value | Required Change |
|----------|--------------|--------------|-----------------|
| **Time Headway** | 2-3s | < 1.0s | **Reduce by 1-2s** |
| **Physical Distance** | 30-50m (highway) | < 5m | **Close gap by 25-45m** |
| **Speed Sync** | Independent | Tight coupling | **Mirror ego's speed** |
| **Reaction Delay** | Normal | Minimal | **Hair-trigger response** |

**Critical Vehicle**: Following vehicle (directly behind ego)

**Attack Execution**:
- **Timing**: Maintain pressure continuously, exploit any ego deceleration
- **Method**: Tight speed synchronization + minimal distance
- **Persistence**: Sustain the threat over extended time (10+ seconds)

**Loss Function Configuration**:
```
L_AdversarialCrash: 0.90-1.00
L_THW:              0.80-0.90  (primary metric for tailgating)
L_TTC:              0.60-0.80
L_DeltaV:           0.30-0.50
L_MotionPrior:      0.10-0.20
```

---

#### Pathway 3B: High Risk → Longtail Extreme

**Additional Transformation**:
- THW: < 0.5s (bumper-to-bumper)
- Distance: < 3m
- Any ego braking = immediate collision

---

## IV. Lateral Drift Scenarios

### 4. LaneDeparture - Progressive Lane Invasion

**Scenario Archetype**: Vehicle gradually drifts out of lane, invading adjacent/oncoming space
**Primary Metrics**: MinDist_lat (to lane boundary), TLC, YawRate
**Collision Type**: lane_departure, head-on

---

#### Pathway 4A: Safe → High Risk

**Starting Conditions (Normal Lane Keeping)**:
- Position: Center of lane
- Distance to lane boundary: > 1.5m
- Lateral velocity: near zero
- No drift or wandering

**Target Conditions (High Risk)**:
- MinDist_lat to boundary: < 1.0m
- TLC: < 1.0s
- Continuous drift toward boundary/adjacent lane

**Transformation Strategy**:

| Variable | Current Value | Target Value | Required Change |
|----------|--------------|--------------|-----------------|
| **Lateral Position** | Lane center | Near boundary | **Drift by 1.0-2.0m** |
| **Lateral Velocity** | ~0 m/s | 0.1-0.3 m/s | **Steady lateral drift** |
| **Yaw Rate** | 0-1 deg/s | 2-5 deg/s | **Sustained angle** |
| **Duration** | N/A | 2-4 seconds | **Gradual invasion** |

**Critical Vehicle**: Drifting vehicle (typically oncoming or adjacent)

**Attack Execution**:
- **Pattern**: Gradual, continuous drift (not sudden jerk)
- **Direction**: Cross center line (head-on) OR invade adjacent lane (sideswipe)
- **Timing**: On curves, areas with limited sight distance

**Loss Function Configuration**:
```
L_AdversarialCrash:      0.80-0.90
L_MinDist_lat:           0.80-1.00  (to lane boundary or ego)
L_YawRate:               0.50-0.70  (sustained drift)
L_EnvironmentCollision:  0.60-0.80  (if head-on scenario)
L_MotionPrior:           0.10-0.20
```

---

## V. Speed Differential Scenarios

### 5. Overspeeding - Velocity-Based Attack

**Scenario Archetype**: Vehicle approaching at excessive speed creating closure rate danger
**Primary Metrics**: DeltaV, TTC, RelativeSpeed
**Collision Type**: rear_end (high-energy)

---

#### Pathway 5A: Safe → High Risk

**Starting Conditions (Normal Traffic)**:
- Speed differential: < 10 km/h
- Approach rate: gradual
- Closing distance: > 50m with ample time

**Target Conditions (High Risk)**:
- DeltaV: > 15 m/s (54 km/h speed difference)
- TTC: < 2.0s
- Speed excess: > 30% over limit

**Transformation Strategy**:

| Variable | Current Value | Target Value | Required Change |
|----------|--------------|--------------|-----------------|
| **Attacker Speed** | 60 km/h (normal) | 90-120 km/h | **+30-60 km/h** |
| **Speed Differential** | 0-10 km/h | > 50 km/h | **Massive increase** |
| **Approach Rate** | Slow | Rapid | **High closure rate** |
| **Warning Time** | Ample | Minimal | **Compress reaction window** |

**Critical Vehicle**: Speeding vehicle (typically approaching from behind)

**Loss Function Configuration**:
```
L_AdversarialCrash: 0.90-1.00
L_DeltaV:           0.80-0.90  (primary for overspeeding)
L_TTC:              0.60-0.80
L_MotionPrior:      0.05-0.10  (way above normal speeds)
```

---

## VI. Intersection Conflict Scenarios

### 6. IntersectionRush - Signal/Right-of-Way Violation

**Scenario Archetype**: Vehicle violates intersection control (red light, yield) creating path conflict
**Primary Metrics**: TTC, MinDist_lat, CollisionAngle
**Collision Type**: intersection_conflict, T-bone

---

#### Pathway 6A: Safe → High Risk

**Starting Conditions (Rule-Compliant Intersection)**:
- All vehicles following signals
- Proper right-of-way yielding
- Safe crossing margins

**Target Conditions (High Risk)**:
- TTC: < 2.0s
- MinDist_lat: < 1.5m at conflict point
- Signal violation: red light running
- Path intersection: direct conflict

**Transformation Strategy**:

| Variable | Current Value | Target Value | Required Change |
|----------|--------------|--------------|-----------------|
| **Signal Compliance** | Green (legal) | Red (violation) | **Ignore signal** |
| **Approach Speed** | Slowing | Maintained/increased | **No deceleration** |
| **Turn Rate** | N/A or gentle | > 15 deg/s | **Sharp turn** |
| **Timing** | Proper gap | Conflict timing | **Intercept ego path** |

**Critical Vehicle**: Intersection-rushing vehicle (crossing from side)

**Attack Execution**:
- **Timing**: Enter intersection precisely when ego vehicle committed
- **Method**: Red light run + high-speed turn (if turning)
- **Angle**: Maximize conflict (T-bone at 90° ideal for damage)

**Loss Function Configuration**:
```
L_AdversarialCrash: 0.90-1.00
L_TTC:              0.80-0.90
L_MinDist_lat:      0.70-0.90  (at conflict point)
L_YawRate:          0.60-0.80  (if turning)
L_MotionPrior:      0.05-0.10  (highly abnormal behavior)
```

---

## VII. Quick Reference Tables

### Risk Level Threshold Summary

| Risk Level | TTC | MinDist_lat | DeltaV | THW | YawRate |
|------------|-----|-------------|--------|-----|---------|
| **Safe/Low** | > 5s | > 2.5m | < 10 m/s | > 2.0s | < 5 deg/s |
| **High Risk** | < 2.0s | < 1.0m | > 15 m/s | < 1.0s | > 15 deg/s |
| **Longtail** | < 0.5s | < 0.3m | > 25 m/s | < 0.5s | > 25 deg/s |

### Scenario → Primary Metric Mapping

| Scenario Type | Primary Metric | Secondary Metrics | Critical Vehicle |
|---------------|----------------|-------------------|------------------|
| **AggressiveCutIn** | MinDist_lat | YawRate, TTC | Adjacent lane vehicle |
| **SuddenBraking** | TTC | DeltaV, Deceleration | Lead vehicle |
| **AggressiveTailgating** | THW | TTC, Distance | Following vehicle |
| **LaneDeparture** | MinDist_lat | TLC, YawRate | Drifting vehicle |
| **Overspeeding** | DeltaV | TTC, Speed | Speeding vehicle |
| **IntersectionRush** | TTC | MinDist_lat, Angle | Cross-traffic vehicle |

### Loss Function Priority by Scenario

| Scenario | Highest Weight | High Weight | Moderate | Low/Suppress |
|----------|----------------|-------------|----------|--------------|
| **Cut-In** | L_AdversarialCrash, L_MinDist_lat | L_YawRate | L_TTC | L_MotionPrior |
| **Braking** | L_AdversarialCrash, L_TTC | L_DeltaV | L_THW | L_MotionPrior |
| **Tailgating** | L_AdversarialCrash, L_THW | L_TTC | L_DeltaV | L_MotionPrior |
| **Departure** | L_MinDist_lat, L_EnvironmentCollision | L_YawRate | - | L_MotionPrior |
| **Overspeeding** | L_AdversarialCrash, L_DeltaV | L_TTC | - | L_MotionPrior |
| **Intersection** | L_AdversarialCrash, L_TTC | L_MinDist_lat, L_YawRate | - | L_MotionPrior |

---

## VIII. General Principles for Risk Escalation

### Principle 1: Identify the Gap
- Calculate the difference between current metric values and target thresholds
- Focus on metrics with largest gaps → these need most change

### Principle 2: Choose Critical Vehicle
- Which vehicle, if modified, can most efficiently create the target risk?
- Usually: closest vehicle in the relevant spatial relationship

### Principle 3: Calculate Required Deltas
- Don't just state targets (e.g., "TTC < 2.0s")
- Specify changes (e.g., "reduce TTC by 3-4 seconds from current 5s")

### Principle 4: Verify Physical Plausibility
- All changes must respect vehicle dynamics constraints
- No teleportation, no impossible accelerations
- Smooth continuous trajectories

### Principle 5: Multi-Metric Coordination
- Usually need to change 2-3 metrics simultaneously
- Example: AggressiveCutIn needs MinDist_lat reduction + YawRate increase + timing coordination

### Principle 6: Loss Function Alignment
- Higher risk levels → suppress L_MotionPrior more
- Scenario-specific weights (see tables above)
- Collision weights always high for adversarial generation

---

## IX. Example: Complete Risk Escalation Analysis

**Given Scenario**: 
- Ego vehicle traveling at 60 km/h in lane 2
- Adjacent vehicle in lane 1 traveling at 62 km/h, 15m ahead, lateral separation 3.2m
- Current TTC: infinite (no projected collision)
- Target: high_risk cut-in scenario

**Step 1: Identify Scenario Type** → AggressiveCutIn (lateral interaction)

**Step 2: Determine Current State**:
- MinDist_lat: 3.2m (safe)
- YawRate: ~2 deg/s (normal)
- Relative speed: +2 km/h (minor)

**Step 3: Determine Target State (High Risk)**:
- MinDist_lat: < 1.0m
- YawRate: > 15 deg/s
- TTC: < 2.0s

**Step 4: Calculate Required Changes**:
- Lateral distance: **reduce by 2.2-2.5m** (from 3.2m to <1.0m)
- Yaw rate: **increase by +13-15 deg/s** (from 2 to >15 deg/s)
- Speed: **increase by +10-15 km/h** relative to ego
- Execute sharp right steering maneuver over 1.0-1.5 seconds

**Step 5: Specify Attack Vector**:
- Critical vehicle: Adjacent vehicle (lane 1)
- Action: Accelerate to 70-75 km/h while executing sharp right turn
- Timing: Begin maneuver when 10m ahead of ego (mid-overlap position)
- Result: Cut directly into ego's path with <1m clearance

**Step 6: Configure Loss Weights**:
```
L_AdversarialCrash: 0.95
L_MinDist_lat: 0.90
L_YawRate: 0.70
L_TTC: 0.40
L_MotionPrior: 0.08
```

**Step 7: Verify Plausibility**:
- Lateral acceleration: ~4 m/s² (feasible)
- Steering angle: ~25° at 70 km/h (feasible)
- Time to complete: 1.2 seconds (realistic for aggressive maneuver)
- ✓ Physically plausible

---

**End of Knowledge Base**



*** 时间邻近场景

### 3. 恶意尾随 - 持续紧密跟随
 - 场景描述：自车（一辆白色丰田凯美瑞）在2车道以50公里/小时的速度平稳行驶。一辆银色宝马在1车道从后方以60公里/小时的速度接近
 - 高风险点：可以增加银色宝马车辆的速度，使得两车的相对速度达到10公里/小时以上，从而达到高风险点，造成紧密跟随
 - 风险行为：追尾行为
