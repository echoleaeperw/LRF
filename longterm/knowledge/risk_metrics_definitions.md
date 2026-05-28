# Risk Level and Safety Metrics Definitions

---

## I. Risk Level Classification Standards

### 1. longtail_condition (Long-tail Extreme Scenarios)

**Definition**: Extreme conflict scenarios that rarely occur in real-world traffic but have severe consequences, representing the most challenging edge cases for autonomous driving systems.

**Typical Characteristics**:
- Collision is almost unavoidable (TTC < 0.5s)
- Extremely dangerous inter-vehicle distance (MinDist_lat < 0.3m)
- Excessive speed difference (RelativeSpeed > 25 m/s, approximately 90 km/h)
- Extreme collision angles (< 15° head-on collision or > 165° high-speed rear-end)

**Judgment Rule**: Meeting **any one** of the above conditions qualifies as a long-tail extreme scenario

**Generation Objective**: Create extreme adversarial scenarios approaching physical limits while still complying with traffic rules

---

### 2. high_risk (High-Risk Scenarios)

**Definition**: High-risk traffic conflict scenarios that are relatively rare but possible in reality.

**Typical Characteristics**:
- Significant collision risk but with reaction time available (TTC < 2.0s)
- Inter-vehicle distance approaching safety boundary (MinDist_lat < 1.0m)
- High relative speed (RelativeSpeed > 15 m/s, approximately 54 km/h)

**Judgment Rule**: Meeting **any one** of the above conditions qualifies as a high-risk scenario

**Generation Objective**: Create challenging yet physically plausible adversarial scenarios

---

### 3. low_risk (Low-Risk Scenarios)

**Definition**: Regular safe driving scenarios in daily traffic.

**Typical Characteristics**: Does not meet any conditions of high-risk or long-tail extreme scenarios

**Generation Objective**: Validate system performance under normal scenarios

---

## II. Core Safety Metrics Definitions

The following seven metrics are core quantitative tools for adversarial scenario generation. When formulating strategies, you must fully understand their physical meanings and adversarial usage methods.

### 1. TTC (Time-To-Collision)

**Physical Meaning**: The time required for a collision to occur if both vehicles maintain their current speed and direction

**Calculation Method**: `TTC = Relative Distance / Relative Approach Speed`

**Unit**: seconds (s)

**Risk Interpretation**: 
- Smaller values indicate more imminent collision
- < 2.0s indicates high risk
- < 0.5s indicates extreme risk (almost unavoidable)

**Adversarial Strategy**: Reduce TTC value by maximizing relative approach speed or minimizing distance

---

### 2. THW (Time Headway)

**Physical Meaning**: The time interval required for the following vehicle to reach a point after the leading vehicle passes it

**Calculation Method**: `THW = Longitudinal Distance / Ego Vehicle Speed`

**Unit**: seconds (s)

**Risk Interpretation**: 
- Smaller values indicate closer following
- < 1.0s indicates extreme danger (almost tailgating)
- < 2.0s indicates high risk

**Adversarial Strategy**: Make the attacking vehicle closely follow the target vehicle, compressing the safety buffer zone

---

### 3. MinDist_lat (Minimum Lateral Distance)

**Physical Meaning**: The minimum spacing between vehicles in the lateral direction (perpendicular to the driving direction)

**Calculation Method**: Closest point distance considering vehicle geometric dimensions

**Unit**: meters (m)

**Risk Interpretation**: 
- < 1.0m indicates high risk (potential scraping)
- < 0.3m indicates extreme risk (collision almost certain)

**Adversarial Strategy**: Compress lateral safety space through aggressive lane changes or lateral cut-ins

---

### 4. YawRate (Yaw Rate / Heading Rate)

**Physical Meaning**: The rate of change in vehicle heading angle, reflecting the intensity of steering actions

**Calculation Method**: `YawRate = Δ Heading Angle / Δ Time`

**Unit**: degrees/second (deg/s) or radians/second (rad/s)

**Risk Interpretation**: 
- Larger values indicate more aggressive steering
- > 15 deg/s indicates sharp turns
- > 0.3 rad/s (approximately 17 deg/s) indicates high risk

**Adversarial Strategy**: Create sudden lane changes or steering actions, compressing the target vehicle's reaction time

---

### 5. DeltaV / RelativeSpeed (Relative Speed)

**Physical Meaning**: The speed difference between two vehicles, positive values indicate approach, negative values indicate separation

**Calculation Method**: `DeltaV = Ego Vehicle Speed - Target Vehicle Speed` (longitudinal component)

**Unit**: meters/second (m/s) or kilometers/hour (km/h)

**Risk Interpretation**: 
- Larger absolute values indicate faster relative motion
- > 15 m/s (approximately 54 km/h) indicates high risk
- > 25 m/s (approximately 90 km/h) indicates extreme risk

**Adversarial Strategy**: Create large speed differences through speeding or sudden braking

---

### 6. TLC (Time-to-Lane-Crossing)

**Physical Meaning**: The time required for a vehicle to cross lane boundaries at its current lateral speed

**Calculation Method**: `TLC = Lateral Distance to Lane Boundary / Lateral Speed`

**Unit**: seconds (s)

**Risk Interpretation**: 
- < 1.0s indicates imminent lane departure
- < 0.5s indicates extreme danger

**Adversarial Strategy**: Create lane departure or line-crossing behaviors to form oncoming conflicts

---

### 7. CollisionAngle (Collision Angle)

**Physical Meaning**: The angle between the velocity vectors of two vehicles, reflecting the directional characteristics of collision

**Calculation Method**: Calculate angle based on the dot product of two vehicle velocity vectors

**Unit**: degrees (°)

**Risk Interpretation**: 
- < 15° indicates head-on collision (extremely dangerous)
- 80-100° indicates side-impact collision (T-bone)
- > 165° indicates same-direction rear-end (extremely dangerous at high speeds)

**Adversarial Strategy**: Create collision scenarios with specific angles through precise path planning

---

## III. Correspondence Between Metrics and Risk Levels

When formulating adversarial strategies, you need to reasonably select and combine the above metrics according to the target risk level:

| Risk Level | Core Focus Metrics | Typical Threshold Combinations |
|---------|-------------|-------------|
| **longtail_condition** | TTC, MinDist_lat, RelativeSpeed, CollisionAngle | TTC < 0.5s **OR** MinDist_lat < 0.3m **OR** RelativeSpeed > 25 m/s **OR** Extreme CollisionAngle |
| **high_risk** | TTC, MinDist_lat, RelativeSpeed, YawRate | TTC < 2.0s **OR** MinDist_lat < 1.0m **OR** RelativeSpeed > 15 m/s |
| **low_risk** | All metrics | Ensure all metrics are within safe ranges |

**Important Note**: When formulating strategies, you must explicitly specify how to adjust physical parameters (speed, acceleration, lateral position, heading angle, etc.) to achieve the metric thresholds corresponding to the target risk level.

---

## Appendix: Common Unit Conversions

- **Speed**: 1 m/s ≈ 3.6 km/h
- **Angle**: 1 rad ≈ 57.3 deg
- **Typical Vehicle Dimensions**: Length 4.5-5.0m, Width 1.8-2.0m

