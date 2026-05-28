# AnalysisAgent Human Message

Analyze the following structured driving scenario JSON and execute the complete 5-step COT reasoning process to identify the best long-tail / **{risk_level}** scenario generation strategy.

## Structured Scenario JSON

```json
{scenario_json}
```

## Critical Instructions

1. **Parse ALL JSON fields** — especially `map_context` for each vehicle (intersection status, lane direction, distance to centerline)
2. **Long-tail focus**: Look for RARE combinations — intersection + high speed, lane boundary + turning, multi-vehicle convergence
3. **Select ONLY active attackers**: `velocity > 0.5 m/s` AND no "近乎静止" in `motion_analysis`
4. **Use `relative_motion_analysis`**: vehicles labeled "快速接近" are high-priority collision candidates
5. **Modify ONLY future trajectory** (`t > 0`); past trajectory (`t ≤ 0`) is fixed
6. **Respect physical constraints**: dt=0.5s, |Δv| ≤ 2.5 m/s per step, |Δheading| ≤ 30° per step
7. **Output ONLY valid JSON** — no markdown, no extra commentary

Target risk level: **{risk_level}**
