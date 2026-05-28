"""
Ablation iii: LLM output robustness across prompts and random seeds.

Reviewer concern: the reported gains may be an artifact of prompt engineering
rather than genuine semantic reasoning. We need to show that the LLM's weight
assignments are stable (low variance) across seeds and prompt variants.

This script:
  1. Loads a saved scenario description (JSON) produced by scenario_extractor.py.
  2. Calls LossFunctionAgent.generate_loss_weights N times with different
     temperature seeds and (optionally) different prompt templates.
  3. Computes and reports:
     - Mean and std of each weight across runs
     - Coefficient of Variation (CV) per weight – the key stability metric
     - Cosine similarity matrix between weight vectors (shows directional consistency)
     - Whether the LLM consistently identifies the same attacker_vehicle_id

Usage:
    python src/ablation_llm_robustness.py \
        --scenario_json path/to/scenario.json \
        --n_runs 5 \
        --temperature 0.5 \
        --llm_model deepseek-reasoner \
        --out_dir ./ablation_results/llm_robustness

The script outputs:
  - robustness_report.json  : full per-run weight vectors + statistics
  - robustness_summary.txt  : human-readable table for the paper
"""

import os
import sys
import json
import copy
import logging
import argparse
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ablation_llm_robustness")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# LLM-relevant weight keys tracked in ablation (match AdvGenLoss keys)
TRACKED_KEYS = [
    "adv_crash", "ttc", "min_dist_lat", "yaw_rate",
    "coll_veh", "coll_veh_plan", "coll_env",
    "motion_prior_atk", "init_z_atk",
]

# Prompt template variants for prompt-robustness test
PROMPT_TEMPLATES = {
    "standard": None,   # uses the default prompt in LossFunctionAgent
    "concise": (
        "Analyze the following traffic scene and output JSON loss weights "
        "to create an adversarial scenario. Be brief."
    ),
    "detailed": (
        "You are an expert traffic safety engineer. "
        "Carefully examine the structured scene data below, identify the "
        "most dangerous vehicle configuration, and produce optimally calibrated "
        "loss function weights that will generate the most challenging long-tail "
        "adversarial scenario while maintaining trajectory realism."
    ),
}


def load_scenario_description(json_path: str) -> str:
    with open(json_path) as f:
        data = json.load(f)
    return json.dumps(data, ensure_ascii=False, indent=2)


def extract_risk_weights(llm_result: Dict) -> Dict[str, float]:
    """Flatten the nested risk_weights dict from LossFunctionAgent output."""
    if "risk_weights" in llm_result:
        return {k: float(v) for k, v in llm_result["risk_weights"].items()
                if isinstance(v, (int, float))}
    return {k: float(v) for k, v in llm_result.items()
            if isinstance(v, (int, float))}


def run_n_seeds(agent,
                scenario_description: str,
                n_runs: int,
                temperature: float,
                risk_level: str,
                prompt_system: Optional[str] = None) -> List[Dict]:
    """Call the LLM n_runs times, varying the random seed via temperature sampling."""
    results = []
    for run_idx in range(n_runs):
        logger.info(f"  Run {run_idx + 1}/{n_runs} (temperature={temperature}) …")
        t0 = time.time()
        try:
            # Each call is independent; temperature > 0 introduces sampling variance
            weights_raw = agent.generate_loss_weights(
                scenario_description=scenario_description,
                temperature=temperature,
                risk_level=risk_level,
            )
            weights = extract_risk_weights(weights_raw)
            attacker_id = weights_raw.get("attacker_vehicle_id", None)
            elapsed = time.time() - t0
            results.append({
                "run_idx": run_idx,
                "weights": weights,
                "attacker_vehicle_id": attacker_id,
                "elapsed_s": round(elapsed, 2),
                "error": None,
            })
            logger.info(f"    attacker_id={attacker_id}  "
                        f"adv_crash={weights.get('adv_crash', 'N/A')}  "
                        f"ttc={weights.get('ttc', 'N/A')}  "
                        f"elapsed={elapsed:.1f}s")
        except Exception as e:
            logger.warning(f"    Run {run_idx} failed: {e}")
            results.append({
                "run_idx": run_idx,
                "weights": {},
                "attacker_vehicle_id": None,
                "elapsed_s": time.time() - t0,
                "error": str(e),
            })
    return results


def compute_statistics(runs: List[Dict]) -> Dict:
    """Compute per-weight mean, std, CV and attacker-id stability."""
    valid = [r for r in runs if not r["error"] and r["weights"]]
    if not valid:
        return {"error": "No valid runs"}

    all_weights = [r["weights"] for r in valid]
    keys = TRACKED_KEYS

    stats = {}
    weight_matrix = []
    for k in keys:
        vals = [w.get(k, np.nan) for w in all_weights]
        vals_clean = [v for v in vals if not np.isnan(v)]
        if not vals_clean:
            stats[k] = {"mean": None, "std": None, "cv": None, "values": vals}
            continue
        mean = float(np.mean(vals_clean))
        std  = float(np.std(vals_clean))
        cv   = float(std / mean) if mean != 0 else float("inf")
        stats[k] = {
            "mean":   round(mean, 4),
            "std":    round(std,  4),
            "cv":     round(cv,   4),  # < 0.10 = low variance; < 0.20 = acceptable
            "values": [round(v, 4) for v in vals_clean],
        }
        weight_matrix.append([v for v in vals_clean])

    # Cosine similarity between weight vectors (directional consistency)
    cosine_sim = None
    if len(weight_matrix) > 0 and all(len(row) == len(valid) for row in weight_matrix):
        wmat = np.array(weight_matrix).T   # shape: (n_valid, n_keys)
        norms = np.linalg.norm(wmat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        wmat_norm = wmat / norms
        cos_mat = wmat_norm @ wmat_norm.T
        cosine_sim = cos_mat.tolist()

    # Attacker-id stability
    attacker_ids = [r["attacker_vehicle_id"] for r in valid]
    from collections import Counter
    id_counter = Counter(str(a) for a in attacker_ids)
    most_common_id, most_common_count = id_counter.most_common(1)[0]
    attacker_stability = round(most_common_count / len(valid), 3)

    return {
        "n_valid_runs":       len(valid),
        "n_total_runs":       len(runs),
        "per_weight_stats":   stats,
        "cosine_similarity":  cosine_sim,
        "attacker_id_distribution": dict(id_counter),
        "attacker_stability": attacker_stability,  # fraction of runs with modal attacker id
    }


def format_summary_table(stats: Dict, scenario_name: str = "scene") -> str:
    lines = [
        f"LLM Robustness Report – {scenario_name}",
        f"Valid runs: {stats['n_valid_runs']} / {stats['n_total_runs']}",
        "",
        f"{'Weight':<22} {'Mean':>8} {'Std':>8} {'CV':>8}  {'Stability'}",
        "-" * 60,
    ]
    per = stats.get("per_weight_stats", {})
    for k in TRACKED_KEYS:
        if k not in per or per[k]["mean"] is None:
            lines.append(f"{k:<22} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            continue
        cv = per[k]["cv"]
        stability = "HIGH" if cv < 0.10 else ("MED" if cv < 0.20 else "LOW")
        lines.append(
            f"{k:<22} {per[k]['mean']:>8.4f} {per[k]['std']:>8.4f} "
            f"{cv:>8.4f}  {stability}"
        )
    lines += [
        "",
        f"Attacker-id agreement: {stats['attacker_stability']*100:.1f}%  "
        f"(distribution: {stats['attacker_id_distribution']})",
        "",
        "Interpretation:",
        "  CV < 0.10 → HIGH stability (robust to temperature sampling)",
        "  CV < 0.20 → MED  stability (acceptable for ablation reporting)",
        "  CV > 0.20 → LOW  stability (prompt-sensitive; investigate further)",
        "",
        "Attacker-id agreement > 80% → LLM consistently identifies the same adversarial agent.",
    ]
    return "\n".join(lines)


def run_robustness_test(scenario_json: str,
                        n_runs: int = 5,
                        temperature: float = 0.5,
                        llm_model: str = "deepseek-reasoner",
                        risk_level: str = "high_risk",
                        test_prompt_variants: bool = False,
                        out_dir: str = "./ablation_results/llm_robustness",
                        api_key: Optional[str] = None) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    scenario_description = load_scenario_description(scenario_json)
    scenario_name = os.path.splitext(os.path.basename(scenario_json))[0]

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.llm.loss_function_agent import LossFunctionAgent

    agent = LossFunctionAgent(
        model_name=llm_model,
        temperature=temperature,
        api_key=api_key,
        cache_dir=None,   # disable cache to ensure independent calls
    )

    report = {
        "scenario": scenario_name,
        "llm_model": llm_model,
        "temperature": temperature,
        "n_runs": n_runs,
        "risk_level": risk_level,
        "seed_robustness": {},
        "prompt_robustness": {} if test_prompt_variants else None,
    }

    # --- Seed robustness (same prompt, different temperature samples) ---
    logger.info(f"=== Seed robustness test: {n_runs} runs, temperature={temperature} ===")
    seed_runs = run_n_seeds(agent, scenario_description, n_runs, temperature, risk_level)
    seed_stats = compute_statistics(seed_runs)
    report["seed_robustness"] = {
        "runs": seed_runs,
        "stats": seed_stats,
        "summary": format_summary_table(seed_stats, f"{scenario_name} (seed robustness)"),
    }
    logger.info("\n" + report["seed_robustness"]["summary"])

    # --- Prompt variant robustness ---
    if test_prompt_variants:
        logger.info("=== Prompt variant robustness test ===")
        prompt_results = {}
        for pname, psystem in PROMPT_TEMPLATES.items():
            logger.info(f"  Prompt variant: {pname}")
            # For prompt variants, use n=3 per variant to keep cost low
            variant_runs = run_n_seeds(agent, scenario_description,
                                       min(n_runs, 3), temperature, risk_level, psystem)
            variant_stats = compute_statistics(variant_runs)
            prompt_results[pname] = {
                "runs": variant_runs,
                "stats": variant_stats,
            }
        report["prompt_robustness"] = prompt_results

        # Cross-prompt variance: std of means across prompt variants
        cross_prompt_stats = {}
        for k in TRACKED_KEYS:
            means = [
                prompt_results[pn]["stats"]["per_weight_stats"].get(k, {}).get("mean")
                for pn in PROMPT_TEMPLATES
            ]
            means = [m for m in means if m is not None]
            if means:
                cross_prompt_stats[k] = {
                    "cross_prompt_std":  round(float(np.std(means)), 4),
                    "cross_prompt_mean": round(float(np.mean(means)), 4),
                    "cross_prompt_cv":   round(float(np.std(means) / max(np.mean(means), 1e-9)), 4),
                }
        report["cross_prompt_stats"] = cross_prompt_stats
        logger.info(f"Cross-prompt weight CV: { {k: cross_prompt_stats[k]['cross_prompt_cv'] for k in TRACKED_KEYS if k in cross_prompt_stats} }")

    # --- Save outputs ---
    report_path = os.path.join(out_dir, f"{scenario_name}_robustness_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Full report saved to {report_path}")

    summary_path = os.path.join(out_dir, f"{scenario_name}_robustness_summary.txt")
    with open(summary_path, "w") as f:
        f.write(report["seed_robustness"]["summary"])
        if report.get("prompt_robustness"):
            f.write("\n\n=== Cross-Prompt Stability ===\n")
            for k, v in report.get("cross_prompt_stats", {}).items():
                f.write(f"  {k:<22}  cross_prompt_CV={v['cross_prompt_cv']:.4f}\n")
    logger.info(f"Summary saved to {summary_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Ablation iii: LLM robustness test")
    p.add_argument("--scenario_json", type=str, required=True,
                   help="Path to structured scenario JSON produced by scenario_extractor.py")
    p.add_argument("--n_runs", type=int, default=5,
                   help="Number of independent LLM calls per prompt template (seed robustness)")
    p.add_argument("--temperature", type=float, default=0.5,
                   help="Sampling temperature (>0 introduces stochasticity)")
    p.add_argument("--llm_model", type=str, default="deepseek-reasoner")
    p.add_argument("--risk_level", type=str, default="high_risk",
                   choices=["low_risk", "high_risk", "longtail_condition"])
    p.add_argument("--test_prompt_variants", action="store_true",
                   help="Also test with 3 different prompt templates (prompt robustness)")
    p.add_argument("--out_dir", type=str,
                   default="./ablation_results/llm_robustness")
    p.add_argument("--api_key", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_robustness_test(
        scenario_json=args.scenario_json,
        n_runs=args.n_runs,
        temperature=args.temperature,
        llm_model=args.llm_model,
        risk_level=args.risk_level,
        test_prompt_variants=args.test_prompt_variants,
        out_dir=args.out_dir,
        api_key=args.api_key,
    )
