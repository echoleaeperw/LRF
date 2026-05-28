"""
Ablation ii: Systematic weight search without LLM.

Reviewer concern: the LLM's contribution may be reducible to a more aggressive
risk-weighting strategy that a grid or Bayesian search could replicate.

This script exhaustively searches over key loss weights on a held-out validation
subset and saves the globally best static weight configuration to a JSON file.
The resulting `grid_search_best_weights.json` is loaded by adv_scenario_gen.py
when --ablation_mode=grid_search.

Usage:
    # Grid search (exhaustive):
    python src/ablation_weight_search.py \
        --search_method grid \
        --config ./configs/adv_gen_rule_based.cfg \
        --ckpt ./model_ckpt/traffic_model.pth \
        --n_scenes 30 \
        --out_dir ./ablation_results/weight_search

    # Bayesian search (requires optuna: pip install optuna):
    python src/ablation_weight_search.py \
        --search_method bayesian \
        --config ./configs/adv_gen_rule_based.cfg \
        --ckpt ./model_ckpt/traffic_model.pth \
        --n_trials 50 \
        --n_scenes 30 \
        --out_dir ./ablation_results/weight_search

    # Dry-run: preview configs without evaluation
    python src/ablation_weight_search.py --dry_run --search_method bayesian --n_trials 20

The primary metric optimised is long-tail coverage rate (fraction of scenes that
produce at least one classified long-tail scenario). Trajectory realism (mean
displacement of non-attacker vehicles from their initial GT-matched trajectories)
is used as a secondary constraint: only configs that keep realism degradation below
a threshold are considered.
"""

import os
import sys
import json
import copy
import shutil
import logging
import argparse
import itertools
import tempfile
from typing import Dict, List, Callable, Optional

import numpy as np

logger = logging.getLogger("ablation_weight_search")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Weight search space definition
# Each key maps to a list of candidate values to try.
# Ranges cover: default value, 2×, 4× and a "saturated" upper bound,
# following standard hyperparameter ablation practice.
# ---------------------------------------------------------------------------
SEARCH_SPACE: Dict[str, List[float]] = {
    "adv_crash":        [2.0, 4.0, 7.0, 10.0],
    "ttc":              [1.5, 3.0, 5.0, 6.0],
    "min_dist_lat":     [1.0, 2.0, 4.0, 5.0],
    "yaw_rate":         [0.8, 1.5, 2.5, 3.0],
    "motion_prior_atk": [0.005, 0.002, 0.001],
    "init_z_atk":       [0.05,  0.02,  0.01],
}

# Fixed weights not included in the search (held constant across all configs)
DEFAULT_FIXED_WEIGHTS: Dict[str, float] = {
    "coll_veh":              20.0,
    "coll_veh_plan":         20.0,
    "coll_env":              20.0,
    "motion_prior":           1.0,
    "init_z":                 0.5,
    "motion_prior_ext":       0.0001,
    "match_ext":             10.0,
    "yaw_rate_ego":           0.5,
    "yaw_rate_non_ego":       1.0,
    "init_match_ext":        10.0,
    "init_motion_prior_ext":  0.1,
}


def build_weight_config(combo: Dict[str, float]) -> Dict[str, float]:
    """Merge a search-space combo with the fixed defaults into a full weight dict."""
    cfg = copy.deepcopy(DEFAULT_FIXED_WEIGHTS)
    cfg.update(combo)
    return cfg


# ---------------------------------------------------------------------------
# Config generators (used for dry-run inspection and grid search)
# ---------------------------------------------------------------------------

def grid_configs() -> List[Dict[str, float]]:
    """Return the full Cartesian product of all search-space values."""
    keys = list(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[k] for k in keys]
    configs = []
    for combo_vals in itertools.product(*values):
        combo = dict(zip(keys, combo_vals))
        configs.append(build_weight_config(combo))
    logger.info(f"Grid search: {len(configs)} total weight configurations")
    return configs


def sample_bayesian_configs(n_trials: int = 50) -> List[Dict[str, float]]:
    """
    Sample n_trials configurations from the search space using Optuna's TPE sampler
    WITHOUT any real objective feedback — for DRY-RUN / INSPECTION ONLY.

    WARNING: This function does NOT perform true Bayesian optimisation. It merely
    samples configs using Optuna's internal prior. The actual sequential Bayesian
    optimisation (where each trial observes the real evaluation result and updates
    the acquisition model) is implemented inside run_weight_search() with method="bayesian".
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Optuna not installed. Run: pip install optuna")

    def _dummy_objective(trial: "optuna.Trial") -> float:
        # Dummy objective — used only to drive config sampling for dry-run.
        # Returns a random value so Optuna has something to report; the actual
        # Bayesian loop uses the real eval_fn inside run_weight_search().
        for k in SEARCH_SPACE:
            trial.suggest_categorical(k, SEARCH_SPACE[k])
        return 0.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(_dummy_objective, n_trials=n_trials, show_progress_bar=False)

    configs = []
    for trial in study.trials:
        combo = {k: trial.params[k] for k in SEARCH_SPACE}
        configs.append(build_weight_config(combo))
    logger.info(f"[dry-run] Sampled {len(configs)} Bayesian configurations (no evaluation)")
    return configs


# ---------------------------------------------------------------------------
# Incremental result persistence
# ---------------------------------------------------------------------------

def _flush_results(entry, path: str) -> None:
    """Append one result entry to a JSON-lines-style results file immediately.

    The file is kept as a JSON array on disk by reading, appending, and
    rewriting; this is safe for the typical trial counts here (≤ 200).
    """
    existing: List[Dict] = []
    if os.path.isfile(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    # entry may be a single dict (bayesian) or a list (grid, already handled)
    if isinstance(entry, list):
        existing = entry  # grid search replaces wholesale
    else:
        existing.append(entry)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


# ---------------------------------------------------------------------------
# Metric computation from saved scenario results
# ---------------------------------------------------------------------------

def _compute_metrics_from_output_dir(out_dir: str) -> Dict[str, float]:
    """
    Scan the scenario_results/ sub-directory produced by run_one_epoch (save=True)
    and compute:
      - longtail_coverage : fraction of saved scenes classified as 'longtail_condition'
      - realism_ade       : mean L2 displacement of non-attacker vehicles between
                            their initial (GT-matched) and final adversarial trajectories

    Directory structure expected:
        out_dir/
          scenario_results/
            longtail_condition/  ← scenes that triggered long-tail criteria
              scene_0001.json
              ...
            high_risk/
            low_risk/
    """
    scenario_root = os.path.join(out_dir, "scenario_results")
    if not os.path.isdir(scenario_root):
        logger.warning(f"  [metrics] scenario_results/ not found in {out_dir}")
        return {"longtail_coverage": 0.0, "realism_ade": float("inf")}

    all_json_files: List[str] = []
    longtail_json_files: List[str] = []

    for risk_dir in os.listdir(scenario_root):
        risk_path = os.path.join(scenario_root, risk_dir)
        if not os.path.isdir(risk_path):
            continue
        for fname in os.listdir(risk_path):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(risk_path, fname)
            all_json_files.append(fpath)
            if risk_dir == "longtail_condition":
                longtail_json_files.append(fpath)

    total = len(all_json_files)
    if total == 0:
        logger.warning(f"  [metrics] No scenario JSON files found in {scenario_root}")
        return {"longtail_coverage": 0.0, "realism_ade": float("inf")}

    longtail_coverage = len(longtail_json_files) / total

    # Compute realism ADE from saved scene dicts
    ade_values: List[float] = []
    for fpath in all_json_files:
        try:
            with open(fpath) as f:
                scene = json.load(f)
            # fut_init: initial GT-matched trajectories  [N, T, 2]
            # fut_adv:  final adversarial trajectories   [N, T, 2]
            fut_init = scene.get("fut_init")   # list[list[list[float]]]
            fut_adv  = scene.get("fut_adv")
            attack_agt = scene.get("attack_agt", -1)  # index of attacker vehicle

            if fut_init is None or fut_adv is None:
                continue

            init_arr = np.array(fut_init)   # [N, T, 2]
            adv_arr  = np.array(fut_adv)    # [N, T, 2]

            if init_arr.shape != adv_arr.shape or init_arr.ndim != 3:
                continue

            N = init_arr.shape[0]
            for n in range(N):
                if n == attack_agt:
                    continue  # skip attacker — we expect it to deviate
                disp = np.linalg.norm(adv_arr[n] - init_arr[n], axis=-1)  # [T]
                ade_values.append(float(np.mean(disp)))
        except Exception as e:
            logger.debug(f"  [metrics] Failed to parse {fpath}: {e}")
            continue

    realism_ade = float(np.mean(ade_values)) if ade_values else float("inf")

    logger.info(f"  [metrics] total={total}  longtail={len(longtail_json_files)}  "
                f"coverage={longtail_coverage:.3f}  ADE={realism_ade:.4f}")
    return {"longtail_coverage": longtail_coverage, "realism_ade": realism_ade}


# ---------------------------------------------------------------------------
# Real eval_fn factory
# ---------------------------------------------------------------------------

def make_eval_fn(cfg_path: str,
                 ckpt_path: str,
                 base_out_dir: str,
                 n_scenes: int) -> Callable[[Dict[str, float], int], Dict[str, float]]:
    """
    Build a real evaluation function that:
      1. Loads the model and dataset once (shared across all weight configs).
      2. For each weight config, runs run_one_epoch on n_scenes scenes in a
         temporary sub-directory.
      3. Parses the saved JSON output to compute longtail_coverage and realism_ade.

    Parameters
    ----------
    cfg_path     : path to the adv_gen .cfg file (same format used by adv_scenario_gen.py)
    ckpt_path    : path to the traffic model checkpoint (.pth)
    base_out_dir : parent directory; a unique sub-dir is created per weight config trial
    n_scenes     : number of scenes to evaluate per config (limits val_size)

    Returns
    -------
    eval_fn : callable(weight_config, n_scenes) → {"longtail_coverage": float,
                                                    "realism_ade": float}
    """
    import types
    import torch

    # Ensure src/ is on the import path (handles running from project root)
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    _root_dir = os.path.dirname(_src_dir)
    for _p in (_src_dir, _root_dir):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    # ── stub out optional langchain* dependencies ────────────────────────────
    # adv_scenario_gen imports longterm.agents.analysis which in turn imports
    # several langchain packages. These are only used in LLM code paths; we
    # inject lightweight stub modules so the file can be imported and its
    # parse_cfg / run_one_epoch functions used without the full LLM stack.
    def _make_stub(mod_name: str, **attrs):
        m = sys.modules.get(mod_name)
        if m is not None:
            return m  # already present (real or stub) — leave untouched
        m = types.ModuleType(mod_name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[mod_name] = m
        return m

    def _noop_class(name: str):
        return type(name, (), {"__init__": lambda s, *a, **kw: None})

    _lc_msgs = _make_stub("langchain_core.messages",
                          HumanMessage=_noop_class("HumanMessage"),
                          SystemMessage=_noop_class("SystemMessage"),
                          AIMessage=_noop_class("AIMessage"))
    _lc_cb   = _make_stub("langchain_core.callbacks",
                          StreamingStdOutCallbackHandler=_noop_class("StreamingStdOutCallbackHandler"))
    _lc_cbs  = _make_stub("langchain_core.callbacks.streaming_stdout",
                          StreamingStdOutCallbackHandler=_noop_class("StreamingStdOutCallbackHandler"))
    _lc_base = _make_stub("langchain.callbacks",
                          BaseCallbackHandler=_noop_class("BaseCallbackHandler"))
    _lc_bb   = _make_stub("langchain.callbacks.base",
                          BaseCallbackHandler=_noop_class("BaseCallbackHandler"))
    _lc_oai  = _make_stub("langchain_openai",
                          ChatOpenAI=_noop_class("ChatOpenAI"))
    _lc_ant  = _make_stub("langchain_anthropic",
                          ChatAnthropic=_noop_class("ChatAnthropic"))
    # Parent packages must expose child modules as attributes
    _make_stub("langchain_core", messages=_lc_msgs, callbacks=_lc_cb)
    _make_stub("langchain",      callbacks=_lc_base)

    from utils.common import mkdir
    from utils.torch import get_device, load_state
    from models.traffic_model import TrafficModel
    from datasets.nuscenes_dataset import NuScenesDataset
    from datasets.map_env import NuScenesMapEnv
    from torch_geometric.data import DataLoader as GraphDataLoader
    from utils.logger import Logger
    from utils.feasibility_reporter import FeasibilityReporter

    # Logger must be initialised before any model/dataset code that calls Logger.log
    os.makedirs(base_out_dir, exist_ok=True)
    Logger.init(os.path.join(base_out_dir, "weight_search.log"))

    # ── parse the full adv_scenario_gen config (all args, including --planner etc.) ──
    # We temporarily replace sys.argv so that adv_scenario_gen.parse_cfg() reads
    # the correct .cfg file and ckpt path, then restore sys.argv immediately.
    import adv_scenario_gen as _asg
    _saved_argv = sys.argv[:]
    sys.argv = [
        "adv_scenario_gen.py",
        "--config", cfg_path,
        "--ckpt",   ckpt_path,
        "--val_size", str(n_scenes),
        "--no_activate_stationary",   # disable side-effects not needed for eval
    ]
    try:
        cfg, _ = _asg.parse_cfg()
    finally:
        sys.argv = _saved_argv  # always restore, even on exception

    cfg.shuffle  = False
    cfg.use_llm  = False   # weight search never needs LLM; gate already closed by weight_manager=None

    device    = get_device()
    data_path = os.path.join(cfg.data_dir, cfg.data_version)

    map_env = NuScenesMapEnv(
        data_path,
        bounds=cfg.map_obs_bounds,
        L=cfg.map_obs_size_pix,
        W=cfg.map_obs_size_pix,
        layers=cfg.map_layers,
        device=device,
        load_lanegraph=(cfg.planner == "hardcode"),
        lanegraph_res_meters=1.0,
    )

    # NuScenesDataset only supports val_size ∈ {200, 400} with randomize_val=True.
    # We always initialise with the minimum supported size (200), then restrict
    # evaluation to the first n_scenes samples via a DataLoader Subset.
    _supported_val_sizes = [200, 400]
    _dataset_val_size = next(
        (s for s in _supported_val_sizes if s >= n_scenes), _supported_val_sizes[0]
    )
    dataset = NuScenesDataset(
        data_path, map_env,
        version=cfg.data_version,
        split=cfg.split,
        categories=cfg.agent_types,
        npast=cfg.past_len,
        nfuture=cfg.future_len,
        seq_interval=cfg.seq_interval,
        randomize_val=True,
        val_size=_dataset_val_size,
        reduce_cats=cfg.reduce_cats,
    )
    logger.info(f"[make_eval_fn] Dataset loaded: {len(dataset)} scenes "
                f"(val_size={_dataset_val_size}); each trial will use first {n_scenes}")

    # torch.utils.data.Subset does not forward arbitrary attribute access to the
    # underlying dataset (e.g. .dt, .vec2cat used by run_one_epoch).
    # _DatasetHead limits __len__ / __getitem__ while proxying all other attrs.
    class _DatasetHead:
        def __init__(self, ds, n_limit):
            object.__setattr__(self, '_ds', ds)
            object.__setattr__(self, '_n', min(n_limit, len(ds)))
        def __len__(self):
            return object.__getattribute__(self, '_n')
        def __getitem__(self, idx):
            if idx >= object.__getattribute__(self, '_n'):
                raise IndexError(idx)
            return object.__getattribute__(self, '_ds')[idx]
        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, '_ds'), name)

    eval_subset = _DatasetHead(dataset, n_scenes)
    loader = GraphDataLoader(eval_subset, batch_size=1, shuffle=False)

    model = TrafficModel(
        cfg.past_len, cfg.future_len, cfg.map_obs_size_pix,
        len(dataset.categories),
        map_feat_size=cfg.map_feat_size,
        past_feat_size=cfg.past_feat_size,
        future_feat_size=cfg.future_feat_size,
        latent_size=cfg.latent_size,
        output_bicycle=cfg.model_output_bicycle,
        conv_channel_in=map_env.num_layers,
        conv_kernel_list=cfg.conv_kernel_list,
        conv_stride_list=cfg.conv_stride_list,
        conv_filter_list=cfg.conv_filter_list,
    ).to(device)
    # load_state(path, model) — matches utils/torch.py signature
    load_state(ckpt_path, model, map_location=device)
    model.set_normalizer(dataset.get_state_normalizer())
    model.set_att_normalizer(dataset.get_att_normalizer())
    if cfg.model_output_bicycle:
        from datasets.utils import NUSC_BIKE_PARAMS
        model.set_bicycle_params(NUSC_BIKE_PARAMS)
    model.eval()
    logger.info(f"[make_eval_fn] Model loaded from {ckpt_path}, dataset size={len(dataset)}")

    trial_counter = [0]

    def eval_fn(weight_config: Dict[str, float], _n_scenes: int) -> Dict[str, float]:
        trial_idx = trial_counter[0]
        trial_counter[0] += 1

        trial_out = os.path.join(base_out_dir, f"trial_{trial_idx:04d}")
        mkdir(trial_out)
        # Re-init Logger for this trial so run_one_epoch writes to the trial log file
        Logger.init(os.path.join(trial_out, "eval_log.txt"))

        # ── save the weights used for this trial immediately ──────────────────
        # (so we can track weight↔metrics even if the run crashes mid-way)
        search_keys = list(SEARCH_SPACE.keys())
        with open(os.path.join(trial_out, "weights.json"), "w") as _wf:
            json.dump({
                "trial": trial_idx,
                "search_weights": {k: weight_config.get(k) for k in search_keys},
                "full_weights":   weight_config,
            }, _wf, indent=2)

        model.train()  # run_one_epoch expects model.train() mode internally
        feasibility_reporter = FeasibilityReporter()
        _asg.run_one_epoch(
            loader, cfg.batch_size, model, map_env, device,
            trial_out, weight_config,
            planner_name=cfg.planner,
            planner_cfg=cfg.planner_cfg,
            feasibility_thresh=cfg.feasibility_thresh,
            feasibility_time=cfg.feasibility_time,
            feasibility_vel=cfg.feasibility_vel,
            feasibility_infront_min=cfg.feasibility_infront_min,
            feasibility_check_sep=cfg.feasibility_check_sep,
            num_iters=cfg.num_iters,
            lr=cfg.lr,
            viz=False,
            save=True,
            adv_attack_with=getattr(cfg, "adv_attack_with", None),
            weight_manager=None,   # no LLM — pure weight search baseline
            config=cfg,
            ablation_mode="grid_search",
            feasibility_reporter=feasibility_reporter,
        )

        metrics = _compute_metrics_from_output_dir(trial_out)

        # ── save metrics alongside weights immediately after evaluation ───────
        with open(os.path.join(trial_out, "metrics.json"), "w") as _mf:
            json.dump({"trial": trial_idx, **metrics}, _mf, indent=2)

        return metrics

    return eval_fn


# ---------------------------------------------------------------------------
# Core search loop
# ---------------------------------------------------------------------------

def run_weight_search(eval_fn: Callable,
                      method: str = "grid",
                      n_scenes: int = 30,
                      n_trials: int = 50,
                      realism_budget: float = 0.15,
                      out_dir: str = "./ablation_results/weight_search") -> Dict[str, float]:
    """
    Main search loop.

    For method="grid"    : evaluates the full Cartesian product of SEARCH_SPACE.
    For method="bayesian": uses Optuna's TPE sampler in a true sequential loop
                           where each trial's result is observed before the next
                           config is proposed — this is the correct Bayesian
                           optimisation protocol.

    Parameters
    ----------
    eval_fn        : callable(weight_dict, n_scenes) → {"longtail_coverage": float,
                                                         "realism_ade": float}
    method         : "grid" | "bayesian"
    n_scenes       : scenes evaluated per weight config
    n_trials       : Bayesian-only — total number of Optuna trials
    realism_budget : maximum allowed relative increase in realism ADE vs. default
    out_dir        : directory to save result JSON files

    Returns
    -------
    Best weight configuration dict (search keys + fixed keys combined)
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── reference baseline (default search-space values) ─────────────────────
    default_combo   = {k: SEARCH_SPACE[k][0] for k in SEARCH_SPACE}
    default_config  = build_weight_config(default_combo)
    logger.info("Evaluating default weight config as reference baseline …")
    default_result  = eval_fn(default_config, n_scenes)
    default_ade     = default_result.get("realism_ade", 1.0)
    default_cov     = default_result.get("longtail_coverage", 0.0)
    logger.info(f"Default config: coverage={default_cov:.3f}, ADE={default_ade:.4f}")

    all_results: List[Dict] = []
    best_coverage = default_cov
    best_config   = copy.deepcopy(default_config)

    def _realism_ok(ade: float) -> bool:
        return (ade - default_ade) / max(default_ade, 1e-6) <= realism_budget

    # ── grid search ──────────────────────────────────────────────────────────
    if method == "grid":
        configs = grid_configs()
        for idx, cfg in enumerate(configs):
            logger.info(
                f"[{idx+1}/{len(configs)}] "
                + "  ".join(f"{k}={cfg[k]}" for k in SEARCH_SPACE)
            )
            result   = eval_fn(cfg, n_scenes)
            coverage = result.get("longtail_coverage", 0.0)
            ade      = result.get("realism_ade", 1.0)
            ok       = _realism_ok(ade)
            logger.info(f"  coverage={coverage:.3f}  ADE={ade:.4f}  realism_ok={ok}")
            entry = {"weights": {k: cfg[k] for k in SEARCH_SPACE},
                     "coverage": coverage, "realism_ade": ade, "realism_ok": ok}
            all_results.append(entry)
            # flush the whole list so partial results survive a crash
            _flush_results(all_results, os.path.join(out_dir, "all_weight_search_results.json"))
            if ok and coverage > best_coverage:
                best_coverage = coverage
                best_config   = copy.deepcopy(cfg)

    # ── true sequential Bayesian optimisation ────────────────────────────────
    elif method == "bayesian":
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Optuna not installed. Run: pip install optuna")

        # The objective MUST call eval_fn internally so that Optuna's TPE
        # sampler can observe each trial's true result and update its
        # acquisition model before proposing the next configuration.
        def _objective(trial: "optuna.Trial") -> float:
            combo = {k: trial.suggest_categorical(k, SEARCH_SPACE[k])
                     for k in SEARCH_SPACE}
            cfg     = build_weight_config(combo)
            logger.info(
                f"[Trial {trial.number+1}/{n_trials}] "
                + "  ".join(f"{k}={cfg[k]}" for k in SEARCH_SPACE)
            )

            result   = eval_fn(cfg, n_scenes)
            coverage = result.get("longtail_coverage", 0.0)
            ade      = result.get("realism_ade", 1.0)
            ok       = _realism_ok(ade)

            # Store extra attributes so we can retrieve them after optimisation
            trial.set_user_attr("coverage",    coverage)
            trial.set_user_attr("realism_ade", ade)
            trial.set_user_attr("realism_ok",  ok)

            logger.info(f"  coverage={coverage:.3f}  ADE={ade:.4f}  realism_ok={ok}")
            entry = {"weights": {k: cfg[k] for k in SEARCH_SPACE},
                     "coverage": coverage, "realism_ade": ade, "realism_ok": ok}
            all_results.append(entry)
            # flush incrementally so partial results survive a crash
            _flush_results(entry, os.path.join(out_dir, "all_weight_search_results.json"))

            # Return 0 for realism-violating configs so Optuna penalises them
            # and learns to stay in the feasible region.
            return coverage if ok else 0.0

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

        # Pick best among realism-feasible trials
        feasible_trials = [
            t for t in study.trials if t.user_attrs.get("realism_ok", False)
        ]
        if feasible_trials:
            best_trial    = max(feasible_trials,
                                key=lambda t: t.user_attrs["coverage"])
            best_coverage = best_trial.user_attrs["coverage"]
            best_config   = build_weight_config(
                {k: best_trial.params[k] for k in SEARCH_SPACE}
            )
            logger.info(f"Bayesian best trial #{best_trial.number}: "
                        f"coverage={best_coverage:.3f}")
        else:
            logger.warning("No realism-feasible Bayesian trial found; "
                           "returning default config.")

    else:
        raise ValueError(f"Unknown search method: {method!r}")

    # ── persist results ───────────────────────────────────────────────────────
    results_path = os.path.join(out_dir, "all_weight_search_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"All results saved to {results_path}")

    best_path            = os.path.join(out_dir, "grid_search_best_weights.json")
    search_weights_only  = {k: best_config[k] for k in SEARCH_SPACE}
    with open(best_path, "w") as f:
        json.dump(search_weights_only, f, indent=2)
    logger.info(f"Best weights (coverage={best_coverage:.3f}) saved to {best_path}")
    logger.info(f"Best config: {search_weights_only}")

    _print_summary(all_results, default_result, best_config, best_coverage)
    return best_config


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(results: List[Dict], default_result: Dict,
                   best_config: Dict, best_coverage: float) -> None:
    logger.info("\n" + "=" * 70)
    logger.info("WEIGHT SEARCH SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total configs evaluated : {len(results)}")
    coverages = [r["coverage"] for r in results]
    if coverages:
        logger.info(f"Coverage range          : [{min(coverages):.3f}, {max(coverages):.3f}]")
    logger.info(f"Default coverage        : {default_result['longtail_coverage']:.3f}")
    logger.info(f"Best coverage (feasible): {best_coverage:.3f}")
    logger.info(f"Best weights found      : { {k: best_config[k] for k in SEARCH_SPACE} }")
    logger.info(
        "\nNote: Compare best_coverage here against full-system (LLM) coverage.\n"
        "If LLM > best_coverage, the gain cannot be attributed to weight magnitude alone."
    )
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation ii: systematic weight search baseline"
    )
    parser.add_argument("--search_method", type=str, default="grid",
                        choices=["grid", "bayesian"],
                        help="Search algorithm: 'grid' for exhaustive search, "
                             "'bayesian' for Optuna TPE sequential search.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to adv_gen .cfg file (required unless --dry_run).")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to traffic model checkpoint .pth (required unless --dry_run).")
    parser.add_argument("--n_scenes", type=int, default=30,
                        help="Number of scenes to evaluate each weight config on.")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Bayesian trials (only used for --search_method bayesian).")
    parser.add_argument("--realism_budget", type=float, default=0.15,
                        help="Max allowed relative increase in realism ADE vs. default config.")
    parser.add_argument("--out_dir", type=str,
                        default="./ablation_results/weight_search",
                        help="Output directory for result JSON files.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print all configs that would be evaluated, then exit.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── dry-run: preview configs without touching the model ──────────────────
    if args.dry_run:
        if args.search_method == "grid":
            configs = grid_configs()
        else:
            try:
                configs = sample_bayesian_configs(args.n_trials)
            except ImportError as e:
                logger.warning(f"{e}  — falling back to grid configs for dry-run preview.")
                configs = grid_configs()
        logger.info(f"Dry run ({args.search_method}): {len(configs)} configs would be evaluated.")
        logger.info("First 5 configs:")
        for c in configs[:5]:
            logger.info({k: c[k] for k in SEARCH_SPACE})
        sys.exit(0)

    # ── production run: requires --config and --ckpt ─────────────────────────
    if args.config is None or args.ckpt is None:
        logger.error("--config and --ckpt are required for a real evaluation run.")
        logger.error("Use --dry_run to preview configs without evaluation.")
        sys.exit(1)

    if not os.path.isfile(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    if not os.path.isfile(args.ckpt):
        logger.error(f"Checkpoint file not found: {args.ckpt}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    trials_dir = os.path.join(args.out_dir, "trials")
    os.makedirs(trials_dir, exist_ok=True)

    logger.info(f"Building eval_fn from config={args.config}, ckpt={args.ckpt} …")
    eval_fn = make_eval_fn(
        cfg_path=args.config,
        ckpt_path=args.ckpt,
        base_out_dir=trials_dir,
        n_scenes=args.n_scenes,
    )

    logger.info(f"Starting {args.search_method} weight search "
                f"(n_scenes={args.n_scenes}, n_trials={args.n_trials})")
    best = run_weight_search(
        eval_fn=eval_fn,
        method=args.search_method,
        n_scenes=args.n_scenes,
        n_trials=args.n_trials,
        realism_budget=args.realism_budget,
        out_dir=args.out_dir,
    )
    logger.info("Weight search complete.")
    logger.info(f"Best weights: {best}")
