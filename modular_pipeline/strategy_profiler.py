# strategy_profiler.py
import numpy as np
import random
from typing import Dict, List, Callable

# Thresholds: tune if you like
FLAT_SCORE_STD_MIN = 1e-3       # if std < this, objective likely flat
FLAT_SCORE_SPREAD_MIN = 1e-2    # if max-min < this, objective likely flat

def random_sample(param_grid: Dict[str, List], n: int = 15, seed: int = 42) -> List[Dict]:
    """Return n random parameter dicts from a param_grid (lists per key)."""
    rnd = random.Random(seed)
    keys = list(param_grid.keys())
    samples = []
    for _ in range(n):
        params = {k: rnd.choice(param_grid[k]) for k in keys}
        samples.append(params)
    return samples

def profile_objective(
    param_grid: Dict[str, List],
    score_fn: Callable[[Dict], Dict],
    n: int = 12
) -> Dict:
    """
    Quickly probe objective by sampling param settings.

    score_fn(params) must return a dict with at least:
      - 'score' (float)
      - optionally 'entropy', 'coverage', 'gini', 'max_freq'
    """
    samples = random_sample(param_grid, n=n)
    rows = [score_fn(p) | p for p in samples]

    # Gather numeric arrays
    metric_names = ["score", "entropy", "coverage", "gini", "max_freq"]
    metrics = {}
    for m in metric_names:
        vals = []
        for r in rows:
            v = r.get(m, None)
            try:
                if v is not None:
                    vals.append(float(v))
            except Exception:
                pass
        if vals:
            metrics[m] = np.array(vals, dtype=float)

    stats = {}
    for k, arr in metrics.items():
        if arr.size:
            stats[k] = {
                "std": float(np.std(arr)),
                "spread": float(np.max(arr) - np.min(arr)),
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n": int(arr.size),
            }

    score_flat = (
        stats.get("score", {}).get("std", 0.0) < FLAT_SCORE_STD_MIN
        or stats.get("score", {}).get("spread", 0.0) < FLAT_SCORE_SPREAD_MIN
    )

    return {
        "rows": rows,        # sampled results (for debugging/inspection)
        "stats": stats,      # variance summary per metric
        "score_flat": score_flat,
    }

def _normalize(d: Dict[str, float]) -> Dict[str, float]:
    total = sum(d.values()) or 1.0
    return {k: (v / total) for k, v in d.items()}

def suggest_weights_from_variance(
    stats: Dict,
    metric_keys=("entropy", "coverage", "gini", "max_freq"),
    priors: Dict[str, float] | None = None
) -> Dict[str, float]:
    """
    Build weights emphasizing metrics with variance (signal).
    Optionally blend with priors (e.g., give 'coverage' slight preference).
    """
    priors = priors or {"coverage": 1.2, "entropy": 1.0, "gini": 1.0, "max_freq": 1.0}
    stds = {k: stats.get(k, {}).get("std", 0.0) for k in metric_keys}
    # Normalize stds to weights, then blend with priors and normalize again
    w_var = _normalize(stds)
    w_blend = {k: w_var.get(k, 0.0) * priors.get(k, 1.0) for k in metric_keys}
    return _normalize(w_blend)

def band_penalty(x: float, lo: float, hi: float) -> float:
    """Return 0 if x in [lo,hi], otherwise a negative penalty proportional to distance."""
    if x < lo:
        return -(lo - x)
    if x > hi:
        return -(x - hi)
    return 0.0

def dynamic_objective(result: Dict, weights: Dict[str, float]) -> float:
    """
    Data-driven single-objective: higher is better.
    - coverage: higher better
    - entropy: lower absolute better (or swap to banded below)
    - gini, max_freq: lower better (diversity)
    """
    entropy  = float(result.get("entropy", 0.0))
    coverage = float(result.get("coverage", 0.0))
    gini     = float(result.get("gini", 0.0))
    max_freq = float(result.get("max_freq", 0.0))

    components = {
        "coverage":  coverage,
        "entropy":  -abs(entropy),
        "gini":     -gini,
        "max_freq": -max_freq,
    }
    return sum(weights.get(k, 0.0) * components[k] for k in components)

def dynamic_objective_banded(result: Dict, weights: Dict[str, float], entropy_band: tuple[float, float]) -> float:
    """
    Variant where you *target a band* for entropy. Inside band is best (0 penalty).
    """
    entropy  = float(result.get("entropy", 0.0))
    coverage = float(result.get("coverage", 0.0))
    gini     = float(result.get("gini", 0.0))
    max_freq = float(result.get("max_freq", 0.0))

    e_term = band_penalty(entropy, entropy_band[0], entropy_band[1])  # 0 is best
    components = {
        "coverage":  coverage,
        "entropy":   e_term,   # closer to band → closer to 0 (better)
        "gini":     -gini,
        "max_freq": -max_freq,
    }
    return sum(weights.get(k, 0.0) * components[k] for k in components)
