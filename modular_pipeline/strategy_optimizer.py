import random
import itertools
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical  # ✅ Add Categorical
from typing import List, Dict, Callable, Optional


# Dummy scoring function (replace with actual logic)
def evaluate_strategy(params: Dict) -> float:
    return 1.0 - (0.5 * params["faiss_k"] + 0.3 * params["bm25_k"]) / 100


def grid_search(param_grid: Dict[str, List], score_fn: Callable[[Dict], Dict], log_fn: Optional[Callable] = None) -> List[Dict]:
    keys, values = zip(*param_grid.items())
    results = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        result = score_fn(params)
        if log_fn:
            log_fn(f"🔎 Grid: {params} → {result['score']:.4f}")
        results.append({"strategy_id": str(params), **result, **params})
    return sorted(results, key=lambda x: x["score"], reverse=True)


def random_search(param_grid: Dict[str, List], score_fn: Callable[[Dict], Dict], n_iter: int = 10, log_fn: Optional[Callable] = None) -> List[Dict]:
    keys = list(param_grid.keys())
    results = []
    for i in range(n_iter):
        params = {k: random.choice(param_grid[k]) for k in keys}
        result = score_fn(params)
        if log_fn:
            log_fn(f"🎲 Random {i+1}/{n_iter}: {params} → {result['score']:.4f}")
        results.append({"strategy_id": str(params), **result, **params})
    return sorted(results, key=lambda x: x["score"], reverse=True)


def bayesian_search(
    search_space: Dict[str, List],
    score_fn: Callable[[Dict, Optional[int], Optional[int]], Dict],
    n_calls: int = 15,
    random_state: int = 42,
    log_fn: Optional[Callable] = None
) -> List[Dict]:
    space = []
    param_keys = []

    # Build skopt-compatible space
    for k, v in search_space.items():
        param_keys.append(k)
        if all(isinstance(i, int) for i in v):
            space.append(Integer(min(v), max(v), name=k))
        elif all(isinstance(i, float) for i in v):
            space.append(Real(min(v), max(v), name=k))
        else:
            v_str = [",".join(map(str, val)) if isinstance(val, tuple) else val for val in v]
            space.append(Categorical(v_str, name=k))

    # Parse stringified weights back into tuples
    def parse_weights(val):
        if isinstance(val, str) and "," in val:
            return tuple(map(float, val.split(",")))
        return val

    # === Track evaluations manually
    eval_history = []

    def objective(x):
        idx = len(eval_history) + 1
        params = dict(zip(param_keys, x))
        if "weights" in params:
            params["weights"] = parse_weights(params["weights"])

        # ✅ Pass strategy index and total into score_fn
        result = score_fn(params, idx, n_calls)
        eval_history.append(result)

        if log_fn:
            log_fn(f"📈 Bayesian: {params} → {result['score']:.4f}")
        return -result["score"]

    # === Run the optimizer
    res = gp_minimize(objective, space, n_calls=n_calls, random_state=random_state)

    # === Reconstruct full results list
    results = []
    for i, x in enumerate(res.x_iters):
        params = dict(zip(param_keys, x))
        if "weights" in params:
            params["weights"] = parse_weights(params["weights"])
        score = -res.func_vals[i]
        results.append({"strategy_id": str(params), "score": score, **params})

    return sorted(results, key=lambda x: x["score"], reverse=True)
