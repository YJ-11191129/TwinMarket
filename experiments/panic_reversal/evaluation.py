"""Delayed-entry conditional targets, not an executable portfolio backtest."""
from collections import defaultdict

import numpy as np


def ranks(x):
    """Average ranks, including ties; no SciPy dependency."""
    x = np.asarray(x)
    order = np.argsort(x, kind="stable")
    out = np.empty(len(x), dtype=float)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        out[order[start:end]] = (start + end - 1) / 2
        start = end
    return out


def conditional_targets(result):
    """Signal at close t; enter close t+1; exit close t+1+h.

    Positions are research cohorts with overlapping holding periods. The same
    close-price convention applies to all mechanisms and controls. Short sales
    are theoretical; the long-only cohort is reported separately for A shares.
    """
    c, prices = result.config, result.prices
    rows = []
    for state in result.market:
        t = state["step"]
        if t + 1 < c.lookback:
            continue
        past = prices[t + 1] / prices[t + 1 - c.lookback] - 1
        if np.ptp(past) < 1e-12:
            continue  # no economically defined ranking
        order = np.argsort(past, kind="stable")
        bucket = max(1, c.assets // 3)
        losers, winners = order[:bucket], order[-bucket:]
        for h in c.horizons:
            entry, exit_ = t + 2, t + 2 + h
            if exit_ >= len(prices):
                continue  # right-censored labels are absent, not zero
            forward = prices[exit_] / prices[entry] - 1
            loser, winner, market = float(forward[losers].mean()), float(forward[winners].mean()), float(forward.mean())
            x, y = ranks(-past), ranks(forward)
            ic = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0 else None
            rows.append({"mechanism": result.mechanism, "control": result.control, "seed": c.seed,
                         "signal_step": t, "entry_step": t + 1, "exit_step": t + 1 + h, "horizon": h,
                         "latent_stage": state["latent_stage"], "proxy_stage": state["proxy_stage"],
                         "disagreement_bin_latent": state["disagreement_bin_latent"],
                         "belief_disagreement_latent": state["belief_disagreement_latent"],
                         "event_window": "pre" if t < c.shock_step else ("stress_window" if t < c.shock_step + c.stress_duration else "post"),
                         "fixed_event_signal": t == c.shock_step + 2,
                         "loser_assets": ";".join(result.initial["asset_codes"][j] for j in losers),
                         "winner_assets": ";".join(result.initial["asset_codes"][j] for j in winners),
                         "loser_gross": loser, "winner_gross": winner, "market_gross": market,
                         "long_only_excess_gross": loser - market, "spread_gross": loser - winner, "rank_ic": ic})
    return rows


def cost_adjust(row, cost_bps):
    """Constant notional commissions + spread + slippage assumption, per side.

    L/S has +1 and -1 exposure (gross 2), four legs; no borrow fee is modeled.
    Long-only has two legs. This is a cost stress test, not fill simulation.
    """
    cost = cost_bps / 10000
    return {**row, "cost_bps_per_side": cost_bps,
            "long_only_excess_net_proxy": row["long_only_excess_gross"] - 2 * cost,
            "spread_net_theoretical": row["spread_gross"] - 4 * cost}


def bootstrap_mean(values, rng, repetitions=1000):
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return None, None
    means = values[rng.integers(0, len(values), (repetitions, len(values)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def summarize(rows, costs, seed=20261106):
    """Average within a run first, then bootstrap independent seeds.

    Stage membership can change under intervention. Conditional tables describe
    that changing population; paired causal simulator contrasts use a fixed
    event signal instead. No t-statistic treats overlapping days as independent.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for kind, state in (("latent", row["latent_stage"]), ("observable_proxy", row["proxy_stage"])):
            for cost in costs:
                key = (row["mechanism"], row["control"], kind, state,
                       row["disagreement_bin_latent"], row["horizon"], cost)
                groups[key][row["seed"]].append(cost_adjust(row, cost))
    output = []
    rng = np.random.default_rng(seed)
    for key, per_seed in sorted(groups.items()):
        means = [np.mean([r["spread_net_theoretical"] for r in rs]) for rs in per_seed.values()]
        lo, hi = bootstrap_mean(means, rng)
        long_means = [np.mean([r["long_only_excess_net_proxy"] for r in rs]) for rs in per_seed.values()]
        ic_means = [np.mean([r["rank_ic"] for r in rs if r["rank_ic"] is not None]) for rs in per_seed.values() if any(r["rank_ic"] is not None for r in rs)]
        output.append(dict(zip(("mechanism", "control", "stage_source", "stage", "disagreement_bin_latent", "horizon", "cost_bps_per_side"), key),
                           n_seeds=len(per_seed), n_overlapping_cohorts=sum(map(len, per_seed.values())),
                           spread_net_theoretical_mean=float(np.mean(means)), seed_mc_ci_low=lo, seed_mc_ci_high=hi,
                           long_only_excess_net_proxy_mean=float(np.mean(long_means)),
                           mean_rank_ic=float(np.mean(ic_means)) if ic_means else None))
    return output


def paired_contrasts(rows, costs, seed=20261107):
    fixed = {(r["mechanism"], r["control"], r["seed"], r["horizon"]): r for r in rows if r["fixed_event_signal"]}
    groups = defaultdict(list)
    for (mechanism, control, run_seed, horizon), row in fixed.items():
        if control == "baseline":
            continue
        base = fixed.get((mechanism, "baseline", run_seed, horizon))
        if base is None:
            continue
        for cost in costs:
            b, r = cost_adjust(base, cost), cost_adjust(row, cost)
            groups[(mechanism, control, horizon, cost)].append((b["spread_net_theoretical"] - r["spread_net_theoretical"],
                                                              b["long_only_excess_net_proxy"] - r["long_only_excess_net_proxy"]))
    output, rng = [], np.random.default_rng(seed)
    for key, values in sorted(groups.items()):
        array = np.asarray(values)
        lo, hi = bootstrap_mean(array[:, 0], rng)
        output.append(dict(zip(("mechanism", "comparator", "horizon", "cost_bps_per_side"), key),
                           n_seed_pairs=len(values), baseline_minus_control_spread=float(array[:, 0].mean()),
                           baseline_minus_control_long_only=float(array[:, 1].mean()), seed_mc_ci_low=lo, seed_mc_ci_high=hi))
    return output
