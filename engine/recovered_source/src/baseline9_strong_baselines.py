#!/usr/bin/env python3
"""Baseline 9: strong causal baselines for online fidelity allocation.

Adds two contextual learning policies and a non-deployable offline dynamic-
programming information comparator.  All deployable policies obey the current
budget before model execution.  All-L2 trajectories remain post-run references,
never deployable-policy inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from baseline4_multidomain import (  # noqa: E402
    HydroL1,
    HydroL2,
    PropulsionL1,
    PropulsionL2,
)
from baseline5_budget_allocator import benchmark_models  # noqa: E402
from baseline7_online_allocation import (  # noqa: E402
    LEVELS,
    causal_rate,
    SwitchAwareEnumerativeAllocator,
    _relative_gap,
    build_trial_rom,
    calibrate_benefit_priors,
    confidence_interval,
    hydro_features,
    make_cost_profile,
    nrmse,
    offline_oracle_benefits,
    propulsion_features,
    run_all_l2_reference,
    simulate_strategy,
    tune_exact_policy,
)
from baseline8_budget_sweep import (  # noqa: E402
    DEFAULT_HEADROOM,
    SCENARIOS,
    budget_trace,
    scenario_conditions,
)


STRATEGIES = (
    "Greedy-Online",
    "Switch-Aware-Enumerative",
    "Discounted-LinUCB",
    "Contextual-Thompson",
    "Offline-DP-Information",
)
DEPLOYABLE = STRATEGIES[:-1]


class DiscountedLinearPosterior:
    """Ridge-regularised discounted linear reward model."""

    def __init__(self, prior, discount=0.995, ridge=1.0):
        self.prior = np.asarray(prior, dtype=float)
        self.discount = float(discount)
        self.ridge = float(ridge)
        self.identity = np.eye(len(self.prior))
        self.A = self.ridge * self.identity
        self.b = self.A @ self.prior
        self.updates = 0
        self._dirty = True
        self._inverse = None
        self._theta = None

    def _refresh_cache(self):
        if self._dirty:
            self._inverse = np.linalg.inv(self.A)
            self._theta = self._inverse @ self.b
            self._dirty = False

    def mean_and_variance(self, features):
        x = np.asarray(features, dtype=float)
        self._refresh_cache()
        mean = float(x @ self._theta)
        variance = float(max(x @ self._inverse @ x, 0.0))
        return mean, variance

    def update(self, features, target):
        x = np.asarray(features, dtype=float)
        gamma = self.discount
        prior_precision = self.ridge * self.identity
        self.A = (gamma * self.A + np.outer(x, x)
                  + (1.0 - gamma) * prior_precision)
        self.b = (gamma * self.b + x * float(np.clip(target, 0.0, 0.50))
                  + (1.0 - gamma) * prior_precision @ self.prior)
        self.updates += 1
        self._dirty = True


class ContextualLearningPolicy:
    """Budget-feasible LinUCB or linear Thompson sampling policy."""

    def __init__(self, costs, priors, mode, seed, discount=0.995,
                 exploration=0.05, min_dwell_steps=5):
        if mode not in ("linucb", "thompson"):
            raise ValueError(mode)
        self.mode = mode
        self.exploration = float(exploration)
        self.rng = np.random.default_rng(seed)
        self.hydro = DiscountedLinearPosterior(priors[0], discount)
        self.propulsion = DiscountedLinearPosterior(priors[1], discount)
        self.allocator = SwitchAwareEnumerativeAllocator(
            costs,
            switch_penalty=0.003,
            utilisation_penalty=0.0,
            decision_interval_steps=1,
            min_dwell_steps=min_dwell_steps,
        )

    def _benefit(self, posterior, features):
        mean, variance = posterior.mean_and_variance(features)
        if self.mode == "linucb":
            value = mean + self.exploration * np.sqrt(variance)
        else:
            value = self.rng.normal(mean, self.exploration * np.sqrt(variance))
        return float(np.clip(value, 0.0, 0.50)), float(np.clip(mean, 0.0, 0.50))

    def choose(self, hydro_x, propulsion_x, budget):
        hydro_value, hydro_mean = self._benefit(self.hydro, hydro_x)
        prop_value, prop_mean = self._benefit(self.propulsion, propulsion_x)
        levels = self.allocator.choose((hydro_value, prop_value), budget)
        return levels, (hydro_value, prop_value), (hydro_mean, prop_mean)


def simulate_learning_policy(mode, t, waves, speed, rom, dt, costs, budgets,
                             priors, config, seed):
    policy = ContextualLearningPolicy(
        costs, priors, mode, seed,
        discount=config["discount"],
        exploration=config["exploration"],
        min_dwell_steps=config.get("min_dwell_steps", 5),
    )
    hydro_models = HydroL1(), HydroL2(rom, dt)
    prop_models = PropulsionL1(), PropulsionL2()
    n = len(t)
    resistance, power = np.empty(n), np.empty(n)
    h_level, p_level = np.empty(n, dtype=int), np.empty(n, dtype=int)
    estimated_cost, decision_latency = np.empty(n), np.empty(n)
    speed_rate, hs_rate = causal_rate(speed, dt), causal_rate(waves, dt)
    previous_resistance = None
    previous_load_rate = 0.0
    previous_prop_output = None
    previous_p_level = 0

    for i, ti in enumerate(t):
        h_x = hydro_features(ti, waves[i], hs_rate[i], speed[i])
        h_mean, _ = policy.hydro.mean_and_variance(h_x)
        p_x = propulsion_features(
            speed_rate[i], previous_load_rate,
            float(np.clip(h_mean, 0.0, 0.50)), speed[i])
        started = time.perf_counter_ns()
        levels, _, _ = policy.choose(h_x, p_x, budgets[i])
        decision_latency[i] = (time.perf_counter_ns() - started) / 1000.0
        h_level[i], p_level[i] = levels

        if h_level[i] == 1:
            h_low = hydro_models[0].step(ti, waves[i], 8.0, speed[i], dt)
            h_out = hydro_models[1].step(ti, waves[i], 8.0, speed[i], dt)
            policy.hydro.update(
                h_x, _relative_gap(h_out.resistance_kN, h_low.resistance_kN))
        else:
            h_out = hydro_models[0].step(ti, waves[i], 8.0, speed[i], dt)

        if p_level[i] == 1 and previous_p_level == 0 and previous_prop_output is not None:
            prop_models[1].initialise_from(previous_prop_output)
        if p_level[i] == 1:
            p_low = prop_models[0].step(h_out.resistance_kN, speed[i], dt)
            p_out = prop_models[1].step(h_out.resistance_kN, speed[i], dt)
            policy.propulsion.update(
                p_x, _relative_gap(p_out.power_kW, p_low.power_kW))
        else:
            p_out = prop_models[0].step(h_out.resistance_kN, speed[i], dt)

        resistance[i], power[i] = h_out.resistance_kN, p_out.power_kW
        estimated_cost[i] = costs.combination_cost(levels)
        if previous_resistance is not None:
            previous_load_rate = abs(resistance[i] - previous_resistance) / (
                max(abs(previous_resistance), 1.0) * dt)
        previous_resistance = resistance[i]
        previous_prop_output = p_out
        previous_p_level = p_level[i]

    return {
        "resistance": resistance,
        "power": power,
        "mean_cost_us": float(np.mean(estimated_cost)),
        "budget_violation_fraction": float(np.mean(estimated_cost > budgets + 1e-12)),
        "hydro_l2_share": float(np.mean(h_level)),
        "propulsion_l2_share": float(np.mean(p_level)),
        "switches": int(np.sum((np.diff(h_level) != 0) | (np.diff(p_level) != 0))),
        "mean_decision_latency_us": float(np.mean(decision_latency)),
        "p95_decision_latency_us": float(np.percentile(decision_latency, 95)),
        "hydro_updates": policy.hydro.updates,
        "propulsion_updates": policy.propulsion.updates,
    }


def offline_dp_actions(hydro_benefit, propulsion_benefit, budgets, costs,
                       switch_penalty=0.003):
    """Exact finite-horizon DP for the declared additive information objective."""

    hydro_benefit = np.asarray(hydro_benefit, dtype=float)
    propulsion_benefit = np.asarray(propulsion_benefit, dtype=float)
    n, actions = len(hydro_benefit), list(LEVELS)
    dp = np.full((n, len(actions)), -np.inf)
    parent = np.full((n, len(actions)), -1, dtype=int)
    for a_index, action in enumerate(actions):
        if costs.combination_cost(action) <= budgets[0] + 1e-12:
            switches = sum(int(level != 0) for level in action)
            dp[0, a_index] = (action[0] * hydro_benefit[0]
                              + action[1] * propulsion_benefit[0]
                              - switch_penalty * switches)
    for i in range(1, n):
        for a_index, action in enumerate(actions):
            if costs.combination_cost(action) > budgets[i] + 1e-12:
                continue
            stage = action[0] * hydro_benefit[i] + action[1] * propulsion_benefit[i]
            candidates = []
            for previous_index, previous in enumerate(actions):
                transition = switch_penalty * sum(
                    int(a != b) for a, b in zip(action, previous))
                candidates.append(dp[i - 1, previous_index] + stage - transition)
            parent[i, a_index] = int(np.argmax(candidates))
            dp[i, a_index] = candidates[parent[i, a_index]]
    end = int(np.argmax(dp[-1]))
    selected = [None] * n
    selected[-1] = actions[end]
    for i in range(n - 1, 0, -1):
        end = parent[i, end]
        selected[i - 1] = actions[end]
    return np.asarray(selected, dtype=int), float(np.max(dp[-1]))


def simulate_forced_actions(actions, t, waves, speed, rom, dt, costs, budgets,
                            planning_latency_us=0.0):
    hydro_models = HydroL1(), HydroL2(rom, dt)
    prop_models = PropulsionL1(), PropulsionL2()
    resistance, power = np.empty(len(t)), np.empty(len(t))
    estimated_cost = np.empty(len(t))
    previous_prop_output = None
    previous_p_level = 0
    for i, ti in enumerate(t):
        h_level, p_level = int(actions[i, 0]), int(actions[i, 1])
        if h_level == 1:
            hydro_models[0].step(ti, waves[i], 8.0, speed[i], dt)
        h_out = hydro_models[h_level].step(ti, waves[i], 8.0, speed[i], dt)
        if p_level == 1 and previous_p_level == 0 and previous_prop_output is not None:
            prop_models[1].initialise_from(previous_prop_output)
        if p_level == 1:
            prop_models[0].step(h_out.resistance_kN, speed[i], dt)
        p_out = prop_models[p_level].step(h_out.resistance_kN, speed[i], dt)
        resistance[i], power[i] = h_out.resistance_kN, p_out.power_kW
        estimated_cost[i] = costs.combination_cost((h_level, p_level))
        previous_prop_output, previous_p_level = p_out, p_level
    return {
        "resistance": resistance,
        "power": power,
        "mean_cost_us": float(np.mean(estimated_cost)),
        "budget_violation_fraction": float(np.mean(estimated_cost > budgets + 1e-12)),
        "hydro_l2_share": float(np.mean(actions[:, 0])),
        "propulsion_l2_share": float(np.mean(actions[:, 1])),
        "switches": int(np.sum(np.any(np.diff(actions, axis=0) != 0, axis=1))),
        "mean_decision_latency_us": float(planning_latency_us),
        "p95_decision_latency_us": float(planning_latency_us),
    }


def tune_learning_policy(mode, database_time, dt, costs, priors):
    seed = 7919
    t = np.arange(0.0, 180.0, dt)
    waves, speed = scenario_conditions(t, "coupled_extreme", seed)
    rom = build_trial_rom(database_time, seed)
    reference_r, reference_p = run_all_l2_reference(t, waves, speed, rom, dt)
    budgets, _ = budget_trace(t, costs, 0.90, seed)
    candidates = []
    for discount in (0.98, 0.995):
        for exploration in (0.01, 0.03, 0.06, 0.12):
            config = {"discount": discount, "exploration": exploration,
                      "min_dwell_steps": 5}
            result = simulate_learning_policy(
                mode, t, waves, speed, rom, dt, costs, budgets, priors,
                config, seed)
            composite = 0.5 * (
                nrmse(result["resistance"], reference_r)
                + nrmse(result["power"], reference_p))
            score = (composite
                     + 0.005 * result["mean_cost_us"] / costs.all_high
                     + 0.01 * result["switches"] / len(t))
            candidates.append({"config": config, "composite_nrmse": composite,
                               "selection_score": score})
    selected = min(candidates, key=lambda item: item["selection_score"])
    return selected["config"], {
        "calibration_seed": seed,
        "mode": mode,
        "candidate_count": len(candidates),
        "selected": selected,
    }


def _summarise(rows):
    metrics = tuple(key for key in rows[0]
                    if key not in ("trial", "scenario", "headroom", "strategy"))
    grouped = {}
    for row in rows:
        grouped.setdefault((row["strategy"], row["headroom"]), []).append(row)
    output = []
    for (strategy, headroom), selected in grouped.items():
        item = {"strategy": strategy, "headroom": headroom,
                "samples": len(selected)}
        for metric in metrics:
            mean, low, high = confidence_interval([row[metric] for row in selected])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_ci95_low"] = low
            item[f"{metric}_ci95_high"] = high
        output.append(item)
    return sorted(output, key=lambda row: (row["strategy"], row["headroom"]))


def paired_summary(rows, baseline):
    grouped = {}
    for row in rows:
        grouped[(row["strategy"], row["headroom"], row["trial"], row["scenario"])] = row
    output = []
    for headroom in sorted(set(float(row["headroom"]) for row in rows)):
        differences = []
        for trial in sorted(set(int(row["trial"]) for row in rows)):
            for scenario in SCENARIOS:
                proposed = grouped.get(("Switch-Aware-Enumerative", headroom, trial, scenario))
                other = grouped.get((baseline, headroom, trial, scenario))
                if proposed is not None and other is not None:
                    differences.append(proposed["composite_nrmse"]
                                       - other["composite_nrmse"])
        mean, low, high = confidence_interval(differences)
        output.append({
            "baseline": baseline,
            "headroom": headroom,
            "paired_samples": len(differences),
            "composite_difference_mean": mean,
            "composite_difference_ci95_low": low,
            "composite_difference_ci95_high": high,
        })
    return output


def run_benchmark(trials=12, duration_s=180.0, dt=0.2,
                  headroom_levels=DEFAULT_HEADROOM):
    t = np.arange(0.0, duration_s, dt)
    database_time = np.arange(0.0, max(duration_s, 180.0), dt)
    timing_rom = build_trial_rom(database_time, 0)
    timing = benchmark_models(timing_rom, dt)
    costs = make_cost_profile(timing)
    h_prior, p_prior, prior_diagnostics = calibrate_benefit_priors(
        database_time, dt)
    priors = h_prior, p_prior
    exact_config, exact_tuning = tune_exact_policy(database_time, dt, costs, priors)
    linucb_config, linucb_tuning = tune_learning_policy(
        "linucb", database_time, dt, costs, priors)
    thompson_config, thompson_tuning = tune_learning_policy(
        "thompson", database_time, dt, costs, priors)
    rows = []

    for trial in range(1, trials + 1):
        rom = build_trial_rom(database_time, trial)
        for scenario_index, scenario in enumerate(SCENARIOS):
            seed = trial * 101 + scenario_index * 1009
            waves, speed = scenario_conditions(t, scenario, seed)
            reference_r, reference_p = run_all_l2_reference(t, waves, speed, rom, dt)
            information = offline_oracle_benefits(
                t, waves, speed, rom, dt, reference_r, reference_p)
            for headroom in np.asarray(headroom_levels, dtype=float):
                budgets, _ = budget_trace(t, costs, float(headroom), seed)
                results = {}
                results["Greedy-Online"] = simulate_strategy(
                    "Greedy-Online", t, waves, speed, rom, dt, costs, budgets,
                    benefit_priors=priors, exact_config=exact_config)
                results["Switch-Aware-Enumerative"] = simulate_strategy(
                    "Proposed-Exact-Online", t, waves, speed, rom, dt, costs,
                    budgets, benefit_priors=priors, exact_config=exact_config)
                results["Discounted-LinUCB"] = simulate_learning_policy(
                    "linucb", t, waves, speed, rom, dt, costs, budgets, priors,
                    linucb_config, seed)
                results["Contextual-Thompson"] = simulate_learning_policy(
                    "thompson", t, waves, speed, rom, dt, costs, budgets, priors,
                    thompson_config, seed)
                started = time.perf_counter_ns()
                actions, _ = offline_dp_actions(
                    information[0], information[1], budgets, costs)
                planning_per_step = ((time.perf_counter_ns() - started)
                                     / 1000.0 / len(t))
                results["Offline-DP-Information"] = simulate_forced_actions(
                    actions, t, waves, speed, rom, dt, costs, budgets,
                    planning_per_step)

                for strategy, result in results.items():
                    r_error = nrmse(result["resistance"], reference_r)
                    p_error = nrmse(result["power"], reference_p)
                    rows.append({
                        "trial": trial,
                        "scenario": scenario,
                        "headroom": float(headroom),
                        "strategy": strategy,
                        "mean_budget_ratio": float(np.mean(budgets) / costs.all_high),
                        "resistance_nrmse": r_error,
                        "power_nrmse": p_error,
                        "composite_nrmse": 0.5 * (r_error + p_error),
                        "mean_cost_us": result["mean_cost_us"],
                        "budget_violation_fraction": result["budget_violation_fraction"],
                        "hydro_l2_share": result["hydro_l2_share"],
                        "propulsion_l2_share": result["propulsion_l2_share"],
                        "switches": result["switches"],
                        "mean_decision_latency_us": result["mean_decision_latency_us"],
                        "p95_decision_latency_us": result["p95_decision_latency_us"],
                    })

    summary = _summarise(rows)
    paired = []
    for baseline in ("Greedy-Online", "Discounted-LinUCB", "Contextual-Thompson"):
        paired.extend(paired_summary(rows, baseline))
    metadata = {
        "scope": "strong-baseline comparison on synthetic/engineering models",
        "trials_per_scenario": trials,
        "scenarios": list(SCENARIOS),
        "budget_headroom_levels": np.asarray(headroom_levels).tolist(),
        "strategies": list(STRATEGIES),
        "executed_policy_runs": len(rows),
        "total_decisions_evaluated": len(rows) * len(t),
        "deployable_online_decisions": (
            trials * len(SCENARIOS) * len(headroom_levels)
            * len(DEPLOYABLE) * len(t)
        ),
        "offline_information_decisions": (
            trials * len(SCENARIOS) * len(headroom_levels) * len(t)
        ),
        "costs_p90_with_shadow_us": costs.as_dict(),
        "benefit_prior_calibration": prior_diagnostics,
        "enumerative_policy_tuning": exact_tuning,
        "linucb_tuning": linucb_tuning,
        "thompson_tuning": thompson_tuning,
        "offline_dp_scope": (
            "exact only for the additive current-information objective; not a "
            "physical end-to-end error bound"
        ),
        "limitations": [
            "synthetic POD source data",
            "engineering-estimate propulsion parameters",
            "two binary modules",
            "replay budget rather than live resource enforcement",
        ],
    }
    return rows, summary, paired, metadata


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _series(summary, strategy):
    return sorted((row for row in summary if row["strategy"] == strategy),
                  key=lambda row: float(row["headroom"]))


def plot_results(summary, paired, metadata, output_path):
    colors = {
        "Greedy-Online": "#2a9d8f",
        "Switch-Aware-Enumerative": "#e76f51",
        "Discounted-LinUCB": "#6a4c93",
        "Contextual-Thompson": "#277da1",
        "Offline-DP-Information": "#222222",
    }
    markers = ("^", "o", "D", "s", "X")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))

    ax = axes[0, 0]
    for strategy, marker in zip(STRATEGIES, markers):
        selected = _series(summary, strategy)
        x = np.array([r["mean_budget_ratio_mean"] for r in selected])
        y = 100 * np.array([r["composite_nrmse_mean"] for r in selected])
        ax.plot(x, y, marker=marker, linewidth=1.8, color=colors[strategy],
                label=strategy)
        if strategy != "Offline-DP-Information":
            low = 100 * np.array([r["composite_nrmse_ci95_low"] for r in selected])
            high = 100 * np.array([r["composite_nrmse_ci95_high"] for r in selected])
            ax.fill_between(x, low, high, color=colors[strategy], alpha=0.10)
    ax.set_xlabel("Mean available budget / all-L2 cost")
    ax.set_ylabel("Composite NRMSE [%]")
    ax.set_title("Strong-baseline budget response")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for strategy, marker in zip(DEPLOYABLE, markers):
        selected = _series(summary, strategy)
        ax.plot([r["mean_cost_us_mean"] for r in selected],
                [100 * r["composite_nrmse_mean"] for r in selected],
                marker=marker, color=colors[strategy], linewidth=1.8,
                label=strategy)
    ax.set_xlabel("Mean allocated model cost [us/step]")
    ax.set_ylabel("Composite NRMSE [%]")
    ax.set_title("Accuracy-cost paths")
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    for strategy, marker in zip(DEPLOYABLE, markers):
        selected = _series(summary, strategy)
        ax.plot([r["mean_budget_ratio_mean"] for r in selected],
                [r["switches_mean"] for r in selected],
                marker=marker, color=colors[strategy], linewidth=1.8,
                label=strategy)
    ax.set_xlabel("Mean available budget / all-L2 cost")
    ax.set_ylabel("Mean switches / run")
    ax.set_title("Decision stability")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    for strategy, marker in zip(DEPLOYABLE, markers):
        selected = _series(summary, strategy)
        ax.plot([r["mean_budget_ratio_mean"] for r in selected],
                [r["p95_decision_latency_us_mean"] for r in selected],
                marker=marker, color=colors[strategy], linewidth=1.8,
                label=strategy)
    ax.set_yscale("log")
    ax.set_xlabel("Mean available budget / all-L2 cost")
    ax.set_ylabel("Per-run P95 decision latency [us, log]")
    ax.set_title("Robust online policy overhead")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    for baseline in ("Greedy-Online", "Discounted-LinUCB", "Contextual-Thompson"):
        selected = sorted((row for row in paired if row["baseline"] == baseline),
                          key=lambda row: float(row["headroom"]))
        x = np.array([float(r["headroom"]) for r in selected])
        y = 100 * np.array([r["composite_difference_mean"] for r in selected])
        low = 100 * np.array([r["composite_difference_ci95_low"] for r in selected])
        high = 100 * np.array([r["composite_difference_ci95_high"] for r in selected])
        ax.plot(x, y, marker="o", color=colors[baseline], label=baseline)
        ax.fill_between(x, low, high, color=colors[baseline], alpha=0.12)
    ax.axhline(0, color="#222222", linestyle="--", linewidth=1)
    ax.set_xlabel("Budget headroom")
    ax.set_ylabel("Enumerative minus baseline NRMSE [pp]")
    ax.set_title("Paired accuracy differences (95% CI)")
    ax.legend(fontsize=7)

    ax = axes[1, 2]
    offline = _series(summary, "Offline-DP-Information")
    proposed = _series(summary, "Switch-Aware-Enumerative")
    x = [r["mean_budget_ratio_mean"] for r in proposed]
    gap = [100 * (p["composite_nrmse_mean"] - o["composite_nrmse_mean"])
           for p, o in zip(proposed, offline)]
    ax.plot(x, gap, "o-", color="#e76f51", linewidth=2)
    ax.axhline(0, color="#222222", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean available budget / all-L2 cost")
    ax.set_ylabel("NRMSE difference [percentage points]")
    ax.set_title("Gap to offline information comparator")

    for ax in axes.flat:
        ax.grid(True, alpha=0.22)
    fig.suptitle(
        "Strong online baselines for causal multi-fidelity scheduling\n"
        f"{metadata['executed_policy_runs']:,} executed policy runs",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_outputs(rows, summary, paired, metadata, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "baseline9_strong_baseline_trials.csv"
    summary_path = output_dir / "baseline9_strong_baseline_summary.csv"
    paired_path = output_dir / "baseline9_paired_comparisons.csv"
    audit_path = output_dir / "baseline9_strong_baseline_audit.json"
    figure_path = output_dir / "baseline9_strong_baselines.png"
    write_csv(raw_path, rows)
    write_csv(summary_path, summary)
    write_csv(paired_path, paired)
    audit_path.write_text(json.dumps(
        {"metadata": metadata, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    plot_results(summary, paired, metadata, figure_path)
    return raw_path, summary_path, paired_path, audit_path, figure_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--budget-points", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs")
    args = parser.parse_args()
    levels = (DEFAULT_HEADROOM if args.budget_points == 0
              else np.linspace(0.0, 1.35, args.budget_points))
    result = run_benchmark(args.trials, args.duration, args.dt, levels)
    paths = save_outputs(*result, args.output_dir)
    print(json.dumps(result[3], ensure_ascii=False, indent=2))
    print("Saved:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
