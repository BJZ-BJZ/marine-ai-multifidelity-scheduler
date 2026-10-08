#!/usr/bin/env python3
"""Baseline 8: multi-scenario budget-response and Pareto experiment.

This module increases experimental density without manufacturing interpolated
points.  Every plotted marker is obtained from an executed causal simulation at
one budget headroom, scenario, and random seed.  It reuses the Baseline-7 online
allocator and preserves its no-reference-leakage boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from baseline5_budget_allocator import benchmark_models  # noqa: E402
from baseline7_online_allocation import (  # noqa: E402
    DEPLOYABLE_STRATEGIES,
    build_trial_rom,
    calibrate_benefit_priors,
    confidence_interval,
    make_cost_profile,
    nrmse,
    run_all_l2_reference,
    simulate_strategy,
    storm_profile,
    speed_profile,
    tune_exact_policy,
)


SCENARIOS = (
    "storm_passage",
    "multi_squall",
    "speed_manoeuvre",
    "coupled_extreme",
)
SWEEP_STRATEGIES = (
    "Fixed-L1",
    "Fixed-L2",
    "Threshold-Projected",
    "Greedy-Online",
    "Proposed-Exact-Online",
)
DEFAULT_HEADROOM = np.array([
    0.00, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90,
    1.05, 1.20, 1.35, 1.70, 2.20, 3.00,
])


def _smooth_noise(length: int, rng, sigma: float, width: int = 21):
    raw = rng.normal(0.0, sigma, length + width - 1)
    return np.convolve(raw, np.ones(width) / width, mode="valid")


def scenario_conditions(t: np.ndarray, scenario: str, seed: int):
    """Generate four distinct, fully observed online operating scenarios."""

    rng = np.random.default_rng(seed + 67867967)
    noise = _smooth_noise(len(t), rng, 0.07)

    if scenario == "storm_passage":
        waves = storm_profile(t) + noise
        speed = speed_profile(t) + 0.06 * np.sin(2.0 * np.pi * t / 43.0)
    elif scenario == "multi_squall":
        waves = np.full(len(t), 0.65)
        for centre, amplitude, width in ((28, 1.5, 7), (67, 2.6, 10),
                                         (112, 1.9, 8), (153, 2.8, 11)):
            waves += amplitude * np.exp(-0.5 * ((t - centre) / width) ** 2)
        speed = 23.5 + 0.9 * np.sin(2.0 * np.pi * t / 58.0)
        speed += 0.6 * np.tanh((t - 90.0) / 8.0)
        waves += noise
    elif scenario == "speed_manoeuvre":
        waves = 1.05 + 0.28 * np.sin(2.0 * np.pi * t / 31.0) + noise
        speed = np.full(len(t), 22.0)
        ramp_1 = (t >= 20.0) & (t < 45.0)
        speed[ramp_1] = 22.0 + 5.0 * (t[ramp_1] - 20.0) / 25.0
        speed[(t >= 45.0) & (t < 82.0)] = 27.0
        ramp_2 = (t >= 82.0) & (t < 104.0)
        speed[ramp_2] = 27.0 - 7.0 * (t[ramp_2] - 82.0) / 22.0
        speed[(t >= 104.0) & (t < 135.0)] = 20.0
        ramp_3 = (t >= 135.0) & (t < 160.0)
        speed[ramp_3] = 20.0 + 4.0 * (t[ramp_3] - 135.0) / 25.0
        speed[t >= 160.0] = 24.0
    elif scenario == "coupled_extreme":
        waves = storm_profile(t)
        waves += 0.45 * np.sin(2.0 * np.pi * t / 27.0) + noise
        waves += 0.8 * np.exp(-0.5 * ((t - 132.0) / 9.0) ** 2)
        speed = speed_profile(t)
        speed += 1.1 * np.sin(2.0 * np.pi * t / 46.0)
        speed -= 1.5 * np.exp(-0.5 * ((t - 88.0) / 7.0) ** 2)
    else:
        raise KeyError(scenario)

    return np.clip(waves, 0.1, 4.8), np.clip(speed, 18.0, 28.0)


def budget_trace(t: np.ndarray, costs, headroom: float, seed: int):
    """Continuously vary feasible compute above the mandatory all-L1 base."""

    rng = np.random.default_rng(seed + 86028121)
    phase = rng.uniform(-5.0, 5.0)
    availability = 0.78 + 0.16 * np.sin(2.0 * np.pi * (t + phase) / 95.0)
    availability -= 0.27 * np.exp(-0.5 * ((t - 82.0 - phase) / 14.0) ** 2)
    availability -= 0.17 * np.exp(-0.5 * ((t - 142.0 + phase) / 10.0) ** 2)
    availability += _smooth_noise(len(t), rng, 0.018, width=15)
    availability = np.clip(availability, 0.35, 1.0)

    base = costs.base * 1.015
    full_increment = costs.all_high - costs.base
    budget = base + float(headroom) * full_increment * availability
    return np.maximum(budget, costs.base * 1.001), availability


def _summary(rows, group_keys, metrics):
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, selected in grouped.items():
        item = dict(zip(group_keys, key))
        item["samples"] = len(selected)
        for metric in metrics:
            mean, low, high = confidence_interval([row[metric] for row in selected])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_ci95_low"] = low
            item[f"{metric}_ci95_high"] = high
        output.append(item)
    return sorted(output, key=lambda row: tuple(str(row[k]) for k in group_keys))


def run_sweep(trials=12, duration_s=180.0, dt=0.2,
              headroom_levels=None, scenarios=SCENARIOS,
              strategies=SWEEP_STRATEGIES):
    t = np.arange(0.0, duration_s, dt)
    database_time = np.arange(0.0, max(duration_s, 180.0), dt)
    headroom_levels = np.asarray(
        DEFAULT_HEADROOM if headroom_levels is None else headroom_levels,
        dtype=float,
    )

    timing_rom = build_trial_rom(database_time, 0)
    timing = benchmark_models(timing_rom, dt)
    costs = make_cost_profile(timing)
    h_prior, p_prior, prior_diagnostics = calibrate_benefit_priors(
        database_time, dt)
    priors = h_prior, p_prior
    exact_config, tuning = tune_exact_policy(database_time, dt, costs, priors)

    rows = []
    representative = {}
    representative_index = int(np.argmin(abs(headroom_levels - 0.75)))
    representative_headroom = float(headroom_levels[representative_index])

    for trial in range(1, trials + 1):
        rom = build_trial_rom(database_time, trial)
        for scenario_index, scenario in enumerate(scenarios):
            condition_seed = trial * 101 + scenario_index * 1009
            waves, speed = scenario_conditions(t, scenario, condition_seed)
            reference_r, reference_p = run_all_l2_reference(
                t, waves, speed, rom, dt)

            for headroom in headroom_levels:
                budget, availability = budget_trace(
                    t, costs, float(headroom), condition_seed)
                mean_budget_ratio = float(np.mean(budget) / costs.all_high)

                for strategy in strategies:
                    keep_trace = (
                        trial == 1
                        and strategy == "Proposed-Exact-Online"
                        and abs(float(headroom) - representative_headroom) < 1e-12
                    )
                    result = simulate_strategy(
                        strategy, t, waves, speed, rom, dt, costs, budget,
                        benefit_priors=priors,
                        exact_config=exact_config,
                        keep_trace=keep_trace,
                    )
                    r_error = nrmse(result["resistance"], reference_r)
                    p_error = nrmse(result["power"], reference_p)
                    rows.append({
                        "trial": trial,
                        "scenario": scenario,
                        "headroom": float(headroom),
                        "mean_budget_ratio": mean_budget_ratio,
                        "strategy": strategy,
                        "resistance_nrmse": r_error,
                        "power_nrmse": p_error,
                        "composite_nrmse": 0.5 * (r_error + p_error),
                        "mean_cost_us": result["mean_cost_us"],
                        "budget_violation_fraction": result["budget_violation_fraction"],
                        "hydro_l2_share": result["hydro_l2_share"],
                        "propulsion_l2_share": result["propulsion_l2_share"],
                        "switches": result["switches"],
                    })
                    if keep_trace:
                        representative[scenario] = result["trace"] | {
                            "availability": availability,
                            "reference_resistance_kN": reference_r,
                            "reference_power_kW": reference_p,
                        }

    metrics = (
        "mean_budget_ratio",
        "resistance_nrmse",
        "power_nrmse",
        "composite_nrmse",
        "mean_cost_us",
        "budget_violation_fraction",
        "hydro_l2_share",
        "propulsion_l2_share",
        "switches",
    )
    summary = _summary(rows, ("strategy", "headroom"), metrics)
    scenario_summary = _summary(
        rows, ("strategy", "scenario", "headroom"), metrics)
    metadata = {
        "scope": (
            "executed multi-scenario causal budget sweep using synthetic POD "
            "and engineering-estimate propulsion models"
        ),
        "trials_per_scenario": trials,
        "scenarios": list(scenarios),
        "scenario_trial_samples_per_budget": trials * len(scenarios),
        "budget_headroom_levels": headroom_levels.tolist(),
        "budget_points": len(headroom_levels),
        "strategies": list(strategies),
        "executed_policy_runs": len(rows),
        "duration_s": duration_s,
        "time_step_s": dt,
        "steps_per_run": len(t),
        "total_online_decisions": len(rows) * len(t),
        "allocation_costs_p90_with_l1_shadow_us": costs.as_dict(),
        "benefit_prior_calibration": prior_diagnostics,
        "exact_policy_calibration": tuning,
        "representative_headroom": representative_headroom,
        "no_interpolated_experiment_points": True,
        "limitations": [
            "synthetic POD source data",
            "engineering-estimate propulsion parameters",
            "two schedulable modules",
            "exogenous replay budget rather than live CPU/RAM enforcement",
            "no physical validation claim",
        ],
    }
    return rows, summary, scenario_summary, metadata, representative


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _strategy_rows(summary, strategy):
    return sorted(
        (row for row in summary if row["strategy"] == strategy),
        key=lambda row: float(row["headroom"]),
    )


def plot_budget_response(rows, summary, metadata, output_path: Path):
    colors = {
        "Fixed-L1": "#6c757d",
        "Fixed-L2": "#212529",
        "Threshold-Projected": "#e9c46a",
        "Greedy-Online": "#2a9d8f",
        "Proposed-Exact-Online": "#e76f51",
    }
    markers = {"Threshold-Projected": "s", "Greedy-Online": "^",
               "Proposed-Exact-Online": "o"}
    adaptive = ("Threshold-Projected", "Greedy-Online", "Proposed-Exact-Online")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))

    ax = axes[0, 0]
    for strategy in adaptive:
        selected = _strategy_rows(summary, strategy)
        x = np.array([r["mean_budget_ratio_mean"] for r in selected])
        y = 100.0 * np.array([r["composite_nrmse_mean"] for r in selected])
        low = 100.0 * np.array([r["composite_nrmse_ci95_low"] for r in selected])
        high = 100.0 * np.array([r["composite_nrmse_ci95_high"] for r in selected])
        ax.plot(x, y, marker=markers[strategy], color=colors[strategy],
                linewidth=2.0, label=strategy)
        ax.fill_between(x, low, high, color=colors[strategy], alpha=0.14)
    ax.set_xlabel("Mean available budget / all-L2 cost")
    ax.set_ylabel("Composite NRMSE [%]")
    ax.set_title("Budget-response curves (95% CI)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for strategy in adaptive:
        selected = _strategy_rows(summary, strategy)
        x = np.array([r["mean_cost_us_mean"] for r in selected])
        y = 100.0 * np.array([r["composite_nrmse_mean"] for r in selected])
        ax.plot(x, y, marker=markers[strategy], color=colors[strategy],
                linewidth=2.0, label=strategy)
    fixed_l1 = _strategy_rows(summary, "Fixed-L1")[0]
    fixed_l2 = _strategy_rows(summary, "Fixed-L2")[-1]
    ax.scatter(fixed_l1["mean_cost_us_mean"],
               100 * fixed_l1["composite_nrmse_mean"], s=70,
               color=colors["Fixed-L1"], label="Fixed-L1")
    ax.scatter(fixed_l2["mean_cost_us_mean"], 0.0, s=70,
               color=colors["Fixed-L2"], label="Fixed-L2 reference")
    ax.set_xlabel("Mean allocated model cost [us/step]")
    ax.set_ylabel("Composite NRMSE [%]")
    ax.set_title("Executed accuracy-cost Pareto paths")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 2]
    selected = _strategy_rows(summary, "Proposed-Exact-Online")
    x = np.array([r["mean_budget_ratio_mean"] for r in selected])
    ax.plot(x, 100 * np.array([r["hydro_l2_share_mean"] for r in selected]),
            "o-", color="#277da1", linewidth=2, label="Hydrodynamics L2")
    ax.plot(x, 100 * np.array([r["propulsion_l2_share_mean"] for r in selected]),
            "s-", color="#f8961e", linewidth=2, label="Propulsion L2")
    ax.set_xlabel("Mean available budget / all-L2 cost")
    ax.set_ylabel("Allocated L2 share [%]")
    ax.set_title("Module-specific fidelity allocation")
    ax.legend()

    ax = axes[1, 0]
    for strategy in adaptive:
        selected = _strategy_rows(summary, strategy)
        x = np.array([r["mean_budget_ratio_mean"] for r in selected])
        y = np.array([r["switches_mean"] for r in selected])
        ax.plot(x, y, marker=markers[strategy], color=colors[strategy],
                linewidth=2.0, label=strategy)
    ax.set_xlabel("Mean available budget / all-L2 cost")
    ax.set_ylabel("Mean allocation switches / run")
    ax.set_title("Switching response to budget headroom")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for strategy in ("Fixed-L2",) + adaptive:
        selected = _strategy_rows(summary, strategy)
        x = np.array([r["mean_budget_ratio_mean"] for r in selected])
        y = 100.0 * np.array([r["budget_violation_fraction_mean"] for r in selected])
        ax.plot(x, y, marker="o", color=colors[strategy], linewidth=2,
                label=strategy)
    ax.set_xlabel("Mean available budget / all-L2 cost")
    ax.set_ylabel("Budget-violation steps [%]")
    ax.set_title("Constraint feasibility across capacity")
    ax.legend(fontsize=7)

    ax = axes[1, 2]
    middle = float(metadata["budget_headroom_levels"][
        len(metadata["budget_headroom_levels"]) // 2])
    selected_raw = [row for row in rows
                    if row["strategy"] == "Proposed-Exact-Online"
                    and abs(float(row["headroom"]) - middle) < 1e-12]
    scenario_names = list(metadata["scenarios"])
    data = [[100.0 * row["composite_nrmse"] for row in selected_raw
             if row["scenario"] == scenario] for scenario in scenario_names]
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for body, color in zip(parts["bodies"], ("#4c78a8", "#72b7b2", "#f2cf5b", "#e45756")):
        body.set_facecolor(color)
        body.set_alpha(0.72)
    parts["cmeans"].set_color("#222222")
    ax.set_xticks(range(1, len(scenario_names) + 1),
                  [name.replace("_", "\n") for name in scenario_names])
    ax.set_ylabel("Composite NRMSE [%]")
    ax.set_title(f"Scenario spread at headroom={middle:.2f}")

    for ax in axes.flat:
        ax.grid(True, alpha=0.22)
    fig.suptitle(
        "Causal multi-fidelity scheduling across scenarios and compute budgets\n"
        f"{metadata['executed_policy_runs']:,} executed policy runs; shaded bands are 95% CI",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_traces(traces, metadata, output_path: Path):
    scenarios = list(metadata["scenarios"])
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(17, 12), sharex=True)
    palette = ("#4c78a8", "#72b7b2", "#f2cf5b", "#e45756")
    for row, (scenario, color) in enumerate(zip(scenarios, palette)):
        trace = traces[scenario]
        t = np.asarray(trace["time_s"])
        ax = axes[row, 0]
        ax.plot(t, trace["Hs_m"], color=color, linewidth=1.7, label="Hs")
        twin = ax.twinx()
        twin.plot(t, trace["speed_kn"], color="#333333", linewidth=1.0,
                  alpha=0.75, label="Speed")
        ax.set_ylabel("Hs [m]")
        twin.set_ylabel("Speed [kn]")
        ax.set_title(scenario.replace("_", " ").title())

        ax = axes[row, 1]
        ax.plot(t, trace["budget_us"], color="#457b9d", linewidth=1.4,
                label="Available budget")
        ax.plot(t, trace["estimated_cost_us"], color="#e76f51", linewidth=1.2,
                label="Allocated cost")
        scale = max(trace["budget_us"])
        ax.fill_between(t, 0, 0.08 * scale,
                        where=np.asarray(trace["hydro_fidelity"]).astype(bool),
                        color="#277da1", alpha=0.55, step="post", label="H-L2")
        ax.fill_between(t, 0.09 * scale, 0.17 * scale,
                        where=np.asarray(trace["propulsion_fidelity"]).astype(bool),
                        color="#f8961e", alpha=0.55, step="post", label="P-L2")
        ax.set_ylabel("Cost [us/step]")
        ax.set_title("Budget, allocation cost, and L2 activity")
        if row == 0:
            ax.legend(ncol=4, fontsize=8, loc="upper right")

    for ax in axes[-1, :]:
        ax.set_xlabel("Simulation time [s]")
    for ax in axes.flat:
        ax.grid(True, alpha=0.20)
    fig.suptitle(
        "Representative causal online traces across four operating scenarios",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_outputs(rows, summary, scenario_summary, metadata, traces,
                 output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "baseline8_budget_sweep_trials.csv"
    summary_path = output_dir / "baseline8_budget_sweep_summary.csv"
    scenario_path = output_dir / "baseline8_scenario_summary.csv"
    audit_path = output_dir / "baseline8_budget_sweep_audit.json"
    response_path = output_dir / "baseline8_budget_response.png"
    traces_path = output_dir / "baseline8_scenario_traces.png"
    write_csv(raw_path, rows)
    write_csv(summary_path, summary)
    write_csv(scenario_path, scenario_summary)
    audit_path.write_text(json.dumps(
        {"metadata": metadata, "summary": summary},
        ensure_ascii=False, indent=2), encoding="utf-8")
    plot_budget_response(rows, summary, metadata, response_path)
    plot_scenario_traces(traces, metadata, traces_path)
    return raw_path, summary_path, scenario_path, audit_path, response_path, traces_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument(
        "--budget-points", type=int, default=0,
        help="0 uses the 13-point nonuniform saturation sweep; positive N uses N linear points from 0 to 1.35",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs")
    args = parser.parse_args()
    levels = (DEFAULT_HEADROOM if args.budget_points == 0
              else np.linspace(0.0, 1.35, args.budget_points))
    result = run_sweep(args.trials, args.duration, args.dt, levels)
    paths = save_outputs(*result, args.output_dir)
    print(json.dumps(result[3], ensure_ascii=False, indent=2))
    print("Saved:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
