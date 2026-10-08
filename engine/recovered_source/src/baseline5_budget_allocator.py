#!/usr/bin/env python3
"""Baseline 5: measured compute-budget allocation across two modules.

The allocator benchmarks the four concrete model implementations, constructs a
per-exchange-step model budget, and selects the feasible fidelity combination
with the largest context-weighted utility.  The default budget deliberately
allows either module to upgrade but not both simultaneously, creating a real
resource conflict to resolve.

The measured times are Python wall-clock observations on the machine executing
the script.  They are not hard-real-time guarantees and should be reported with
the hardware/software environment and repeated-trial statistics.
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
    build_rom,
    speed_profile,
    storm_profile,
)


LEVELS = ((0, 0), (1, 0), (0, 1), (1, 1))


def benchmark_call(callable_, batches=20, calls_per_batch=500):
    """Return median and p90 execution time per call in microseconds."""
    for _ in range(50):
        callable_()
    samples = []
    for _ in range(batches):
        start = time.perf_counter_ns()
        for _ in range(calls_per_batch):
            callable_()
        samples.append((time.perf_counter_ns() - start) / calls_per_batch / 1000.0)
    return {
        "median_us": float(np.median(samples)),
        "p90_us": float(np.percentile(samples, 90)),
    }


def benchmark_models(rom, dt):
    h1, h2 = HydroL1(), HydroL2(rom, dt)
    p1, p2 = PropulsionL1(), PropulsionL2()
    return {
        "H-L1": benchmark_call(lambda: h1.step(80.0, 2.5, 8.0, 24.0, dt)),
        "H-L2": benchmark_call(lambda: h2.step(80.0, 2.5, 8.0, 24.0, dt)),
        "P-L1": benchmark_call(lambda: p1.step(2400.0, 24.0, dt)),
        "P-L2": benchmark_call(lambda: p2.step(2400.0, 24.0, dt)),
    }


class BudgetAllocator:
    """Enumerate the four fidelity combinations and maximize feasible utility."""

    def __init__(self, costs_us, budget_us, hydro_weight=1.0, propulsion_weight=0.85,
                 persistence_bonus=0.03, switch_penalty=0.10,
                 decision_interval_steps=5, min_dwell_steps=25):
        self.costs = costs_us
        self.budget_us = budget_us
        self.hydro_weight = hydro_weight
        self.propulsion_weight = propulsion_weight
        self.persistence_bonus = persistence_bonus
        self.switch_penalty = switch_penalty
        self.decision_interval_steps = decision_interval_steps
        self.min_dwell_steps = min_dwell_steps
        self.previous = (0, 0)
        self.steps_since_decision = decision_interval_steps
        self.steps_since_switch = min_dwell_steps

    def combination_cost(self, levels):
        h, p = levels
        return self.costs[f"H-L{h + 1}"] + self.costs[f"P-L{p + 1}"]

    def choose(self, hydro_utility, propulsion_utility):
        feasible = [levels for levels in LEVELS
                    if self.combination_cost(levels) <= self.budget_us]
        if not feasible:
            raise ValueError("Budget is below the measured all-L1 base cost")

        self.steps_since_decision += 1
        self.steps_since_switch += 1
        if (self.previous in feasible
                and self.steps_since_decision < self.decision_interval_steps):
            return self.previous
        self.steps_since_decision = 0

        # Release resources after the context has become unimportant.  A pure
        # persistence bonus would otherwise keep an unnecessary L2 allocation
        # forever once both utilities reach zero.
        if (hydro_utility <= 0.01 and propulsion_utility <= 0.01
                and (0, 0) in feasible
                and self.steps_since_switch >= self.min_dwell_steps):
            if self.previous != (0, 0):
                self.steps_since_switch = 0
            self.previous = (0, 0)
            return self.previous

        def score(levels):
            h, p = levels
            value = (h * self.hydro_weight * hydro_utility
                     + p * self.propulsion_weight * propulsion_utility)
            if levels == self.previous:
                value += self.persistence_bonus
            value -= self.switch_penalty * sum(
                int(new != old) for new, old in zip(levels, self.previous)
            )
            return value

        chosen = max(feasible, key=lambda levels: (score(levels), -self.combination_cost(levels)))
        if chosen != self.previous and self.steps_since_switch < self.min_dwell_steps:
            return self.previous
        if chosen != self.previous:
            self.steps_since_switch = 0
        self.previous = chosen
        return chosen


def context_utilities(H_s, speed_rate, previous_load_rate):
    hydro = float(np.clip((H_s - 0.8) / (3.5 - 0.8), 0.0, 1.0))
    propulsion = float(np.clip(max(abs(speed_rate) / 0.20,
                                   previous_load_rate / 0.15), 0.0, 1.0))
    return hydro, propulsion


def run_budgeted_simulation(duration_s=180.0, dt=0.2, budget_us=None,
                            benchmark_batches=20, calls_per_batch=500):
    t = np.arange(0.0, duration_s, dt)
    waves, speed = storm_profile(t), speed_profile(t)
    speed_rate = np.gradient(speed, dt)
    database_time = np.arange(0.0, max(duration_s, 180.0), dt)
    rom = build_rom(database_time)

    # Keep CLI options available for shorter CI checks while preserving the
    # repeated-batch measurement method used in the full experiment.
    global benchmark_call
    original_benchmark = benchmark_call
    if benchmark_batches != 20 or calls_per_batch != 500:
        def configured(callable_, batches=benchmark_batches, calls_per_batch=calls_per_batch):
            return original_benchmark(callable_, batches, calls_per_batch)
        benchmark_call = configured
    try:
        timings = benchmark_models(rom, dt)
    finally:
        benchmark_call = original_benchmark

    p90 = {name: value["p90_us"] for name, value in timings.items()}
    base = p90["H-L1"] + p90["P-L1"]
    h_increment = max(0.0, p90["H-L2"] - p90["H-L1"])
    p_increment = max(0.0, p90["P-L2"] - p90["P-L1"])
    if budget_us is None:
        budget_us = base + 1.15 * max(h_increment, p_increment)
        # Preserve the intended conflict even if timings are unusual.
        all_high = p90["H-L2"] + p90["P-L2"]
        budget_us = min(budget_us, np.nextafter(all_high, 0.0))

    allocator = BudgetAllocator(p90, budget_us)
    h_models, p_models = (HydroL1(), HydroL2(rom, dt)), (PropulsionL1(), PropulsionL2())

    n = len(t)
    resistance = np.empty(n)
    power, rpm, fuel = np.empty(n), np.empty(n), np.empty(n)
    h_level, p_level = np.empty(n, dtype=int), np.empty(n, dtype=int)
    estimated_cost, observed_cost = np.empty(n), np.empty(n)
    hydro_utility, prop_utility = np.empty(n), np.empty(n)
    conflict = np.zeros(n, dtype=int)
    previous_resistance = None
    previous_load_rate = 0.0
    previous_prop_output = None
    previous_p_level = 0

    for i, ti in enumerate(t):
        hu, pu = context_utilities(waves[i], speed_rate[i], previous_load_rate)
        hydro_utility[i], prop_utility[i] = hu, pu
        conflict[i] = int(hu > 0.0 and pu > 0.0)
        h_level[i], p_level[i] = allocator.choose(hu, pu)
        estimated_cost[i] = allocator.combination_cost((h_level[i], p_level[i]))

        if p_level[i] == 1 and previous_p_level == 0 and previous_prop_output is not None:
            p_models[1].initialise_from(previous_prop_output)

        start = time.perf_counter_ns()
        h_out = h_models[h_level[i]].step(ti, waves[i], 8.0, speed[i], dt)
        p_out = p_models[p_level[i]].step(h_out.resistance_kN, speed[i], dt)
        observed_cost[i] = (time.perf_counter_ns() - start) / 1000.0

        resistance[i], power[i] = h_out.resistance_kN, p_out.power_kW
        rpm[i], fuel[i] = p_out.shaft_rpm, p_out.fuel_kg_s
        if previous_resistance is not None:
            previous_load_rate = abs(resistance[i] - previous_resistance) / (
                max(abs(previous_resistance), 1.0) * dt
            )
        previous_resistance = resistance[i]
        previous_prop_output = p_out
        previous_p_level = p_level[i]

    allocation_counts = {
        f"H-L{h + 1}+P-L{p + 1}": int(np.sum((h_level == h) & (p_level == p)))
        for h, p in LEVELS
    }
    allocation_switches = int(np.sum(
        (np.diff(h_level) != 0) | (np.diff(p_level) != 0)
    ))
    summary = {
        "scope": "measured Python model-call budget; not a hard-real-time guarantee",
        "duration_s": duration_s,
        "time_step_s": dt,
        "steps": n,
        "benchmark": timings,
        "budget_us": float(budget_us),
        "all_L1_estimated_us": float(base),
        "all_L2_estimated_us": float(p90["H-L2"] + p90["P-L2"]),
        "allocation_counts": allocation_counts,
        "allocation_switches": allocation_switches,
        "context_conflict_steps": int(np.sum(conflict)),
        "context_conflict_fraction": float(np.mean(conflict)),
        "estimated_budget_violations": int(np.sum(estimated_cost > budget_us)),
        "observed_model_call_p50_us": float(np.percentile(observed_cost, 50)),
        "observed_model_call_p90_us": float(np.percentile(observed_cost, 90)),
        "observed_model_call_p99_us": float(np.percentile(observed_cost, 99)),
        "observed_over_budget_fraction": float(np.mean(observed_cost > budget_us)),
        "note": "Observed overruns include interpreter/OS jitter; allocation uses repeated-batch p90 costs.",
    }
    data = {
        "time_s": t, "Hs_m": waves, "speed_kn": speed,
        "hydro_utility": hydro_utility, "propulsion_utility": prop_utility,
        "hydro_fidelity": h_level, "propulsion_fidelity": p_level,
        "conflict": conflict, "estimated_cost_us": estimated_cost,
        "observed_cost_us": observed_cost, "resistance_kN": resistance,
        "power_kW": power, "shaft_rpm": rpm, "fuel_kg_s": fuel,
    }
    return data, summary


def save_outputs(data, summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "baseline5_budget_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))
    json_path = output_dir / "baseline5_budget_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    t = data["time_s"]
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(t, data["hydro_utility"], label="Hydrodynamics utility")
    axes[0].plot(t, data["propulsion_utility"], label="Propulsion utility")
    axes[0].fill_between(t, 0, 1, where=data["conflict"].astype(bool), alpha=0.10,
                         color="red", label="Competing requests")
    axes[0].set_ylabel("Utility")
    axes[0].legend(ncol=3, loc="upper left")

    axes[1].step(t, data["hydro_fidelity"], where="post", label="Hydro L2")
    axes[1].step(t, data["propulsion_fidelity"], where="post", label="Propulsion L2")
    axes[1].set_yticks([0, 1], ["L1", "L2"])
    axes[1].set_ylabel("Allocation")
    axes[1].legend(loc="upper left", ncol=2)

    axes[2].plot(t, data["estimated_cost_us"], label="Estimated allocated cost")
    axes[2].axhline(summary["budget_us"], color="red", linestyle="--", label="Budget")
    axes[2].set_ylabel("Estimated [us]")
    axes[2].legend(loc="upper left")

    axes[3].plot(t, data["observed_cost_us"], color="#6a4c93", linewidth=0.8)
    axes[3].axhline(summary["budget_us"], color="red", linestyle="--")
    axes[3].set_yscale("log")
    axes[3].set_ylabel("Observed [us]")
    axes[3].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Baseline 5: context utility under a measured model-call budget")
    fig.tight_layout()
    png_path = output_dir / "baseline5_budget_allocator.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return csv_path, json_path, png_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--budget-us", type=float)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs")
    args = parser.parse_args()
    data, summary = run_budgeted_simulation(args.duration, args.dt, args.budget_us)
    paths = save_outputs(data, summary, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
