#!/usr/bin/env python3
"""Baseline 7: causal online allocation under a time-varying compute budget.

This experiment is deliberately separate from Baselines 4--6.  In particular,
the deployable policies in this file never read the all-L2 reference trajectory
when making a decision.  The reference is generated only for post-run scoring.

The online problem is a small multiple-choice resource-allocation problem.  At
each exchange step, the coordinator chooses one fidelity for hydrodynamics and
one for propulsion.  The choice maximises estimated error reduction minus
switching and compute-use penalties, subject to the currently available model-
call budget.  The exact policy enumerates the four combinations; its interface
is module-generic so a MILP solver can replace enumeration as the system grows.

Evidence boundary: the hydrodynamic ROM and propulsion parameters remain
synthetic/engineering estimates.  Results demonstrate scheduler mechanics and
causal resource allocation, not CFD, EFD, engine-bench, or sea-trial validation.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).parent))
from baseline2_rom import PODReducedOrderModel, build_snapshot_database  # noqa: E402
from baseline4_multidomain import (  # noqa: E402
    HydroL1,
    HydroL2,
    PropulsionL1,
    PropulsionL2,
    speed_profile,
    storm_profile,
)
from baseline5_budget_allocator import benchmark_models  # noqa: E402


MODULES = ("hydro", "propulsion")
LEVELS = tuple(itertools.product((0, 1), repeat=len(MODULES)))
DEPLOYABLE_STRATEGIES = (
    "Fixed-L1",
    "Fixed-L2",
    "Threshold-Projected",
    "Greedy-Online",
    "Proposed-Exact-Online",
)
ALL_STRATEGIES = DEPLOYABLE_STRATEGIES + ("Oracle-Myopic-Offline",)


@dataclass(frozen=True)
class CostProfile:
    """Measured per-call costs used by the online budget constraint."""

    hydro_l1: float
    hydro_l2: float
    propulsion_l1: float
    propulsion_l2: float

    def option_cost(self, module: str, level: int) -> float:
        if module == "hydro":
            return self.hydro_l1 if level == 0 else self.hydro_l2
        if module == "propulsion":
            return self.propulsion_l1 if level == 0 else self.propulsion_l2
        raise KeyError(module)

    def combination_cost(self, levels: tuple[int, int]) -> float:
        return sum(self.option_cost(module, level)
                   for module, level in zip(MODULES, levels))

    @property
    def base(self) -> float:
        return self.combination_cost((0, 0))

    @property
    def all_high(self) -> float:
        return self.combination_cost((1, 1))

    def as_dict(self) -> dict[str, float]:
        return {
            "H-L1": self.hydro_l1,
            "H-L2": self.hydro_l2,
            "P-L1": self.propulsion_l1,
            "P-L2": self.propulsion_l2,
        }


class RecursiveBenefitEstimator:
    """Causal recursive least-squares estimator of L2 error reduction.

    A target becomes observable only while L2 is selected.  The update therefore
    uses the current and past selected-model outputs; it never queries an unseen
    all-L2 trajectory.  The prior prevents a cold-start policy from assigning
    zero value to every upgrade before its first observation.
    """

    def __init__(self, prior: np.ndarray, forgetting: float = 0.995,
                 covariance: float = 12.0):
        self.theta = np.asarray(prior, dtype=float).copy()
        self.P = np.eye(len(self.theta)) * float(covariance)
        self.forgetting = float(forgetting)
        self.updates = 0

    def predict(self, features: np.ndarray) -> float:
        value = float(np.dot(self.theta, np.asarray(features, dtype=float)))
        return float(np.clip(value, 0.0, 0.50))

    def update(self, features: np.ndarray, target: float) -> None:
        x = np.asarray(features, dtype=float)
        px = self.P @ x
        gain = px / (self.forgetting + x @ px)
        residual = float(np.clip(target, 0.0, 0.50) - x @ self.theta)
        self.theta += gain * residual
        self.P = (self.P - np.outer(gain, x) @ self.P) / self.forgetting
        self.updates += 1


def causal_rate(values, dt):
    """Backward difference; no initial history means an initial rate of zero."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Finite one-dimensional observations and positive dt required")
    rate = np.zeros_like(values)
    rate[1:] = np.diff(values) / dt
    return rate


def hydro_features(t: float, hs: float, hs_rate: float, speed_kn: float,
                   wave_period: float = 8.0) -> np.ndarray:
    h = float(np.clip(hs / 3.5, 0.0, 1.5))
    omega_0 = 2.0 * np.pi / wave_period
    wave_number = omega_0**2 / 9.81
    encounter = omega_0 + wave_number * speed_kn * 0.514444
    phase_1 = abs(np.cos(encounter * t))
    phase_2 = abs(np.cos(2.0 * encounter * t + 0.3))
    return np.array([
        1.0,
        h,
        h * h,
        float(np.clip(abs(hs_rate) / 0.25, 0.0, 2.0)),
        float(np.clip(abs(speed_kn - 24.0) / 5.0, 0.0, 1.5)),
        phase_1,
        phase_2,
        h * h * phase_1,
        h**1.8 * phase_2,
    ])


def propulsion_features(speed_rate: float, previous_load_rate: float,
                        hydro_gap_estimate: float, speed_kn: float) -> np.ndarray:
    speed_offset = float(np.clip((speed_kn - 22.0) / 5.0, -0.5, 1.5))
    return np.array([
        1.0,
        float(np.clip(abs(speed_rate) / 0.20, 0.0, 2.0)),
        float(np.clip(previous_load_rate / 0.15, 0.0, 2.0)),
        float(np.clip(hydro_gap_estimate / 0.20, 0.0, 2.0)),
        speed_offset,
        speed_offset * speed_offset,
    ])


def make_cost_profile(timing: dict[str, dict[str, float]]) -> CostProfile:
    """Use batch-p90 timings; L2 includes a cheap L1 shadow error monitor."""

    p90 = {name: float(stats["p90_us"]) for name, stats in timing.items()}
    return CostProfile(
        hydro_l1=p90["H-L1"],
        hydro_l2=p90["H-L2"] + p90["H-L1"],
        propulsion_l1=p90["P-L1"],
        propulsion_l2=p90["P-L2"] + p90["P-L1"],
    )


def budget_landmarks(costs: CostProfile) -> dict[str, float]:
    """Budgets with known feasibility: base, cheaper single, either single, both."""

    single_h = costs.combination_cost((1, 0))
    single_p = costs.combination_cost((0, 1))
    all_high = costs.all_high
    return {
        "base": costs.base * 1.03,
        "low": min(single_h, single_p) * 1.03,
        "medium": min(max(single_h, single_p) * 1.03,
                      np.nextafter(all_high, 0.0)),
        "high": all_high * 1.05,
    }


def dynamic_budget_profile(t: np.ndarray, costs: CostProfile, seed: int):
    """Return a causal replay of capacity left by unrelated workstation tasks.

    The state changes are exogenous to the simulated ship and available to the
    scheduler at the current step.  Small seeded timing variation is clipped so
    each named band retains its intended feasibility class.
    """

    marks = budget_landmarks(costs)
    rng = np.random.default_rng(seed + 32452843)
    phase_shift = float(rng.uniform(-4.0, 4.0))
    phase = np.mod(t + phase_shift, 180.0)
    labels = np.full(len(t), "high", dtype=object)
    labels[(phase >= 30.0) & (phase < 65.0)] = "medium"
    labels[(phase >= 65.0) & (phase < 105.0)] = "low"
    labels[(phase >= 105.0) & (phase < 145.0)] = "medium"

    # Correlated variation represents background-load jitter without allowing a
    # nominal band to silently cross into another feasibility regime.
    raw = rng.normal(0.0, 0.006, len(t) + 14)
    jitter = np.convolve(raw, np.ones(15) / 15.0, mode="valid")
    budget = np.array([marks[str(label)] for label in labels], dtype=float)
    budget *= 1.0 + jitter
    budget = np.maximum(budget, costs.base * 1.001)
    return budget, labels, marks


class SwitchAwareEnumerativeAllocator:
    """Enumerate every feasible combination at one decision epoch.

    "Exact" applies only to the declared *single-epoch finite action set*.  The
    method is not a globally optimal policy over future time and its brute-force
    complexity is O(product_i L_i).  Baseline 7 has two binary modules, so only
    four combinations are examined.  The former class name remains as a
    compatibility alias below, but new paper text should use this explicit name.
    """

    def __init__(self, costs: CostProfile, switch_penalty: float = 0.006,
                 utilisation_penalty: float = 0.004,
                 decision_interval_steps: int = 5, min_dwell_steps: int = 15):
        self.costs = costs
        self.switch_penalty = float(switch_penalty)
        self.utilisation_penalty = float(utilisation_penalty)
        self.decision_interval_steps = int(decision_interval_steps)
        self.min_dwell_steps = int(min_dwell_steps)
        self.previous = (0, 0)
        self.steps_since_decision = self.decision_interval_steps
        self.steps_since_switch = self.min_dwell_steps

    def feasible(self, budget: float):
        return [levels for levels in LEVELS
                if self.costs.combination_cost(levels) <= budget + 1e-12]

    def choose(self, benefits: tuple[float, float], budget: float):
        feasible = self.feasible(budget)
        if not feasible:
            raise ValueError("Available budget is below the all-L1 base cost")

        self.steps_since_decision += 1
        self.steps_since_switch += 1
        previous_feasible = self.previous in feasible
        if (previous_feasible
                and self.steps_since_decision < self.decision_interval_steps):
            return self.previous
        self.steps_since_decision = 0

        # Dwell time is waived if the budget has made the old allocation
        # infeasible; budget safety takes precedence over switching smoothness.
        if previous_feasible and self.steps_since_switch < self.min_dwell_steps:
            return self.previous

        def score(levels):
            avoided_error = sum(level * benefit
                                for level, benefit in zip(levels, benefits))
            changes = sum(int(new != old)
                          for new, old in zip(levels, self.previous))
            utilisation = self.costs.combination_cost(levels) / max(budget, 1e-12)
            return (avoided_error
                    - self.switch_penalty * changes
                    - self.utilisation_penalty * utilisation)

        chosen = max(feasible,
                     key=lambda levels: (score(levels),
                                         -self.costs.combination_cost(levels)))
        if chosen != self.previous:
            self.steps_since_switch = 0
        self.previous = chosen
        return chosen


# Backward compatibility for Baseline 7/8 result reproduction and old tests.
ExactOnlineAllocator = SwitchAwareEnumerativeAllocator


def greedy_choice(costs: CostProfile, benefits: tuple[float, float], budget: float):
    levels = [0, 0]
    remaining = budget - costs.base
    upgrades = []
    for index, module in enumerate(MODULES):
        increment = (costs.option_cost(module, 1)
                     - costs.option_cost(module, 0))
        upgrades.append((benefits[index] / max(increment, 1e-12),
                         benefits[index], -increment, index, increment))
    for _, _, _, index, increment in sorted(upgrades, reverse=True):
        if increment <= remaining + 1e-12 and benefits[index] > 0.002:
            levels[index] = 1
            remaining -= increment
    return tuple(levels)


def threshold_projected_choice(costs: CostProfile, hs: float, speed_rate: float,
                               load_rate: float, budget: float):
    requested = [int(hs >= 1.5),
                 int(abs(speed_rate) >= 0.04 or load_rate >= 0.015)]
    if costs.combination_cost(tuple(requested)) <= budget + 1e-12:
        return tuple(requested)
    # Fixed hydro-first arbitration is intentionally simple and reproducible.
    for levels in ((requested[0], 0), (0, requested[1]), (0, 0)):
        if costs.combination_cost(levels) <= budget + 1e-12:
            return levels
    raise ValueError("Available budget is below the all-L1 base cost")


def build_trial_rom(database_time: np.ndarray, seed: int):
    snapshots, params = build_snapshot_database(
        database_time,
        speed_range=[18, 21, 24, 27],
        wave_range=[0.5, 1.5, 2.5, 3.5],
        period_range=[6, 8, 10],
        seed_offset=seed,
    )
    return PODReducedOrderModel(energy_threshold=0.99).fit(snapshots, params)


def trial_conditions(t: np.ndarray, seed: int):
    rng = np.random.default_rng(seed + 49979687)
    raw = rng.normal(0.0, 0.055, len(t) + 20)
    perturbation = np.convolve(raw, np.ones(21) / 21.0, mode="valid")
    waves = np.clip(storm_profile(t) + perturbation, 0.1, None)
    speed = speed_profile(t).copy()
    speed += 0.08 * np.sin(2.0 * np.pi * t / (35.0 + seed % 7))
    return waves, speed


def _ridge_prior(features: np.ndarray, targets: np.ndarray,
                 ridge: float = 0.25) -> np.ndarray:
    """Fit a stable offline prior on a calibration scenario, not a test trial."""

    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = ridge * 0.05
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def calibrate_benefit_priors(database_time: np.ndarray, dt: float):
    """Calibrate error-benefit priors on a disjoint seeded scenario.

    This is an offline model-identification stage analogous to fitting an error
    surrogate from historical runs.  Seed 7919 is never used by evaluation
    trials (which are numbered 1..N), so no test-trajectory value is exposed to
    a deployable policy.
    """

    seed = 7919
    t = np.arange(0.0, 180.0, dt)
    waves, speed = trial_conditions(t, seed)
    rom = build_trial_rom(database_time, seed)
    h1_model, h2_model = HydroL1(), HydroL2(rom, dt)
    p1_model, p2_model = PropulsionL1(), PropulsionL2()
    hs_rate, speed_rate = causal_rate(waves, dt), causal_rate(speed, dt)
    previous_resistance = None
    previous_load_rate = 0.0
    hx, hy, px, py = [], [], [], []

    for i, ti in enumerate(t):
        hf = hydro_features(ti, waves[i], hs_rate[i], speed[i])
        h1 = h1_model.step(ti, waves[i], 8.0, speed[i], dt)
        h2 = h2_model.step(ti, waves[i], 8.0, speed[i], dt)
        h_gap = _relative_gap(h2.resistance_kN, h1.resistance_kN)
        pf = propulsion_features(
            speed_rate[i], previous_load_rate, h_gap, speed[i])
        p1 = p1_model.step(h2.resistance_kN, speed[i], dt)
        p2 = p2_model.step(h2.resistance_kN, speed[i], dt)
        hx.append(hf)
        hy.append(h_gap)
        px.append(pf)
        py.append(_relative_gap(p2.power_kW, p1.power_kW))
        if previous_resistance is not None:
            previous_load_rate = abs(h2.resistance_kN - previous_resistance) / (
                max(abs(previous_resistance), 1.0) * dt)
        previous_resistance = h2.resistance_kN

    h_prior = _ridge_prior(np.asarray(hx), np.asarray(hy))
    p_prior = _ridge_prior(np.asarray(px), np.asarray(py))
    h_fit = np.clip(np.asarray(hx) @ h_prior, 0.0, 0.50)
    p_fit = np.clip(np.asarray(px) @ p_prior, 0.0, 0.50)
    diagnostics = {
        "calibration_seed": seed,
        "samples": len(t),
        "hydro_prior": h_prior.tolist(),
        "propulsion_prior": p_prior.tolist(),
        "hydro_gap_rmse": float(np.sqrt(np.mean((h_fit - hy) ** 2))),
        "propulsion_gap_rmse": float(np.sqrt(np.mean((p_fit - py) ** 2))),
    }
    return h_prior, p_prior, diagnostics


def run_all_l2_reference(t, waves, speed, rom, dt):
    """Generate the scoring reference outside every deployable policy."""

    hydro, propulsion = HydroL2(rom, dt), PropulsionL2()
    resistance, power = np.empty(len(t)), np.empty(len(t))
    for i, ti in enumerate(t):
        h = hydro.step(ti, waves[i], 8.0, speed[i], dt)
        p = propulsion.step(h.resistance_kN, speed[i], dt)
        resistance[i], power[i] = h.resistance_kN, p.power_kW
    return resistance, power


def offline_oracle_benefits(t, waves, speed, rom, dt,
                            reference_resistance, reference_power):
    """Non-deployable current-step information comparator.

    It is not a global sequence optimum because propulsion is stateful and the
    policy retains dwell constraints.  The label therefore avoids claiming a
    formal upper or lower performance bound.
    """

    h_low, p_low = HydroL1(), PropulsionL1()
    h_benefit, p_benefit = np.empty(len(t)), np.empty(len(t))
    for i, ti in enumerate(t):
        h1 = h_low.step(ti, waves[i], 8.0, speed[i], dt)
        p1 = p_low.step(reference_resistance[i], speed[i], dt)
        h_benefit[i] = abs(h1.resistance_kN - reference_resistance[i]) / max(
            abs(reference_resistance[i]), 1.0)
        p_benefit[i] = abs(p1.power_kW - reference_power[i]) / max(
            abs(reference_power[i]), 1.0)
    return h_benefit, p_benefit


def _relative_gap(high: float, low: float) -> float:
    return float(abs(high - low) / max(abs(high), 1.0))


def simulate_strategy(strategy: str, t: np.ndarray, waves: np.ndarray,
                      speed: np.ndarray, rom, dt: float, costs: CostProfile,
                      budgets: np.ndarray, oracle_benefits=None,
                      benefit_priors=None,
                      exact_config=None,
                      keep_trace: bool = False, monitor_unused: bool = False,
                      allocator_factory=None):
    """Sequentially execute a policy using only information available at step i."""

    hydro_models = HydroL1(), HydroL2(rom, dt)
    prop_models = PropulsionL1(), PropulsionL2()
    factory = allocator_factory or SwitchAwareEnumerativeAllocator
    allocator = factory(costs, **(exact_config or {}))
    if benefit_priors is None:
        benefit_priors = (
            np.array([0.002, 0.015, 0.035, 0.004, 0.006,
                      0.008, 0.006, 0.025, 0.015]),
            np.array([0.002, 0.018, 0.055, 0.018, 0.008, 0.004]),
        )
    hydro_estimator = RecursiveBenefitEstimator(benefit_priors[0])
    prop_estimator = RecursiveBenefitEstimator(benefit_priors[1])

    n = len(t)
    resistance, power = np.empty(n), np.empty(n)
    h_level, p_level = np.empty(n, dtype=int), np.empty(n, dtype=int)
    estimated_cost, estimated_benefit = np.empty(n), np.empty(n)
    decision_latency_us = np.empty(n)
    h_pred_log, p_pred_log = np.empty(n), np.empty(n)
    previous_resistance = None
    previous_load_rate = 0.0
    previous_prop_output = None
    previous_p_level = 0
    speed_rate = causal_rate(speed, dt)
    hs_rate = causal_rate(waves, dt)

    monitor = monitor_unused or strategy in ("Greedy-Online", "Proposed-Exact-Online")
    for i, ti in enumerate(t):
        hydro_pred = prop_pred = hydro_benefit = prop_benefit = 0.0
        if monitor:
            hf = hydro_features(ti, waves[i], hs_rate[i], speed[i])
            hydro_pred = hydro_estimator.predict(hf)
            pf = propulsion_features(
                speed_rate[i], previous_load_rate, hydro_pred, speed[i])
            prop_pred = prop_estimator.predict(pf)
            hydro_benefit = hydro_pred * (1.0 + 0.45 * np.clip(waves[i] / 3.5, 0, 1))
            prop_benefit = prop_pred * (1.0 + 0.35 * np.clip(abs(speed_rate[i]) / 0.2, 0, 1))

        decision_started = time.perf_counter_ns()
        if strategy == "Fixed-L1":
            levels = (0, 0)
        elif strategy == "Fixed-L2":
            levels = (1, 1)
        elif strategy == "Threshold-Projected":
            levels = threshold_projected_choice(
                costs, waves[i], speed_rate[i], previous_load_rate, budgets[i])
        elif strategy == "Greedy-Online":
            levels = greedy_choice(costs, (hydro_benefit, prop_benefit), budgets[i])
        elif strategy == "Proposed-Exact-Online":
            levels = allocator.choose((hydro_benefit, prop_benefit), budgets[i])
        elif strategy == "Oracle-Myopic-Offline":
            if oracle_benefits is None:
                raise ValueError("Offline oracle requires post-hoc reference benefits")
            levels = allocator.choose(
                (float(oracle_benefits[0][i]), float(oracle_benefits[1][i])),
                budgets[i],
            )
        else:
            raise KeyError(strategy)
        decision_latency_us[i] = (time.perf_counter_ns() - decision_started) / 1000.0
        h_level[i], p_level[i] = levels

        # Learning policies need the L1 shadow discrepancy. Other policies
        # omit it; the legacy CostProfile remains a conservative budget bound.
        if h_level[i] == 1:
            if monitor:
                h_low = hydro_models[0].step(ti, waves[i], 8.0, speed[i], dt)
            h_out = hydro_models[1].step(ti, waves[i], 8.0, speed[i], dt)
            if monitor:
                hydro_estimator.update(
                    hf, _relative_gap(h_out.resistance_kN, h_low.resistance_kN))
        else:
            h_out = hydro_models[0].step(ti, waves[i], 8.0, speed[i], dt)

        if p_level[i] == 1 and previous_p_level == 0 and previous_prop_output is not None:
            prop_models[1].initialise_from(previous_prop_output)
        if p_level[i] == 1:
            if monitor:
                p_low = prop_models[0].step(h_out.resistance_kN, speed[i], dt)
            p_out = prop_models[1].step(h_out.resistance_kN, speed[i], dt)
            if monitor:
                prop_estimator.update(
                    pf, _relative_gap(p_out.power_kW, p_low.power_kW))
        else:
            p_out = prop_models[0].step(h_out.resistance_kN, speed[i], dt)

        resistance[i], power[i] = h_out.resistance_kN, p_out.power_kW
        estimated_cost[i] = costs.combination_cost(levels)
        estimated_benefit[i] = levels[0] * hydro_benefit + levels[1] * prop_benefit
        h_pred_log[i], p_pred_log[i] = hydro_pred, prop_pred

        if previous_resistance is not None:
            previous_load_rate = abs(resistance[i] - previous_resistance) / (
                max(abs(previous_resistance), 1.0) * dt)
        previous_resistance = resistance[i]
        previous_prop_output = p_out
        previous_p_level = p_level[i]

    switches = int(np.sum((np.diff(h_level) != 0) | (np.diff(p_level) != 0)))
    result = {
        "monitoring_enabled": bool(monitor),
        "cost_accounting": "Legacy conservative call-budget estimate includes shadows even when omitted; use wall time for actual savings",
        "resistance": resistance,
        "power": power,
        "mean_cost_us": float(np.mean(estimated_cost)),
        "p95_cost_us": float(np.percentile(estimated_cost, 95)),
        "budget_violation_fraction": float(np.mean(estimated_cost > budgets + 1e-12)),
        "mean_budget_utilisation": float(np.mean(estimated_cost / budgets)),
        "hydro_l2_share": float(np.mean(h_level)),
        "propulsion_l2_share": float(np.mean(p_level)),
        "switches": switches,
        "mean_decision_latency_us": float(np.mean(decision_latency_us)),
        "p95_decision_latency_us": float(np.percentile(decision_latency_us, 95)),
        "hydro_estimator_updates": hydro_estimator.updates,
        "propulsion_estimator_updates": prop_estimator.updates,
    }
    if keep_trace:
        result["trace"] = {
            "time_s": t,
            "Hs_m": waves,
            "speed_kn": speed,
            "budget_us": budgets,
            "hydro_predicted_gap": h_pred_log,
            "propulsion_predicted_gap": p_pred_log,
            "hydro_fidelity": h_level,
            "propulsion_fidelity": p_level,
            "estimated_cost_us": estimated_cost,
            "estimated_selected_benefit": estimated_benefit,
            "resistance_kN": resistance,
            "power_kW": power,
        }
    return result


def tune_exact_policy(database_time: np.ndarray, dt: float, costs: CostProfile,
                      priors):
    """Select policy regularisation on the disjoint calibration scenario.

    The score balances end-to-end error, allocated compute, and switching.  It
    is evaluated only on seed 7919 and frozen before trials 1..N are generated.
    This prevents manual tuning on the reported test trials.
    """

    seed = 7919
    t = np.arange(0.0, 180.0, dt)
    waves, speed = trial_conditions(t, seed)
    rom = build_trial_rom(database_time, seed)
    budgets, _, _ = dynamic_budget_profile(t, costs, seed)
    reference_r, reference_p = run_all_l2_reference(t, waves, speed, rom, dt)
    candidates = []
    for utilisation_penalty in (0.0, 0.004, 0.015, 0.030, 0.050):
        for switch_penalty in (0.0, 0.003, 0.008):
            for decision_interval in (1, 5):
                for min_dwell in (0, 5, 15):
                    config = {
                        "utilisation_penalty": utilisation_penalty,
                        "switch_penalty": switch_penalty,
                        "decision_interval_steps": decision_interval,
                        "min_dwell_steps": min_dwell,
                    }
                    result = simulate_strategy(
                        "Proposed-Exact-Online", t, waves, speed, rom, dt, costs,
                        budgets, benefit_priors=priors, exact_config=config)
                    composite = 0.5 * (
                        nrmse(result["resistance"], reference_r)
                        + nrmse(result["power"], reference_p)
                    )
                    normalized_cost = result["mean_cost_us"] / costs.all_high
                    switch_rate = result["switches"] / len(t)
                    score = composite + 0.005 * normalized_cost + 0.01 * switch_rate
                    candidates.append({
                        "config": config,
                        "calibration_composite_nrmse": composite,
                        "calibration_normalized_cost": normalized_cost,
                        "calibration_switch_rate": switch_rate,
                        "calibration_selection_score": score,
                    })
    selected = min(candidates, key=lambda item: item["calibration_selection_score"])
    return selected["config"], {
        "selection_seed": seed,
        "selection_score": (
            "composite_nrmse + 0.005*(mean_cost/all_L2_cost) "
            "+ 0.01*(switches/steps)"
        ),
        "selected": selected,
        "candidate_count": len(candidates),
    }


def nrmse(values, reference):
    return float(np.sqrt(np.mean((values - reference) ** 2))
                 / max(abs(np.mean(reference)), 1e-12))


def confidence_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    sem = np.std(values, ddof=1) / np.sqrt(len(values))
    half = float(student_t.ppf((1.0 + confidence) / 2.0,
                               len(values) - 1) * sem)
    return mean, mean - half, mean + half


def paired_comparison(rows, proposed: str, baseline: str):
    """Paired trial differences; negative error/cost/switch values favour proposed."""

    proposed_rows = {int(row["trial"]): row for row in rows
                     if row["strategy"] == proposed}
    baseline_rows = {int(row["trial"]): row for row in rows
                     if row["strategy"] == baseline}
    trials = sorted(set(proposed_rows) & set(baseline_rows))
    metrics = ("resistance_nrmse", "power_nrmse", "composite_nrmse",
               "mean_cost_us", "budget_violation_fraction", "switches")
    result = {
        "proposed": proposed,
        "baseline": baseline,
        "paired_trials": len(trials),
        "difference_definition": "proposed minus baseline",
    }
    for metric in metrics:
        differences = [proposed_rows[trial][metric] - baseline_rows[trial][metric]
                       for trial in trials]
        mean, low, high = confidence_interval(differences)
        result[metric] = {"mean": mean, "ci95_low": low, "ci95_high": high}
    return result


def run_experiment(trials=30, duration_s=180.0, dt=0.2):
    t = np.arange(0.0, duration_s, dt)
    database_time = np.arange(0.0, max(duration_s, 180.0), dt)
    timing_rom = build_trial_rom(database_time, 0)
    timing = benchmark_models(timing_rom, dt)
    costs = make_cost_profile(timing)
    hydro_prior, propulsion_prior, calibration = calibrate_benefit_priors(
        database_time, dt)
    priors = hydro_prior, propulsion_prior
    exact_config, policy_tuning = tune_exact_policy(
        database_time, dt, costs, priors)
    rows = []
    representative = None

    for trial in range(1, trials + 1):
        rom = build_trial_rom(database_time, trial)
        waves, speed = trial_conditions(t, trial)
        budgets, budget_labels, marks = dynamic_budget_profile(t, costs, trial)
        reference_r, reference_p = run_all_l2_reference(t, waves, speed, rom, dt)
        oracle = offline_oracle_benefits(
            t, waves, speed, rom, dt, reference_r, reference_p)

        for strategy in ALL_STRATEGIES:
            result = simulate_strategy(
                strategy, t, waves, speed, rom, dt, costs, budgets,
                oracle_benefits=oracle,
                benefit_priors=priors,
                exact_config=exact_config,
                keep_trace=(trial == 1 and strategy == "Proposed-Exact-Online"),
            )
            resistance_error = nrmse(result["resistance"], reference_r)
            power_error = nrmse(result["power"], reference_p)
            row = {
                "trial": trial,
                "strategy": strategy,
                "resistance_nrmse": resistance_error,
                "power_nrmse": power_error,
                "composite_nrmse": 0.5 * (resistance_error + power_error),
                "mean_cost_us": result["mean_cost_us"],
                "p95_cost_us": result["p95_cost_us"],
                "budget_violation_fraction": result["budget_violation_fraction"],
                "mean_budget_utilisation": result["mean_budget_utilisation"],
                "hydro_l2_share": result["hydro_l2_share"],
                "propulsion_l2_share": result["propulsion_l2_share"],
                "switches": result["switches"],
            }
            rows.append(row)
            if "trace" in result:
                representative = result["trace"] | {
                    "budget_band": budget_labels,
                    "reference_resistance_kN": reference_r,
                    "reference_power_kW": reference_p,
                }

    metrics = tuple(key for key in rows[0]
                    if key not in ("trial", "strategy"))
    summaries = []
    for strategy in ALL_STRATEGIES:
        selected = [row for row in rows if row["strategy"] == strategy]
        summary = {"strategy": strategy, "trials": trials}
        for metric in metrics:
            mean, low, high = confidence_interval([row[metric] for row in selected])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_ci95_low"] = low
            summary[f"{metric}_ci95_high"] = high
        summaries.append(summary)

    comparisons = {
        baseline: paired_comparison(
            rows, "Proposed-Exact-Online", baseline)
        for baseline in ("Fixed-L1", "Threshold-Projected", "Greedy-Online")
    }

    metadata = {
        "scope": (
            "causal online scheduling experiment with synthetic/engineering "
            "models; All-L2 is an internal scoring reference only"
        ),
        "no_reference_leakage": True,
        "deployable_strategies": list(DEPLOYABLE_STRATEGIES),
        "offline_information_comparator": "Oracle-Myopic-Offline",
        "trials": trials,
        "duration_s": duration_s,
        "time_step_s": dt,
        "steps_per_trial": len(t),
        "timing_benchmark_raw": timing,
        "allocation_costs_p90_with_l1_shadow_us": costs.as_dict(),
        "budget_landmarks_us": budget_landmarks(costs),
        "benefit_prior_calibration": calibration,
        "exact_policy_calibration": policy_tuning,
        "objective": (
            "maximise predicted weighted error reduction minus switching and "
            "utilisation penalties subject to the current measured-cost budget"
        ),
        "solver_scope": (
            "single-epoch exhaustive enumeration of four actions; not a "
            "global-horizon exact optimiser"
        ),
        "limitations": [
            "synthetic POD source data",
            "engineering-estimate propulsion parameters",
            "two schedulable modules",
            "model-call wall-time budget rather than hard deadline enforcement",
            "no CFD/EFD/sea-trial accuracy validation",
        ],
        "paired_comparisons": comparisons,
    }
    return rows, summaries, metadata, representative


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(rows, summaries, metadata, trace, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "baseline7_online_trials.csv"
    summary_path = output_dir / "baseline7_online_summary.csv"
    json_path = output_dir / "baseline7_online_audit.json"
    trace_path = output_dir / "baseline7_online_trace.csv"
    figure_path = output_dir / "baseline7_online_allocation.png"
    write_csv(raw_path, rows)
    write_csv(summary_path, summaries)
    json_path.write_text(json.dumps({"metadata": metadata, "summary": summaries},
                                    indent=2, ensure_ascii=False), encoding="utf-8")

    trace_rows = []
    keys = list(trace)
    for values in zip(*(trace[key] for key in keys)):
        trace_rows.append(dict(zip(keys, values)))
    write_csv(trace_path, trace_rows)

    deployable = [row for row in summaries
                  if row["strategy"] in DEPLOYABLE_STRATEGIES]
    colors = ["#6c757d", "#212529", "#e9c46a", "#2a9d8f", "#e76f51"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    t = np.asarray(trace["time_s"])
    ax.plot(t, trace["budget_us"], color="#457b9d", label="Available budget")
    ax.plot(t, trace["estimated_cost_us"], color="#e76f51", label="Allocated cost")
    ax.set_ylabel("Model cost [us/step]")
    ax.set_title("Causal allocation remains within dynamic budget")
    ax.legend()

    ax = axes[0, 1]
    ax.step(t, trace["hydro_fidelity"], where="post", label="Hydrodynamics")
    ax.step(t, np.asarray(trace["propulsion_fidelity"]) + 0.05,
            where="post", label="Propulsion")
    ax.set_yticks([0, 1], ["L1", "L2"])
    ax.set_title("Independent online fidelity decisions")
    ax.legend()

    ax = axes[1, 0]
    for row, color in zip(deployable, colors):
        x = row["mean_cost_us_mean"]
        y = 100.0 * row["composite_nrmse_mean"]
        xerr = row["mean_cost_us_ci95_high"] - x
        yerr = 100.0 * (row["composite_nrmse_ci95_high"]
                        - row["composite_nrmse_mean"])
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", color=color,
                    capsize=3, label=row["strategy"])
    ax.set_xlabel("Mean allocated cost [us/step]")
    ax.set_ylabel("Composite NRMSE [%]")
    ax.set_title("Accuracy-cost trade-off (95% CI)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    x = np.arange(len(deployable))
    violation = [100.0 * row["budget_violation_fraction_mean"]
                 for row in deployable]
    ax.bar(x, violation, color=colors)
    ax.set_xticks(x, [row["strategy"] for row in deployable],
                  rotation=25, ha="right")
    ax.set_ylabel("Budget violations [% steps]")
    ax.set_title("Feasibility under changing capacity")

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        f"Baseline 7: causal online multi-fidelity allocation ({metadata['trials']} trials)"
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return raw_path, summary_path, json_path, trace_path, figure_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs")
    args = parser.parse_args()
    rows, summaries, metadata, trace = run_experiment(
        args.trials, args.duration, args.dt)
    paths = save_outputs(rows, summaries, metadata, trace, args.output_dir)
    print(json.dumps({"metadata": metadata, "summary": summaries},
                     indent=2, ensure_ascii=False))
    print("Saved:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
