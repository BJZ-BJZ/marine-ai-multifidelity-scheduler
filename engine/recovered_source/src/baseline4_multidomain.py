#!/usr/bin/env python3
"""Baseline 4: two-module adaptive multi-fidelity co-simulation.

This executable prototype couples a variable-speed hydrodynamic resistance
module to a propulsion module.  Each module has its own two-level fidelity
decision, so the experiment tests *independent* allocation rather than merely
renaming one global model switch.

Important scope statement: the POD snapshots and propulsion parameters are
synthetic/engineering estimates.  The experiment demonstrates scheduler and
state-transfer behaviour; it is not CFD-, engine-bench-, or sea-trial-validated.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from baseline1_lf import (  # noqa: E402
    heave_response_amplitude,
    holtrop_mennen_total_resistance,
    pitch_response_amplitude,
    wave_elevation,
)
from baseline2_rom import (  # noqa: E402
    PODReducedOrderModel,
    build_snapshot_database,
)


L_PP, B, DRAFT = 230.0, 32.2, 10.8
C_B, C_M = 0.6505, 0.9849
RHO, NU, GRAVITY = 1025.0, 1.19e-6, 9.81
A_M, DISPLACEMENT_VOLUME = B * DRAFT * C_M, 52030.0


@dataclass
class HydroOutput:
    resistance_kN: float
    heave_m: float
    pitch_deg: float


@dataclass
class PropulsionOutput:
    power_kW: float
    shaft_rpm: float
    fuel_kg_s: float


def _blend_dataclass(old, new, weight):
    values = {
        key: (1.0 - weight) * getattr(old, key) + weight * getattr(new, key)
        for key in asdict(old)
    }
    return type(old)(**values)


class HydroL1:
    """Quasi-steady Holtrop-style resistance plus linear motions."""

    name = "H-L1"

    def step(self, t, H_s, T_p, speed_knots, dt):
        del dt
        speed_ms = speed_knots * 0.514444
        resistance, _, _ = holtrop_mennen_total_resistance(
            speed_ms, L_PP, B, DRAFT, C_B, C_M, A_M,
            DISPLACEMENT_VOLUME, RHO, NU, GRAVITY,
        )
        zeta = wave_elevation(t, H_s, T_p, "head")
        omega = 2.0 * np.pi / T_p
        added = 0.5 * RHO * GRAVITY * B * zeta**2 * 0.1
        return HydroOutput(
            resistance_kN=(resistance + added) / 1000.0,
            heave_m=heave_response_amplitude(zeta, omega, speed_ms, L_PP),
            pitch_deg=pitch_response_amplitude(zeta, omega, speed_ms, L_PP),
        )


class HydroL2:
    """POD resistance surrogate with the same exchange interface as H-L1."""

    name = "H-L2"

    def __init__(self, rom, database_dt=0.2):
        self.rom = rom
        self.database_dt = database_dt

    def step(self, t, H_s, T_p, speed_knots, dt):
        del dt
        prediction, _ = self.rom.predict(speed_knots, H_s, T_p)
        idx = min(int(round(t / self.database_dt)), len(prediction) - 1)
        zeta = wave_elevation(t, H_s, T_p, "head")
        omega = 2.0 * np.pi / T_p
        speed_ms = speed_knots * 0.514444
        return HydroOutput(
            resistance_kN=float(prediction[idx]),
            heave_m=heave_response_amplitude(zeta, omega, speed_ms, L_PP),
            pitch_deg=pitch_response_amplitude(zeta, omega, speed_ms, L_PP),
        )


class PropulsionL1:
    """Instantaneous propulsive balance with fixed overall efficiency."""

    name = "P-L1"

    def __init__(self, efficiency=0.70, design_power_kW=40000.0,
                 design_rpm=105.0, sfoc_g_kWh=185.0):
        self.efficiency = efficiency
        self.design_power_kW = design_power_kW
        self.design_rpm = design_rpm
        self.sfoc_g_kWh = sfoc_g_kWh

    def step(self, resistance_kN, speed_knots, dt):
        del dt
        speed_ms = speed_knots * 0.514444
        power = max(1.0, resistance_kN * speed_ms / self.efficiency)
        rpm = self.design_rpm * (power / self.design_power_kW) ** (1.0 / 3.0)
        fuel = power * self.sfoc_g_kWh / 3.6e6
        return PropulsionOutput(power, rpm, fuel)


class PropulsionL2:
    """Stateful mean-value shaft/engine model with load-dependent efficiency."""

    name = "P-L2"

    def __init__(self, design_power_kW=40000.0, design_rpm=105.0,
                 tau_up_s=8.0, tau_down_s=12.0, tau_rpm_s=5.0):
        self.design_power_kW = design_power_kW
        self.design_rpm = design_rpm
        self.tau_up_s = tau_up_s
        self.tau_down_s = tau_down_s
        self.tau_rpm_s = tau_rpm_s
        self.power_kW = None
        self.shaft_rpm = None

    def initialise_from(self, output):
        self.power_kW = float(output.power_kW)
        self.shaft_rpm = float(output.shaft_rpm)

    def step(self, resistance_kN, speed_knots, dt):
        speed_ms = speed_knots * 0.514444
        preliminary_load = np.clip(
            resistance_kN * speed_ms / (0.70 * self.design_power_kW), 0.05, 1.20
        )
        efficiency = 0.64 + 0.08 * np.exp(-((preliminary_load - 0.78) / 0.30) ** 2)
        demand = max(1.0, resistance_kN * speed_ms / efficiency)
        target_rpm = self.design_rpm * (demand / self.design_power_kW) ** (1.0 / 3.0)

        if self.power_kW is None:
            self.power_kW, self.shaft_rpm = demand, target_rpm
        tau = self.tau_up_s if demand >= self.power_kW else self.tau_down_s
        self.power_kW += min(dt / tau, 1.0) * (demand - self.power_kW)
        self.shaft_rpm += min(dt / self.tau_rpm_s, 1.0) * (target_rpm - self.shaft_rpm)

        load = np.clip(self.power_kW / self.design_power_kW, 0.05, 1.20)
        sfoc = 178.0 + 35.0 * (load - 0.78) ** 2
        fuel = self.power_kW * sfoc / 3.6e6
        return PropulsionOutput(self.power_kW, self.shaft_rpm, fuel)


class HydroScheduler:
    """Wave-context scheduler, independent of the propulsion decision."""

    def __init__(self, low, high, check_interval=2.0, high_threshold=1.5,
                 low_threshold=0.8, transition_steps=10):
        self.low, self.high, self.active = low, high, low
        self.check_interval = check_interval
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.transition_steps = transition_steps
        self.elapsed = 0.0
        self.high_count = self.low_count = 0
        self.old = None
        self.transition_counter = 0
        self.events = []

    @property
    def level(self):
        return 1 if self.active is self.high else 0

    def _switch(self, target, t, context):
        self.old, self.active = self.active, target
        self.transition_counter = self.transition_steps
        self.events.append({"time_s": round(float(t), 3), "from": self.old.name,
                            "to": target.name, "context": round(float(context), 4)})

    def step(self, t, H_s, T_p, speed_knots, dt):
        self.elapsed += dt
        if self.elapsed + 1e-12 >= self.check_interval:
            self.elapsed = 0.0
            self.high_count = self.high_count + 1 if H_s >= self.high_threshold else 0
            self.low_count = self.low_count + 1 if H_s <= self.low_threshold else 0
            if self.active is self.low and self.high_count >= 2:
                self._switch(self.high, t, H_s)
            elif self.active is self.high and self.low_count >= 5:
                self._switch(self.low, t, H_s)

        if self.transition_counter:
            old = self.old.step(t, H_s, T_p, speed_knots, dt)
            new = self.active.step(t, H_s, T_p, speed_knots, dt)
            completed = self.transition_steps - self.transition_counter + 1
            self.transition_counter -= 1
            return _blend_dataclass(old, new, completed / self.transition_steps)
        return self.active.step(t, H_s, T_p, speed_knots, dt)


class PropulsionScheduler:
    """Speed-transient scheduler with explicit state transfer to P-L2."""

    def __init__(self, low, high, check_interval=1.0, high_rate=0.04,
                 low_rate=0.005, high_load_rate=0.015,
                 low_load_rate=0.003, transition_steps=10):
        self.low, self.high, self.active = low, high, low
        self.check_interval = check_interval
        self.high_rate = high_rate
        self.low_rate = low_rate
        self.high_load_rate = high_load_rate
        self.low_load_rate = low_load_rate
        self.transition_steps = transition_steps
        self.elapsed = 0.0
        self.high_count = self.low_count = 0
        self.old = None
        self.transition_counter = 0
        self.events = []
        self.previous_resistance = None

    @property
    def level(self):
        return 1 if self.active is self.high else 0

    def _switch(self, target, t, rate, resistance, speed, dt):
        self.old = self.active
        if target is self.high:
            self.high.initialise_from(self.low.step(resistance, speed, dt))
        self.active = target
        self.transition_counter = self.transition_steps
        self.events.append({"time_s": round(float(t), 3), "from": self.old.name,
                            "to": target.name, "context": round(float(rate), 5)})

    def step(self, t, resistance_kN, speed_knots, speed_rate_kn_s, dt):
        self.elapsed += dt
        rate = abs(speed_rate_kn_s)
        if self.previous_resistance is None:
            load_rate = 0.0
        else:
            load_rate = abs(resistance_kN - self.previous_resistance) / (
                max(abs(self.previous_resistance), 1.0) * dt
            )
        self.previous_resistance = resistance_kN
        if self.elapsed + 1e-12 >= self.check_interval:
            self.elapsed = 0.0
            demanding = rate >= self.high_rate or load_rate >= self.high_load_rate
            steady = rate <= self.low_rate and load_rate <= self.low_load_rate
            self.high_count = self.high_count + 1 if demanding else 0
            self.low_count = self.low_count + 1 if steady else 0
            if self.active is self.low and self.high_count >= 2:
                context = max(rate / self.high_rate, load_rate / self.high_load_rate)
                self._switch(self.high, t, context, resistance_kN, speed_knots, dt)
            elif self.active is self.high and self.low_count >= 5:
                context = max(rate / self.high_rate, load_rate / self.high_load_rate)
                self._switch(self.low, t, context, resistance_kN, speed_knots, dt)

        if self.transition_counter:
            old = self.old.step(resistance_kN, speed_knots, dt)
            new = self.active.step(resistance_kN, speed_knots, dt)
            completed = self.transition_steps - self.transition_counter + 1
            self.transition_counter -= 1
            return _blend_dataclass(old, new, completed / self.transition_steps)
        return self.active.step(resistance_kN, speed_knots, dt)


def storm_profile(t):
    H_s = np.full_like(t, 0.5, dtype=float)
    up = (t > 60.0) & (t <= 75.0)
    H_s[up] = 0.5 + 3.0 * (t[up] - 60.0) / 15.0
    H_s[(t > 75.0) & (t <= 105.0)] = 3.5
    down = (t > 105.0) & (t <= 120.0)
    H_s[down] = 3.5 - 3.0 * (t[down] - 105.0) / 15.0
    H_s += 0.1 * np.sin(2.0 * np.pi * t / 20.0) * np.minimum(H_s, 1.0)
    return H_s


def speed_profile(t):
    speed = np.full_like(t, 24.0, dtype=float)
    up = (t > 35.0) & (t <= 55.0)
    speed[up] = 24.0 + 2.0 * (t[up] - 35.0) / 20.0
    speed[(t > 55.0) & (t <= 105.0)] = 26.0
    down = (t > 105.0) & (t <= 125.0)
    speed[down] = 26.0 - 4.0 * (t[down] - 105.0) / 20.0
    speed[t > 125.0] = 22.0
    return speed


def build_rom(database_time):
    snapshots, params = build_snapshot_database(
        database_time,
        speed_range=[18, 21, 24, 27],
        wave_range=[0.5, 1.5, 2.5, 3.5],
        period_range=[6, 8, 10],
    )
    return PODReducedOrderModel(energy_threshold=0.99).fit(snapshots, params)


def run_simulation(duration_s=180.0, dt=0.2, T_p=8.0):
    t = np.arange(0.0, duration_s, dt)
    H_s = storm_profile(t)
    speed = speed_profile(t)
    speed_rate = np.gradient(speed, dt)

    database_time = np.arange(0.0, max(duration_s, 180.0), dt)
    rom = build_rom(database_time)
    hydro = HydroScheduler(HydroL1(), HydroL2(rom, dt))
    prop = PropulsionScheduler(PropulsionL1(), PropulsionL2())

    resistance = np.empty_like(t)
    power = np.empty_like(t)
    rpm = np.empty_like(t)
    fuel = np.empty_like(t)
    hydro_level = np.empty_like(t, dtype=int)
    prop_level = np.empty_like(t, dtype=int)

    for i, ti in enumerate(t):
        h = hydro.step(ti, H_s[i], T_p, speed[i], dt)
        p = prop.step(ti, h.resistance_kN, speed[i], speed_rate[i], dt)
        resistance[i], power[i] = h.resistance_kN, p.power_kW
        rpm[i], fuel[i] = p.shaft_rpm, p.fuel_kg_s
        hydro_level[i], prop_level[i] = hydro.level, prop.level

    # Counterfactual all-L2 propulsion reference driven by the same resistance.
    reference_model = PropulsionL2()
    reference_power = np.empty_like(t)
    for i in range(len(t)):
        reference_power[i] = reference_model.step(
            resistance[i], speed[i], dt
        ).power_kW

    summary = {
        "scope": "two-module scheduler demonstration using synthetic ROM and estimated propulsion parameters",
        "duration_s": duration_s,
        "time_step_s": dt,
        "steps": len(t),
        "hydro_L1_fraction": float(np.mean(hydro_level == 0)),
        "hydro_L2_fraction": float(np.mean(hydro_level == 1)),
        "propulsion_L1_fraction": float(np.mean(prop_level == 0)),
        "propulsion_L2_fraction": float(np.mean(prop_level == 1)),
        "hydro_switch_events": hydro.events,
        "propulsion_switch_events": prop.events,
        "adaptive_vs_all_L2_power_nrmse": float(
            np.sqrt(np.mean((power - reference_power) ** 2)) / np.mean(reference_power)
        ),
        "fuel_consumed_kg": float(np.trapezoid(fuel, t)),
    }
    data = {
        "time_s": t, "Hs_m": H_s, "speed_kn": speed,
        "speed_rate_kn_s": speed_rate, "resistance_kN": resistance,
        "power_kW": power, "all_L2_power_kW": reference_power,
        "shaft_rpm": rpm, "fuel_kg_s": fuel,
        "hydro_fidelity": hydro_level, "propulsion_fidelity": prop_level,
    }
    return data, summary


def save_outputs(data, summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "baseline4_multidomain_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))

    json_path = output_dir / "baseline4_multidomain_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    t = data["time_s"]
    fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True)
    axes[0].plot(t, data["Hs_m"], color="#277da1", label="Hs")
    axes[0].step(t, data["hydro_fidelity"] * 3.5, where="post", color="#f94144",
                 alpha=0.65, label="Hydro L2 active (scaled)")
    axes[0].set_ylabel("Hs [m]")
    axes[0].legend(loc="upper left", ncol=2)

    axes[1].plot(t, data["speed_kn"], color="#43aa8b", label="Speed command")
    axes[1].step(t, 21.5 + data["propulsion_fidelity"] * 5.0, where="post",
                 color="#f3722c", alpha=0.75, label="Propulsion L2 active")
    axes[1].set_ylabel("Speed [kn]")
    axes[1].legend(loc="upper left", ncol=2)

    axes[2].plot(t, data["resistance_kN"], color="#577590")
    axes[2].set_ylabel("Resistance [kN]")

    axes[3].plot(t, data["power_kW"] / 1000.0, label="Adaptive", color="#9b5de5")
    axes[3].plot(t, data["all_L2_power_kW"] / 1000.0, "--", label="All P-L2 reference",
                 color="#22223b", linewidth=1.0)
    axes[3].set_ylabel("Power [MW]")
    axes[3].legend(loc="upper left")

    axes[4].plot(t, data["fuel_kg_s"], color="#f8961e")
    axes[4].set_ylabel("Fuel [kg/s]")
    axes[4].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Baseline 4: independently scheduled hydrodynamics and propulsion")
    fig.tight_layout()
    png_path = output_dir / "baseline4_multidomain_scheduler.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return csv_path, json_path, png_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs")
    args = parser.parse_args()
    data, summary = run_simulation(args.duration, args.dt)
    paths = save_outputs(data, summary, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
