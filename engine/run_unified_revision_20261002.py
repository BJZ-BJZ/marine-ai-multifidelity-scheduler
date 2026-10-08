"""Frozen-cost replication with contextual baselines and matched switching ablation.

Never overwrites archived raw runs. Model-call quotas are not wall-clock deadlines.
References are post-run scoring only; physical models are unchanged.
"""
from pathlib import Path
import csv
import hashlib
import json
import platform
import time
import numpy as np
from scipy.stats import t as student_t
from recovered_source.src.baseline7_online_allocation import (
    CostProfile, SwitchAwareEnumerativeAllocator, build_trial_rom,
    confidence_interval, nrmse, run_all_l2_reference, simulate_strategy,
)
from recovered_source.src.baseline8_budget_sweep import SCENARIOS, budget_trace, scenario_conditions
from recovered_source.src.baseline9_strong_baselines import simulate_learning_policy, tune_learning_policy

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs' / 'unified_revision_20261002'
POLICIES = ('Fixed-L1', 'Fixed-L2', 'Threshold-Projected', 'Greedy-Online',
            'Switch-Aware-Greedy', 'Discounted-LinUCB', 'Contextual-Thompson',
            'Proposed-Exact-Online')
HEADROOMS = (.35, .60, .90, 1.35)
TRIALS = tuple(range(1, 9))
REPEATS = 2
CONFIG = dict(switch_penalty=.006, utilisation_penalty=.004,
              decision_interval_steps=5, min_dwell_steps=15)


class SwitchAwareGreedy(SwitchAwareEnumerativeAllocator):
    """One-bit ascent with exactly the enumerator's score, cadence and dwell.

    Start from the retained feasible allocation, or all-L1 after a budget drop.
    Consider only immediate one-bit neighbours; no four-action maximization.
    Accept strictly positive score gains. Reference a fixed pre-decision state
    in every switching penalty so local search does not erase transition costs.
    """
    def choose(self, benefits, budget):
        if budget < self.costs.base - 1e-12:
            raise ValueError('Budget below all-L1 cost')
        self.steps_since_decision += 1
        self.steps_since_switch += 1
        previous_feasible = self.costs.combination_cost(self.previous) <= budget + 1e-12
        if previous_feasible and self.steps_since_decision < self.decision_interval_steps:
            return self.previous
        self.steps_since_decision = 0
        if previous_feasible and self.steps_since_switch < self.min_dwell_steps:
            return self.previous
        old = self.previous
        def score(action):
            return (sum(a*b for a, b in zip(action, benefits))
                    - self.switch_penalty * sum(a != b for a, b in zip(action, old))
                    - self.utilisation_penalty * self.costs.combination_cost(action) / budget)
        action = old if previous_feasible else (0, 0)
        # Four states and strict ascent imply at most three moves.
        for _ in range(3):
            neighbours = [tuple(1-v if i == j else v for j, v in enumerate(action))
                          for i in range(len(action))]
            neighbours = [a for a in neighbours
                          if self.costs.combination_cost(a) <= budget + 1e-12]
            if not neighbours:
                break
            best = max(neighbours, key=lambda a: (score(a), -self.costs.combination_cost(a)))
            if score(best) <= score(action) + 1e-12:
                break
            action = best
        if action != old:
            self.steps_since_switch = 0
        self.previous = action
        return action


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def cluster_values(rows, policy, headroom, metric):
    values = []
    for trial in TRIALS:
        selected = [float(r[metric]) for r in rows if r['policy'] == policy
                    and float(r['headroom']) == headroom and int(r['trial']) == trial]
        assert len(selected) == len(SCENARIOS) * REPEATS
        values.append(float(np.mean(selected)))
    return np.asarray(values)


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.zeros(len(order))
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(order)-rank)*pvalues[index]))
        adjusted[index] = previous
    return adjusted


def paired_record(rows, baseline, headroom):
    delta = (cluster_values(rows, POLICIES[-1], headroom, 'composite_nrmse')
             - cluster_values(rows, baseline, headroom, 'composite_nrmse')) * 100
    mean, low, high = confidence_interval(delta)
    se = float(np.std(delta, ddof=1)/np.sqrt(len(delta)))
    p = float(2*student_t.sf(abs(mean/se), len(delta)-1)) if se else (1. if mean == 0 else 0.)
    return dict(headroom=headroom, baseline=baseline, difference_pp=mean,
                ci95_low_pp=low, ci95_high_pp=high, p_two_sided=p,
                n_clusters=len(delta), negative_clusters=int(np.sum(delta < 0)))


def analyze(rows, direct):
    metrics = ('composite_nrmse', 'elapsed_ms', 'switches',
               'budget_violation_fraction', 'estimated_cost_us',
               'decision_latency_us', 'hydro_l2_share', 'propulsion_l2_share')
    summary = []
    for h in HEADROOMS:
        for policy in POLICIES:
            item = dict(headroom=h, policy=policy)
            for metric in metrics:
                mean, low, high = confidence_interval(cluster_values(rows, policy, h, metric))
                item.update({metric+'_mean': mean, metric+'_ci95_low': low, metric+'_ci95_high': high})
            summary.append(item)
    paired = [paired_record(rows, b, h) for b in ('Greedy-Online', 'Discounted-LinUCB',
                                               'Contextual-Thompson') for h in HEADROOMS]
    adjusted = holm([r['p_two_sided'] for r in paired])
    for row, p in zip(paired, adjusted):
        row['p_holm_12'] = float(p)
    ablation = [paired_record(rows, 'Switch-Aware-Greedy', h) for h in HEADROOMS]
    for row, p in zip(ablation, holm([r['p_two_sided'] for r in ablation])):
        row['p_holm_4'] = float(p)
    direct_means = [np.mean([r['elapsed_ms'] for r in direct if r['trial'] == trial]) for trial in TRIALS]
    dm, dl, dh = confidence_interval(direct_means)
    return dict(summary=summary, paired=paired, switching_ablation=ablation,
                direct_all_l2_ms=dict(mean=dm, ci95_low=dl, ci95_high=dh))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'raw_runs.csv').exists():
        raise FileExistsError('Existing raw data will not be overwritten')
    archived = json.loads((ROOT/'runs/multiscenario_corrected_20260924_v2/plan.json').read_text())
    c = archived['costs_us']
    costs = CostProfile(c['H-L1'], c['H-L2'], c['P-L1'], c['P-L2'])
    calibration = archived['prior_calibration']
    priors = (np.array(calibration['hydro_prior']), np.array(calibration['propulsion_prior']))
    t = np.arange(0., 180., .2)
    modes = {'Discounted-LinUCB': 'linucb', 'Contextual-Thompson': 'thompson'}
    tuned = {policy: tune_learning_policy(mode, t, .2, costs, priors)
             for policy, mode in modes.items()}
    plan = dict(policies=POLICIES, headrooms=HEADROOMS, trials=TRIALS,
                scenarios=SCENARIOS, repeats=REPEATS, duration_s=180., dt_s=.2,
                model='unchanged synthetic POD + engineering propulsion',
                costs_us=costs.as_dict(), cost_source='frozen 20260924 v2 plan, not fresh model timing',
                benefit_prior=calibration, proposed_config=CONFIG,
                learning_calibration={p: diag for p, (_, diag) in tuned.items()},
                learning_score_note='historical contextual policies retain raw local-gap scores; Greedy/Proposed use existing urgency weights',
                primary_comparisons='Proposed vs Greedy, LinUCB, Thompson across four budgets, Holm family of 12',
                ablation_comparisons='Proposed vs switch-aware Greedy across four budgets, separate Holm family of 4',
                inference='8 ROM clusters; 4 scenarios and 2 repeats averaged within each cluster',
                timing='whole simulation only; shared ROM training/calibration excluded and recorded separately',
                platform=platform.platform(), python=platform.python_version(),
                source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (Path(__file__), ROOT/'recovered_source/src/baseline7_online_allocation.py',
                                         ROOT/'recovered_source/src/baseline9_strong_baselines.py',
                                         ROOT/'recovered_source/src/baseline8_budget_sweep.py')})
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print('Protocol frozen before evaluation', flush=True)
    rows, direct = [], []
    rng = np.random.default_rng(20261002)
    for trial in TRIALS:
        rom = build_trial_rom(t, trial)
        for si, scenario in enumerate(SCENARIOS):
            seed = trial*101 + si*1009
            waves, speed = scenario_conditions(t, scenario, seed)
            rr, rp = run_all_l2_reference(t, waves, speed, rom, .2)
            cache = {h: budget_trace(t, costs, h, seed)[0] for h in HEADROOMS}
            order = [(repeat, h, p) for repeat in range(REPEATS)
                     for h in HEADROOMS for p in POLICIES]
            order += [(repeat, None, 'Direct-All-L2') for repeat in range(REPEATS)]
            rng.shuffle(order)
            for repeat, h, policy in order:
                started = time.perf_counter_ns()
                if policy == 'Direct-All-L2':
                    out_r, out_p = run_all_l2_reference(t, waves, speed, rom, .2)
                    elapsed = (time.perf_counter_ns()-started)/1e6
                    np.testing.assert_array_equal(out_r, rr)
                    np.testing.assert_array_equal(out_p, rp)
                    direct.append(dict(trial=trial, scenario=scenario, repeat=repeat, elapsed_ms=elapsed))
                    continue
                if policy in modes:
                    result = simulate_learning_policy(modes[policy], t, waves, speed, rom,
                                                      .2, costs, cache[h], priors, tuned[policy][0], seed)
                else:
                    result = simulate_strategy(
                        'Proposed-Exact-Online' if policy == 'Switch-Aware-Greedy' else policy,
                        t, waves, speed, rom, .2, costs, cache[h], benefit_priors=priors,
                        exact_config=CONFIG,
                        allocator_factory=SwitchAwareGreedy if policy == 'Switch-Aware-Greedy' else None)
                elapsed = (time.perf_counter_ns()-started)/1e6
                re, pe = nrmse(result['resistance'], rr), nrmse(result['power'], rp)
                rows.append(dict(trial=trial, scenario=scenario, repeat=repeat, headroom=h,
                                 policy=policy, resistance_nrmse=re, power_nrmse=pe,
                                 composite_nrmse=(re+pe)/2, elapsed_ms=elapsed,
                                 estimated_cost_us=result['mean_cost_us'],
                                 decision_latency_us=result['mean_decision_latency_us'],
                                 switches=result['switches'],
                                 budget_violation_fraction=result['budget_violation_fraction'],
                                 hydro_l2_share=result['hydro_l2_share'],
                                 propulsion_l2_share=result['propulsion_l2_share']))
            print(f'Completed trial {trial}/8 {scenario}', flush=True)
    write_csv(OUT/'raw_runs.csv', rows)
    write_csv(OUT/'direct_all_l2.csv', direct)
    report = analyze(rows, direct)
    report.update(plan=plan, executed_runs=len(rows), direct_runs=len(direct),
                  physical_validation=False, live_resource_enforcement=False)
    write_csv(OUT/'summary.csv', report['summary'])
    write_csv(OUT/'paired.csv', report['paired'])
    write_csv(OUT/'switching_ablation.csv', report['switching_ablation'])
    (OUT/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'runs':len(rows), 'direct':len(direct), 'output':str(OUT)}), flush=True)


if __name__ == '__main__':
    main()
