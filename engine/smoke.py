"""Fresh synthetic coupled simulation, using restored original research code."""

if not __debug__:
    raise RuntimeError('Verification requires assertions: do not use -O, -OO or PYTHONOPTIMIZE')
from pathlib import Path
import json
import time
import numpy as np
from run_unified_revision_20261002 import CONFIG, POLICIES, HEADROOMS, CostProfile, SwitchAwareGreedy
from recovered_source.src.baseline7_online_allocation import build_trial_rom, nrmse, run_all_l2_reference, simulate_strategy
from recovered_source.src.baseline8_budget_sweep import scenario_conditions, budget_trace
from recovered_source.src.baseline9_strong_baselines import simulate_learning_policy

ROOT=Path(__file__).resolve().parent

def main():
    plan=json.loads((ROOT.parent/'data/plan.json').read_text(encoding='utf-8'))
    c=plan['costs_us']; costs=CostProfile(c['H-L1'],c['H-L2'],c['P-L1'],c['P-L2'])
    prior=plan['benefit_prior']
    priors=(np.array(prior['hydro_prior']),np.array(prior['propulsion_prior']))
    t=np.arange(0.,180.,.2)
    # Fit new POD weights from the original generated 48-condition database.
    rom=build_trial_rom(t,1)
    waves,speed=scenario_conditions(t,'storm_passage',101)
    reference_r,reference_p=run_all_l2_reference(t,waves,speed,rom,.2)
    records=[]
    for headroom in HEADROOMS:
        budgets,_=budget_trace(t,costs,headroom,101)
        for policy in POLICIES:
            started=time.perf_counter()
            if policy in ('Discounted-LinUCB','Contextual-Thompson'):
                mode='linucb' if policy=='Discounted-LinUCB' else 'thompson'
                config=plan['learning_calibration'][policy]['selected']['config']
                result=simulate_learning_policy(mode,t,waves,speed,rom,.2,costs,budgets,priors,config,101)
            else:
                result=simulate_strategy(
                    'Proposed-Exact-Online' if policy=='Switch-Aware-Greedy' else policy,
                    t,waves,speed,rom,.2,costs,budgets,benefit_priors=priors,
                    exact_config=CONFIG,
                    allocator_factory=SwitchAwareGreedy if policy=='Switch-Aware-Greedy' else None)
            assert result['resistance'].shape==t.shape and result['power'].shape==t.shape
            assert np.isfinite(result['resistance']).all() and np.isfinite(result['power']).all()
            if policy!='Fixed-L2':
                assert result['budget_violation_fraction']==0.,(policy,headroom)
            else:
                np.testing.assert_allclose(result['resistance'],reference_r,rtol=0,atol=0)
                np.testing.assert_allclose(result['power'],reference_p,rtol=0,atol=0)
            error=(nrmse(result['resistance'],reference_r)+nrmse(result['power'],reference_p))/2
            records.append(dict(policy=policy,headroom=headroom,composite_nrmse=error,
                switches=result['switches'],budget_violation_fraction=result['budget_violation_fraction'],
                elapsed_ms=(time.perf_counter()-started)*1000))
    report=dict(status='PASS_FRESH_COUPLED_SIMULATION',steps=900,fresh_strategy_runs=len(records),
        scenario='storm_passage',ROM_training='fresh synthetic 48-condition POD database, seed 1',
        reused='original frozen call costs, benefit priors and learning hyperparameters',
        physical_validation=False,records=records,
        scope='One ROM and one scenario; execution check, not a new statistical replication of 2048 runs.')
    output=ROOT.parent/'reports';output.mkdir(exist_ok=True)
    (output/'multifidelity_fresh_smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

if __name__=='__main__':main()
