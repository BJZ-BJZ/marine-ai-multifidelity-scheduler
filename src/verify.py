"""Recompute cluster means, paired differences and Holm from archived p-values."""

if not __debug__:
    raise RuntimeError('Verification requires assertions: do not use -O, -OO or PYTHONOPTIMIZE')
from pathlib import Path
from collections import Counter
import csv
import json
import numpy as np

DATA=Path(__file__).resolve().parents[1]/'data'
def read(name):
    with (DATA/name).open(encoding='utf-8') as stream:return list(csv.DictReader(stream))
raw=read('raw_runs.csv');direct=read('direct_all_l2.csv')
summary=read('summary.csv');paired=read('paired.csv');ablation=read('switching_ablation.csv')
report=json.loads((DATA/'report.json').read_text(encoding='utf-8'))
assert len(raw)==report['executed_runs']==2048 and len(direct)==report['direct_runs']==64
metrics=('composite_nrmse','elapsed_ms','switches','budget_violation_fraction','estimated_cost_us',
         'decision_latency_us','hydro_l2_share','propulsion_l2_share')
def clusters(policy,headroom,metric):
    values=[]
    for trial in range(1,9):
        selected=[r for r in raw if r['policy']==policy and float(r['headroom'])==headroom and int(r['trial'])==trial]
        assert len(selected)==8 and len({(r['scenario'],r['repeat']) for r in selected})==8
        values.append(np.mean([float(r[metric]) for r in selected]))
    return np.array(values)
checks=0
for row in summary:
    for metric in metrics:
        value=clusters(row['policy'],float(row['headroom']),metric).mean()
        assert np.isclose(value,float(row[metric+'_mean']),rtol=0,atol=1e-9)
        checks+=1
assert len(summary)==32
for family,key in [(paired,'p_holm_12'),(ablation,'p_holm_4')]:
    p=np.array([float(row['p_two_sided']) for row in family]);order=np.argsort(p)
    previous=0.;adjusted=np.zeros(len(p))
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,(len(p)-rank)*p[index]));adjusted[index]=previous
    for index,row in enumerate(family):
        h=float(row['headroom'])
        delta=(clusters('Proposed-Exact-Online',h,'composite_nrmse')-clusters(row['baseline'],h,'composite_nrmse'))*100
        assert np.isclose(delta.mean(),float(row['difference_pp']),rtol=0,atol=1e-9)
        assert int(np.sum(delta<0))==int(row['negative_clusters'])
        assert np.isclose(adjusted[index],float(row[key]),rtol=0,atol=1e-12)
direct_means=[]
for trial in range(1,9):
    group=[float(r['elapsed_ms']) for r in direct if int(r['trial'])==trial]
    assert len(group)==8;direct_means.append(np.mean(group))
assert np.isclose(np.mean(direct_means),report['direct_all_l2_ms']['mean'],rtol=0,atol=1e-9)
assert report['physical_validation'] is False and report['live_resource_enforcement'] is False
print(json.dumps(dict(status='PASS',policy_runs=2048,direct_timing_runs=64,independent_rom_clusters=8,
    mean_checks=checks,paired_contrasts=16,holm_families=[12,4],
    scope='Archived numerical statistics; t-test p-values and confidence intervals are not independently recomputed. No fresh simulation or timing.')))
