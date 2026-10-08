"""Recompute original cluster means, t intervals, p-values and Holm from raw runs."""

if not __debug__:
    raise RuntimeError('Verification requires assertions: do not use -O, -OO or PYTHONOPTIMIZE')
from pathlib import Path
import csv
import json
import numpy as np
from run_unified_revision_20261002 import analyze

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'data'

def read(name):
    with (DATA/name).open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        for key,value in list(row.items()):
            try:row[key]=float(value)
            except ValueError:pass
    return rows

def main():
    raw,direct=read('raw_runs.csv'),read('direct_all_l2.csv')
    assert len(raw)==2048 and len(direct)==64
    computed=analyze(raw,direct)
    checked=0; maximum=0.
    for label,filename in [('summary','summary.csv'),('paired','paired.csv'),('switching_ablation','switching_ablation.csv')]:
        saved=read(filename)
        assert len(saved)==len(computed[label])
        for expected,actual in zip(saved,computed[label]):
            assert set(expected)==set(actual)
            for key,value in actual.items():
                if isinstance(value,(float,int,np.number)):
                    error=abs(float(value)-expected[key]); maximum=max(maximum,error)
                    assert error<1e-9,(label,key,error)
                    checked+=1
                else:assert value==expected[key]
    saved=json.loads((DATA/'report.json').read_text(encoding='utf-8'))
    for key,value in computed['direct_all_l2_ms'].items():
        error=abs(value-saved['direct_all_l2_ms'][key]);maximum=max(maximum,error)
        assert error<1e-9;checked+=1
    report=dict(status='PASS_FULL_STATISTICS_RECOMPUTATION',raw_runs=len(raw),direct_runs=len(direct),
        numeric_values_checked=checked,max_absolute_error=maximum,
        scope='Original 8-ROM cluster means, Student t confidence intervals and paired p-values; two separate Holm families.')
    output=ROOT.parent/'reports';output.mkdir(exist_ok=True)
    (output/'multifidelity_full_statistics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
