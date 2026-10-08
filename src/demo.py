"""Synthetic budget-drop demonstration, not a new physical simulation."""
import json
from allocator import CostProfile,SwitchAwareEnumerativeAllocator

costs=CostProfile(1.,4.,1.,4.)
allocator=SwitchAwareEnumerativeAllocator(costs,decision_interval_steps=5,min_dwell_steps=15)
first=allocator.choose((1.,1.),8.)
second=allocator.choose((1.,1.),2.)
assert first==(1,1) and second==(0,0)
assert costs.combination_cost(second)<=2.
try:allocator.choose((1.,1.),1.)
except ValueError:rejected=True
else:rejected=False
assert rejected
print(json.dumps(dict(status='PASS',input='Synthetic benefits and model-call quotas',first=first,budget_drop=second,
    infeasible_budget_rejected=True,scope='Finite-action mechanics; no OS resource enforcement or wall-clock guarantee.')))
