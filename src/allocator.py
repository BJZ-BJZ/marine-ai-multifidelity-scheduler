"""Original cost profile and finite-action allocator, with portable imports."""
from dataclasses import dataclass
import itertools
MODULES=("hydro","propulsion")
LEVELS=tuple(itertools.product((0,1),repeat=2))

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
