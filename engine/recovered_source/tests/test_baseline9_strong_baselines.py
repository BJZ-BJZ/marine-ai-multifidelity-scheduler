import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline7_online_allocation import CostProfile  # noqa: E402
from baseline9_strong_baselines import (  # noqa: E402
    ContextualLearningPolicy,
    DiscountedLinearPosterior,
    offline_dp_actions,
)


class StrongBaselineTests(unittest.TestCase):
    def setUp(self):
        self.costs = CostProfile(5.0, 27.0, 1.0, 11.0)

    def test_discounted_model_learns_positive_target(self):
        model = DiscountedLinearPosterior(np.zeros(3), discount=0.99)
        x = np.array([1.0, 0.5, 0.2])
        before, _ = model.mean_and_variance(x)
        for _ in range(8):
            model.update(x, 0.2)
        after, variance = model.mean_and_variance(x)
        self.assertGreater(after, before)
        self.assertGreaterEqual(variance, 0.0)

    def test_learning_policies_are_budget_feasible(self):
        priors = np.zeros(3), np.zeros(3)
        for mode in ("linucb", "thompson"):
            policy = ContextualLearningPolicy(
                self.costs, priors, mode, seed=4, min_dwell_steps=0)
            for budget in (6.1, 16.1, 29.0, 40.0):
                levels, _, _ = policy.choose(np.ones(3), np.ones(3), budget)
                self.assertLessEqual(self.costs.combination_cost(levels), budget)

    def test_offline_dp_obeys_every_budget(self):
        hydro = np.linspace(0.0, 0.2, 20)
        propulsion = np.linspace(0.2, 0.0, 20)
        budgets = np.r_[np.full(10, 16.1), np.full(10, 40.0)]
        actions, _ = offline_dp_actions(
            hydro, propulsion, budgets, self.costs)
        for action, budget in zip(actions, budgets):
            self.assertLessEqual(
                self.costs.combination_cost(tuple(action)), budget)


if __name__ == "__main__":
    unittest.main()
