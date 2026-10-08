import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline7_online_allocation import (  # noqa: E402
    CostProfile,
    ExactOnlineAllocator,
    RecursiveBenefitEstimator,
    budget_landmarks,
    dynamic_budget_profile,
    greedy_choice,
)


class OnlineAllocationTests(unittest.TestCase):
    def setUp(self):
        self.costs = CostProfile(5.0, 27.0, 1.0, 11.0)

    def test_exact_allocator_never_returns_infeasible_combination(self):
        allocator = ExactOnlineAllocator(
            self.costs, decision_interval_steps=1, min_dwell_steps=0)
        for budget in (6.1, 16.0, 29.0, 40.0):
            for benefits in ((0.0, 0.0), (0.2, 0.1), (0.1, 0.3)):
                levels = allocator.choose(benefits, budget)
                self.assertLessEqual(self.costs.combination_cost(levels), budget)

    def test_budget_drop_overrides_dwell(self):
        allocator = ExactOnlineAllocator(
            self.costs, decision_interval_steps=1, min_dwell_steps=100)
        self.assertEqual(allocator.choose((1.0, 1.0), 40.0), (1, 1))
        levels = allocator.choose((1.0, 1.0), 6.1)
        self.assertEqual(levels, (0, 0))

    def test_greedy_respects_budget(self):
        for budget in (6.1, 16.0, 29.0, 40.0):
            levels = greedy_choice(self.costs, (0.2, 0.3), budget)
            self.assertLessEqual(self.costs.combination_cost(levels), budget)

    def test_dynamic_budget_contains_binding_bands(self):
        t = np.arange(0.0, 180.0, 0.2)
        budget, labels, marks = dynamic_budget_profile(t, self.costs, seed=1)
        self.assertTrue(set(("low", "medium", "high")).issubset(set(labels)))
        self.assertTrue(np.all(budget >= self.costs.base))
        self.assertLess(marks["medium"], self.costs.all_high)
        self.assertGreater(marks["high"], self.costs.all_high)

    def test_recursive_estimator_updates_from_observation(self):
        estimator = RecursiveBenefitEstimator(np.zeros(3))
        features = np.array([1.0, 0.5, 0.25])
        before = estimator.predict(features)
        estimator.update(features, 0.2)
        after = estimator.predict(features)
        self.assertEqual(estimator.updates, 1)
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
