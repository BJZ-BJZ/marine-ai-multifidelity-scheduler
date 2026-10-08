import unittest
from run_unified_revision_20261002 import SwitchAwareGreedy, CostProfile, holm
from recovered_source.src.baseline7_online_allocation import SwitchAwareEnumerativeAllocator


class RevisionTests(unittest.TestCase):
    def test_holm_monotone(self):
        values = holm([.0015, .0313, .2124, .7815])
        self.assertAlmostEqual(values[0], .006)
        self.assertAlmostEqual(values[1], .0939)
        self.assertAlmostEqual(values[2], .4248)
        self.assertAlmostEqual(values[3], .7815)

    def test_greedy_dwell_and_budget(self):
        c = CostProfile(5, 27, 1, 11)
        g = SwitchAwareGreedy(c, min_dwell_steps=100, decision_interval_steps=5)
        self.assertEqual(g.choose((1., 1.), 40), (1, 1))
        self.assertEqual(g.choose((0., 0.), 40), (1, 1))
        self.assertEqual(g.choose((1., 1.), 6.1), (0, 0))

    def test_greedy_does_not_exchange_both_bits(self):
        c = CostProfile(5, 27, 1, 11)
        kw = dict(switch_penalty=.006, utilisation_penalty=0., min_dwell_steps=0,
                  decision_interval_steps=1)
        g, e = SwitchAwareGreedy(c, **kw), SwitchAwareEnumerativeAllocator(c, **kw)
        self.assertEqual(g.choose((.05, .10), 30), (0, 1))
        self.assertEqual(e.choose((.05, .10), 30), (0, 1))
        self.assertEqual(g.choose((.20, .10), 30), (0, 1))
        self.assertEqual(e.choose((.20, .10), 30), (1, 0))


if __name__ == '__main__':
    unittest.main()
