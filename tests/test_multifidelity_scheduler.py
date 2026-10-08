"""Multifidelity scheduler: verification entry points must refuse to run with assertions disabled."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = ['src/verify.py', 'engine/smoke.py', 'engine/verify_statistics.py', 'engine/reproduce.py']

class VerificationGuards(unittest.TestCase):
    def test_optimized_entry_points_refuse_to_report_success(self):
        for rel in SCRIPTS:
            path = ROOT / rel
            with self.subTest(path=rel):
                result = subprocess.run([sys.executable, '-O', str(path)],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Verification requires assertions', result.stderr)


if __name__ == '__main__':
    unittest.main()
