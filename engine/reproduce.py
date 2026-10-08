"""Portable entry to the unchanged original 2048-run experiment."""

if not __debug__:
    raise RuntimeError('Verification requires assertions: do not use -O, -OO or PYTHONOPTIMIZE')
from pathlib import Path
import argparse
from datetime import datetime, timezone
import run_unified_revision_20261002 as experiment

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,help='A new/empty directory for the experiment')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=(args.out or root/'reports'/('multifidelity-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output must be new or empty: {output}')
    experiment.OUT=output
    experiment.main()

if __name__=='__main__':main()
