"""E4 — sweep the explore band constants on the explore harness.

The spec fixes each constant's ROLE and direction; this script fixes the
VALUE by measurement. It runs the explore harness (tests/recall) once per
grid point with the module constants patched, and prints one row per point:
differential, adj MRR, adj recall, precision, divergence, empty count.
Nothing is written back — read the table, pick, then edit ranking.py.

Usage (from the repo root, inside the venv):
    python tools/experiments/e4-explore-rescale/sweep.py
    python tools/experiments/e4-explore-rescale/sweep.py --offset 0.3 0.4 --scale 0.5 0.7 --floor 0.8 1.0
"""
from __future__ import annotations

import argparse
import itertools
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import logging
from starlette.testclient import TestClient  # noqa: E402

# Same shim as tests/conftest.py: in-process calls are loopback traffic.
_orig_init = TestClient.__init__
TestClient.__init__ = lambda self, *a, **kw: _orig_init(self, *a, **{"client": ("127.0.0.1", 50000), **kw})
logging.disable(logging.INFO)

from agentb import ranking  # noqa: E402
from agentb.config import RankingConfig  # noqa: E402
from tests.recall.embed_fixtures import load_cache, load_explore_fixtures, load_fixtures  # noqa: E402
from tests.recall.test_explore_harness import run_explore_harness, summarize  # noqa: E402
from tests.recall.test_recall_harness import _make_client, _seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    # Defaults = the union of the two grids swept on 2026-09-02 (coarse 5×5×3,
    # then fine 7×4×3 around the middle; 159 points, tables in RESULTS.md),
    # so a re-run lands on the shipped point (0.05 / 0.30 / 0.80) and the
    # runner-up (floor 0.90). 8×8×5 = 320 runs, ~15 minutes.
    ap.add_argument("--offset", nargs="+", type=float,
                    default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5])
    ap.add_argument("--scale", nargs="+", type=float,
                    default=[0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0])
    ap.add_argument("--floor", nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.9, 1.0])
    args = ap.parse_args()

    fixtures, explore, vectors = load_fixtures(), load_explore_fixtures(), load_cache()["vectors"]
    print(f"{'offset':>6} {'scale':>6} {'floor':>6} | {'diff':>5} {'adjMRR':>6} {'adjR@5':>6} {'prec':>5} {'diverg':>6} empty")
    for offset, scale, floor in itertools.product(args.offset, args.scale, args.floor):
        ranking.EXPLORE_OFFSET, ranking.EXPLORE_SCALE, ranking.EXPLORE_FLOOR = offset, scale, floor
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            index_path = _seed(tmp_path, fixtures, vectors)
            with _make_client(tmp_path, vectors, RankingConfig()) as client:
                s = summarize(run_explore_harness(client, index_path, fixtures, explore))
        print(f"{offset:>6.2f} {scale:>6.2f} {floor:>6.2f} | {s['differential']:>5.2f} {s['adj_mrr']:>6.3f} "
              f"{s['adj_recall']:>6.3f} {s['precision']:>5.2f} {s['divergence']:>6.3f} {len(s['empty'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
