"""Fidelity of the paper cards in papers/ against the AUTHORS' code (GitHub Actions: verify-papers).

    python scripts/verify_papers.py [--cards papers] [--work /tmp/papers]

For every card with an `official` section: clone the authors' repository at the pinned commit,
build their model and the card's network, copy the authors' (perturbed) weights into the card's
network and compare the outputs (dlnet.paper.verify). Prints one line per card and the content
of every <card>.fidelity.json, which is then committed next to the card. Exit code 1 unless every
card reaches level B.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dlnet import paper

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="papers")
    ap.add_argument("--work", default="/tmp/papers")
    a = ap.parse_args()
    bad = 0
    for path in sorted(Path(a.cards).glob("*.yaml")):
        card = paper.load(path)
        off = card.get("official")
        if not off or "architecture" not in card:
            print(f"{path.stem:<18} skipped (no `official` + `architecture`: nothing to compare)")
            continue
        root = Path(a.work) / paper.slug(card)
        if not root.exists():
            subprocess.run(["git", "clone", "-q", off["repo"], str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "checkout", "-q", off["commit"]], check=True)
        try:
            r = paper.verify(card, root)
        except Exception as e:  # noqa: BLE001  report every card, then fail
            print(f"{path.stem:<18} ERROR {type(e).__name__}: {e}")
            bad += 1
            continue
        print(f"{path.stem:<18} level {r['level']}  params {r['params_card']:,} vs {r['params_official']:,}  "
              f"max|diff| {r.get('max_abs_diff')}  [{r['compared']}]  {r.get('note', '')}")  # fmt: skip
        print(f"FIDELITY {path.stem} {json.dumps(r)}")
        for k in [k for k in sys.modules if k.startswith("official_")]:
            del sys.modules[k]
        bad += r["level"] != "B"
    print("ALL LEVEL B" if not bad else f"{bad} CARD(S) NOT AT LEVEL B")
    raise SystemExit(1 if bad else 0)
