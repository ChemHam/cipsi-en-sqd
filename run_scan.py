#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a scan.

    python run_scan.py inputs/cn_16o.json
    python run_scan.py inputs/cn_16o.json --geometries 2.90
    python run_scan.py inputs/oh_18o.json --geometries 1.0:1.6 --out results/oh_a
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import systems
from core import config, driver


def parse_geometries(spec, available):
    """A geometry selection: values, a:b range, or every nth."""
    if not spec:
        return list(available)
    picked = []
    for part in spec:
        if ":" in part:
            lo, _, hi = part.partition(":")
            lo = float(lo) if lo else float("-inf")
            hi = float(hi) if hi else float("inf")
            picked += [g for g in available if lo <= g <= hi]
        elif part.startswith("every"):
            picked += list(available)[::int(part[5:])]
        else:
            want = float(part)
            near = min(available, key=lambda g: abs(g - want))
            if abs(near - want) > 1e-6:
                raise SystemExit(f"!! no geometry at {want}; nearest is {near}")
            picked.append(near)
    return sorted(set(picked))


def provenance(cfg):
    """What the numbers came from. Recorded per point, because an environment
    that changes mid-scan is otherwise invisible in the output."""
    info = {"python": platform.python_version(),
            "platform": platform.platform(),
            "config": cfg.to_dict()}
    for mod in ("numpy", "pyscf", "ffsim", "qiskit"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info


def main():
    ap = argparse.ArgumentParser(description="CIPSI-EN-SQD scan")
    ap.add_argument("input", help="an input file under inputs/")
    ap.add_argument("--geometries", nargs="+", metavar="G",
                    help="values, a:b ranges, or everyN; default all")
    ap.add_argument("--out", metavar="DIR",
                    help="where to write; default results/<name>")
    ap.add_argument("--threads", type=int, default=24,
                    help="PySCF threads; OMP_NUM_THREADS wins if set")
    ap.add_argument("--redo", action="store_true",
                    help="recompute geometries that are already written")
    args = ap.parse_args()

    from pyscf import lib as pyscf_lib
    n_threads = int(os.environ.get("OMP_NUM_THREADS", args.threads))
    pyscf_lib.num_threads(n_threads)
    print(f"threads  {n_threads}")

    cfg = config.load(args.input)
    system = systems.load(getattr(cfg, "system", cfg.name.split("_")[0]))
    config.bind(cfg)

    out = Path(args.out or f"results/{cfg.name}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(provenance(cfg), indent=2),
                                   encoding="utf-8")

    wanted = parse_geometries(args.geometries, cfg.geometries)
    todo = [g for g in wanted
            if args.redo or not (out / f"{system.COORDINATE}_{g:.3f}.json").exists()]

    print(f"system   {cfg.name}  ({cfg.norb} orbitals, {cfg.nelec} electrons)")
    print(f"labeling {cfg.labeling}   roots {cfg.nroots_int}")
    print(f"scan     {len(todo)} of {len(wanted)} geometries in "
          f"{system.COORDINATE}   -> {out}")
    if not todo:
        print("nothing to do")
        return

    t0 = time.time()
    for i, x in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {system.COORDINATE} = {x}", flush=True)
        t = time.time()
        point = driver.compute_point(x, system)
        point["wall_s"] = round(time.time() - t, 1)
        path = out / f"{system.COORDINATE}_{x:.3f}.json"
        path.write_text(json.dumps(point, indent=1, default=float),
                        encoding="utf-8")
        print(f"  {path}  [{point['wall_s']:.0f} s]", flush=True)
    print(f"\n{len(todo)} geometries in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
