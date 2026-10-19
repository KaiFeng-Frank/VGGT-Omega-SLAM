#!/usr/bin/env python3
"""Contact sheet of the P_loop pairs (CPU) so a human can confirm they show the same place.

  OMP_NUM_THREADS=4 <python> tools/calib_contact_sheet.py [--pairs runs/calibration/pairs_defs.json]
Writes runs/calibration/p_loop_contact_sheet.png (left = query/late frame, right = retrieved/early frame).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vos import env  # noqa: E402

env.prepare(upstream=False)

from vos import calib_plots  # noqa: E402

OUT = env.PROJECT / "runs" / "calibration"


def with_paths(defs):
    kf = {k["image"]: k["path"] for k in json.loads((OUT / "keyframes.json").read_text())["keyframes"]}
    out = []
    for pair in defs["pairs"]:
        out.append(dict(pair, path_late=kf[pair["image_late"]], path_early=kf[pair["image_early"]]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=Path, default=OUT / "pairs_defs.json")
    p.add_argument("--set", default="P_loop")
    p.add_argument("--output", type=Path, default=OUT / "p_loop_contact_sheet.png")
    args = p.parse_args()
    defs = json.loads(args.pairs.read_text())
    chosen = [q for q in with_paths(defs) if q["set"] == args.set]
    calib_plots.contact_sheet(
        chosen, args.output,
        f"{args.set} 全部 {len(chosen)} 对(office_loop,{defs['n_keyframes']} 个关键帧;每对左=查询帧,右=检索帧)")
    print(args.output, len(chosen))


if __name__ == "__main__":
    main()
