"""Freeze alternate assignment laws offline; never accesses the chamber."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from collect import CHAMBER, FIXED, save_manifest, timestamp


def joint_law(design, midpoint=0.0):
    """Return P(X,H), sampling H first through Bayes' rule during preparation."""
    xs = np.arange(8, 25)
    if design == "uniform":
        joint = np.array([[1/26 if abs(x-(14+4*h)) <= 6 else 0
                           for h in (0, 1)] for x in xs])
    else:
        if abs(midpoint) >= .2:
            raise ValueError("Optical midpoint too far from the symmetric coordinate; review calibration.")
        k = abs(xs-16)
        lo, hi = np.maximum(0, (k-.5)/8), np.minimum(1, (k+.5)/8)
        mass0 = beta.cdf(hi, 2, 5)-beta.cdf(lo, 2, 5)
        mass1 = beta.cdf(hi, 5, 2)-beta.cdf(lo, 5, 2)
        p = mass1/(mass0+mass1)
        px = .5*(mass0+mass1)*np.where(xs > 16, .8, np.where(xs < 16, .2, 1.))
        m = .7*(2*p-1)
        v = np.where(xs > 16, .9, np.where(xs < 16, .75, .87))
        # Source coordinates are (-1, midpoint, +1). Correct the distribution
        # using calibration only so its conditional optical mean equals m(K).
        signed_outer_mean = m-midpoint*(1-v)
        posterior = np.column_stack(((v-signed_outer_mean)/2, 1-v,
                                     (v+signed_outer_mean)/2))
        joint = px[:, None]*posterior
    if np.any(joint < 0) or not np.isclose(joint.sum(), 1):
        raise ValueError("Invalid joint assignment law")
    return xs, joint


def prepare(args):
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"manifest.json").exists():
        raise ValueError("Manifest exists; use a new directory")
    xs, joint = joint_law(args.design, args.midpoint)
    ph, px = joint.sum(axis=0), joint.sum(axis=1)
    reds = [32, 192] if args.design == "uniform" else [32, 112, 192]
    rng = np.random.default_rng(args.seed)
    rows = []
    for block in range(args.blocks):
        for kind in rng.permutation(["confounded", "reference"]):
            count = args.train_per_block if kind == "confounded" else args.reference_per_block
            for e in rng.permutation(2):
                hs = rng.choice(len(ph), p=ph, size=count)
                # H is drawn first. The treatment distribution depends on H
                # only in the confounded data. Reference uses independent X.
                xvals = [rng.choice(xs, p=joint[:, h]/ph[h] if kind == "confounded" else px)
                         for h in hs]
                for x, h in zip(xvals, hs):
                    rows.append(dict(kind=kind, block_id=block, E=int(e), H=int(h),
                                     X=int(x), red=reds[h], pol_2=[0, 60][e],
                                     replicate_id=args.replicate_id))
    schedule = pd.DataFrame(rows)
    schedule.insert(0, "flag", np.arange(1, len(schedule)+1))
    schedule["led_3_ir"] = schedule.X
    schedule.to_csv(directory/"schedule.csv", index=False)
    pd.DataFrame([dict(X=int(x), H=h, probability=float(joint[i, h]))
                  for i, x in enumerate(xs) for h in range(len(ph))]).to_csv(directory/"assignment_law.csv", index=False)
    manifest = dict(stage="campaign", design=args.design, created_utc=timestamp(),
                    replicate_id=args.replicate_id, chamber_id=CHAMBER,
                    config="standard", seed=args.seed, n_measurements=len(schedule),
                    n_blocks=args.blocks, fixed_controls=FIXED, wait_ms=50,
                    red_levels=reds, angles=[0, 60], center=16, scale=8,
                    train_per_environment=args.blocks*args.train_per_block,
                    reference_per_environment=args.blocks*args.reference_per_block,
                    optical_midpoint=args.midpoint, h_prior=ph.tolist(),
                    schedule_sha256=hashlib.sha256((directory/"schedule.csv").read_bytes()).hexdigest(),
                    assignment_sha256=hashlib.sha256((directory/"assignment_law.csv").read_bytes()).hexdigest(),
                    status="PREPARED", calibration_basis=args.calibration_basis,
                    fitting="Unchanged synthetic-paper OVR; seeds 0,1,2; whole-block nested n/E="
                            + ",".join(str(n) for n in (500, 1000, 2500, 5000) if n <= args.blocks*args.train_per_block)
                            + "; no reference tuning.")
    if args.design == "ternary":
        manifest.update(mean_amplitude=.7, outer_mass_plus=.9, outer_mass_minus=.75,
                        note="Posterior mean follows original binary score; source distribution also varies with sign. Midpoint correction uses calibration, not estimator results.")
    else:
        manifest.update(treatment_equation="X=16+2*(2H-1)+U; U uniform over integers -6,...,6, independent of H.")
    save_manifest(directory, manifest)
    print(json.dumps({"directory": str(directory), "design": args.design, "rows": len(schedule), "h_prior": ph.tolist()}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--design", choices=["ternary", "uniform"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--replicate-id", required=True)
    p.add_argument("--blocks", type=int, default=50)
    p.add_argument("--train-per-block", type=int, default=50)
    p.add_argument("--reference-per-block", type=int, default=5)
    p.add_argument("--midpoint", type=float, default=0)
    p.add_argument("--calibration-basis", default="Original binary-source calibration; X range 8 through 24.")
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
