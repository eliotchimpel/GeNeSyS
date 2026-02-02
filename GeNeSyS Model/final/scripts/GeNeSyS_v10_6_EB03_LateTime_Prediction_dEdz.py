#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GeNeSyS_v10_6_EB03_LateTime_Prediction_dEdz.py
Late-time internal diagnostic (EB03 scope only).
Author : Eliot CHIMPEL
"""

from __future__ import annotations
import argparse, os, json, csv, hashlib
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_monotonic_z(z, *arrays):
    idx = np.argsort(z)
    z_sorted = z[idx]
    arrays_sorted = [a[idx] for a in arrays]
    z_unique, uidx = np.unique(z_sorted, return_index=True)
    arrays_unique = [a[uidx] for a in arrays_sorted]
    info = {
        "n_raw": int(len(z)),
        "n_unique": int(len(z_unique)),
        "removed_duplicates": int(len(z) - len(z_unique)),
    }
    return z_unique, arrays_unique, info


def lcdm_dEdz(z, Om=0.3, Ol=0.7):
    zp1 = 1 + z
    E = np.sqrt(Om * zp1**3 + Ol)
    return 0.5 * (3 * Om * zp1**2) / E


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geom-npz", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--lcdm-Om", type=float, default=0.3)
    p.add_argument("--lcdm-Ol", type=float, default=0.7)
    p.add_argument("--z-min", type=float, default=None)
    p.add_argument("--z-max", type=float, default=None)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    geom = dict(np.load(args.geom_npz))
    z_raw = geom["z"]
    E_raw = geom["E"]

    z, (E,), zinfo = prepare_monotonic_z(z_raw, E_raw)

    mask = np.ones_like(z, dtype=bool)
    if args.z_min is not None:
        mask &= z >= args.z_min
    if args.z_max is not None:
        mask &= z <= args.z_max

    z = z[mask]
    E = E[mask]

    dEdz_gen = np.gradient(E, z)
    dEdz_lcdm = lcdm_dEdz(z, args.lcdm_Om, args.lcdm_Ol)

    diff = dEdz_gen - dEdz_lcdm

    figpath = os.path.join(
        args.output_dir, f"eb03_late_prediction_dEdz_{args.tag}.png"
    )
    plt.figure(figsize=(10, 6))
    plt.plot(z, dEdz_gen, label="GeNeSyS (EB03)")
    plt.plot(z, dEdz_lcdm, "--", label="ΛCDM reference", color="red")
    plt.xlabel("z")
    plt.ylabel("dE/dz")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figpath, dpi=200)
    plt.close()

    summary = {
        "tag": args.tag,
        "z_range": [float(z.min()), float(z.max())],
        "dEdz_genesys": {
            "min": float(dEdz_gen.min()),
            "max": float(dEdz_gen.max()),
            "mean": float(dEdz_gen.mean()),
        },
        "comparison_lcdm": {
            "MAE": float(np.mean(np.abs(diff))),
            "RMSE": float(np.sqrt(np.mean(diff**2))),
        },
        "z_cleaning": zinfo,
        "notes": [
            "Late-time internal diagnostic only",
            "No observational calibration",
            "ΛCDM curve is reference only",
        ],
    }

    sumpath = os.path.join(
        args.output_dir, f"eb03_late_prediction_dEdz_summary_{args.tag}.json"
    )
    with open(sumpath, "w") as f:
        json.dump(summary, f, indent=2)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {args.geom_npz: sha256_file(args.geom_npz)},
        "outputs": {
            figpath: sha256_file(figpath),
            sumpath: sha256_file(sumpath),
        },
    }

    manpath = os.path.join(
        args.output_dir, f"manifest_eb03_late_prediction_dEdz_{args.tag}.json"
    )
    with open(manpath, "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 72)
    print("EB03 LATE-TIME PREDICTION COMPLETE")
    print("=" * 72)
    print(f"[SAVED] {figpath}")
    print(f"[SAVED] {sumpath}")
    print(f"[SAVED] {manpath}")
    print(f"z range        : [{z.min()}, {z.max()}]")
    print(
        f"dE/dz (min,max): ({dEdz_gen.min():.4e}, {dEdz_gen.max():.4e})"
    )
    print(f"dE/dz mean     : {dEdz_gen.mean():.4e}")
    print("[NOTE] No observational calibration applied.")
    print("=" * 72)


if __name__ == "__main__":
    main()