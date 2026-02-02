#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeNeSyS_v10_6_EB2.py
====================
Author : Eliot CHIMPEL

EB2 (v10.6) — Reconstruction / diagnosis of the emergent gamma from EB01 outputs.

Philosophy (anti-tuning, pipeline continuity)
----------------------------------------------
- EB01 produces a geometry (EB0) plus an auxiliary solution (EB1).
- EB2 does not "choose" gamma: it reconstructs an effective gamma(x) from
  the dynamical identities already encoded in the EB01 outputs (U, phi, etc.).
- Two independent estimators:
    (A) from the U equation: Upp = -2H Up - a^2 (R + <gamma, phi>)
    (B) from the phi equation: phipp = -2H phip - a^2 (m^2 phi) - a^2 (beta R + gamma U)
  -> (A) and (B) are compared as a consistency test (no fitting).

Inputs
------
- geom_out NPZ (EB0): x, a, Hconf, R, ...
- aux_out  NPZ (EB1): U, Up, Upp, phi, phip, phipp, beta, gamma, m, ...

Outputs
-------
- eb2_out_npz: gamma_U_est, gamma_phi_est, projections (nf > 1), masks, residuals
- eb2_summary_json: statistics + paths + sha256
- manifest_json: traceability (inputs/outputs + sha256) + runtime parameters

Important note
--------------
If EB01 was run with gamma = 0 (as in your EB01 regime),
then the reconstructions should yield ~0 (up to numerical errors).
This is not "imposed" here: it is *inferred* from the identities.

"note": "Input gamma from EB01 is ignored by EB2; gamma is reconstructed independently."
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

SCRIPT_VERSION = "v10.6"

# -------------------------
# Logging
# -------------------------
LOGGER = logging.getLogger("GeNeSyS_EB2")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

def setup_logging(log_path: Optional[Path] = None) -> None:
    if LOGGER.handlers:
        return
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    LOGGER.addHandler(sh)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)
        LOGGER.addHandler(fh)

def log_info(msg: str) -> None:
    LOGGER.info(msg)

def log_warn(msg: str) -> None:
    LOGGER.warning(msg)

def die(msg: str, code: int = 1) -> None:
    LOGGER.error(msg)
    raise SystemExit(code)

# -------------------------
# Small utils
# -------------------------
def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def npz_load_or_die(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        die(f"[EB2] Fichier introuvable ({label}): {path}")
    try:
        return dict(np.load(path, allow_pickle=False))
    except Exception as e:
        die(f"[EB2] Impossible de charger {label} ({path}): {e}")

def safe_div(num: np.ndarray, den: np.ndarray, den_min_abs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retourne (num/den, mask_ok) où mask_ok indique den "suffisant".
    """
    den = np.asarray(den, float)
    num = np.asarray(num, float)
    mask = np.isfinite(num) & np.isfinite(den) & (np.abs(den) >= float(den_min_abs))
    out = np.full_like(num, np.nan, dtype=float)
    out[mask] = num[mask] / den[mask]
    return out, mask

@dataclass
class EB2Config:
    den_min_abs_phi: float = 1e-14
    den_min_abs_U: float = 1e-14

# -------------------------
# Core reconstruction
# -------------------------
def reconstruct_gamma(
    geom: Dict[str, Any],
    aux: Dict[str, Any],
    cfg: EB2Config,
) -> Dict[str, Any]:
    """
    Reconstruit gamma_effective à partir des sorties EB01.

    Support:
    - nf=1 : phi shape (N,)
    - nf>1 : phi shape (N,nf) ; gamma reconstruit comme projection minimale-norme (direction phi)
    """
    # geometry essentials
    for k in ("x", "a", "Hconf", "R"):
        if k not in geom:
            die(f"[EB2] Clé manquante dans geom NPZ: '{k}'")
    x = np.asarray(geom["x"], float)
    a = np.asarray(geom["a"], float)
    H = np.asarray(geom["Hconf"], float)
    R = np.asarray(geom["R"], float)

    # aux essentials
    for k in ("U", "Up", "Upp", "phi", "phip", "phipp", "beta", "m"):
        if k not in aux:
            die(f"[EB2] Clé manquante dans aux NPZ: '{k}'")
    U = np.asarray(aux["U"], float)
    Up = np.asarray(aux["Up"], float)
    Upp = np.asarray(aux["Upp"], float)
    phi = np.asarray(aux["phi"], float)
    phip = np.asarray(aux["phip"], float)
    phipp = np.asarray(aux["phipp"], float)
    beta = np.asarray(aux["beta"], float)
    m = np.asarray(aux["m"], float)

    N = x.size
    if any(arr.shape[0] != N for arr in (a, H, R, U, Up, Upp)):
        die("[EB2] Incohérence de taille: geometry/aux ne partagent pas le même N.")

    # Determine nf
    if phi.ndim == 1:
        nf = 1
    elif phi.ndim == 2:
        nf = int(phi.shape[1])
    else:
        die(f"[EB2] phi ndim inattendu: {phi.ndim}")

    # ---------- (A) depuis eq U : <gamma,phi> = S_U
    # Upp = -2H Up - a^2 (R + <gamma,phi>)
    # => <gamma,phi> = -(Upp + 2HUp)/a^2 - R
    a2 = a**2
    SU = -(Upp + 2.0 * H * Up) / a2 - R  # scalar over x

    # ---------- (B) depuis eq phi : gamma * U = S_phi  (par composante)
    # phipp = -2H phip - a^2(m^2 phi) - a^2(beta R + gamma U)
    # => gamma = -(phipp + 2H phip + a^2(m^2 phi) + a^2 beta R) / (a^2 U)
    if nf == 1:
        # beta, m could be profiles or constants; ensure shape (N,)
        beta1 = beta if beta.ndim == 1 else beta[:, 0]
        m1 = m if m.ndim == 1 else m[:, 0]

        num_phi = -(phipp + 2.0 * H * phip + a2 * (m1**2) * phi + a2 * (beta1 * R))
        den_phi = a2 * U
        gamma_phi_est, mask_phi = safe_div(num_phi, den_phi, cfg.den_min_abs_U)

        # Now gamma from U equation:
        # gamma = SU / phi
        gamma_U_est, mask_U = safe_div(SU, phi, cfg.den_min_abs_phi)

        # residual check: dot(gamma_U_est,phi) - SU should be ~0 wherever mask_U True
        resid_U = np.full(N, np.nan, float)
        resid_U[mask_U] = gamma_U_est[mask_U] * phi[mask_U] - SU[mask_U]

        resid_phi = np.full(N, np.nan, float)
        resid_phi[mask_phi] = gamma_phi_est[mask_phi] * U[mask_phi] - (
            -(phipp[mask_phi] + 2.0 * H[mask_phi] * phip[mask_phi] + a2[mask_phi] * (m1[mask_phi]**2) * phi[mask_phi]
              + a2[mask_phi] * (beta1[mask_phi] * R[mask_phi])) / a2[mask_phi]
        )

        return {
            "nf": int(nf),
            "SU": SU,
            "gamma_U_est": gamma_U_est,
            "gamma_phi_est": gamma_phi_est,
            "mask_gamma_from_U": mask_U.astype(np.int8),
            "mask_gamma_from_phi": mask_phi.astype(np.int8),
            "resid_U": resid_U,
            "resid_phi": resid_phi,
        }

    # nf > 1 :
    # (A) gives only scalar SU = <gamma,phi>. We choose minimal-norm gamma_parallel = SU * phi / ||phi||^2
    # (B) gives vector gamma per component from phi-equation (componentwise) -> compare projection.
    phi2 = np.sum(phi * phi, axis=1)  # (N,)
    gammaU_par = np.zeros_like(phi, dtype=float)  # (N,nf)
    mask_phi2 = np.isfinite(SU) & np.isfinite(phi2) & (phi2 >= cfg.den_min_abs_phi)
    gammaU_par[:] = np.nan
    gammaU_par[mask_phi2, :] = (SU[mask_phi2, None] * phi[mask_phi2, :]) / (phi2[mask_phi2, None])

    # From phi eq: componentwise gamma
    num = -(phipp + 2.0 * H[:, None] * phip + a2[:, None] * ((m**2) * phi) + a2[:, None] * (beta * R[:, None]))
    den = a2[:, None] * U[:, None]
    gamma_phi_est = np.full_like(phi, np.nan, dtype=float)
    mask_phi = np.isfinite(num) & np.isfinite(den) & (np.abs(den) >= cfg.den_min_abs_U)
    gamma_phi_est[mask_phi] = num[mask_phi] / den[mask_phi]

    # Residual for U eq: <gammaU_par,phi> - SU
    dotU = np.sum(gammaU_par * phi, axis=1)
    resid_U = np.full(N, np.nan, float)
    resid_U[mask_phi2] = dotU[mask_phi2] - SU[mask_phi2]

    # Compare phi-based estimate projected onto phi direction
    # gamma_phi_parallel = ( (gamma_phi_est·phi) / (phi·phi) ) * phi
    gdotphi = np.sum(gamma_phi_est * phi, axis=1)
    gammaPhi_par = np.full_like(phi, np.nan, float)
    ok = np.isfinite(gdotphi) & mask_phi2
    gammaPhi_par[ok, :] = (gdotphi[ok, None] * phi[ok, :]) / (phi2[ok, None])

    return {
        "nf": int(nf),
        "SU": SU,
        "gamma_U_par": gammaU_par,
        "gamma_phi_est": gamma_phi_est,
        "gamma_phi_par": gammaPhi_par,
        "mask_phi2": mask_phi2.astype(np.int8),
        "mask_gamma_from_phi": mask_phi.astype(np.int8),
        "resid_U": resid_U,
    }

def summarize(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    a = np.asarray(arr, float)
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        a = a[m]
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"N": 0}
    return {
        "N": int(a.size),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "p16": float(np.quantile(a, 0.16)),
        "p50": float(np.quantile(a, 0.50)),
        "p84": float(np.quantile(a, 0.84)),
    }

# -------------------------
# CLI command
# -------------------------
def cmd_eb2(args) -> None:
    geom_npz = Path(args.geom_npz).resolve()
    aux_npz = Path(args.aux_npz).resolve()

    geom = npz_load_or_die(geom_npz, "geom_npz")
    aux = npz_load_or_die(aux_npz, "aux_npz")

    cfg = EB2Config(
        den_min_abs_phi=float(args.den_min_abs_phi),
        den_min_abs_U=float(args.den_min_abs_U),
    )

    out_npz = Path(args.eb2_out_npz).resolve()
    out_summary = Path(args.eb2_summary_json).resolve()
    out_manifest = Path(args.manifest_json).resolve()

    ensure_parent(out_npz)
    ensure_parent(out_summary)
    ensure_parent(out_manifest)

    log_info("[EB2] Reconstruction gamma_effective...")
    rec = reconstruct_gamma(geom, aux, cfg)

    # Save NPZ
    np.savez(out_npz, **{k: v for k, v in rec.items()})
    log_info(f"[EB2] NPZ saved: {out_npz}")

    # Summary JSON
    nf = int(rec["nf"])
    summary = {
        "model": "GeNeSyS_EB2",
        "version": SCRIPT_VERSION,
        "timestamp_unix": int(time.time()),
        "inputs": {
            "geom_npz": str(geom_npz),
            "aux_npz": str(aux_npz),
        },
        "outputs": {
            "eb2_out_npz": str(out_npz),
        },
        "sha256": {
            "geom_npz": sha256_file(geom_npz),
            "aux_npz": sha256_file(aux_npz),
            "eb2_out_npz": sha256_file(out_npz),
        },
        "config": {
            "den_min_abs_phi": cfg.den_min_abs_phi,
            "den_min_abs_U": cfg.den_min_abs_U,
        },
        "nf": nf,
    }

    if nf == 1:
        summary["stats"] = {
            "gamma_U_est": summarize(rec["gamma_U_est"], rec["mask_gamma_from_U"]),
            "gamma_phi_est": summarize(rec["gamma_phi_est"], rec["mask_gamma_from_phi"]),
            "resid_U": summarize(rec["resid_U"], rec["mask_gamma_from_U"]),
        }
    else:
        # For nf>1, summarize norms along x
        gU = rec["gamma_U_par"]
        gP = rec["gamma_phi_par"]
        norm_gU = np.sqrt(np.nansum(gU * gU, axis=1))
        norm_gP = np.sqrt(np.nansum(gP * gP, axis=1))
        summary["stats"] = {
            "||gamma_U_par||": summarize(norm_gU, rec["mask_phi2"]),
            "||gamma_phi_par||": summarize(norm_gP, rec["mask_phi2"]),
            "resid_U": summarize(rec["resid_U"], rec["mask_phi2"]),
        }

    write_json(out_summary, summary)
    log_info(f"[EB2] Summary JSON saved: {out_summary}")

    # Manifest JSON (traceability)
    manifest = {
        "script": "GeNeSyS_v10_6_EB2.py",
        "script_version": SCRIPT_VERSION,
        "stage": "EB2",
        "timestamp_unix": int(time.time()),
        "status": "SUCCESS",
        "inputs": {
            "geom_npz": str(geom_npz),
            "aux_npz": str(aux_npz),
        },
        "outputs": {
            "eb2_out_npz": str(out_npz),
            "eb2_summary_json": str(out_summary),
        },
        "sha256": summary["sha256"],
        "config": summary["config"],
        "notes": "EB2 reconstructs gamma from EB01 identities; no scanning/tuning.",
    }
    write_json(out_manifest, manifest)
    log_info(f"[EB2] Manifest saved: {out_manifest}")

# -------------------------
# Robust global arg parsing (allow --log-file anywhere)
# -------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="GeNeSyS_v10_6_EB2.py", add_help=True)

    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eb2", help="Reconstruct gamma_effective from EB01 outputs (geom+aux NPZ).")
    pe.add_argument("--geom-npz", required=True, help="geom_output NPZ (from EB01 / EB0).")
    pe.add_argument("--aux-npz", required=True, help="aux_output NPZ (from EB01 / EB1).")

    pe.add_argument("--den-min-abs-phi", type=float, default=1e-14, help="Min |phi| (or ||phi||^2) threshold.")
    pe.add_argument("--den-min-abs-U", type=float, default=1e-14, help="Min |U| threshold in phi-equation inversion.")

    pe.add_argument("--eb2-out-npz", required=True, help="Output NPZ with reconstructed gamma diagnostics.")
    pe.add_argument("--eb2-summary-json", required=True, help="Output summary JSON.")
    pe.add_argument("--manifest-json", required=True, help="Output manifest JSON (sha256, inputs/outputs).")
    pe.set_defaults(func=cmd_eb2)

    return p

def main() -> None:
    # Parse --log-file first
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--log-file", default=None)
    pre_args, remaining_args = pre_parser.parse_known_args()

    setup_logging(Path(pre_args.log_file).resolve() if pre_args.log_file else None)

    # Parse the rest of the arguments
    parser = build_parser()
    args = parser.parse_args(remaining_args)

    args.func(args)

if __name__ == "__main__":
    main()
