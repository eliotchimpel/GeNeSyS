#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeNeSyS_v10_6_EB3.py

EB3 (v10.6) — Scalar perturbations (production, robust, anti-triche)
-------------------------------------------------------------------
Author: Eliot CHIMPEL

This EB3 integrates the comoving curvature mode R_k directly:

    R'' + 2 (z'/z) R' + k^2 R = 0

Why direct R_k?
- Avoids the common super-horizon overflow issues seen with v'' + (k^2 - z''/z) v = 0.
- Keeps the numerical evolution stable even when k/(aH) is tiny.

CRITICAL CONSISTENCY RULE (pipeline + physics)
----------------------------------------------
- If 'z' is present in the EB0/EB01 geom NPZ, EB3 MUST use it.
- EB3 must NOT recompute z from (a, phip, Hconf) unless explicitly allowed,
  because unit/convention mismatches can silently rescale z by huge factors and
  create artificial runaways.

Inputs
------
--geom-npz : EB0/EB01 background NPZ (eta, a, Hconf, [x], and ideally z)
--aux-npz  : EB01 auxiliary NPZ (phi, phip) (kept for traceability; not used if z is in geom)
--eb2-npz  : optional, only for traceability
--perturbations-json : config JSON with keys:
    k_list (required), integrator (rk4 only), ic_mode, save_every, substeps,
    z_min_abs, err_raise

Outputs
-------
--eb3-out-npz, --eb3-summary-json, --manifest-json

Notes
-----
- Anti-triche: no tuning, no scanning; strict float error policy optional.
- Pre-parse --log-file so it can appear anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
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
LOGGER = logging.getLogger("GeNeSyS_EB3")
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


def log_error(msg: str) -> None:
    LOGGER.error(msg)


def die(msg: str, code: int = 1) -> None:
    log_error(msg)
    raise SystemExit(code)


# -------------------------
# Utils
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
        die(f"[EB3] File not found ({label}): {path}")
    try:
        return dict(np.load(path, allow_pickle=False))
    except Exception as e:
        die(f"[EB3] Failed to load {label} ({path}): {e}")


def summarize(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    a = np.asarray(arr, float)
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        if m.shape != a.shape:
            # allow masking by mode only if arr is (nk,...) or (...,nk) — but keep safe and explicit
            die(f"[EB3] summarize(): mask shape {m.shape} != data shape {a.shape}")
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
# Config
# -------------------------
@dataclass
class EB3Config:
    integrator: str = "rk4"          # rk4 only (prod)
    ic_mode: str = "bunch_davies"    # bunch_davies | zeros
    save_every: int = 1
    substeps: int = 1               # RK4 substeps per eta interval
    z_min_abs: float = 1e-14        # if |z| < threshold, shift start
    err_raise: bool = True          # raise on overflow/invalid


def read_perturbations_json(path: Path) -> Tuple[EB3Config, np.ndarray]:
    d = read_json(path)

    k_list = d.get("k_list", None)
    if k_list is None:
        die(f"[EB3] perturbations-json missing key 'k_list' in {path}")

    cfg = EB3Config(
        integrator=str(d.get("integrator", "rk4")).lower().strip(),
        ic_mode=str(d.get("ic_mode", "bunch_davies")).lower().strip(),
        save_every=int(d.get("save_every", 1)),
        substeps=int(d.get("substeps", 1)),
        z_min_abs=float(d.get("z_min_abs", 1e-14)),
        err_raise=bool(d.get("err_raise", True)),
    )

    if cfg.integrator != "rk4":
        die("[EB3] Only rk4 is supported in v10.6 EB3 (prod).")
    if cfg.ic_mode not in ("bunch_davies", "zeros"):
        die("[EB3] ic_mode must be 'bunch_davies' or 'zeros'.")
    if cfg.save_every < 1:
        die("[EB3] save_every must be >= 1")
    if cfg.substeps < 1:
        die("[EB3] substeps must be >= 1")

    k_arr = np.asarray(k_list, float)
    if k_arr.ndim != 1 or k_arr.size < 1:
        die("[EB3] k_list must be a 1D non-empty array.")
    if not np.all(np.isfinite(k_arr)) or np.any(k_arr <= 0):
        die("[EB3] k_list contains invalid (non-finite or <=0) entries.")

    return cfg, k_arr


# -------------------------
# Background: z, z'/z, z''/z
# -------------------------
def compute_z_diagnostics(
    geom: Dict[str, Any],
    aux: Dict[str, Any],
    *,
    allow_recompute_z: bool = False,
) -> Dict[str, Any]:
    """
    Build {eta, x(optional), z, z_ok, zprime_over_z, zpp_over_z}.

    Anti-triche / pipeline rule:
    - If geom contains 'z', use it (authoritative).
    - If geom lacks 'z', then:
        - if allow_recompute_z is False -> abort (forces pipeline consistency)
        - else -> recompute z from (a, phip, Hconf) with a loud warning
    """
    for k in ("eta", "a", "Hconf"):
        if k not in geom:
            die(f"[EB3] Missing key in geom NPZ: '{k}'")

    eta = np.asarray(geom["eta"], float)
    a = np.asarray(geom["a"], float)
    Hc = np.asarray(geom["Hconf"], float)

    if eta.ndim != 1 or a.shape != eta.shape or Hc.shape != eta.shape:
        die("[EB3] Geometry shapes inconsistent (eta,a,Hconf).")

    if "z" in geom:
        z = np.asarray(geom["z"], float)
        if z.shape != eta.shape:
            die(f"[EB3] geom['z'] shape {z.shape} inconsistent with eta {eta.shape}")
        log_info("[EB3] ✅ Using pre-computed z from geom NPZ (pipeline-consistent).")
    else:
        if not allow_recompute_z:
            die("[EB3] geom NPZ does not contain 'z'. Refusing to recompute z to avoid unit/convention drift. "
                "Re-run EB01/EB0 to export z, or pass --allow-recompute-z (not recommended).")
        # Fallback (legacy) with explicit warning
        for k in ("phi", "phip"):
            if k not in aux:
                die(f"[EB3] Missing key in aux NPZ needed to recompute z: '{k}'")
        phip = np.asarray(aux["phip"], float)
        # if multi-field, take norm (more physically robust than first component)
        if phip.ndim == 2:
            phip_eff = np.sqrt(np.sum(phip**2, axis=1))
        else:
            phip_eff = phip
        if phip_eff.shape != eta.shape:
            die("[EB3] Aux shapes inconsistent (phip vs eta) for recompute z.")
        z = a * phip_eff / Hc
        log_warn("[EB3] ⚠️ Recomputing z = a*|phip|/Hconf because geom lacks 'z'. "
                 "This may break pipeline-consistency if conventions differ. Prefer exporting 'z' from EB01.")

    z_ok = np.isfinite(z) & (np.abs(z) > 0)

    # robust z'/z using np.gradient wrt eta
    dz_deta = np.gradient(z, eta, edge_order=2)
    zprime_over_z = np.full_like(z, np.nan, dtype=float)
    good = z_ok & np.isfinite(dz_deta) & (np.abs(z) >= 1e-300)
    zprime_over_z[good] = dz_deta[good] / z[good]

    # z''/z (diagnostic only)
    d2z = np.gradient(dz_deta, eta, edge_order=2)
    zpp_over_z = np.full_like(z, np.nan, dtype=float)
    good2 = z_ok & np.isfinite(d2z) & (np.abs(z) >= 1e-300)
    zpp_over_z[good2] = d2z[good2] / z[good2]

    out = {
        "eta": eta,
        "z": z,
        "z_ok": z_ok.astype(np.int8),
        "zprime_over_z": zprime_over_z,
        "zpp_over_z": zpp_over_z,
    }
    if "x" in geom:
        x = np.asarray(geom["x"], float)
        if x.shape == eta.shape:
            out["x"] = x
    return out


# -------------------------
# Integrate R_k
# -------------------------
def rk4_step_R(
    R: complex,
    Rp: complex,
    h: float,
    k: float,
    zprime_over_z: float,
) -> Tuple[complex, complex]:
    # R'' = -2 (z'/z) R' - k^2 R
    def f(stateR: complex, stateRp: complex, zpo: float) -> Tuple[complex, complex]:
        return stateRp, (-2.0 * zpo) * stateRp - (k * k) * stateR

    k1R, k1Rp = f(R, Rp, zprime_over_z)
    k2R, k2Rp = f(R + 0.5 * h * k1R, Rp + 0.5 * h * k1Rp, zprime_over_z)
    k3R, k3Rp = f(R + 0.5 * h * k2R, Rp + 0.5 * h * k2Rp, zprime_over_z)
    k4R, k4Rp = f(R + h * k3R, Rp + h * k3Rp, zprime_over_z)

    Rn = R + (h / 6.0) * (k1R + 2 * k2R + 2 * k3R + k4R)
    Rpn = Rp + (h / 6.0) * (k1Rp + 2 * k2Rp + 2 * k3Rp + k4Rp)
    return Rn, Rpn


def integrate_R_modes(bg: Dict[str, Any], k_list: np.ndarray, cfg: EB3Config) -> Dict[str, Any]:
    eta = np.asarray(bg["eta"], float)
    z = np.asarray(bg["z"], float)
    z_ok = np.asarray(bg["z_ok"]).astype(bool)
    zpo = np.asarray(bg["zprime_over_z"], float)

    N = eta.size
    save_every = int(cfg.save_every)
    substeps = int(cfg.substeps)

    # Save indices
    idx_save = np.arange(0, N, save_every, dtype=int)
    if idx_save[-1] != N - 1:
        idx_save = np.append(idx_save, N - 1)
    Ns = idx_save.size

    # Choose start index where z usable and z'/z finite (for BD conversion)
    start_idx = 0
    while start_idx < N and (
        (not z_ok[start_idx])
        or (abs(z[start_idx]) < cfg.z_min_abs)
        or (not np.isfinite(zpo[start_idx]))
    ):
        start_idx += 1
    if start_idx >= N - 2:
        die("[EB3] No valid start index found for z / z'/z. Check EB01 outputs (z, z'/z).")

    nk = k_list.size
    R_re = np.full((nk, Ns), np.nan, dtype=float)
    R_im = np.full((nk, Ns), np.nan, dtype=float)
    Rp_re = np.full((nk, Ns), np.nan, dtype=float)
    Rp_im = np.full((nk, Ns), np.nan, dtype=float)
    fail_mask = np.zeros(nk, dtype=np.int8)

    # Fast mapping from full-grid index to save slot (or -1)
    save_slot = -np.ones(N, dtype=int)
    save_slot[idx_save] = np.arange(Ns, dtype=int)

    err_ctx = np.errstate(over="raise", invalid="raise", divide="raise") if cfg.err_raise else np.errstate()

    with err_ctx:
        for ik, k in enumerate(k_list):
            k = float(k)

            # Initial conditions defined on v_k then mapped to (R, R')
            if cfg.ic_mode == "zeros":
                v0 = 0.0 + 0.0j
                vp0 = 0.0 + 0.0j
            else:
                v0 = np.exp(-1j * k * eta[start_idx]) / np.sqrt(2.0 * k)
                vp0 = (-1j * k) * v0

            z0 = float(z[start_idx])
            zpo0 = float(zpo[start_idx])

            # R = v/z ; R' = (v' - v z'/z)/z
            R = v0 / z0
            Rp = (vp0 - v0 * zpo0) / z0

            # Save at start if in save grid
            s0 = save_slot[start_idx]
            if s0 >= 0:
                R_re[ik, s0] = R.real
                R_im[ik, s0] = R.imag
                Rp_re[ik, s0] = Rp.real
                Rp_im[ik, s0] = Rp.imag

            try:
                for i in range(start_idx, N - 1):
                    h_big = float(eta[i + 1] - eta[i])
                    if not (h_big > 0 and np.isfinite(h_big)):
                        raise FloatingPointError("Non-positive or non-finite eta step.")

                    # If z'/z not finite, we refuse to step (keeps NaNs downstream, avoids fake physics)
                    if not np.isfinite(zpo[i]):
                        continue

                    # Use a simple linear interpolation of z'/z inside the interval (more faithful than freezing)
                    zpo_i = float(zpo[i])
                    zpo_ip1 = float(zpo[i + 1]) if np.isfinite(zpo[i + 1]) else zpo_i

                    h = h_big / float(substeps)
                    for s in range(substeps):
                        alpha = (s + 0.5) / float(substeps)
                        zpo_mid = (1.0 - alpha) * zpo_i + alpha * zpo_ip1
                        R, Rp = rk4_step_R(R, Rp, h, k, zpo_mid)

                    # Save at i+1 if needed
                    ss = save_slot[i + 1]
                    if ss >= 0:
                        R_re[ik, ss] = R.real
                        R_im[ik, ss] = R.imag
                        Rp_re[ik, ss] = Rp.real
                        Rp_im[ik, ss] = Rp.imag

            except FloatingPointError:
                fail_mask[ik] = 1
                continue

    # Power spectrum (diagnostic): P_R(k) at last saved time where R finite
    Rk = (R_re + 1j * R_im)
    last = Rk[:, -1]
    Pk_R = (k_list ** 3 / (2.0 * np.pi ** 2)) * (np.abs(last) ** 2)

    return {
        "start_idx": np.array([start_idx], dtype=int),
        "eta_save": eta[idx_save],
        "x_save": (np.asarray(bg["x"], float)[idx_save] if "x" in bg else np.full((Ns,), np.nan)),
        "Rk_re": R_re,
        "Rk_im": R_im,
        "Rkp_re": Rp_re,
        "Rkp_im": Rp_im,
        "Pk_R": Pk_R,
        "fail_mask": fail_mask,
        "k_list": k_list,
        "save_every": np.array([save_every], dtype=int),
        "substeps": np.array([substeps], dtype=int),
        "ic_mode": np.array([cfg.ic_mode], dtype="U32"),
        "integrator": np.array([cfg.integrator], dtype="U16"),
    }


# -------------------------
# CLI
# -------------------------
def cmd_eb3(args) -> None:
    geom_npz = Path(args.geom_npz).resolve()
    aux_npz = Path(args.aux_npz).resolve()
    eb2_npz = Path(args.eb2_npz).resolve() if args.eb2_npz else None
    pert_json = Path(args.perturbations_json).resolve()

    geom = npz_load_or_die(geom_npz, "geom_npz")
    aux = npz_load_or_die(aux_npz, "aux_npz")

    if eb2_npz is not None and not eb2_npz.exists():
        die(f"[EB3] eb2-npz not found: {eb2_npz}")

    cfg, k_list = read_perturbations_json(pert_json)
    log_info(f"[EB3] Loaded k_list (nk={k_list.size}), integrator={cfg.integrator}, save_every={cfg.save_every}, substeps={cfg.substeps}")

    # Background diagnostics (z must come from pipeline unless explicit override)
    bg = compute_z_diagnostics(geom, aux, allow_recompute_z=bool(args.allow_recompute_z))

    z_ok_frac = float(np.mean(bg["z_ok"]))
    V_ok_frac = float(np.mean(np.isfinite(bg["zpp_over_z"])))
    log_info(f"[EB3] z_ok_frac={z_ok_frac:.6f}, V_ok_frac={V_ok_frac:.6f}")

    # Integrate R_k
    out = integrate_R_modes(bg, k_list, cfg)

    # Save outputs
    out_npz = Path(args.eb3_out_npz).resolve()
    out_summary = Path(args.eb3_summary_json).resolve()
    out_manifest = Path(args.manifest_json).resolve()
    ensure_parent(out_npz)
    ensure_parent(out_summary)
    ensure_parent(out_manifest)

    np.savez(
        out_npz,
        # background diagnostics
        eta=bg["eta"],
        z=bg["z"],
        z_ok=bg["z_ok"],
        zprime_over_z=bg["zprime_over_z"],
        zpp_over_z=bg["zpp_over_z"],
        # results
        **out,
        script_version=np.array([SCRIPT_VERSION], dtype="U16"),
        z_source=np.array(["geom" if "z" in geom else ("recomputed" if args.allow_recompute_z else "missing")], dtype="U16"),
    )
    log_info(f"[EB3] NPZ saved: {out_npz}")

    # Summary
    Pk_R = out["Pk_R"]
    finite = np.isfinite(Pk_R)

    summary = {
        "model": "GeNeSyS_EB3",
        "version": SCRIPT_VERSION,
        "timestamp_unix": int(time.time()),
        "inputs": {
            "geom_npz": str(geom_npz),
            "aux_npz": str(aux_npz),
            "eb2_npz": str(eb2_npz) if eb2_npz is not None else None,
            "perturbations_json": str(pert_json),
        },
        "outputs": {
            "eb3_out_npz": str(out_npz),
        },
        "sha256": {
            "geom_npz": sha256_file(geom_npz),
            "aux_npz": sha256_file(aux_npz),
            "perturbations_json": sha256_file(pert_json),
            "eb3_out_npz": sha256_file(out_npz),
            **({"eb2_npz": sha256_file(eb2_npz)} if eb2_npz is not None else {}),
        },
        "config": {
            "integrator": cfg.integrator,
            "ic_mode": cfg.ic_mode,
            "save_every": cfg.save_every,
            "substeps": cfg.substeps,
            "z_min_abs": cfg.z_min_abs,
            "err_raise": cfg.err_raise,
            "allow_recompute_z": bool(args.allow_recompute_z),
        },
        "k_list": k_list.tolist(),
        "z_source": "geom" if "z" in geom else ("recomputed" if args.allow_recompute_z else "missing"),
        "z_ok_frac": float(np.mean(bg["z_ok"])),
        "V_ok_frac": float(np.mean(np.isfinite(bg["zpp_over_z"]))),
        "start_idx": int(out["start_idx"][0]),
        "fail_frac": float(np.mean(out["fail_mask"])),
        "stats": {
            "Pk_R": summarize(Pk_R),
            "finite_frac_Pk_R": float(np.mean(finite)),
        },
    }
    write_json(out_summary, summary)
    log_info(f"[EB3] Summary JSON saved: {out_summary}")

    # Manifest
    manifest = {
        "script": "GeNeSyS_v10_6_EB3.py",
        "script_version": SCRIPT_VERSION,
        "stage": "EB3",
        "timestamp_unix": int(time.time()),
        "status": "SUCCESS",
        "inputs": summary["inputs"],
        "outputs": {
            "eb3_out_npz": str(out_npz),
            "eb3_summary_json": str(out_summary),
        },
        "sha256": summary["sha256"],
        "notes": (
            "EB3 integrates R_k directly (anti-overflow). "
            "Pipeline-consistent z is used from geom NPZ when available; recompute is forbidden by default."
        ),
    }
    write_json(out_manifest, manifest)
    log_info(f"[EB3] Manifest saved: {out_manifest}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="GeNeSyS_v10_6_EB3.py", add_help=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eb3", help="Run EB3 perturbations from EB01 outputs (geom+aux).")
    pe.add_argument("--geom-npz", required=True)
    pe.add_argument("--aux-npz", required=True)
    pe.add_argument("--eb2-npz", default=None, help="Optional EB2 NPZ (traceability only).")
    pe.add_argument("--perturbations-json", required=True)
    pe.add_argument("--eb3-out-npz", required=True)
    pe.add_argument("--eb3-summary-json", required=True)
    pe.add_argument("--manifest-json", required=True)

    # Strict by default: do not recompute z unless user explicitly asks
    pe.add_argument(
        "--allow-recompute-z",
        action="store_true",
        help="Allow EB3 to recompute z if geom NPZ lacks it (NOT recommended; can break conventions).",
    )

    pe.set_defaults(func=cmd_eb3)
    return p


def main() -> None:
    # pre-parse --log-file so it's allowed anywhere
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--log-file", default=None)
    pre_args, remaining = pre.parse_known_args()
    setup_logging(Path(pre_args.log_file).resolve() if pre_args.log_file else None)

    parser = build_parser()
    args = parser.parse_args(remaining)
    args.func(args)


if __name__ == "__main__":
    main()