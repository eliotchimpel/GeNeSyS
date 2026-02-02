#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeNeSyS_v10_6_EB01.py
=====================
Author: Eliot CHIMPEL

GeNeSyS – EB0/EB1/EB01 production helper (v10.6)

Goals (rigorous + continuity EB2/EB3)
-------------------------------------
EB0:
  - Build geometry from pipeline outputs (best_Ez.csv + H0 from params/verdict)
  - Produce consistent x-grid, a(x), z(x), E(x), conformal Hubble ℋ(x), Ricci scalar R(x), conformal time η(x)
  - Strong sanitation: finite, monotonic x, E>0, a>0; optional smoothing before derivatives
  - SHA256 of inputs/outputs, reproducibility manifests

EB1:
  - Solve auxiliary background ODE system (U, phi_i) using geometry
  - Load EB profiles (beta, gamma, m) from an EB JSON that can be:
      (A) legacy arrays: {"x":[...],"beta":[...],"gamma":[...],"m":[...]}
      (B) v10.6 structured profiles: {"profiles": {"x": {"policy": "from_geometry", "description": "..."}, "beta": {"type": "constant", "value": 1.0}, "gamma": {"type": "constant", "value": 0.0}, "m": {"type": "constant", "value": 1.0}}}
  - Robust x policy:
      - strict (requires identical grid)
      - interp (interpolate EB profiles onto geometry grid)
  - Runaway/non-finite guard with early stop (FloatingPointError) + manifest diagnostics

Continuity EB2/EB3 (what we preserve/export)
--------------------------------------------
- aux_output NPZ includes:
    eta, x, a, z, E, Hconf, R,
    beta, gamma, m,
    U, Up, Upp, phi, phip, phipp,
    integrator config,
    *and* a "provenance" dict dumped into manifest (plus hashes)
- manifests include: pipeline refs, EB params hash, geom hash, aux hash, and key runtime settings
- (Optional) gamma_scan support can exist but is DISABLED by default. EB01 regime should not "fix"
  gamma: default gamma=0 unless your EB params explicitly gives something else.

Usage
-----
EB01 chained:
  python scripts/GeNeSyS_v10_6_EB01.py eb01 --verdict-json outputs_v67/_GeNeSyS_v10_5v2_full_cov_verdict.json \
      --eb-params-json data/eb01_from_pipeline_H67.json --eb-x-policy interp \
      --x-min -2 --x-max 0 --n-x 2001 --rk4-geom interp --integrator rk4 \
      --geom-out-npz outputs_v2/geom_output_v10_6_H67_xm2_0.npz \
      --aux-out-npz  outputs_v2/aux_output_v10_6_H67_xm2_0.npz \
      --manifest-eb0-json outputs_v2/manifest_eb0_v10_6_H67_xm2_0.json \
      --manifest-eb1-json outputs_v2/manifest_eb1_v10_6_H67_xm2_0.json

Author: GeNeSyS collaboration (production hardening)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

C_KM_S = 299792.458
SCRIPT_VERSION = "v10.6"

# ============================================================
# Logging (console + file)
# ============================================================

LOGGER = logging.getLogger("GeNeSyS_EB01")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

def setup_logging(log_path: Optional[Path] = None) -> None:
    """Configure logging once (stdout + optional file)."""
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
    LOGGER.error(msg)
    raise SystemExit(code)

# ============================================================
# Small utilities: IO, hash, checks
# ============================================================

def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def is_strictly_increasing(x: np.ndarray) -> bool:
    return bool(np.all(np.diff(x) > 0))

def cumulative_trapz(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Cumulative integral int_{x0}^{x} y(x') dx' using trapezoidal rule.
    Requires x strictly increasing.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("cumulative_trapz requires strictly increasing x.")
    out = np.zeros_like(x, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return out

def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average with reflect padding. window must be odd >= 3."""
    y = np.asarray(y, float)
    w = int(window)
    if w < 3 or (w % 2) == 0:
        raise ValueError("smooth_window must be an odd integer >= 3.")
    pad = w // 2
    yp = np.pad(y, (pad, pad), mode="reflect")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(yp, kernel, mode="valid")

# ============================================================
# Pipeline resolution helpers
# ============================================================

@dataclass
class PipelineRefs:
    best_ez: Path
    params_json: Path
    verdict_json: Optional[Path] = None
    scan_meta_json: Optional[Path] = None

def _resolve_from_verdict(verdict_json: Path) -> Tuple[Path, Path]:
    """Infer best_Ez.csv + best_fit_params.json from a pipeline verdict metadata."""
    v = read_json(verdict_json)
    meta = v.get("metadata", {})

    ez_name = meta.get("Ez_file") or meta.get("best_Ez_file")
    params_name = meta.get("params_file") or meta.get("best_fit_params_file")

    base = verdict_json.parent

    if ez_name and params_name:
        ezp = (base / ez_name).resolve() if not os.path.isabs(str(ez_name)) else Path(ez_name).resolve()
        pjp = (base / params_name).resolve() if not os.path.isabs(str(params_name)) else Path(params_name).resolve()
        if ezp.exists() and pjp.exists():
            return ezp, pjp

    candidates_ez = sorted(base.glob("*best_Ez*.csv"))
    candidates_params = sorted(base.glob("*best_fit_params*.json")) + sorted(base.glob("*fit_params*.json"))
    if candidates_ez and candidates_params:
        return candidates_ez[0].resolve(), candidates_params[0].resolve()

    raise FileNotFoundError(
        "Could not infer best_Ez.csv and best_fit_params.json from verdict. "
        "Provide --best-ez and --params-json explicitly."
    )

def resolve_pipeline_refs(
    best_ez: Optional[str],
    params_json: Optional[str],
    verdict_json: Optional[str],
    scan_meta_json: Optional[str],
) -> PipelineRefs:
    if best_ez and params_json:
        return PipelineRefs(
            best_ez=Path(best_ez).resolve(),
            params_json=Path(params_json).resolve(),
            verdict_json=Path(verdict_json).resolve() if verdict_json else None,
            scan_meta_json=Path(scan_meta_json).resolve() if scan_meta_json else None,
        )
    if not verdict_json:
        raise ValueError("Provide --best-ez and --params-json, or provide --verdict-json to infer them.")
    vpath = Path(verdict_json).resolve()
    ezp, pjp = _resolve_from_verdict(vpath)
    return PipelineRefs(
        best_ez=ezp,
        params_json=pjp,
        verdict_json=vpath,
        scan_meta_json=Path(scan_meta_json).resolve() if scan_meta_json else None,
    )

# ============================================================
# EB parameter JSON (v10.6 structured)
# ============================================================

def read_eb_params_v10_6(path: Path) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, Optional[Dict]]:
    """Read EB01 parameter JSON.

    v10.6 supports two equivalent shapes:

    1) Profile form (recommended, prod):
        {"profiles": {"x": {"policy": "from_geometry", "description": "..."}, "beta": {"type": "constant", "value": 1.0}, "gamma": {"type": "constant", "value": 0.0}, "m": {"type": "constant", "value": 1.0}}}
    """
    d = read_json(path)

    # Vérifie si la structure est imbriquée
    if "profiles" in d:
        profiles = d["profiles"]
        if "x" not in profiles:
            raise KeyError(f"EB params JSON missing key 'x' in profiles block in {path}")

        # Gestion de la clé x qui peut être un dictionnaire
        x_info = profiles["x"]
        if isinstance(x_info, dict):
            if x_info.get("policy") != "from_geometry":
                raise ValueError(f"Unsupported x policy: {x_info.get('policy')}")
            # Dans ce cas, on utilise la grille de géométrie
            x = None
        else:
            x = np.asarray(x_info, dtype=float)
    else:
        if "x" not in d:
            raise KeyError(f"EB params JSON missing key 'x' in {path}")
        x = np.asarray(d["x"], dtype=float)

    def _as_array(name: str):
        if "profiles" in d:
            if name not in d["profiles"]:
                raise KeyError(f"EB params JSON missing key '{name}' in profiles block in {path}")
            v = d["profiles"][name]
        else:
            if name not in d:
                raise KeyError(f"EB params JSON missing key '{name}' in {path}")
            v = d[name]

        if isinstance(v, dict):
            if v.get("type") == "constant":
                return float(v.get("value"))
            else:
                raise ValueError(f"Unsupported type for {name}: {v.get('type')}")
        elif isinstance(v, (int, float)):
            return float(v)
        else:
            return np.asarray(v, dtype=float)

    beta = _as_array("beta")
    gamma = _as_array("gamma")
    m = _as_array("m")

    # Si x est None, on utilise la grille de géométrie
    if x is None:
        return None, beta, gamma, m, None

    N = int(x.size)
    if N < 3:
        raise ValueError(f"EB params JSON has too few x points (N={N}) in {path}")

    def _broadcast(v):
        if isinstance(v, float):
            return np.full((N,), v, dtype=float)
        return v

    beta = _broadcast(beta)
    gamma = _broadcast(gamma)
    m = _broadcast(m)

    for name, arr in (("beta", beta), ("gamma", gamma), ("m", m)):
        if arr.ndim == 1:
            if arr.shape[0] != N:
                raise ValueError(
                    f"EB params '{name}' must have length N=len(x)={N} (got {arr.shape[0]}) in {path}"
                )
        elif arr.ndim == 2:
            if arr.shape[0] != N:
                raise ValueError(
                    f"EB params '{name}' must have shape (N, nf) with N=len(x)={N} (got {arr.shape}) in {path}"
                )
        else:
            raise ValueError(f"EB params '{name}' must be 1D or 2D (got ndim={arr.ndim}) in {path}")

    if not (beta.shape == gamma.shape == m.shape):
        raise ValueError(f"EB params shapes mismatch: beta{beta.shape}, gamma{gamma.shape}, m{m.shape} in {path}")

    return x, beta, gamma, m, None

# ============================================================
# EB0 — Geometry builder
# ============================================================

def _read_best_ez(best_ez: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(best_ez)
    cols = [c.strip() for c in df.columns]
    if "z" in cols:
        z = df["z"].to_numpy(dtype=float)
        if "E" in cols:
            E = df["E"].to_numpy(dtype=float)
        elif "E_genesys" in cols:
            E = df["E_genesys"].to_numpy(dtype=float)
        else:
            E = df.iloc[:, 1].to_numpy(dtype=float)
    else:
        z = df.iloc[:, 0].to_numpy(dtype=float)
        E = df.iloc[:, 1].to_numpy(dtype=float)
    return z, E

def _read_H0(params_json: Path, verdict_json: Optional[Path] = None) -> float:
    meta = read_json(params_json)
    best = meta.get("best_fit_parameters", meta.get("best_fit", meta))
    if isinstance(best, dict) and "H0" in best:
        return float(best["H0"])

    if verdict_json is not None and verdict_json.exists():
        v = read_json(verdict_json)
        m = v.get("metadata", {})
        if isinstance(m, dict) and "H0_used_km_s_Mpc" in m:
            return float(m["H0_used_km_s_Mpc"])

    raise KeyError("Could not find H0 in params_json (or verdict fallback).")

def build_geometry(
    best_ez: Path,
    params_json: Path,
    *,
    verdict_json: Optional[Path] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    n_x: int = 2001,
    smooth_window: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Build a robust geometry dict on a uniform x grid.
    Outputs: eta,x,a,z,E,H0_km_s_Mpc,Hconf,HHp,R
    """
    z_raw, E_raw = _read_best_ez(best_ez)

    m = np.isfinite(z_raw) & np.isfinite(E_raw)
    z_raw = z_raw[m]
    E_raw = E_raw[m]
    if z_raw.size < 3:
        raise ValueError("best_Ez has too few finite points after sanitation.")

    a_raw = 1.0 / (1.0 + z_raw)
    m2 = np.isfinite(a_raw) & (a_raw > 0) & (E_raw > 0)
    z_raw = z_raw[m2]
    E_raw = E_raw[m2]
    if z_raw.size < 3:
        raise ValueError("best_Ez has too few valid points after enforcing a>0 and E>0.")

    _, idx = np.unique(z_raw, return_index=True)
    idx = np.sort(idx)
    z = z_raw[idx]
    E = E_raw[idx]

    a = 1.0 / (1.0 + z)
    x = np.log(a)
    order = np.argsort(x)
    x, z, a, E = x[order], z[order], a[order], E[order]

    if not is_strictly_increasing(x):
        raise ValueError("Non-strictly-increasing x detected after sorting; check best_Ez duplicates/quality.")

    if int(n_x) < 3:
        raise ValueError("n_x must be >= 3.")
    xmin = float(x[0]) if x_min is None else float(x_min)
    xmax = float(x[-1]) if x_max is None else float(x_max)
    if not (xmin < xmax):
        raise ValueError(f"x_min must be < x_max (got {xmin} >= {xmax}).")

    xg = np.linspace(xmin, xmax, int(n_x), dtype=float)
    Eg = np.interp(xg, x, E, left=float(E[0]), right=float(E[-1]))
    x = xg
    a = np.exp(x)
    z = (1.0 / a) - 1.0
    E = Eg

    if smooth_window and int(smooth_window) > 0:
        E = moving_average(E, int(smooth_window))

    H0 = _read_H0(params_json, verdict_json=verdict_json)
    Hphys = (H0 * E) / C_KM_S
    Hconf = a * Hphys

    if not np.all(np.isfinite(Hconf)) or np.any(Hconf <= 0):
        raise ValueError("Non-finite or non-positive ℋ detected; check H0/E inputs.")

    eta = cumulative_trapz(x, 1.0 / Hconf)

    Hc_for = Hconf
    if smooth_window and int(smooth_window) > 0:
        Hc_for = moving_average(Hconf, int(smooth_window))

    dHdx = np.gradient(Hc_for, x, edge_order=2)
    HHp = Hc_for * dHdx

    R = 6.0 * (HHp + Hc_for**2) / (a**2)

    if not np.all(np.isfinite(eta)) or not np.all(np.isfinite(R)):
        raise ValueError("Non-finite η or R detected after geometry build.")

    return {
        "eta": eta,
        "x": x,
        "a": a,
        "z": z,
        "E": E,
        "H0_km_s_Mpc": np.array([H0], dtype=float),
        "Hconf": Hconf,
        "HHp": HHp,
        "R": R,
    }

# ============================================================
# EB1 — Auxiliary background solver
# ============================================================

@dataclass
class IntegratorCfg:
    method: str = "rk4"
    rk4_geom: str = "interp"
    stop_on_nonfinite: bool = True
    maxabs_stop: float = 1e200

def _align_profiles(
    x_geom: np.ndarray,
    x_p: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    m: np.ndarray,
    policy: str = "strict",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align EB parameter profiles (beta/gamma/m) onto geometry grid.
    Supports 0D (scalar), 1D profiles [N] and 2D profiles [N, nf].
    """
    policy = str(policy).lower()
    if policy not in ("strict", "interp"):
        raise ValueError("eb_x_policy must be 'strict' or 'interp'.")

    x_geom = np.asarray(x_geom, float)

    if x_p is None:
        x_p = x_geom
    else:
        x_p = np.asarray(x_p, float)

    if policy == "strict":
        if len(x_geom) != len(x_p):
            raise ValueError(f"Strict x policy: len(x_geom)={len(x_geom)} != len(x_params)={len(x_p)}")
        if np.max(np.abs(x_geom - x_p)) > 1e-10:
            raise ValueError("Strict x policy: x grids differ (max |Δx| > 1e-10).")
        return beta, gamma, m

    if np.any(np.diff(x_p) <= 0):
        order = np.argsort(x_p)
        x_p = x_p[order]
        beta = beta[order, ...] if beta.ndim > 1 else beta[order] if beta.ndim == 1 else beta
        gamma = gamma[order, ...] if gamma.ndim > 1 else gamma[order] if gamma.ndim == 1 else gamma
        m = m[order, ...] if m.ndim > 1 else m[order] if m.ndim == 1 else m
        if np.any(np.diff(x_p) <= 0):
            raise ValueError("Cannot interpolate: x in EB params is not strictly increasing.")

    def _interp(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, float)
        if arr.ndim == 0:  # Si c'est un scalaire
            return np.full_like(x_geom, arr)
        elif arr.ndim == 1:
            return np.interp(x_geom, x_p, arr, left=float(arr[0]), right=float(arr[-1]))
        elif arr.ndim == 2:
            nf = arr.shape[1]
            out = np.zeros((x_geom.size, nf), dtype=float)
            for j in range(nf):
                col = arr[:, j]
                out[:, j] = np.interp(x_geom, x_p, col, left=float(col[0]), right=float(col[-1]))
            return out
        else:
            raise ValueError(f"Unsupported array dimension: {arr.ndim}")

    return _interp(beta), _interp(gamma), _interp(m)

def solve_aux_background(
    geom: Dict[str, np.ndarray],
    x_params: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    m: np.ndarray,
    *,
    U0: float,
    Up0: float,
    cfg: IntegratorCfg,
    eb_x_policy: str = "strict",
    n_fields: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """EB1 auxiliary background solver.

    v10.6 semantics:
    - beta/gamma/m are PROFILES along x (aligned to geometry grid).
      Shapes supported: [N] or [N, nf]
    - U is scalar over x.
    - phi is [N] if nf=1, else [N, nf].

    This fixes the old ambiguity that created phi of shape (N×N).
    """
    eta = np.asarray(geom["eta"], float)
    xg = np.asarray(geom["x"], float)
    a = np.asarray(geom["a"], float)
    H = np.asarray(geom["Hconf"], float)
    R = np.asarray(geom["R"], float)

    if np.any(np.diff(eta) <= 0) or not np.all(np.isfinite(eta)):
        raise ValueError("Geometry has non-increasing or non-finite eta.")

    beta, gamma, m = _align_profiles(xg, x_params, beta, gamma, m, policy=eb_x_policy)

    nf = 1 if beta.ndim == 1 else int(beta.shape[1])

    if n_fields is not None:
        k = int(n_fields)
        if k < 1:
            raise ValueError("n_fields must be >= 1")
        if nf > 1:
            nf = min(nf, k)
            beta = beta[:, :nf]
            gamma = gamma[:, :nf]
            m = m[:, :nf]
        else:
            nf = 1

    N = int(eta.size)
    U = np.zeros(N, dtype=float)
    Up = np.zeros(N, dtype=float)

    if nf == 1:
        phi = np.zeros(N, dtype=float)
        phip = np.zeros(N, dtype=float)
    else:
        phi = np.zeros((N, nf), dtype=float)
        phip = np.zeros((N, nf), dtype=float)

    U[0] = float(U0)
    Up[0] = float(Up0)

    def _check(i: int) -> None:
        if not cfg.stop_on_nonfinite:
            return
        if (not np.isfinite(U[i])) or (not np.isfinite(Up[i])):
            raise FloatingPointError(f"Non-finite detected at step i={i}, x={xg[i]:.6g}, eta={eta[i]:.6g}")
        if nf == 1:
            if (not np.isfinite(phi[i])) or (not np.isfinite(phip[i])):
                raise FloatingPointError(f"Non-finite detected at step i={i}, x={xg[i]:.6g}, eta={eta[i]:.6g}")
            mx = max(abs(phi[i]), abs(phip[i]), abs(U[i]), abs(Up[i]))
        else:
            if (not np.all(np.isfinite(phi[i, :]))) or (not np.all(np.isfinite(phip[i, :]))):
                raise FloatingPointError(f"Non-finite detected at step i={i}, x={xg[i]:.6g}, eta={eta[i]:.6g}")
            mx = max(float(np.max(np.abs(phi[i, :]))), float(np.max(np.abs(phip[i, :]))), abs(U[i]), abs(Up[i]))
        if mx > float(cfg.maxabs_stop):
            raise FloatingPointError(f"Runaway detected at step i={i}, x={xg[i]:.6g}: max|state|={mx:.3e}")

    def geom_at(i: int, frac: float):
        if cfg.method != "rk4" or cfg.rk4_geom == "frozen" or frac == 0.0:
            return float(a[i]), float(H[i]), float(R[i])
        if i >= N - 1:
            return float(a[-1]), float(H[-1]), float(R[-1])
        j = i + 1
        aa = (1.0 - frac) * a[i] + frac * a[j]
        HH = (1.0 - frac) * H[i] + frac * H[j]
        RR = (1.0 - frac) * R[i] + frac * R[j]
        return float(aa), float(HH), float(RR)

    def prof_at(i: int, frac: float):
        if cfg.method != "rk4" or frac == 0.0 or i >= N - 1:
            return beta[i], gamma[i], m[i]
        j = i + 1
        b = (1.0 - frac) * beta[i] + frac * beta[j]
        g = (1.0 - frac) * gamma[i] + frac * gamma[j]
        mm = (1.0 - frac) * m[i] + frac * m[j]
        return b, g, mm

    def rhs(i: int, frac: float, Uv, Upv, phiv, phipv):
        ai, Hi, Ri = geom_at(i, frac)
        bi, gi, mi = prof_at(i, frac)

        if nf == 1:
            gamma_phi = float(gi) * float(phiv)
            Uppv = -2.0 * Hi * Upv - (ai ** 2) * (Ri + gamma_phi)
            phippv = (
                -2.0 * Hi * phipv
                - (ai ** 2) * ((float(mi) ** 2) * phiv)
                - (ai ** 2) * (float(bi) * Ri + float(gi) * Uv)
            )
            return Upv, Uppv, phipv, phippv

        gamma_dot_phi = float(np.dot(gi, phiv))
        Uppv = -2.0 * Hi * Upv - (ai ** 2) * (Ri + gamma_dot_phi)
        phippv = (
            -2.0 * Hi * phipv
            - (ai ** 2) * ((mi ** 2) * phiv)
            - (ai ** 2) * (bi * Ri + gi * Uv)
        )
        return Upv, Uppv, phipv, phippv

    method = cfg.method.lower().strip()
    if method not in {"euler", "rk4"}:
        raise ValueError("--integrator must be 'euler' or 'rk4'")

    for i in range(N - 1):
        h = float(eta[i + 1] - eta[i])
        if h <= 0:
            raise RuntimeError("Non-increasing eta grid; check geometry build.")

        if method == "euler":
            dU, dUp, dphi, dphip = rhs(i, 0.0, U[i], Up[i], phi[i], phip[i])
            U[i + 1] = U[i] + h * dU
            Up[i + 1] = Up[i] + h * dUp
            phi[i + 1] = phi[i] + h * dphi
            phip[i + 1] = phip[i] + h * dphip
            _check(i + 1)
            continue

        k1 = rhs(i, 0.0, U[i], Up[i], phi[i], phip[i])
        k2 = rhs(i, 0.5,
                 U[i] + 0.5*h*k1[0], Up[i] + 0.5*h*k1[1],
                 phi[i] + 0.5*h*k1[2], phip[i] + 0.5*h*k1[3])
        k3 = rhs(i, 0.5,
                 U[i] + 0.5*h*k2[0], Up[i] + 0.5*h*k2[1],
                 phi[i] + 0.5*h*k2[2], phip[i] + 0.5*h*k2[3])
        k4 = rhs(i, 1.0,
                 U[i] + h*k3[0], Up[i] + h*k3[1],
                 phi[i] + h*k3[2], phip[i] + h*k3[3])

        U[i + 1] = U[i] + (h/6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        Up[i + 1] = Up[i] + (h/6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        phi[i + 1] = phi[i] + (h/6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        phip[i + 1] = phip[i] + (h/6.0) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

        _check(i + 1)

    if nf == 1:
        gamma_phi = gamma * phi
        Upp = -2.0 * H * Up - (a ** 2) * (R + gamma_phi)
        phipp = -2.0 * H * phip - (a ** 2) * ((m ** 2) * phi) - (a ** 2) * (beta * R + gamma * U)
    else:
        gamma_dot_phi = (phi * gamma).sum(axis=1)
        Upp = -2.0 * H * Up - (a ** 2) * (R + gamma_dot_phi)
        phipp = (
            -2.0 * H[:, None] * phip
            - (a[:, None] ** 2) * ((m ** 2) * phi)
            - (a[:, None] ** 2) * (beta * R[:, None] + gamma * U[:, None])
        )

    return {
        "eta": eta,
        "x": xg,
        "a": a,
        "Hconf": H,
        "R": R,
        "beta": beta,
        "gamma": gamma,
        "m": m,
        "U": U,
        "Up": Up,
        "Upp": Upp,
        "phi": phi,
        "phip": phip,
        "phipp": phipp,
        "integrator": np.array([method], dtype="U8"),
        "rk4_geom": np.array([cfg.rk4_geom], dtype="U8"),
        "eb_x_policy": np.array([eb_x_policy], dtype="U8"),
        "n_fields": np.array([nf], dtype=int),
        "script_version": np.array([SCRIPT_VERSION], dtype="U16"),
    }

# ============================================================
# Manifests (with SHA256)
# ============================================================

def make_common_manifest_header(stage: str) -> dict:
    return {
        "script": "GeNeSyS_v10_6_EB01.py",
        "script_version": SCRIPT_VERSION,
        "stage": stage,
        "timestamp_unix": int(time.time()),
    }

# ============================================================
# CLI commands
# ============================================================

def cmd_eb01(args) -> None:
    refs = resolve_pipeline_refs(args.best_ez, args.params_json, args.verdict_json, args.scan_meta_json)
    geom = build_geometry(
        refs.best_ez,
        refs.params_json,
        verdict_json=refs.verdict_json,
        x_min=args.x_min,
        x_max=args.x_max,
        n_x=args.n_x,
        smooth_window=args.smooth_window,
    )
    geom_out = Path(args.geom_out_npz).resolve()
    ensure_dir(geom_out)
    np.savez(geom_out, **geom)
    log_info(f"[EB01] geom saved: {geom_out}")

    # EB0 manifest written immediately
    eb0_manifest = Path(args.manifest_eb0_json).resolve()
    ensure_dir(eb0_manifest)
    eb0_payload = {
        "stage": "EB0",
        "timestamp_unix": int(time.time()),
        "status": "SUCCESS",
        "script_version": SCRIPT_VERSION,
        "inputs": {
            "best_ez": str(refs.best_ez),
            "params_json": str(refs.params_json),
            "verdict_json": str(refs.verdict_json) if refs.verdict_json else None,
            "scan_meta_json": str(refs.scan_meta_json) if refs.scan_meta_json else None,
        },
        "outputs": {"geom_npz": str(geom_out)},
        "geometry": {
            "n": int(len(geom["x"])),
            "x_min": float(geom["x"][0]),
            "x_max": float(geom["x"][-1]),
            "eta_min": float(geom["eta"][0]),
            "eta_max": float(geom["eta"][-1]),
            "requested": {"n_x": int(args.n_x), "x_min": args.x_min, "x_max": args.x_max},
            "smooth_window": int(args.smooth_window),
        },
        "sha256": {
            "geom_npz": _sha256_file(geom_out),
            "best_ez": _sha256_file(refs.best_ez),
            "params_json": _sha256_file(refs.params_json),
            "verdict_json": _sha256_file(refs.verdict_json) if refs.verdict_json else None,
        },
    }
    eb0_manifest.write_text(json.dumps(eb0_payload, indent=2), encoding="utf-8")
    log_info(f"[EB01] manifest EB0 saved: {eb0_manifest}")

    # EB1
    eb_json = Path(args.eb_params_json).resolve()
    cfg = IntegratorCfg(
        method=args.integrator,
        rk4_geom=args.rk4_geom,
        stop_on_nonfinite=not args.no_stop_on_nonfinite,
        maxabs_stop=float(args.maxabs_stop),
    )

    aux_out = Path(args.aux_out_npz).resolve()
    eb1_manifest = Path(args.manifest_eb1_json).resolve()
    ensure_dir(aux_out)
    ensure_dir(eb1_manifest)

    eb1_payload = {
        "stage": "EB1",
        "timestamp_unix": int(time.time()),
        "status": "INITIALIZED",
        "script_version": SCRIPT_VERSION,
        "inputs": {
            "geom_npz": str(geom_out),
            "eb_params_json": str(eb_json),
            "integrator": args.integrator,
            "rk4_geom": args.rk4_geom,
            "eb_x_policy": args.eb_x_policy,
            "n_fields": args.n_fields,
            "U0": float(args.U0),
            "Up0": float(args.Up0),
            "maxabs_stop": float(args.maxabs_stop),
            "stop_on_nonfinite": (not args.no_stop_on_nonfinite),
        },
        "outputs": {"aux_npz": str(aux_out)},
        "gamma_semantics": "profile_from_input_json (not fixed internally)",
        "sha256": {
            "geom_npz": _sha256_file(geom_out),
            "eb_params_json": _sha256_file(eb_json),
        },
    }

    try:
        x_p, beta, gamma, m, _ = read_eb_params_v10_6(eb_json)

        # Si x_p est None, on utilise la grille de géométrie
        if x_p is None:
            x_p = geom["x"]

        aux = solve_aux_background(
            geom,
            x_p, beta, gamma, m,
            U0=float(args.U0),
            Up0=float(args.Up0),
            cfg=cfg,
            eb_x_policy=args.eb_x_policy,
            n_fields=args.n_fields,
        )

        np.savez(aux_out, **aux)
        log_info(f"[EB01] aux saved: {aux_out}")

        eb1_payload["status"] = "SUCCESS"
        eb1_payload["sha256"]["aux_npz"] = _sha256_file(aux_out)
        eb1_payload["geometry"] = {
            "n": int(len(geom["x"])),
            "x_min": float(geom["x"][0]),
            "x_max": float(geom["x"][-1]),
        }

        eb1_manifest.write_text(json.dumps(eb1_payload, indent=2), encoding="utf-8")
        log_info(f"[EB01] manifest EB1 saved: {eb1_manifest}")
        return

    except Exception as e:
        eb1_payload["status"] = "ERROR"
        eb1_payload["error"] = str(e)
        eb1_payload["traceback"] = traceback.format_exc(limit=30)
        eb1_manifest.write_text(json.dumps(eb1_payload, indent=2), encoding="utf-8")
        log_error(f"[EB01] EB1 failed: {e}")
        raise

# ============================================================
# Parser
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="GeNeSyS_v10_6_EB01.py")

    p.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path (recommended for traceability).",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # EB01
    p01 = sub.add_parser("eb01", help="Run EB0 then EB1 (chained)")
    p01.add_argument("--best-ez", default=None)
    p01.add_argument("--params-json", default=None)
    p01.add_argument("--verdict-json", default=None)
    p01.add_argument("--scan-meta-json", default=None)
    p01.add_argument("--x-min", dest="x_min", type=float, default=None)
    p01.add_argument("--x-max", dest="x_max", type=float, default=None)
    p01.add_argument("--n-x", dest="n_x", type=int, default=2001)
    p01.add_argument("--smooth-window", type=int, default=0)
    p01.add_argument("--geom-out-npz", required=True)

    p01.add_argument("--eb-params-json", required=True)
    p01.add_argument("--integrator", choices=["euler", "rk4"], default="rk4")
    p01.add_argument("--rk4-geom", choices=["frozen", "interp"], default="interp")
    p01.add_argument("--eb-x-policy", choices=["strict", "interp"], default="strict")
    p01.add_argument("--n-fields", type=int, default=None)
    p01.add_argument("--U0", type=float, default=0.0)
    p01.add_argument("--Up0", type=float, default=0.0)
    p01.add_argument("--maxabs-stop", type=float, default=1e200)
    p01.add_argument("--no-stop-on-nonfinite", action="store_true")
    p01.add_argument("--aux-out-npz", required=True)

    p01.add_argument("--manifest-eb0-json", required=True)
    p01.add_argument("--manifest-eb1-json", required=True)
    p01.set_defaults(func=cmd_eb01)

    return p

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log_path = Path(args.log_file).resolve() if args.log_file else None
    setup_logging(log_path)

    if args.cmd in ("eb01",):
        if not (getattr(args, "verdict_json", None) or (getattr(args, "best_ez", None) and getattr(args, "params_json", None))):
            die("Provide --verdict-json OR provide both --best-ez and --params-json.")

    args.func(args)

if __name__ == "__main__":
    main()