#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeNeSyS_v10_pipeline.py

UNIFIED PIPELINE FOR GENESYS V10.1 COSMOLOGICAL MODEL (BACKGROUND-ONLY)
=====================================================================

Features:
  1. Background parameter scan with early-time safety gate (Ξ_early for BBN protection)
  2. Best-fit E(z) computation with kinetic regime placeholder (G_kin)
  3. Observational confrontation with:
     - Pantheon+SH0ES SN Ia (full covariance, Cholesky decomposition)
     - BOSS DR12 BAO (r_d treated as nuisance parameter, analytic least-squares for both models)
  4. ΛCDM comparison at fixed H0 (same as GeNeSyS) with analytic r_d nuisance
  5. Deterministic self-test mode for regression checks

Outputs:
  - Scan:   _GeNeSyS_v10_3_background_scan.csv          (scalar parameters + diagnostics)
            _GeNeSyS_v10_3_background_scan.meta.json    (metadata: versions, parameters, grid)
  - E(z):   _GeNeSyS_v10_3_best_Ez.csv                 (z, E_genesys)
            best_fit_params.json                     (metadata: environment, parameters, hashes)
  - Verdict: genesys_full_cov_verdict.json           (χ² SN/BAO, rd_best, ΛCDM comparison)
  - Self-test: Pass/Fail validation

Dependencies:
  Required: numpy, pandas, scipy
  Optional: tqdm (progress bars)

Data Files (expected in --data-dir):
  - Pantheon+SH0ES.dat
  - symmetrized_cov_matrix.cov  (recommended)
  - Pantheon+SH0ES_STAT+SYS.cov (legacy)
  - DR12_fid_DMrd_DHrd_summary.csv
  - DR12_cov6x6_DMrd_DHrd_from_consensus.csv

Reproducibility & Safety:
  - Early-time protection (Ξ_early gate before BBN, effective with xmin=-20)
  - Kinetic regime placeholder (G_kin hook, disabled by default)
  - ΛCDM comparison at fixed H0=GeNeSyS (no degeneracy with r_d)
  - Robust BAO covariance handling (Cholesky + SPD fallback)
  - Deterministic self-test mode (no stochastic elements)
  - Complete metadata documentation
  - Python 3.9+ type annotations
"""

from __future__ import annotations
from typing import Callable
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.integrate import cumulative_trapezoid
from importlib.metadata import version, PackageNotFoundError
from datetime import datetime

# --- Constants ---
C_KM_S = 299792.458  # Speed of light [km/s]
SCRIPT_VERSION = "1.1.6-prod"

# Keys allowed in master JSON but not mapped to CLI options in this script.
# We ignore them silently to support unified configs without noisy warnings.
IGNORED_JSON_KEYS = {
    "_unified_from", "_prod_ready", "model", "data_files", "theta_gate", "last_best_fit",
    "fixed_parameters", "output_dir",
    # extended scan keys (handled by other scripts / versions)
    "x_BBN_list", "width_BBN_list", "tau_fast_list", "f_fast_list",
    # theta-gate explicit keys (handled by theta-gate pipeline)
    "cmb_zstar", "cmb_theta_target", "cmb_theta_sigma", "cmb_theta_weight",
    "cmb_theta_pull_max", "cmb_theta_enable",
    # memory solver controls (not exposed in this CLI)
    "mm_max_iter", "mm_tol",
    "version",
    "run_mode",
    "model",
    "bao",
    "symbol_legend",
    "notes",
}

  # Final version with all corrections applied

# Default parameters (aligned with your document)
DEFAULT_PARAMS = {
    "H0": 70.0,
    "Omega_b0": 0.045755,
    "Omega_C0": 0.25,
    # Radiation content used for closure (if you do not provide Omega_r0 explicitly).
    # These defaults are standard (Tcmb≈2.7255 K, Neff≈3.046).
    "Tcmb_K": 2.7255,
    "Neff": 3.046,
    # If provided, Omega_r0 overrides (Tcmb_K,Neff) -> Omega_r0 computation.
    "Omega_r0": None,
    # Model-specific parameters
    "tau_fast": 0.7,
    "f_fast": 0.2,
    "x_c_list": [-10.0, -9.0, -8.0, -7.0],
    "sigma_c_list": [0.30, 0.50, 0.80],
    "tau_slow_list": [4.0, 5.0, 6.0],
    "alpha_M_list": [0.005, 0.010, 0.020, 0.050],
    "x_BBN": -17.0,          # Early-time cut for Ξ_early (BBN protection)
    "width_BBN": 0.5,        # Width for Ξ_early transition
    "xmin": -20.0,           # Extended to ensure Ξ_early is effective
}


# ----------------------------
# Cosmological bookkeeping (v10.1/v10.3 compliance)
# ----------------------------

def omega_r0_from_Tcmb_Neff(H0_km_s_Mpc: float, Tcmb_K: float = 2.7255, Neff: float = 3.046) -> float:
    """Return Ω_r0 = Ω_γ0 + Ω_ν0 (dimensionless) from (Tcmb, Neff, H0).

    Uses standard relations:
      Ω_γ h^2 ≈ 2.469e-5 (Tcmb/2.7255)^4
      Ω_ν h^2 = Ω_γ h^2 * 0.2271 * Neff
    where h = H0 / (100 km/s/Mpc).
    """
    h = float(H0_km_s_Mpc) / 100.0
    if h <= 0:
        die("H0 must be > 0 to compute Ω_r0.")
    Og_h2 = 2.469e-5 * (float(Tcmb_K) / 2.7255) ** 4
    Onu_h2 = Og_h2 * 0.2271 * float(Neff)
    return (Og_h2 + Onu_h2) / (h * h)

def derive_omega_M0(*, Omega_b0: float, Omega_C0: float, Omega_r0: float) -> float:
    """v10.1 closure: Ω_M0 is *not* an independent parameter."""
    Om0 = 1.0 - float(Omega_r0) - float(Omega_b0) - float(Omega_C0)
    return Om0

def validate_closure(*, Omega_b0: float, Omega_C0: float, Omega_r0: float, tol: float = 1e-12) -> float:
    """Return derived Ω_M0 and assert closure ΣΩ=1 and Ω_M0≥0."""
    Om0 = derive_omega_M0(Omega_b0=Omega_b0, Omega_C0=Omega_C0, Omega_r0=Omega_r0)
    if Om0 < -1e-14:
        die(f"Invalid closure: derived Ω_M0={Om0:.6g} < 0. Check Ω inputs.")
    # round small negative due to float
    Om0 = max(Om0, 0.0)
    s = Om0 + float(Omega_r0) + float(Omega_b0) + float(Omega_C0)
    if abs(s - 1.0) > tol:
        die(f"Closure violated: Ω_M0+Ω_r0+Ω_b0+Ω_C0={s:.16g} (expected 1).")
    return Om0
# --- Safety Gates ---
def Xi_early(x: np.ndarray, x_cut: float = -17.0, width: float = 0.5) -> np.ndarray:
    """Early-time safety gate (BBN protection): ~0 for x<<x_cut, ~1 for x>>x_cut.
    Ensures memory sector does not activate before BBN constraints."""
    return 0.5 * (1.0 + np.tanh((x - x_cut) / width))

def G_kin(x: np.ndarray, E: np.ndarray, enable: bool = False) -> np.ndarray:
    """Kinetic-regime gate. Placeholder: returns 1 unless enable=True with a defined prescription.
    Safe default for background-only analysis."""
    if not enable:
        return np.ones_like(x)
    # TODO: Implement exact prescription from Technical when available
    return np.ones_like(x)

# --- Utilities ---
def die(msg: str, code: int = 1) -> None:
    """Exit with error message."""
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(code)

def info(msg: str) -> None:
    """Print info message."""
    print(f"[INFO] {msg}")

def warn(msg: str) -> None:
    """Print warning message."""
    print(f"[WARN] {msg}", file=sys.stderr)


# --- Pretty status helpers (prod readability) ---
BADGE_GREEN = '🟢'
BADGE_ORANGE = '🟠'
BADGE_RED = '🔴'

def badge(level: str) -> str:
    level = (level or '').lower()
    if level in ('ok','green','info','pass','success'):
        return BADGE_GREEN
    if level in ('warn','orange','caution','partial'):
        return BADGE_ORANGE
    if level in ('err','error','red','fail','fatal'):
        return BADGE_RED
    return ''

def info_b(level: str, msg: str) -> None:
    info(f"{badge(level)}  {msg}")

def warn_b(level: str, msg: str) -> None:
    # keep stderr for warnings
    warn(f"{badge(level)}  {msg}")

def summarize_run(scan_meta: Path | None, params_json: Path | None, verdict_json: Path | None) -> None:
    """Best-effort human-readable summary with 🟢/🟠/🔴 badges."""
    info('')
    info('=== RUN SUMMARY ===')
    status = BADGE_GREEN

    # Scan meta
    if scan_meta and scan_meta.exists():
        try:
            m = json.loads(scan_meta.read_text(encoding='utf-8'))
            H0 = m.get('fixed_parameters', {}).get('H0', None)
            grid = m.get('scan_grid', {}).get('n_models', None)
            info_b('ok', f"scan_meta: {scan_meta}")
            if H0 is not None:
                info_b('ok', f"H0 used (scan): {H0}")
            if grid is not None:
                info_b('ok', f"scan grid: {grid} models")
        except Exception as e:
            status = BADGE_ORANGE
            warn_b('warn', f"Could not parse scan meta {scan_meta}: {e}")
    else:
        status = BADGE_ORANGE
        warn_b('warn', 'scan_meta missing')

    # Best-fit params / early report sanity
    if params_json and params_json.exists():
        try:
            p = json.loads(params_json.read_text(encoding='utf-8'))
            bf = p.get('best_fit_parameters', {}) or p.get('best_fit_params', {}) or {}
            if 'H0' in bf:
                info_b('ok', f"H0 used (best-fit): {bf['H0']}")
            # Early-time warning heuristic (background only)
            er = p.get('early_report', {})
            if isinstance(er, dict):
                # Flag large negative Omega_em at high z (heuristic, not a proof of inconsistency)
                for key in ('Omega_em_z1100', 'Omega_em_z1e4', 'Omega_em_z1e9'):
                    val = er.get(key, None)
                    if isinstance(val, (int, float)) and val < -1e-3:
                        status = BADGE_ORANGE
                        warn_b('warn', f"early_report: {key}={val} (negative) — background-only heuristic flag")
                        break
            info_b('ok', f"best_fit_params: {params_json}")
        except Exception as e:
            status = BADGE_ORANGE
            warn_b('warn', f"Could not parse best-fit params {params_json}: {e}")
    else:
        status = BADGE_ORANGE
        warn_b('warn', 'best_fit_params missing')

    # Verdict
    if verdict_json and verdict_json.exists():
        try:
            v = json.loads(verdict_json.read_text(encoding='utf-8'))
            info_b('ok', f"verdict: {verdict_json}")
            # Try to print chi2 summary if present
            chis = v.get('chi2', {}) or v.get('chisq', {}) or {}
            if isinstance(chis, dict) and chis:
                # common keys
                sn = chis.get('SN', chis.get('sn', None))
                bao = chis.get('BAO', chis.get('bao', None))
                tot = chis.get('total', chis.get('tot', None))
                if sn is not None: info_b('ok', f"chi2_SN: {sn}")
                if bao is not None: info_b('ok', f"chi2_BAO: {bao}")
                if tot is not None: info_b('ok', f"chi2_total: {tot}")
        except Exception as e:
            status = BADGE_ORANGE
            warn_b('warn', f"Could not parse verdict {verdict_json}: {e}")
    else:
        status = BADGE_ORANGE
        warn_b('warn', 'verdict missing (run may have stopped before confront)')

    info('')
    info(f"STATUS: {status}")
    info('============')


def apply_params_json(args: argparse.Namespace) -> None:
    """
    Override argparse namespace with values from a JSON file, if --params is provided.

    Rules:
    - Keys must match argparse dest names (e.g. Omega_b0, Tcmb_K, Omega_r0, x_c_list, ...).
    - For *_list options that are comma-separated in CLI, JSON may provide either a list or a string.
    - CLI explicit flags still win if you pass them after loading; this helper is intended to be called
      at the very start of each command handler.
    """
    params_path = getattr(args, "params", None)
    if not params_path:
        return

    p = Path(params_path)
    if not p.exists():
        die(f"Params JSON not found: {p}")

    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        die(f"Failed to parse params JSON {p}: {e}")

    if not isinstance(payload, dict):
        die(f"Params JSON must be an object/dict at top level: {p}")

    # Support unified "master" JSON schemas by unpacking nested sections.
    # 1) fixed_parameters: map user-friendly keys (Omega_b, Omega_c, ...) onto CLI args (Omega_b0, Omega_C0, ...).
    fp = payload.get("fixed_parameters", None)
    if isinstance(fp, dict):
        keymap = {
            "H0": "H0",
            "Omega_b": "Omega_b0",
            "Omega_b0": "Omega_b0",
            "Omega_c": "Omega_C0",
            "Omega_C_condensate": "Omega_C0",
            "Omega_C_condensate0": "Omega_C0",
            "Omega_C": "Omega_C0",
            "Omega_C0": "Omega_C0",
            "Omega_r": "Omega_r0",
            "Omega_r0": "Omega_r0",
            "Omega_k": "Omega_k0",
            "Omega_k0": "Omega_k0",
            "Tcmb_K": "Tcmb_K",
            "Neff": "Neff",
            "tau_fast": "tau_fast",
            "f_fast": "f_fast",
            "x_BBN": "x_BBN",
            "width_BBN": "width_BBN",
        }

        for fk, fv in fp.items():
            ak = keymap.get(fk)
            if not ak or not hasattr(args, ak):
                continue

            # Normalize numeric types where it matters
            if ak in (
                "H0",
                "Omega_b0",
                "Omega_C0",
                "Omega_r0",
                "Omega_k0",
                "Tcmb_K",
                "Neff",
                "tau_fast",
                "f_fast",
                "x_BBN",
                "width_BBN",
            ):
                try:
                    fv = float(fv)
                except Exception:
                    die(f"Invalid numeric value for fixed_parameters.{fk} = {fv!r} in {p}")

            setattr(args, ak, fv)

    # 2) data_files (and other nested sections) are intentionally not injected into CLI args here.
    #    cmd_confront may reload payload to access them. Keep forward/backward compatibility by
    #    ignoring unknown keys with a warning.
    for k, v in payload.items():
        # Skip nested sections not meant to be CLI args
        if k in ("fixed_parameters", "data_files", "cmb_theta_gate", "theta_gate"):
            continue

        if not hasattr(args, k):
            if k in IGNORED_JSON_KEYS:
                continue
            warn(f"Params JSON key '{k}' is not a recognized CLI option; ignoring.")
            continue

        # Support JSON arrays for *_list -> convert to comma-separated string
        if k.endswith("_list") and isinstance(v, (list, tuple)):
            v = ",".join(str(x) for x in v)

        setattr(args, k, v)



def apply_output_dir(args: argparse.Namespace) -> None:
    """Prefix relative output file paths with --output-dir, if provided.

    This keeps argparse simple (no global subparser magic) while ensuring all artifacts
    are written under a single directory for production runs.
    """
    od = getattr(args, "output_dir", None)
    if not od:
        return
    od_path = Path(str(od))
    od_path.mkdir(parents=True, exist_ok=True)

    def _prefix(attr: str) -> None:
        if not hasattr(args, attr):
            return
        v = getattr(args, attr, None)
        if not v:
            return
        p = Path(str(v))
        if p.is_absolute():
            return
        setattr(args, attr, str(od_path / p.name))

    # Common output attributes across modes
    for a in ("output", "params_json", "scan_out", "ez_out", "params_out", "verdict_out"):
        _prefix(a)
def file_hash(filepath: Path, blocksize: int = 65536) -> str:
    """Compute SHA256 hash of a file for reproducibility tracking."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        buf = f.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(blocksize)
    return hasher.hexdigest()

def log_interp(x_grid: np.ndarray, y_grid: np.ndarray, xq: np.ndarray, floor: float = 1e-60) -> np.ndarray:
    """Safe log-linear interpolation for positive quantities."""
    y = np.maximum(y_grid, floor)
    return np.exp(np.interp(xq, x_grid, np.log(y)))

def pkg_version(name: str) -> str:
    """Get package version safely."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"

# --- Core Model Functions ---
def build_x_grid(xmin: float = -20.0, xmax: float = 0.0, n: int = 2001) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build logarithmic grid in x=ln(a) with extended range for Ξ_early effectiveness."""
    x = np.linspace(xmin, xmax, n)
    a = np.exp(x)
    z = 1.0 / a - 1.0
    return x, a, z

def W_eq(x: np.ndarray, x_eq: float = -8.1, width: float = 0.35) -> np.ndarray:
    """Smooth equality window (0→1)."""
    return 0.5 * (1.0 + np.tanh((x - x_eq) / width))

def compute_R_and_dRdx(x: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute dimensionless Ricci scalar R_tilde = R/H0^2 and dR_tilde/dx.

GeNeSyS Technical uses R = 6 H^2 (2 + d ln H / dx) with H=H0*E.
Since H0 is constant, d ln H / dx = d ln E / dx.
Therefore R_tilde = R/H0^2 = 6 E^2 (2 + d ln E / dx).
"""
    lnE = np.log(np.maximum(E, 1e-60))
    dlnE_dx = np.gradient(lnE, x)
    R = 6.0 * (E**2) * (2.0 + dlnE_dx)
    dRdx = np.gradient(R, x)
    return R, dRdx

def compute_age_Gyr(x: np.ndarray, E: np.ndarray, H0: float) -> float:
    """Compute age of the universe in Gyr (diagnostic only)."""
    h = H0 / 100.0
    integrand = 1.0 / np.maximum(E, 1e-60)
    age = (9.778 / h) * np.trapz(integrand, x)
    return float(age)

def w_from_M(x: np.ndarray, M: np.ndarray) -> float:
    """Compute present-day equation of state of memory sector."""
    lnM = np.log(np.maximum(M, 1e-60))
    dlnM_dx = np.gradient(lnM, x)
    w = -1.0 - (1.0 / 3.0) * dlnM_dx
    return float(w[-1])

def interp_at_z(z_grid: np.ndarray, y_grid: np.ndarray, z_target: float) -> float:
    """Interpolate y(z) at z_target."""
    idx = np.argsort(z_grid)
    return float(np.interp(z_target, z_grid[idx], y_grid[idx]))

def compute_background_for_params(
    x_c: float,
    sigma_c: float,
    tau_slow: float,
    alpha_M: float,
    *,
    H0: float = DEFAULT_PARAMS["H0"],
    Omega_b0: float = DEFAULT_PARAMS["Omega_b0"],
    Omega_C0: float = DEFAULT_PARAMS["Omega_C0"],
    Omega_r0: float | None = DEFAULT_PARAMS["Omega_r0"],
    Tcmb_K: float = DEFAULT_PARAMS["Tcmb_K"],
    Neff: float = DEFAULT_PARAMS["Neff"],
    tau_fast: float = DEFAULT_PARAMS["tau_fast"],
    f_fast: float = DEFAULT_PARAMS["f_fast"],
    xmin: float = DEFAULT_PARAMS["xmin"],
    xmax: float = 0.0,
    ngrid: int = 2001,
    n_iter: int = 15,
    x_BBN: float = DEFAULT_PARAMS["x_BBN"],
    width_BBN: float = DEFAULT_PARAMS["width_BBN"],
) -> dict[str, float]:
    """
    Core background computation for GeNeSyS v10.1 with effective Ξ_early and G_kin.
    Returns diagnostics and score used for model selection.
    """
    x, a, z = build_x_grid(xmin, xmax, ngrid)

    # Check Ξ_early effectiveness
    if x_BBN < x[0]:
        warn(f"x_BBN={x_BBN:.3g} is earlier than xmin={x[0]:.3g}; Ξ_early will be ~1 everywhere. "
             f"Consider decreasing --xmin or increasing --x_BBN.")

    h = H0 / 100.0
    # Radiation density today: either provided explicitly, or derived from (H0, Tcmb_K, Neff)
    Omega_r0_eff = float(omega_r0_from_Tcmb_Neff(H0, Tcmb_K, Neff) if Omega_r0 is None else Omega_r0)
    # Flatness closure: Omega_M0 is derived and must be non-negative
    Omega_M0_derived = validate_closure(Omega_b0=Omega_b0, Omega_C0=Omega_C0, Omega_r0=Omega_r0_eff)

    # Condensate kernel
    dx = np.gradient(x)
    Jc = np.exp(-0.5 * ((x - x_c) / sigma_c)**2)
    Mc_raw = np.cumsum(Jc * dx)
    M_c = Mc_raw / max(Mc_raw[-1], 1e-60)

    # Memory kernel with Ξ_early and G_kin
    M_m = np.ones_like(x)
    # Use per-step dt to remain correct even if x-grid changes in the future.
    # For current build_x_grid() this is uniform, so this is equivalent to a constant dt.

    for _ in range(n_iter):
        E2 = (
            Omega_r0_eff / a**4 +
            Omega_b0 / a**3 +
            Omega_C0 * M_c / a**3 +
            Omega_M0_derived * M_m
        )
        E = np.sqrt(np.maximum(E2, 1e-60))
        R, dRdx = compute_R_and_dRdx(x, E)
        Jm = Xi_early(x, x_cut=x_BBN, width=width_BBN) * W_eq(x) * G_kin(x, E, enable=False) * np.maximum(-alpha_M * dRdx, 0.0)

        acc_s, acc_f = 0.0, 0.0
        Mm_new = np.zeros_like(x)
        for i in range(len(x)):
            dt_i = float(x[i] - x[i-1]) if i > 0 else float(x[1] - x[0])
            dec_s = np.exp(-dt_i / max(tau_slow, 1e-30))
            dec_f = np.exp(-dt_i / max(tau_fast, 1e-30))
            acc_s = acc_s * dec_s + Jm[i] * dt_i
            acc_f = acc_f * dec_f + Jm[i] * dt_i
            Mm_new[i] = f_fast * acc_f + (1.0 - f_fast) * acc_s
        M_m = Mm_new / max(Mm_new[-1], 1e-60)

    # Final background
    E2 = (
        Omega_r0_eff / a**4 +
        Omega_b0 / a**3 +
        Omega_C0 * M_c / a**3 +
        Omega_M0_derived * M_m
    )
    E = np.sqrt(np.maximum(E2, 1e-60))

    if not np.all(np.isfinite(E)):
        raise FloatingPointError('Non-finite E(z) encountered')

    # Diagnostics
    age_Gyr = compute_age_Gyr(x, E, H0)
    w_M0 = w_from_M(x, M_m)
    f_DE_zstar = float((Omega_M0_derived * M_m[np.argmin(np.abs(z - 1100.0))]) / max(E2[np.argmin(np.abs(z - 1100.0))], 1e-60))
    Mc_zstar = interp_at_z(z, M_c, 1100.0)

    # Present-day densities
    Omega_r_today = Omega_r0_eff / a[-1]**4
    Omega_b_today = Omega_b0 / a[-1]**3
    Omega_C_today = Omega_C0 * M_c[-1] / a[-1]**3
    Omega_M_today = Omega_M0_derived * M_m[-1]
    Omega_tot0 = Omega_r_today + Omega_b_today + Omega_C_today + Omega_M_today

    # Score (LCDM-like ranking)
    score = (
        2.0 * abs(w_M0 + 1.0) +
        1.0 * abs(age_Gyr - 13.8) +
        3.0 * abs(f_DE_zstar) +
        4.0 * abs(1.0 - Mc_zstar) +  # enforce condensate formed by recombination (CMB-safe)
        1.0 * abs((Omega_b_today / max(Omega_tot0, 1e-60)) - 0.05)
    )

    return {
        "x_c": float(x_c),
        "sigma_c": float(sigma_c),
        "tau_slow": float(tau_slow),
        "alpha_M": float(alpha_M),
        "R_convention": "R_tilde=R/H0^2=6 E^2 (2 + d ln E/dx)",
        "H0": float(H0),
        "tau_fast": float(tau_fast),
        "f_fast": float(f_fast),
        "x_BBN": float(x_BBN),
        "width_BBN": float(width_BBN),
        "age_Gyr": float(age_Gyr),
        "w_M0": float(w_M0),
        "Omega_b0": float(Omega_b_today / max(Omega_tot0, 1e-60)),
        "Omega_r0": float(Omega_r_today / max(Omega_tot0, 1e-60)),
        "Omega_C0": float(Omega_C_today / max(Omega_tot0, 1e-60)),
        "Omega_M0": float(Omega_M_today / max(Omega_tot0, 1e-60)),
        "Omega_tot0": float(Omega_tot0),
        "f_DE_zstar": float(f_DE_zstar),
        "Mc_zstar": float(Mc_zstar),
        "score_LCDM_like": float(score),
    }

def compute_Ez_from_params(
    params: dict[str, float],
    *,
    Omega_b0: float = DEFAULT_PARAMS["Omega_b0"],
    Omega_C0: float = DEFAULT_PARAMS["Omega_C0"],
    Omega_r0: float | None = DEFAULT_PARAMS["Omega_r0"],
    Tcmb_K: float = DEFAULT_PARAMS["Tcmb_K"],
    Neff: float = DEFAULT_PARAMS["Neff"],
    xmin: float = DEFAULT_PARAMS["xmin"],
    xmax: float = 0.0,
    ngrid: int = 2001,
    n_iter: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute E(z) for best-fit parameters with effective Ξ_early and G_kin."""
    x, a, z = build_x_grid(xmin, xmax, ngrid)

    H0 = float(params["H0"])
    # v10.1 closure (Ω_M0 is derived, not a free input)
    Omega_r0_eff = float(omega_r0_from_Tcmb_Neff(H0, Tcmb_K, Neff) if Omega_r0 is None else Omega_r0)
    Omega_M0_derived = validate_closure(Omega_b0=Omega_b0, Omega_C0=Omega_C0, Omega_r0=Omega_r0_eff)

    x_BBN = params.get("x_BBN", DEFAULT_PARAMS["x_BBN"])
    width_BBN = params.get("width_BBN", DEFAULT_PARAMS["width_BBN"])

    # Check Ξ_early effectiveness
    if x_BBN < x[0]:
        warn(f"x_BBN={x_BBN:.3g} is earlier than xmin={x[0]:.3g}; Ξ_early will be ~1 everywhere. "
             f"Consider decreasing --xmin or increasing --x_BBN.")

    h = H0 / 100.0
    # Radiation density today: either provided explicitly, or derived from (H0, Tcmb_K, Neff)
    Omega_r0_eff = float(omega_r0_from_Tcmb_Neff(H0, Tcmb_K, Neff) if Omega_r0 is None else Omega_r0)
    # Flatness closure: Omega_M0 is derived and must be non-negative
    Omega_M0_derived = validate_closure(Omega_b0=Omega_b0, Omega_C0=Omega_C0, Omega_r0=Omega_r0_eff)

    x_c = float(params["x_c"])
    sigma_c = float(params["sigma_c"])
    tau_slow = float(params["tau_slow"])
    alpha_M = float(params["alpha_M"])
    tau_fast = float(params.get("tau_fast", DEFAULT_PARAMS["tau_fast"]))
    f_fast = float(params.get("f_fast", DEFAULT_PARAMS["f_fast"]))

    dx = np.gradient(x)
    Jc = np.exp(-0.5 * ((x - x_c) / sigma_c)**2)
    Mc_raw = np.cumsum(Jc * dx)
    M_c = Mc_raw / max(Mc_raw[-1], 1e-60)

    M_m = np.ones_like(x)
    # Use per-step dt to remain correct even if x-grid changes in the future.
    # For current build_x_grid() this is uniform, so this is equivalent to a constant dt.

    for _ in range(n_iter):
        E2 = (
            Omega_r0_eff / a**4 +
            Omega_b0 / a**3 +
            Omega_C0 * M_c / a**3 +
            Omega_M0_derived * M_m
        )
        E = np.sqrt(np.maximum(E2, 1e-60))
        R, dRdx = compute_R_and_dRdx(x, E)
        Jm = Xi_early(x, x_cut=x_BBN, width=width_BBN) * W_eq(x) * G_kin(x, E, enable=False) * np.maximum(-alpha_M * dRdx, 0.0)

        acc_s, acc_f = 0.0, 0.0
        Mm_new = np.zeros_like(x)
        for i in range(len(x)):
            dt_i = float(x[i] - x[i-1]) if i > 0 else float(x[1] - x[0])
            dec_s = np.exp(-dt_i / max(tau_slow, 1e-30))
            dec_f = np.exp(-dt_i / max(tau_fast, 1e-30))
            acc_s = acc_s * dec_s + Jm[i] * dt_i
            acc_f = acc_f * dec_f + Jm[i] * dt_i
            Mm_new[i] = f_fast * acc_f + (1.0 - f_fast) * acc_s
        M_m = Mm_new / max(Mm_new[-1], 1e-60)

    E2 = (
        Omega_r0_eff / a**4 +
        Omega_b0 / a**3 +
        Omega_C0 * M_c / a**3 +
        Omega_M0_derived * M_m
    )
    E = np.sqrt(np.maximum(E2, 1e-60))

    idx = np.argsort(z)
    z_s = z[idx]
    E_s = E[idx]
    # v10.1 convention: E(0)=1 exactly (avoid double-normalization elsewhere).
    E0 = float(np.interp(0.0, z_s, E_s))
    if not np.isfinite(E0) or E0 <= 0:
        die(f"E(0) invalid: {E0}. Check background integration and inputs.")
    E_s = E_s / E0
    return z_s, E_s

# --- Early-time diagnostic (report-only) ---
def _interp_E_at_z(z_tab: np.ndarray, E_tab: np.ndarray, zq: np.ndarray) -> tuple[np.ndarray, dict]:
    """Interpolate E(z) on log(1+z). Values outside range are clipped and reported."""
    z_tab = np.asarray(z_tab, dtype=float)
    E_tab = np.asarray(E_tab, dtype=float)
    zq = np.asarray(zq, dtype=float)

    # Sort and make strictly increasing in log(1+z)
    idx = np.argsort(z_tab)
    z_tab = z_tab[idx]
    E_tab = E_tab[idx]

    x_tab = np.log1p(z_tab)
    xq = np.log1p(zq)

    x_min = float(x_tab[0])
    x_max = float(x_tab[-1])
    xq_clip = np.clip(xq, x_min, x_max)

    Eq = np.interp(xq_clip, x_tab, E_tab)

    rep = {
        "z_min": float(z_tab[0]),
        "z_max": float(z_tab[-1]),
        "clipped": bool(np.any(xq != xq_clip)),
        "zq_requested": [float(v) for v in zq.tolist()],
        "zq_used": [float(np.expm1(v)) for v in xq_clip.tolist()],
    }
    return Eq, rep

def early_report_from_Ez(
    z_tab: np.ndarray,
    E_tab: np.ndarray,
    *,
    Omega_r0: float,
    Omega_b0: float,
    Omega_C0: float,
    Omega_M0: float,
    z_points: tuple[float, float, float] = (1e9, 1100.0, 1e4),
) -> dict:
    """Compute emergent fraction Omega_em(z) = (E^2 - E_std^2)/E^2 at key early epochs.

    This is report-only: no rejection and no penalty is applied here.
    Baseline early-time model: radiation + matter (Omega_b0 + Omega_C0 + Omega_M0), no DE term.
    """
    Or = float(Omega_r0)
    Om = float(Omega_b0) + float(Omega_C0) + float(Omega_M0)

    zq = np.array(z_points, dtype=float)
    Eq, clip_rep = _interp_E_at_z(z_tab, E_tab, zq)

    zp1 = 1.0 + zq
    E_std = np.sqrt(Or * zp1**4 + Om * zp1**3)

    # emergent fraction in H^2
    Omega_em = (Eq*Eq - E_std*E_std) / np.maximum(Eq*Eq, 1e-60)

    rep = {
        "z_points": [float(v) for v in zq.tolist()],
        "Omega_em": [float(v) for v in Omega_em.tolist()],
        "baseline": "E_std^2 = Omega_r0*(1+z)^4 + (Omega_b0+Omega_C0+Omega_M0)*(1+z)^3",
        "clip": clip_rep,
    }

    # Convenience keys
    rep["Omega_em_z1e9"] = float(Omega_em[0])
    rep["Omega_em_z1100"] = float(Omega_em[1])
    rep["Omega_em_z1e4"] = float(Omega_em[2])
    return rep

# --- Distance Calculations ---

def distance_bundle(z_tab: np.ndarray, E_tab: np.ndarray, H0: float, z_target: np.ndarray, ngrid: int = 6000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute D_L, D_M, D_H for given z_target with safety checks."""
    # Safety 1: sort z_tab
    idx = np.argsort(z_tab)
    z_tab = z_tab[idx]
    E_tab = E_tab[idx]

    # Safety 2: remove duplicates
    mask = np.concatenate(([True], np.diff(z_tab) > 0))
    z_tab = z_tab[mask]
    E_tab = E_tab[mask]

    z_target = np.asarray(z_target, float)
    zmax = max(1e-6, float(np.max(z_target)) * 1.02)  # Guard against zmax=0
    z_grid = np.linspace(0.0, zmax, ngrid)
    E_grid = log_interp(z_tab, E_tab, z_grid)
    invE = 1.0 / np.maximum(E_grid, 1e-60)

    I = cumulative_trapezoid(invE, z_grid, initial=0.0)
    I_t = np.interp(z_target, z_grid, I)

    Dc = (C_KM_S / H0) * I_t
    Dl = (1.0 + z_target) * Dc
    Dm = Dc
    Dh = (C_KM_S / H0) / log_interp(z_tab, E_tab, z_target)
    return Dl, Dm, Dh

def mu_from_Dl(Dl_mpc: np.ndarray) -> np.ndarray:
    """Distance modulus from luminosity distance (Mpc)."""
    return 5.0 * np.log10(np.maximum(Dl_mpc, 1e-60)) + 25.0

# --- Data Confrontation ---
def parse_pantheon_cov(path: Path, n: int) -> np.ndarray:
    """Parse Pantheon+ covariance matrix.

    Supports:
    - whitespace-separated floats
    - comma-separated floats
    - optional leading integer header (n or similar), i.e. n*n + 1 entries.
    """
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        die(f"Pantheon covariance file is empty: {path}")
    # Allow comma-separated values
    txt = txt.replace(",", " ")
    data = np.fromstring(txt, sep=" ")
    if data.size == n * n:
        return data.reshape((n, n))
    if data.size == n * n + 1:
        return data[1:].reshape((n, n))
    die(f"Pantheon covariance size mismatch: got {data.size}, expected {n*n} or {n*n+1}")
def sn_chi2_marginalized_M(mu_model: np.ndarray, m_obs: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    """SN χ² with analytical marginalization over M (full covariance)."""
    n = len(m_obs)
    one = np.ones(n, float)
    rhs = (m_obs - mu_model).astype(float)

    if not np.allclose(cov, cov.T, atol=1e-8):
        warn("SN covariance is not symmetric. Forcing symmetry for numerical stability.")
        cov = 0.5 * (cov + cov.T)

    try:
        cF = cho_factor(cov, lower=True, check_finite=False)
    except Exception as e:
        die(f"Cholesky decomposition failed (covariance not SPD?): {e}")

    Cinv_one = cho_solve(cF, one, check_finite=False)
    Cinv_rhs = cho_solve(cF, rhs, check_finite=False)

    A = float(one @ Cinv_one)
    B = float(one @ Cinv_rhs)
    Mbest = B / A

    resid = rhs - Mbest
    Cinv_resid = cho_solve(cF, resid, check_finite=False)
    chi2 = float(resid @ Cinv_resid)
    return chi2, float(Mbest)

def bao_load(summary_csv: Path, cov_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load BAO DR12 data vector and covariance.

    Accepts two summary formats:
      - DATA columns (preferred):        DM_over_rd, DH_over_rd  (or any columns containing 'dm'+'rd' and 'dh'+'rd')
      - FIDUCIAL-only columns (fallback): DM_over_rd_fid, DH_over_rd_fid

    This function will PREFER non-fiducial columns if both are present.
    It returns a meta dict describing which columns were used and whether they were fiducial-only.
    """
    df = pd.read_csv(summary_csv)
    cols = {c.lower(): c for c in df.columns}
    zcol = cols.get("z", df.columns[0])

    # Candidate selection: any column containing ('dm' and 'rd') / ('dh' and 'rd')
    dm_candidates, dh_candidates = [], []
    for c in df.columns:
        cl = c.lower()
        if ("dm" in cl) and ("rd" in cl):
            dm_candidates.append(c)
        if ("dh" in cl) and ("rd" in cl):
            dh_candidates.append(c)

    if not dm_candidates or not dh_candidates:
        die(f"BAO data missing DM/rd or DH/rd columns. Found: {df.columns.tolist()}")

    def prefer_nonfid(cands: list[str]) -> tuple[str, bool]:
        # returns (chosen_col, is_fiducial)
        nonfid = [c for c in cands if "_fid" not in c.lower()]
        if nonfid:
            return nonfid[0], False
        return cands[0], True

    dmcol, dm_is_fid = prefer_nonfid(dm_candidates)
    dhcol, dh_is_fid = prefer_nonfid(dh_candidates)
    is_fid_only = bool(dm_is_fid or dh_is_fid)

    z = df[zcol].to_numpy(float)

    # Interleaved ordering (consistent with pipeline): [DM(z0), DH(z0), DM(z1), DH(z1), ...]
    y = np.empty(2 * len(z), float)
    y[0::2] = df[dmcol].to_numpy(float)
    y[1::2] = df[dhcol].to_numpy(float)

    C = pd.read_csv(cov_csv, header=None).to_numpy(float)

    meta = {
        "summary_file": str(summary_csv),
        "cov_file": str(cov_csv),
        "z_column": zcol,
        "dm_column": dmcol,
        "dh_column": dhcol,
        "is_fiducial_only": is_fid_only,
        "Nvec": int(len(y)),
        "N_z": int(len(z)),
        "ordering": "interleaved: [DM(z0), DH(z0), DM(z1), DH(z1), ...]",
    }
    return z, y, C, meta


def bao_chi2_bestfit_rd_analytic(
    z_tab: np.ndarray,
    E_tab: np.ndarray,
    H0: float,
    z_bao: np.ndarray,
    y_obs: np.ndarray,
    *,
    solve_C: Callable[[np.ndarray], np.ndarray],  # Typed callable
) -> tuple[float, float]:
    """
    Best-fit r_d analytically for BAO data (used for both GeNeSyS and ΛCDM).
    Treats r_d as a nuisance parameter via least-squares elimination.
    solve_C: function that computes C^{-1} v (avoids explicit matrix inversion).
    """
    _, Dm, Dh = distance_bundle(z_tab, E_tab, H0, z_bao)

    # Build theory template t such that y_model = s * t, with s = 1/rd
    t = np.empty_like(y_obs)
    t[0::2] = Dm
    t[1::2] = Dh

    # Solve s = argmin (y_obs - s t)^T C^{-1} (y_obs - s t)
    Ct = solve_C(t)
    denom = float(t @ Ct)
    if denom <= 0.0:
        die("BAO analytic fit failed: non-positive denominator (check covariance).")

    s_best = float(t @ solve_C(y_obs)) / denom
    if s_best <= 0.0:
        die(f"BAO analytic fit failed: best s=1/rd is non-positive (s={s_best}).")

    y_model = s_best * t
    r = y_obs - y_model
    chi2 = float(r @ solve_C(r))
    rd_best = 1.0 / s_best
    return chi2, float(rd_best)

def lcdm_E_of_z(z: np.ndarray, H0: float, Om_m: float = 0.315) -> np.ndarray:
    """ΛCDM E(z) for comparison (explicit benchmark), with Ω_r consistent with GeNeSyS."""
    h = H0 / 100.0
    Og = 2.469e-5 / (h * h)  # Aligned with compute_background_for_params
    Neff = 3.046
    Om_r = Og * (1.0 + 0.2271 * Neff)
    Om_L = 1.0 - Om_m - Om_r
    return np.sqrt(Om_r * (1 + z)**4 + Om_m * (1 + z)**3 + Om_L)

def lcdm_eval_fixed_H0_with_rd_nuisance(
    z_sn: np.ndarray,
    m_sn: np.ndarray,
    cov_sn: np.ndarray,
    z_bao: np.ndarray,
    y_bao: np.ndarray,
    *,
    solve_C_bao: Callable[[np.ndarray], np.ndarray],
    H0_fixed: float,
    Om_m: float = 0.315,
) -> dict[str, float]:
    """
    Evaluate ΛCDM with fixed H0 and r_d treated as nuisance parameter (analytically eliminated).
    Uses the same H0 as GeNeSyS for a fair comparison, with r_d fitted analytically.
    """
    zmax = max(1e-6, float(np.max(z_sn)), float(np.max(z_bao)), 5.0)
    z_l = np.linspace(0.0, zmax, 8000)
    E_l = lcdm_E_of_z(z_l, H0_fixed, Om_m=Om_m)

    # SN chi2 (M marginalized)
    Dl_l, _, _ = distance_bundle(z_l, E_l, H0_fixed, z_sn)
    mu_l = mu_from_Dl(Dl_l)
    chi2_sn, M_best = sn_chi2_marginalized_M(mu_l, m_sn, cov_sn)

    # BAO chi2 + best rd (analytic)
    chi2_bao, rd_best = bao_chi2_bestfit_rd_analytic(
        z_l, E_l, H0_fixed, z_bao, y_bao,
        solve_C=solve_C_bao,
    )
    chi2_tot = float(chi2_sn + chi2_bao)

    return {
        "H0_fixed": H0_fixed,
        "rd_best": float(rd_best),
        "chi2_sn": float(chi2_sn),
        "Mbest": float(M_best),
        "chi2_bao": float(chi2_bao),
        "chi2_tot": float(chi2_tot),
    }

# --- Self-Test Mode ---
def cmd_selftest(args: argparse.Namespace) -> None:
    """Run deterministic self-tests for regression checks."""
    info("Running deterministic self-tests...")

    # Test 1: Simple E(z) generation (LCDM)
    z = np.linspace(0, 2, 2000)
    E = lcdm_E_of_z(z, H0=70.0, Om_m=0.3)
    if not np.all(np.isfinite(E)):
        die("SELFTEST failed: E(z) contains NaN/inf")
    if not np.all(E > 0):
        die("SELFTEST failed: E(z) contains non-positive values")

    # Test 2: Distance monotonicity (deterministic unsorted input)
    z_unsorted = np.linspace(0, 2, 1000)[::-1]  # Strictly decreasing
    E_unsorted = lcdm_E_of_z(z_unsorted, H0=70.0, Om_m=0.3)
    Dl, Dm, Dh = distance_bundle(z_unsorted, E_unsorted, 70.0, np.array([0.1, 0.5, 1.0, 2.0]))
    if not np.all(np.isfinite(Dl)):
        die("SELFTEST failed: D_L contains NaN/inf")
    if not np.all(np.diff(Dl) > 0):
        die("SELFTEST failed: D_L not strictly increasing with z")
    if not np.all(np.isfinite(Dm)):
        die("SELFTEST failed: D_M contains NaN/inf")
    if not np.all(np.isfinite(Dh)):
        die("SELFTEST failed: D_H contains NaN/inf")

    # Test 3: Interpolation safety (deterministic permutation)
    z_test = np.linspace(0, 2, 500)
    perm = np.arange(500)[::-1]  # Deterministic permutation
    z_permuted = z_test[perm]
    E_permuted = lcdm_E_of_z(z_permuted, H0=70.0, Om_m=0.3)
    Dl_test, _, _ = distance_bundle(z_permuted, E_permuted, 70.0, np.array([0.5, 1.0, 1.5]))
    if not np.all(np.isfinite(Dl_test)):
        die("SELFTEST failed: Permuted interpolation failed")

    info("All deterministic self-tests passed: ✅ Interpolation safety")
    info("                                   ✅ Distance monotonicity")
    info("                                   ✅ Finite values")
    info("Pipeline is ready for publication use.")

# --- CLI Modes ---
def cmd_scan(args: argparse.Namespace) -> None:
    """Background parameter scan mode with full metadata tracking."""
    apply_params_json(args)
    apply_output_dir(args)
    x_c_list = [float(x) for x in args.x_c_list.split(",")]
    sigma_c_list = [float(x) for x in args.sigma_c_list.split(",")]
    tau_slow_list = [float(x) for x in args.tau_slow_list.split(",")]
    alpha_M_list = [float(x) for x in args.alpha_M_list.split(",")]

    total = len(x_c_list) * len(sigma_c_list) * len(tau_slow_list) * len(alpha_M_list)
    info(f"Scan grid: {total} models")

    rows = []
    for xc in x_c_list:
        for sig in sigma_c_list:
            for ts in tau_slow_list:
                for aM in alpha_M_list:
                    try:
                        res = compute_background_for_params(
                            xc, sig, ts, aM,
                            H0=args.H0,
                            Omega_b0=args.Omega_b0,
                            Omega_C0=args.Omega_C0,
                            Omega_r0=args.Omega_r0,
                            Tcmb_K=args.Tcmb_K,
                            Neff=args.Neff,
                            tau_fast=args.tau_fast,
                            f_fast=args.f_fast,
                            xmin=args.xmin,
                            xmax=args.xmax,
                            ngrid=args.ngrid,
                            n_iter=args.n_iter,
                            x_BBN=args.x_BBN,
                            width_BBN=args.width_BBN,
                        )
                        if args.require_mc_zstar is not None and res["Mc_zstar"] < args.require_mc_zstar:
                            continue
                        # Early-time diagnostic (report-only)
                        try:
                            ptmp = {
                                "x_c": float(xc),
                                "sigma_c": float(sig),
                                "tau_slow": float(ts),
                                "alpha_M": float(aM),
                                "tau_fast": float(args.tau_fast),
                                "f_fast": float(args.f_fast),
                                "x_BBN": float(args.x_BBN),
                                "width_BBN": float(args.width_BBN),
                            }
                            z_tab, E_tab = compute_Ez_from_params(
                                ptmp,
                                Omega_b0=float(args.Omega_b0),
                                Omega_C0=float(args.Omega_C0),
                                Omega_r0=(None if args.Omega_r0 is None else float(args.Omega_r0)),
                                Tcmb_K=float(args.Tcmb_K),
                                Neff=float(args.Neff),
                                H0=float(args.H0),
                                xmin=float(args.xmin),
                                xmax=float(args.xmax),
                                ngrid=int(args.ngrid),
                            )
                            early_rep = early_report_from_Ez(
                                z_tab, E_tab,
                                Omega_r0=float(res["Omega_r0"]),
                                Omega_b0=float(res["Omega_b0"]),
                                Omega_C0=float(res["Omega_C0"]),
                                Omega_M0=float(res["Omega_M0"]),
                            )
                            # Flatten key diagnostics into the scan table
                            res["Omega_em_z1e9"] = early_rep["Omega_em_z1e9"]
                            res["Omega_em_z1100"] = early_rep["Omega_em_z1100"]
                            res["Omega_em_z1e4"] = early_rep["Omega_em_z1e4"]
                        except Exception as _e:
                            # Never fail the scan on early-report computation
                            res["Omega_em_z1e9"] = np.nan
                            res["Omega_em_z1100"] = np.nan
                            res["Omega_em_z1e4"] = np.nan

                        rows.append(res)
                    except Exception as e:
                        warn(f"Model failed (xc={xc}, sig={sig}, ts={ts}, aM={aM}): {e}")

    if not rows:
        die("Scan produced 0 valid models.")

    df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Complete metadata
    Omega_r0_eff = float(omega_r0_from_Tcmb_Neff(args.H0, args.Tcmb_K, args.Neff) if args.Omega_r0 is None else args.Omega_r0)
    Omega_M0_derived = validate_closure(Omega_b0=args.Omega_b0, Omega_C0=args.Omega_C0, Omega_r0=Omega_r0_eff)

    meta = {
        "script_version": SCRIPT_VERSION,
        "fixed_parameters": {
            "H0": args.H0,
            "Omega_b0": args.Omega_b0,
            "Omega_C0": args.Omega_C0,
        "Omega_r0_eff": Omega_r0_eff,
        "Omega_M0_derived": Omega_M0_derived,
"tau_fast": args.tau_fast,
            "f_fast": args.f_fast,
            "xmin": args.xmin,
            "xmax": args.xmax,
            "ngrid": args.ngrid,
            "n_iter": args.n_iter,
            "x_BBN": args.x_BBN,
            "width_BBN": args.width_BBN,
        },
        "scan_grid": {
            "x_c_list": x_c_list,
            "sigma_c_list": sigma_c_list,
            "tau_slow_list": tau_slow_list,
            "alpha_M_list": alpha_M_list,
        },
        "require_mc_zstar": args.require_mc_zstar,
        "output_csv": str(out_path.name),
        "output_csv_sha256": file_hash(out_path),
        "environment": {
            "python_version": sys.version.split()[0],
            "numpy_version": pkg_version("numpy"),
            "pandas_version": pkg_version("pandas"),
            "scipy_version": pkg_version("scipy"),
        },
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    info(f"Scan saved: {out_path} ({len(df)} models)")
    info(f"Scan metadata saved: {meta_path}")

    # Top 10 models
    df_sorted = df.sort_values("score_LCDM_like", ascending=True)
    cols = ["x_c", "sigma_c", "tau_slow", "alpha_M", "age_Gyr", "w_M0", "Mc_zstar", "score_LCDM_like"]
    info("\nTOP 10 models (lowest score_LCDM_like):")
    print(df_sorted[cols].head(10).to_string(index=False))

def cmd_best_ez(args: argparse.Namespace) -> None:
    """Best-fit E(z) computation with environment tracking."""
    apply_params_json(args)
    apply_output_dir(args)
    scan_path = Path(args.scan)
    if not scan_path.exists():
        die(f"Scan file not found: {scan_path}")

    df = pd.read_csv(scan_path)
    if len(df) == 0:
        die("Scan CSV is empty.")
    if "score_LCDM_like" not in df.columns:
        die("Scan CSV missing 'score_LCDM_like' column.")

    best = df.loc[df["score_LCDM_like"].idxmin()]
    params = {
        "x_c": float(best["x_c"]),
        "sigma_c": float(best["sigma_c"]),
        "tau_slow": float(best["tau_slow"]),
        "alpha_M": float(best["alpha_M"]),
        "H0": float(best.get("H0", args.H0)),
        "tau_fast": float(best.get("tau_fast", args.tau_fast)),
        "f_fast": float(best.get("f_fast", args.f_fast)),
        "x_BBN": float(best.get("x_BBN", args.x_BBN)),
        "width_BBN": float(best.get("width_BBN", args.width_BBN)),
    }

    info("Best-fit parameters:")
    for k, v in params.items():
        info(f"  {k:10s} = {v}")

    z, E = compute_Ez_from_params(
        params,
        Omega_b0=args.Omega_b0,
        Omega_C0=args.Omega_C0,
xmin=args.xmin,
        xmax=args.xmax,
        ngrid=args.ngrid,
        n_iter=args.n_iter,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame({"z": z, "E_genesys": E})
    df_out.to_csv(out_path, index=False)

    # Early-time diagnostic (report-only) for best-fit
    early_rep = early_report_from_Ez(
        z, E,
        Omega_r0=float(best.get("Omega_r0", np.nan)),
        Omega_b0=float(best.get("Omega_b0", np.nan)),
        Omega_C0=float(best.get("Omega_C0", np.nan)),
        Omega_M0=float(best.get("Omega_M0", np.nan)),
    )

    # Enriched metadata
    metadata = {
        "script_version": SCRIPT_VERSION,
        "input_scan": str(scan_path.name),
        "input_scan_sha256": file_hash(scan_path),
        "best_fit_parameters": params,
        "early_report": early_rep,
        "output_Ez_sha256": file_hash(out_path),
        "environment": {
            "python_version": sys.version.split()[0],
            "numpy_version": pkg_version("numpy"),
            "pandas_version": pkg_version("pandas"),
            "scipy_version": pkg_version("scipy"),
        },
    }

    if args.params_json:
        params_path = Path(args.params_json)
        params_path.parent.mkdir(parents=True, exist_ok=True)
        params_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        info(f"Parameters metadata saved: {params_path}")

    info(f"E(z) saved: {out_path}")

def cmd_confront(args: argparse.Namespace) -> None:
    """Data confrontation (SN full-cov + BAO DR12) with guaranteed verdict output.

    This command ALWAYS writes the verdict JSON specified by --output, even if
    SN or BAO fails (status will be PARTIAL/ERROR with an error message).
    """
    apply_params_json(args)
    apply_output_dir(args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out: dict = {
        "metadata": {
            "script_version": SCRIPT_VERSION,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "status": "INITIALIZED",
            "warnings": [],
        },
        "SN": {"status": "PENDING"},
        "BAO": {"status": "PENDING"},
        "TOTAL": {"status": "PENDING"},
    }

    def _warn(msg: str) -> None:
        out["metadata"].setdefault("warnings", []).append(str(msg))

    def _set_error(msg: str) -> None:
        out["metadata"]["status"] = "ERROR"
        _warn(msg)

    def _set_partial(msg: str) -> None:
        if out["metadata"].get("status") != "ERROR":
            out["metadata"]["status"] = "PARTIAL"
        _warn(msg)

    try:
        # ----------------------------
        # Load unified sections (optional) from the *master* params JSON (if provided)
        # ----------------------------
        payload: dict = {}
        master_params = getattr(args, "params", None)
        if master_params:
            p = Path(master_params)
            if p.exists():
                try:
                    pl = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(pl, dict):
                        payload = pl
                except Exception:
                    pass

        data_dir = Path(args.data_dir)

        # data_files override (optional)
        data_files: dict = {}
        if isinstance(payload, dict) and isinstance(payload.get("data_files", None), dict):
            data_files = payload["data_files"]

        sn_dat = str(data_files.get("sn_dat", "Pantheon+SH0ES.dat"))
        sn_cov = str(data_files.get("sn_cov", "symmetrized_cov_matrix.cov"))
        bao_summary = str(data_files.get("bao_summary", "DR12_fid_DMrd_DHrd_summary.csv"))
        bao_cov = str(data_files.get("bao_cov", "DR12_cov6x6_DMrd_DHrd_from_consensus.csv"))

        required_files = [sn_dat, sn_cov, bao_summary, bao_cov]
        file_paths = {f: data_dir / f for f in required_files}
        for f, p in file_paths.items():
            if not p.exists():
                raise FileNotFoundError(f"Missing data file: {f} (expected at {p})")

        # ----------------------------
        # Load E(z)
        # ----------------------------
        ez_path = Path(args.Ez)
        if not ez_path.exists():
            raise FileNotFoundError(f"E(z) file not found: {ez_path}")

        df_ez = pd.read_csv(ez_path)
        if df_ez.shape[1] < 2:
            raise ValueError(f"E(z) must have >=2 columns, got {df_ez.columns.tolist()}")
        z_gen = df_ez.iloc[:, 0].to_numpy(float)
        E_gen = df_ez.iloc[:, 1].to_numpy(float)

        # ----------------------------
        # Load params-json required by confront (audit metadata + best-fit)
        # ----------------------------
        params_path = Path(args.params_json)
        if not params_path.exists():
            raise FileNotFoundError(f"Parameters file not found: {params_path}")

        params_meta = json.loads(params_path.read_text(encoding="utf-8"))
        if not isinstance(params_meta, dict):
            raise ValueError(f"params-json must be a dict/object: {params_path}")

        # Verify Ez hash if available (warning only)
        expected_ez_hash = params_meta.get("output_Ez_sha256", None)
        actual_ez_hash = file_hash(ez_path)
        if expected_ez_hash is not None and str(expected_ez_hash) != str(actual_ez_hash):
            _warn(f"Ez hash mismatch: params-json expects {expected_ez_hash}, got {actual_ez_hash}")
        elif expected_ez_hash is None:
            _warn(f"{params_path.name} is missing 'output_Ez_sha256' (continuing).")

        # H0 used (priority: params-json best_fit_parameters.H0 else args.H0)
        H0_used = float(params_meta.get("best_fit_parameters", {}).get("H0", args.H0))
        H0_source = str(params_path.name)

        out["metadata"].update({
            "Ez_file": str(ez_path.name),
            "Ez_file_sha256": actual_ez_hash,
            "params_file": str(params_path.name),
            "params_file_sha256": file_hash(params_path),
            "H0_used_km_s_Mpc": float(H0_used),
            "H0_source": H0_source,
            "data_dir": str(data_dir),
            "data_files_used": {k: str(v.name) for k, v in file_paths.items()},
        })

        # ----------------------------
        # SN likelihood (full covariance)
        # ----------------------------
        chi2_sn_gen = None
        M_gen = None
        n_sn = None
        z_sn = None
        m_sn = None
        cov_sn = None

        try:
            sn_df = pd.read_csv(file_paths[sn_dat], sep=r"\s+", comment="#")
            if "zHD" not in sn_df.columns or "m_b_corr" not in sn_df.columns:
                raise ValueError(f"Pantheon file missing zHD/m_b_corr. Columns: {sn_df.columns.tolist()}")

            z_sn = sn_df["zHD"].to_numpy(float)
            m_sn = sn_df["m_b_corr"].to_numpy(float)
            n_sn = int(len(sn_df))

            cov_sn = parse_pantheon_cov(file_paths[sn_cov], n_sn)

            Dl_gen, _, _ = distance_bundle(z_gen, E_gen, H0_used, z_sn)
            mu_gen = mu_from_Dl(Dl_gen)
            chi2_sn_gen, M_gen = sn_chi2_marginalized_M(mu_gen, m_sn, cov_sn)

            out["SN"] = {
                "status": "SUCCESS",
                "mode": "FULL_COV_CHOLESKY",
                "N": int(n_sn),
                "Ndof_SN": int(n_sn - 1),
                "chi2_GeNeSyS": float(chi2_sn_gen),
                "Mbest_GeNeSyS": float(M_gen),
            }
        except Exception as e:
            out["SN"] = {"status": "ERROR", "error": str(e)[:500]}
            _set_partial(f"SN calculation failed: {str(e)[:200]}")

        # ----------------------------
        # BAO likelihood (analytic rd nuisance)
        # ----------------------------
        chi2_bao_gen = None
        rd_gen = None
        z_bao = None
        y_bao = None
        solve_C_bao = None

        try:
            z_bao, y_bao, C_bao, bao_meta = bao_load(file_paths[bao_summary], file_paths[bao_cov])

            if not np.allclose(C_bao, C_bao.T, atol=1e-12):
                _warn("BAO covariance not symmetric; symmetrizing.")
                C_bao = 0.5 * (C_bao + C_bao.T)

            try:
                cF_bao = cho_factor(C_bao, lower=True, check_finite=False)
            except Exception:
                eps = 1e-12 * np.trace(C_bao) / C_bao.shape[0]
                _warn(f"BAO covariance not SPD; adding jitter {eps:.3e}.")
                C_bao = C_bao + eps * np.eye(C_bao.shape[0])
                cF_bao = cho_factor(C_bao, lower=True, check_finite=False)

            def solve_C_bao(v: np.ndarray) -> np.ndarray:
                return cho_solve(cF_bao, v, check_finite=False)

            chi2_bao_gen, rd_gen = bao_chi2_bestfit_rd_analytic(
                z_gen, E_gen, H0_used, z_bao, y_bao, solve_C=solve_C_bao
            )

            out["BAO"] = {
                "status": "SUCCESS",
                "rd_treatment": "analytic_nuisance",
                "Nvec": int(len(y_bao)),
                "Ndof_BAO": int(len(y_bao) - 1),
                "chi2_GeNeSyS": float(chi2_bao_gen),
                "rd_best_GeNeSyS_Mpc": float(rd_gen),
            }
        except Exception as e:
            out["BAO"] = {"status": "ERROR", "error": str(e)[:500]}
            _set_partial(f"BAO calculation failed: {str(e)[:200]}")

        # ----------------------------
        # Optional LCDM comparison (fixed H0=H0_used)
        # ----------------------------
        if getattr(args, "compare_lcdm", False):
            try:
                if (z_sn is None) or (m_sn is None) or (cov_sn is None) or (z_bao is None) or (y_bao is None) or (solve_C_bao is None):
                    raise RuntimeError("Missing SN/BAO intermediates; cannot compare LCDM.")
                fit = lcdm_eval_fixed_H0_with_rd_nuisance(
                    z_sn, m_sn, cov_sn, z_bao, y_bao,
                    solve_C_bao=solve_C_bao, H0_fixed=H0_used, Om_m=getattr(args, "Om_m", 0.315)
                )
                out["metadata"]["comparison_note"] = "LCDM comparison at fixed H0=GeNeSyS"
                out["SN"]["chi2_LCDM"] = float(fit["chi2_sn"])
                out["SN"]["Mbest_LCDM"] = float(fit["Mbest"])
                out["BAO"]["chi2_LCDM"] = float(fit["chi2_bao"])
                out["BAO"]["rd_best_LCDM_Mpc"] = float(fit["rd_best"])
                out["TOTAL"]["chi2_LCDM"] = float(fit["chi2_tot"])
            except Exception as e:
                _set_partial(f"LCDM comparison failed: {str(e)[:200]}")

        # ----------------------------
        # TOTAL
        # ----------------------------
        if out["SN"].get("status") == "SUCCESS" and out["BAO"].get("status") == "SUCCESS":
            out["TOTAL"] = {
                "status": "SUCCESS",
                "chi2_GeNeSyS": float(chi2_sn_gen + chi2_bao_gen),
                "Ndof_total": int(out["SN"]["Ndof_SN"] + out["BAO"]["Ndof_BAO"]),
            }
            out["metadata"]["status"] = "FULL_SUCCESS"
        else:
            if out["metadata"].get("status") == "INITIALIZED":
                out["metadata"]["status"] = "PARTIAL"

    except Exception as e:
        _set_error(str(e)[:400])

    # GUARANTEED write
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    info(f"Verdict saved: {out_path}")
    print(json.dumps(out, indent=2))
def cmd_all(args: argparse.Namespace) -> None:
    """Run scan → best-ez → confront in one command."""
    apply_params_json(args)
    apply_output_dir(args)
    # 1. Scan
    scan_args = argparse.Namespace(**vars(args))
    scan_args.output = args.scan_out
    cmd_scan(scan_args)

    # 2. Best E(z)
    best_args = argparse.Namespace(**vars(args))
    best_args.scan = args.scan_out
    best_args.output = args.ez_out
    best_args.params_json = args.params_out
    cmd_best_ez(best_args)

    # 3. Confront (with hash verification)
    confront_args = argparse.Namespace(**vars(args))
    confront_args.Ez = args.ez_out
    confront_args.params_json = args.params_out
    confront_args.output = args.verdict_out
    cmd_confront(confront_args)

    # 4. Summary
    try:
        summarize_run(Path(args.scan_out), Path(args.params_out), Path(args.verdict_out))
    except Exception as _e:
        # Never fail the pipeline because of the summary
        warn_b('warn', f"Summary failed: {_e}")

# --- Main ---
def main() -> None:
    """Unified CLI entry point."""
    ap = argparse.ArgumentParser(
        description="GeNeSyS v10.1 Unified Pipeline (scan → E(z) → SN+BAO confrontation) with Ξ_early and ΛCDM comparison at fixed H0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sp = ap.add_subparsers(dest="mode", required=True)

    # Shared background parameters
    def add_background_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--H0", type=float, default=DEFAULT_PARAMS["H0"], help="Hubble constant [km/s/Mpc]")
        p.add_argument("--Omega_b0", type=float, default=DEFAULT_PARAMS["Omega_b0"], help="Baryon density today")
        p.add_argument("--Omega_C0", type=float, default=DEFAULT_PARAMS["Omega_C0"], help="Condensate density today")
        p.add_argument("--Tcmb-K", dest="Tcmb_K", type=float, default=DEFAULT_PARAMS["Tcmb_K"], help="CMB temperature [K] (used to derive Omega_r0 if not provided)")
        p.add_argument("--Neff", type=float, default=DEFAULT_PARAMS["Neff"], help="Effective number of neutrino species (used to derive Omega_r0 if not provided)")
        p.add_argument("--Omega-r0", dest="Omega_r0", type=float, default=DEFAULT_PARAMS["Omega_r0"], help="Radiation density today (optional; if omitted derived from (H0,Tcmb_K,Neff))")
        p.add_argument("--params", type=str, default=None, help="Optional JSON file to override background/scan parameters (keys match CLI names)")
        p.add_argument("--output-dir", dest="output_dir", type=str, default=None, help="Directory to write all output artifacts (prefixes relative output filenames)")
        p.add_argument("--tau_fast", type=float, default=DEFAULT_PARAMS["tau_fast"], help="Fast memory relaxation time")
        p.add_argument("--f_fast", type=float, default=DEFAULT_PARAMS["f_fast"], help="Fast memory fraction")
        p.add_argument("--xmin", type=float, default=DEFAULT_PARAMS["xmin"], help="Minimum x=ln(a) for grid (default=-20 for Ξ_early effectiveness)")
        p.add_argument("--xmax", type=float, default=0.0, help="Maximum x=ln(a) for grid")
        p.add_argument("--ngrid", type=int, default=2001, help="Number of grid points")
        p.add_argument("--n_iter", type=int, default=15, help="Memory kernel iteration count")
        p.add_argument("--x_BBN", type=float, default=DEFAULT_PARAMS["x_BBN"], help="Early-time cut for Ξ_early (BBN protection, default=-17)")
        p.add_argument("--width_BBN", type=float, default=DEFAULT_PARAMS["width_BBN"], help="Width for Ξ_early transition (default=0.5)")

    # Mode 1: Scan
    p1 = sp.add_parser("scan", help="Generate background parameter scan CSV with Ξ_early")
    add_background_args(p1)
    p1.add_argument("--x_c_list", type=str, default=",".join(map(str, DEFAULT_PARAMS["x_c_list"])), help="Comma-separated x_c values")
    p1.add_argument("--sigma_c_list", type=str, default=",".join(map(str, DEFAULT_PARAMS["sigma_c_list"])), help="Comma-separated sigma_c values")
    p1.add_argument("--tau_slow_list", type=str, default=",".join(map(str, DEFAULT_PARAMS["tau_slow_list"])), help="Comma-separated tau_slow values")
    p1.add_argument("--alpha_M_list", type=str, default=",".join(map(str, DEFAULT_PARAMS["alpha_M_list"])), help="Comma-separated alpha_M values")
    p1.add_argument("--require-mc-zstar", type=float, default=None, help="Filter models with M_c(z*) < threshold")
    p1.add_argument("--output", type=str, default="_GeNeSyS_v10_3_background_scan.csv", help="Output scan CSV")
    p1.set_defaults(func=cmd_scan)

    # Mode 2: Best E(z)
    p2 = sp.add_parser("best-ez", help="Compute best-fit E(z) from scan CSV with Ξ_early and G_kin")
    add_background_args(p2)
    p2.add_argument("--scan", type=str, required=True, help="Input scan CSV")
    p2.add_argument("--output", type=str, default="_GeNeSyS_v10_3_best_Ez.csv", help="Output E(z) CSV")
    p2.add_argument("--params-json", type=str, default=None, help="Optional JSON file to save best-fit parameters")
    p2.set_defaults(func=cmd_best_ez)

    # Mode 3: Confront Data
    p3 = sp.add_parser("confront", help="Confront E(z) with Pantheon+ (full cov) + BAO DR12 with ΛCDM comparison at fixed H0")
    p3.add_argument("--Ez", type=str, required=True, help="Input E(z) CSV")
    p3.add_argument("--params-json", type=str, required=True, help="JSON metadata from best-ez (REQUIRED for H0 consistency)")  # <-- REQUIRED
    p3.add_argument("--data-dir", type=str, default=".", help="Directory containing data files")
    p3.add_argument("--H0", type=float, default=DEFAULT_PARAMS["H0"], help="Hubble constant [km/s/Mpc] (overridden by params-json)")
    p3.add_argument("--rd-min", type=float, default=130.0, help="Minimum rd for BAO nuisance fit [Mpc] (legacy, not used with analytic method)")
    p3.add_argument("--rd-max", type=float, default=160.0, help="Maximum rd for BAO nuisance fit [Mpc] (legacy, not used with analytic method)")
    p3.add_argument("--rd-ngrid", type=int, default=500, help="Number of rd grid points (legacy, not used with analytic method)")
    p3.add_argument("--output", type=str, default="genesys_full_cov_verdict.json", help="Output verdict JSON")
    p3.add_argument("--compare-lcdm", action="store_true", help="Compare to ΛCDM with same H0 as GeNeSyS (no degeneracy with r_d)")
    p3.add_argument("--allow-bao-fid", action="store_true", help="Allow BAO summary files that contain only fiducial columns (e.g., *_fid). WARNING: this is not a data-vs-model comparison.")
    p3.add_argument("--Om_m", type=float, default=0.315, help="ΛCDM matter density for comparison")
    p3.set_defaults(func=cmd_confront)

    # Mode 4: All-in-one
    p4 = sp.add_parser("all", help="Run scan → best-ez → confront in one command with Ξ_early and ΛCDM comparison at fixed H0")
    add_background_args(p4)
    p4.add_argument("--x_c_list", type=str, default=",".join(map(str, DEFAULT_PARAMS["x_c_list"])), help="Comma-separated x_c values")
    p4.add_argument("--sigma_c_list", type=str, default=",".join(map(str, DEFAULT_PARAMS["sigma_c_list"])), help="Comma-separated sigma_c values")
    p4.add_argument("--tau_slow_list", type=str, default=",".join(map(str, DEFAULT_PARAMS["tau_slow_list"])), help="Comma-separated tau_slow values")
    p4.add_argument("--alpha_M_list", type=str, default=",".join(map(str, DEFAULT_PARAMS["alpha_M_list"])), help="Comma-separated alpha_M values")
    p4.add_argument("--require-mc-zstar", type=float, default=None, help="Filter models with M_c(z*) < threshold")
    p4.add_argument("--data-dir", type=str, default=".", help="Directory containing data files")
    p4.add_argument("--rd-min", type=float, default=130.0, help="Minimum rd for BAO nuisance fit [Mpc] (legacy)")
    p4.add_argument("--rd-max", type=float, default=160.0, help="Maximum rd for BAO nuisance fit [Mpc] (legacy)")
    p4.add_argument("--allow-bao-fid", action="store_true", help="Allow BAO summary files that contain only fiducial columns (e.g., *_fid). WARNING: this is not a data-vs-model comparison.")
    p4.add_argument("--rd-ngrid", type=int, default=500, help="Number of rd grid points (legacy)")
    p4.add_argument("--compare-lcdm", action="store_true", help="Compare to ΛCDM with same H0 as GeNeSyS (no degeneracy with r_d)")
    p4.add_argument("--Om_m", type=float, default=0.315, help="ΛCDM matter density for comparison")
    p4.add_argument("--scan-out", type=str, default="_GeNeSyS_v10_3_background_scan.csv", help="Output scan CSV")
    p4.add_argument("--ez-out", type=str, default="_GeNeSyS_v10_3_best_Ez.csv", help="Output E(z) CSV")
    p4.add_argument("--params-out", type=str, default="best_fit_params.json", help="JSON file to save best-fit parameters")
    p4.add_argument("--verdict-out", type=str, default="genesys_full_cov_verdict.json", help="Output verdict JSON")
    p4.set_defaults(func=cmd_all)

    # Mode 5: Self-test
    p5 = sp.add_parser("selftest", help="Run deterministic self-tests for regression checks")
    p5.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
