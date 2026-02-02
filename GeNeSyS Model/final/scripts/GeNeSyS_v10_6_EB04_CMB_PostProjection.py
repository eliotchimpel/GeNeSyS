#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeNeSyS_v10_6_EB04_CMB_PostProjection.py
========================================
Author: Eliot CHIMPEL

EB04 (CMB post-projection) in strict continuity with EB03.

Scope (strict GeNeSyS framing):
- This script does NOT generate the primordial CMB spectrum.
- It evaluates whether the late-time effective geometry (and, if available, late-time potentials)
  produced by EB03 are compatible with CMB photon propagation (distances / angular scales) and
  late-time integrated effects (ISW/lensing proxies).

Important honesty note:
- If EB03 did not output a usable (Phi, Psi) or (Phi+Psi) time series, ISW diagnostics are NOT computed.
  The summary will mark these as NOT_AVAILABLE, without any inference.

Optional comparative mode:
- A --compare-planck switch adds a clearly labeled comparison using Planck 2018/2020 angular acoustic scale:
  100*theta_* = 1.04109 ± 0.00030. (Planck 2018 results VI)   [user-visible citations must be in the paper, not here]
"""

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Utilities: hashing / IO
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def npz_to_dict(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def ensure_strictly_increasing(x: np.ndarray, name: str) -> None:
    dx = np.diff(x)
    if not np.all(dx > 0):
        raise ValueError(f"{name} is not strictly increasing; cannot safely integrate/project.")


# =============================================================================
# Core: background geometry projections
# =============================================================================

def infer_z_from_a(a: np.ndarray) -> np.ndarray:
    return 1.0 / a - 1.0


def choose_time_axis(geom: dict) -> tuple[np.ndarray, str]:
    if "eta" in geom:
        return np.asarray(geom["eta"], dtype=float), "eta"
    if "x" in geom:
        return np.asarray(geom["x"], dtype=float), "x"
    raise KeyError("geom npz must contain 'eta' or 'x'.")


def compute_comoving_distance_from_eta(eta: np.ndarray, a: np.ndarray, z_target: float) -> dict:
    z = infer_z_from_a(a)
    idx = int(np.argmin(np.abs(z - z_target)))
    eta0 = float(eta[-1])
    eta_z = float(eta[idx])
    chi = float(eta0 - eta_z)
    D_A = chi / (1.0 + float(z_target))
    return {
        "z_target": float(z_target),
        "idx": idx,
        "eta0": eta0,
        "eta_z": eta_z,
        "chi": chi,
        "D_A": D_A,
    }


def trapz_integral(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


# =============================================================================
# Optional unit conversion (ONLY if user specifies eta units)
# =============================================================================

def convert_conformal_length_to_Mpc(value_conformal: float, eta_unit: str, H0_kmsMpc: float | None) -> dict:
    """
    Convert a conformal-length-like quantity (chi or D_A) to Mpc, ONLY when units are declared.

    eta_unit:
      - "Mpc": means eta is already in Mpc (c=1 convention already absorbed)
      - "H0inv": means eta is in units of H0^{-1} (dimensionless time scaled by 1/H0)
      - "unknown": no conversion performed

    If eta_unit == "H0inv", H0_kmsMpc is required and we use:
      (c/H0) in Mpc, with c in km/s and H0 in km/s/Mpc.

    Returns dict with status and value_Mpc if available.
    """
    eta_unit = (eta_unit or "unknown").lower().strip()
    if eta_unit == "mpc":
        return {"status": "OK", "value_Mpc": float(value_conformal), "assumption": "eta already in Mpc"}
    if eta_unit == "h0inv":
        if H0_kmsMpc is None or H0_kmsMpc <= 0:
            return {"status": "NOT_AVAILABLE", "reason": "eta_unit=H0inv requires positive --H0-kmsMpc"}
        c_km_s = 299792.458
        scale_Mpc = c_km_s / float(H0_kmsMpc)  # Mpc
        return {"status": "OK", "value_Mpc": float(value_conformal) * scale_Mpc, "assumption": "eta in H0^{-1} units"}
    return {"status": "NOT_AVAILABLE", "reason": "eta_unit unknown; no conversion performed"}


# =============================================================================
# Optional: ISW / lensing proxies (only if potentials are present)
# =============================================================================

def detect_potential_series(eb3: dict, geom: dict) -> tuple[np.ndarray | None, str]:
    # Direct sum series
    for k in ["Phi_plus_Psi", "phi_plus_psi", "PhiPlusPsi", "phi_psi_sum", "PhiPsi"]:
        if k in eb3:
            s = np.asarray(eb3[k], dtype=float).squeeze()
            if s.ndim == 1:
                return s, k
    # Sum Phi and Psi if both exist
    phi_key = None
    psi_key = None
    for k in ["Phi", "phi"]:
        if k in eb3:
            phi_key = k
            break
    for k in ["Psi", "psi"]:
        if k in eb3:
            psi_key = k
            break
    if phi_key and psi_key:
        phi = np.asarray(eb3[phi_key], dtype=float).squeeze()
        psi = np.asarray(eb3[psi_key], dtype=float).squeeze()
        if phi.ndim == 1 and psi.ndim == 1 and phi.shape == psi.shape:
            return phi + psi, f"{phi_key}+{psi_key}"
    # fallback check geom (rare)
    for k in ["Phi_plus_Psi", "phi_plus_psi"]:
        if k in geom:
            s = np.asarray(geom[k], dtype=float).squeeze()
            if s.ndim == 1:
                return s, f"geom:{k}"
    return None, "NOT_AVAILABLE"


def compute_isw_proxy(z: np.ndarray, pot: np.ndarray, late_z_max: float) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    ISW proxy: d/dz (Phi+Psi) and norms over z in [0, late_z_max] (if sufficient coverage).
    This is a diagnostic, not a C_ell computation.
    """
    z = np.asarray(z, dtype=float)
    pot = np.asarray(pot, dtype=float)
    if z.shape[0] != pot.shape[0]:
        return ({"status": "NOT_AVAILABLE", "reason": "z and potential length mismatch"}, np.array([]), np.array([]))

    # Work with increasing z for stable derivative (many arrays are time-ordered, z decreasing)
    z_inc = z[::-1].copy()
    p_inc = pot[::-1].copy()
    ensure_strictly_increasing(z_inc, "z (increasing for ISW proxy)")

    dp_dz = np.gradient(p_inc, z_inc)

    m = (z_inc >= 0.0) & (z_inc <= float(late_z_max))
    if np.sum(m) < 5:
        # not enough samples in the requested window -> fall back to full range
        m = slice(None)

    l2 = float(np.sqrt(trapz_integral(z_inc[m], dp_dz[m] ** 2)))
    maxabs = float(np.max(np.abs(dp_dz[m])))

    payload = {
        "status": "OK",
        "late_z_max_used": float(late_z_max) if not isinstance(m, slice) else "FULL_RANGE",
        "l2_norm_dp_dz": l2,
        "maxabs_dp_dz": maxabs,
    }
    return payload, z_inc, dp_dz


def compute_lensing_kernel_proxy(z_sorted: np.ndarray, chi_sorted: np.ndarray, z_star: float) -> tuple[dict, np.ndarray | None]:
    """
    Geometric lensing kernel proxy:
      W(z) ∝ chi(z) * (chi_* - chi(z)) / chi_*
    Pure geometry sanity check only (no growth, no prefactors).
    """
    z_sorted = np.asarray(z_sorted, dtype=float)
    chi_sorted = np.asarray(chi_sorted, dtype=float)
    idx = int(np.argmin(np.abs(z_sorted - float(z_star))))
    chi_star = float(chi_sorted[idx])
    if chi_star <= 0:
        return {"status": "NOT_AVAILABLE", "reason": "chi_star_nonpositive"}, None

    W = chi_sorted * (chi_star - chi_sorted) / chi_star
    m = (z_sorted >= 0.0) & (z_sorted <= float(z_star))
    if np.sum(m) < 5:
        m = slice(None)

    payload = {
        "status": "OK",
        "chi_star": chi_star,
        "W_max": float(np.max(W[m])),
        "W_argmax_z": float(z_sorted[m][int(np.argmax(W[m]))]) if not isinstance(m, slice) else float(z_sorted[int(np.argmax(W))]),
        "note": "geometry-only proxy; no growth included",
    }
    return payload, W


# =============================================================================
# Plotting
# =============================================================================

def save_plot(path: Path, fig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# Optional comparative Planck constants (ONLY for labeled comparison)
# =============================================================================
PLANCK_2018_100_THETA_STAR = 1.04109
PLANCK_2018_100_THETA_STAR_SIGMA = 0.00030


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="GeNeSyS EB04 CMB post-projection (strictly from EB03 outputs)")
    ap.add_argument("--geom-npz", required=True, help="Path to geom_output_*.npz")
    ap.add_argument("--aux-npz", default=None, help="Path to aux_output_*.npz (optional)")
    ap.add_argument("--eb3-npz", required=True, help="Path to eb3_output_*.npz")
    ap.add_argument("--output-dir", default="output_EB04", help="Directory for EB04 outputs")
    ap.add_argument("--z-star", type=float, default=1090.0, help="Redshift of last scattering surface (default 1090)")
    ap.add_argument("--rs-fid", type=float, default=None,
                    help="Optional fiducial sound horizon r_s in SAME UNITS as eta/chi (if provided, theta_* is computed)")
    ap.add_argument("--late-z-max", type=float, default=6.0, help="Upper bound of late-time window for ISW proxy (default 6)")
    ap.add_argument("--tag", default="v10_6", help="Tag string for outputs")

    # New options (requested improvements)
    ap.add_argument("--eta-unit", default="unknown", choices=["unknown", "Mpc", "H0inv"],
                    help="Declare unit of eta/chi: 'Mpc' or 'H0inv'. If unknown, no physical conversion is attempted.")
    ap.add_argument("--H0-kmsMpc", type=float, default=None,
                    help="H0 in km/s/Mpc (required if --eta-unit H0inv and you want Mpc conversion).")
    ap.add_argument("--compare-planck", action="store_true",
                    help="Add a clearly labeled comparison to Planck 2018/2020 100*theta_* (requires theta_* computed).")

    # Automated verdict thresholds (do NOT assume Planck unless user asks)
    ap.add_argument("--theta-pass-sigma", type=float, default=5.0,
                    help="If --compare-planck, PASS when |100*theta - Planck| <= N*sigma (default 5).")
    args = ap.parse_args()

    geom_path = Path(args.geom_npz)
    eb3_path = Path(args.eb3_npz)
    aux_path = Path(args.aux_npz) if args.aux_npz else None
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    geom = npz_to_dict(geom_path)
    eb3 = npz_to_dict(eb3_path)
    aux = npz_to_dict(aux_path) if aux_path else {}

    a = np.asarray(geom.get("a", None), dtype=float) if "a" in geom else None
    if a is None:
        raise KeyError("geom npz must contain 'a' for EB04 distance projections.")

    eta, time_kind = choose_time_axis(geom)
    if time_kind != "eta":
        raise ValueError("EB04 distance projection requires 'eta' (conformal time) in geom npz. Found only 'x'.")

    if eta.shape[0] != a.shape[0]:
        raise ValueError(f"Shape mismatch: len(eta)={eta.shape[0]} vs len(a)={a.shape[0]}")

    z_geom = infer_z_from_a(a)
    z_eb3 = np.asarray(eb3["z"], dtype=float) if "z" in eb3 else None

    # Distances to z_star
    dist_star = compute_comoving_distance_from_eta(eta=eta, a=a, z_target=args.z_star)

    # Full chi(z), D_A(z) arrays for plots
    eta0 = float(eta[-1])
    chi_z = (eta0 - eta).astype(float)
    order = np.argsort(z_geom)
    z_sorted = z_geom[order]
    chi_sorted = chi_z[order]
    DA_sorted = chi_sorted / (1.0 + z_sorted)

    # Optional physical conversion if units declared
    chi_star_Mpc = convert_conformal_length_to_Mpc(dist_star["chi"], args.eta_unit, args.H0_kmsMpc)
    DA_star_Mpc = convert_conformal_length_to_Mpc(dist_star["D_A"], args.eta_unit, args.H0_kmsMpc)

    # Theta* if rs_fid provided (in SAME units as eta)
    theta_star = {"status": "NOT_COMPUTED", "reason": "rs_fid not provided"}
    if args.rs_fid is not None:
        if dist_star["D_A"] <= 0:
            theta_star = {"status": "NOT_AVAILABLE", "reason": "nonpositive_D_A"}
        else:
            theta = float(args.rs_fid / dist_star["D_A"])
            theta_star = {
                "status": "OK",
                "rs_fid_same_units_as_eta": float(args.rs_fid),
                "theta_star_rad": theta,
                "theta_star_deg": float(np.degrees(theta)),
                "theta_star_times_100": float(100.0 * theta),
            }

    # Optional Planck comparison (clearly labeled)
    planck_cmp = {"status": "NOT_REQUESTED"}
    automated_verdict = {}

    if args.compare_planck:
        if theta_star.get("status") != "OK":
            planck_cmp = {"status": "NOT_AVAILABLE", "reason": "theta_* not computed; provide --rs-fid in eta units"}
        else:
            v = theta_star["theta_star_times_100"]
            delta = float(v - PLANCK_2018_100_THETA_STAR)
            nsig = abs(delta) / PLANCK_2018_100_THETA_STAR_SIGMA
            planck_cmp = {
                "status": "OK",
                "planck_2018_100_theta_star": PLANCK_2018_100_THETA_STAR,
                "planck_2018_sigma": PLANCK_2018_100_THETA_STAR_SIGMA,
                "genesys_100_theta_star": v,
                "delta": delta,
                "n_sigma": nsig,
                "note": "Comparative diagnostic only; does not imply primordial modeling.",
            }
            automated_verdict["theta_star_planck"] = "PASS" if nsig <= float(args.theta_pass_sigma) else "FAIL"

    # Potential series and ISW proxy
    pot_series, pot_label = detect_potential_series(eb3=eb3, geom=geom)
    isw = {"status": "NOT_AVAILABLE", "reason": "No usable potential time series in EB03 outputs."}
    dpdz_payload = None

    if pot_series is not None:
        if z_eb3 is not None and z_eb3.shape[0] == pot_series.shape[0]:
            z_for_pot = z_eb3
        elif z_geom.shape[0] == pot_series.shape[0]:
            z_for_pot = z_geom
        else:
            z_for_pot = None

        if z_for_pot is None:
            isw = {"status": "NOT_AVAILABLE", "reason": f"Potential '{pot_label}' length does not match available z grids."}
        else:
            isw_payload, z_inc, dp_dz = compute_isw_proxy(z=z_for_pot, pot=pot_series, late_z_max=args.late_z_max)
            isw = dict(isw_payload)
            isw["potential_label"] = pot_label
            dpdz_payload = (z_inc, dp_dz)
            automated_verdict["isw_proxy"] = "OK"

    if isw.get("status") != "OK":
        automated_verdict["isw_proxy"] = "NOT_AVAILABLE"

    # Lensing kernel proxy (geometry only)
    lensing_payload, W_proxy = compute_lensing_kernel_proxy(z_sorted=z_sorted, chi_sorted=chi_sorted, z_star=float(args.z_star))
    lensing = dict(lensing_payload)
    automated_verdict["lensing_kernel_proxy"] = lensing.get("status", "NOT_AVAILABLE")

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(10, 5))
    ax1 = fig1.add_subplot(111)
    ax1.plot(z_sorted, DA_sorted)
    ax1.set_xlabel("z")
    ax1.set_ylabel(r"$D_A(z)$ (units of conformal length)")
    ax1.set_title("EB04: Angular diameter distance from EB03 geometry")
    ax1.grid(True, alpha=0.3)
    p_DA = outdir / f"eb04_DA_vs_z_{args.tag}.png"
    save_plot(p_DA, fig1)

    if theta_star.get("status") == "OK":
        fig2 = plt.figure(figsize=(10, 4))
        ax2 = fig2.add_subplot(111)
        ax2.axis("off")
        txt = (
            f"EB04 theta* (using external r_s in eta-units)\n\n"
            f"z* = {args.z_star}\n"
            f"D_A(z*) = {dist_star['D_A']:.6e} (eta units)\n"
            f"r_s(fid) = {theta_star['rs_fid_same_units_as_eta']:.6e} (eta units)\n"
            f"theta* = {theta_star['theta_star_rad']:.6e} rad\n"
            f"100*theta* = {theta_star['theta_star_times_100']:.6f}\n"
        )
        if planck_cmp.get("status") == "OK":
            txt += (
                f"\nPlanck comparative (100*theta*): {PLANCK_2018_100_THETA_STAR:.5f} ± {PLANCK_2018_100_THETA_STAR_SIGMA:.5f}\n"
                f"Δ = {planck_cmp['delta']:+.6f}  (|Δ|/σ = {planck_cmp['n_sigma']:.2f})\n"
            )
        ax2.text(0.02, 0.98, txt, va="top")
        p_theta = outdir / f"eb04_theta_star_{args.tag}.png"
        save_plot(p_theta, fig2)
    else:
        p_theta = None

    if dpdz_payload is not None:
        z_inc, dp_dz = dpdz_payload
        fig3 = plt.figure(figsize=(10, 5))
        ax3 = fig3.add_subplot(111)
        ax3.plot(z_inc, dp_dz)
        ax3.set_xlabel("z (increasing)")
        ax3.set_ylabel(r"$d(\Phi+\Psi)/dz$ (proxy units)")
        ax3.set_title(f"EB04: ISW proxy from potential '{pot_label}', window z∈[0,{args.late_z_max}]")
        ax3.grid(True, alpha=0.3)
        p_isw = outdir / f"eb04_ISW_proxy_dPhiPsi_dz_{args.tag}.png"
        save_plot(p_isw, fig3)
    else:
        p_isw = None

    if W_proxy is not None and lensing.get("status") == "OK":
        fig4 = plt.figure(figsize=(10, 5))
        ax4 = fig4.add_subplot(111)
        ax4.plot(z_sorted, W_proxy)
        ax4.set_xlabel("z")
        ax4.set_ylabel("W(z) (geometric proxy)")
        ax4.set_title("EB04: Lensing kernel proxy (geometry only)")
        ax4.grid(True, alpha=0.3)
        p_lens = outdir / f"eb04_lensing_kernel_proxy_{args.tag}.png"
        save_plot(p_lens, fig4)
    else:
        p_lens = None

    # -------------------------------------------------------------------------
    # Summary JSON
    # -------------------------------------------------------------------------
    summary = {
        "eb_level": "EB04",
        "scope": "CMB post-projection (geometry + optional late-time potentials) strictly from EB03 outputs; no primordial generation.",
        "inputs": {
            "geom_npz": str(geom_path),
            "aux_npz": str(aux_path) if aux_path else None,
            "eb3_npz": str(eb3_path),
        },
        "unit_declarations": {
            "eta_unit": args.eta_unit,
            "H0_kmsMpc": args.H0_kmsMpc,
        },
        "z_star": float(args.z_star),
        "distance": {
            "eta0": dist_star["eta0"],
            "eta_at_zstar": dist_star["eta_z"],
            "chi_zstar_eta_units": dist_star["chi"],
            "D_A_zstar_eta_units": dist_star["D_A"],
            "chi_zstar_Mpc": chi_star_Mpc,
            "D_A_zstar_Mpc": DA_star_Mpc,
            "units_note": "Physical conversion is performed ONLY if eta_unit is declared as Mpc or H0inv.",
        },
        "theta_star": theta_star,
        "planck_comparison": planck_cmp,
        "isw_proxy": isw,
        "lensing_kernel_proxy": lensing,
        "automated_verdict": automated_verdict,
        "data_availability": {
            "geom_keys": sorted(list(geom.keys())),
            "eb3_keys": sorted(list(eb3.keys())),
            "aux_keys": sorted(list(aux.keys())) if aux else [],
        },
        "verdict_logic_note": "EB04 is a compatibility check; absence of strong late-time signatures is not a failure.",
    }

    summary_path = outdir / f"eb04_summary_{args.tag}.json"
    save_json(summary_path, summary)

    # -------------------------------------------------------------------------
    # Manifest with SHA256
    # -------------------------------------------------------------------------
    outputs = [summary_path, p_DA]
    if p_theta is not None:
        outputs.append(p_theta)
    if p_isw is not None:
        outputs.append(p_isw)
    if p_lens is not None:
        outputs.append(p_lens)

    manifest = {
        "eb_level": "EB04",
        "tag": args.tag,
        "sha256": {
            "inputs": {
                str(geom_path): sha256_file(geom_path),
                str(eb3_path): sha256_file(eb3_path),
            },
            "outputs": {},
        },
    }
    if aux_path:
        manifest["sha256"]["inputs"][str(aux_path)] = sha256_file(aux_path)

    for p in outputs:
        if p and Path(p).exists():
            manifest["sha256"]["outputs"][str(p)] = sha256_file(Path(p))

    manifest_path = outdir / f"manifest_eb4_{args.tag}.json"
    save_json(manifest_path, manifest)

    print("=" * 80)
    print("EB04 COMPLETE")
    print("=" * 80)
    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {manifest_path}")
    for p in outputs:
        if p and Path(p).exists():
            print(f"[SAVED] {p}")
    if isw.get("status") != "OK":
        print(f"[NOTE] ISW proxy not computed: {isw.get('reason')}")
    if theta_star.get("status") != "OK":
        print(f"[NOTE] theta* not computed/available: {theta_star}")
    if args.compare_planck:
        print(f"[NOTE] Planck comparative mode enabled (uses 100*theta* reference).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())