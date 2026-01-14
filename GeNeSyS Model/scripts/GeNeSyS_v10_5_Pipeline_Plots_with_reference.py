#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

import scipy.linalg as la

C_KM_S = 299792.458


def die(msg: str):
    raise SystemExit(f"[FATAL] {msg}")

def log_line(log_file: Path, msg: str):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")
    print(msg, flush=True)

def first_existing(paths):
    for p in paths:
        if p is not None and p.exists():
            return p
    return None

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def cumulative_trapz(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    out = np.zeros_like(x, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return out

def build_Dc_grid(z_grid: np.ndarray, E_grid: np.ndarray, H0: float) -> np.ndarray:
    invE = 1.0 / np.maximum(E_grid, 1e-60)
    I = cumulative_trapz(z_grid, invE)
    return (C_KM_S / H0) * I  # Mpc

def mu_from_Dc(z: np.ndarray, Dc: np.ndarray) -> np.ndarray:
    Dl = (1.0 + z) * Dc  # Mpc
    return 5.0 * np.log10(np.maximum(Dl, 1e-30)) + 25.0

def load_pantheon_cov_STAT_SYS(path: Path) -> np.ndarray:
    # Format Pantheon+ classique: 1er nombre = N puis N*N valeurs
    raw = np.loadtxt(path)
    if raw.size < 2:
        raise RuntimeError(f"SN cov too small: {path}")
    N = int(raw[0])
    flat = raw[1:]
    if flat.size != N * N:
        raise RuntimeError(f"SN cov size mismatch: got {flat.size}, expected {N*N} (N={N}) in {path}")
    cov = flat.reshape((N, N))
    return 0.5 * (cov + cov.T)

def load_pantheon_cov_symmetrized(path: Path, N_expected: int) -> np.ndarray:
    """
    symmetrized_cov_matrix.cov :
      - parfois whitespace-separated
      - parfois comma-separated (CSV)
    On autodétecte via la 1ère ligne non vide/comment.
    """
    # Détecter le séparateur sur la 1ère ligne utile
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        first = ""
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            first = s
            break

    if not first:
        raise RuntimeError(f"SN sym cov empty/unreadable: {path}")

    delim = "," if ("," in first) else None  # None => whitespace
    try:
        cov = np.loadtxt(path, delimiter=delim)
    except Exception:
        # fallback plus tolérant
        cov = np.genfromtxt(path, delimiter=delim)

    if cov.ndim != 2:
        raise RuntimeError(f"SN sym cov not 2D: shape={getattr(cov,'shape',None)} in {path}")

    if cov.shape != (N_expected, N_expected):
        raise RuntimeError(f"SN sym cov shape {cov.shape}, expected ({N_expected},{N_expected}) in {path}")

    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    return 0.5 * (cov + cov.T)

def fit_offset_fullcov(residual: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    """Fit additive offset a minimizing (residual-a)^T C^-1 (residual-a). Returns (a, chi2)."""
    N = residual.size
    ones = np.ones(N, dtype=float)
    cov = 0.5 * (cov + cov.T)
    cho = la.cho_factor(cov, lower=True, check_finite=False)

    def Cinv(v): return la.cho_solve(cho, v, check_finite=False)

    den = float(ones @ Cinv(ones))
    num = float(ones @ Cinv(residual))
    a = num / den
    r = residual - a * ones
    chi2 = float(r @ Cinv(r))
    return a, chi2

def whiten_residuals(d_minus_m: np.ndarray, cov: np.ndarray) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    L = la.cholesky(cov, lower=True, check_finite=False)
    return la.solve_triangular(L, d_minus_m, lower=True, check_finite=False)

def find_pipeline_outputs(out_dir: Path):
    ez = first_existing(sorted(out_dir.glob("*best_Ez.csv")))
    params = first_existing(sorted(out_dir.glob("*best_fit_params.json"))) or first_existing(sorted(out_dir.glob("*best_fit_params*.json")))
    scan_meta = first_existing(sorted(out_dir.glob("*background_scan.meta.json")))
    verdict = first_existing(sorted(out_dir.glob("*full_cov_verdict*.json"))) or first_existing(sorted(out_dir.glob("*verdict*.json")))
    return ez, params, scan_meta, verdict

def load_bao_dr12(summary_path: Path):
    bao = pd.read_csv(summary_path)
    if {"z","DM_over_rd","DH_over_rd"}.issubset(bao.columns):
        z = bao["z"].to_numpy(float)
        DM = bao["DM_over_rd"].to_numpy(float)
        DH = bao["DH_over_rd"].to_numpy(float)
        return z, DM, DH, "observed"
    if {"z","DM_over_rd_fid","DH_over_rd_fid"}.issubset(bao.columns):
        z = bao["z"].to_numpy(float)
        DM = bao["DM_over_rd_fid"].to_numpy(float)
        DH = bao["DH_over_rd_fid"].to_numpy(float)
        return z, DM, DH, "fiducial_cols"
    die(f"BAO summary colonnes inconnues: {list(bao.columns)}")

def split_bao_vector(z, DM, DH, cov6):
    # Convention DR12 anisotropic: [DM(z1),DM(z2),DM(z3), DH(z1),DH(z2),DH(z3)]
    z = np.asarray(z)
    if z.size != 3:
        die("BAO summary doit contenir 3 redshifts pour DR12 anisotropic.")
    if cov6.shape != (6,6):
        die(f"BAO cov doit être 6x6, trouvé {cov6.shape}")

    y = np.concatenate([DM, DH])
    err = np.sqrt(np.clip(np.diag(cov6), 0.0, None))
    eDM = err[:3]
    eDH = err[3:]
    return z, DM, eDM, z, DH, eDH, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dossier outputs du pipeline (ex: outputs_v73)")
    ap.add_argument("--data-dir", default="data", help="dossier data (ex: data)")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not out_dir.exists(): die(f"--out introuvable: {out_dir}")
    if not data_dir.exists(): die(f"--data-dir introuvable: {data_dir}")

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    log_file = plots_dir / "run_plots.log"
    (plots_dir / "HEARTBEAT_plots.txt").write_text("OK\n", encoding="utf-8")

    log_line(log_file, f"[INFO] out_dir={out_dir}")
    log_line(log_file, f"[INFO] data_dir={data_dir}")
    log_line(log_file, f"[INFO] plots_dir={plots_dir}")

    ez_file, params_file, scan_meta_file, verdict_file = find_pipeline_outputs(out_dir)
    if ez_file is None: die(f"Ez introuvable dans {out_dir}")
    if params_file is None: die(f"best_fit_params introuvable dans {out_dir}")

    log_line(log_file, f"[INFO] Ez_file={ez_file.name}")
    log_line(log_file, f"[INFO] params_file={params_file.name}")
    log_line(log_file, f"[INFO] scan_meta_file={scan_meta_file.name if scan_meta_file else '(none)'}")
    log_line(log_file, f"[INFO] verdict_file={verdict_file.name if verdict_file else '(none)'}")

    ez = pd.read_csv(ez_file)
    if not {"z","E_genesys"}.issubset(ez.columns):
        die(f"Colonnes Ez inattendues: {list(ez.columns)} (attendu z,E_genesys)")
    zE = ez["z"].to_numpy(float)
    Eg = ez["E_genesys"].to_numpy(float)

    params = load_json(params_file)
    best = params.get("best_fit_parameters", params)
    if "H0" not in best:
        die("H0 introuvable dans best_fit_params.")
    H0 = float(best["H0"])
    log_line(log_file, f"[INFO] H0={H0:.6f} km/s/Mpc")

    Om_r_eff = None
    if scan_meta_file and scan_meta_file.exists():
        meta = load_json(scan_meta_file)
        try:
            Om_r_eff = float(meta["fixed_parameters"]["Omega_r0_eff"])
        except Exception:
            Om_r_eff = None
    if Om_r_eff is None:
        Om_r_eff = 8.5e-5
    log_line(log_file, f"[INFO] Omega_r0_eff(ref)={Om_r_eff:.8g}")

    # LCDM référence (positionnement, pas un fit)
    Om_m_LCDM = 0.315
    Om_r_LCDM = Om_r_eff
    Om_L_LCDM = 1.0 - Om_m_LCDM - Om_r_LCDM
    if Om_L_LCDM <= 0:
        die("Omega_L LCDM <= 0 (check Omega_r0_eff)")

    def E_lcdm(z):
        zp1 = 1.0 + z
        return np.sqrt(Om_r_LCDM * zp1**4 + Om_m_LCDM * zp1**3 + Om_L_LCDM)

    El = E_lcdm(zE)

    Dc_g = build_Dc_grid(zE, Eg, H0)
    Dc_l = build_Dc_grid(zE, El, H0)

    # -------------------------
    # E(z) zoom late-time
    # -------------------------
    zmax_late = 2.5
    mlate = zE <= zmax_late
    if mlate.sum() < 20:
        mlate = np.arange(min(800, zE.size))

    plt.figure(figsize=(8,5))
    plt.plot(zE[mlate], Eg[mlate], label="GeNeSyS")
    plt.plot(zE[mlate], El[mlate], label=r"$\Lambda$CDM ref ($\Omega_m=0.315$, $H_0$ fixé)")
    plt.xlabel("z"); plt.ylabel("E(z)=H(z)/H0")
    plt.title("Expansion history (zoom z<2.5)")
    plt.legend(); plt.tight_layout()
    plt.savefig(plots_dir/"fig_Ez_late_zoom.png", dpi=200)
    plt.close()

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(El[mlate] > 0, Eg[mlate]/El[mlate], np.nan)

    plt.figure(figsize=(8,5))
    plt.axhline(1.0, linewidth=1)
    plt.plot(zE[mlate], ratio, label="E_GeNeSyS / E_LCDM")
    plt.xlabel("z"); plt.ylabel("ratio")
    plt.title("E(z) ratio (zoom z<2.5)")
    plt.legend(); plt.tight_layout()
    plt.savefig(plots_dir/"fig_Ez_ratio_late_zoom.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(zE[mlate], Dc_l[mlate], label=r"$\Lambda$CDM $D_C(z)$")
    plt.plot(zE[mlate], Dc_g[mlate], label=r"GeNeSyS $D_C(z)$")
    plt.xlabel("z"); plt.ylabel(r"$D_C(z)$ [Mpc]")
    plt.title("Comoving distance (zoom z<2.5)")
    plt.legend(); plt.tight_layout()
    plt.savefig(plots_dir/"fig_Dc_late_zoom.png", dpi=200)
    plt.close()

    # -------------------------
    # SN plots (fallback using Pantheon+SH0ES.dat + cov)
    # -------------------------
    sn_dat = data_dir / "Pantheon+SH0ES.dat"
    if not sn_dat.exists():
        log_line(log_file, "[WARN] SN data introuvable: Pantheon+SH0ES.dat -> plots SN SKIPPED")
    else:
        sn = pd.read_csv(sn_dat, sep=r"\s+", comment="#")
        if not {"zHD","MU_SH0ES"}.issubset(sn.columns):
            die(f"Pantheon+SH0ES.dat colonnes inattendues: {list(sn.columns)} (attendu zHD, MU_SH0ES)")
        z_sn = sn["zHD"].to_numpy(float)
        mu_obs = sn["MU_SH0ES"].to_numpy(float)

        Dc_g_sn = np.interp(z_sn, zE, Dc_g)
        Dc_l_sn = np.interp(z_sn, zE, Dc_l)
        mu_g = mu_from_Dc(z_sn, Dc_g_sn)
        mu_l = mu_from_Dc(z_sn, Dc_l_sn)

        cov_sym = data_dir / "symmetrized_cov_matrix.cov"
        cov_stat_sys = data_dir / "Pantheon+SH0ES_STAT+SYS.cov"
        cov = None
        if cov_sym.exists():
            cov = load_pantheon_cov_symmetrized(cov_sym, N_expected=mu_obs.size)
            cov_name = cov_sym.name
        elif cov_stat_sys.exists():
            cov = load_pantheon_cov_STAT_SYS(cov_stat_sys)
            if cov.shape[0] != mu_obs.size:
                die(f"SN cov N={cov.shape[0]} != SN data N={mu_obs.size} (check file)")
            cov_name = cov_stat_sys.name
        else:
            die("Aucune covariance SN trouvée: symmetrized_cov_matrix.cov ou Pantheon+SH0ES_STAT+SYS.cov")

        a_g, chi2_g = fit_offset_fullcov(mu_obs - mu_g, cov)
        a_l, chi2_l = fit_offset_fullcov(mu_obs - mu_l, cov)

        res_g = mu_obs - (mu_g + a_g)
        res_l = mu_obs - (mu_l + a_l)
        sig_diag = np.sqrt(np.maximum(np.diag(cov), 1e-30))

        log_line(log_file, f"[INFO] SN cov used: {cov_name}")
        log_line(log_file, f"[INFO] SN chi2: GeNeSyS={chi2_g:.3f}  LCDM={chi2_l:.3f}")

        plt.figure(figsize=(10,7))
        plt.subplot(2,1,1)
        plt.scatter(z_sn, mu_obs, s=4, alpha=0.35, label="Pantheon+SH0ES")
        zz = np.linspace(max(1e-4, z_sn.min()), z_sn.max(), 1200)
        mu_g_line = mu_from_Dc(zz, np.interp(zz, zE, Dc_g)) + a_g
        plt.plot(zz, mu_g_line, linewidth=2, label=f"GeNeSyS (χ²/ν≈{chi2_g/(mu_obs.size-1):.2f})")
        plt.xscale("log")
        plt.ylabel("Distance modulus μ")
        plt.title("Pantheon+SH0ES — Hubble diagram")
        plt.legend()

        plt.subplot(2,1,2)
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.scatter(z_sn, res_g/sig_diag, s=4, alpha=0.35)
        plt.xscale("log")
        plt.xlabel("Redshift z")
        plt.ylabel("Residuals (σ units)")
        plt.tight_layout()
        plt.savefig(plots_dir/"fig_SN_hubble_and_residuals.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10,5))
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.scatter(z_sn, res_g, s=4, alpha=0.35, label=f"GeNeSyS (χ²={chi2_g:.1f})")
        plt.scatter(z_sn, res_l, s=4, alpha=0.20, label=f"ΛCDM ref (χ²={chi2_l:.1f})")
        plt.xlabel("Redshift z")
        plt.ylabel("Residuals Δμ (mag)")
        plt.title("Pantheon+ residuals (GeNeSyS vs ΛCDM ref)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir/"fig_SN_residuals_linear.png", dpi=200)
        plt.close()

    # -------------------------
    # BAO DR12 plots
    # -------------------------
    bao_sum = data_dir / "DR12_fid_DMrd_DHrd_summary.csv"
    bao_cov = data_dir / "DR12_cov6x6_DMrd_DHrd_from_consensus.csv"
    if not (bao_sum.exists() and bao_cov.exists()):
        log_line(log_file, "[WARN] BAO files introuvables -> plots BAO SKIPPED")
    else:
        z3, DM3, DH3, mode = load_bao_dr12(bao_sum)
        C6 = np.loadtxt(bao_cov, delimiter=",")
        C6 = 0.5 * (C6 + C6.T)
        z_dm, y_dm, e_dm, z_dh, y_dh, e_dh, d6 = split_bao_vector(z3, DM3, DH3, C6)

        rd_g = 143.0
        rd_l = 143.0
        if verdict_file and verdict_file.exists():
            v = load_json(verdict_file)
            try:
                rd_g = float(v["BAO"]["rd_best_GeNeSyS_Mpc"])
                rd_l = float(v["BAO"]["rd_best_LCDM_Mpc"])
            except Exception:
                pass

        def model_DM_over_rd(z_pts, Dc_grid, rd):
            return np.interp(z_pts, zE, Dc_grid) / rd

        def model_DH_over_rd(z_pts, E_grid, rd):
            Epts = np.interp(z_pts, zE, E_grid)
            Hz = H0 * Epts
            return (C_KM_S / Hz) / rd

        DMrd_g = model_DM_over_rd(z_dm, Dc_g, rd_g)
        DMrd_l = model_DM_over_rd(z_dm, Dc_l, rd_l)
        DHrd_g = model_DH_over_rd(z_dh, Eg, rd_g)
        DHrd_l = model_DH_over_rd(z_dh, El, rd_l)

        m6_g = np.concatenate([DMrd_g, DHrd_g])
        m6_l = np.concatenate([DMrd_l, DHrd_l])
        w_g = whiten_residuals(d6 - m6_g, C6)
        w_l = whiten_residuals(d6 - m6_l, C6)
        chi2_bao_g = float(np.sum(w_g*w_g))
        chi2_bao_l = float(np.sum(w_l*w_l))
        log_line(log_file, f"[INFO] BAO chi2 (whitened check): GeNeSyS={chi2_bao_g:.3f}  LCDM={chi2_bao_l:.3f}")

        plt.figure(figsize=(8,5))
        plt.errorbar(z_dm, y_dm, yerr=e_dm, fmt="o", capsize=3, label="DR12 DM/rd")
        plt.plot(z_dm, DMrd_g, linewidth=2, label=f"GeNeSyS (rd={rd_g:.2f})")
        plt.plot(z_dm, DMrd_l, linewidth=2, label=f"ΛCDM ref (rd={rd_l:.2f})")
        plt.xlabel("z"); plt.ylabel(r"$D_M(z)/r_d$")
        plt.title(f"BAO DR12 transverse (mode={mode})")
        plt.legend(); plt.tight_layout()
        plt.savefig(plots_dir/"fig_BAO_DR12_DMrd.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8,5))
        plt.errorbar(z_dh, y_dh, yerr=e_dh, fmt="s", capsize=3, label="DR12 DH/rd")
        plt.plot(z_dh, DHrd_g, linewidth=2, label=f"GeNeSyS (rd={rd_g:.2f})")
        plt.plot(z_dh, DHrd_l, linewidth=2, label=f"ΛCDM ref (rd={rd_l:.2f})")
        plt.xlabel("z"); plt.ylabel(r"$D_H(z)/r_d$")
        plt.title(f"BAO DR12 radial (mode={mode})")
        plt.legend(); plt.tight_layout()
        plt.savefig(plots_dir/"fig_BAO_DR12_DHrd.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10,5))
        plt.axhline(0, linewidth=1)
        x = np.arange(6)
        labels = [f"DM@{z_dm[i]:.2f}" for i in range(3)] + [f"DH@{z_dh[i]:.2f}" for i in range(3)]
        plt.plot(x, w_g, marker="o", label=f"GeNeSyS whitened (χ²={chi2_bao_g:.2f})")
        plt.plot(x, w_l, marker="s", label=f"ΛCDM ref whitened (χ²={chi2_bao_l:.2f})")
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.ylabel(r"$L^{-1}(d-m)$")
        plt.title("BAO whitened residuals (DR12 6-vector)")
        plt.legend(); plt.tight_layout()
        plt.savefig(plots_dir/"fig_BAO_whitened_residuals.png", dpi=200)
        plt.close()

    generated = sorted([p.name for p in plots_dir.glob("fig_*.png")])
    log_line(log_file, "[OK] Finished.")
    log_line(log_file, f"[OK] Generated {len(generated)} figures in {plots_dir}:")
    for n in generated:
        log_line(log_file, f"  - {n}")


if __name__ == "__main__":
    main()