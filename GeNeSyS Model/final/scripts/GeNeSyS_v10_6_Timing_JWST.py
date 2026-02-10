#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Eliot CHIMPEL

import argparse
import numpy as np
import os
import sys

def _load_npz(path: str) -> np.lib.npyio.NpzFile:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"NPZ introuvable: {path}")
    return np.load(path)

def _ensure_increasing_z(z: np.ndarray, *arrays):
    """
    Les NPZ peuvent être stockés en z décroissant (ex: 12 -> 0).
    On renvoie z croissant et les tableaux réordonnés de façon cohérente.
    """
    z = np.asarray(z)
    if z.ndim != 1:
        raise ValueError("z doit être 1D")
    if np.all(np.diff(z) > 0):
        return (z, *arrays)
    if np.all(np.diff(z) < 0):
        z2 = z[::-1]
        arr2 = []
        for a in arrays:
            a = np.asarray(a)
            if a.shape[-1] == z.shape[0]:
                arr2.append(a[..., ::-1])
            elif a.shape[0] == z.shape[0]:
                arr2.append(a[::-1, ...])
            else:
                arr2.append(a)  # tableau non-indexé par z
        return (z2, *arr2)
    raise ValueError("z n'est ni strictement croissant ni strictement décroissant.")

def _compute_A_from_Rk(Rk_re: np.ndarray, Rk_im: np.ndarray) -> np.ndarray:
    """
    Rk_re, Rk_im attendus en forme (nk, nz).
    A(z) = sqrt( mean_k ( Rk_re^2 + Rk_im^2 ) )
    """
    if Rk_re.shape != Rk_im.shape:
        raise ValueError(f"Rk_re et Rk_im shapes différentes: {Rk_re.shape} vs {Rk_im.shape}")
    if Rk_re.ndim != 2:
        raise ValueError(f"Rk_re doit être 2D (nk,nz). Reçu: {Rk_re.ndim}D")
    power = Rk_re**2 + Rk_im**2
    A = np.sqrt(np.mean(power, axis=0))
    if np.any(~np.isfinite(A)) or np.any(A <= 0):
        raise ValueError("A(z) contient des valeurs non-finies ou non-positives.")
    return A

def _nuG_from_A(z: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    nu_G(z) = d ln A / d ln a
            = (d ln A / dz) / (d ln a / dz)
    avec a = 1/(1+z) => ln a = -ln(1+z) => d ln a / dz = -(1/(1+z))
    donc nu_G = -(1+z) * d ln A / dz
    """
    lnA = np.log(A)
    dlnA_dz = np.gradient(lnA, z)  # gère z non-uniforme
    nuG = -(1.0 + z) * dlnA_dz
    return nuG

def _print_key_points(z: np.ndarray, nuG: np.ndarray):
    def interp(z0):
        return float(np.interp(z0, z, nuG))
    for z0 in [6.0, 8.0, 10.0, float(z[-1])]:
        val = interp(z0)
        ratio = val / 1.0
        print(f"z={z0:5.2f} | nu_G={val:6.3f} | f_LCDM≈1.00 | ratio={ratio:5.2f}")

def main():
    ap = argparse.ArgumentParser(
        description="GeNeSyS — JWST Timing (Local chronology) from NPZ outputs"
    )
    ap.add_argument("--geom-npz", required=True)
    ap.add_argument("--eb3-npz", required=True)
    ap.add_argument("--aux-npz", required=True)
    ap.add_argument("--z-min", type=float, default=3.0)
    ap.add_argument("--z-max", type=float, default=12.0)
    ap.add_argument("--out-fig", required=True)
    args = ap.parse_args()

    eb3 = _load_npz(args.eb3_npz)
    aux = _load_npz(args.aux_npz)

    # --- EB3: z, Rk modes
    for k in ["z", "Rk_re", "Rk_im"]:
        if k not in eb3.files:
            raise KeyError(f"Clé manquante dans EB3 NPZ: '{k}'. Présent: {eb3.files}")

    z_raw = eb3["z"]
    Rk_re_raw = eb3["Rk_re"]
    Rk_im_raw = eb3["Rk_im"]

    # z en ordre croissant + modes alignés
    z, Rk_re, Rk_im = _ensure_increasing_z(z_raw, Rk_re_raw, Rk_im_raw)

    # A(z) et nu_G(z)
    A = _compute_A_from_Rk(Rk_re, Rk_im)
    nuG = _nuG_from_A(z, A)

    # fenêtre z
    zmin = float(args.z_min)
    zmax = float(args.z_max)
    if zmin >= zmax:
        raise ValueError("--z-min doit être < --z-max")
    mask = (z >= zmin) & (z <= zmax)
    if mask.sum() < 5:
        raise ValueError("Fenêtre z trop petite / pas assez de points.")
    z_sel = z[mask]
    nu_sel = nuG[mask]

    # --- Preuves numériques sur la fenêtre
    # 1) Toujours au-dessus de 1 ?
    min_excess = float(np.min(nu_sel - 1.0))
    # 2) Monotonie (nu décroît quand z augmente) ?
    monotone = bool(np.all(np.diff(nu_sel) <= 0.0))

    print("\n=== JWST Local Chronology Check (NPZ-driven) ===\n")
    _print_key_points(z_sel, nu_sel)
    print("\n--- Proofs on selected window ---")
    print(f"Window: z in [{z_sel[0]:.3f}, {z_sel[-1]:.3f}]  (N={z_sel.size})")
    print(f"min(nu_G - 1) = {min_excess:.6f}  -> {'OK (nu_G>1)' if min_excess>0 else 'NOT OK'}")
    print(f"nu_G monotone decreasing with z? {monotone}")

    # --- Indice “mémoire éteinte” au bord haut (aux: phi, U)
    phi0 = None
    U0 = None
    if "phi" in aux.files and "U" in aux.files:
        a_raw = aux["a"]
        phi_raw = aux["phi"]
        U_raw = aux["U"]
        # convertir a->z (cohérence interne)
        z_aux_raw = 1.0 / a_raw - 1.0
        z_aux, phi_aux, U_aux = _ensure_increasing_z(z_aux_raw, phi_raw, U_raw)
        # prendre la valeur au plus grand z disponible
        phi0 = float(phi_aux[-1])
        U0 = float(U_aux[-1])
        print("\n--- Auxiliary fields at highest-z in aux NPZ (indicator) ---")
        print(f"aux z_max ≈ {float(z_aux[-1]):.6f} | phi(z_max)={phi0:.6e} | U(z_max)={U0:.6e}")

    # --- Figure (propre, sans spike de z~0)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10.5, 7.5))
    plt.plot(z_sel, nu_sel, linewidth=2.5, label=r"GeNeSyS $\nu_G(z)=d\ln A/d\ln a$")
    plt.axhline(1.0, linestyle="--", linewidth=2.0, label=r"$\Lambda$CDM $f(z)\simeq 1$ (high-$z$)")
    plt.title("JWST Timing — Local Chronology (H67)")
    plt.xlabel("Redshift z")
    plt.ylabel("Local structuration rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(args.out_fig) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=160)
    print(f"\n[OK] Figure saved → {args.out_fig}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)