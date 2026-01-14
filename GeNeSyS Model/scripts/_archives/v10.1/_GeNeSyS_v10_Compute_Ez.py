#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genesys_v31v3_compute_Ez.py

Recalcule la vraie courbe E(z) = H(z)/H0 du fond GeNeSyS v31v3
pour le modèle "best fit" trouvé dans _GeNeSyS_v10_background_scan.csv,
en réutilisant exactement les mêmes équations que le script de scan.

Sortie : un fichier CSV '_GeNeSyS_v10_best_Ez.csv' contenant :
    z, E_genesys
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# On réutilise les briques physiques déjà définies dans le script de scan
from genesys_v31v3_background_scan import (
    build_x_grid,
    compute_densities_today,
    equality_scale_factor,
    equality_window,
    DELTA_X_EQ,
    TAU_FAST,
    F_FAST,
    H0_REF,
)


# ============================================================================
# 1. Chargement du best-fit depuis le fichier de scan
# ============================================================================

def load_best_params(scan_csv: Path,
                     score_col: str = "score_LCDM_like") -> dict:
    """
    Charge le fichier de scan et renvoie un dictionnaire de paramètres
    pour la ligne présentant le meilleur score (score le plus petit).

    Paramètres
    ----------
    scan_csv : Path
        Chemin vers '_GeNeSyS_v10_background_scan.csv'.
    score_col : str
        Nom de la colonne contenant le score de similarité ΛCDM.

    Retour
    ------
    params : dict
        Paramètres numériques du best fit (x_c, sigma_c, tau_slow, alpha_M, etc.).
    """
    df = pd.read_csv(scan_csv)

    if score_col not in df.columns:
        raise ValueError(f"Colonne '{score_col}' introuvable dans {scan_csv}")

    best = df.loc[df[score_col].idxmin()]

    # Adapter ces noms si jamais ton CSV a des colonnes légèrement différentes
    params = {
        "x_c":      float(best["x_c"]),
        "sigma_c":  float(best["sigma_c"]),
        "tau_slow": float(best["tau_slow"]),
        "alpha_M":  float(best["alpha_M"]),
        "H0":       float(best.get("H0", H0_REF)),
        # On laisse tau_fast et f_fast aux valeurs globales par défaut
        "tau_fast": TAU_FAST,
        "f_fast":   F_FAST,
    }

    print("[INFO] Best-fit parameters from scan:")
    for k, v in params.items():
        print(f"  - {k:10s} = {v}")

    return params


# ============================================================================
# 2. Résolution du fond GeNeSyS pour un jeu de paramètres donné
#    (copie fidèle de la logique de compute_background_for_params,
#     mais en renvoyant la courbe E(z))
# ============================================================================

def compute_E_of_z_from_params(params: dict):
    """
    Calcule la vraie courbe E(z) = H(z)/H0 pour le fond GeNeSyS v31v3,
    en reprenant la structure de compute_background_for_params().

    Retour
    ------
    z : ndarray
        Redshift.
    E : ndarray
        E(z) = H(z)/H0.
    """
    xc       = params["x_c"]
    sig      = params["sigma_c"]
    tau_slow = params["tau_slow"]
    alpha_M  = params["alpha_M"]
    H0       = params["H0"]
    tau_fast = params["tau_fast"]
    f_fast   = params["f_fast"]

    # Grille en x = ln a, et conversion a, z
    x, a, z = build_x_grid()

    # Densités aujourd'hui à partir de H0 et des Ωh² fixés
    Omega_b0, Omega_r0, Omega_C0, Omega_M0, h = compute_densities_today(H0)

    # Égalité matière–rayonnement et fenêtre post-égalité
    a_eq = equality_scale_factor(Omega_b0, Omega_r0, Omega_C0)
    x_eq = np.log(a_eq)
    W_eq = equality_window(x, x_eq, DELTA_X_EQ)

    # Condensat M_C(x) (~ gaussienne en x)
    Jc = np.exp(-0.5 * ((x - xc) / sig)**2)
    dx = np.gradient(x)
    M_C_raw = np.cumsum(Jc * dx)
    if M_C_raw[-1] != 0.0:
        M_C = M_C_raw / M_C_raw[-1]
    else:
        M_C = np.zeros_like(x)

    # Mémoire M_M(x) (initialement 1 pour que tout soit bien défini)
    M_M = np.ones_like(x)

    dt = dx[0]  # x uniforme
    dec_s = np.exp(-dt / tau_slow)
    dec_f = np.exp(-dt / tau_fast)

    # Itérations pour convergence de M_M (comme dans compute_background_for_params)
    for _ in range(15):
        # E2 avec M_M courant
        E2 = (
            Omega_r0 / a**4 +
            Omega_b0 / a**3 +
            Omega_C0 * M_C / a**3 +
            Omega_M0 * M_M
        )
        E2 = np.maximum(E2, 1e-30)
        E  = np.sqrt(E2)

        # Courbure scalaire R(x) et source mémoire brute
        dE2_dx = np.gradient(E2, x)
        R      = 12.0 * E2 + 3.0 * a * dE2_dx / E
        dR_dx  = np.gradient(R, x)

        J_M_raw = np.maximum(-alpha_M * dR_dx, 0.0)

        # Fenêtre post-égalité appliquée à la source
        J_M = W_eq * J_M_raw

        # Intégration récursive du noyau mémoire bi-exponentiel
        acc_s = 0.0
        acc_f = 0.0
        M_new = np.zeros_like(x)
        for i in range(len(x)):
            acc_s = acc_s * dec_s + J_M[i] * dt
            acc_f = acc_f * dec_f + J_M[i] * dt
            M_new[i] = f_fast * acc_f + (1.0 - f_fast) * acc_s

        if M_new[-1] > 0.0:
            M_M = M_new / M_new[-1]
        else:
            # fallback : pas de mémoire
            M_M[:] = 1.0

    # Fond final avec M_M convergée
    E2 = (
        Omega_r0 / a**4 +
        Omega_b0 / a**3 +
        Omega_C0 * M_C / a**3 +
        Omega_M0 * M_M
    )
    E2 = np.maximum(E2, 1e-30)
    E  = np.sqrt(E2)

    return z, E


# ============================================================================
# 3. Sauvegarde dans un CSV : z, E_genesys
# ============================================================================

def save_E_of_z(z, E, output_csv: Path):
    df = pd.DataFrame({
        "z": z,
        "E_genesys": E
    })
    output_csv = output_csv.resolve()
    df.to_csv(output_csv, index=False)
    print(f"[OK] Courbe E(z) sauvegardée dans '{output_csv}'")


# ============================================================================
# 4. Interface ligne de commande
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Recalcule la courbe E(z) du fond GeNeSyS v31v3 "
                    "pour le best-fit du scan, et la sauvegarde dans un CSV."
    )
    parser.add_argument(
        "--scan",
        type=Path,
        default=Path("_GeNeSyS_v10_background_scan.csv"),
        help="Chemin vers le fichier de scan CSV (défaut: _GeNeSyS_v10_background_scan.csv)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("_GeNeSyS_v10_best_Ez.csv"),
        help="Fichier de sortie pour z, E(z) (défaut: _GeNeSyS_v10_best_Ez.csv)."
    )

    args = parser.parse_args()

    params = load_best_params(args.scan)
    z, E = compute_E_of_z_from_params(params)
    save_E_of_z(z, E, args.output)


if __name__ == "__main__":
    main()