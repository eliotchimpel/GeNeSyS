#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeNeSyS v31v3 — Scan FOND-ONLY avec source mémoire EFT-like + fenêtre post-égalité

Objectif :
    - Étudier le fond cosmologique GeNeSyS (sans data tardive),
    - Avec une source mémoire J_M dérivée de R(x),
    - MAIS activée uniquement après l’égalité matière–radiation via une fenêtre W_eq(x).

Ce script :
    1. Définit les paramètres physiques (Ω_b h², Ω_r h², Ω_C0 h², H0_ref).
    2. Pour chaque jeu (x_c, sigma_c, tau_slow, alpha_M) d’une grille :
        - Construit le condensat M_C(x),
        - Construit la mémoire M_M(x) via un noyau bi-exponentiel,
        - Avec J_M(x) = W_eq(x) * max[-alpha_M * dR/dx, 0],
        - Calcule E(a), H(a), t0, w_M0, f_DE(z*) et Ω_i(0),
        - Évalue un score "ΛCDM-like".
    3. Sauvegarde tous les résultats dans _GeNeSyS_v10_background_scan.csv
       et imprime les 10 meilleurs modèles (fond le plus ΛCDM-like).
"""

import numpy as np
import pandas as pd

# =============================================================================
# 1. CONSTANTES ET PARAMÈTRES GLOBAUX
# =============================================================================

C_LIGHT = 299792.458       # km/s
H0_REF  = 70.0             # km/s/Mpc, uniquement comme échelle temporelle
OMBH2   = 0.02242          # baryons physiques (Planck-like)
OMRH2   = 4.18e-5          # rayonnement (photons + ν effectif)
OMC0H2  = 0.120            # CDM physique (condensat émergent)

# Noyau mémoire bi-exponentiel (on les fixe ici, comme en v30)
TAU_FAST = 0.7
F_FAST   = 0.2

# Fenêtre post-égalité (Verlinde-inspired)
DELTA_X_EQ = 0.5  # largeur de la transition en x = ln a

# Grille micro-physique (identique à v31v2)
X_C_LIST      = [-10.0, -9.0, -8.0, -7.0]
SIGMA_C_LIST  = [0.30, 0.50, 0.80]
TAU_SLOW_LIST = [4.0, 5.0, 6.0]
ALPHA_M_LIST  = [0.005, 0.010, 0.020, 0.050]

# Grille en x = ln a
X_MIN = -15.0
X_MAX = 0.0
NPTS  = 2000

# =============================================================================
# 2. OUTILS NUMÉRIQUES
# =============================================================================

def build_x_grid(x_min=X_MIN, x_max=X_MAX, npts=NPTS):
    x = np.linspace(x_min, x_max, npts)
    a = np.exp(x)
    z = 1.0 / a - 1.0
    return x, a, z

def compute_densities_today(H0):
    """
    Retourne (Omega_b0, Omega_r0, Omega_C0, Omega_M0, h)
    avec Omega_M0 = 1 - Omega_b0 - Omega_r0 - Omega_C0.
    """
    h = H0 / 100.0
    Omega_b0 = OMBH2  / h**2
    Omega_r0 = OMRH2  / h**2
    Omega_C0 = OMC0H2 / h**2
    Omega_M0 = 1.0 - Omega_b0 - Omega_r0 - Omega_C0
    return Omega_b0, Omega_r0, Omega_C0, Omega_M0, h

def equality_scale_factor(Omega_b0, Omega_r0, Omega_C0):
    """
    Estime a_eq via Ω_r0 / a^4 = Ω_m0_eff / a^3,
    avec Ω_m0_eff = Ω_b0 + Ω_C0 (on néglige la mémoire avant égalité).
      => a_eq = Ω_r0 / Ω_m0_eff
    """
    Omega_m_eff = Omega_b0 + Omega_C0
    a_eq = Omega_r0 / max(Omega_m_eff, 1e-12)
    return a_eq

def equality_window(x, x_eq, delta_x):
    """
    Fenêtre sigmoïde post-égalité :
        W_eq(x) = 0.5 * [1 + tanh((x - x_eq)/delta_x)]
    """
    return 0.5 * (1.0 + np.tanh((x - x_eq) / delta_x))

def compute_age_in_Gyr(x, E, h):
    """
    t0 = (1/H0) ∫ dx / E(x)
    avec H0 = 100 h km/s/Mpc, 1/H0 ≈ 9.778/h Gyr
    """
    integrand = 1.0 / np.maximum(E, 1e-30)
    integral = np.trapz(integrand, x)
    age_Gyr = (9.778 / h) * integral
    return age_Gyr

def compute_w_M0(x, a, M_M):
    """
    Approximation de w_M0 via :
       w_M = -1 - (1/3) d ln ρ_M / d ln a
    avec ρ_M ∝ M_M.
    """
    # dérivée en ln a au bord tardif
    ln_a = x  # car x = ln a
    ln_M = np.log(np.maximum(M_M, 1e-30))
    dlnM_dlnA = np.gradient(ln_M, ln_a)

    # on prend la valeur au dernier point (a ~ 1)
    w_M0 = -1.0 - (1.0 / 3.0) * dlnM_dlnA[-1]
    return w_M0

def compute_f_DE_zstar(a, x, E2, M_M, Omega_M0, z_star=1100.0):
    """
    Fraction d'énergie "mémoire" à recombinaison :
        f_DE(z*) = ρ_M / ρ_tot = Ω_M0 M_M(a*) / E2(a*)
    avec E2(a) = Ω_r0/a^4 + Ω_b0/a^3 + Ω_C0 M_C/a^3 + Ω_M0 M_M.
    """
    a_star = 1.0 / (1.0 + z_star)
    x_star = np.log(a_star)
    # interpolation
    E2_star = np.interp(x_star, x, E2)
    M_M_star = np.interp(x_star, x, M_M)
    f_DE = (Omega_M0 * M_M_star) / max(E2_star, 1e-30)
    return f_DE

def score_LCDM_like(w_M0, age_Gyr, Omega_tot0, f_DE_zstar):
    """
    Score simple de "distance" à ΛCDM-like sur le FOND :
      - w_M0 ≈ -1 (tolérance ~0.05),
      - age ≈ 13.8 Gyr (tolérance ~0.5),
      - Omega_tot0 ≈ 1 (tolérance ~0.01),
      - f_DE(z*) ≪ 1 (on pénalise 1e-3 comme scale).
    """
    term_w   = ((w_M0 + 1.0) / 0.05)**2
    term_age = ((age_Gyr - 13.8) / 0.5)**2
    term_Ot  = ((Omega_tot0 - 1.0) / 0.01)**2
    term_fDE = (f_DE_zstar / 1e-3)**2
    return term_w + term_age + term_Ot + term_fDE

# =============================================================================
# 3. FOND GeNeSyS POUR UN JEU DE PARAMÈTRES
# =============================================================================

def compute_background_for_params(xc, sig, tau_slow, alpha_M,
                                  H0=H0_REF,
                                  tau_fast=TAU_FAST,
                                  f_fast=F_FAST):
    """
    Calcule le fond GeNeSyS pour un jeu (xc, sig, tau_slow, alpha_M) :

      - Condensat M_C(x) gaussien,
      - Mémoire M_M(x) par noyau bi-exponentiel,
      - Source J_M(x) = W_eq(x) * max[-alpha_M dR/dx, 0],
      - E2(a), E(a) et diagnostics (t0, w_M0, f_DE(z*), Ω_i(0)).
    """
    # Grille
    x, a, z = build_x_grid()

    # Densités aujourd'hui
    Omega_b0, Omega_r0, Omega_C0, Omega_M0, h = compute_densities_today(H0)

    # Égalité matière–radiation
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

    # Itérations pour convergence de M_M
    for _ in range(15):
        # E2 sans écraser M_M (on utilise la valeur courante)
        E2 = (
            Omega_r0 / a**4 +
            Omega_b0 / a**3 +
            Omega_C0 * M_C / a**3 +
            Omega_M0 * M_M
        )
        E2 = np.maximum(E2, 1e-30)
        E  = np.sqrt(E2)

        # R(x) et source mémoire brute
        dE2_dx = np.gradient(E2, x)
        R      = 12.0 * E2 + 3.0 * a * dE2_dx / E
        dR_dx  = np.gradient(R, x)

        J_M_raw = np.maximum(-alpha_M * dR_dx, 0.0)

        # Fenêtre post-égalité
        J_M = W_eq * J_M_raw

        # Intégration récursive du noyau mémoire
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

    # Fond final
    E2 = (
        Omega_r0 / a**4 +
        Omega_b0 / a**3 +
        Omega_C0 * M_C / a**3 +
        Omega_M0 * M_M
    )
    E2 = np.maximum(E2, 1e-30)
    E  = np.sqrt(E2)

    # Diagnostics
    age_Gyr = compute_age_in_Gyr(x, E, h)
    w_M0    = compute_w_M0(x, a, M_M)
    f_DE_star = compute_f_DE_zstar(a, x, E2, M_M, Omega_M0, z_star=1100.0)

    # Densités aujourd’hui (a=1 ~ dernier point)
    Omega_r_today = Omega_r0 / a[-1]**4
    Omega_b_today = Omega_b0 / a[-1]**3
    Omega_C_today = Omega_C0 * M_C[-1] / a[-1]**3
    Omega_M_today = Omega_M0 * M_M[-1]
    Omega_tot0    = Omega_r_today + Omega_b_today + Omega_C_today + Omega_M_today

    return dict(
    x_c=xc,
    sigma_c=sig,
    tau_slow=tau_slow,
    alpha_M=alpha_M,
    tau_fast=tau_fast,
    f_fast=f_fast,
    H0=H0,

    # Ajoutés pour le mini-scan (fond tabulé)
    x=x,
    a=a,
    z=z,
    E=E,

    # Diagnostics 
    age_Gyr=age_Gyr,
    w_M0=w_M0,
    Omega_b0=Omega_b_today / Omega_tot0,
    Omega_r0=Omega_r_today / Omega_tot0,
    Omega_C0=Omega_C_today / Omega_tot0,
    Omega_M0=Omega_M_today / Omega_tot0,
    Omega_tot0=Omega_tot0,
    f_DE_zstar=f_DE_star,
)

# =============================================================================
# 4. SCAN DE LA GRILLE ET SAUVEGARDE
# =============================================================================

def main():
    results = []
    total = len(X_C_LIST) * len(SIGMA_C_LIST) * len(TAU_SLOW_LIST) * len(ALPHA_M_LIST)
    idx = 0

    print("=== SCAN FOND GeNeSyS v31v3 (fond-only, fenêtre post-égalité) ===")
    for xc in X_C_LIST:
        for sig in SIGMA_C_LIST:
            for tau_s in TAU_SLOW_LIST:
                for aM in ALPHA_M_LIST:
                    idx += 1
                    print(f"[{idx:3d}/{total}] x_c={xc}, σ_c={sig}, τ_slow={tau_s}, α_M={aM}")
                    try:
                        res = compute_background_for_params(
                            xc, sig, tau_s,
                            aM,
                            H0=H0_REF,
                            tau_fast=TAU_FAST,
                            f_fast=F_FAST
                        )
                        # Score ΛCDM-like
                        s = score_LCDM_like(
                            res["w_M0"],
                            res["age_Gyr"],
                            res["Omega_tot0"],
                            res["f_DE_zstar"]
                        )
                        res["score_LCDM_like"] = s
                        results.append(res)
                    except Exception as e:
                        print(f"    -> Échec numérique : {e}")
                        continue

    if len(results) == 0:
        print("Aucun modèle calculé (problème numérique global ?).")
        return

    df = pd.DataFrame(results)
    outfile = "_GeNeSyS_v10_background_scan.csv"
    df.to_csv(outfile, index=False)
    print(f"[SAVE] Résultats fond -> {outfile}")

    # Tri par score croissant (plus ΛCDM-like en haut)
    df_sorted = df.sort_values(by="score_LCDM_like", ascending=True)
    top = df_sorted.head(10)

    print("\n=== TOP 10 modèles (fond le plus ΛCDM-like, v31v3) ===")
    cols_show = [
        "x_c", "sigma_c", "tau_slow", "alpha_M",
        "tau_fast", "f_fast", "H0",
        "age_Gyr", "w_M0",
        "Omega_b0", "Omega_r0", "Omega_C0", "Omega_M0",
        "f_DE_zstar", "score_LCDM_like"
    ]
    print(top[cols_show].to_string(index=False))

if __name__ == "__main__":
    main()