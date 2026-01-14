#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
_GeNeSyS_v10_SN_BAO_pipeline.py

Pipeline SN Ia (Pantheon+SH0ES) + BAO (BOSS DR12) pour le fond GeNeSyS tabulé.

Ce script :
  - charge le fond cosmologique GeNeSyS tabulé dans _GeNeSyS_v10_best_Ez.csv
    (redshift z, fonction de Hubble normalisée E(z) = H(z)/H0),
  - reconstruit la distance comobile D_M(z) et la distance de luminosité D_L(z),
  - calcule le chi carré complet pour les supernovae de type Ia (Pantheon+SH0ES),
  - calcule le chi carré BAO pour les mesures BOSS DR12 (D_M/r_d et D_H/r_d),
  - génère des figures explicites (diagramme de Hubble, BAO, H(z)),
  - écrit un fichier JSON résumant les résultats numériques.

Usage minimal (dans le répertoire des fichiers) :

    python _GeNeSyS_v10_SN_BAO_pipeline.py

Les options par défaut supposent :

    - _GeNeSyS_v10_best_Ez.csv
    - Pantheon+SH0ES.dat
    - Pantheon+SH0ES_STAT+SYS.cov
    - DR12_fid_DMrd_DHrd_summary.csv
    - DR12_cov6x6_DMrd_DHrd_from_consensus.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Constantes physiques
# -------------------------------------------------------------------

C_KMS = 299792.458  # vitesse de la lumière en km/s


# -------------------------------------------------------------------
# 1. Chargement du fond GeNeSyS : z, E(z)
# -------------------------------------------------------------------

def load_genesys_best_Ez(path_csv: str, H0: float):
    """
    Charge le fond GeNeSyS tabulé : redshift z, fonction E(z) = H(z)/H0.

    Le fichier _GeNeSyS_v10_best_Ez.csv contient typiquement :
        z, E_genesys

    Paramètres
    ----------
    path_csv : str
        Chemin vers le fichier CSV du fond tabulé.
    H0 : float
        Valeur du paramètre de Hubble aujourd'hui (en km/s/Mpc).

    Renvoie
    -------
    z_sorted : np.ndarray
        Tableau des redshifts, triés par ordre croissant.
    E_sorted : np.ndarray
        Tableau de E(z) correspondant à H(z)/H0.
    H0 : float
        La valeur de H0 utilisée (pour cohérence avec le reste du pipeline).
    """
    print(f"Chargement du fond GeNeSyS depuis : {path_csv}")
    df = pd.read_csv(path_csv)

    if "z" not in df.columns:
        raise RuntimeError(
            f"Le fichier {path_csv} ne contient pas de colonne 'z'. "
            f"Colonnes disponibles : {df.columns}"
        )

    # On cherche une colonne plausible pour E(z)
    E_col = None
    for cand in ["E_genesys", "E", "Ez"]:
        if cand in df.columns:
            E_col = cand
            break
    if E_col is None:
        raise RuntimeError(
            f"Impossible de trouver une colonne E(z) dans {path_csv}. "
            f"Colonnes disponibles : {df.columns}"
        )

    z = df["z"].to_numpy(dtype=float)
    E = df[E_col].to_numpy(dtype=float)

    # Tri en redshift croissant
    idx = np.argsort(z)
    z_sorted = z[idx]
    E_sorted = E[idx]

    print(
        f"Fond GeNeSyS chargé : {len(z_sorted)} points, "
        f"z in [{z_sorted[0]:.3e}, {z_sorted[-1]:.3e}]"
    )

    return z_sorted, E_sorted, H0


def compute_DM_from_Ez(z_grid: np.ndarray, E_grid: np.ndarray, H0: float) -> np.ndarray:
    """
    Calcule la distance comobile D_M(z) (en Mpc) à partir de E(z) = H(z)/H0.

    Définitions :
        D_C(z) = (c / H0) * ∫_0^z [dz' / E(z')]
        D_M(z) = D_C(z) en univers plat.

    Paramètres
    ----------
    z_grid : np.ndarray
        Grille de redshifts (triée dans l'ordre croissant).
    E_grid : np.ndarray
        Fonction E(z) = H(z)/H0 tabulée sur z_grid.
    H0 : float
        Paramètre de Hubble aujourd'hui (en km/s/Mpc).

    Renvoie
    -------
    DM : np.ndarray
        Distance comobile D_M(z) en Mpc, pour chaque z de z_grid.
    """
    integrand = 1.0 / E_grid  # f(z) = 1 / E(z)

    Dc_int = np.zeros_like(z_grid)
    for i in range(1, len(z_grid)):
        dz = z_grid[i] - z_grid[i - 1]
        Dc_int[i] = Dc_int[i - 1] + 0.5 * (integrand[i] + integrand[i - 1]) * dz

    DM = (C_KMS / H0) * Dc_int  # Mpc
    return DM


# -------------------------------------------------------------------
# 2. Supernovae de type Ia : Pantheon+SH0ES
# -------------------------------------------------------------------

def load_sn_pantheon_sh0es(sn_table: str, sn_cov: str):
    """
    Charge les données Pantheon+SH0ES et la matrice de covariance complète.

    Fichier sn_table (par ex. "Pantheon+SH0ES.dat") :
        - colonne 'zHD'      : redshift corrigé du flot de Hubble
        - colonne 'MU_SH0ES' : distance modulaire µ(z) déjà calibrée par SH0ES

    Fichier sn_cov (par ex. "Pantheon+SH0ES_STAT+SYS.cov") :
        - première ligne : N (nombre de supernovae, ici N = 1701)
        - lignes suivantes : N^2 valeurs de covariance, aplaties

    Paramètres
    ----------
    sn_table : str
        Chemin vers la table de supernovae.
    sn_cov : str
        Chemin vers la covariance STAT+SYS.

    Renvoie
    -------
    z_sn : np.ndarray
        Redshifts des supernovae.
    mu_obs : np.ndarray
        Distances modulaires observées (en magnitudes).
    icov : np.ndarray
        Matrice inverse de covariance (N x N).
    """
    sn = pd.read_csv(sn_table, sep=r"\s+", comment="#")

    if "zHD" not in sn.columns:
        raise RuntimeError(
            f"Colonne 'zHD' absente de {sn_table}. Colonnes : {sn.columns}"
        )
    if "MU_SH0ES" not in sn.columns:
        raise RuntimeError(
            f"Colonne 'MU_SH0ES' absente de {sn_table}. Colonnes : {sn.columns}"
        )

    z_sn = sn["zHD"].to_numpy(dtype=float)
    mu_obs = sn["MU_SH0ES"].to_numpy(dtype=float)

    # Lecture de la covariance
    with open(sn_cov, "r") as f:
        first = f.readline().strip()
        try:
            N = int(first)
        except ValueError as e:
            raise RuntimeError(
                f"Première ligne de {sn_cov} = '{first}', attendu un entier N (ex : 1701)."
            ) from e

        arr = np.loadtxt(f)

    if arr.size != N * N:
        raise RuntimeError(
            f"Taille de la covariance SN incorrecte : {arr.size}, attendu {N*N}."
        )

    cov = arr.reshape((N, N))
    try:
        icov = np.linalg.inv(cov)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Impossible d'inverser la covariance SN. "
            f"Vérifie que le fichier {sn_cov} est correct."
        ) from e

    return z_sn, mu_obs, icov


def compute_mu_model(z_sn: np.ndarray,
                     z_grid: np.ndarray,
                     DM_grid: np.ndarray) -> np.ndarray:
    """
    Calcule la distance modulaire µ(z) du modèle à partir de D_M(z).

    Définitions cosmologiques :
      - Distance comobile D_M(z) en Mpc (univers plat),
      - Distance de luminosité : D_L(z) = (1+z) * D_M(z),
      - Distance modulaire (appliquée aux supernovae de type Ia) :
            µ(z) = 5 * log10(D_L(z) / 1 Mpc) + 25.

    Paramètres
    ----------
    z_sn : np.ndarray
        Redshifts des supernovae.
    z_grid : np.ndarray
        Grille de redshifts sur laquelle D_M(z) est tabulée.
    DM_grid : np.ndarray
        Distance comobile D_M(z) du modèle, en Mpc.

    Renvoie
    -------
    mu_mod : np.ndarray
        Distance modulaire prédite par le modèle GeNeSyS pour chaque supernova.
    """
    # Interpolation de D_M(z) sur les redshifts des SN
    DM_sn = np.interp(z_sn, z_grid, DM_grid)
    # Distance de luminosité
    DL_sn = (1.0 + z_sn) * DM_sn  # en Mpc

    if np.any(DL_sn <= 0.0):
        raise RuntimeError(
            "Certaines distances de luminosité D_L(z) sont <= 0. "
            "Vérifie la cohérence du fond E(z) et des redshifts SN."
        )

    # Distance modulaire
    mu_mod = 5.0 * np.log10(DL_sn) + 25.0
    return mu_mod


def chi2_sn_full(mu_obs: np.ndarray,
                 mu_mod: np.ndarray,
                 icov: np.ndarray) -> float:
    """
    Calcule le chi carré complet pour les supernovae de type Ia.

    On utilise la distance modulaire calibrée µ_obs = MU_SH0ES,
    et la matrice de covariance STAT+SYS fournie par Pantheon+SH0ES.

    Paramètres
    ----------
    mu_obs : np.ndarray
        Distances modulaires observées (Pantheon+SH0ES).
    mu_mod : np.ndarray
        Distances modulaires prédites par le modèle GeNeSyS.
    icov : np.ndarray
        Matrice inverse de covariance (N x N).

    Renvoie
    -------
    chi2 : float
        Valeur du chi carré pour l'ensemble des supernovae.
    """
    d = mu_obs - mu_mod
    chi2 = float(d @ icov @ d)
    return chi2


# -------------------------------------------------------------------
# 3. BAO BOSS DR12 (D_M/r_d, D_H/r_d)
# -------------------------------------------------------------------

def load_bao_dr12(summary_path: str, cov_path: str):
    """
    Charge les données BAO BOSS DR12 (D_M/r_d et D_H/r_d) et la covariance.

    Fichier summary (par ex. "DR12_fid_DMrd_DHrd_summary.csv") :
        - colonnes : z, DM_over_rd_fid, DH_over_rd_fid, ...

    Fichier covariance (par ex. "DR12_cov6x6_DMrd_DHrd_from_consensus.csv") :
        - matrice 6 x 6 (après nettoyage des séparateurs).

    Renvoie
    -------
    z_bao : np.ndarray
        Redshifts BAO (3 valeurs : 0.38, 0.51, 0.61).
    DM_over_rd_obs : np.ndarray
        Mesures observées D_M(z)/r_d.
    DH_over_rd_obs : np.ndarray
        Mesures observées D_H(z)/r_d.
    icov : np.ndarray
        Matrice inverse de covariance 6 x 6.
    """
    # Lecture du fichier de summary BAO
    bao = pd.read_csv(summary_path, sep=",")
    # Nettoyage des éventuellement noms de colonnes avec espaces
    bao.columns = [c.strip() for c in bao.columns]

    # On attend des noms de colonnes compatibles avec :
    #   'z', 'DM_over_rd_fid', 'DH_over_rd_fid'
    if "z" not in bao.columns:
        raise RuntimeError(
            f"Aucune colonne 'z' dans {summary_path}. Colonnes = {bao.columns}"
        )

    # On tente les colonnes pour D_M/r_d et D_H/r_d
    dm_col = None
    dh_col = None
    for cand in ["DM_over_rd_fid", "DM/rd_fid", "DMrd_fid", "DM_over_rd"]:
        if cand in bao.columns:
            dm_col = cand
            break
    for cand in ["DH_over_rd_fid", "DH/rd_fid", "DHrd_fid", "DH_over_rd"]:
        if cand in bao.columns:
            dh_col = cand
            break

    if dm_col is None or dh_col is None:
        raise RuntimeError(
            f"Impossible de trouver des colonnes DM/rd et DH/rd dans {summary_path}. "
            f"Colonnes = {bao.columns}"
        )

    z_bao = bao["z"].to_numpy(dtype=float)
    DM_over_rd_obs = bao[dm_col].to_numpy(dtype=float)
    DH_over_rd_obs = bao[dh_col].to_numpy(dtype=float)

    # Lecture de la covariance BAO (6 x 6)
    # Lecture de la covariance BAO : fichier CSV avec virgules
    cov = pd.read_csv(cov_path, header=None).to_numpy(dtype=float)
    
    if cov.shape != (6, 6):
        raise RuntimeError(
            f"Covariance BAO doit être 6x6, trouvé {cov.shape}. "
            f"Vérifie le fichier {cov_path}."
        )

    try:
        icov = np.linalg.inv(cov)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Impossible d'inverser la covariance BAO dans {cov_path}."
        ) from e

    return z_bao, DM_over_rd_obs, DH_over_rd_obs, icov


def chi2_bao_dr12(z_bao: np.ndarray,
                  DM_over_rd_obs: np.ndarray,
                  DH_over_rd_obs: np.ndarray,
                  icov: np.ndarray,
                  z_grid: np.ndarray,
                  DM_grid: np.ndarray,
                  E_grid: np.ndarray,
                  H0: float,
                  rd: float):
    """
    Calcule le chi carré BAO pour les mesures BOSS DR12.

    On construit pour chaque redshift BAO :
        - la distance comobile transverse prédite : D_M(z) / r_d
        - la distance radiale prédite : D_H(z) / r_d = c / (H(z) r_d)

    puis on forme un vecteur de longueur 2*N :
        y_obs = (DM/rd_obs_1, DH/rd_obs_1, ..., DM/rd_obs_N, DH/rd_obs_N)
        y_mod = (DM/rd_mod_1, DH/rd_mod_1, ..., DM/rd_mod_N, DH/rd_mod_N)

    et on calcule :
        chi2 = (y_mod - y_obs)^T * C^{-1} * (y_mod - y_obs).

    Paramètres
    ----------
    z_bao : np.ndarray
        Redshifts BAO (3 valeurs typiquement : 0.38, 0.51, 0.61).
    DM_over_rd_obs : np.ndarray
        Mesures observées de D_M(z) / r_d.
    DH_over_rd_obs : np.ndarray
        Mesures observées de D_H(z) / r_d.
    icov : np.ndarray
        Matrice inverse de covariance BAO (6 x 6).
    z_grid : np.ndarray
        Grille de redshifts du fond GeNeSyS.
    DM_grid : np.ndarray
        Distance comobile GeNeSyS D_M(z) sur la grille.
    E_grid : np.ndarray
        Fonction E(z) = H(z)/H0 sur la grille.
    H0 : float
        Paramètre de Hubble aujourd'hui (km/s/Mpc).
    rd : float
        Règle acoustique effective r_d (Mpc).

    Renvoie
    -------
    chi2 : float
        Valeur du chi carré BAO.
    details : dict
        Dictionnaire contenant les valeurs DM/rd et DH/rd modélisées.
    """
    # Interpolation de D_M sur les z BAO
    DM_bao = np.interp(z_bao, z_grid, DM_grid)
    # H(z) = E(z) * H0
    E_bao = np.interp(z_bao, z_grid, E_grid)
    H_bao = E_bao * H0

    # Distances BAO modélisées
    DM_over_rd_mod = DM_bao / rd
    DH_over_rd_mod = C_KMS / (H_bao * rd)

    # Construction des vecteurs y_obs, y_mod
    y_obs = np.zeros(2 * len(z_bao))
    y_mod = np.zeros(2 * len(z_bao))

    # Convention : (DM/rd_1, DH/rd_1, DM/rd_2, DH/rd_2, ...)
    for i in range(len(z_bao)):
        y_obs[2 * i] = DM_over_rd_obs[i]
        y_obs[2 * i + 1] = DH_over_rd_obs[i]
        y_mod[2 * i] = DM_over_rd_mod[i]
        y_mod[2 * i + 1] = DH_over_rd_mod[i]

    d = y_mod - y_obs
    chi2 = float(d @ icov @ d)

    details = {
        "z_BAO": z_bao.tolist(),
        "DM_over_rd_obs": DM_over_rd_obs.tolist(),
        "DM_over_rd_mod": DM_over_rd_mod.tolist(),
        "DH_over_rd_obs": DH_over_rd_obs.tolist(),
        "DH_over_rd_mod": DH_over_rd_mod.tolist(),
    }

    return chi2, details


# -------------------------------------------------------------------
# 4. Tracés : Hubble, BAO, H(z)
# -------------------------------------------------------------------

def make_plots(z_grid: np.ndarray,
               E_grid: np.ndarray,
               H0: float,
               z_sn: np.ndarray,
               mu_obs: np.ndarray,
               mu_mod: np.ndarray,
               bao_details: dict,
               outdir: str):
    """
    Génère et sauvegarde les figures :
      - Diagramme de Hubble SN Ia (µ_obs vs µ_mod),
      - BAO D_M/r_d (observé vs modèle),
      - BAO D_H/r_d (observé vs modèle),
      - H(z) = E(z) H0.

    Les fichiers PNG sont enregistrés dans le répertoire 'outdir'.
    """

    os.makedirs(outdir, exist_ok=True)

    # 1) Diagramme de Hubble SN Ia
    plt.figure()
    plt.scatter(z_sn, mu_obs, s=5, alpha=0.5, label="Données SN Ia (Pantheon+SH0ES)")
    plt.plot(z_sn, mu_mod, linewidth=1.0, label="Modèle GeNeSyS (fond prédictif)")
    plt.xlabel("Redshift z")
    plt.ylabel("Distance modulaire µ(z) [mag]")
    plt.title("Diagramme de Hubble SN Ia : données vs modèle GeNeSyS")
    plt.legend()
    plt.grid(True, alpha=0.3)
    sn_fig_path = os.path.join(outdir, "sn_hubble_diagram.png")
    plt.savefig(sn_fig_path, dpi=150)
    plt.close()

    # 2) BAO : D_M / r_d
    z_bao = np.array(bao_details["z_BAO"])
    DM_over_rd_obs = np.array(bao_details["DM_over_rd_obs"])
    DM_over_rd_mod = np.array(bao_details["DM_over_rd_mod"])

    plt.figure()
    plt.errorbar(
        z_bao,
        DM_over_rd_obs,
        fmt="o",
        label="Données BAO BOSS DR12 : D_M / r_d",
    )
    plt.plot(
        z_bao,
        DM_over_rd_mod,
        "-",
        label="Modèle GeNeSyS : D_M / r_d",
    )
    plt.xlabel("Redshift z")
    plt.ylabel("Distance comobile transverse D_M(z) / r_d")
    plt.title("BAO BOSS DR12 : D_M(z) / r_d (données vs GeNeSyS)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    bao_dm_fig_path = os.path.join(outdir, "bao_DM_over_rd.png")
    plt.savefig(bao_dm_fig_path, dpi=150)
    plt.close()

    # 3) BAO : D_H / r_d
    DH_over_rd_obs = np.array(bao_details["DH_over_rd_obs"])
    DH_over_rd_mod = np.array(bao_details["DH_over_rd_mod"])

    plt.figure()
    plt.errorbar(
        z_bao,
        DH_over_rd_obs,
        fmt="o",
        label="Données BAO BOSS DR12 : D_H / r_d",
    )
    plt.plot(
        z_bao,
        DH_over_rd_mod,
        "-",
        label="Modèle GeNeSyS : D_H / r_d",
    )
    plt.xlabel("Redshift z")
    plt.ylabel("Distance radiale D_H(z) / r_d")
    plt.title("BAO BOSS DR12 : D_H(z) / r_d (données vs GeNeSyS)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    bao_dh_fig_path = os.path.join(outdir, "bao_DH_over_rd.png")
    plt.savefig(bao_dh_fig_path, dpi=150)
    plt.close()

    # 4) H(z) = E(z) H0
    H_z = E_grid * H0
    plt.figure()
    plt.plot(z_grid, H_z, label="Modèle GeNeSyS : H(z)")
    plt.xlabel("Redshift z")
    plt.ylabel("Taux d'expansion H(z) [km/s/Mpc]")
    plt.title("Historique du taux d'expansion H(z) dans GeNeSyS")
    plt.grid(True, alpha=0.3)
    plt.legend()
    H_fig_path = os.path.join(outdir, "H_of_z.png")
    plt.savefig(H_fig_path, dpi=150)
    plt.close()

    return {
        "sn_hubble_diagram": sn_fig_path,
        "bao_DM_over_rd": bao_dm_fig_path,
        "bao_DH_over_rd": bao_dh_fig_path,
        "H_of_z": H_fig_path,
    }


# -------------------------------------------------------------------
# 5. Programme principal
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline SN Ia (Pantheon+SH0ES) + BAO (BOSS DR12) pour le fond GeNeSyS "
            "tabulé dans _GeNeSyS_v10_best_Ez.csv."
        )
    )
    parser.add_argument(
        "--best-ez",
        default="_GeNeSyS_v10_best_Ez.csv",
        help="Fichier CSV contenant le fond GeNeSyS (z, E(z)).",
    )
    parser.add_argument(
        "--H0",
        type=float,
        default=70.0,
        help="Paramètre de Hubble H0 en km/s/Mpc (doit correspondre au fond tabulé).",
    )
    parser.add_argument(
        "--sn-table",
        default="Pantheon+SH0ES.dat",
        help="Table Pantheon+SH0ES (.dat) avec zHD et MU_SH0ES.",
    )
    parser.add_argument(
        "--sn-cov",
        default="Pantheon+SH0ES_STAT+SYS.cov",
        help="Matrice de covariance STAT+SYS pour Pantheon+SH0ES.",
    )
    parser.add_argument(
        "--bao-summary",
        default="DR12_fid_DMrd_DHrd_summary.csv",
        help="Fichier résumé BAO DR12 (z, DM/rd, DH/rd).",
    )
    parser.add_argument(
        "--bao-cov",
        default="DR12_cov6x6_DMrd_DHrd_from_consensus.csv",
        help="Covariance 6x6 pour BAO DR12 (DM/rd, DH/rd).",
    )
    parser.add_argument(
        "--rd",
        type=float,
        default=147.0,
        help="Règle acoustique effective r_d en Mpc (GeNeSyS v10).",
    )
    parser.add_argument(
        "--outdir",
        default="results",
        help="Répertoire de sortie pour les figures et le JSON.",
    )

    args = parser.parse_args()

    # 1) Fond GeNeSyS : z, E(z), D_M(z)
    z_grid, E_grid, H0 = load_genesys_best_Ez(args.best_ez, args.H0)
    DM_grid = compute_DM_from_Ez(z_grid, E_grid, H0)

    # 2) Supernovae Pantheon+SH0ES
    z_sn, mu_obs, icov_sn = load_sn_pantheon_sh0es(args.sn_table, args.sn_cov)
    mu_mod = compute_mu_model(z_sn, z_grid, DM_grid)
    chi2_SN = chi2_sn_full(mu_obs, mu_mod, icov_sn)
    print(
        f"Pantheon+SH0ES chargé : {len(z_sn)} supernovae, "
        f"chi2_SN = {chi2_SN:.3f}"
    )

    # 3) BAO BOSS DR12
    z_bao, DM_over_rd_obs, DH_over_rd_obs, icov_bao = load_bao_dr12(
        args.bao_summary, args.bao_cov
    )
    chi2_BAO, bao_details = chi2_bao_dr12(
        z_bao,
        DM_over_rd_obs,
        DH_over_rd_obs,
        icov_bao,
        z_grid,
        DM_grid,
        E_grid,
        H0,
        args.rd,
    )
    print("BAO DR12 chargées.")
    print(f"z_BAO = {z_bao}")

    chi2_tot = chi2_SN + chi2_BAO

    print("\n=== Résultats GeNeSyS SN+BAO ===")
    print(f"Paramètre de Hubble H0          = {H0:.3f} km/s/Mpc")
    print(f"Règle acoustique effective r_d   = {args.rd:.3f} Mpc")
    print(f"Chi carré SN Ia (Pantheon+SH0ES) = {chi2_SN:.3f}")
    print(f"Chi carré BAO (BOSS DR12)        = {chi2_BAO:.3f}")
    print(f"Chi carré total SN+BAO           = {chi2_tot:.3f}")

    # 4) Figures
    fig_paths = make_plots(
        z_grid,
        E_grid,
        H0,
        z_sn,
        mu_obs,
        mu_mod,
        bao_details,
        args.outdir,
    )

    # 5) Sauvegarde JSON
    os.makedirs(args.outdir, exist_ok=True)
    json_path = os.path.join(args.outdir, "genesys_SN_BAO_results.json")
    summary = {
        "H0_km_per_s_per_Mpc": H0,
        "rd_Mpc": args.rd,
        "chi2_SN": chi2_SN,
        "chi2_BAO": chi2_BAO,
        "chi2_tot": chi2_tot,
        "z_BAO": bao_details["z_BAO"],
        "DM_over_rd_obs": bao_details["DM_over_rd_obs"],
        "DM_over_rd_mod": bao_details["DM_over_rd_mod"],
        "DH_over_rd_obs": bao_details["DH_over_rd_obs"],
        "DH_over_rd_mod": bao_details["DH_over_rd_mod"],
        "figures": fig_paths,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nRésumé JSON sauvegardé dans : {json_path}")
    print(f"Figure SN sauvegardée dans : {fig_paths['sn_hubble_diagram']}")
    print(f"Figure BAO (D_M/r_d) sauvegardée dans : {fig_paths['bao_DM_over_rd']}")
    print(f"Figure BAO (D_H/r_d) sauvegardée dans : {fig_paths['bao_DH_over_rd']}")
    print(f"Figure H(z) sauvegardée dans : {fig_paths['H_of_z']}")


if __name__ == "__main__":
    main()