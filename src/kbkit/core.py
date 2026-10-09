"""Kirkwood-Buff integrals from simulation RDFs, with finite-size corrections.

Three estimators of G_ij (same length unit cubed as r, e.g. nm^3):

* ``running``   : G(R) = 4 pi int_0^R [g(r) - 1] r^2 dr on the raw RDF (closed-system bias).
* ``gvdv``      : the same on the Ganguly-van der Vegt corrected RDF (JCTC 9, 1347, 2013), which removes the
                  depletion of a closed box: g_c = g N_j (1 - V_r/V) / [N_j (1 - V_r/V) - dN_ij(r) - delta_ij].
* ``kv``        : finite-volume integral of Kruger, Vlugt et al. (JPCL 4, 235, 2013),
                  G_V(L) = int_0^L [g - 1] 4 pi r^2 (1 - 3x/2 + x^3/2) dr,  x = r/L,
                  which scales as G_inf + F/L; G_inf is the slope of L*G_V(L) versus L.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NA_NM3_TO_CM3_PER_MOL = 602.214076          # 1 nm^3 per molecule = 602.2 cm^3/mol
KB = {"kJ/mol": 0.0083144626, "kcal/mol": 0.0019872043}
PRESSURE_UNIT_GPA = 1.66053907e-3           # (kJ/mol)/nm^3 in GPa


_trapz = getattr(np, "trapezoid", None) or np.trapz


def _cumtrapz(y, x):
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def running_kbi(r, g):
    r, g = np.asarray(r, float), np.asarray(g, float)
    return _cumtrapz(4 * np.pi * (g - 1.0) * r ** 2, r)


def gvdv_correct(r, g, n_j: float, volume: float, same_species: bool):
    """Ganguly-van der Vegt finite-size correction of g_ij(r)."""
    r, g = np.asarray(r, float), np.asarray(g, float)
    rho_j = n_j / volume
    dN = rho_j * running_kbi(r, g)                    # excess number of j within r
    vr = 4.0 / 3.0 * np.pi * r ** 3
    bulk = n_j * (1.0 - vr / volume)
    denom = bulk - dN - (1.0 if same_species else 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        gc = np.where(denom > 0, g * bulk / denom, np.nan)
    return gc


def kv_integral(r, g, L):
    """Finite-volume KB integral G_V(L) with the Kruger-Vlugt weight."""
    r, g = np.asarray(r, float), np.asarray(g, float)
    m = r <= L
    x = r[m] / L
    w = 1 - 1.5 * x + 0.5 * x ** 3
    return float(_trapz(4 * np.pi * (g[m] - 1.0) * r[m] ** 2 * w, r[m])) if m.sum() > 1 else 0.0


@dataclass
class KBIResult:
    pair: str
    r: np.ndarray
    g: np.ndarray
    g_corr: np.ndarray | None
    G_running: np.ndarray
    G_gvdv: np.ndarray | None
    L: np.ndarray
    G_V: np.ndarray
    plateau_raw: tuple[float, float]
    plateau_gvdv: tuple[float, float] | None
    kv_inf: tuple[float, float]
    fit_range: tuple[float, float]

    @property
    def best(self) -> tuple[float, float]:
        """Recommended estimate: KV extrapolation (or GvdV plateau when the fit is unstable)."""
        if self.plateau_gvdv and np.isfinite(self.kv_inf[1]) and self.kv_inf[1] > 3 * max(self.plateau_gvdv[1], 1e-12):
            return self.plateau_gvdv
        return self.kv_inf


def _plateau(r, G, lo, hi):
    m = (r >= lo) & (r <= hi) & np.isfinite(G)
    if m.sum() < 3:
        return (float("nan"), float("nan"))
    return (float(G[m].mean()), float(G[m].std(ddof=1)))


def kbi(r, g, pair: str = "ij", n_j: float | None = None, volume: float | None = None, same_species: bool = False,
        plateau: tuple[float, float] | None = None, fit: tuple[float, float] | None = None) -> KBIResult:
    """Compute all three KBI estimators for one RDF.

    ``plateau`` and ``fit`` are (start, end) ranges in r units. Defaults: plateau over the last 30% of the RDF;
    KV fit for L between 40% and 90% of r_max.
    """
    r, g = np.asarray(r, float), np.asarray(g, float)
    if r[0] > 0:
        r = np.concatenate([[0.0], r])
        g = np.concatenate([[g[0] if g[0] < 1e-6 else 0.0], g])
    rmax = r[-1]
    plateau = plateau or (0.7 * rmax, rmax)
    fit = fit or (0.4 * rmax, 0.9 * rmax)
    G_run = running_kbi(r, g)
    g_corr = G_gv = None
    if n_j is not None and volume is not None:
        g_corr = gvdv_correct(r, g, n_j, volume, same_species)
        G_gv = running_kbi(r, np.nan_to_num(g_corr, nan=1.0))
    L = np.linspace(rmax * 0.1, rmax, 120)
    gv_src = np.nan_to_num(g_corr, nan=1.0) if g_corr is not None else g
    G_V = np.array([kv_integral(r, gv_src, l) for l in L])
    m = (L >= fit[0]) & (L <= fit[1])
    A = np.vstack([L[m], np.ones(m.sum())]).T
    coef, res, *_ = np.linalg.lstsq(A, L[m] * G_V[m], rcond=None)
    dof = max(1, m.sum() - 2)
    resid = L[m] * G_V[m] - A @ coef
    cov = np.linalg.inv(A.T @ A) * (resid @ resid) / dof
    kv = (float(coef[0]), float(np.sqrt(cov[0, 0])))
    return KBIResult(pair, r, g, g_corr, G_run, G_gv, L, G_V, _plateau(r, G_run, *plateau),
                     _plateau(r, G_gv, *plateau) if G_gv is not None else None, kv, fit)


def binary_thermo(G_ww: float, G_cc: float, G_cw: float, rho_w: float, rho_c: float, T: float = 298.15,
                  unit: str = "kJ/mol") -> dict:
    """Kirkwood-Buff thermodynamics of a binary mixture (w = solvent, c = cosolvent), lengths in nm.

    Returns the KB stability term, activity derivatives, partial molar volumes (cm^3/mol), isothermal
    compressibility (1/GPa) and excess coordination numbers.
    """
    rho = rho_w + rho_c
    x_c, x_w = rho_c / rho, rho_w / rho
    delta = G_cc + G_ww - 2 * G_cw
    eta = rho_w + rho_c + rho_w * rho_c * delta
    zeta = 1 + rho_w * G_ww + rho_c * G_cc + rho_w * rho_c * (G_ww * G_cc - G_cw ** 2)
    kt = KB[unit] * T
    return {
        "x_c": x_c,
        "Delta_cw": delta,
        "dlna_dlnx": 1.0 / (1.0 + rho * x_w * delta),
        "dlna_dlnrho": 1.0 / (1.0 + rho_c * (G_cc - G_cw)),
        "V_c": (1 + rho_w * (G_ww - G_cw)) / eta * NA_NM3_TO_CM3_PER_MOL,
        "V_w": (1 + rho_c * (G_cc - G_cw)) / eta * NA_NM3_TO_CM3_PER_MOL,
        "kappa_T_per_GPa": zeta / (eta * kt) / PRESSURE_UNIT_GPA,
        "N_cc": rho_c * G_cc, "N_cw": rho_w * G_cw, "N_wc": rho_c * G_cw, "N_ww": rho_w * G_ww,
    }


def preferential_binding(G_pc: float, G_pw: float, rho_c: float) -> float:
    """Preferential binding coefficient of cosolvent c to solute p: nu = rho_c (G_pc - G_pw)."""
    return rho_c * (G_pc - G_pw)


def one_component_structure_factor(G: float, rho: float) -> float:
    """S(0) = 1 + rho G = rho k_B T kappa_T for a one-component fluid."""
    return 1.0 + rho * G
