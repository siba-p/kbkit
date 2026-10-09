"""Command-line interface: ``kbkit kbi|binary|validate``."""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from .core import binary_thermo, kbi, preferential_binding
from .io import read_rdf


def _report(res, unit="nm"):
    print(f"pair {res.pair}")
    print(f"  raw running KBI plateau   : {res.plateau_raw[0]: .5f} ± {res.plateau_raw[1]:.5f} {unit}^3")
    if res.plateau_gvdv:
        print(f"  GvdV-corrected plateau    : {res.plateau_gvdv[0]: .5f} ± {res.plateau_gvdv[1]:.5f} {unit}^3")
    print(f"  Kruger-Vlugt extrapolation: {res.kv_inf[0]: .5f} ± {res.kv_inf[1]:.5f} {unit}^3   (fit L in {res.fit_range[0]:.2f}-{res.fit_range[1]:.2f})")
    print(f"  recommended               : {res.best[0]: .5f} ± {res.best[1]:.5f} {unit}^3  ({res.best[0] * 602.214:.2f} cm^3/mol)")


def _plot(res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    ax[0].plot(res.r, res.G_running, color="#9aa3b2", lw=1.5, label="raw")
    if res.G_gvdv is not None:
        ax[0].plot(res.r, res.G_gvdv, color="#2a78d6", lw=2, label="GvdV corrected")
    ax[0].axhline(res.best[0], color="#eb6834", lw=1, ls="--", label="recommended")
    ax[0].set_xlabel("R")
    ax[0].set_ylabel("running G(R)")
    ax[0].legend(frameon=False)
    ax[1].plot(res.L, res.L * res.G_V, color="#2a78d6", lw=2)
    lo, hi = res.fit_range
    Lf = np.linspace(lo, hi, 10)
    ax[1].plot(Lf, res.kv_inf[0] * Lf + (np.interp(lo, res.L, res.L * res.G_V) - res.kv_inf[0] * lo), color="#eb6834", ls="--")
    ax[1].set_xlabel("L")
    ax[1].set_ylabel("L · G_V(L)   (slope = G∞)")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    print(f"wrote {path}")


def cmd_kbi(a):
    r, g = read_rdf(a.rdf, a.column)
    res = kbi(r, g, a.pair or a.rdf, a.nj, a.volume, a.same, tuple(a.plateau) if a.plateau else None, tuple(a.fit) if a.fit else None)
    _report(res, a.length_unit)
    if a.output:
        with open(a.output + "_kbi.csv", "w") as fh:
            fh.write("r,g,g_corrected,G_running,G_gvdv\n")
            for i in range(len(res.r)):
                gc = res.g_corr[i] if res.g_corr is not None else np.nan
                gg = res.G_gvdv[i] if res.G_gvdv is not None else np.nan
                fh.write(f"{res.r[i]:.5f},{res.g[i]:.6f},{gc:.6f},{res.G_running[i]:.6f},{gg:.6f}\n")
        print(f"wrote {a.output}_kbi.csv")
        _plot(res, a.output + "_kbi.png")


def cmd_binary(a):
    rho_w, rho_c = a.nw / a.volume, a.nc / a.volume
    Gs = {}
    for key, path, nj, same in (("ww", a.ww, a.nw, True), ("cc", a.cc, a.nc, True), ("cw", a.cw, a.nw, False)):
        r, g = read_rdf(path, a.column)
        res = kbi(r, g, key, nj, a.volume, same)
        _report(res, "nm")
        Gs[key] = res.best[0]
    t = binary_thermo(Gs["ww"], Gs["cc"], Gs["cw"], rho_w, rho_c, a.temperature)
    print(f"\nBinary mixture  x_c = {t['x_c']:.4f}")
    print(f"  stability  dln a_c/dln x_c  = {t['dlna_dlnx']:.4f}   (1 = ideal, >1 favourable mixing, <1 aggregation)")
    print(f"  dln a_c/dln rho_c          = {t['dlna_dlnrho']:.4f}")
    print(f"  Delta_cw = Gcc+Gww-2Gcw    = {t['Delta_cw']:.5f} nm^3")
    print(f"  partial molar volumes      : V_c = {t['V_c']:.2f}, V_w = {t['V_w']:.2f} cm^3/mol")
    print(f"  isothermal compressibility : {t['kappa_T_per_GPa']:.4f} 1/GPa")
    print(f"  excess coordination        : N_cc {t['N_cc']:.3f}, N_cw {t['N_cw']:.3f}, N_ww {t['N_ww']:.3f}")


def cmd_pb(a):
    def best(path, nj, same):
        r, g = read_rdf(path, a.column)
        return kbi(r, g, path, nj, a.volume, same).best[0]
    G_pc = best(a.pc, a.nc, False)
    G_pw = best(a.pw, a.nw, False)
    nu = preferential_binding(G_pc, G_pw, a.nc / a.volume)
    print(f"G_pc = {G_pc:.4f}, G_pw = {G_pw:.4f} nm^3  ->  preferential binding nu_pc = {nu:.3f}"
          f"  ({'cosolvent accumulates' if nu > 0 else 'cosolvent is excluded'})")


def cmd_validate(a):
    from .validation import hard_sphere_G, hard_spheres, ideal_gas, rdf_from_frames
    print("Hard-sphere fluid (Metropolis MC, N = 512) vs Carnahan-Starling; ideal gas (exact G = 0).")
    print(f"{'system':22s}{'exact':>10s}{'raw':>10s}{'GvdV':>10s}{'KV':>10s}")
    for eta in a.eta:
        frames, box = hard_spheres(8, eta, a.sweeps, a.sweeps // 4, 2, delta=0.15 if eta < 0.35 else 0.1)
        r, g = rdf_from_frames(frames, box)
        res = kbi(r, g, "HS", 512, box ** 3, True)
        print(f"{'hard spheres eta=' + str(eta):22s}{hard_sphere_G(eta):10.4f}{res.plateau_raw[0]:10.4f}{res.plateau_gvdv[0]:10.4f}{res.kv_inf[0]:10.4f}")
    fr, b = ideal_gas(400, 8.0, 150)
    r, g = rdf_from_frames(fr, b, dr=0.04)
    res = kbi(r, g, "ideal", 400, b ** 3, True)
    print(f"{'ideal gas':22s}{0.0:10.4f}{res.plateau_raw[0]:10.4f}{res.plateau_gvdv[0]:10.4f}{res.kv_inf[0]:10.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kbkit", description="Kirkwood-Buff integrals with finite-size corrections.")
    ap.add_argument("--version", action="version", version=f"kbkit {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("kbi", help="KB integral of one RDF")
    p.add_argument("rdf")
    p.add_argument("--nj", type=float, help="number of j particles in the box (enables GvdV correction)")
    p.add_argument("--volume", type=float, help="box volume (length unit^3)")
    p.add_argument("--same", action="store_true", help="i and j are the same species")
    p.add_argument("--pair", help="label")
    p.add_argument("--plateau", type=float, nargs=2, metavar=("R1", "R2"))
    p.add_argument("--fit", type=float, nargs=2, metavar=("L1", "L2"), help="L range for the KV extrapolation")
    p.add_argument("--column", type=int, default=1)
    p.add_argument("--length-unit", default="nm")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_kbi)

    p = sub.add_parser("binary", help="KB thermodynamics of a binary mixture from three RDFs")
    p.add_argument("--ww", required=True, help="solvent-solvent RDF")
    p.add_argument("--cc", required=True, help="cosolvent-cosolvent RDF")
    p.add_argument("--cw", required=True, help="cosolvent-solvent RDF")
    p.add_argument("--nw", type=float, required=True)
    p.add_argument("--nc", type=float, required=True)
    p.add_argument("--volume", type=float, required=True, help="box volume in nm^3")
    p.add_argument("-T", "--temperature", type=float, default=298.15)
    p.add_argument("--column", type=int, default=1)
    p.set_defaults(func=cmd_binary)

    p = sub.add_parser("pb", help="preferential binding of a cosolvent to a solute")
    p.add_argument("--pc", required=True, help="solute-cosolvent RDF")
    p.add_argument("--pw", required=True, help="solute-water RDF")
    p.add_argument("--nc", type=float, required=True)
    p.add_argument("--nw", type=float, required=True)
    p.add_argument("--volume", type=float, required=True)
    p.add_argument("--column", type=int, default=1)
    p.set_defaults(func=cmd_pb)

    p = sub.add_parser("validate", help="reproduce the hard-sphere / ideal-gas validation")
    p.add_argument("--eta", type=float, nargs="+", default=[0.2, 0.3])
    p.add_argument("--sweeps", type=int, default=300)
    p.set_defaults(func=cmd_validate)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":  # pragma: no cover
    main()
