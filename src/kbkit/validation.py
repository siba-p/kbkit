"""Reference systems with exactly known Kirkwood-Buff integrals, for validating any KBI workflow."""
from __future__ import annotations

import numpy as np


def carnahan_starling_S0(eta: float) -> float:
    """S(0) = rho k_B T kappa_T of the hard-sphere fluid at packing fraction eta (Carnahan-Starling)."""
    return (1 - eta) ** 4 / (1 + 4 * eta + 4 * eta ** 2 - 4 * eta ** 3 + eta ** 4)


def hard_sphere_G(eta: float, sigma: float = 1.0) -> float:
    """Exact (Carnahan-Starling) KB integral of the hard-sphere fluid: G = (S(0) - 1) / rho."""
    rho = 6 * eta / (np.pi * sigma ** 3)
    return (carnahan_starling_S0(eta) - 1) / rho


def rdf_from_frames(frames, box: float, dr: float = 0.02, rmax: float | None = None):
    """Same-species g(r) normalised with rho = N/V (as GROMACS does), from cubic periodic frames."""
    rmax = rmax or box / 2
    edges = np.arange(0, rmax + dr / 2, dr)
    hist = np.zeros(len(edges) - 1)
    n = frames[0].shape[0]
    iu = np.triu_indices(n, 1)
    for X in frames:
        d = X[:, None, :] - X[None, :, :]
        d -= box * np.round(d / box)
        r = np.sqrt((d ** 2).sum(-1))[iu]
        hist += np.histogram(r, bins=edges)[0]
    rho = n / box ** 3
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = 2 * hist / (len(frames) * n * rho * shell)
    return 0.5 * (edges[1:] + edges[:-1]), g


def hard_spheres(n_side: int = 8, eta: float = 0.3, sweeps: int = 400, equil: int = 100, every: int = 2,
                 delta: float = 0.15, seed: int = 1):
    """Metropolis Monte Carlo of hard spheres (sigma = 1) in a cubic box.

    Returns (frames, box) with N = n_side**3 particles started from a simple-cubic lattice.
    """
    rng = np.random.default_rng(seed)
    n = n_side ** 3
    box = (n * np.pi / (6 * eta)) ** (1 / 3)
    a = box / n_side
    grid = np.arange(n_side) * a + a / 2
    X = np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T.copy()
    frames = []
    for sweep in range(sweeps):
        order = rng.permutation(n)
        moves = rng.uniform(-delta, delta, size=(n, 3))
        for i in order:
            trial = (X[i] + moves[i]) % box
            d = X - trial
            d -= box * np.round(d / box)
            r2 = (d ** 2).sum(1)
            r2[i] = 10.0
            if r2.min() >= 1.0:
                X[i] = trial
        if sweep >= equil and (sweep - equil) % every == 0:
            frames.append(X.copy())
    return frames, box


def ideal_gas(n: int = 500, box: float = 8.0, frames: int = 200, seed: int = 2):
    """Uncorrelated particles at fixed N (exact open-system G = 0); exposes the closed-box bias."""
    rng = np.random.default_rng(seed)
    return [rng.uniform(0, box, size=(n, 3)) for _ in range(frames)], box
