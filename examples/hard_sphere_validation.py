"""Compare raw, GvdV and Kruger-Vlugt KB integrals with the exact hard-sphere result.

Run:  python examples/hard_sphere_validation.py
"""
import kbkit as kb
from kbkit.validation import hard_sphere_G, hard_spheres, rdf_from_frames

frames, box = hard_spheres(n_side=8, eta=0.3, sweeps=400, equil=100)
r, g = rdf_from_frames(frames, box)
res = kb.kbi(r, g, "HS", n_j=512, volume=box ** 3, same_species=True)
print(f"exact (Carnahan-Starling) : {hard_sphere_G(0.3):.4f}")
print(f"raw running integral      : {res.plateau_raw[0]:.4f}")
print(f"Ganguly-van der Vegt      : {res.plateau_gvdv[0]:.4f}")
print(f"Kruger-Vlugt extrapolation: {res.kv_inf[0]:.4f}")
