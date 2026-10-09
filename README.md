# KBkit

**Kirkwood–Buff integrals from simulation RDFs, done right: finite-size corrections and KB thermodynamics in one tool.**

[![tests](https://github.com/siba-p/kbkit/actions/workflows/ci.yml/badge.svg)](https://github.com/siba-p/kbkit/actions/workflows/ci.yml)
[![web app](https://img.shields.io/badge/web%20app-siba--p.github.io%2Fkbkit-3b50e0)](https://siba-p.github.io/kbkit/)
![python](https://img.shields.io/badge/python-3.9%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

Kirkwood–Buff (KB) theory links radial distribution functions to measurable thermodynamics: preferential binding, activity coefficients, partial molar volumes and compressibility. The difficulty is that the naive running integral of an RDF from a closed, finite simulation box does not converge to the right answer. KBkit implements the two standard remedies side by side, so you can see whether your number is trustworthy:

| estimator | what it does |
|---|---|
| **raw** | G(R) = 4π∫₀ᴿ [g(r) − 1] r² dr. Biased by the closed box. |
| **GvdV** | Ganguly–van der Vegt correction of g(r) for the depletion of a finite box (*JCTC* **9**, 1347, 2013), then the running integral. |
| **KV** | Krüger–Vlugt finite-volume integral G_V(L) with weight 1 − 3x/2 + x³/2, extrapolated to the thermodynamic limit via L·G_V(L) = G∞L + F (*J. Phys. Chem. Lett.* **4**, 235, 2013). |

It then turns the G's into thermodynamics: the stability term ∂ln a/∂ln x, partial molar volumes, isothermal compressibility, excess coordination numbers and preferential binding coefficients.

Use it **in the browser** at **https://siba-p.github.io/kbkit/**, or install the package below.

## Install

```bash
pip install "git+https://github.com/siba-p/kbkit.git#egg=kbkit[plot]"
```

## Quick start

```bash
# one RDF from `gmx rdf` (r in nm); N_j and box volume enable the finite-size corrections
kbkit kbi rdf_OW_OW.xvg --nj 2180 --volume 66.2 --same -o water

# binary mixture (e.g. urea c in water w): three RDFs -> full KB thermodynamics
kbkit binary --ww rdf_ww.xvg --cc rdf_cc.xvg --cw rdf_cw.xvg --nw 1800 --nc 200 --volume 65.4 -T 298

# preferential binding of a cosolvent to a solute (protein, nanoparticle, polymer...)
kbkit pb --pc rdf_np_cosolvent.xvg --pw rdf_np_water.xvg --nc 300 --nw 6000 --volume 220

# reproduce the validation below on your machine (about 30 s)
kbkit validate --eta 0.2 0.3 0.4
```

Python API:

```python
import kbkit as kb
r, g = kb.read_rdf("rdf_cc.xvg")
res = kb.kbi(r, g, "cc", n_j=200, volume=65.4, same_species=True)
print(res.plateau_raw, res.plateau_gvdv, res.kv_inf, res.best)
thermo = kb.binary_thermo(G_ww, G_cc, G_cw, rho_w, rho_c, T=298.15)
```

## Validation against exact results

The hard-sphere fluid has an exactly known KB integral, G = (S(0) − 1)/ρ with the Carnahan–Starling S(0). The ideal gas has G = 0, but a closed box makes the raw integral look attractive. Metropolis MC, N = 512, cubic periodic box (`kbkit validate`):

| system | exact | raw | GvdV | KV |
|---|---|---|---|---|
| hard spheres, η = 0.2 (250 frames) | −2.062 | −2.260 | −2.079 | **−2.055** |
| hard spheres, η = 0.3 | −1.575 | −1.613 | −1.554 | **−1.564** |
| hard spheres, η = 0.4 | −1.253 | −1.300 | −1.289 | **−1.290** |
| ideal gas | 0 | −0.456 | −0.044 | **−0.027** |

Both corrections remove most of the closed-box bias. The remaining differences are sampling noise that shrinks with more frames. Note that the "±" reported for KV is the uncertainty of the linear fit only; for a sampling error, split your trajectory into blocks and compare.

The thermodynamic layer is checked against limits: an ideal mixture gives ∂ln a/∂ln x = 1, and pure water with G_ww = −0.028 nm³ gives κ_T ≈ 0.45 GPa⁻¹.

## Formulas (binary mixture, c = cosolvent, w = solvent)

- Δ = G_cc + G_ww − 2G_cw; η = ρ_w + ρ_c + ρ_wρ_cΔ; ζ = 1 + ρ_wG_ww + ρ_cG_cc + ρ_wρ_c(G_wwG_cc − G_cw²)
- ∂ln a_c/∂ln x_c = 1/(1 + ρ x_w Δ); ∂ln a_c/∂ln ρ_c = 1/(1 + ρ_c(G_cc − G_cw))
- V̄_c = (1 + ρ_w(G_ww − G_cw))/η; V̄_w = (1 + ρ_c(G_cc − G_cw))/η; κ_T = ζ/(η k_BT)
- Preferential binding of c to a solute p: ν_pc = ρ_c(G_pc − G_pw)

## Citation

> S. Panigrahy, *KBkit: Kirkwood–Buff integrals with finite-size corrections*, 2026. https://github.com/siba-p/kbkit

This tool grew out of the KB analysis of crowded amino acid solutions and DNA-functionalised gold nanoparticles in my PhD work (Panigrahy & Nayar, *J. Chem. Phys.* **160**, 014901, 2024).

## License

MIT © Sibasankar Panigrahy
