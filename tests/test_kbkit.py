import numpy as np
import pytest

import kbkit as kb
from kbkit.validation import carnahan_starling_S0, hard_sphere_G, hard_spheres, ideal_gas, rdf_from_frames


@pytest.fixture(scope="module")
def hs():
    frames, box = hard_spheres(n_side=8, eta=0.3, sweeps=240, equil=60, every=2, seed=1)
    r, g = rdf_from_frames(frames, box)
    return r, g, 512, box ** 3


def test_carnahan_starling_limits():
    assert carnahan_starling_S0(0.0) == pytest.approx(1.0)
    assert hard_sphere_G(1e-6) == pytest.approx(-4 / 3 * np.pi, rel=1e-3)   # low density: excluded volume


def test_hard_sphere_kbi_matches_carnahan_starling(hs):
    r, g, n, V = hs
    res = kb.kbi(r, g, "HS", n_j=n, volume=V, same_species=True)
    exact = hard_sphere_G(0.3)
    assert res.kv_inf[0] == pytest.approx(exact, rel=0.03)
    assert res.plateau_gvdv[0] == pytest.approx(exact, rel=0.06)
    # the correction must move the raw estimate towards the exact value
    assert abs(res.kv_inf[0] - exact) < abs(res.plateau_raw[0] - exact)


def test_ideal_gas_closed_box_bias_removed():
    frames, box = ideal_gas(400, 8.0, 120)
    r, g = rdf_from_frames(frames, box, dr=0.04)
    res = kb.kbi(r, g, "ideal", n_j=400, volume=box ** 3, same_species=True)
    assert res.plateau_raw[0] < -0.3            # closed-system artefact
    assert abs(res.plateau_gvdv[0]) < 0.1
    assert abs(res.kv_inf[0]) < 0.1


def test_step_function_rdf_exact():
    r = np.linspace(0, 3, 3001)
    g = (r >= 1.0).astype(float)
    G = kb.running_kbi(r, g)
    assert G[-1] == pytest.approx(-4 / 3 * np.pi, rel=2e-3)


def test_binary_thermo_ideal_and_water():
    t = kb.binary_thermo(0.0, 0.0, 0.0, 30.0, 3.0)
    assert t["dlna_dlnx"] == pytest.approx(1.0)
    assert t["V_c"] == pytest.approx(t["V_w"]) == pytest.approx(602.214 / 33.0, rel=1e-6)
    # pure-water limit: G_ww = -0.0281 nm^3 at rho = 33.4 nm^-3 gives kappa_T ~ 0.45 / GPa
    w = kb.binary_thermo(-0.0281, 0.0, 0.0, 33.4, 1e-9, T=298.15)
    assert w["kappa_T_per_GPa"] == pytest.approx(0.45, rel=0.05)


def test_preferential_binding_sign():
    assert kb.preferential_binding(0.5, 0.1, 1.0) > 0
    assert kb.preferential_binding(-0.1, 0.1, 1.0) < 0


def test_read_rdf(tmp_path):
    f = tmp_path / "rdf.xvg"
    f.write_text('# gmx rdf\n@ title "RDF"\n@TYPE xy\n0.000 0.0\n0.002 0.0\n0.004 0.5\n')
    r, g = kb.read_rdf(f)
    assert len(r) == 3 and g[-1] == 0.5


def test_cli_kbi(tmp_path, hs, capsys):
    r, g, n, V = hs
    f = tmp_path / "hs.xvg"
    f.write_text("\n".join(f"{a:.4f} {b:.6f}" for a, b in zip(r, g)))
    from kbkit.cli import main
    main(["kbi", str(f), "--nj", str(n), "--volume", str(V), "--same"])
    assert "Kruger-Vlugt" in capsys.readouterr().out
