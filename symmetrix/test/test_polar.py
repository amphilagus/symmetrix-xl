"""Oracle tests for the mixed native/PyTorch POLAR calculator.

The reduced model has the same one-interaction, scalar-output nonlinear
backbone contract as MACE-POLAR-1-S. It is a real PolarMACE, including two
field updates and both long-range energy terms; no model is downloaded.
"""

import copy
import os
from pathlib import Path

import numpy as np
import pytest


def _polar_dependencies():
    torch = pytest.importorskip("torch")
    pytest.importorskip("graph_longrange")
    extensions = pytest.importorskip("mace.modules.extensions")
    if not hasattr(extensions, "PolarMACE"):
        pytest.skip("requires mace-torch with PolarMACE (pinned runtime: 0.3.16)")
    return torch, extensions.PolarMACE


@pytest.fixture(scope="module")
def reduced_polar_model():
    torch, polar_type = _polar_dependencies()
    from e3nn import o3
    from mace.modules.blocks import (
        LinearReadoutBlock,
        RealAgnosticResidualNonLinearInteractionBlock,
    )

    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(51037)
            model = (
                polar_type(
                    r_max=4.5,
                    num_bessel=4,
                    num_polynomial_cutoff=5,
                    max_ell=2,
                    interaction_cls=RealAgnosticResidualNonLinearInteractionBlock,
                    interaction_cls_first=RealAgnosticResidualNonLinearInteractionBlock,
                    num_interactions=1,
                    num_elements=3,
                    hidden_irreps=o3.Irreps("4x0e"),
                    edge_irreps=o3.Irreps("4x0e+4x1o+4x2e"),
                    MLP_irreps=o3.Irreps("4x0e"),
                    readout_cls=LinearReadoutBlock,
                    atomic_energies=np.array([[0.2, -0.4, 0.1]]),
                    atomic_inter_scale=[1.3],
                    atomic_inter_shift=[-0.02],
                    avg_num_neighbors=4.0,
                    atomic_numbers=[1, 3, 8],
                    correlation=3,
                    gate=torch.nn.functional.silu,
                    radial_MLP=[8, 8],
                    radial_type="bessel",
                    kspace_cutoff_factor=1.0,
                    atomic_multipoles_max_l=1,
                    atomic_multipoles_smearing_width=1.5,
                    field_feature_max_l=1,
                    field_feature_widths=[1.5],
                    field_feature_norms=[1.0, 1.0],
                    num_recursion_steps=2,
                    field_si=False,
                    include_electrostatic_self_interaction=True,
                    add_local_electron_energy=True,
                    return_electrostatic_potentials=False,
                    heads=["Default"],
                    field_norm_factor=1.0,
                    fixedpoint_update_config={
                        "type": "AgnosticEmbeddedOneBodyVariableUpdate",
                        "potential_embedding_cls": "AgnosticChargeBiasedLinearPotentialEmbedding",
                        "nonlinearity_cls": "MLPNonLinearity",
                    },
                    field_readout_config={"type": "OneBodyMLPFieldReadout"},
                )
                .double()
                .eval()
            )
    finally:
        torch.set_default_dtype(old_dtype)
    return model


@pytest.fixture
def lithium_water():
    from ase import Atoms

    # Asymmetric, nonplanar geometry exercises vector components and PBC edges.
    oxygens = np.array(
        [[2.4, 2.5, 4.4], [6.4, 2.6, 4.6], [4.5, 6.4, 4.3], [4.4, 4.5, 6.7]]
    )
    offsets = np.array([[0.0, 0.0, 0.0], [0.9572, 0.02, 0.03], [-0.239, 0.927, 0.05]])
    positions = [[4.5, 4.5, 4.5]]
    numbers = [3]
    for index, oxygen in enumerate(oxygens):
        positions.extend(oxygen + np.roll(offsets, index % 3, axis=1))
        numbers.extend([8, 1, 1])
    atoms = Atoms(numbers=numbers, positions=positions, cell=[9.0, 9.0, 9.0], pbc=True)
    atoms.info.update(charge=1.0, spin=1.0, external_field=[0.0, 0.0, 0.0])
    return atoms


def _batch(model, atoms, device="cpu"):
    torch, _ = _polar_dependencies()
    from mace import data
    from mace.tools import torch_geometric, utils

    keys = data.KeySpecification(
        info_keys={
            "total_charge": "charge",
            "total_spin": "spin",
            "external_field": "external_field",
        }
    )
    config = data.config_from_atoms(
        atoms, key_specification=keys, head_name=model.heads[0]
    )
    graph = data.AtomicData.from_config(
        config,
        z_table=utils.AtomicNumberTable([int(z) for z in model.atomic_numbers]),
        cutoff=float(model.r_max),
        heads=model.heads,
    )
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[graph], batch_size=1, shuffle=False, drop_last=False
    )
    batch = next(iter(loader)).to(device).to_dict()
    return {
        key: value.to(dtype=torch.float64)
        if torch.is_tensor(value) and value.is_floating_point()
        else value
        for key, value in batch.items()
    }


def _reference(model, atoms, device="cpu"):
    from symmetrix._polar_compat import repair_polar_checkpoint

    model = copy.deepcopy(model).to(
        device=device, dtype=_polar_dependencies()[0].float64
    )
    repair_polar_checkpoint(model, pbc_handling="pbc")
    output = model(_batch(model, atoms, device), compute_force=True)
    return {
        name: value.detach().cpu().numpy()
        for name, value in output.items()
        if value is not None and hasattr(value, "detach")
    }


def _selected_device():
    import symmetrix

    native = symmetrix.load_backend()
    if not native._kokkos_is_initialized():
        native._init_kokkos()
    space = native._kokkos_default_execution_space()
    if space == "Cuda":
        torch, _ = _polar_dependencies()
        if not torch.cuda.is_available():
            pytest.skip("CUDA native backend requires a working PyTorch CUDA runtime")
        return "cuda"
    if space not in ("Serial", "OpenMP", "Threads"):
        pytest.skip(f"POLAR v1 qualifies CPU/CUDA; selected execution space is {space}")
    return "cpu"


def _assert_polar_result(actual, expected, atoms, atol=2e-8):
    for name in (
        "energy",
        "forces",
        "charges",
        "density_coefficients",
        "spin_charge_density",
        "electrostatic_energy",
        "electron_energy",
    ):
        np.testing.assert_allclose(
            np.asarray(actual[name]).reshape(-1),
            np.asarray(expected[name]).reshape(-1),
            rtol=2e-8,
            atol=atol,
            err_msg=name,
        )
    np.testing.assert_allclose(
        np.sum(actual["charges"]), atoms.info["charge"], atol=2e-10
    )
    density = np.asarray(actual["spin_charge_density"])
    np.testing.assert_allclose(
        np.sum(density[:, 0, 0] - density[:, 1, 0]), atoms.info["spin"] - 1, atol=2e-10
    )
    assert actual["free_energy"] == actual["energy"]
    diagnostics = actual["polar_diagnostics"]
    assert diagnostics["backbone"] == "native_feature_forward"
    assert diagnostics["prefix_fallback"] is False
    assert diagnostics["precision"] == "float64"
    assert diagnostics["num_interactions"] == 1
    assert diagnostics["num_recursion_steps"] == 2
    assert diagnostics["spherical_vector_permutation"] == [1, 2, 0]
    assert diagnostics["electrostatic_boundary"] == "pbc_k0_removed"
    terms = diagnostics["energy_terms"]
    np.testing.assert_allclose(
        terms["local"] + terms["electron"] + terms["electrostatic"],
        terms["total"],
        atol=atol,
    )
    targets = [
        (atoms.info["charge"] + atoms.info["spin"] - 1) / 2,
        (atoms.info["charge"] - atoms.info["spin"] + 1) / 2,
    ]
    assert len(diagnostics["normalized_spin_charge_totals"]) == 3
    for populations in diagnostics["normalized_spin_charge_totals"]:
        np.testing.assert_allclose(populations, targets, atol=2e-10)


def test_ordinary_converter_rejects_complete_polar_checkpoint(
    reduced_polar_model, tmp_path
):
    torch, _ = _polar_dependencies()
    from symmetrix.extract_mace_data import extract_mace_data

    checkpoint = tmp_path / "reduced-polar.model"
    torch.save(reduced_polar_model, checkpoint)
    with pytest.raises(ValueError, match="SymmetrixPolar"):
        extract_mace_data(checkpoint)


@pytest.mark.parametrize("bridge", ["host", "dlpack"])
def test_complete_polar_energy_force_and_density_oracle(
    reduced_polar_model, lithium_water, bridge
):
    from symmetrix.polar import SymmetrixPolar

    device = _selected_device()
    expected = _reference(reduced_polar_model, lithium_water, device)
    calculator = SymmetrixPolar(reduced_polar_model, device=device, bridge=bridge)
    calculator.calculate(lithium_water, properties=["energy", "forces"])
    _assert_polar_result(calculator.results, expected, lithium_water)


def test_candidate_never_calls_original_prefix(
    reduced_polar_model, lithium_water, monkeypatch
):
    from symmetrix.polar import SymmetrixPolar

    device = _selected_device()
    expected = _reference(reduced_polar_model, lithium_water, device)

    def forbidden(*args, **kwargs):
        raise AssertionError("candidate entered original PyTorch backbone")

    # Class patches also cover modules reconstructed or copied by the adapter.
    for module in (
        reduced_polar_model,
        *reduced_polar_model.interactions,
        *reduced_polar_model.products,
    ):
        monkeypatch.setattr(type(module), "forward", forbidden)
    calculator = SymmetrixPolar(reduced_polar_model, device=device)
    calculator.calculate(lithium_water, properties=["energy", "forces"])
    _assert_polar_result(calculator.results, expected, lithium_water)


@pytest.mark.parametrize("change", ["translate", "wrap", "perturb", "charge", "spin"])
def test_geometry_and_electronic_state_recalculation(
    reduced_polar_model, lithium_water, change
):
    from symmetrix.polar import SymmetrixPolar

    device = _selected_device()
    calculator = SymmetrixPolar(reduced_polar_model, device=device)
    lithium_water.calc = calculator
    first_energy = lithium_water.get_potential_energy()
    if change == "translate":
        lithium_water.positions += [0.137, -0.271, 0.193]
    elif change == "wrap":
        lithium_water.positions[2] += lithium_water.cell[0]
    elif change == "perturb":
        lithium_water.positions[2] += [0.013, -0.021, 0.008]
    elif change == "charge":
        lithium_water.info["charge"] = 0.0
    else:
        lithium_water.info["spin"] = 3.0
    energy = lithium_water.get_potential_energy()
    forces = lithium_water.get_forces()
    expected = _reference(reduced_polar_model, lithium_water, device)
    _assert_polar_result(calculator.results, expected, lithium_water)
    assert np.isfinite(forces).all()
    if change in ("translate", "wrap"):
        np.testing.assert_allclose(energy, first_energy, rtol=2e-8, atol=2e-8)


def test_complete_force_matches_energy_finite_difference(
    reduced_polar_model, lithium_water
):
    from symmetrix.polar import SymmetrixPolar

    calculator = SymmetrixPolar(reduced_polar_model, device=_selected_device())
    lithium_water.calc = calculator
    force = lithium_water.get_forces()[2, 1]
    step = 2e-5
    lithium_water.positions[2, 1] += step
    plus = lithium_water.get_potential_energy()
    lithium_water.positions[2, 1] -= 2 * step
    minus = lithium_water.get_potential_energy()
    np.testing.assert_allclose(
        force, -(plus - minus) / (2 * step), rtol=2e-5, atol=2e-6
    )


@pytest.mark.parametrize(
    "invalid", ["partial_pbc", "field", "spin", "charge", "stress"]
)
def test_unsupported_input_is_rejected(reduced_polar_model, lithium_water, invalid):
    from symmetrix.polar import SymmetrixPolar

    calculator = SymmetrixPolar(reduced_polar_model, device=_selected_device())
    properties = ["energy", "forces"]
    if invalid == "partial_pbc":
        lithium_water.pbc = [True, True, False]
    elif invalid == "field":
        lithium_water.info["external_field"] = [0.001, 0, 0]
    elif invalid == "spin":
        lithium_water.info["spin"] = 0.5
    elif invalid == "charge":
        lithium_water.info["charge"] = float("nan")
    else:
        properties.append("stress")
    with pytest.raises((ValueError, NotImplementedError)):
        calculator.calculate(lithium_water, properties=properties)


def test_float32_request_is_rejected(reduced_polar_model):
    from symmetrix.polar import SymmetrixPolar

    with pytest.raises((ValueError, NotImplementedError), match="float64|FP64"):
        SymmetrixPolar(reduced_polar_model, precision="float32")


def test_real_h512_s_checkpoint_oracle(lithium_water):
    """Opt in explicitly; this test never downloads the foundation checkpoint."""
    checkpoint = os.environ.get("SYMMETRIX_POLAR_MODEL")
    if not checkpoint:
        pytest.skip("set SYMMETRIX_POLAR_MODEL to the MACE-POLAR-1-S checkpoint")
    path = Path(checkpoint)
    if not path.is_file():
        pytest.fail(f"requested POLAR checkpoint does not exist: {path}")
    torch, _ = _polar_dependencies()
    from symmetrix.polar import SymmetrixPolar

    model = torch.load(path, map_location="cpu", weights_only=False).double().eval()
    assert len(model.interactions) == 1
    assert model.products[0].linear.irreps_out.dim == 512
    device = _selected_device()
    expected = _reference(model, lithium_water, device)
    calculator = SymmetrixPolar(path, device=device, bridge="dlpack")
    calculator.calculate(lithium_water, properties=["energy", "forces"])
    _assert_polar_result(calculator.results, expected, lithium_water, atol=5e-7)
