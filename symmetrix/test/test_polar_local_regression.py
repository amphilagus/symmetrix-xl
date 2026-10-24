"""Ordinary nonlinear-MACE inference must survive feature/VJP bridge use.

These oracles are genuine ScaleShiftMACE models, without any POLAR modules or
external checkpoint. All native evaluators are test-local: pytest fixtures
retain only Torch models, geometry and JSON paths through session teardown.
"""

import json

import numpy as np
import pytest


@pytest.fixture(
    scope="module",
    params=[(1, 4), (2, 4), (2, 8)],
    ids=["one-layer-c4", "two-layer-c4", "two-layer-c8"],
)
def ordinary_artifact(request, tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("mace")
    from e3nn import o3
    from mace.modules.blocks import RealAgnosticResidualNonLinearInteractionBlock
    from mace.modules.models import ScaleShiftMACE
    from symmetrix.extract_mace_data import extract_mace_data

    layers, channels = request.param
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3781 + 17 * layers + channels)
            model = (
                ScaleShiftMACE(
                    atomic_inter_scale=1.7,
                    atomic_inter_shift=-0.13,
                    r_max=4.3,
                    num_bessel=4,
                    num_polynomial_cutoff=5,
                    max_ell=2,
                    interaction_cls=RealAgnosticResidualNonLinearInteractionBlock,
                    interaction_cls_first=RealAgnosticResidualNonLinearInteractionBlock,
                    num_interactions=layers,
                    num_elements=3,
                    hidden_irreps=o3.Irreps(f"{channels}x0e+{channels}x1o"),
                    MLP_irreps=o3.Irreps("5x0e"),
                    atomic_energies=np.array([[0.25, -0.35, 0.8]]),
                    atomic_numbers=[1, 8, 14],
                    avg_num_neighbors=4.0,
                    correlation=3,
                    gate=torch.nn.functional.silu,
                    use_agnostic_product=True,
                    edge_irreps=o3.Irreps("3x0e+3x1o+3x2e"),
                    use_edge_irreps_first=True,
                    radial_MLP=[8, 8],
                    heads=["Default"],
                )
                .double()
                .eval()
            )
    finally:
        torch.set_default_dtype(previous_dtype)
    assert type(model).__name__ == "ScaleShiftMACE"
    assert model.products[-1].linear.irreps_out.lmax == 0
    # Exercise the public ordinary converter, including its nonlinear dispatch.
    directory = tmp_path_factory.mktemp(f"ordinary-local-{layers}-{channels}")
    checkpoint = directory / "ordinary.model"
    torch.save(model, checkpoint)
    payload = extract_mace_data(checkpoint)
    assert payload["model_type"] == "MACE_Nonlinear"
    path = directory / "model.json"
    path.write_text(json.dumps(payload))
    return model, path


@pytest.fixture
def ordinary_geometry():
    from ase import Atoms

    return Atoms(
        numbers=[14, 8, 1, 1, 8, 1, 1],
        positions=[
            [0.35, 2.7, 3.1],
            [2.5, 2.6, 3.3],
            [3.35, 2.9, 3.55],
            [2.0, 3.35, 3.1],
            [6.45, 2.75, 3.4],
            [6.1, 3.55, 3.7],
            [5.75, 2.2, 3.2],
        ],
        cell=[7.0, 7.3, 7.5],
        pbc=True,
    )


def _graph(model, atoms):
    import torch
    from mace import data
    from mace.tools import torch_geometric, utils

    config = data.config_from_atoms(atoms, head_name="Default")
    graph = data.AtomicData.from_config(
        config,
        z_table=utils.AtomicNumberTable([int(value) for value in model.atomic_numbers]),
        cutoff=float(model.r_max),
        heads=model.heads,
    )
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[graph], batch_size=1, shuffle=False
    )
    batch = next(iter(loader)).to("cpu").to_dict()
    return {
        name: value.to(torch.float64)
        if torch.is_tensor(value) and value.is_floating_point()
        else value
        for name, value in batch.items()
    }


def _oracle_and_inputs(model, atoms):
    from mace.modules.utils import get_edge_vectors_and_lengths

    graph = _graph(model, atoms)
    output = model(graph, compute_force=True, compute_edge_forces=True)
    expected = {
        name: output[name].detach().cpu().numpy().copy()
        for name in ("energy", "node_energy", "forces", "edge_forces")
    }
    vectors, _ = get_edge_vectors_and_lengths(
        graph["positions"], graph["edge_index"], graph["shifts"]
    )
    # Ordinary MACE uses these vectors directly; POLAR's [y,z,x] permutation
    # must not leak into this existing inference path.
    vectors = vectors.detach().cpu().numpy()
    source, receiver = graph["edge_index"].cpu().numpy()
    order = np.argsort(receiver, kind="stable")
    node_types = graph["node_attrs"].argmax(dim=1).cpu().numpy().astype(np.int32)
    neighbors = source[order].astype(np.int32)
    ordered_vectors = np.ascontiguousarray(vectors[order])
    inputs = (
        len(atoms),
        node_types,
        np.bincount(receiver, minlength=len(atoms)).astype(np.int32),
        neighbors,
        node_types[neighbors],
        ordered_vectors.reshape(-1),
        np.linalg.norm(ordered_vectors, axis=1),
    )
    # Pinned MACE utils.get_outputs negates compute_forces(vectors) once more
    # "to match LAMMPS sign convention": its public edge_forces is +dE/dv.
    # The native API returns -dE/d(input xyz), so compare the documented forms.
    expected["edge_forces"] = -expected["edge_forces"][order]
    return expected, inputs, source[order], receiver[order]


def _native_evaluator(path):
    import symmetrix

    native = symmetrix.load_backend()
    if not native._kokkos_is_initialized():
        native._init_kokkos()
    if native._kokkos_default_execution_space() not in (
        "Serial",
        "OpenMP",
        "Threads",
        "Cuda",
    ):
        pytest.skip("Ordinary feature-bridge regression qualifies CPU/CUDA FP64")
    return native.MACENonlinearKokkos(str(path))


def _compute(evaluator, inputs):
    evaluator.compute_node_energies_forces(*inputs)
    evaluator.fence()
    return {
        "node_energy": np.array(evaluator.node_energies, copy=True),
        "edge_forces": np.array(evaluator.node_forces, copy=True).reshape(-1, 3),
    }


def _assert_ordinary_oracle(actual, expected, sources, receivers, atom_count):
    np.testing.assert_allclose(
        actual["node_energy"], expected["node_energy"], rtol=2e-10, atol=2e-10
    )
    np.testing.assert_allclose(
        actual["node_energy"].sum(), expected["energy"].sum(), rtol=2e-10, atol=2e-10
    )
    np.testing.assert_allclose(
        actual["edge_forces"], expected["edge_forces"], rtol=2e-10, atol=2e-10
    )
    forces = np.zeros((atom_count, 3))
    np.add.at(forces, receivers, actual["edge_forces"])
    np.add.at(forces, sources, -actual["edge_forces"])
    np.testing.assert_allclose(forces, expected["forces"], rtol=2e-10, atol=2e-10)


def test_ordinary_energy_forces_are_isolated_from_feature_vjp(
    ordinary_artifact, ordinary_geometry
):
    model, path = ordinary_artifact
    expected, inputs, sources, receivers = _oracle_and_inputs(model, ordinary_geometry)
    evaluator = _native_evaluator(path)
    try:
        before = _compute(evaluator, inputs)
        _assert_ordinary_oracle(
            before, expected, sources, receivers, len(ordinary_geometry)
        )
        assert np.max(np.abs(before["edge_forces"])) > 1e-7
        token = evaluator._forward_features(*inputs)
        features = evaluator._feature_arrays(token)
        rng = np.random.default_rng(729)
        seeds = [
            np.ascontiguousarray(11 * rng.normal(size=value.shape))
            for value in features
        ]
        evaluator._reverse_features_host(token, seeds)
        pure_feature_force = np.array(evaluator.node_forces).reshape(-1, 3)
        assert np.max(np.abs(pure_feature_force - before["edge_forces"])) > 1e-6
        after = _compute(evaluator, inputs)
        _assert_ordinary_oracle(
            after, expected, sources, receivers, len(ordinary_geometry)
        )
        for name in before:
            np.testing.assert_allclose(
                after[name], before[name], rtol=5e-12, atol=5e-12
            )

        displaced = ordinary_geometry.copy()
        displaced.positions[3] += [0.023, -0.017, 0.011]
        other_expected, other_inputs, other_sources, other_receivers = (
            _oracle_and_inputs(model, displaced)
        )
        _assert_ordinary_oracle(
            _compute(evaluator, other_inputs),
            other_expected,
            other_sources,
            other_receivers,
            len(displaced),
        )
        restored = _compute(evaluator, inputs)
        _assert_ordinary_oracle(
            restored, expected, sources, receivers, len(ordinary_geometry)
        )
    finally:
        # Keep a live native owner out of pytest's last-item fixture retention.
        del evaluator


def test_ordinary_compute_invalidates_an_unconsumed_feature_tape(
    ordinary_artifact, ordinary_geometry
):
    model, path = ordinary_artifact
    expected, inputs, sources, receivers = _oracle_and_inputs(model, ordinary_geometry)
    evaluator = _native_evaluator(path)
    try:
        token = evaluator._forward_features(*inputs)
        actual = _compute(evaluator, inputs)
        _assert_ordinary_oracle(
            actual, expected, sources, receivers, len(ordinary_geometry)
        )
        with pytest.raises(
            (ValueError, RuntimeError), match="token|tape|stale|consum|invalid"
        ):
            evaluator._feature_arrays(token)
    finally:
        del evaluator
