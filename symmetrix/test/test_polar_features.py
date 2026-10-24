"""Independent feature and arbitrary-seed VJP oracles for the native bridge."""

import json

import numpy as np
import pytest
from test_polar import (
    _batch,
    _polar_dependencies,
    _selected_device,
    lithium_water,  # noqa: F401 -- imported pytest fixtures
    reduced_polar_model,  # noqa: F401
)


def _backbone_oracle(model, graph):
    """Use upstream blocks and a leaf edge vector, without any POLAR suffix."""
    torch, _ = _polar_dependencies()
    from mace.modules.utils import get_edge_vectors_and_lengths

    vectors, _ = get_edge_vectors_and_lengths(
        graph["positions"], graph["edge_index"], graph["shifts"]
    )
    # PolarMACE applies this permutation before its spherical harmonics.
    spherical_vectors = vectors[:, [1, 2, 0]].detach().requires_grad_(True)
    lengths = spherical_vectors.norm(dim=-1, keepdim=True)
    edge_attrs = model.spherical_harmonics(spherical_vectors)
    edge_feats, cutoff = model.radial_embedding(
        lengths, graph["node_attrs"], graph["edge_index"], model.atomic_numbers
    )
    features = model.node_embedding(graph["node_attrs"])
    outputs = []
    for index, (interaction, product) in enumerate(
        zip(model.interactions, model.products, strict=True)
    ):
        features, skip = interaction(
            node_attrs=graph["node_attrs"],
            node_feats=features,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=graph["edge_index"],
            cutoff=cutoff,
            first_layer=index == 0,
        )
        features = product(node_feats=features, sc=skip, node_attrs=graph["node_attrs"])
        outputs.append(features)
    generator = torch.Generator(device="cpu").manual_seed(8713)
    seeds = [
        torch.randn(output.shape, dtype=torch.float64, generator=generator)
        for output in outputs
    ]
    gradient = torch.autograd.grad(
        sum((value * seed).sum() for value, seed in zip(outputs, seeds, strict=True)),
        spherical_vectors,
    )[0]
    return outputs, seeds, spherical_vectors, -gradient


@pytest.fixture
def native_feature_case(reduced_polar_model, lithium_water, tmp_path):
    import symmetrix
    from symmetrix.extract_mace_nonlinear import extract_mace_nonlinear_data

    # This internal serializer intentionally extracts only the native prefix;
    # the public converter must reject the same whole-model POLAR checkpoint.
    path = tmp_path / "polar-prefix.json"
    path.write_text(json.dumps(extract_mace_nonlinear_data(reduced_polar_model)))
    _selected_device()
    native = symmetrix.load_backend().MACENonlinearKokkos(str(path))
    graph = _batch(reduced_polar_model, lithium_water)
    outputs, seeds, xyz, negative_gradient = _backbone_oracle(
        reduced_polar_model, graph
    )
    source, receiver = graph["edge_index"].cpu().numpy()
    order = np.argsort(receiver, kind="stable")
    node_types = graph["node_attrs"].argmax(dim=1).cpu().numpy().astype(np.int32)
    neighbors = source[order].astype(np.int32)
    vectors = xyz.detach().cpu().numpy()[order]
    inputs = (
        len(lithium_water),
        node_types,
        np.bincount(receiver, minlength=len(lithium_water)).astype(np.int32),
        neighbors,
        node_types[neighbors],
        np.ascontiguousarray(vectors.reshape(-1)),
        np.linalg.norm(vectors, axis=1),
    )
    return (
        native,
        inputs,
        outputs,
        seeds,
        negative_gradient.detach().cpu().numpy()[order],
    )


def test_native_features_and_arbitrary_seed_vjp_match_upstream(native_feature_case):
    native, inputs, expected, seeds, negative_gradient = native_feature_case
    token = native._forward_features(*inputs)
    actual = native._feature_arrays(token)
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            np.asarray(result), reference.detach().numpy(), atol=2e-10, rtol=2e-10
        )
    native._reverse_features_host(token, [value.numpy() for value in seeds])
    np.testing.assert_allclose(
        np.asarray(native.node_forces).reshape(-1, 3),
        negative_gradient,
        atol=2e-10,
        rtol=2e-10,
    )


def test_dlpack_feature_snapshot_and_vjp_match_host_bridge(native_feature_case):
    torch, _ = _polar_dependencies()
    native, inputs, expected, seeds, negative_gradient = native_feature_case
    token = native._forward_features(*inputs)
    snapshots = [
        torch.utils.dlpack.from_dlpack(capsule)
        for capsule in native._feature_tensors(token)
    ]
    device = _selected_device()
    for result, reference in zip(snapshots, expected, strict=True):
        assert result.dtype == torch.float64
        assert result.device.type == device
        torch.testing.assert_close(
            result.cpu(), reference.detach(), atol=2e-10, rtol=2e-10
        )
    device_seeds = [value.to(device).contiguous() for value in seeds]
    native._reverse_features(
        token, [torch.utils.dlpack.to_dlpack(value) for value in device_seeds]
    )
    forces = torch.utils.dlpack.from_dlpack(native._edge_force_tensor())
    assert forces.device.type == device
    np.testing.assert_allclose(
        forces.cpu().numpy(), negative_gradient, atol=2e-10, rtol=2e-10
    )
    # Exported tensors own snapshots, not aliases to the evaluator's next tape.
    snapshot_values = [value.clone() for value in snapshots]
    native._forward_features(*inputs)
    for before, after in zip(snapshot_values, snapshots, strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize("invalidate", ["forward", "reverse", "ordinary_compute"])
def test_feature_tape_rejects_stale_tokens(native_feature_case, invalidate):
    native, inputs, _, seeds, _ = native_feature_case
    token = native._forward_features(*inputs)
    if invalidate == "forward":
        native._forward_features(*inputs)
    elif invalidate == "reverse":
        native._reverse_features_host(token, [value.numpy() for value in seeds])
    else:
        native.compute_node_energies_forces(*inputs)
    with pytest.raises(
        (ValueError, RuntimeError), match="token|tape|stale|consum|invalid"
    ):
        native._feature_arrays(token)
    with pytest.raises(
        (ValueError, RuntimeError), match="token|tape|stale|consum|invalid"
    ):
        native._reverse_features_host(token, [value.numpy() for value in seeds])


def test_feature_seed_shape_is_validated_before_reverse(native_feature_case):
    native, inputs, _, seeds, _ = native_feature_case
    token = native._forward_features(*inputs)
    malformed = [np.zeros((1, 1), dtype=np.float64)]
    with pytest.raises(
        (ValueError, RuntimeError), match="shape|size|extent|dimension|adjoint"
    ):
        native._reverse_features_host(token, malformed)
    # Failed validation must not turn a valid tape into a silently partial VJP.
    native._reverse_features_host(token, [value.numpy() for value in seeds])


def test_feature_tape_owns_inputs_after_forward_returns(native_feature_case):
    native, inputs, _, seeds, negative_gradient = native_feature_case
    token = native._forward_features(*inputs)
    # These host arrays can be reused by a neighbor builder before backward.
    # A tape that only retained borrowed NumPy views would now use wrong data.
    for array in inputs[1:]:
        array.fill(0)
    native._reverse_features_host(token, [value.numpy() for value in seeds])
    np.testing.assert_allclose(
        np.asarray(native.node_forces).reshape(-1, 3),
        negative_gradient,
        atol=2e-10,
        rtol=2e-10,
    )
