"""Restore derived Polar checkpoint metadata without changing a single tensor.

Adapted from ACEsuit/mace's fixed upstream compatibility helper.  The pinned
v0.3.16 model lacks the later model-level PBC setter, so its two public graph
setters are called directly.  This is not a model or weight reconstruction.
"""

# The metadata derivation follows ACEsuit/mace, MIT licensed,
# mace/tools/polar_conversion.py at 2133e56642ca6380887bc7f8c0f3448c2fdc32ac.
# Its source identity is retained below. No upstream model tensors are replaced.
from __future__ import annotations

import hashlib
import json
from pathlib import Path

UPSTREAM = {
    "repository": "https://github.com/ACEsuit/mace",
    "commit": "2133e56642ca6380887bc7f8c0f3448c2fdc32ac",
    "path": "mace/tools/polar_conversion.py",
    "sha256": "5d9d15b0b8b94928576e47a2c466be79ef0c71954d9035c1dac090de231b5c17",
    "tests_path": "tests/extensions/polar/test_polar_conversion.py",
    "tests_sha256": "daa014426f8f93afc297e62614066dba8c8446bdcce54d111309c5b1e0c9349a",
}


def tensor_inventory(model):
    result = {}
    for name, value in sorted(model.state_dict().items()):
        data = value.detach().cpu().contiguous().numpy()
        result[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        }
    return result


def inventory_sha256(inventory):
    return hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _same_objects(before, after):
    return before.keys() == after.keys() and all(
        before[name] is after[name] for name in before
    )


def repair_polar_checkpoint(model, *, pbc_handling="pbc"):
    """Validate everything first, restore only missing dimensions, then audit."""
    from graph_longrange.energy import GTOElectrostaticEnergy
    from graph_longrange.features import GTOElectrostaticFeatures
    from graph_longrange.gto_utils import GTOSelfInteractionBlock

    if pbc_handling != "pbc":
        raise ValueError(
            "This benchmark requires explicit three-dimensional periodic electrostatics"
        )
    if not any(cls.__name__ == "PolarMACE" for cls in type(model).__mro__):
        raise TypeError("Expected a PolarMACE whole model")
    roots = (
        (
            "electric_potential_descriptor",
            model.electric_potential_descriptor,
            GTOElectrostaticFeatures,
        ),
        ("coulomb_energy", model.coulomb_energy, GTOElectrostaticEnergy),
    )
    checks, repairs, seen = [], [], set()
    for name, root, expected_type in roots:
        if not isinstance(root, expected_type) or not callable(
            getattr(root, "set_pbc_handling", None)
        ):
            raise TypeError(
                f"{name} lacks the fixed graph_longrange public PBC interface"
            )
        for child_name, module in root.named_modules():
            if not isinstance(module, GTOSelfInteractionBlock) or id(module) in seen:
                continue
            seen.add(id(module))
            dimension = int(module.features_irreps.dim)
            if getattr(module, "features_dim", dimension) != dimension:
                raise ValueError(
                    f"Conflicting self-interaction features_dim: {name}.{child_name}"
                )
            missing = not hasattr(module, "features_dim")
            path = name + ("." + child_name if child_name else "")
            checks.append(
                {
                    "module": path,
                    "features_dim": dimension,
                    "derivation": "existing features_irreps.dim",
                    "restored": missing,
                }
            )
            if missing:
                repairs.append((module, dimension))
    descriptor = model.electric_potential_descriptor
    expected = descriptor._build_output_permutation(
        model.field_feature_max_l, len(model.field_feature_widths)
    )
    permutation = descriptor._buffers.get("output_permutation")
    if permutation is None or not permutation.cpu().equal(expected):
        raise ValueError("Unsupported electrostatic output_permutation buffer schema")

    # Keep actual objects as well as independent value hashes: even replacement
    # by an equal tensor/module would violate the intended migration boundary.
    before = tensor_inventory(model)
    modules_before = dict(model.named_modules())
    parameters_before = dict(model.named_parameters(remove_duplicate=False))
    buffers_before = dict(model.named_buffers(remove_duplicate=False))
    for module, dimension in repairs:
        module.features_dim = dimension
    for _, root, _ in roots:
        root.set_pbc_handling(pbc_handling)
    descriptor.static_quantities = None
    dispatch = (
        (descriptor, "_precompute_geometry_impl"),
        (descriptor, "_forward_dynamic_impl"),
        (model.coulomb_energy, "_forward_impl"),
    )
    for root, name in dispatch:
        if (
            root.pbc_handling != pbc_handling
            or getattr(getattr(root, name, None), "__self__", None) is not root
        ):
            raise RuntimeError(
                "Polar PBC dispatch did not bind to the current graph object"
            )
    after = tensor_inventory(model)
    same_modules = _same_objects(modules_before, dict(model.named_modules()))
    same_parameters = _same_objects(
        parameters_before, dict(model.named_parameters(remove_duplicate=False))
    )
    same_buffers = _same_objects(
        buffers_before, dict(model.named_buffers(remove_duplicate=False))
    )
    if before != after or not same_modules or not same_parameters or not same_buffers:
        raise RuntimeError(
            "Polar metadata restoration changed model tensors or object ownership"
        )
    return {
        "passed": True,
        "upstream": dict(UPSTREAM),
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy": "restore_missing_derived_metadata_preserve_all_tensors_and_modules",
        "self_interaction_blocks": checks,
        "restored_attribute_count": len(repairs),
        "output_permutation_validated": True,
        "output_permutation_replaced": False,
        "pbc_handling": pbc_handling,
        "dispatch_bound_to_current_objects": True,
        "cleared_transient_cache": "electric_potential_descriptor.static_quantities",
        "state_dict_tensor_count": len(before),
        "state_dict_sha256_before": inventory_sha256(before),
        "state_dict_sha256_after": inventory_sha256(after),
        "all_state_tensors_unchanged": True,
        "module_objects_unchanged": same_modules,
        "parameter_objects_unchanged": same_parameters,
        "buffer_objects_unchanged": same_buffers,
    }
