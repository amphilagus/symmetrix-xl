#!/usr/bin/env python3
"""Inventory a trusted whole-model POLAR checkpoint without running forward.

Run with the locked profile interpreter inside a Slurm job. The report records
the loaded checkpoint, including inactive boundary-condition branches; coverage
is serialization/ownership coverage, not a claim that every tensor is active in
the chosen PBC forward. Reports must be saved outside the source checkout.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def type_name(value):
    cls = value if isinstance(value, type) else type(value)
    return cls.__module__ + "." + cls.__qualname__


def tensor_record(value, *, small_values=False):
    import torch

    if value.layout != torch.strided:
        raise TypeError(f"Unsupported checkpoint tensor layout {value.layout}")
    contiguous = value.detach().cpu().contiguous()
    raw = contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()
    record = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": value.numel(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_order": sys.byteorder,
        "requires_grad": value.requires_grad,
    }
    if small_values and value.numel() <= 32:
        record["values"] = clean(contiguous.tolist(), {})
    return record


def clean(value, module_paths, seen=None):
    """Serialize config values without unstable object reprs or tensor payloads."""
    import torch

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if torch.is_tensor(value):
        return {"tensor": tensor_record(value, small_values=True)}
    if isinstance(value, torch.nn.Module):
        return {"module_ref": module_paths.get(id(value)), "class": type_name(value)}
    if type(value).__module__.startswith("e3nn") and hasattr(value, "dim"):
        return {"class": type_name(value), "irreps": str(value), "dim": int(value.dim)}
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, type):
        return {"class": type_name(value)}
    if callable(value):
        return {
            "callable": getattr(value, "__module__", type(value).__module__)
            + "."
            + getattr(value, "__qualname__", type(value).__qualname__),
            "bound_module": module_paths.get(id(getattr(value, "__self__", None))),
        }
    seen = set() if seen is None else seen
    if id(value) in seen:
        return {"recursive_container": type_name(value)}
    seen = seen | {id(value)}
    if hasattr(value, "_asdict"):
        return clean(value._asdict(), module_paths, seen)
    if isinstance(value, dict):
        return {
            str(key): clean(item, module_paths, seen)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [clean(item, module_paths, seen) for item in value]
    if isinstance(value, (set, frozenset)):
        return [clean(item, module_paths, seen) for item in sorted(value, key=str)]
    if hasattr(value, "tolist"):
        return clean(value.tolist(), module_paths, seen)
    return {"class": type_name(value), "opaque_config_value": True}


def tensor_owner(name):
    root = name.split(".", 1)[0]
    if root in {"node_embedding", "interactions", "products"}:
        return "native_local_features"
    if root in {"radial_embedding", "spherical_harmonics"}:
        return "native_local_features_and_torch_suffix_geometry"
    if root in {
        "atomic_numbers",
        "r_max",
        "num_interactions",
        "kspace_cutoff",
        "field_feature_norms",
    }:
        return "shared_model_constants"
    if root == "pair_repulsion_fn":
        return "retained_upstream_auxiliary_not_added_to_polar_energy"
    if root in {
        "atomic_energies_fn",
        "scale_shift",
        "readouts",
        "lr_source_maps",
        "electric_potential_descriptor",
        "fukui_source_map",
        "field_dependent_charges_maps",
        "local_electron_energy",
        "external_field_contribution",
        "coulomb_energy",
        "layer_feature_mixer",
        "_charges_to_mul_ir",
        "_charges_from_mul_ir",
        "_field_from_mul_ir",
        "_fukui_to_mul_ir",
    }:
        return "retained_torch_polar_suffix"
    return "unclassified"


def inventories(model):
    entries = {}
    aliases = {}
    for kind, values in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in values:
            entry = tensor_record(tensor)
            entry.update(kind=kind, owner=tensor_owner(name))
            if id(tensor) in aliases:
                entry["alias_of"] = aliases[id(tensor)]
            else:
                aliases[id(tensor)] = name
            entries[name] = entry
    return entries


def repository_record(source):
    def git(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(source), *arguments], text=True, stderr=subprocess.DEVNULL
        ).strip()

    return {
        "root": str(source),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "script_sha256": sha256_file(__file__),
    }


def collect_contract(checkpoint, source):
    import torch

    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Expected a whole-model checkpoint, not a state dictionary")
    before = inventories(model)
    named_modules = list(model.named_modules(remove_duplicate=False))
    module_paths = {}
    for name, module in named_modules:
        module_paths.setdefault(id(module), name)
    internal = set(vars(torch.nn.Module())) - {"training"}
    modules = {}
    source_files = {}
    unavailable_sources = set()
    derived_metadata = []
    for name, module in named_modules:
        config = {
            key: clean(value, module_paths)
            for key, value in sorted(vars(module).items())
            if key not in internal
        }
        modules[name] = {"class": type_name(module), "config": config}
        try:
            filename = inspect.getsourcefile(type(module))
        except (TypeError, OSError):
            filename = None
        if filename and Path(filename).is_file():
            filename = str(Path(filename).resolve())
            source_files.setdefault(
                filename, {"sha256": sha256_file(filename), "classes": []}
            )
            if type_name(module) not in source_files[filename]["classes"]:
                source_files[filename]["classes"].append(type_name(module))
        else:
            unavailable_sources.add(type_name(module))
        if type(module).__name__ == "GTOSelfInteractionBlock":
            expected = int(module.features_irreps.dim)
            present = hasattr(module, "features_dim")
            actual = getattr(module, "features_dim", None)
            derived_metadata.append(
                {
                    "module": name,
                    "attribute": "features_dim",
                    "present": present,
                    "value": actual,
                    "unique_derivation": "existing features_irreps.dim",
                    "expected": expected,
                    "conflict": present and actual != expected,
                }
            )
    for relative in (
        "symmetrix/source/symmetrix/polar.py",
        "symmetrix/source/symmetrix/_polar_compat.py",
        "symmetrix/source/symmetrix/extract_mace_nonlinear.py",
        "libsymmetrix/source/mace_nonlinear_kokkos.hpp",
        "libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp",
        "symmetrix/source/cpp/mace_nonlinear_kokkos.cpp",
    ):
        path = source / relative
        if path.is_file():
            source_files[str(path)] = {
                "sha256": sha256_file(path),
                "candidate_source": True,
            }
    package_sources = {}
    for package in ("mace", "graph_longrange"):
        specification = importlib.util.find_spec(package)
        if specification is None or specification.origin is None:
            continue
        directory = Path(specification.origin).resolve().parent
        files = {
            str(path.relative_to(directory)): sha256_file(path)
            for path in sorted(directory.rglob("*.py"))
        }
        package_sources[package] = {
            "root": str(directory),
            "file_count": len(files),
            "files": files,
            "source_tree_sha256": hashlib.sha256(
                json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    after = inventories(model)
    state_names = sorted(model.state_dict())
    state_without_named_tensor = sorted(set(state_names) - set(before))
    unclassified = [
        name for name, entry in before.items() if entry["owner"] == "unclassified"
    ]
    versions = {}
    for package in (
        "torch",
        "mace-torch",
        "graph-longrange",
        "graph-electrostatics",
        "e3nn",
        "ase",
        "numpy",
        "symmetrix-xl",
        "cuequivariance",
        "cuequivariance-torch",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    layers = []
    for index, (interaction, product) in enumerate(
        zip(model.interactions, model.products)
    ):
        layers.append(
            {
                "interaction": f"interactions.{index}",
                "interaction_class": type_name(interaction),
                "product": f"products.{index}",
                "product_class": type_name(product),
                "product_output_irreps": str(product.linear.irreps_out),
                "product_output_dim": int(product.linear.irreps_out.dim),
                "use_agnostic_product": bool(
                    getattr(product, "use_agnostic_product", False)
                ),
                "use_sc": bool(product.use_sc),
                "correlation": [
                    int(block.correlation)
                    for block in product.symmetric_contractions.contractions
                ],
            }
        )
    attributes = (
        "r_max",
        "num_interactions",
        "heads",
        "num_recursion_steps",
        "atomic_multipoles_max_l",
        "atomic_multipoles_smearing_width",
        "field_feature_max_l",
        "field_feature_widths",
        "field_feature_norms",
        "kspace_cutoff",
        "kspace_cutoff_factor",
        "field_norm_factor",
        "field_si",
        "include_electrostatic_self_interaction",
        "add_local_electron_energy",
        "charges_irreps",
        "field_irreps",
        "potential_irreps",
        "_fixedpoint_update_config",
        "_field_readout_config",
        "use_last_readout_only",
        "keep_last_layer_irreps",
        "pair_repulsion",
    )
    resolved = {
        name: clean(getattr(model, name), module_paths)
        for name in attributes
        if hasattr(model, name)
    }
    checks = {
        "whole_polar_model": any(
            cls.__name__ == "PolarMACE" for cls in type(model).__mro__
        ),
        "one_local_layer": len(model.interactions) == len(model.products) == 1,
        "scalar_product_output": all(
            ir.l == 0 and ir.p == 1
            for product in model.products
            for _, ir in product.linear.irreps_out
        ),
        "single_linear_readout": len(model.readouts) == 1
        and type(model.readouts[0]).__name__ == "LinearReadoutBlock",
        "two_field_steps": int(model.num_recursion_steps)
        == len(model.field_dependent_charges_maps)
        == 2,
        "derived_metadata_consistent": not any(
            row["conflict"] for row in derived_metadata
        ),
        "all_named_tensors_classified": not unclassified,
        "all_state_dict_entries_inventoried": not state_without_named_tensor,
        "all_named_tensors_unchanged": before == after,
    }
    inventory_fingerprint = hashlib.sha256(
        json.dumps(before, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "kind": "POLAR_CHECKPOINT_CONTRACT",
        "status": "ACCEPTED" if all(checks.values()) else "REJECTED",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_class": type_name(model),
        "atomic_numbers": model.atomic_numbers.tolist(),
        "resolved_model_attributes": resolved,
        "local_layers": layers,
        "named_tensors": before,
        "named_modules": modules,
        "state_dict_tensor_names": state_names,
        "tensor_inventory_sha256": inventory_fingerprint,
        "full_weight_coverage": {
            "passed": not unclassified and not state_without_named_tensor,
            "meaning": "Every named tensor is retained and assigned an owner; inactive boundary branches are included. This does not certify forward execution or numerical parity.",
            "named_parameter_count": sum(
                row["kind"] == "parameter" for row in before.values()
            ),
            "named_buffer_count": sum(
                row["kind"] == "buffer" for row in before.values()
            ),
            "unique_tensor_objects": sum(
                "alias_of" not in row for row in before.values()
            ),
            "unclassified_tensor_names": unclassified,
            "state_dict_entries_without_named_tensor": state_without_named_tensor,
        },
        "derived_metadata": derived_metadata,
        "missing_derived_metadata_requires_runtime_repair": any(
            not row["present"] for row in derived_metadata
        ),
        "source": repository_record(source),
        "source_files": source_files,
        "package_source_trees": package_sources,
        "classes_without_source_file": sorted(unavailable_sources),
        "versions": versions,
        "python": {"executable": sys.executable, "version": sys.version},
        "inspection": {
            "device": "cpu",
            "forward_called": False,
            "model_saved": False,
            "model_tensors_unchanged": before == after,
        },
        "checks": checks,
    }


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name + ".", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = Path(__file__).resolve().parents[2]
    output = args.output.expanduser().resolve()
    if output.is_relative_to(source):
        parser.error("Save checkpoint contract reports outside the source checkout")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    context = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "output": str(output),
    }
    try:
        report = collect_contract(checkpoint, source)
    except Exception as error:
        report = {
            "schema_version": 1,
            "kind": "POLAR_CHECKPOINT_CONTRACT",
            "status": "FAILED",
            "checkpoint": str(checkpoint),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
    report.update(context)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_report(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "report_sha256": sha256_file(output),
            }
        )
    )
    return 0 if report["status"] == "ACCEPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
