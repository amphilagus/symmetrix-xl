#!/usr/bin/env python3
"""Isolate native features from POLAR suffix differences in an allocated job.

Uses the checkout's complete official-forward oracle and its small Li+water
fixture. No forward implementation or acceptance threshold is replaced. The
report contains comparisons, not a qualification pass/fail decision.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch


def comparison(actual, expected, rtol=1e-7, atol=1e-9):
    import numpy as np

    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.size == expected.size == 1:
        actual, expected = actual.reshape(()), expected.reshape(())
    if actual.shape != expected.shape:
        return {
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }
    delta = np.abs(actual - expected)
    ratio = delta / (atol + rtol * np.abs(expected))
    index = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
    return {
        "shape": list(actual.shape),
        "rtol": rtol,
        "atol": atol,
        "max_abs": float(delta.max()),
        "max_tolerance_ratio": float(ratio.max()),
        "outside_tolerance": int(np.count_nonzero(ratio > 1)),
        "worst_index": list(map(int, index)),
        "worst_actual": float(actual[index]),
        "worst_expected": float(expected[index]),
    }


def observed_values(output):
    names = (
        "energy",
        "forces",
        "charges",
        "density_coefficients",
        "spin_charge_density",
        "electron_energy",
        "electrostatic_energy",
        "density_history",
        "fukui_history",
        "potential_history",
    )
    values = {}
    for name in names:
        if name not in output:
            continue
        value = output[name]
        if isinstance(value, list):
            values[name] = [item.detach().cpu().numpy().copy() for item in value]
        else:
            values[name] = value.detach().cpu().numpy().copy()
    return values


def compare_outputs(actual, expected):
    result = {}
    for name, value in actual.items():
        if name not in expected:
            continue
        if isinstance(value, list):
            result[name] = [
                comparison(a, b) for a, b in zip(value, expected[name], strict=True)
            ]
        else:
            result[name] = comparison(value, expected[name])
    return result


def wigner_inventory(model):
    import torch
    from e3nn import o3

    result = []
    for name, module in model.named_modules():
        if not all(
            hasattr(module, item)
            for item in ("irreps_in1", "irreps_in2", "irreps_out", "instructions")
        ):
            continue
        triples = set()
        for instruction in module.instructions:
            if not all(
                hasattr(instruction, item) for item in ("i_in1", "i_in2", "i_out")
            ):
                continue
            triples.add(
                (
                    module.irreps_in1[instruction.i_in1].ir.l,
                    module.irreps_in2[instruction.i_in2].ir.l,
                    module.irreps_out[instruction.i_out].ir.l,
                )
            )
        buffers = dict(module.named_buffers())
        for triple in sorted(triples):
            generated32 = o3.wigner_3j(*triple, dtype=torch.float32).double()
            generated64 = o3.wigner_3j(*triple, dtype=torch.float64)
            row = {
                "module": name,
                "l1_l2_lout": list(triple),
                "scalar_input": triple[0] == 0 or triple[1] == 0,
                "f32_upcast_vs_f64": comparison(
                    generated32.numpy(), generated64.numpy(), rtol=0, atol=1e-15
                ),
                "checkpoint_buffers": [],
            }
            for buffer_name, tensor in buffers.items():
                match = re.search(r"(?:w3j|wigner)_(\d+)_(\d+)_(\d+)$", buffer_name)
                if not match or tuple(map(int, match.groups())) != triple:
                    continue
                raw = tensor.detach().cpu().contiguous()
                value = raw.double().numpy()
                row["checkpoint_buffers"].append(
                    {
                        "name": buffer_name,
                        "dtype": str(raw.dtype),
                        "shape": list(raw.shape),
                        "sha256": hashlib.sha256(
                            raw.reshape(-1).view(torch.uint8).numpy().tobytes()
                        ).hexdigest(),
                        "vs_f32_upcast": comparison(
                            value, generated32.numpy(), rtol=0, atol=1e-15
                        ),
                        "vs_direct_f64": comparison(
                            value, generated64.numpy(), rtol=0, atol=1e-15
                        ),
                    }
                )
            result.append(row)
    return result


def diagnose(checkpoint, source, device, bridge):
    import torch
    import symmetrix.polar as polar

    sys.path.insert(0, str(source / "symmetrix/test"))
    from test_polar import _reference, lithium_water

    model = (
        torch.load(checkpoint, map_location="cpu", weights_only=False).double().eval()
    )
    atoms = lithium_water.__wrapped__()
    official_features = []

    def product_hook(module, args, output):
        official_features.append(output.detach().clone())

    handles = [
        product.register_forward_hook(product_hook) for product in model.products
    ]
    try:
        # _reference deep-copies the model and keeps the original forward;
        # observation hooks are copied too and retain this capture closure.
        reference = _reference(model, atoms, device)
    finally:
        for handle in handles:
            handle.remove()

    observed = {}
    original_suffix = polar._suffix

    def suffix_observer(*args, **kwargs):
        observed["args"] = args
        observed["native_features"] = [value.detach().clone() for value in args[1]]
        output = original_suffix(*args, **kwargs)
        observed["output"] = observed_values(output)
        return output

    calculator = polar.SymmetrixPolar(checkpoint, device=device, bridge=bridge)
    with patch.object(polar, "_suffix", side_effect=suffix_observer):
        calculator.calculate(atoms, properties=["energy", "forces"])
    if len(official_features) != len(observed["native_features"]):
        raise RuntimeError("Official product hooks did not capture every local layer")
    arguments = list(observed["args"])
    arguments[1] = official_features
    with torch.no_grad():
        suffix_from_official = observed_values(original_suffix(*arguments))
    artifact = Path(calculator._artifact_directory.name) / "local.json"
    artifact_data = json.loads(artifact.read_text())
    serialized_wigner = []
    for layer, interaction in enumerate(artifact_data["interactions"]):
        for instruction in interaction["conv_tp"]["instructions"]:
            serialized_wigner.append(
                {
                    "layer": layer,
                    "i_in1": instruction["i_in1"],
                    "i_in2": instruction["i_in2"],
                    "i_out": instruction["i_out"],
                    "path_weight": instruction["path_weight"],
                    "wigner_3j": instruction["wigner_3j"],
                }
            )
    return {
        "kind": "POLAR_STAGE_DIAGNOSTIC",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_version": torch.__version__,
        "native_diagnostics": {
            key: value
            for key, value in calculator.results["polar_diagnostics"].items()
            if not key.endswith("_history")
        },
        "official_feature_suffix_vs_official_forward": compare_outputs(
            suffix_from_official, reference
        ),
        "native_feature_suffix_vs_official_forward": compare_outputs(
            observed["output"], reference
        ),
        "native_features_vs_official_features": [
            comparison(a.cpu().numpy(), b.cpu().numpy())
            for a, b in zip(observed["native_features"], official_features, strict=True)
        ],
        "candidate_forces_vs_official": comparison(
            calculator.results["forces"], reference["forces"]
        ),
        "tensor_product_wigner_inventory": wigner_inventory(calculator.model),
        "serialized_conv_wigner": serialized_wigner,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bridge", choices=("auto", "host", "dlpack"), default="auto")
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error(
            "Run this diagnostic inside a Slurm allocation with the locked profile interpreter"
        )
    source = Path(__file__).resolve().parents[2]
    output = args.output.expanduser().resolve()
    if output.is_relative_to(source):
        parser.error("Save diagnostic reports outside the source checkout")
    with redirect_stdout(sys.stderr):
        report = diagnose(
            args.checkpoint.expanduser().resolve(strict=True),
            source,
            args.device,
            args.bridge,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({"output": str(output), "kind": report["kind"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
