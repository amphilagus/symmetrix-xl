#!/usr/bin/env python3
"""Synchronized POLAR module diagnostics, excluded from throughput measurements.

Example (use the selected profile's interpreter/backend in a separate Slurm job):
  python benchmarks/polar_module_profile.py --checkpoint MODEL --input STATE \
      --device cuda --calls 3 --output module-profile.json

Only forward module regions are split. The complete Torch autograd stage includes
all suffix derivatives; this tool does not isolate long-range backward time.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import functools
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter


def pin_one_cpu():
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {allowed[0]})
    for name in (
        "OMP_NUM_THREADS",
        "KOKKOS_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    os.environ["OMP_PROC_BIND"] = "false"
    return {
        "allowed_before": allowed,
        "actual_affinity": sorted(os.sched_getaffinity(0)),
        "policy": "one logical CPU; SMT siblings excluded; before numerical imports",
    }


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ForwardTimer:
    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        self.reset()

    def reset(self):
        self.events, self.stack = [], []

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def start(self, label):
        self.synchronize()
        self.stack.append([label, perf_counter(), 0.0])

    def stop(self, label):
        self.synchronize()
        actual, started, children = self.stack.pop()
        if actual != label:
            raise RuntimeError("Module timing regions are not properly nested")
        elapsed = (perf_counter() - started) * 1000
        self.events.append(
            {
                "label": label,
                "inclusive_ms": elapsed,
                "exclusive_ms": elapsed - children,
            }
        )
        if self.stack:
            self.stack[-1][2] += elapsed

    def regions(self):
        if self.stack:
            raise RuntimeError("An instrumented forward region did not finish")
        result = {}
        for event in self.events:
            values = result.setdefault(
                event["label"], {"calls": 0, "inclusive_ms": 0.0, "exclusive_ms": 0.0}
            )
            values["calls"] += 1
            for key in ("inclusive_ms", "exclusive_ms"):
                values[key] += event[key]
        return result


@contextmanager
def instrument_forward(model, timer):
    """Restore original methods and remove hooks even if inference raises."""
    expected = {}
    with ExitStack() as cleanup:
        descriptor = model.electric_potential_descriptor
        missing = object()
        for name, count in (("precompute_geometry", 1), ("forward_dynamic", 4)):
            label = "electric_potential_descriptor." + name
            previous = descriptor.__dict__.get(name, missing)
            original = getattr(descriptor, name)

            def wrap(function, region):
                @functools.wraps(function)
                def measured(*args, **kwargs):
                    timer.start(region)
                    try:
                        return function(*args, **kwargs)
                    finally:
                        timer.stop(region)

                return measured

            setattr(descriptor, name, wrap(original, label))
            if previous is missing:
                cleanup.callback(delattr, descriptor, name)
            else:
                cleanup.callback(setattr, descriptor, name, previous)
            expected[label] = count
        modules = [
            ("coulomb_energy", model.coulomb_energy),
            ("local_electron_energy", model.local_electron_energy),
            ("fukui_source_map", model.fukui_source_map),
            ("layer_feature_mixer", model.layer_feature_mixer),
        ]
        for name in ("readouts", "lr_source_maps", "field_dependent_charges_maps"):
            modules.extend(
                (f"{name}[{i}]", module)
                for i, module in enumerate(getattr(model, name))
            )
        for label, module in modules:

            def before(_module, _arguments, region=label):
                timer.start(region)

            def after(_module, _arguments, _output, region=label):
                timer.stop(region)

            cleanup.callback(module.register_forward_pre_hook(before).remove)
            cleanup.callback(
                module.register_forward_hook(after, always_call=True).remove
            )
            expected[label] = 1
        yield expected


def calculate(calculator, atoms, timer, *, synchronize_before=True):
    from ase.calculators.calculator import all_changes

    if synchronize_before:
        timer.synchronize()
    started = perf_counter()
    # Force a complete fresh call at the same initial geometry, bypassing ASE cache.
    calculator.calculate(
        atoms, properties=["energy", "forces"], system_changes=all_changes
    )
    timer.synchronize()
    return {
        "wall_ms": (perf_counter() - started) * 1000,
        "energy_eV": float(calculator.results["energy"]),
        "forces": calculator.results["forces"].copy(),
        "stage_timings_ms": dict(
            calculator.results["polar_diagnostics"].get("timings_ms", {})
        ),
    }


def unchanged(actual, reference):
    import numpy as np

    energy = abs(actual["energy_eV"] - reference["energy_eV"])
    force = float(np.max(np.abs(actual["forces"] - reference["forces"])))
    # These FP64 instrumentation checks are stricter than the physics oracle;
    # they are not a replacement for original-model correctness qualification.
    if not np.isfinite([energy, force]).all() or energy > 1e-8 or force > 1e-10:
        raise AssertionError(
            f"Instrumentation changed E/F: energy={energy}, force={force}"
        )
    return {
        "energy_max_abs_eV": energy,
        "force_max_abs_eV_A": force,
        "energy_atol_eV": 1e-8,
        "force_atol_eV_A": 1e-10,
        "bitwise_identical": energy == 0.0
        and np.array_equal(actual["forces"], reference["forces"]),
    }


def distribution(values):
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend")
    parser.add_argument("--bridge", choices=("host", "dlpack"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Bracket one normal profile=False call with cudaProfilerStart/Stop before installing module hooks",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.calls < 1:
        parser.error("--warmup and --calls must be positive")
    if args.output.exists():
        parser.error("--output already exists; select a fresh diagnostic output")
    if args.cuda_profiler_range and not args.device.startswith("cuda"):
        parser.error("--cuda-profiler-range requires a CUDA device")
    cpu_policy = pin_one_cpu()
    import numpy as np
    import torch
    from ase.io import read
    import symmetrix
    from symmetrix.polar import SymmetrixPolar

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    atoms = read(args.input)
    calculator = SymmetrixPolar(
        args.checkpoint,
        device=args.device,
        backend=args.backend,
        bridge=args.bridge,
        diagnostics=False,
        profile=False,
    )
    timer = ForwardTimer(torch, calculator.device)
    for _ in range(args.warmup):
        calculate(calculator, atoms, timer)
    baseline = calculate(calculator, atoms, timer)
    trace_parity = None
    if args.cuda_profiler_range:
        # The initial synchronization is outside the capture. Inside it, only
        # the normal calculation and final completion synchronization run.
        timer.synchronize()
        with torch.cuda.profiler.profile():
            traced = calculate(calculator, atoms, timer, synchronize_before=False)
        trace_parity = unchanged(traced, baseline)
    calls = []
    calculator.profile = True
    with instrument_forward(calculator.model, timer) as expected:
        for index in range(args.calls):
            timer.reset()
            result = calculate(calculator, atoms, timer)
            parity = unchanged(result, baseline)
            regions = timer.regions()
            observed = {name: values["calls"] for name, values in regions.items()}
            if observed != expected:
                raise AssertionError(
                    f"Unexpected S forward call counts: {observed}; expected {expected}"
                )
            lr = sum(
                value["exclusive_ms"]
                for name, value in regions.items()
                if name.startswith("electric_potential_descriptor.")
                or name == "coulomb_energy"
            )
            calls.append(
                {
                    "call": index,
                    "wall_ms": result["wall_ms"],
                    "stage_timings_ms": result["stage_timings_ms"],
                    "forward_regions": regions,
                    "long_range_electrostatic_forward_ms": lr,
                    "whole_suffix_autograd_ms": result["stage_timings_ms"][
                        "torch_autograd"
                    ],
                    "suffix_forward_unattributed_ms": result["stage_timings_ms"][
                        "torch_suffix_forward"
                    ]
                    - sum(value["exclusive_ms"] for value in regions.values()),
                    "instrumentation_parity": parity,
                }
            )
    calculator.profile = False
    restored = calculate(calculator, atoms, timer)
    restored_parity = unchanged(restored, baseline)
    native = symmetrix.load_backend(args.backend)
    extension = Path(native.__file__)
    actual_affinity = sorted(os.sched_getaffinity(0))
    if actual_affinity != cpu_policy["actual_affinity"]:
        raise RuntimeError("Numerical runtime changed the selected single-CPU affinity")
    record = {
        "command": "polar_module_profile",
        "passed": True,
        "precision": "float64",
        "scope": "synchronized single-call diagnostics; excluded from formal throughput",
        "warmup_calls": args.warmup,
        "diagnostic_calls": args.calls,
        "python": sys.executable,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "checkpoint_sha256": calculator.checkpoint_sha256,
        "atoms": len(atoms),
        "species_counts": {
            str(z): int(np.sum(atoms.numbers == z)) for z in np.unique(atoms.numbers)
        },
        "initial_positions_sha256": hashlib.sha256(
            atoms.positions.astype("<f8").tobytes()
        ).hexdigest(),
        "cpu_policy": cpu_policy,
        "actual_cpu_affinity": actual_affinity,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "KOKKOS_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "OMP_MAX_ACTIVE_LEVELS",
            )
        },
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "device": str(calculator.device),
        "bridge": calculator.bridge,
        "native_build_info": calculator.native_build_info,
        "native_device_environment": native._execution_device_execution_environment(),
        "native_openmp_runtime": native._openmp_runtime_info()
        if calculator.device.type == "cpu"
        else None,
        "cutoff_A": calculator.cutoff,
        "k_vectors_count": calculator.results["polar_diagnostics"]["k_vectors_count"],
        "feature_shapes": calculator.results["polar_diagnostics"]["feature_shapes"],
        "extension_path": str(extension),
        "extension_sha256": sha256(extension),
        "script_sha256": sha256(__file__),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "e3nn",
                "mace-torch",
                "graph-longrange",
                "symmetrix-xl",
            )
        },
        "expected_forward_calls_per_calculation": expected,
        "baseline_uninstrumented_wall_ms": baseline["wall_ms"],
        "restored_uninstrumented_wall_ms": restored["wall_ms"],
        "restored_instrumentation_parity": restored_parity,
        "cuda_profiler_range": {
            "requested": args.cuda_profiler_range,
            "scope": "one profile=False, hook-free call plus final synchronization; before module instrumentation",
            "parity": trace_parity,
        },
        "calls": calls,
        "summary": {
            "stage_timings_ms": {
                name: distribution([call["stage_timings_ms"][name] for call in calls])
                for name in calls[0]["stage_timings_ms"]
            },
            "long_range_electrostatic_forward_ms": distribution(
                [call["long_range_electrostatic_forward_ms"] for call in calls]
            ),
            "whole_suffix_autograd_ms": distribution(
                [call["whole_suffix_autograd_ms"] for call in calls]
            ),
            "forward_regions": {
                name: {
                    key: distribution(
                        [call["forward_regions"][name][key] for call in calls]
                    )
                    for key in ("inclusive_ms", "exclusive_ms")
                }
                for name in expected
            },
        },
        "interpretation": {
            "module_times": "Forward-only. Inclusive regions can overlap; exclusive times subtract instrumented children. GPU synchronized before/after each region; Python/hooks/synchronization overhead is included.",
            "long_range_forward": "Selected regions: descriptor geometry + four dynamic fields (two updates x two spins) + Coulomb energy. Reciprocal-vector enumeration, normalization and equilibration outside these modules stay in suffix_forward_unattributed_ms; field-update MLPs, local electron energy and local readout are listed separately.",
            "autograd": "Whole Torch suffix gradient stage, including local, response and long-range derivatives. Long-range backward is not independently measured.",
            "comparison": "Before/after uninstrumented walls are controls, not throughput or a speedup estimate.",
            "graph": "ASE neighbor creation plus Torch graph tensors, including host-to-device copies on CUDA.",
            "native_forward": "Native input staging/copies plus local forward kernels and the stage fence; not kernel-only.",
            "feature_export": "DLPack: native device snapshot/copy and synchronization; host bridge: native host snapshot and Torch tensor staging. Not zero-copy.",
            "native_vjp": "Adjoint import/copies, bridge synchronization and native local VJP; host bridge includes downloading Torch seeds. Not kernel-only.",
            "force_result_assembly": "Native edge-force export snapshot/readback after native_vjp, permutation/reduction and final host result transfers.",
        },
    }
    del calculator
    gc.collect()
    if native._kokkos_live_object_count():
        raise RuntimeError("Diagnostic retained a native owner after restoration")
    native._finalize_kokkos()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "passed": True}))


if __name__ == "__main__":
    main()
