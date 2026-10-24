#!/usr/bin/env python3
"""Prepare, qualify and time charged periodic Li/water MACE-POLAR calculations.

Each box contains exactly one Li and N complete waters. No charged box is
repeated to create a larger system. Outputs are written only under --output.
The original and candidate timing workers always run in separate processes.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path


def _early_execution_environment():
    # Set affinity before NumPy, BLAS, PyTorch or Kokkos can create workers.
    early = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    early.add_argument("--cpu-threads", type=int, default=1)
    early.add_argument("--device", default="cpu")
    options, _ = early.parse_known_args()
    threads = options.cpu_threads
    if threads < 1 or (
        threads != 1 and (sys.argv[1:2] != ["md"] or options.device != "cpu")
    ):
        raise ValueError(
            "--cpu-threads >1 is supported only for CPU md correctness runs"
        )
    record = {
        "policy": "one logical CPU per physical core; SMT siblings excluded",
        "requested_threads": threads,
        "formal_benchmark_threads": 1,
        "blas_policy": "thread environment set to one; OpenMP BLAS shares the OpenMP team outside worker regions; nested parallelism disabled",
    }
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        selected = allowed[0]
        chosen, physical = [], set()
        for logical in allowed:
            directory = Path(f"/sys/devices/system/cpu/cpu{logical}/topology")
            package = directory / "physical_package_id"
            core = directory / "core_id"
            if not (package.is_file() and core.is_file()):
                if threads > 1:
                    raise RuntimeError(
                        "CPU MD threads require verified physical-core topology"
                    )
                chosen = [selected]
                break
            identity = (package.read_text().strip(), core.read_text().strip())
            if identity not in physical:
                chosen.append(logical)
                physical.add(identity)
            if len(chosen) == threads:
                break
        if len(chosen) != threads:
            raise RuntimeError(
                f"Requested {threads} CPU MD cores, allocation has {len(chosen)}"
            )
        os.sched_setaffinity(0, set(chosen))
        topology = Path(f"/sys/devices/system/cpu/cpu{selected}/topology")
        record.update(
            applied=True,
            allowed_before=allowed,
            selected_logical_cpu=selected,
            selected_logical_cpus=chosen,
            selected_physical_cores=sorted(physical),
            allowed_after=sorted(os.sched_getaffinity(0)),
            physical_topology={
                name: (topology / name).read_text().strip()
                if (topology / name).is_file()
                else None
                for name in ("physical_package_id", "core_id", "thread_siblings_list")
            },
        )
    else:
        if threads > 1:
            raise RuntimeError("CPU MD threads require OS affinity support")
        record.update(
            applied=False,
            reason="OS lacks sched_setaffinity; not a qualified Linux CPU timing",
        )
    for name in (
        "OMP_NUM_THREADS",
        "KOKKOS_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["KOKKOS_NUM_THREADS"] = str(threads)
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    return record


EARLY_EXECUTION_ENVIRONMENT = _early_execution_environment()

import numpy as np  # noqa: E402 -- pin CPU affinity before importing numerical runtimes

ROOT = Path(__file__).resolve().parents[1]
FORMAL_WATER_COUNTS = (64, 128, 256, 512)
STATE = {"charge": 1.0, "spin": 1.0, "external_field": [0.0, 0.0, 0.0]}


def clean(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(record), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def command_output(arguments):
    result = subprocess.run(
        arguments, text=True, capture_output=True, check=False, timeout=15
    )
    return result.stdout.strip() if result.returncode == 0 else None


def apply_state(atoms):
    atoms.info.update(STATE)
    return atoms


def inspect_structure(atoms):
    numbers = atoms.numbers
    counts = {int(z): int(np.sum(numbers == z)) for z in np.unique(numbers)}
    waters = counts.get(8, 0)
    if counts != {1: 2 * waters, 3: 1, 8: waters} or waters < 1:
        raise ValueError(f"Expected exactly one Li and N waters, found {counts}")
    if not np.all(atoms.pbc) or atoms.get_volume() <= 0:
        raise ValueError(
            "Li/water qualification requires full 3D PBC and positive volume"
        )
    state = {key: clean(atoms.info.get(key)) for key in STATE}
    if state != STATE:
        raise ValueError(f"Expected Q=+1, spin multiplicity 1, zero field; got {state}")
    return {
        "atoms": len(atoms),
        "waters": waters,
        "lithium_atoms": 1,
        "cell_A": atoms.cell.array,
        "volume_A3": atoms.get_volume(),
        "water_density_g_cm3": waters
        * 18.01528
        / 6.02214076e23
        * 1e24
        / atoms.get_volume(),
        "polar_state": state,
    }


def load_structure(path):
    from ase.io import read

    atoms = read(path)
    inspect_structure(atoms)
    return atoms


def _rotation(rng):
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    matrix[:, 0] *= np.linalg.det(matrix)
    return matrix


def initial_box(waters, seed, density=1.0):
    from ase import Atoms

    if waters < 1 or density <= 0:
        raise ValueError("Water count and density must be positive")
    rng = np.random.default_rng(seed)
    length = (waters * 18.01528 / 6.02214076e23 * 1e24 / density) ** (1 / 3)
    side = math.ceil(waters ** (1 / 3) - 1e-12)
    while True:
        coordinates = (np.indices((side,) * 3).reshape(3, -1).T + 0.5) * length / side
        center = np.full(3, length / 2)
        eligible = coordinates[np.linalg.norm(coordinates - center, axis=1) >= 1.8]
        if len(eligible) >= waters:
            break
        side += 1
    # A shuffled lattice avoids the strong spatial bias of taking its first N sites.
    centers = eligible[rng.permutation(len(eligible))[:waters]]
    positions, numbers = [center], [3]
    water = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.927297, 0.0]])
    for oxygen in centers:
        for _ in range(2000):
            proposal = water @ _rotation(rng) + oxygen
            differences = proposal[:, None, :] - np.asarray(positions)[None, :, :]
            differences -= length * np.rint(differences / length)
            distances = np.linalg.norm(differences, axis=-1)
            minimum = np.full_like(distances, 0.85)
            minimum[0, :] = 1.35
            minimum[:, 0] = [1.7, 1.1, 1.1]
            if np.all(distances > minimum):
                positions.extend(proposal)
                numbers.extend([8, 1, 1])
                break
        else:
            raise RuntimeError(
                "Could not orient nonoverlapping water; change the preparation seed"
            )
    atoms = apply_state(
        Atoms(numbers=numbers, positions=positions, cell=[length] * 3, pbc=True)
    )
    atoms.set_array("water_id", np.concatenate(([-1], np.repeat(np.arange(waters), 3))))
    atoms.wrap()
    inspect_structure(atoms)
    return atoms


def velocities(atoms, temperature, seed):
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=temperature,
        force_temp=True,
        rng=np.random.default_rng(seed),
    )
    Stationary(atoms, preserve_temperature=True)


def state_hashes(atoms):
    """Hash the actual state arrays, including shape and a fixed byte order."""

    def array_hash(values):
        array = np.ascontiguousarray(values, dtype="<f8")
        digest = hashlib.sha256(str(array.shape).encode() + array.tobytes())
        return digest.hexdigest()

    has_momenta = atoms.has("momenta")
    return {
        "positions_sha256": array_hash(atoms.positions),
        "cell_sha256": array_hash(atoms.cell.array),
        "velocities_sha256": array_hash(atoms.get_velocities())
        if has_momenta
        else None,
        "momenta_sha256": array_hash(atoms.get_momenta()) if has_momenta else None,
        "momenta_present": has_momenta,
        "temperature_K": atoms.get_temperature() if has_momenta else None,
        "hash_schema": "shape text followed by C-order little-endian float64 bytes; ASE native units",
    }


def preserve_or_initialize_velocities(atoms, temperature, seed):
    if atoms.has("momenta"):
        momenta = atoms.get_momenta()
        if not np.isfinite(momenta).all():
            raise ValueError("Prepared momenta contain non-finite values")
        policy = {"policy": "preserve_prepared_momenta", "seed_used": None}
    else:
        velocities(atoms, temperature, seed)
        policy = {
            "policy": "deterministic_maxwell_fallback_when_momenta_absent",
            "seed_used": seed,
            "temperature_requested_K": temperature,
        }
    return policy | {"actual_state": state_hashes(atoms)}


def make_calculator(args, executor, capture=False):
    import torch

    if not args.checkpoint or not Path(args.checkpoint).is_file():
        raise FileNotFoundError("--checkpoint must name a local MACE-POLAR-1-S model")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this PyTorch process cannot use CUDA"
        )
    if executor == "original":
        from mace.calculators.foundations_models import mace_polar
        from symmetrix._polar_compat import repair_polar_checkpoint

        calculator = mace_polar(
            model=str(Path(args.checkpoint).resolve()),
            device=args.device,
            default_dtype="float64",
            pbc_handling="pbc",
            enable_cueq=False,
        )
        calculator._li_water_compat = [
            repair_polar_checkpoint(model, pbc_handling="pbc")
            for model in calculator.models
        ]
        for model in calculator.models:
            if type(model).__name__ != "PolarMACE":
                raise RuntimeError(
                    f"Original executor loaded {type(model).__name__}, not PolarMACE"
                )
            if any(
                parameter.dtype != torch.float64 for parameter in model.parameters()
            ):
                raise RuntimeError("Original executor is not wholly FP64")
            if any(
                parameter.device.type != torch.device(args.device).type
                for parameter in model.parameters()
            ):
                raise RuntimeError(
                    "Original executor used a device different from --device"
                )
        if capture:
            _observe_original(calculator)
    else:
        from symmetrix.polar import SymmetrixPolar

        calculator = SymmetrixPolar(
            args.checkpoint,
            device=args.device,
            bridge=args.bridge,
            precision="float64",
            diagnostics=capture,
        )
    return calculator


def _observe_original(calculator):
    """Observe original normalized populations without replacing any forward."""
    observation = {"populations": [], "output": {}}

    def before_update(module, positional, keywords):
        density = (
            keywords["local_charges"]
            .detach()
            .reshape(-1, 2, calculator.models[0].charges_irreps.dim // 2)
        )
        observation["populations"].append(density[:, :, 0].sum(dim=0).cpu().numpy())

    def after_model(module, positional, output):
        for key in (
            "charges",
            "spins",
            "spin_charge_density",
            "density_coefficients",
            "electrostatic_energy",
            "electron_energy",
        ):
            value = output.get(key)
            if value is not None:
                observation["output"][key] = value.detach().cpu().numpy().copy()
        density = output["spin_charge_density"].detach()
        observation["populations"].append(density[:, :, 0].sum(dim=0).cpu().numpy())

    handles = []
    for model in calculator.models:
        handles.extend(
            block.register_forward_pre_hook(before_update, with_kwargs=True)
            for block in model.field_dependent_charges_maps
        )
        handles.append(model.register_forward_hook(after_model))
    calculator._li_water_observation = observation
    calculator._li_water_hooks = handles


def evaluate(calculator, atoms):
    observation = getattr(calculator, "_li_water_observation", None)
    if observation is not None:
        observation["populations"].clear()
        observation["output"].clear()
    calculator.calculate(atoms.copy(), properties=["energy", "forces"])
    result = dict(calculator.results)
    if observation is not None:
        result.update(observation["output"])
        result["normalized_spin_charge_totals"] = observation["populations"].copy()
    else:
        diagnostic = result["polar_diagnostics"]
        if (
            diagnostic["backbone"] != "native_feature_forward"
            or diagnostic["prefix_fallback"]
        ):
            raise RuntimeError("Candidate did not execute the native feature/VJP path")
        result["normalized_spin_charge_totals"] = diagnostic[
            "normalized_spin_charge_totals"
        ]
    energy = float(np.asarray(result["energy"]).reshape(-1)[0])
    force = np.asarray(result["forces"])
    if (
        not math.isfinite(energy)
        or force.shape != (len(atoms), 3)
        or not np.isfinite(force).all()
    ):
        raise FloatingPointError("Non-finite or malformed complete model output")
    result["energy"] = energy
    result["forces"] = force
    return result


def synchronize(device):
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def hardware_provenance(device):
    selected_cpu = EARLY_EXECUTION_ENVIRONMENT.get("selected_logical_cpu")
    cpu = {"selected_logical_cpu": selected_cpu, "source": "/proc/cpuinfo"}
    path = Path("/proc/cpuinfo")
    if path.is_file():
        records = []
        for block in path.read_text(errors="replace").split("\n\n"):
            fields = {}
            for line in block.splitlines():
                key, separator, value = line.partition(":")
                if separator:
                    fields[key.strip()] = value.strip()
            if fields:
                records.append(fields)
        selected = next(
            (
                record
                for record in records
                if record.get("processor") == str(selected_cpu)
            ),
            None,
        )
        if selected is not None:
            cpu.update(
                {
                    name: selected.get(name)
                    for name in (
                        "processor",
                        "vendor_id",
                        "model name",
                        "cpu family",
                        "model",
                        "stepping",
                        "physical id",
                        "core id",
                    )
                }
            )
            cpu["available"] = True
        else:
            cpu.update(
                available=False,
                reason="Selected logical CPU has no /proc/cpuinfo stanza",
            )
    else:
        cpu.update(available=False, reason="/proc/cpuinfo is unavailable")
    result = {
        "nodename": platform.node(),
        "cpu": cpu,
        "slurm": {
            name: os.environ.get(name)
            for name in (
                "SLURM_JOB_ID",
                "SLURM_STEP_ID",
                "SLURMD_NODENAME",
                "SLURM_JOB_NODELIST",
                "SLURM_NODEID",
                "SLURM_PROCID",
                "SLURM_LOCALID",
            )
        },
        "gpu": {
            "queried": False,
            "reason": "CPU execution; CUDA device APIs were not queried",
        },
    }
    if device and device.startswith("cuda"):
        import torch

        current = torch.cuda.current_device()
        ordinal = torch.device(device).index
        ordinal = current if ordinal is None else ordinal
        properties = torch.cuda.get_device_properties(ordinal)
        uuid = getattr(properties, "uuid", None)
        result["gpu"] = {
            "queried": True,
            "source": "torch.cuda.get_device_properties",
            "device_name": properties.name,
            "device_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "uuid": str(uuid) if uuid is not None else None,
            "device_ordinal": ordinal,
            "current_visible_ordinal": current,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_build_version": torch.version.cuda,
            "visible_ordinals_are_not_physical_gpu_indices": True,
        }
    return result


def provenance(args, calculator=None, executor=None):
    versions = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "")
        if name.lower().replace("_", "-") in (
            "torch",
            "numpy",
            "ase",
            "mace-torch",
            "e3nn",
            "graph-longrange",
        ) or name.lower().startswith("symmetrix"):
            versions[name] = distribution.version
    sources = {}
    for name in (
        "mace.modules.extensions",
        "mace.modules.field_blocks",
        "graph_longrange.features",
        "symmetrix.polar",
        "symmetrix._polar_compat",
    ):
        with contextlib.suppress(ImportError):
            module = importlib.import_module(name)
            path = getattr(module, "__file__", None)
            if path and Path(path).is_file():
                sources[name] = {"path": path, "sha256": sha256(path)}
    fingerprint = hashlib.sha256(
        json.dumps({"versions": versions, "sources": sources}, sort_keys=True).encode()
    ).hexdigest()
    result = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hardware": hardware_provenance(getattr(args, "device", None)),
        "pid": os.getpid(),
        "early_execution_environment": EARLY_EXECUTION_ENVIRONMENT,
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "git_sha": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_dirty": command_output(["git", "-C", str(ROOT), "status", "--porcelain"]),
        "script_sha256": sha256(__file__),
        "packages": versions,
        "source_modules": sources,
        "package_fingerprint": fingerprint,
        "checkpoint": str(Path(args.checkpoint).resolve())
        if getattr(args, "checkpoint", None)
        else None,
        "checkpoint_sha256": sha256(args.checkpoint)
        if getattr(args, "checkpoint", None)
        else None,
        "device_requested": getattr(args, "device", None),
        "precision": "float64",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
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
    }
    sys.path.insert(0, str(ROOT))
    try:
        from tools._symmetrix_build.source_provenance import (
            symmetrix_native_source_fingerprint,
            symmetrix_source_fingerprint,
        )

        result["tree_source_content_sha256"] = symmetrix_source_fingerprint(ROOT)
        result["tree_native_source_content_sha256"] = (
            symmetrix_native_source_fingerprint(ROOT)
        )
    except ImportError as error:
        result["tree_fingerprint_error"] = repr(error)
    finally:
        sys.path.remove(str(ROOT))
    if calculator is not None:
        result["actual_executor"] = {
            "request": executor,
            "calculator_class": f"{type(calculator).__module__}.{type(calculator).__name__}",
        }
        if executor == "candidate":
            import symmetrix

            native = symmetrix.load_backend()
            path = native.__file__
            space = native._kokkos_default_execution_space()
            if (space == "Cuda") != args.device.startswith("cuda"):
                raise RuntimeError(
                    f"Requested device {args.device} conflicts with actual Kokkos {space}"
                )
            result["actual_executor"].update(
                backbone="native_feature_forward",
                bridge=args.bridge,
                native_class=f"{type(calculator.evaluator).__module__}.{type(calculator.evaluator).__name__}",
                kokkos_execution_space=space,
                extension_path=path,
                extension_sha256=sha256(path),
                native_build_info=calculator.native_build_info,
            )
            if space == "OpenMP":
                runtime = dict(native._openmp_runtime_info())
                result["actual_executor"]["openmp_runtime"] = runtime
                if (
                    runtime.get("kokkos_concurrency")
                    != EARLY_EXECUTION_ENVIRONMENT["requested_threads"]
                ):
                    raise RuntimeError(
                        "Native concurrency differs from the recorded CPU policy"
                    )
        else:
            result["actual_executor"].update(
                model_classes=[
                    f"{type(model).__module__}.{type(model).__name__}"
                    for model in calculator.models
                ],
                devices=sorted(
                    {
                        str(parameter.device)
                        for model in calculator.models
                        for parameter in model.parameters()
                    }
                ),
                compatibility=calculator._li_water_compat,
            )
    return result


def conserved_populations(result, tolerance=1e-9):
    totals = np.asarray(result["normalized_spin_charge_totals"], dtype=float)
    expected = np.full((3, 2), 0.5)
    if totals.shape != expected.shape:
        raise RuntimeError(
            f"Expected initial plus two normalized populations, got {totals.shape}"
        )
    error = float(np.max(np.abs(totals - expected)))
    charge_error = float(np.max(np.abs(totals.sum(axis=1) - 1.0)))
    spin_error = float(np.max(np.abs(totals[:, 0] - totals[:, 1])))
    return {
        "passed": bool(
            np.isfinite(totals).all()
            and charge_error <= tolerance
            and spin_error <= tolerance
        ),
        "alpha_beta_totals": totals,
        "charge_totals": totals.sum(axis=1),
        "spin_populations": totals[:, 0] - totals[:, 1],
        "maximum_population_error": error,
        "maximum_charge_error_e": charge_error,
        "maximum_spin_population_error": spin_error,
        "charge_and_spin_tolerance": tolerance,
    }


def regional_force_errors(atoms, delta, tolerance, shell_radius=3.2):
    lithium = int(np.flatnonzero(atoms.numbers == 3)[0])
    labels = atoms.arrays.get("water_id")
    if labels is None:
        # Files produced before water_id was added retain this preparation order.
        water_atoms = np.flatnonzero(atoms.numbers != 3)
        if len(water_atoms) % 3 or not np.all(
            atoms.numbers[water_atoms].reshape(-1, 3) == [8, 1, 1]
        ):
            raise ValueError(
                "Region analysis needs water_id labels or the prepared O,H,H molecule ordering"
            )
        labels = np.full(len(atoms), -1, dtype=int)
        labels[water_atoms] = np.repeat(np.arange(len(water_atoms) // 3), 3)
    labels = np.asarray(labels)
    shell_ids, other_ids, oxygen_distances = [], [], {}
    for identifier in sorted(set(labels[atoms.numbers != 3].tolist())):
        members = np.flatnonzero(labels == identifier)
        if len(members) != 3 or sorted(atoms.numbers[members].tolist()) != [1, 1, 8]:
            raise ValueError(f"Invalid whole-water identity {identifier}")
        oxygen = int(members[atoms.numbers[members] == 8][0])
        distance = float(atoms.get_distance(lithium, oxygen, mic=True))
        oxygen_distances[int(identifier)] = distance
        (shell_ids if distance <= shell_radius else other_ids).append(identifier)
    regions = {
        "Li": (np.array([lithium]), []),
        "hydration_shell_water": (
            np.flatnonzero(np.isin(labels, shell_ids)),
            shell_ids,
        ),
        "remaining_water": (np.flatnonzero(np.isin(labels, other_ids)), other_ids),
    }
    result = {}
    for name, (indices, water_ids) in regions.items():
        values = delta[indices]
        maximum = float(np.max(np.abs(values))) if len(indices) else None
        result[name] = {
            "atom_count": len(indices),
            "water_count": len(water_ids),
            "atom_indices": indices,
            "water_ids": water_ids,
            "maximum_force_component_error_eV_A": maximum,
            "maximum_force_vector_error_eV_A": float(
                np.linalg.norm(values, axis=1).max()
            )
            if len(indices)
            else None,
            "rms_force_component_error_eV_A": float(np.sqrt(np.mean(values * values)))
            if len(indices)
            else None,
            "passed": maximum is None or maximum <= tolerance,
        }
    result["definition"] = {
        "shell_radius_A": shell_radius,
        "oxygen_Li_distance": "minimum image",
        "selection": "all three atoms of each water whose O is within radius",
        "water_oxygen_distances_A": oxygen_distances,
    }
    return result


def compare_results(original, candidate, atoms, args):
    delta = candidate["forces"] - original["forces"]
    lithium = int(np.flatnonzero(atoms.numbers == 3)[0])
    original_conservation, candidate_conservation = (
        conserved_populations(original),
        conserved_populations(candidate),
    )
    energy_error = abs(candidate["energy"] - original["energy"])
    force_error = float(np.max(np.abs(delta)))
    return {
        "passed": energy_error <= args.energy_tolerance
        and force_error <= args.force_tolerance
        and original_conservation["passed"]
        and candidate_conservation["passed"],
        "original_energy_eV": original["energy"],
        "candidate_energy_eV": candidate["energy"],
        "absolute_total_energy_error_eV": energy_error,
        "maximum_force_component_error_eV_A": force_error,
        "maximum_force_vector_error_eV_A": float(np.linalg.norm(delta, axis=1).max()),
        "Li_force_original_eV_A": original["forces"][lithium],
        "Li_force_candidate_eV_A": candidate["forces"][lithium],
        "Li_force_vector_error_eV_A": float(np.linalg.norm(delta[lithium])),
        "force_error_by_region": regional_force_errors(
            atoms, delta, args.force_tolerance
        ),
        "original_populations": original_conservation,
        "candidate_populations": candidate_conservation,
        "candidate_diagnostics": candidate["polar_diagnostics"],
    }


def prepare(args):
    from ase import units
    from ase.io import write
    from ase.md.langevin import Langevin
    from ase.optimize import BFGS, LBFGS

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.input and len(args.waters) != 1:
        raise ValueError("--input can resume only one water count per invocation")
    calculator = None if args.stage == "initial" else make_calculator(args, "original")
    for waters in args.waters:
        prefix = output / f"water{waters:04d}"
        manifest = {
            "stage": args.stage,
            "seed": args.seed,
            "temperature_K": args.temperature,
            "density_requested_g_cm3": args.density,
            "formal_water_count": waters in FORMAL_WATER_COUNTS,
            "status": "running",
            "provenance": provenance(args, calculator, "original"),
        }
        try:
            if args.input:
                atoms = load_structure(args.input)
                if inspect_structure(atoms)["waters"] != waters:
                    raise ValueError(
                        "--waters differs from the resumed input structure"
                    )
            elif args.stage == "equilibrate":
                prior = json.loads(prefix.with_suffix(".prepare.json").read_text())
                if not prior.get("optimization", {}).get("converged"):
                    raise RuntimeError(
                        "Equilibration requires an explicitly converged optimization"
                    )
                if (
                    prior["provenance"]["checkpoint_sha256"]
                    != manifest["provenance"]["checkpoint_sha256"]
                ):
                    raise RuntimeError(
                        "Optimization and equilibration checkpoints differ"
                    )
                atoms = load_structure(prefix.with_suffix(".optimized.extxyz"))
                manifest["optimization"] = prior["optimization"]
            else:
                atoms = initial_box(waters, args.seed + waters, args.density)
                write(prefix.with_suffix(".initial.extxyz"), atoms)
            manifest["structure"] = inspect_structure(atoms)
            if args.stage == "initial":
                write(prefix.with_suffix(".initial.extxyz"), atoms)
            if args.stage in ("optimize", "all"):
                atoms.calc = calculator
                selected_optimizer = (
                    ("lbfgs" if waters >= 256 else "bfgs")
                    if args.optimizer == "auto"
                    else args.optimizer
                )
                optimizer_class = LBFGS if selected_optimizer == "lbfgs" else BFGS
                optimizer = optimizer_class(
                    atoms,
                    logfile=str(prefix.with_suffix(f".{selected_optimizer}.log")),
                    trajectory=str(prefix.with_suffix(f".{selected_optimizer}.traj")),
                )
                converged = bool(
                    optimizer.run(fmax=args.fmax, steps=args.max_opt_steps)
                )
                fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
                manifest["optimization"] = {
                    "optimizer_requested": args.optimizer,
                    "optimizer": selected_optimizer,
                    "optimizer_class": f"{optimizer_class.__module__}.{optimizer_class.__name__}",
                    "converged": converged,
                    "steps": optimizer.nsteps,
                    "maximum_force_eV_A": fmax,
                    "fmax_target_eV_A": args.fmax,
                    "fixed_cell": True,
                }
                path = prefix.with_suffix(
                    ".optimized.extxyz" if converged else ".unconverged.extxyz"
                )
                write(path, atoms)
                if not converged:
                    raise RuntimeError(
                        f"{selected_optimizer.upper()} did not converge in {args.max_opt_steps} steps; fmax={fmax}"
                    )
            if args.stage in ("equilibrate", "all"):
                if args.input and not manifest.get("optimization"):
                    # Explicit resumed inputs are verified by their actual force.
                    atoms.calc = calculator
                    fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
                    if fmax > args.fmax:
                        raise RuntimeError(
                            f"Resumed equilibration input is not force converged: {fmax}"
                        )
                    manifest["optimization"] = {
                        "converged": True,
                        "maximum_force_eV_A": fmax,
                        "verified_resumed_input": True,
                    }
                atoms.calc = calculator
                velocities(atoms, args.temperature, args.seed + waters)
                nvt_initial_state = state_hashes(atoms)
                dynamics = Langevin(
                    atoms,
                    args.timestep * units.fs,
                    temperature_K=args.temperature,
                    friction=0.01 / units.fs,
                    rng=np.random.default_rng(args.seed + waters + 1),
                    logfile=str(prefix.with_suffix(".nvt.log")),
                    loginterval=10,
                    trajectory=str(prefix.with_suffix(".nvt.traj")),
                )
                dynamics.run(args.nvt_steps)
                synchronize(args.device)
                prepared_path = prefix.with_suffix(".equilibrated.extxyz")
                final_state = state_hashes(atoms)
                write(prepared_path, atoms)
                manifest["nvt"] = {
                    "steps": args.nvt_steps,
                    "timestep_fs": args.timestep,
                    "temperature_target_K": args.temperature,
                    "final_temperature_K": atoms.get_temperature(),
                    "friction_fs_inverse": 0.01,
                    "initial_state": nvt_initial_state,
                    "final_state_in_memory": final_state,
                    "prepared_output_state": state_hashes(
                        load_structure(prepared_path)
                    ),
                    "prepared_output_sha256": sha256(prepared_path),
                }
            manifest["status"] = "passed"
        except Exception as error:
            manifest.update(status="failed", error=repr(error))
            raise
        finally:
            write_json(prefix.with_suffix(".prepare.json"), manifest)


def validate(args):
    atoms = load_structure(args.input)
    original, candidate = (
        make_calculator(args, "original", True),
        make_calculator(args, "candidate", True),
    )
    frames = {"base": atoms.copy(), "translated": atoms.copy(), "wrapped": atoms.copy()}
    frames["translated"].positions += (
        np.array([0.173, -0.291, 0.137]) @ atoms.cell.array
    )
    frames["wrapped"].positions[::2] += atoms.cell[0] - 2 * atoms.cell[1]
    frames["wrapped"].wrap()
    report = {
        "command": "validate",
        "structure": inspect_structure(atoms),
        "input_sha256": sha256(args.input),
        "thresholds": {
            "total_energy_eV": args.energy_tolerance,
            "force_component_eV_A": args.force_tolerance,
        },
        "original": provenance(args, original, "original"),
        "candidate": provenance(args, candidate, "candidate"),
        "cases": {},
    }
    baseline = None
    for name, frame in frames.items():
        expected, actual = evaluate(original, frame), evaluate(candidate, frame)
        case = compare_results(expected, actual, frame, args)
        if baseline is None:
            baseline = expected
        else:
            case["original_invariance_energy_error_eV"] = abs(
                expected["energy"] - baseline["energy"]
            )
            case["original_invariance_force_error_eV_A"] = float(
                np.max(np.abs(expected["forces"] - baseline["forces"]))
            )
            case["passed"] &= (
                case["original_invariance_energy_error_eV"] <= args.energy_tolerance
                and case["original_invariance_force_error_eV_A"] <= args.force_tolerance
            )
        report["cases"][name] = case
    report["passed"] = all(case["passed"] for case in report["cases"].values())
    write_json(Path(args.output) / "validation.json", report)
    if not report["passed"]:
        raise RuntimeError(
            "Complete POLAR parity/invariance qualification failed; see validation.json"
        )


def md(args):
    from ase import units
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io.trajectory import Trajectory
    from ase.md.verlet import VelocityVerlet

    atoms = load_structure(args.input)
    candidate, original = (
        make_calculator(args, "candidate", True),
        make_calculator(args, "original", True),
    )
    atoms.calc = candidate
    velocity_initialization = preserve_or_initialize_velocities(
        atoms, args.temperature, args.seed
    )
    dynamics = VelocityVerlet(atoms, args.timestep * units.fs)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "command": "md",
        "ensemble": "NVE",
        "same_trajectory_oracle": True,
        "structure": inspect_structure(atoms),
        "input_sha256": sha256(args.input),
        "velocity_initialization": velocity_initialization,
        "initial_state": state_hashes(atoms),
        "timestep_fs": args.timestep,
        "steps_requested": args.steps,
        "samples": [],
        "original": provenance(args, original, "original"),
        "candidate": provenance(args, candidate, "candidate"),
        "passed": False,
    }
    trajectory = Trajectory(str(output / "candidate-nve.traj"), "w")
    try:
        for step in range(args.steps + 1):
            if step:
                dynamics.run(1)
            if step % args.sample_interval and step != args.steps:
                continue
            # Both potentials are evaluated on exactly the same coordinates and momenta.
            actual = evaluate(candidate, atoms)
            expected = evaluate(original, atoms)
            sample = compare_results(expected, actual, atoms, args)
            sample.update(
                step=step,
                time_fs=step * args.timestep,
                kinetic_energy_eV=atoms.get_kinetic_energy(),
                total_energy_eV=actual["energy"] + atoms.get_kinetic_energy(),
                temperature_K=atoms.get_temperature(),
            )
            report["samples"].append(sample)
            frame = atoms.copy()
            frame.info.update(md_step=step, md_time_fs=step * args.timestep)
            frame.calc = SinglePointCalculator(
                frame, energy=actual["energy"], forces=actual["forces"]
            )
            trajectory.write(frame)
            write_json(output / "md.json", report)
            if not sample["passed"]:
                raise RuntimeError(
                    f"Same-trajectory original/candidate parity failed at MD step {step}"
                )
        energies = np.array([sample["total_energy_eV"] for sample in report["samples"]])
        times = np.array([sample["time_fs"] for sample in report["samples"]])
        report["drift"] = {
            "initial_total_energy_eV": energies[0],
            "final_total_energy_eV": energies[-1],
            "final_minus_initial_eV_per_atom": (energies[-1] - energies[0])
            / len(atoms),
            "maximum_absolute_eV_per_atom": float(
                np.max(np.abs(energies - energies[0]))
            )
            / len(atoms),
            "linear_eV_per_atom_ps": float(
                np.polyfit(times / 1000, energies / len(atoms), 1)[0]
            )
            if len(times) > 1
            else None,
        }
        report["passed"] = True
    finally:
        trajectory.close()
        write_json(output / "md.json", report)


class MemoryMonitor:
    """Poll total GPU process allocation, including Torch and Kokkos allocators."""

    def __init__(self, device, interval=0.2):
        self.device, self.interval = device, interval
        self.stop = threading.Event()
        self.process_samples, self.global_samples, self.errors = [], [], []
        self.competing_pids = set()
        self.thread = None

    def sample(self):
        if not self.device.startswith("cuda"):
            return
        try:
            text = command_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ]
            )
            if text is not None:
                rows = [line.split(",") for line in text.splitlines() if line.strip()]
                self.competing_pids.update(
                    int(pid.strip())
                    for pid, _ in rows
                    if int(pid.strip()) != os.getpid()
                )
                values = [
                    float(memory.strip())
                    for pid, memory in rows
                    if int(pid.strip()) == os.getpid()
                    and memory.strip() not in ("N/A", "[N/A]")
                ]
                if values:
                    self.process_samples.append(sum(values))
            text = command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ]
            )
            if text is not None:
                self.global_samples.append(
                    sum(float(value.strip()) for value in text.splitlines())
                )
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            if not self.errors:
                self.errors.append(repr(error))

    def run(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *ignored):
        self.sample()
        self.stop.set()
        self.thread.join(timeout=10)

    def record(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record = {
            "process_peak_rss_bytes": int(
                rss if sys.platform == "darwin" else rss * 1024
            ),
            "poll_interval_seconds": self.interval,
            "nvidia_process_samples": len(self.process_samples),
            "nvidia_process_peak_MiB": max(self.process_samples)
            if self.process_samples
            else None,
            "global_all_gpus_baseline_MiB": self.global_samples[0]
            if self.global_samples
            else None,
            "global_all_gpus_peak_MiB": max(self.global_samples)
            if self.global_samples
            else None,
            "global_scope": "all GPUs on host; includes competing processes",
            "errors": self.errors,
            "competing_compute_pids_observed": sorted(self.competing_pids),
            "poller": "background nvidia-smi subprocesses at the same interval for both executors",
        }
        if self.global_samples:
            record["global_all_gpus_peak_minus_baseline_MiB"] = (
                max(self.global_samples) - self.global_samples[0]
            )
        if self.device.startswith("cuda"):
            import torch

            record["torch_allocator_only_peak_bytes"] = torch.cuda.max_memory_allocated(
                self.device
            )
            record["torch_allocator_only_reserved_peak_bytes"] = (
                torch.cuda.max_memory_reserved(self.device)
            )
        return record


def benchmark_worker(args):
    from ase import units
    from ase.md.verlet import VelocityVerlet
    from ase.neighborlist import neighbor_list

    atoms = load_structure(args.input)
    with MemoryMonitor(args.device) as memory:
        calculator = make_calculator(args, args.executor)
        atoms.calc = calculator
        velocity_initialization = preserve_or_initialize_velocities(
            atoms, args.temperature, args.seed
        )
        initial_state = state_hashes(atoms)
        dynamics = VelocityVerlet(atoms, args.timestep * units.fs)
        cutoff = float(
            calculator.model.r_max
            if args.executor == "candidate"
            else calculator.models[0].r_max
        )
        edge_count = len(neighbor_list("i", atoms, cutoff))
        info = provenance(args, calculator, args.executor)
        atoms.get_forces()
        dynamics.run(args.warmup_steps)
        synchronize(args.device)
        if args.device.startswith("cuda"):
            import torch

            torch.cuda.reset_peak_memory_stats(args.device)
        start = time.perf_counter()
        steps = 0
        while time.perf_counter() - start < args.seconds:
            dynamics.run(1)
            steps += 1
        synchronize(args.device)
        elapsed = time.perf_counter() - start
        final_energy = atoms.get_potential_energy()
        if not math.isfinite(final_energy) or not np.isfinite(atoms.get_forces()).all():
            raise FloatingPointError("Non-finite MD state in timed block")
        native_diagnostics = calculator.results.get("polar_diagnostics")
        if args.executor == "candidate" and (
            not native_diagnostics
            or native_diagnostics.get("backbone") != "native_feature_forward"
            or native_diagnostics.get("prefix_fallback") is not False
        ):
            raise RuntimeError(
                "Timed candidate did not use its native feature/VJP executor"
            )
    measured_memory = memory.record()
    stage_profile = None
    if args.executor == "candidate":
        # One synchronized diagnostic call is outside timing and memory samples.
        calculator.profile = True
        try:
            calculator.calculate(atoms.copy(), properties=["energy", "forces"])
            synchronize(args.device)
            stage_profile = calculator.results["polar_diagnostics"]["timings_ms"]
            if abs(calculator.results["energy"] - final_energy) > 1e-8:
                raise RuntimeError("Diagnostic profiling changed the candidate energy")
        finally:
            calculator.profile = False
    record = {
        "command": "benchmark_worker",
        "executor": args.executor,
        "repeat": args.repeat,
        "structure": inspect_structure(atoms),
        "input_sha256": sha256(args.input),
        "velocity_initialization": velocity_initialization,
        "initial_state": initial_state,
        "provenance": info,
        "timing_scope": "complete ASE VelocityVerlet MD, including calculator and CPU force transfer; final CUDA synchronize included",
        "ensemble": "NVE",
        "precision": "float64",
        "properties": ["energy", "forces"],
        "timestep_fs": args.timestep,
        "temperature_initial_K": initial_state["temperature_K"],
        "temperature_requested_when_momenta_absent_K": args.temperature,
        "seed": args.seed,
        "warmup_steps_excluded": args.warmup_steps,
        "block_seconds_requested": args.seconds,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "us_per_atom_step": elapsed * 1e6 / (len(atoms) * steps),
        "steps_per_second": steps / elapsed,
        "cutoff_A": cutoff,
        "skin_A": 0.0,
        "effective_cutoff_A": cutoff,
        "initial_directed_edges": edge_count,
        "neighbor_construction_included": True,
        "memory": measured_memory,
        "memory_scope": "worker initialization, warmup and timed MD; extra diagnostic profile excluded",
        "candidate_diagnostics": native_diagnostics,
        "separate_single_call_profile_timings_ms": stage_profile,
        "separate_single_call_profile_in_throughput": False,
        "final_energy_eV": final_energy,
        "passed": True,
    }
    write_json(args.output, record)


def benchmark(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for repeat in range(args.repeats):
        for executor, interpreter in (
            ("original", args.original_python),
            ("candidate", args.candidate_python),
        ):
            path = output / f"{executor}-{repeat:02d}.json"
            command = [
                interpreter,
                str(Path(__file__).resolve()),
                "_worker",
                "--executor",
                executor,
                "--checkpoint",
                str(Path(args.checkpoint).resolve()),
                "--input",
                str(Path(args.input).resolve()),
                "--output",
                str(path.resolve()),
                "--device",
                args.device,
                "--bridge",
                args.bridge,
                "--seconds",
                str(args.seconds),
                "--repeat",
                str(repeat),
                "--timestep",
                str(args.timestep),
                "--temperature",
                str(args.temperature),
                "--seed",
                str(args.seed + repeat),
                "--warmup-steps",
                str(args.warmup_steps),
            ]
            log = output / f"{executor}-{repeat:02d}.log"
            with log.open("w") as stream:
                result = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            if result.returncode:
                write_json(
                    output / "benchmark-failure.json",
                    {
                        "passed": False,
                        "command": command,
                        "returncode": result.returncode,
                        "log": log,
                        "completed_records": records,
                    },
                )
                raise RuntimeError(f"{executor} repeat {repeat} failed; see {log}")
            record = json.loads(path.read_text())
            if (
                record.get("executor") != executor
                or record.get("repeat") != repeat
                or not record.get("passed")
            ):
                raise RuntimeError(
                    f"Worker result does not match requested invocation: {path}"
                )
            records.append(str(path.resolve()))
    summarize_records(
        [Path(path) for path in records], output / "benchmark-report.json"
    )


def summarize_records(paths, output):
    records = [json.loads(Path(path).read_text()) for path in paths]
    records = [
        record for record in records if record.get("command") == "benchmark_worker"
    ]
    if not records:
        raise ValueError("No benchmark_worker JSON records found")

    def identity(record):
        return (
            record["input_sha256"],
            record["provenance"]["checkpoint_sha256"],
            record["structure"]["atoms"],
            record["structure"]["waters"],
            record["precision"],
            record["timestep_fs"],
            record["cutoff_A"],
            tuple(record["properties"]),
            record["block_seconds_requested"],
            record["temperature_initial_K"],
            record["warmup_steps_excluded"],
            record["provenance"]["device_requested"],
        )

    if len({identity(record) for record in records}) != 1:
        raise ValueError(
            "Reports mix different inputs/checkpoints/precision/timestep/cutoffs"
        )
    summary = {
        "command": "report",
        "structure": records[0]["structure"],
        "records": [str(path) for path in paths],
        "executors": {},
    }
    for executor in ("original", "candidate"):
        rows = [record for record in records if record["executor"] == executor]
        if not rows:
            raise ValueError(f"Missing {executor} timing records")
        if len({row["repeat"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate {executor} repeats")
        values = [row["us_per_atom_step"] for row in rows]
        median = statistics.median(values)
        summary["executors"][executor] = {
            "repeats": len(rows),
            "median_us_per_atom_step": median,
            "median_absolute_deviation_us_per_atom_step": statistics.median(
                abs(value - median) for value in values
            ),
            "samples_us_per_atom_step": values,
            "steps": [row["steps"] for row in rows],
            "elapsed_seconds": [row["elapsed_seconds"] for row in rows],
            "nvidia_process_peak_MiB": [
                row["memory"]["nvidia_process_peak_MiB"] for row in rows
            ],
            "process_peak_rss_bytes": [
                row["memory"]["process_peak_rss_bytes"] for row in rows
            ],
            "provenance": [row["provenance"] for row in rows],
        }
    original = summary["executors"]["original"]
    candidate = summary["executors"]["candidate"]
    if original["repeats"] != candidate["repeats"]:
        raise ValueError("Original and candidate repeat counts differ")
    repeated_inputs = {
        executor: {
            record["repeat"]: record["seed"]
            for record in records
            if record["executor"] == executor
        }
        for executor in ("original", "candidate")
    }
    if repeated_inputs["original"] != repeated_inputs["candidate"]:
        raise ValueError(
            "Original and candidate repeat identities or velocity seeds differ"
        )
    for repeat in repeated_inputs["original"]:
        paired = [record for record in records if record["repeat"] == repeat]
        states = {
            (
                record["initial_state"]["positions_sha256"],
                record["initial_state"]["velocities_sha256"],
            )
            for record in paired
        }
        if len(states) != 1:
            raise ValueError(
                f"Original and candidate initial positions or velocities differ in repeat {repeat}"
            )
    summary["speedup_original_over_candidate"] = (
        original["median_us_per_atom_step"] / candidate["median_us_per_atom_step"]
    )
    summary["passed"] = all(record.get("passed", False) for record in records)
    write_json(output, summary)
    print(
        json.dumps(
            clean(
                {
                    "output": str(output),
                    "speedup_original_over_candidate": summary[
                        "speedup_original_over_candidate"
                    ],
                    "executors": {
                        name: data["median_us_per_atom_step"]
                        for name, data in summary["executors"].items()
                    },
                }
            )
        )
    )


def parser():
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = result.add_subparsers(dest="command", required=True)

    def common(name, help_text):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument(
            "--output",
            required=True,
            type=Path,
            help="Explicit output directory (worker: JSON file)",
        )
        sub.add_argument("--checkpoint", type=Path)
        sub.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
        sub.add_argument("--bridge", choices=("host", "dlpack"), default="dlpack")
        sub.add_argument("--seed", type=int, default=20261024)
        sub.add_argument("--temperature", type=float, default=300.0)
        sub.add_argument(
            "--timestep", type=float, default=0.25, help="MD timestep in fs"
        )
        return sub

    preparation = common(
        "prepare", "Generate one-Li boxes, fixed-cell BFGS, and original-model NVT"
    )
    preparation.add_argument(
        "--waters", nargs="+", type=int, default=list(FORMAL_WATER_COUNTS)
    )
    preparation.add_argument(
        "--stage", choices=("initial", "optimize", "equilibrate", "all"), default="all"
    )
    preparation.add_argument(
        "--input", type=Path, help="Resume one existing explicit structure"
    )
    preparation.add_argument("--density", type=float, default=1.0)
    preparation.add_argument("--max-opt-steps", type=int, default=2000)
    preparation.add_argument(
        "--optimizer",
        choices=("auto", "bfgs", "lbfgs"),
        default="auto",
        help="auto uses LBFGS for >=256 waters, otherwise BFGS; cell and convergence tolerance are unchanged",
    )
    preparation.add_argument("--fmax", type=float, default=0.05)
    preparation.add_argument("--nvt-steps", type=int, default=500)
    preparation.set_defaults(function=prepare)
    for name, function in (("validate", validate), ("md", md)):
        sub = common(
            name,
            "Compare complete original/candidate physics"
            if name == "validate"
            else "NVE trajectory with same-frame original-model checks",
        )
        sub.add_argument("--input", required=True, type=Path)
        sub.add_argument(
            "--energy-tolerance",
            type=float,
            default=1e-4,
            help="Absolute total-cell energy tolerance, eV",
        )
        sub.add_argument(
            "--force-tolerance",
            type=float,
            default=1e-5,
            help="Maximum force-component tolerance, eV/A",
        )
        if name == "md":
            sub.add_argument(
                "--cpu-threads",
                type=int,
                default=1,
                help="CPU correctness MD only; distinct physical cores, shared OpenMP team and no nested parallelism. Formal benchmarks always use one core.",
            )
            sub.add_argument("--steps", type=int, default=100)
            sub.add_argument("--sample-interval", type=int, default=10)
        sub.set_defaults(function=function)
    for name, function in (("benchmark", benchmark), ("_worker", benchmark_worker)):
        sub = common(
            name, "Independent original/candidate complete ASE MD timing processes"
        )
        sub.add_argument("--input", required=True, type=Path)
        sub.add_argument("--seconds", type=float, default=10.0)
        sub.add_argument("--warmup-steps", type=int, default=10)
        if name == "benchmark":
            sub.add_argument("--repeats", type=int, default=3)
            sub.add_argument("--original-python", default=sys.executable)
            sub.add_argument("--candidate-python", default=sys.executable)
        else:
            sub.add_argument(
                "--executor", choices=("original", "candidate"), required=True
            )
            sub.add_argument("--repeat", type=int, required=True)
        sub.set_defaults(function=function)
    report = commands.add_parser(
        "report", help="Summarize worker JSON files from one structure"
    )
    report.add_argument("--input", nargs="+", required=True, type=Path)
    report.add_argument(
        "--output", required=True, type=Path, help="Output summary JSON file"
    )
    report.set_defaults(
        function=lambda args: summarize_records(args.input, args.output)
    )
    return result


def main():
    args = parser().parse_args()
    if hasattr(args, "timestep") and (args.timestep <= 0 or args.temperature <= 0):
        raise ValueError("Timestep and initial temperature must be positive")
    for name in (
        "steps",
        "sample_interval",
        "seconds",
        "repeats",
        "max_opt_steps",
        "fmax",
        "nvt_steps",
    ):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if (
        args.command not in ("report",)
        and not (args.command == "prepare" and args.stage == "initial")
        and not args.checkpoint
    ):
        raise ValueError("--checkpoint is required for model execution")
    try:
        args.function(args)
    except Exception as error:
        failure = {
            "command": args.command,
            "passed": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "arguments": vars(args) | {"function": None},
        }
        try:
            failure["provenance"] = provenance(args)
        except Exception as provenance_error:  # noqa: BLE001 -- preserve the primary failure
            failure["provenance_error"] = repr(provenance_error)
        output = (
            args.output
            if args.command == "_worker"
            else Path(args.output) / "failure.json"
        )
        if args.command == "report":
            output = Path(args.output).with_suffix(".failure.json")
        write_json(output, failure)
        raise


if __name__ == "__main__":
    main()
