#!/usr/bin/env python3
"""Qualify SM80 and SM90 concurrently inside one three-GPU Slurm allocation.

The controller imports only the standard library. Each profile has one allocated
GPU and one distinct physical CPU core; all model calls use polar_li_water.py
in fresh profile processes. Existing result files are never silently reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from select_device import allocation_evidence, visible_tokens


WATERS = (64, 128, 256, 512)
PROFILES = {"sm80": [8, 0], "sm90": [9, 0]}
STATE = {"charge": 1.0, "spin": 1.0, "external_field": [0.0, 0.0, 0.0]}
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "KOKKOS_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def physical_cores(allowed, topology_root=Path("/sys/devices/system/cpu")):
    selected, seen = [], set()
    for cpu in sorted(allowed):
        topology = topology_root / f"cpu{cpu}" / "topology"
        row = {
            key: (topology / key).read_text().strip()
            for key in ("physical_package_id", "core_id", "thread_siblings_list")
        }
        identity = (row["physical_package_id"], row["core_id"])
        if identity not in seen:
            seen.add(identity)
            selected.append({"logical_cpu": cpu, **row})
    require(
        len(selected) >= 2, "Allocation must expose at least two physical CPU cores"
    )
    return selected[:2]


def select_profile(root, profile, tokens, output):
    command = [
        str(root / "opt" / profile / "bin/python"),
        str(Path(__file__).with_name("select_device.py")),
        "--profile",
        profile,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (output / f"{profile}-selection.log").write_text(result.stderr)
    require(result.returncode == 0, f"{profile} GPU selection failed: {result.stderr}")
    token = result.stdout.strip()
    require(
        token in tokens, "GPU selector returned a token outside allocation visibility"
    )
    records = []
    for line in result.stderr.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("kind") == "POLAR_ALLOCATED_GPU_SELECTION":
            records.append(row)
    require(len(records) == 1, f"Missing unambiguous {profile} selection evidence")
    evidence = records[0]
    require(
        evidence["selected"]["capability"] == PROFILES[profile],
        "GPU capability mismatch",
    )
    require(
        evidence["selected"]["original_token"] == token, "GPU token evidence mismatch"
    )
    return evidence


def check_structure(row, waters):
    require(row.get("waters") == waters, "Water count differs from requested case")
    require(
        row.get("atoms") == 3 * waters + 1, "Atom count differs from requested case"
    )
    require(row.get("lithium_atoms") == 1, "Expected exactly one Li")
    require(
        row.get("polar_state") == STATE,
        "Expected Q=+1, spin multiplicity 1 and zero field",
    )


def prepared_inputs(args):
    checkpoint_sha = sha256(args.checkpoint)
    records = {}
    for waters in WATERS:
        prefix = args.prepared / f"water{waters:04d}"
        path = prefix.with_suffix(".equilibrated.extxyz")
        manifest_path = prefix.with_suffix(".prepare.json")
        manifest = read_json(manifest_path)
        require(
            manifest.get("status") == "passed",
            f"Preparation has not passed: {manifest_path}",
        )
        check_structure(manifest["structure"], waters)
        optimization = manifest["optimization"]
        require(
            optimization.get("converged") is True,
            "Preparation optimization did not converge",
        )
        require(
            optimization["maximum_force_eV_A"] <= 0.05,
            "Preparation force exceeds 0.05 eV/A",
        )
        nvt = manifest["nvt"]
        require(
            nvt["steps"] >= 500 and nvt["timestep_fs"] == 0.25,
            "Preparation needs 500 NVT steps at 0.25 fs",
        )
        require(
            nvt["prepared_output_sha256"] == sha256(path),
            f"Prepared file changed: {path}",
        )
        require(
            manifest["provenance"]["checkpoint_sha256"] == checkpoint_sha,
            "Preparation checkpoint differs",
        )
        require(
            nvt["prepared_output_state"]["momenta_present"] is True,
            "Prepared momenta are required",
        )
        records[str(waters)] = {
            "path": str(path),
            "sha256": sha256(path),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "initial_state": nvt["prepared_output_state"],
        }
    return records


def check_state(row, expected):
    for key in (
        "positions_sha256",
        "cell_sha256",
        "velocities_sha256",
        "momenta_sha256",
    ):
        require(
            row.get(key) == expected[key], f"Prepared starting state changed: {key}"
        )


def check_provenance(row, profile, cpu, checkpoint_sha, candidate=False):
    require(
        row["checkpoint_sha256"] == checkpoint_sha, "Checkpoint provenance mismatch"
    )
    require(row["precision"] == "float64", "Qualification requires FP64")
    require(
        row["cpu_affinity"] == [cpu], "Worker CPU affinity differs from assigned core"
    )
    require(
        row["device_requested"] in ("cuda", "cuda:0"),
        "Expected logical CUDA device zero",
    )
    require(
        row["cuda_visible_devices"] == os.environ["CUDA_VISIBLE_DEVICES"],
        "GPU visibility changed",
    )
    actual = row["actual_executor"]
    if candidate:
        require(
            actual["backbone"] == "native_feature_forward",
            "Candidate backbone fallback",
        )
        require(actual["bridge"] == "dlpack", "Candidate must use DLPack")
        require(
            actual["kokkos_execution_space"] == "Cuda",
            "Candidate native execution must use Cuda",
        )
        require(bool(actual["extension_sha256"]), "Missing native binary identity")
    else:
        require(
            actual["devices"] == ["cuda:0"], "Official model is not on the selected GPU"
        )
    gpu = row["hardware"]["gpu"]
    require(gpu["queried"] is True, "Missing actual worker GPU hardware evidence")
    require(
        gpu["device_capability"] == PROFILES[profile],
        "Actual worker GPU architecture differs",
    )
    require(
        gpu["device_ordinal"] == 0 and gpu["current_visible_ordinal"] == 0,
        "Worker CUDA ordinal is not zero inside its isolated visibility",
    )


def check_diagnostics(row):
    require(
        row["backbone"] == "native_feature_forward" and row["prefix_fallback"] is False,
        "Candidate fallback detected",
    )
    require(
        row["bridge"] == "dlpack" and row["precision"] == "float64",
        "Expected FP64 DLPack bridge",
    )
    require(
        row["native_execution_space"] == "Cuda", "Native feature execution must be Cuda"
    )
    require(
        row["feature_device"] == "cuda:0", "Feature tensor is not on logical GPU zero"
    )


def check_comparison(row):
    require(row.get("passed") is True, "Physics comparison failed")
    require(
        0 <= row["absolute_total_energy_error_eV"] <= 1e-4,
        "Total energy threshold exceeded",
    )
    require(
        0 <= row["maximum_force_component_error_eV_A"] <= 1e-5,
        "Force threshold exceeded",
    )
    for name in ("Li", "hydration_shell_water", "remaining_water"):
        region = row["force_error_by_region"][name]
        require(region["passed"] is True, f"Force region failed: {name}")
        error = region["maximum_force_component_error_eV_A"]
        require(
            error is None or 0 <= error <= 1e-5, f"Force threshold exceeded: {name}"
        )
    for name in ("original_populations", "candidate_populations"):
        populations = row[name]
        require(populations["passed"] is True, f"Conserved populations failed: {name}")
        require(
            len(populations["charge_totals"]) == 3,
            "Missing initial/two update populations",
        )
        require(
            all(abs(value - 1.0) <= 1e-9 for value in populations["charge_totals"]),
            "Charge conservation threshold exceeded",
        )
        require(
            all(abs(value) <= 1e-9 for value in populations["spin_populations"]),
            "Spin conservation threshold exceeded",
        )
    check_diagnostics(row["candidate_diagnostics"])


def check_result(path, command, waters, args, prepared, steps=None):
    record = read_json(path)
    require(record.get("passed") is True, f"Result has not passed: {path}")
    check_structure(record["structure"], waters)
    require(
        record["input_sha256"] == prepared["sha256"],
        "Result input differs from prepared file",
    )
    for executor in ("original", "candidate"):
        check_provenance(
            record[executor],
            args.worker_profile,
            args.worker_cpu,
            args.checkpoint_sha,
            executor == "candidate",
        )
    if command == "validate":
        require(
            set(record["cases"]) == {"base", "translated", "wrapped"},
            "Incomplete geometry validation",
        )
        for case in record["cases"].values():
            check_comparison(case)
    else:
        require(
            record["steps_requested"] == steps and record["timestep_fs"] == 0.25,
            "MD protocol mismatch",
        )
        require(
            record["same_trajectory_oracle"] is True,
            "MD must use same-frame official oracle",
        )
        require(
            [row["step"] for row in record["samples"]] == list(range(0, steps + 1, 10)),
            "Incomplete MD sample coverage",
        )
        require(
            record["velocity_initialization"]["policy"] == "preserve_prepared_momenta",
            "MD resampled prepared momenta",
        )
        check_state(record["initial_state"], prepared["initial_state"])
        for sample in record["samples"]:
            check_comparison(sample)
        require(
            all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in record["drift"].values()
            ),
            "Missing/nonfinite MD drift evidence",
        )


def check_performance(output, waters, args, prepared):
    path = output / "benchmark-report.json"
    summary = read_json(path)
    require(summary.get("passed") is True, "Performance summary did not pass")
    check_structure(summary["structure"], waters)
    require(
        len(summary["records"]) == 6, "Expected three independent repeats per executor"
    )
    identities, pids = set(), set()
    for worker_path in summary["records"]:
        row = read_json(worker_path)
        identity = (row["executor"], row["repeat"])
        require(identity not in identities, "Duplicate timing repeat")
        identities.add(identity)
        require(row.get("passed") is True, f"Timing worker failed: {worker_path}")
        check_structure(row["structure"], waters)
        require(row["input_sha256"] == prepared["sha256"], "Timing input mismatch")
        require(
            row["block_seconds_requested"] >= 10 and row["elapsed_seconds"] >= 10,
            "Timing block is shorter than ten seconds",
        )
        require(
            row["steps"] > 0 and row["timestep_fs"] == 0.25,
            "Timing MD protocol mismatch",
        )
        require(
            row["precision"] == "float64" and row["properties"] == ["energy", "forces"],
            "Timing properties/precision mismatch",
        )
        require(
            row["neighbor_construction_included"] is True,
            "Timing scope omitted neighbor construction",
        )
        require(
            row["separate_single_call_profile_in_throughput"] is False,
            "Diagnostic profiling entered throughput",
        )
        require(
            row["velocity_initialization"]["policy"] == "preserve_prepared_momenta",
            "Timing resampled prepared momenta",
        )
        check_state(row["initial_state"], prepared["initial_state"])
        candidate = row["executor"] == "candidate"
        check_provenance(
            row["provenance"],
            args.worker_profile,
            args.worker_cpu,
            args.checkpoint_sha,
            candidate,
        )
        pids.add(row["provenance"]["pid"])
        if candidate:
            check_diagnostics(row["candidate_diagnostics"])
            timings = row["separate_single_call_profile_timings_ms"]
            require(
                timings
                and all(
                    math.isfinite(value) and value >= 0 for value in timings.values()
                ),
                "Missing separate stage profile",
            )
    require(
        identities
        == {
            (name, repeat) for name in ("original", "candidate") for repeat in range(3)
        },
        "Timing repeat set differs from protocol",
    )
    require(len(pids) == 6, "Timing executors did not use six independent processes")
    write_json(output / "summary.json", summary)


def stage_plan(args):
    stages = []
    for waters in WATERS:
        stages.append(
            (
                "validate",
                waters,
                None,
                args.output_root
                / "qualification"
                / args.worker_profile
                / f"water{waters:04d}"
                / "validation",
            )
        )
    for steps in (100, 1000):
        stages.append(
            (
                "md",
                64,
                steps,
                args.output_root
                / "qualification"
                / args.worker_profile
                / "water0064"
                / f"md{steps}",
            )
        )
    for waters in WATERS:
        stages.append(
            (
                "benchmark",
                waters,
                None,
                args.output_root
                / "performance"
                / args.worker_profile
                / f"water{waters:04d}",
            )
        )
    return stages


def worker(args):
    os.sched_setaffinity(0, {args.worker_cpu})
    require(
        os.sched_getaffinity(0) == {args.worker_cpu},
        "Cannot pin worker to assigned physical core",
    )
    require(
        len(visible_tokens(os.environ)) == 1,
        "Profile worker requires exactly one visible GPU",
    )
    args.checkpoint_sha = sha256(args.checkpoint)
    evidence = read_json(args.evidence / "selection.json")
    prepared = evidence["prepared"]
    summary = {
        "profile": args.worker_profile,
        "status": "running",
        "cpu_affinity": [args.worker_cpu],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "planned_stages": [
            {
                "command": command,
                "waters": waters,
                "steps": steps,
                "output": str(output),
            }
            for command, waters, steps, output in stage_plan(args)
        ],
        "stages": [],
    }
    summary_path = args.evidence / f"{args.worker_profile}-pipeline.json"
    write_json(summary_path, summary)
    try:
        for number, (command, waters, steps, output) in enumerate(stage_plan(args)):
            prepared_row = prepared[str(waters)]
            require(
                sha256(prepared_row["path"]) == prepared_row["sha256"],
                "Prepared file changed during pipeline",
            )
            # Reject stale records before launch; a retry must use an explicit new
            # output root, keeping previous failure evidence reviewable.
            result_name = {
                "validate": "validation.json",
                "md": "md.json",
                "benchmark": "benchmark-report.json",
            }[command]
            require(
                not output.exists() or not any(output.iterdir()),
                f"Output directory is not empty: {output}",
            )
            invocation = [
                sys.executable,
                str(args.source / "benchmarks/polar_li_water.py"),
                command,
                "--input",
                prepared_row["path"],
                "--output",
                str(output),
                "--checkpoint",
                str(args.checkpoint),
                "--device",
                "cuda:0",
                "--bridge",
                "dlpack",
                "--timestep",
                "0.25",
            ]
            if command in ("validate", "md"):
                invocation += [
                    "--energy-tolerance",
                    "0.0001",
                    "--force-tolerance",
                    "0.00001",
                ]
            if command == "md":
                invocation += ["--steps", str(steps), "--sample-interval", "10"]
            elif command == "benchmark":
                invocation += [
                    "--repeats",
                    "3",
                    "--seconds",
                    "10",
                    "--warmup-steps",
                    "10",
                    "--original-python",
                    sys.executable,
                    "--candidate-python",
                    sys.executable,
                ]
            log = (
                args.evidence
                / f"{args.worker_profile}-{number:02d}-{command}-water{waters:04d}.log"
            )
            stage = {
                "command": invocation,
                "output": str(output),
                "log": str(log),
                "status": "running",
                "started_unix": time.time(),
            }
            summary["stages"].append(stage)
            write_json(summary_path, summary)
            with log.open("w") as stream:
                result = subprocess.run(
                    invocation,
                    cwd=args.source,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            stage.update(returncode=result.returncode, completed_unix=time.time())
            require(
                result.returncode == 0,
                f"{command} failed for {waters} waters; see {log}",
            )
            if command == "benchmark":
                check_performance(output, waters, args, prepared_row)
            else:
                check_result(
                    output / result_name, command, waters, args, prepared_row, steps
                )
            stage["status"] = "passed"
            write_json(summary_path, summary)
        summary["status"] = "passed"
        return 0
    except Exception as error:
        summary.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
        if summary["stages"] and summary["stages"][-1]["status"] == "running":
            summary["stages"][-1]["status"] = "failed"
        print(summary["traceback"], file=sys.stderr)
        return 1
    finally:
        write_json(summary_path, summary)


def controller(args):
    allocation, count = allocation_evidence(os.environ)
    tokens = visible_tokens(os.environ)
    require(
        len(tokens) == 3 and (count is None or count == 3),
        "Pair qualification requires exactly three allocated visible GPUs",
    )
    require(
        not args.evidence.exists(),
        f"Evidence directory already exists: {args.evidence}",
    )
    args.evidence.mkdir(parents=True)
    summary = {
        "status": "running",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "allocation": allocation,
        "original_cuda_visible_devices": tokens,
        "source": str(args.source),
        "script_sha256": sha256(__file__),
        "profiles": {},
    }
    processes, streams = [], []
    try:
        source_commit = subprocess.run(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
        summary["source_git_sha"] = source_commit
        cores = physical_cores(os.sched_getaffinity(0))
        prepared = prepared_inputs(args)
        selections = {
            profile: select_profile(args.root, profile, tokens, args.evidence)
            for profile in PROFILES
        }
        chosen_tokens = [
            row["selected"]["original_token"] for row in selections.values()
        ]
        require(
            len(set(chosen_tokens)) == 2, "Profiles must have different allocated GPUs"
        )
        write_json(
            args.evidence / "selection.json",
            {
                **summary,
                "prepared": prepared,
                "checkpoint_sha256": sha256(args.checkpoint),
                "allowed_cpu_affinity": sorted(os.sched_getaffinity(0)),
                "assigned_physical_cores": cores,
                "selections": selections,
            },
        )
        # nvidia-smi is inventory evidence only. GPU selection above uses only the
        # original Slurm visibility and never expands it from physical indices.
        with (args.evidence / "nvidia-smi.txt").open("w") as stream:
            subprocess.run(
                ["nvidia-smi"],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=30,
            )
        for (profile, selection), core in zip(selections.items(), cores, strict=True):
            token, cpu = selection["selected"]["original_token"], core["logical_cpu"]
            environment = os.environ | {name: "1" for name in THREAD_VARIABLES}
            environment.update(
                CUDA_VISIBLE_DEVICES=token, SYMMETRIX_BACKEND=f"cuda12-{profile}"
            )
            environment["PATH"] = (
                f"{args.root / 'opt' / profile / 'bin'}:{os.environ.get('PATH', '')}"
            )
            command = [
                str(args.root / "opt" / profile / "bin/python"),
                str(Path(__file__).resolve()),
                "--root",
                str(args.root),
                "--source",
                str(args.source),
                "--prepared",
                str(args.prepared),
                "--output-root",
                str(args.output_root),
                "--checkpoint",
                str(args.checkpoint),
                "--evidence",
                str(args.evidence),
                "--worker-profile",
                profile,
                "--worker-cpu",
                str(cpu),
            ]
            stream = (args.evidence / f"{profile}-controller.log").open("w")
            streams.append(stream)
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                cwd=args.source,
            )
            processes.append((profile, process))
            summary["profiles"][profile] = {
                "pid": process.pid,
                "command": command,
                "gpu": selection["selected"],
                "cpu": core,
                "status": "running",
            }
        write_json(args.evidence / "pipeline.json", summary)
        for profile, process in processes:
            returncode = process.wait()
            worker_summary = args.evidence / f"{profile}-pipeline.json"
            status = (
                read_json(worker_summary).get("status")
                if worker_summary.is_file()
                else "failed"
            )
            summary["profiles"][profile].update(returncode=returncode, status=status)
            write_json(args.evidence / "pipeline.json", summary)
        summary["status"] = (
            "passed"
            if all(
                row["returncode"] == 0 and row["status"] == "passed"
                for row in summary["profiles"].values()
            )
            else "failed"
        )
        return 0 if summary["status"] == "passed" else 1
    except Exception as error:
        summary.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
        print(summary["traceback"], file=sys.stderr)
        return 1
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for stream in streams:
            stream.close()
        write_json(args.evidence / "pipeline.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument(
        "--worker-profile", choices=tuple(PROFILES), help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-cpu", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.source = (args.source or args.root / "src/symmetrix-xl").resolve()
    args.prepared = (
        args.prepared or args.root / "benchmark/prepared-reference"
    ).resolve()
    args.output_root = (args.output_root or args.root / "benchmark").resolve()
    args.checkpoint = (
        args.checkpoint or args.root / "forcefields/polar1s/MACE-POLAR-1-S.model"
    ).resolve()
    args.evidence = (
        args.evidence
        or args.output_root
        / "gpu-pair"
        / os.environ.get("SLURM_JOB_ID", "no-allocation")
    ).resolve()
    for path in (args.output_root, args.evidence):
        require(
            not path.is_relative_to(args.source),
            "Result paths must be outside the source checkout",
        )
    return worker(args) if args.worker_profile else controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
