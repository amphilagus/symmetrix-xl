#!/usr/bin/env python3
"""Run qualified cpu-omp correctness MD, then isolated one-core ASE timings.

The controller is standard-library only. Existing prepared-state verification
is shared with the GPU controller; no GPU selection or numerical code runs in
this process. Every model stage starts a fresh locked-profile interpreter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

from bootstrap import sha256, write_json
from qualify_gpu_pair import WATERS, check_state, check_structure, prepared_inputs


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def physical_cores(allowed, topology=Path("/sys/devices/system/cpu")):
    result, seen = [], set()
    for cpu in sorted(allowed):
        folder = topology / f"cpu{cpu}/topology"
        identity = tuple(
            (folder / key).read_text().strip()
            for key in ("physical_package_id", "core_id")
        )
        if identity not in seen:
            seen.add(identity)
            result.append(
                {"logical_cpu": cpu, "package": identity[0], "core": identity[1]}
            )
    require(
        len(result) >= 8, "The allocation must expose eight distinct physical cores"
    )
    return result[:8]


def build_gate(root, source):
    manifest = root / "manifests/build/cpu-omp"
    built = read_json(manifest / "built.json")
    doctor = read_json(manifest / "doctor-torch-polar.json")
    require(built["profile"] == "cpu-omp", "Wrong qualified native profile")
    require(
        doctor.get("status") == "ok", "Strict Torch/POLAR doctor must have status ok"
    )
    cases = ET.parse(manifest / "polar-tests.xml").getroot().findall(".//testcase")
    require(len(cases) >= 40, "At least forty complete POLAR tests are required")
    require(
        not any(
            child.tag in {"failure", "error", "skipped"}
            for case in cases
            for child in case
        ),
        "POLAR qualification contains a failed, errored, or skipped test",
    )
    require(
        sha256(built["extension_path"]) == built["extension_sha256"],
        "Qualified native binary changed",
    )
    require(
        sha256(root / "opt/cpu-omp/bin/python") == built["python_wrapper_sha256"],
        "Qualified CPU wrapper changed",
    )
    require(
        sha256(root / "manifests/cpu-toolchain.json")
        == built["cpu_toolchain_manifest_sha256"],
        "Qualified CPU toolchain changed",
    )
    sys.path.insert(0, str(source))
    try:
        from tools._symmetrix_build.source_provenance import (
            symmetrix_native_source_fingerprint,
            symmetrix_source_fingerprint,
        )

        expected = {
            "source_content_sha256": symmetrix_source_fingerprint(source),
            "native_source_content_sha256": symmetrix_native_source_fingerprint(source),
        }
    finally:
        sys.path.pop(0)
    require(
        expected == built["source_fingerprints"],
        "Current numerical source differs from qualified build",
    )
    for key, value in expected.items():
        require(
            doctor["qualification_context"]["native_build_info"][key] == value,
            "Doctor used another native source",
        )
    return built, {
        "doctor_sha256": sha256(manifest / "doctor-torch-polar.json"),
        "polar_tests_sha256": sha256(manifest / "polar-tests.xml"),
        "polar_tests_passed": len(cases),
        "built_sha256": sha256(manifest / "built.json"),
    }


def provenance_gate(row, args, cpus, candidate):
    require(
        row["precision"] == "float64" and row["device_requested"] == "cpu",
        "Expected complete FP64 CPU execution",
    )
    require(
        row["checkpoint_sha256"] == args.checkpoint_sha,
        "Checkpoint identity changed",
    )
    require(
        row["cpu_affinity"] == cpus,
        "Worker affinity differs from the selected physical cores",
    )
    require(
        row["early_execution_environment"]["requested_threads"] == len(cpus),
        "Worker thread policy changed",
    )
    for key, value in row["thread_environment"].items():
        require(
            value
            == (
                str(len(cpus))
                if key in {"OMP_NUM_THREADS", "KOKKOS_NUM_THREADS"}
                else "1"
            ),
            "Unexpected numerical thread count: " + key,
        )
    actual = row["actual_executor"]
    if candidate:
        require(
            actual["backbone"] == "native_feature_forward"
            and actual["bridge"] == "host",
            "Candidate fallback or wrong bridge",
        )
        require(
            actual["kokkos_execution_space"] == "OpenMP",
            "Candidate must use native OpenMP",
        )
        require(
            actual["extension_sha256"] == args.built["extension_sha256"],
            "Candidate binary differs from qualified build",
        )
        require(
            actual["openmp_runtime"]["kokkos_concurrency"] == len(cpus),
            "Actual Kokkos concurrency differs",
        )
        for key, value in args.built["source_fingerprints"].items():
            require(
                actual["native_build_info"][key] == value,
                "Candidate source identity differs",
            )
    else:
        require(actual["devices"] == ["cpu"], "Official model must use CPU")


def comparison_gate(row):
    require(row.get("passed") is True, "Same-frame physics comparison failed")
    require(
        0 <= row["absolute_total_energy_error_eV"] <= 1e-4, "Energy threshold exceeded"
    )
    require(
        0 <= row["maximum_force_component_error_eV_A"] <= 1e-5,
        "Force threshold exceeded",
    )
    for name in ("Li", "hydration_shell_water", "remaining_water"):
        require(
            row["force_error_by_region"][name]["passed"] is True,
            "Regional force comparison failed",
        )
    for name in ("original_populations", "candidate_populations"):
        population = row[name]
        require(population["passed"] is True, "Charge/spin conservation failed")
        require(
            len(population["charge_totals"]) == 3,
            "Missing initial/two-update normalization evidence",
        )
        require(
            all(abs(value - 1) <= 1e-9 for value in population["charge_totals"]),
            "Q=+1 was not conserved",
        )
        require(
            all(abs(value) <= 1e-9 for value in population["spin_populations"]),
            "Singlet population was not conserved",
        )
    diag = row["candidate_diagnostics"]
    require(
        diag["backbone"] == "native_feature_forward"
        and diag["prefix_fallback"] is False,
        "Native feature fallback detected",
    )


def result_gate(path, command, waters, steps, interval, args, prepared, cpus):
    report = read_json(path)
    require(report.get("passed") is True, f"Stage failed: {path}")
    check_structure(report["structure"], waters)
    if command == "benchmark":
        require(
            len(report["records"]) == 6,
            "Expected three independent processes per executor",
        )
        identities, pids = set(), set()
        for worker_path in report["records"]:
            worker_path = Path(worker_path).resolve()
            require(
                worker_path.is_relative_to(path.parent),
                "Worker result escaped its output directory",
            )
            row = read_json(worker_path)
            identity = row["executor"], row["repeat"]
            require(identity not in identities, "Duplicate timing repeat")
            identities.add(identity)
            pids.add(row["provenance"]["pid"])
            require(
                row["passed"] is True and row["input_sha256"] == prepared["sha256"],
                "Worker failed or used a different input",
            )
            require(
                row["block_seconds_requested"] >= 10
                and row["elapsed_seconds"] >= 10
                and row["steps"] > 0,
                "Timing needs at least ten seconds and one full MD step",
            )
            require(
                row["warmup_steps_excluded"] == 2,
                "Both executors require two excluded warmup steps",
            )
            require(
                row["neighbor_construction_included"] is True
                and row["separate_single_call_profile_in_throughput"] is False,
                "Timing scope differs from the complete ASE MD protocol",
            )
            require(
                "complete ASE VelocityVerlet MD" in row["timing_scope"],
                "Missing actual full-MD timing scope",
            )
            require(
                row["precision"] == "float64"
                and row["properties"] == ["energy", "forces"]
                and row["timestep_fs"] == 0.25,
                "Timing precision/properties/timestep changed",
            )
            check_state(row["initial_state"], prepared["initial_state"])
            provenance_gate(
                row["provenance"], args, cpus, row["executor"] == "candidate"
            )
        require(
            identities
            == {
                (executor, repeat)
                for executor in ("original", "candidate")
                for repeat in range(3)
            }
            and len(pids) == 6,
            "Independent timing repeat identities are incomplete",
        )
    else:
        require(
            report["input_sha256"] == prepared["sha256"],
            "Stage input differs from prepared structure",
        )
        for executor in ("original", "candidate"):
            provenance_gate(report[executor], args, cpus, executor == "candidate")
        if command == "validate":
            require(
                set(report["cases"]) == {"base", "translated", "wrapped"},
                "Geometry validation is incomplete",
            )
            for case in report["cases"].values():
                comparison_gate(case)
        else:
            require(
                report["steps_requested"] == steps
                and report["timestep_fs"] == 0.25
                and report["same_trajectory_oracle"] is True,
                "MD protocol changed",
            )
            require(
                [row["step"] for row in report["samples"]]
                == list(range(0, steps + 1, interval)),
                "MD sample coverage is incomplete",
            )
            require(
                report["velocity_initialization"]["policy"]
                == "preserve_prepared_momenta",
                "MD resampled prepared momenta",
            )
            check_state(report["initial_state"], prepared["initial_state"])
            for row in report["samples"]:
                comparison_gate(row)
            require(
                all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in report["drift"].values()
                ),
                "Missing finite NVE drift report",
            )


def stages(root):
    qualification, performance = (
        root / "benchmark/qualification/cpu-omp",
        root / "benchmark/performance/cpu-omp",
    )
    result = [
        ("md", 64, 100, 10, qualification / "water0064/md100"),
        ("md", 64, 1000, 100, qualification / "water0064/md1000"),
    ]
    result += [
        (
            "validate",
            waters,
            None,
            None,
            qualification / f"water{waters:04d}/validation",
        )
        for waters in WATERS
    ]
    result += [
        ("benchmark", waters, None, None, performance / f"water{waters:04d}")
        for waters in WATERS
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    require(bool(os.environ.get("SLURM_JOB_ID")), "Run inside a Slurm allocation")
    require(
        int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) >= 16,
        "Request at least sixteen Slurm logical CPUs",
    )
    args.root = args.root.resolve()
    source = args.root / "src/symmetrix-xl"
    args.prepared = args.root / "benchmark/prepared-reference"
    args.checkpoint = args.root / "forcefields/polar1s/MACE-POLAR-1-S.model"
    args.checkpoint_sha = sha256(args.checkpoint)
    expected_python = args.root / "opt/cpu-omp/bin/python"
    require(
        Path(sys.executable).resolve() == expected_python.resolve(),
        "Controller must use the locked cpu-omp interpreter",
    )
    allowed = sorted(os.sched_getaffinity(0))
    require(
        len(allowed) >= 16,
        "Slurm must expose sixteen logical CPUs, covering eight physical cores",
    )
    cores = physical_cores(allowed)
    md_cpus = sorted(row["logical_cpu"] for row in cores)
    one_cpu = [md_cpus[0]]
    require(
        not subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip(),
        "Commit the source before qualification",
    )
    args.built, gates = build_gate(args.root, source)
    prepared = prepared_inputs(args)
    planned = stages(args.root)
    evidence = (
        args.root
        / "manifests/qualification"
        / ("cpu-omp-" + os.environ["SLURM_JOB_ID"])
    )
    require(not evidence.exists(), "Pipeline evidence directory already exists")
    for directory in (
        args.root / "benchmark/qualification/cpu-omp",
        args.root / "benchmark/performance/cpu-omp",
    ):
        require(
            not directory.exists(),
            f"Refusing to overwrite previous evidence: {directory}",
        )
    evidence.mkdir(parents=True)
    summary = {
        "status": "running",
        "profile": "cpu-omp",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "script_sha256": sha256(__file__),
        "source_git_sha": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "build_gate": gates,
        "prepared": prepared,
        "allocated_cpu_affinity": allowed,
        "md_physical_cores": cores,
        "formal_performance_affinity": one_cpu,
        "stages": [],
    }
    summary_path = evidence / "pipeline.json"
    write_json(summary_path, summary)
    try:
        for number, (command, waters, steps, interval, output) in enumerate(planned):
            row = prepared[str(waters)]
            require(
                sha256(row["path"]) == row["sha256"],
                "Prepared input changed during the pipeline",
            )
            require(
                not output.exists(), f"Refusing to overwrite stage evidence: {output}"
            )
            cpus = md_cpus if command == "md" else one_cpu
            invocation = [
                str(expected_python),
                str(source / "benchmarks/polar_li_water.py"),
                command,
                "--input",
                row["path"],
                "--output",
                str(output),
                "--checkpoint",
                str(args.checkpoint),
                "--device",
                "cpu",
                "--bridge",
                "host",
                "--timestep",
                "0.25",
            ]
            if command == "md":
                invocation += [
                    "--cpu-threads",
                    "8",
                    "--steps",
                    str(steps),
                    "--sample-interval",
                    str(interval),
                ]
            elif command == "benchmark":
                invocation += [
                    "--repeats",
                    "3",
                    "--seconds",
                    "10",
                    "--warmup-steps",
                    "2",
                    "--original-python",
                    str(expected_python),
                    "--candidate-python",
                    str(expected_python),
                ]
            if command in {"validate", "md"}:
                invocation += [
                    "--energy-tolerance",
                    "0.0001",
                    "--force-tolerance",
                    "0.00001",
                ]
            log = evidence / f"{number:02d}-{command}-water{waters:04d}.log"
            stage = {
                "command": invocation,
                "cpu_affinity": cpus,
                "output": str(output),
                "log": str(log),
                "status": "running",
                "started_unix": time.time(),
            }
            summary["stages"].append(stage)
            write_json(summary_path, summary)
            with log.open("x") as stream:
                result = subprocess.run(
                    invocation,
                    cwd=source,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                    preexec_fn=lambda: os.sched_setaffinity(0, set(cpus)),
                )
            stage.update(returncode=result.returncode, completed_unix=time.time())
            require(result.returncode == 0, f"{command} failed; see {log}")
            filename = {
                "md": "md.json",
                "validate": "validation.json",
                "benchmark": "benchmark-report.json",
            }[command]
            result_gate(
                output / filename, command, waters, steps, interval, args, row, cpus
            )
            stage.update(status="passed", result_sha256=sha256(output / filename))
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


if __name__ == "__main__":
    raise SystemExit(main())
