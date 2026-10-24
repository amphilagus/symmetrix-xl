#!/usr/bin/env python3
"""Build one independently qualified POLAR native backend under Slurm."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil

from bootstrap import environment, install_wrappers, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("cpu", "cpu-omp", "sm80", "sm90"), required=True
    )
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument(
        "--qualify-only",
        action="store_true",
        help="Reuse installed binaries only when both source fingerprints match",
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Native compilation requires a Slurm compute job")
    root, donor = args.root.resolve(), args.donor.resolve()
    if root == donor or root.parent != donor.parent:
        raise ValueError("Build root must be independent of the donor")
    profile = args.profile
    source = root / "src/symmetrix-xl"
    is_cpu = profile in ("cpu", "cpu-omp")
    env = environment(root, donor, profile)
    env["SYMMETRIX_POLAR_BUILD_MODE"] = "1"
    env["PIP_FIND_LINKS"] = " ".join(
        str(p)
        for p in (
            root / "src/wheels",
            donor / "src/wheels_symmetrix",
            donor
            / "src/polar/polar-mace0316-torch260cu126-cueq091-a4470e17e377/wheels",
        )
    )
    env["CMAKE_ARGS"] = "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
    python = root / "opt" / profile / "bin/python"
    loader_links = install_wrappers(root, profile)
    env["PATH"] = str(python.parent) + os.pathsep + env["PATH"]
    env["UV_CACHE_DIR"] = str(root / "build/cache" / profile / "uv")
    # uv treats this environment value as one location, unlike pip's list.
    env["UV_FIND_LINKS"] = str(root / "src/wheels")
    env["UV_PYTHON_DOWNLOADS"] = "never"
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-index", "uv==0.8.22"],
        env=env,
        check=True,
    )
    wheels = root / "build/wheels"
    if profile == "cpu-omp":
        wheels /= profile
    wheels.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifests/build" / profile
    manifest.mkdir(parents=True, exist_ok=True)
    previous = [p for p in manifest.iterdir() if p.is_file()]
    if previous:
        archive = manifest / "attempts" / ("before-" + os.environ["SLURM_JOB_ID"])
        archive.mkdir(parents=True, exist_ok=False)
        for path in previous:
            shutil.copy2(path, archive / path.name)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, env=env, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=source,
        env=env,
        text=True,
    )
    if dirty:
        raise RuntimeError("Native build source has tracked modifications")
    common = [
        "--generator",
        "Unix Makefiles",
        "--build-root",
        str(root / "build/native" / profile),
        "--jobs",
        str(args.jobs),
    ]
    frontend = [str(python), "tools/symmetrix_build.py"]
    if args.qualify_only:
        pass
    elif is_cpu:
        command = [
            *frontend,
            "wheel",
            "--backend",
            "cpu",
            "--cpu-target",
            "x86-64-v3",
            "--blas",
            "openblas",
            "--cxx",
            env["CXX"],
            "--wheel-dir",
            str(wheels),
            *common,
        ]
        subprocess.run(command, cwd=source, env=env, check=True)
        candidates = list(wheels.glob("symmetrix_xl-0.1.1-*.whl"))
        if len(candidates) != 1:
            raise RuntimeError("Expected exactly one CPU frontend wheel")
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                str(candidates[0]),
            ],
            env=env,
            check=True,
        )
    else:
        candidates = list(wheels.glob("symmetrix_xl-0.1.1-*.whl"))
        if len(candidates) != 1:
            raise RuntimeError("Build the matching CPU frontend wheel first")
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                str(candidates[0]),
            ],
            env=env,
            check=True,
        )
        site = root / "opt" / profile / "lib/python3.12/site-packages"
        cublas = site / "nvidia/cublas"
        devel = root / "build/cublas" / profile
        (devel / "lib64").mkdir(parents=True, exist_ok=True)
        for path, target in (
            (devel / "include", cublas / "include"),
            (devel / "lib64/libcublas.so", cublas / "lib/libcublas.so.12"),
            (devel / "lib64/libcublasLt.so", cublas / "lib/libcublasLt.so.12"),
        ):
            path.unlink(missing_ok=True)
            path.symlink_to(target)
        env["CMAKE_ARGS"] += " -DCUBLAS_ROOT=" + str(devel)
        env["CMAKE_ARGS"] += " -DCUDA_cublas_LIBRARY=" + str(
            devel / "lib64/libcublas.so"
        )
        env["CMAKE_ARGS"] += " -DCUDA_cublasLt_LIBRARY=" + str(
            devel / "lib64/libcublasLt.so"
        )
        gpu_lite = next((donor / "src").glob("gpu-lite-*"))
        env["CMAKE_ARGS"] += " -DFETCHCONTENT_SOURCE_DIR_GPU_LITE=" + str(gpu_lite)
        command = [
            *frontend,
            "install",
            "--backend",
            "cuda",
            "--arch",
            profile,
            "--cuda-root",
            env["CUDA_HOME"],
            "--host-cxx",
            env["CXX"],
            *common,
        ]
        subprocess.run(command, cwd=source, env=env, check=True)
    env.pop("SYMMETRIX_POLAR_BUILD_MODE", None)
    env["SYMMETRIX_BACKEND"] = "cpu" if is_cpu else "cuda12-" + profile
    env["SYMMETRIX_POLAR_MODEL"] = str(
        root / "forcefields/polar1s/MACE-POLAR-1-S.model"
    )
    for name in (
        "OMP_NUM_THREADS",
        "KOKKOS_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    for label, command in (
        (
            "version",
            [str(python), "-m", "symmetrix.cli.main", "version", "--probe", "--json"],
        ),
        ("pip-check", [str(python), "-m", "pip", "check"]),
    ):
        result = subprocess.run(
            command, env=env, cwd=source, capture_output=True, text=True
        )
        (manifest / (label + ".txt")).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(label + " probe failed")
        if label == "version":
            report = json.loads(result.stdout)
            write_json(manifest / "version.json", report)
            if report.get("probe_status") != "ok":
                raise RuntimeError(
                    "Native probe unavailable: " + str(report.get("probe_error"))
                )
            probe = report["probe"]
            descriptor, compiled = probe["backend"], probe["build_info"]
            if descriptor.get("selector") != env["SYMMETRIX_BACKEND"]:
                raise RuntimeError("The probe selected a different backend")
            expected = json.loads(
                subprocess.check_output(
                    [
                        str(python),
                        "-c",
                        "import json; from pathlib import Path; "
                        "from tools._symmetrix_build.source_provenance import "
                        "symmetrix_source_fingerprint, symmetrix_native_source_fingerprint; "
                        "print(json.dumps({'source_content_sha256': symmetrix_source_fingerprint(Path.cwd()), "
                        "'native_source_content_sha256': symmetrix_native_source_fingerprint(Path.cwd())}))",
                    ],
                    cwd=source,
                    env=env,
                    text=True,
                )
            )
            compiled_commit = compiled.get("source_commit")
            if args.qualify_only:
                if descriptor.get("source_commit") != compiled_commit:
                    raise RuntimeError(
                        "Installed binary and descriptor revisions differ"
                    )
                subprocess.run(
                    ["git", "cat-file", "-e", compiled_commit + "^{commit}"],
                    cwd=source,
                    env=env,
                    check=True,
                )
            for identity in (descriptor, compiled):
                if (
                    not args.qualify_only and identity.get("source_commit") != commit
                ) or identity.get("source_dirty") is not False:
                    raise RuntimeError(
                        "Native probe source revision differs from the clean checkout"
                    )
                for key, value in expected.items():
                    if len(value) != 64 or identity.get(key) != value:
                        raise RuntimeError(
                            "Native probe source fingerprint mismatch: " + key
                        )
    if profile == "cpu-omp":
        subprocess.run(
            [
                str(python),
                "deployment/polar/runtime_doctor.py",
                "--checkpoint",
                env["SYMMETRIX_POLAR_MODEL"],
                "--output",
                str(manifest / "doctor-torch-polar.json"),
            ],
            cwd=source,
            env=env,
            check=True,
        )
    subprocess.run(
        [
            str(python),
            "-m",
            "pytest",
            "-q",
            "symmetrix/test/test_polar.py",
            "symmetrix/test/test_polar_features.py",
            "symmetrix/test/test_polar_local_regression.py",
            "--junitxml=" + str(manifest / "polar-tests.xml"),
        ],
        cwd=source,
        env=env,
        check=True,
    )
    write_json(
        manifest / "built.json",
        {
            "source_commit": compiled_commit,
            "qualification_source_commit": commit,
            "qualification_only": args.qualify_only,
            "profile": profile,
            "python": str(python),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "runtime_manifest_sha256": sha256(
                root / "manifests" / ("runtime-" + profile + ".json")
            ),
            "wheel_sha256": {p.name: sha256(p) for p in wheels.glob("*.whl")},
            "python_wrapper_sha256": sha256(python),
            "cuda_loader_links": loader_links,
            "version_probe_sha256": sha256(manifest / "version.json"),
            "extension_path": str(probe["extension"]),
            "extension_sha256": sha256(probe["extension"]),
            "source_fingerprints": expected,
            "submodules": subprocess.check_output(
                ["git", "submodule", "status", "--recursive"],
                cwd=source,
                env=env,
                text=True,
            ).splitlines(),
            "openblas_sha256": {
                str(p): sha256(p)
                for p in sorted(
                    (
                        root / "opt/openblas-omp/lib"
                        if profile == "cpu-omp"
                        else root.parent / "opt/openblas/0.3.28-gcc9.4.0/lib"
                    ).glob("*.so*")
                )
                if p.is_file()
            },
            "cpu_toolchain_manifest_sha256": (
                sha256(root / "manifests/cpu-toolchain.json")
                if profile == "cpu-omp"
                else None
            ),
        },
    )


if __name__ == "__main__":
    main()
