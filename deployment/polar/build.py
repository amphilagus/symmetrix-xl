#!/usr/bin/env python3
"""Build one independently qualified POLAR native backend under Slurm."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from bootstrap import environment, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--profile", choices=("cpu", "sm80", "sm90"), required=True)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Native compilation requires a Slurm compute job")
    root, donor = args.root.resolve(), args.donor.resolve()
    if root == donor or root.parent != donor.parent:
        raise ValueError("Build root must be independent of the donor")
    profile = args.profile
    source = root / "src/symmetrix-xl"
    env = environment(root, donor)
    env["SYMMETRIX_POLAR_BUILD_MODE"] = "1"
    env["PIP_FIND_LINKS"] = " ".join(str(p) for p in (
        root / "src/wheels", donor / "src/wheels_symmetrix",
        donor / "src/polar/polar-mace0316-torch260cu126-cueq091-a4470e17e377/wheels"))
    env["CMAKE_ARGS"] = "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
    python = root / "opt" / profile / "bin/python"
    wheels = root / "build/wheels"
    wheels.mkdir(exist_ok=True)
    manifest = root / "manifests/build" / profile
    manifest.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, env=env, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, env=env, text=True)
    if dirty:
        raise RuntimeError("Native build source has tracked modifications")
    common = ["--generator", "Unix Makefiles", "--build-root", str(root / "build/native" / profile), "--jobs", str(args.jobs)]
    frontend = [str(python), "tools/symmetrix_build.py"]
    if profile == "cpu":
        command = [*frontend, "wheel", "--backend", "cpu", "--cpu-target", "x86-64-v3", "--blas", "openblas",
                   "--cxx", env["CXX"], "--wheel-dir", str(wheels), *common]
        subprocess.run(command, cwd=source, env=env, check=True)
        candidates = list(wheels.glob("symmetrix_xl-0.1.1-*.whl"))
        if len(candidates) != 1:
            raise RuntimeError("Expected exactly one CPU frontend wheel")
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(candidates[0])], env=env, check=True)
    else:
        candidates = list(wheels.glob("symmetrix_xl-0.1.1-*.whl"))
        if len(candidates) != 1:
            raise RuntimeError("Build the matching CPU frontend wheel first")
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(candidates[0])], env=env, check=True)
        site = root / "opt" / profile / "lib/python3.12/site-packages"
        cublas = site / "nvidia/cublas"
        devel = root / "build/cublas" / profile
        (devel / "lib64").mkdir(parents=True, exist_ok=True)
        for path, target in ((devel / "include", cublas / "include"),
                             (devel / "lib64/libcublas.so", cublas / "lib/libcublas.so.12"),
                             (devel / "lib64/libcublasLt.so", cublas / "lib/libcublasLt.so.12")):
            path.unlink(missing_ok=True)
            path.symlink_to(target)
        env["CMAKE_ARGS"] += " -DCUBLAS_ROOT=" + str(devel)
        env["CMAKE_ARGS"] += " -DCUDA_cublas_LIBRARY=" + str(devel / "lib64/libcublas.so")
        env["CMAKE_ARGS"] += " -DCUDA_cublasLt_LIBRARY=" + str(devel / "lib64/libcublasLt.so")
        gpu_lite = next((donor / "src").glob("gpu-lite-*"))
        env["CMAKE_ARGS"] += " -DFETCHCONTENT_SOURCE_DIR_GPU_LITE=" + str(gpu_lite)
        command = [*frontend, "install", "--backend", "cuda", "--arch", profile,
                   "--cuda-root", env["CUDA_HOME"], "--host-cxx", env["CXX"], *common]
        subprocess.run(command, cwd=source, env=env, check=True)
    env.pop("SYMMETRIX_POLAR_BUILD_MODE", None)
    env["SYMMETRIX_BACKEND"] = "cpu" if profile == "cpu" else "cuda12-" + profile
    for label, command in (("version", [str(python), "-m", "symmetrix.cli.main", "version", "--probe", "--json"]),
                           ("pip-check", [str(python), "-m", "pip", "check"])):
        result = subprocess.run(command, env=env, cwd=source, capture_output=True, text=True)
        (manifest / (label + ".txt")).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(label + " probe failed")
    subprocess.run([str(python), "-m", "pytest", "-q", "symmetrix/test/test_polar.py", "symmetrix/test/test_polar_features.py"], cwd=source, env=env, check=True)
    write_json(manifest / "built.json", {"source_commit": commit, "profile": profile,
        "python": str(python), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "runtime_manifest_sha256": sha256(root / "manifests" / ("runtime-" + profile + ".json")),
        "wheel_sha256": {p.name: sha256(p) for p in wheels.glob("*.whl")}})


if __name__ == "__main__":
    main()
