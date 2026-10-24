#!/usr/bin/env python3
"""Create independent POLAR test runtimes from a verified, read-only donor.

The donor contains the previously qualified public dependencies. Virtual
environments are always created afresh; only immutable runtime components,
locked wheels and application source archives are reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import zipfile


RUNTIME_ID = "polar-mace0316-torch260cu126-cueq091-a4470e17e377"
MODEL_SHA256 = "e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def stage(root, donor):
    if root == donor or root.parent != donor.parent:
        raise ValueError("The new deployment must be a distinct sibling of the donor")
    for name in (
        "src",
        "opt",
        "build",
        "deployment",
        "forcefields",
        "benchmark",
        "logs",
        "manifests",
        "tests",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    capsule = donor / "opt/polar" / RUNTIME_ID
    base = root / "opt/base"
    base.mkdir(exist_ok=True)
    inventory = {}
    for name in ("python-base", "glibc-2.28", "gcc-runtime"):
        target = base / name
        if not target.exists():
            shutil.copytree(capsule / name, target, symlinks=True)
        inventory[name] = {
            str(p.relative_to(target)): sha256(p)
            for p in sorted(target.rglob("*"))
            if p.is_file()
        }
    checkpoint = root / "forcefields/polar1s/MACE-POLAR-1-S.model"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    original = donor / "forcefields/polar1s" / checkpoint.name
    if sha256(original) != MODEL_SHA256:
        raise ValueError("Donor checkpoint differs from the official locked checkpoint")
    if not checkpoint.exists():
        shutil.copy2(original, checkpoint)
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("Staged checkpoint changed")
    for name in ("assets.lock.json", "requirements.lock.txt"):
        target = root / "deployment" / name
        original = donor / "deployment/polar" / name
        if target.exists() and sha256(target) != sha256(original):
            raise ValueError("Existing dependency lock differs from donor")
        shutil.copy2(original, target)
    write_json(
        root / "manifests/staging.json",
        {
            "schema_version": 1,
            "donor": str(donor),
            "root": str(root),
            "donor_runtime_id": RUNTIME_ID,
            "immutable_components": inventory,
            "checkpoint_sha256": MODEL_SHA256,
            "locks": {
                name: sha256(root / "deployment" / name)
                for name in ("assets.lock.json", "requirements.lock.txt")
            },
        },
    )


def library_paths(root, profile):
    base = root / "opt/base"
    site = root / "opt" / profile / "lib/python3.12/site-packages"
    native = [str(p) for p in sorted((site / "nvidia").glob("*/lib"))]
    native += [str(p) for p in sorted(site.glob("cuequivariance_ops*/lib"))]
    return [
        str(base / "glibc-2.28/lib"),
        str(base / "gcc-runtime/lib64"),
        str(base / "python-base/lib"),
        str(root.parent / "opt/openblas/0.3.28-gcc9.4.0/lib"),
        str(root / "opt" / profile / "lib/polar-cuda-loader-links"),
        str(site / "torch/lib"),
        *native,
        "/usr/lib64",
        "/lib64",
    ]


def install_cuda_loader_links(root, profile):
    """Resolve dlopen's unversioned names only against this locked profile."""
    prefix = root / "opt" / profile
    site = prefix / "lib/python3.12/site-packages"
    directory = prefix / "lib/polar-cuda-loader-links"
    directory.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for alias, component, filename in (
        ("libcudart.so", "cuda_runtime", "libcudart.so.12"),
        ("libnvrtc.so", "cuda_nvrtc", "libnvrtc.so.12"),
        ("libnvJitLink.so", "nvjitlink", "libnvJitLink.so.12"),
    ):
        target = site / "nvidia" / component / "lib" / filename
        link = directory / alias
        if link.exists() and not link.is_symlink():
            raise FileExistsError(
                f"Refusing to replace non-symlink CUDA loader alias: {link}"
            )
        if not target.is_file():
            # An optional wheel may be absent during initial environment
            # creation. Never retain a stale alias to another environment.
            if link.is_symlink():
                link.unlink()
            continue
        resolved = target.resolve(strict=True)
        if not resolved.is_relative_to(prefix.resolve()):
            raise RuntimeError(f"CUDA loader target escapes its own profile: {target}")
        relative = os.path.relpath(target, directory)
        if link.is_symlink() and os.readlink(link) != relative:
            link.unlink()
        if not link.is_symlink():
            link.symlink_to(relative)
        inventory[alias] = {
            "path": str(link),
            "link_target": relative,
            "source": str(target),
            "resolved_source": str(resolved),
            "source_sha256": sha256(resolved),
        }
    return inventory


def install_wrappers(root, profile):
    prefix = root / "opt" / profile
    cuda_loader_links = install_cuda_loader_links(root, profile)
    loader = root / "opt/base/glibc-2.28/lib/ld-2.28.so"
    script = "#!/bin/bash\nset -euo pipefail\n"
    script += "unset PYTHONHOME PYTHONPATH LD_PRELOAD LD_AUDIT\n"
    script += 'if [ "${SYMMETRIX_POLAR_BUILD_MODE:-0}" != 1 ]; then unset LD_LIBRARY_PATH; fi\n'
    script += "export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 LC_ALL=C LANG=C\n"
    script += "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
    script += (
        "export NVIDIA_TF32_OVERRIDE=0 PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1\n"
    )
    for key, suffix in (
        ("XDG_CACHE_HOME", "xdg"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("TORCH_EXTENSIONS_DIR", "torch"),
        ("CUEQUIVARIANCE_CACHE_DIR", "cueq"),
        ("SYMMETRIX_JIT_CACHE", "symmetrix"),
    ):
        target = root / "build/cache" / profile / suffix
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        script += "export " + key + "=" + shlex.quote(str(target)) + "\n"
    script += "exec " + shlex.quote(str(loader)) + " --library-path "
    script += shlex.quote(":".join(library_paths(root, profile))) + " "
    script += shlex.quote(str(prefix / "bin/python3.real")) + ' "$@"\n'
    for name in ("python", "python3", "python3.12"):
        path = prefix / "bin" / name
        path.unlink(missing_ok=True)
        path.write_text(script)
        path.chmod(0o755)
    site = prefix / "lib/python3.12/site-packages"
    (site / "sitecustomize.py").write_text(
        "import sys\n" + "sys.executable = " + repr(str(prefix / "bin/python")) + "\n"
    )
    return cuda_loader_links


def environment(root, donor):
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "LD_PRELOAD", "LD_AUDIT"):
        env.pop(key, None)
    gcc = donor / "opt/gcc/12.4.0"
    public = donor.parent / "opt"
    cuda = public / "nvhpc/Linux_x86_64/24.3/cuda/12.3"
    env.update(
        CC=str(gcc / "bin/gcc"),
        CXX=str(gcc / "bin/g++"),
        FC=str(gcc / "bin/gfortran"),
        LC_ALL="C",
        LANG="C",
        PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUDA_HOME=str(cuda),
        CUDA_PATH=str(cuda),
        CUDAToolkit_ROOT=str(cuda),
        PATH=":".join(
            map(
                str,
                [
                    donor / "opt/git/bin",
                    gcc / "bin",
                    public / "cmake/3.29.6-gcc9.4.0/bin",
                    cuda / "bin",
                    Path("/usr/bin"),
                    Path("/bin"),
                ],
            )
        ),
        LD_LIBRARY_PATH=":".join(
            map(
                str,
                [
                    gcc / "lib64",
                    donor / "opt/python/lib",
                    public / "openblas/0.3.28-gcc9.4.0/lib",
                    cuda / "lib64",
                ],
            )
        ),
        CMAKE_PREFIX_PATH=str(public / "openblas/0.3.28-gcc9.4.0"),
        LIBRARY_PATH=str(public / "openblas/0.3.28-gcc9.4.0/lib"),
        CPATH=str(public / "openblas/0.3.28-gcc9.4.0/include"),
        PIP_NO_INDEX="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        UV_OFFLINE="1",
    )
    return env


def prepare(root, donor, profile):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Environment installation requires a Slurm job")
    lock = json.loads((root / "deployment/assets.lock.json").read_text())
    source = donor / "src/polar" / RUNTIME_ID
    prefix = root / "opt" / profile
    env = environment(root, donor)
    base = root / "opt/base"
    loader = base / "glibc-2.28/lib/ld-2.28.so"
    command = [
        str(loader),
        "--library-path",
        ":".join(library_paths(root, profile)),
        str(base / "python-base/bin/python3"),
    ]
    if not (prefix / "pyvenv.cfg").exists():
        subprocess.run(
            [*command, "-m", "venv", "--copies", "--without-pip", str(prefix)],
            check=True,
            env=env,
        )
    real = prefix / "bin/python3.real"
    if not real.exists():
        shutil.copy2(prefix / "bin/python3", real, follow_symlinks=True)
    install_wrappers(root, profile)
    site = prefix / "lib/python3.12/site-packages"
    pip = next(p for p in lock["packages"] if p["name"] == "pip")
    with zipfile.ZipFile(source / "wheels" / pip["filename"]) as archive:
        archive.extractall(site)
    python = str(prefix / "bin/python")
    find_links = [
        source / "wheels",
        donor / "src/wheels_symmetrix",
        root / "src/wheels",
    ]
    links = [
        part
        for path in find_links
        if path.exists()
        for part in ("--find-links", str(path))
    ]
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(source / "wheels"),
            "--require-hashes",
            "-r",
            str(root / "deployment/requirements.lock.txt"),
        ],
        env=env,
        check=True,
    )
    for asset in lock["sources"]:
        if asset["name"] == "glibc":
            continue
        archive = source / "downloads" / asset["filename"]
        if sha256(archive) != asset["sha256"]:
            raise ValueError("Application source archive hash mismatch")
        unpacked = root / "src/dependency-sources" / profile
        unpacked.mkdir(parents=True, exist_ok=True)
        tree = unpacked / asset["directory"]
        if not tree.exists():
            with tarfile.open(archive) as bundle:
                bundle.extractall(unpacked, filter="data")
        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--no-build-isolation",
                str(tree),
            ],
            check=True,
            env=env,
        )
    # Reproduce the two already-qualified, API-only MACE/graph compatibility fixes.
    patched = (
        donor
        / "opt/polar"
        / RUNTIME_ID
        / "venv/lib/python3.12/site-packages/mace/modules/extensions.py"
    )
    destination = site / "mace/modules/extensions.py"
    before = sha256(destination)
    shutil.copy2(patched, destination)
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-index",
            *links,
            "scikit-build-core",
            "pybind11",
            "ninja",
            "pytest",
            "ruff",
        ],
        check=True,
        env=env,
    )
    cuda_loader_links = install_wrappers(root, profile)
    subprocess.run([python, "-m", "pip", "check"], check=True, env=env)
    write_json(
        root / "manifests" / ("runtime-" + profile + ".json"),
        {
            "schema_version": 1,
            "profile": profile,
            "python": python,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "staging_sha256": sha256(root / "manifests/staging.json"),
            "mace_compatibility": {
                "before_sha256": before,
                "after_sha256": sha256(destination),
                "qualified_donor_file": str(patched),
            },
            "requirements_sha256": sha256(root / "deployment/requirements.lock.txt"),
            "cuda_loader_links": cuda_loader_links,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "prepare"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--profile", choices=("cpu", "sm80", "sm90"), default="cpu")
    args = parser.parse_args()
    root, donor = args.root.resolve(), args.donor.resolve()
    if root == donor or root.parent != donor.parent:
        raise ValueError("Deployment must be a distinct sibling of the read-only donor")
    if args.action == "stage":
        stage(root, donor)
    else:
        prepare(root, donor, args.profile)


if __name__ == "__main__":
    main()
