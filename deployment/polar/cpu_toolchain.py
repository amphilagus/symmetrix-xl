#!/usr/bin/env python3
"""Build a fresh OpenMP OpenBLAS and stage verified GCC12 GOMP under Slurm."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from bootstrap import environment, library_paths, sha256, write_json


OPENBLAS_SHA256 = "f1003466ad074e9b0c8d421a204121100b0751c96fc6fcf3d1456bd12f8a00a1"
GOMP_SHA256 = "b16672394348cbc11df46cc6445a7ef4f3b3112de8ca48b8c548e3e43db6652e"
FORTRAN_RUNTIME_SHA256 = {
    "libgfortran.so.5": "7a27115b7aa238c109b65f185273919ce4e9d86ceaa8e30e1c3a8e8bb618a3e1",
    "libquadmath.so.0": "ba9491cfb23a6d8bc2263849f76435072cbb7203658a1367cdd782d586c770d3",
}


def verified_copy(source, target, expected):
    if sha256(source) != expected:
        raise RuntimeError(f"Source digest mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != expected:
            raise RuntimeError(f"Existing staged file digest mismatch: {target}")
    else:
        shutil.copy2(source, target)
    if sha256(target) != expected:
        raise RuntimeError(f"Staged copy digest mismatch: {target}")


def build(root, donor, archive, jobs):
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise RuntimeError("CPU toolchain compilation requires a Slurm compute job")
    if root == donor or root.parent != donor.parent:
        raise ValueError("Deployment must be a distinct sibling of the read-only donor")
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if not 1 <= jobs <= allocated:
        raise ValueError(f"Requested {jobs} workers but Slurm allocated {allocated}")
    env = environment(root, donor)
    for key in (
        "NO_LAPACK",
        "NO_LAPACKE",
        "NOFORTRAN",
        "ONLY_CBLAS",
        "MAKEFLAGS",
        "MFLAGS",
    ):
        env.pop(key, None)
    compilers = {}
    for name in ("CC", "CXX", "FC"):
        compiler = Path(env[name])
        version = subprocess.check_output(
            [str(compiler), "-dumpfullversion"], env=env, text=True
        ).strip()
        if version != "12.4.0":
            raise RuntimeError(f"Expected GCC12.4.0, found {name}={version}")
        compilers[name] = {
            "path": str(compiler),
            "sha256": sha256(compiler),
            "version": version,
        }
    gomp = Path(
        subprocess.check_output(
            [env["CC"], "-print-file-name=libgomp.so.1"], env=env, text=True
        ).strip()
    ).resolve(strict=True)
    if not gomp.is_relative_to(donor / "opt/gcc/12.4.0"):
        raise RuntimeError(
            f"Compiler GOMP is outside the verified GCC toolchain: {gomp}"
        )
    own_gomp = root / "opt/base/openmp/libgomp.so.1.0.0"
    verified_copy(gomp, own_gomp, GOMP_SHA256)
    for name in ("libgomp.so", "libgomp.so.1"):
        alias = own_gomp.parent / name
        if alias.is_symlink() and os.readlink(alias) == own_gomp.name:
            continue
        if alias.exists() or alias.is_symlink():
            raise FileExistsError(f"Unexpected existing OpenMP alias: {alias}")
        alias.symlink_to(own_gomp.name)
    fortran_runtime = {}
    for name in ("libgfortran.so.5", "libquadmath.so.0", "libgcc_s.so.1"):
        source = Path(
            subprocess.check_output(
                [env["FC"], "-print-file-name=" + name], env=env, text=True
            ).strip()
        ).resolve(strict=True)
        if not source.is_relative_to(donor / "opt/gcc/12.4.0"):
            raise RuntimeError(
                f"Fortran dependency escaped the compiler toolchain: {source}"
            )
        expected = FORTRAN_RUNTIME_SHA256.get(name, sha256(source))
        if sha256(source) != expected:
            raise RuntimeError(f"Fixed compiler runtime digest mismatch: {source}")
        owned = root / "opt/base/gcc-runtime/lib64" / name
        mode = "reuse_verified_owned_runtime"
        if not owned.is_file() or sha256(owned) != expected:
            # Keep shared runtime files used by existing CPU/GPU jobs immutable.
            owned = own_gomp.parent / name
            verified_copy(source, owned, expected)
            mode = "stage_cpu_only_runtime"
        fortran_runtime[name] = {
            "compiler_source": str(source),
            "path": str(owned),
            "sha256": expected,
            "mode": mode,
        }
    staged_archive = root / "src/dependency-archives/OpenBLAS-0.3.28.tar.gz"
    verified_copy(archive, staged_archive, OPENBLAS_SHA256)
    prefix = root / "opt/openblas-omp"
    build_root = root / "build/cpu-toolchain" / job
    if prefix.exists() or build_root.exists():
        raise FileExistsError(
            "OpenBLAS qualification requires a fresh install/build directory"
        )
    build_root.mkdir(parents=True)
    with tarfile.open(staged_archive) as bundle:
        bundle.extractall(build_root, filter="data")
    tree = build_root / "OpenBLAS-0.3.28"
    flags = [
        "TARGET=HASWELL",
        "DYNAMIC_ARCH=1",
        "USE_OPENMP=1",
        "NUM_THREADS=64",
        "NO_AFFINITY=1",
        "NO_SHARED=0",
        "NO_LAPACKE=1",
        "CFLAGS=-O2 -fPIC",
        "FFLAGS=-O2 -fPIC",
        "CC=" + env["CC"],
        "FC=" + env["FC"],
    ]
    commands = [
        ["make", "-j" + str(jobs), *flags],
        ["make", *flags, "PREFIX=" + str(prefix), "install"],
    ]
    for command in commands:
        subprocess.run(command, cwd=tree, env=env, check=True)
    library = prefix / "lib/libopenblas.so"
    if not library.is_file():
        raise RuntimeError("OpenBLAS shared library was not installed")
    probe_source = build_root / "probe.c"
    probe_source.write_text(
        '#include <stdio.h>\n#include <omp.h>\n#include "cblas.h"\n'
        "extern char *openblas_get_config(void);\nextern int openblas_get_parallel(void);\n"
        "int main(void) { double a=2,b=3,c=0;\n"
        "cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,1,1,1,1,&a,1,&b,1,0,&c,1);\n"
        'printf("%s\\nparallel=%d\\ngemm=%.17g\\nomp=%d\\n",openblas_get_config(),openblas_get_parallel(),c,omp_get_max_threads());\n'
        "return !(openblas_get_parallel()==2 && c==6); }\n"
    )
    probe = build_root / "probe"
    subprocess.run(
        [
            env["CC"],
            "-fopenmp",
            str(probe_source),
            "-I" + str(prefix / "include"),
            "-L" + str(prefix / "lib"),
            "-lopenblas",
            "-o",
            str(probe),
        ],
        env=env,
        check=True,
    )
    loader = root / "opt/base/glibc-2.28/lib/ld-2.28.so"
    probe_env = env.copy()
    probe_env.update(
        LD_PRELOAD=str(own_gomp),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_MAX_ACTIVE_LEVELS="1",
    )
    probe_command = [
        str(loader),
        "--library-path",
        ":".join(library_paths(root, "cpu-omp")),
        str(probe),
    ]
    probe_result = subprocess.check_output(probe_command, env=probe_env, text=True)
    dependency_list = subprocess.check_output(
        [
            str(loader),
            "--library-path",
            ":".join(library_paths(root, "cpu-omp")),
            "--list",
            str(probe),
        ],
        env=probe_env,
        text=True,
    )
    lapack_source = build_root / "probe_lapack.f90"
    lapack_source.write_text(
        "program probe_lapack\n"
        "  implicit none\n"
        "  complex :: a(2,2), work(10)\n"
        "  real :: w(2), rwork(4)\n"
        "  integer :: info\n"
        "  a = (0.0,0.0)\n"
        "  a(1,1) = (1.0,0.0)\n"
        "  a(2,2) = (3.0,0.0)\n"
        "  call cheev('N','U',2,a,2,w,work,10,rwork,info)\n"
        "  if (info /= 0) stop 1\n"
        "  if (abs(w(1)-1.0)>1.e-5 .or. abs(w(2)-3.0)>1.e-5) stop 2\n"
        "  print *, 'cheev_ linked and executed; eigenvalues=', w\n"
        "end program probe_lapack\n"
    )
    lapack_probe = build_root / "probe_lapack"
    lapack_compile = [
        env["FC"],
        "-fopenmp",
        str(lapack_source),
        "-L" + str(prefix / "lib"),
        "-lopenblas",
        "-o",
        str(lapack_probe),
    ]
    subprocess.run(lapack_compile, env=env, check=True)
    lapack_result = subprocess.check_output(
        [
            str(loader),
            "--library-path",
            ":".join(library_paths(root, "cpu-omp")),
            str(lapack_probe),
        ],
        env=probe_env,
        text=True,
    )
    files = {
        str(p.relative_to(prefix)): sha256(p)
        for p in sorted(prefix.rglob("*"))
        if p.is_file()
    }
    write_json(
        root / "manifests/cpu-toolchain.json",
        {
            "schema_version": 1,
            "status": "BUILT",
            "slurm_job_id": job,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "profile": "cpu-omp",
            "script_sha256": sha256(__file__),
            "source_archive": str(archive),
            "source_archive_sha256": OPENBLAS_SHA256,
            "staged_archive": str(staged_archive),
            "source_version": "0.3.28",
            "source_release": "https://github.com/OpenMathLib/OpenBLAS/releases/tag/v0.3.28",
            "compilers": compilers,
            "commands": commands,
            "build_root": str(build_root),
            "openmp": {
                "compiler_source": str(gomp),
                "path": str(own_gomp),
                "sha256": GOMP_SHA256,
            },
            "fortran_runtime": fortran_runtime,
            "openblas": {
                "prefix": str(prefix),
                "lapack_enabled": True,
                "lapack_configuration": "upstream default; inherited disabling flags removed",
                "lapack_probe_compile": lapack_compile,
                "lapack_probe": lapack_result,
                "files": files,
                "probe": probe_result,
                "dependencies": dependency_list,
            },
            "qualification_pending": "fresh native build and strict doctor after Torch+SymmetrixPolar initialization",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--donor", required=True, type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    root, donor = args.root.resolve(), args.donor.resolve()
    archive = args.source_archive or donor.parent / "src/v0.3.28.tar.gz"
    build(root, donor, archive.resolve(strict=True), args.jobs)


if __name__ == "__main__":
    main()
