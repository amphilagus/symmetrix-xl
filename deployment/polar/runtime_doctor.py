#!/usr/bin/env python3
"""Qualify one OpenMP runtime after real Torch and native POLAR inference."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Pin before importing numerical libraries; exclude the SMT sibling.
    allocated = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {allocated[0]})
    import torch
    from ase import Atoms
    import symmetrix
    from symmetrix import SymmetrixPolar
    from symmetrix.runtime_diagnostics import openmp_runtime_diagnostics

    atoms = Atoms(
        "LiOH2",
        positions=[[1, 1, 1], [3, 1, 1], [3.9572, 1, 1], [2.76, 1.927, 1]],
        cell=[8, 8, 8],
        pbc=True,
    )
    calc = SymmetrixPolar(args.checkpoint, device="cpu", bridge="host")
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    native = symmetrix.load_backend()
    report = openmp_runtime_diagnostics(
        native_module=native,
        strict=True,
        expected_threads=1,
        allocated_cpus=[allocated[0]],
    )
    report["qualification_context"] = {
        "torch_version": torch.__version__,
        "energy_eV": energy,
        "maximum_force_eV_A": float(abs(forces).max()),
        "executor": {
            "calculator": type(calc).__name__,
            "native_class": type(calc.evaluator).__module__
            + "."
            + type(calc.evaluator).__name__,
            "kokkos_execution_space": calc.native_execution_space,
            "bridge": calc.bridge,
        },
        "native_build_info": calc.native_build_info,
        "allocated_before_pinning": allocated,
        "actual_affinity": sorted(os.sched_getaffinity(0)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    atoms.calc = None
    del calc
    gc.collect()
    if report["status"] != "ok":
        raise RuntimeError(
            "Strict Torch/POLAR OpenMP qualification failed; see " + str(args.output)
        )


if __name__ == "__main__":
    main()
