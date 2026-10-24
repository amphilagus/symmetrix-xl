#!/usr/bin/env python3
"""Collect metadata-only /proc archive evidence where direct SSH is disabled."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pwd
import socket
import subprocess
import sys
import time

from archive_cpu_attempt import plan, process_probe_code, require
from bootstrap import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--toolchain-job", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require(
        bool(os.environ.get("SLURM_JOB_ID")), "Run this probe in a Slurm CPU allocation"
    )
    root = args.root.resolve(strict=True)
    require(
        root.name == "mace-symmetrix-xl-polar",
        "Probe is restricted to this task's deployment",
    )
    paths = plan(root, args.toolchain_job)
    require(
        not any(
            name == "torch" or name.startswith(("torch.", "numpy", "symmetrix"))
            for name in sys.modules
        ),
        "Probe must not import numerical runtimes",
    )
    node = socket.gethostname().split(".")[0]
    slurm_node = os.environ.get(
        "SLURMD_NODENAME", os.environ.get("SLURM_JOB_NODELIST", "")
    ).split(".")[0]
    require(node == slurm_node, "Probe must run on its Slurm allocated node")
    output = (
        args.output
        or root
        / "manifests/archive-probes"
        / f"{node}-{os.environ['SLURM_JOB_ID']}.json"
    ).resolve()
    require(
        output.is_relative_to(root / "manifests/archive-probes")
        and not output.exists(),
        "Use a fresh output in this deployment's archive-probes directory",
    )
    result = subprocess.run(
        [sys.executable, "-"],
        input=process_probe_code(paths),
        capture_output=True,
        text=True,
        check=True,
    )
    processes = json.loads(result.stdout)
    record = {
        "kind": "POLAR_CPU_ARCHIVE_PROCESS_PROBE",
        "root": str(root),
        "paths": [str(path) for path in paths],
        "toolchain_job": args.toolchain_job,
        "observed_unix": time.time(),
        "uid": os.getuid(),
        "user": pwd.getpwuid(os.getuid()).pw_name,
        "node": node,
        "slurm_node": slurm_node,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "numerical_imports": False,
        "gpu_requested": False,
        "probe_script_sha256": sha256(__file__),
        "scanner_script_sha256": sha256(
            Path(__file__).with_name("archive_cpu_attempt.py")
        ),
        "processes": processes,
    }
    write_json(output, record)
    print(
        json.dumps(
            {
                "output": str(output),
                "node": node,
                "hits": processes["hits"],
                "unreadable_own_processes": processes["unreadable_own_processes"],
            }
        )
    )
    return (
        0 if not processes["hits"] and not processes["unreadable_own_processes"] else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
