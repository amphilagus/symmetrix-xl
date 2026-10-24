#!/usr/bin/env python3
"""Choose one allocated, already visible CUDA device for an SM80/SM90 profile.

Stdout contains only the selected original CUDA_VISIBLE_DEVICES token. Run in
the allocated Slurm job, capture that token, then use it when launching a fresh
profile process. This script never widens visibility or initializes Kokkos.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
import re
import sys


def allocation_evidence(environment):
    if not environment.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU selection requires a Slurm job allocation")
    evidence = {}
    for key in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        value = environment.get(key, "").strip()
        if value and value.lower() not in {"none", "n/a", "(null)", "-1"}:
            evidence[key] = value
    counts = []
    for key in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS", "SLURM_GPUS_PER_NODE"):
        value = environment.get(key, "").strip()
        if not value:
            continue
        match = re.fullmatch(r"(?:[^:,]+:)?([1-9][0-9]*)", value)
        if match:
            evidence[key] = value
            if key in {"SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE"}:
                counts.append(int(match.group(1)))
    for key in ("SLURM_TRES_PER_NODE", "SLURM_JOB_GRES", "SLURM_STEP_GRES"):
        value = environment.get(key, "").strip()
        if re.search(r"(?:^|,)(?:gres/)?gpu(?::[^:,]+)?:[1-9][0-9]*(?:,|$)", value):
            evidence[key] = value
    if not evidence:
        raise RuntimeError(
            "Slurm job exists but no GPU allocation evidence was found "
            "(SLURM_JOB_GPUS/STEP_GPUS, GPU counts, or GPU TRES/GRES)"
        )
    return evidence, min(counts) if counts else None


def visible_tokens(environment):
    raw = environment.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must explicitly list this Slurm allocation; "
            "refusing to discover or expose devices outside that list"
        )
    tokens = [value.strip() for value in raw.split(",")]
    if any(
        not re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/.-]+", token)
        for token in tokens
    ):
        raise RuntimeError(
            f"Unsupported or disabled CUDA_VISIBLE_DEVICES list: {raw!r}"
        )
    if len(set(tokens)) != len(tokens):
        raise RuntimeError("CUDA_VISIBLE_DEVICES contains duplicate device tokens")
    return tokens


def select(profile, environment=None):
    environment = os.environ if environment is None else environment
    evidence, allocated_count = allocation_evidence(environment)
    tokens = visible_tokens(environment)
    if allocated_count is not None and len(tokens) > allocated_count:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES lists more devices than the Slurm per-node "
            f"GPU allocation ({len(tokens)} > {allocated_count})"
        )
    # Import/runtime diagnostics from dependencies must not contaminate the token
    # captured by a shell. Only inspect ordinals inside existing visibility.
    with redirect_stdout(sys.stderr):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access CUDA inside this GPU allocation")
        count = torch.cuda.device_count()
        if count != len(tokens):
            raise RuntimeError(
                "Cannot establish a one-to-one CUDA visibility mapping: "
                f"Torch sees {count} devices but CUDA_VISIBLE_DEVICES has {len(tokens)} tokens"
            )
        devices = []
        for ordinal, token in enumerate(tokens):
            capability = tuple(torch.cuda.get_device_capability(ordinal))
            properties = torch.cuda.get_device_properties(ordinal)
            devices.append(
                {
                    "local_ordinal": ordinal,
                    "original_token": token,
                    "name": properties.name,
                    "capability": list(capability),
                    "uuid": str(getattr(properties, "uuid", "unavailable")),
                }
            )
    target = {"sm80": (8, 0), "sm90": (9, 0)}[profile]
    candidates = [row for row in devices if tuple(row["capability"]) == target]
    if not candidates:
        visible = ", ".join(
            f"{row['local_ordinal']}:{row['name']} sm{row['capability'][0]}{row['capability'][1]}"
            for row in devices
        )
        raise RuntimeError(
            f"The allocated visible devices do not contain profile {profile}; "
            f"visible devices: {visible}. No visibility expansion or architecture fallback was attempted."
        )
    selected = candidates[0]
    print(
        json.dumps(
            {
                "kind": "POLAR_ALLOCATED_GPU_SELECTION",
                "profile": profile,
                "slurm_job_id": environment["SLURM_JOB_ID"],
                "allocation_evidence": evidence,
                "original_cuda_visible_devices": tokens,
                "visible_devices": devices,
                "selected": selected,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return selected["original_token"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("sm80", "sm90"), required=True)
    args = parser.parse_args(argv)
    try:
        token = select(args.profile)
    except Exception as error:
        print(f"GPU selection failed: {error}", file=sys.stderr)
        return 1
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
