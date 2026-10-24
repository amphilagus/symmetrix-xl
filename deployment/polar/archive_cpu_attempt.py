#!/usr/bin/env python3
"""Archive this deployment's failed cpu-omp attempt before a clean retry.

Default is a read-only plan. --apply checks Slurm and /proc on every cluster
compute node before moving only the enumerated cpu-omp paths. Logs and science
locks remain in place; their hashes and all moves are recorded in the archive.
Use a control interpreter outside the cpu-omp environment being archived.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd
import re
import subprocess
import sys
import time

from bootstrap import sha256, write_json


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def plan(root, toolchain_job):
    manifest = json.loads((root / "manifests/cpu-toolchain.json").read_text())
    require(
        manifest["profile"] == "cpu-omp" and manifest["slurm_job_id"] == toolchain_job,
        "Toolchain attempt does not match the explicitly requested archive",
    )
    build = root / "build/cpu-toolchain" / toolchain_job
    require(
        Path(manifest["build_root"]).resolve() == build,
        "Toolchain manifest build path escaped its task scope",
    )
    relative = [
        "opt/openblas-omp",
        "opt/cpu-omp",
        "opt/base/openmp",
        "build/native/cpu-omp",
        "build/wheels/cpu-omp",
        "build/cache/cpu-omp",
        "manifests/cpu-toolchain.json",
        "manifests/runtime-cpu-omp.json",
        "manifests/build/cpu-omp",
        "benchmark/qualification/cpu-omp",
        "benchmark/performance/cpu-omp",
        str(build.relative_to(root)),
    ]
    paths = [root / value for value in relative if (root / value).exists()]
    for path in paths:
        require(
            not path.is_symlink() and path.resolve().is_relative_to(root),
            "Archive path must be owned by this deployment: " + str(path),
        )
    return paths


def active_jobs(root):
    user = pwd.getpwuid(os.getuid()).pw_name
    identifiers = subprocess.check_output(
        ["squeue", "-h", "-u", user, "-o", "%i"], text=True
    ).split()
    matches = []
    for identifier in identifiers:
        record = subprocess.check_output(
            ["scontrol", "show", "job", "-o", identifier], text=True
        )
        if str(root) in record and (
            "cpu-omp" in record
            or any(
                name in record
                for name in ("cpu_toolchain.sbatch", "qualify_cpu.sbatch")
            )
        ):
            matches.append({"job_id": identifier, "record": record.strip()})
    require(
        not matches,
        "CPU attempt still has active/pending Slurm jobs: " + json.dumps(matches),
    )
    return identifiers


def process_probe_code(paths):
    # Compatible with compute nodes' system Python 2.7; no numerical imports.
    return """import errno,json,os,pwd,socket
prefixes=PREFIXES
hits=[]
errors=[]
excluded=[]
uid=os.getuid()
user=pwd.getpwuid(uid).pw_name
for name in os.listdir('/proc'):
 if not name.isdigit() or int(name)==os.getpid(): continue
 base='/proc/'+name
 try:
  if os.stat(base).st_uid != uid: continue
 except OSError: continue
 values=[]
 unreadable=[]
 denied_only=True
 argv0=''
 for leaf in ('cmdline','maps'):
  try:
   with open(base+'/'+leaf,'rb') as stream: value=stream.read().decode('utf-8','replace')
   values.append(value)
   if leaf=='cmdline': argv0=value.split('\\x00',1)[0]
  except IOError as error:
   if error.errno not in (errno.ENOENT,errno.ESRCH):
    unreadable.append([name,leaf,str(error)])
    denied_only=denied_only and error.errno in (errno.EACCES,errno.EPERM)
 for leaf in ('cwd','exe'):
  try: values.append(os.readlink(base+'/'+leaf))
  except OSError as error:
   if error.errno not in (errno.ENOENT,errno.ESRCH):
    unreadable.append([name,leaf,str(error)])
    denied_only=denied_only and error.errno in (errno.EACCES,errno.EPERM)
 try:
  for fd in os.listdir(base+'/fd'):
   try: values.append(os.readlink(base+'/fd/'+fd))
   except OSError: pass
 except OSError as error:
  if error.errno not in (errno.ENOENT,errno.ESRCH):
   unreadable.append([name,'fd',str(error)])
   denied_only=denied_only and error.errno in (errno.EACCES,errno.EPERM)
 matched=[prefix for prefix in prefixes if any(prefix in value for value in values)]
 if matched: hits.append({'pid':int(name),'paths':matched})
 if not unreadable: continue
 status={}
 comm=''
 try:
  with open(base+'/status') as stream:
   for line in stream:
    key,separator,value=line.partition(':')
    if key in ('Name','Uid'): status[key]=value.strip()
  with open(base+'/comm') as stream: comm=stream.read().strip()
 except (IOError,OSError): pass
 own_uids=status.get('Uid','').split()==[str(uid)]*4
 known_auth=(
  status.get('Name')==comm=='sshd' and argv0.startswith('sshd: '+user+'@')
  or status.get('Name')==comm=='ssh-agent' and argv0 in ('ssh-agent','/usr/bin/ssh-agent')
  or status.get('Name')==comm=='gnome-keyring-d' and argv0=='/usr/bin/gnome-keyring-daemon'
 )
 if denied_only and own_uids and known_auth:
  excluded.append({'pid':int(name),'name':status['Name'],'comm':comm,'uid':uid,
   'program':'sshd' if comm=='sshd' else argv0,
   'unreadable_leaves':[row[1] for row in unreadable]})
 else: errors.extend(unreadable)
print(json.dumps({'node':socket.gethostname().split('.')[0],'uid':uid,'hits':hits,'unreadable_own_processes':errors,'excluded_auth_daemons':excluded}))
""".replace("PREFIXES", repr([str(path) for path in paths]))


def read_proc_evidence(files, root, paths):
    records = {}
    expected_paths = sorted(str(path) for path in paths)
    for filename in files:
        filename = filename.resolve(strict=True)
        require(
            filename.is_relative_to(root / "manifests/archive-probes"),
            "Process evidence must be in this deployment's archive-probes directory",
        )
        record = json.loads(filename.read_text())
        require(
            record.get("kind") == "POLAR_CPU_ARCHIVE_PROCESS_PROBE",
            "Wrong process evidence kind",
        )
        require(
            record["root"] == str(root) and sorted(record["paths"]) == expected_paths,
            "Process evidence prefixes differ from this exact archive plan",
        )
        require(
            record["uid"] == os.getuid()
            and record["user"] == pwd.getpwuid(os.getuid()).pw_name,
            "Process evidence belongs to another user",
        )
        age = time.time() - record["observed_unix"]
        require(
            -30 <= age <= 900
            and abs(filename.stat().st_mtime - record["observed_unix"]) <= 60,
            "Process evidence is stale or its timestamp is inconsistent",
        )
        require(
            bool(record["slurm_job_id"])
            and record["numerical_imports"] is False
            and record["gpu_requested"] is False,
            "Evidence must come from the metadata-only Slurm probe",
        )
        require(
            record["probe_script_sha256"]
            == sha256(Path(__file__).with_name("archive_cpu_probe.py"))
            and record["scanner_script_sha256"] == sha256(__file__),
            "Process evidence used another probe implementation",
        )
        node = record["node"]
        require(
            node == record["slurm_node"]
            and node == record["processes"]["node"]
            and record["processes"]["uid"] == os.getuid(),
            "Process evidence node/user identity disagrees",
        )
        require(node not in records, "Duplicate process evidence for node " + node)
        records[node] = {
            **record["processes"],
            "evidence_path": str(filename),
            "evidence_sha256": sha256(filename),
            "observed_unix": record["observed_unix"],
            "slurm_job_id": record["slurm_job_id"],
        }
    return records


def process_scan(paths, evidence_files=(), root=None):
    code = process_probe_code(paths)
    evidence = read_proc_evidence(evidence_files, root, paths) if evidence_files else {}
    nodes = sorted(
        set(
            subprocess.check_output(
                ["sinfo", "-N", "-h", "-o", "%N"], text=True
            ).split()
        )
    )
    require(bool(nodes), "Slurm returned no compute-node inventory")
    require(
        all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", node) for node in nodes),
        "Slurm node inventory contains an unsupported name",
    )
    require(
        set(evidence).issubset(nodes),
        "Process evidence refers to a node outside the Slurm inventory",
    )

    def scan(node):
        if node in evidence:
            return evidence[node]
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                node,
                "/usr/bin/python",
                "-",
            ],
            input=code,
            capture_output=True,
            text=True,
            timeout=60,
        )
        require(
            result.returncode == 0,
            "Cannot verify live users on node "
            + node
            + ": "
            + result.stderr
            + ". Submit archive_cpu_probe.sbatch on that node, then pass its fresh --proc-evidence JSON.",
        )
        return json.loads(result.stdout)

    local = subprocess.run(
        [sys.executable, "-"], input=code, capture_output=True, text=True, check=True
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        records = [json.loads(local.stdout), *executor.map(scan, nodes)]
    require(
        not any(row["hits"] or row["unreadable_own_processes"] for row in records),
        "Live users or unverified processes prevent archive: " + json.dumps(records),
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--toolchain-job", required=True)
    parser.add_argument("--build-job", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--proc-evidence", type=Path, action="append", default=[])
    args = parser.parse_args()
    require(
        args.toolchain_job.isdigit() and args.build_job.isdigit(),
        "Use numeric attempt job IDs",
    )
    root = args.root.resolve(strict=True)
    require(
        root.name == "mace-symmetrix-xl-polar",
        "This retry tool is restricted to this task's new deployment",
    )
    paths = plan(root, args.toolchain_job)
    require(
        not any(
            Path(sys.executable).resolve().is_relative_to(path)
            or Path.cwd().resolve().is_relative_to(path)
            for path in paths
        ),
        "The control interpreter/cwd must be outside archived paths",
    )
    record = {
        "status": "plan",
        "root": str(root),
        "toolchain_job": args.toolchain_job,
        "failed_build_job": args.build_job,
        "paths": [str(path) for path in paths],
    }
    if not args.apply:
        print(json.dumps(record, indent=2))
        return 0
    record["other_active_jobs"] = active_jobs(root)
    record["process_checks"] = process_scan(paths, args.proc_evidence, root)
    active_jobs(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = root / "archives" / ("cpu-omp-before-lapack-" + stamp)
    require(not archive.exists(), "Archive destination already exists")
    archive.mkdir(parents=True)
    record.update(
        status="archiving",
        archive=str(archive),
        script_sha256=sha256(__file__),
        moves=[],
    )
    record["unchanged_scientific_locks"] = {
        str(path): sha256(path)
        for path in (
            root / "deployment/assets.lock.json",
            root / "deployment/requirements.lock.txt",
        )
    }
    record["preserved_logs"] = {
        str(path): sha256(path)
        for identifier in (args.toolchain_job, args.build_job)
        for path in sorted((root / "logs").glob("*" + identifier + "*"))
        if path.is_file()
    }
    require(
        any(args.build_job in name for name in record["preserved_logs"]),
        "Failed build log was not found",
    )
    record_path = archive / "archive.json"
    write_json(record_path, record)
    try:
        for path in paths:
            target = archive / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)
            record["moves"].append({"source": str(path), "archive": str(target)})
            write_json(record_path, record)
        require(
            all(
                sha256(path) == digest
                for path, digest in record["unchanged_scientific_locks"].items()
            ),
            "Scientific dependency locks changed",
        )
        record["status"] = "archived"
    except Exception as error:
        record.update(status="archive_incomplete", error=repr(error))
        raise
    finally:
        write_json(record_path, record)
    print(json.dumps({"status": record["status"], "archive": str(archive)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
