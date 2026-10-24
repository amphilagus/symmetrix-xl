#!/usr/bin/env python3
"""Generate an evidence-only Markdown POLAR report without importing model code.

Read deployment manifests and experiment records; never run a model, a test,
or a benchmark. Missing and partial evidence stays pending. --output must be
outside every Git checkout, including the deployed source checkout.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


PROFILES = ("cpu", "sm80", "sm90")
WATERS = (64, 128, 256, 512)
LABELS = {"PASSED": "通过", "PENDING": "待完成", "FAILED": "失败"}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def fmt(value):
    if number(value):
        return f"{value:.8g}"
    return "未提供" if value is None else str(value)


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def link(path):
    return f"[{path.name}](<{path.resolve()}>)"


def table(headers, rows):
    if not rows:
        return "未找到记录。\n"
    lines = [
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    return "\n".join(lines) + "\n"


class Evidence:
    def __init__(self, root):
        self.root = root
        self.records = []
        self.file_inventory = {}
        self.superseded_failures = []

    def read(self, path, *, version=False):
        if not path.is_file():
            return None, None
        self.file_inventory[str(path)] = file_hash(path)
        text = path.read_text(errors="replace")
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            value = None
            if version:
                decoder = json.JSONDecoder()
                for index, char in enumerate(text):
                    if char != "{":
                        continue
                    try:
                        candidate, _ = decoder.raw_decode(text[index:])
                    except json.JSONDecodeError:
                        continue
                    if isinstance(candidate, dict) and (
                        "probe_status" in candidate or "probe" in candidate
                    ):
                        value = candidate
                        break
            if value is None:
                return None, f"无法解析 JSON：{error}"
        if not isinstance(value, dict):
            return None, "JSON 顶层不是对象"
        self.records.append((path, value))
        return value, None

    def first(self, paths, *, version=False):
        for path in paths:
            if path.is_file():
                value, error = self.read(path, version=version)
                return path, value, error
        return paths[0], None, None


def explicit_status(data):
    if data is None:
        return "PENDING"
    status = str(data.get("status", "")).upper()
    if data.get("error") or status in ("FAILED", "FAILURE", "REJECTED", "ERROR"):
        return "FAILED"
    if (
        status in ("ACCEPTED", "PASSED", "COMPLETE", "COMPLETED", "OK", "SUCCESS")
        or data.get("passed") is True
    ):
        return "PASSED"
    if data.get("passed") is False:
        return "FAILED"
    return "PENDING"


def failed_marker(evidence, directory, completed_path=None, complete=False):
    for name in ("failure.json", "benchmark-failure.json"):
        path = directory / name
        value, error = evidence.read(path)
        if error or value is not None:
            if (
                complete
                and completed_path
                and completed_path.is_file()
                and completed_path.stat().st_mtime_ns > path.stat().st_mtime_ns
            ):
                evidence.superseded_failures.append(
                    (path, error or str(value.get("error", "失败记录")), completed_path)
                )
                continue
            return path, error or str(value.get("error", "明确的失败记录"))
    return None, None


def measurements(data):
    if not data:
        return []
    if isinstance(data.get("cases"), dict):
        return list(data["cases"].values())
    return [row for row in data.get("samples", []) if isinstance(row, dict)]


def maximum(rows, key):
    values = [row.get(key) for row in rows if number(row.get(key))]
    return max(values) if values else None


def population_errors(rows):
    charges, spins = [], []
    for row in rows:
        for name in ("original_populations", "candidate_populations"):
            record = row.get(name, {})
            if number(record.get("maximum_charge_error_e")):
                charges.append(record["maximum_charge_error_e"])
            else:
                charges.extend(
                    abs(value - 1.0)
                    for value in record.get("charge_totals", [])
                    if number(value)
                )
            if number(record.get("maximum_spin_population_error")):
                spins.append(record["maximum_spin_population_error"])
            else:
                spins.extend(
                    abs(value)
                    for value in record.get("spin_populations", [])
                    if number(value)
                )
    return max(charges) if charges else None, max(spins) if spins else None


def physics_status(data, error, phase, waters):
    if error:
        return "FAILED", error
    if data is None:
        return "PENDING", "未找到记录"
    if data.get("error"):
        return "FAILED", str(data["error"])
    actual_waters = data.get("structure", {}).get("waters")
    if actual_waters is not None and actual_waters != waters:
        return "FAILED", f"实际 water={actual_waters}，与目标 {waters} 不符"
    rows = measurements(data)
    if any(row.get("passed") is False for row in rows):
        return "FAILED", "至少一个实际采样未通过"
    if phase.startswith("md"):
        required = int(phase[2:])
        last = max((row.get("step", -1) for row in rows), default=-1)
        if (
            data.get("steps_requested") != required
            or last < required
            or data.get("passed") is not True
        ):
            return (
                "PENDING",
                f"最后采样 step={last}；目标 {required} 步，尚无完整通过记录",
            )
        if data.get("timestep_fs") != 0.25:
            return "FAILED", f"实际 timestep={data.get('timestep_fs')} fs，不是 0.25 fs"
    elif not {"base", "translated", "wrapped"}.issubset(data.get("cases", {})):
        return "PENDING", "缺少 base / translated / wrapped 完整对照"
    if data.get("passed") is not True:
        return explicit_status(data), "没有明确的完整通过标记"
    if not rows:
        return "PENDING", "没有数值采样证据"
    energy = maximum(rows, "absolute_total_energy_error_eV")
    force = maximum(rows, "maximum_force_component_error_eV_A")
    charge, spin = population_errors(rows)
    if any(value is None for value in (energy, force, charge, spin)):
        return "PENDING", "能量、力或守恒误差证据不完整"
    if energy > 1e-4 or force > 1e-5 or charge > 1e-9 or spin > 1e-9:
        return "FAILED", "实际误差超过正式阈值"
    regions = ("Li", "hydration_shell_water", "remaining_water")
    if any(
        any(region not in row.get("force_error_by_region", {}) for region in regions)
        for row in rows
    ):
        return "PENDING", "总体对照有记录，三类区域力误差尚未齐全"
    return "PASSED", "完整数值记录满足正式阈值"


def preparation_status(data, error, waters):
    if error:
        return "FAILED", error
    status = explicit_status(data)
    if status == "FAILED":
        return status, str(data.get("error", "准备阶段失败"))
    if not data:
        return "PENDING", "未找到准备记录"
    if data.get("structure", {}).get("waters") != waters:
        return "FAILED", "准备记录的实际 water 数不匹配"
    if not data.get("optimization", {}).get("converged"):
        return "PENDING", "尚无 BFGS 收敛证据；仅初始构型不计完整准备"
    optimization = data["optimization"]
    if (
        not number(optimization.get("maximum_force_eV_A"))
        or optimization["maximum_force_eV_A"] > 0.05
    ):
        return "PENDING", "尚无最终 fmax ≤ 0.05 eV/Å 的数值证据"
    if number(optimization.get("steps")) and optimization["steps"] > 2000:
        return "PENDING", "优化超过正式协议的 2000 步上限"
    nvt = data.get("nvt", {})
    if nvt.get("steps", 0) < 500 or nvt.get("timestep_fs") != 0.25:
        return "PENDING", "尚无 500 步、0.25 fs NVT 完成证据"
    if status != "PASSED":
        return "PENDING", "优化/NVT 有记录，但缺少完成标记"
    return "PASSED", "原模型固定晶胞优化收敛，NVT 完成"


def junit(evidence, profile):
    roots = (
        evidence.root / "tests",
        evidence.root / "manifests/tests",
        evidence.root / "manifests/build" / profile,
    )
    paths = set()
    for base in roots:
        if base.is_dir():
            for path in base.rglob("*.xml"):
                if (
                    base.name == profile
                    or profile in path.parts
                    or profile in path.name
                ):
                    paths.add(path)
    rows = []
    for path in sorted(paths):
        evidence.file_inventory[str(path)] = file_hash(path)
        try:
            tree = ET.parse(path).getroot()
        except (ET.ParseError, OSError) as error:
            rows.append(("FAILED", path, str(error), None))
            continue
        cases = list(tree.iter("testcase"))
        if not cases:
            rows.append(("PENDING", path, "无 testcase；不推断测试已通过", None))
            continue
        failed = sum(case.find("failure") is not None for case in cases)
        errors = sum(case.find("error") is not None for case in cases)
        skipped = sum(case.find("skipped") is not None for case in cases)
        status = (
            "FAILED"
            if failed or errors
            else ("PASSED" if len(cases) > skipped else "PENDING")
        )
        real = [
            case
            for case in cases
            if "real_h512_s_checkpoint_oracle" in case.get("name", "")
        ]
        real_status = "未收集"
        if real:
            real_status = (
                "跳过"
                if any(case.find("skipped") is not None for case in real)
                else (
                    "失败"
                    if any(
                        case.find("failure") is not None
                        or case.find("error") is not None
                        for case in real
                    )
                    else "通过"
                )
            )
        rows.append(
            (
                status,
                path,
                f"{len(cases) - failed - errors - skipped} passed / {failed} failed / {errors} error / {skipped} skipped；真实 h512 S：{real_status}",
                len(cases),
            )
        )
    return rows


def worker_records(evidence, directory, summary):
    paths = set(directory.glob("original-*.json")) | set(
        directory.glob("candidate-*.json")
    )
    for value in (summary or {}).get("records", []):
        path = Path(value)
        if not path.is_absolute():
            path = directory / path
        if not path.is_file():
            path = directory / path.name
        if path.is_file():
            paths.add(path)
    result = []
    for path in sorted(paths):
        data, error = evidence.read(path)
        if data and data.get("command") == "benchmark_worker":
            result.append((path, data))
        elif error:
            result.append((path, {"passed": False, "error": error}))
    return result


def performance_status(summary, error, workers, waters):
    if error:
        return "FAILED", error
    if any(data.get("error") or data.get("passed") is False for _, data in workers):
        return "FAILED", "性能 worker 有失败记录"
    if not summary:
        return "PENDING", f"未找到汇总；已找到 {len(workers)} 个 worker 记录"
    if summary.get("passed") is False:
        return "FAILED", "汇总明确标记失败"
    executors = summary.get("executors", {})
    if any(
        executors.get(name, {}).get("repeats") != 3
        for name in ("original", "candidate")
    ):
        return "PENDING", "尚无原模型/候选各 3 次重复的汇总"
    if summary.get("structure", {}).get("waters") != waters:
        return "FAILED", "实际 water 数与目标不符"
    if len(workers) < 6:
        return "PENDING", "完整 6 个独立 worker 证据尚未齐全"
    repeats = {
        name: [
            data.get("repeat") for _, data in workers if data.get("executor") == name
        ]
        for name in ("original", "candidate")
    }
    if any(
        len(values) != 3 or len(set(values)) != 3 for values in repeats.values()
    ) or set(repeats["original"]) != set(repeats["candidate"]):
        return "PENDING", "原模型/候选各 3 个独立 worker 的重复身份尚未齐全"
    for _, data in workers:
        if (
            data.get("precision") != "float64"
            or data.get("elapsed_seconds", 0) < 10
            or data.get("block_seconds_requested", 0) < 10
        ):
            return "PENDING", "worker 尚未满足 FP64、每块至少 10 秒的正式协议"
    if summary.get("passed") is not True:
        return "PENDING", "无明确通过汇总"
    return "PASSED", "已记录原模型/候选各 3 次独立完整 MD 计时"


def identities(evidence):
    keys = {
        "checkpoint_sha256",
        "source_commit",
        "commit",
        "git_sha",
        "source_content_sha256",
        "native_source_content_sha256",
        "tree_source_content_sha256",
        "tree_native_source_content_sha256",
        "extension_sha256",
        "extension_path",
        "python_wrapper_sha256",
        "runtime_manifest_sha256",
        "package_fingerprint",
        "kokkos_execution_space",
        "execution_space",
        "device",
        "device_name",
        "device_ordinal",
        "device_requested",
        "feature_device",
        "selector",
        "tensor_inventory_sha256",
        "positions_sha256",
        "velocities_sha256",
    }
    rows = []

    def visit(value, prefix=""):
        if isinstance(value, dict):
            for name, item in value.items():
                path = prefix + ("." if prefix else "") + name
                if name in keys and item is not None:
                    yield path, item
                elif name in ("wheel_sha256", "shared_openblas_sha256") and isinstance(
                    item, dict
                ):
                    for key, digest in item.items():
                        yield path + "." + key, digest
                if name not in (
                    "named_tensors",
                    "named_modules",
                    "source_files",
                    "package_source_trees",
                    "density_history",
                    "potential_history",
                    "fukui_history",
                    "samples",
                    "cases",
                ):
                    yield from visit(item, path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from visit(item, prefix + f"[{index}]")

    seen = set()
    for source, data in evidence.records:
        for name, value in visit(data):
            identity = (str(source), name, str(value))
            if identity not in seen:
                rows.append((source, name, value))
                seen.add(identity)
    return rows


def render(root):
    evidence = Evidence(root)
    (
        checks,
        build_rows,
        test_rows,
        prep_rows,
        physics_rows,
        region_rows,
        drift_rows,
        perf_rows,
        profile_rows,
        memory_rows,
        state_rows,
    ) = ([] for _ in range(11))
    contract_path, contract, error = evidence.first(
        [root / "manifests/model-contract.json"]
    )
    contract_status = "FAILED" if error else explicit_status(contract)
    checks.append(
        (
            contract_status,
            "checkpoint 合约清点",
            error or (contract or {}).get("status", "未找到记录"),
        )
    )
    for profile in PROFILES:
        directory = root / "manifests/build" / profile
        built_path, built, built_error = evidence.first([directory / "built.json"])
        build_status = (
            "FAILED"
            if built_error
            else (
                "PASSED"
                if built
                and built.get("source_commit")
                and built.get("profile") == profile
                and not built.get("error")
                else explicit_status(built)
            )
        )
        version_path, version, version_error = evidence.first(
            [directory / "version.json", directory / "version.txt"], version=True
        )
        version_status = (
            "FAILED"
            if version_error
            or (version and version.get("probe_status") not in (None, "ok"))
            else (
                "PASSED"
                if version and version.get("probe_status") == "ok"
                else explicit_status(version)
            )
        )
        checks.extend(
            (
                (
                    build_status,
                    profile + " 构建完成标记",
                    built_error
                    or (built or {}).get(
                        "error", "built.json 存在" if built else "未找到记录"
                    ),
                ),
                (
                    version_status,
                    profile + " native probe",
                    version_error
                    or (version or {}).get("probe_status", "未找到有效 probe"),
                ),
            )
        )
        probe = (version or {}).get("probe", {})
        compiled = probe.get("build_info", {})
        descriptor = probe.get("backend", {})
        build_rows.append(
            (
                profile,
                LABELS[build_status],
                LABELS[version_status],
                (built or {}).get("source_commit", "未提供"),
                descriptor.get("selector", "未提供"),
                compiled.get(
                    "execution_space", compiled.get("kokkos_execution_space", "未提供")
                ),
                link(built_path) if built_path.is_file() else "—",
                link(version_path) if version_path.is_file() else "—",
            )
        )
        tests = junit(evidence, profile)
        if not tests:
            checks.append(
                (
                    "PENDING",
                    profile + " JUnit 测试",
                    "未找到 JUnit，构建标记不替代独立测试计数",
                )
            )
            test_rows.append((profile, "待完成", "未找到 JUnit", "—"))
        for status, path, note, _ in tests:
            checks.append((status, profile + " JUnit " + path.name, note))
            test_rows.append((profile, LABELS[status], note, link(path)))
    for waters in WATERS:
        path, data, error = evidence.first(
            [root / f"benchmark/prepared-reference/water{waters:04d}.prepare.json"]
        )
        status, note = preparation_status(data, error, waters)
        checks.append((status, f"{waters} 水共同准备态", note))
        optimization, nvt = (
            (data or {}).get("optimization", {}),
            (data or {}).get("nvt", {}),
        )
        prep_rows.append(
            (
                waters,
                LABELS[status],
                fmt(optimization.get("steps")),
                fmt(optimization.get("maximum_force_eV_A")),
                fmt(nvt.get("steps")),
                fmt(nvt.get("final_temperature_K")),
                note,
                link(path) if path.is_file() else "—",
            )
        )
        state = nvt.get("prepared_output_state", {})
        if state:
            state_rows.append(
                (
                    "prepared",
                    waters,
                    "NVT 写出并回读",
                    state.get("positions_sha256", "未提供"),
                    state.get("velocities_sha256", "未提供"),
                    "—",
                )
            )
        for profile in PROFILES:
            for phase, filename in (
                ("validation", "validation.json"),
                ("md100", "md.json"),
                ("md1000", "md.json"),
            ):
                directory = (
                    root
                    / "benchmark/qualification"
                    / profile
                    / f"water{waters:04d}"
                    / phase
                )
                path, data, error = evidence.first([directory / filename])
                status, note = physics_status(data, error, phase, waters)
                failure_path, failure = failed_marker(
                    evidence, directory, path, status == "PASSED"
                )
                if failure:
                    status, note = "FAILED", failure
                checks.append((status, f"{profile}/{waters}/{phase}", note))
                rows = measurements(data)
                charge, spin = population_errors(rows)
                physics_rows.append(
                    (
                        profile,
                        waters,
                        phase,
                        LABELS[status],
                        fmt(maximum(rows, "absolute_total_energy_error_eV")),
                        fmt(maximum(rows, "maximum_force_component_error_eV_A")),
                        fmt(charge),
                        fmt(spin),
                        note,
                        link(path)
                        if path.is_file()
                        else (link(failure_path) if failure_path else "—"),
                    )
                )
                for region in ("Li", "hydration_shell_water", "remaining_water"):
                    values = [
                        row.get("force_error_by_region", {}).get(region, {})
                        for row in rows
                    ]
                    if any(values):
                        region_rows.append(
                            (
                                profile,
                                waters,
                                phase,
                                region,
                                fmt(
                                    maximum(
                                        values, "maximum_force_component_error_eV_A"
                                    )
                                ),
                                fmt(maximum(values, "maximum_force_vector_error_eV_A")),
                                ", ".join(
                                    sorted(
                                        {
                                            str(value.get("atom_count"))
                                            for value in values
                                            if "atom_count" in value
                                        }
                                    )
                                ),
                            )
                        )
                drift = (data or {}).get("drift")
                if drift:
                    drift_rows.append(
                        (
                            profile,
                            waters,
                            phase,
                            fmt(drift.get("final_minus_initial_eV_per_atom")),
                            fmt(drift.get("maximum_absolute_eV_per_atom")),
                            fmt(drift.get("linear_eV_per_atom_ps")),
                        )
                    )
                initial = (data or {}).get("initial_state", {})
                if initial:
                    state_rows.append(
                        (
                            profile,
                            waters,
                            phase,
                            initial.get("positions_sha256", "未提供"),
                            initial.get("velocities_sha256", "未提供"),
                            (data or {})
                            .get("velocity_initialization", {})
                            .get("policy", "未提供"),
                        )
                    )
            directory = root / "benchmark/performance" / profile / f"water{waters:04d}"
            path, summary, error = evidence.first(
                [directory / "summary.json", directory / "benchmark-report.json"]
            )
            workers = worker_records(evidence, directory, summary)
            status, note = performance_status(summary, error, workers, waters)
            failure_path, failure = failed_marker(
                evidence, directory, path, status == "PASSED"
            )
            if failure:
                status, note = "FAILED", failure
            checks.append((status, f"{profile}/{waters}/performance", note))
            executors = (summary or {}).get("executors", {})
            perf_rows.append(
                (
                    profile,
                    waters,
                    LABELS[status],
                    fmt(executors.get("original", {}).get("repeats")),
                    fmt(executors.get("candidate", {}).get("repeats")),
                    fmt(executors.get("original", {}).get("median_us_per_atom_step")),
                    fmt(executors.get("candidate", {}).get("median_us_per_atom_step")),
                    fmt((summary or {}).get("speedup_original_over_candidate")),
                    note,
                    link(path)
                    if path.is_file()
                    else (link(failure_path) if failure_path else "—"),
                )
            )
            stages = {}
            for worker_path, worker in workers:
                for name, value in (
                    worker.get("separate_single_call_profile_timings_ms") or {}
                ).items():
                    if number(value):
                        stages.setdefault(name, []).append(value)
                memory = worker.get("memory", {})
                if memory:
                    memory_rows.append(
                        (
                            profile,
                            waters,
                            worker.get("executor", "未提供"),
                            worker.get("repeat", "未提供"),
                            fmt(memory.get("nvidia_process_peak_MiB")),
                            fmt(memory.get("torch_allocator_only_peak_bytes")),
                            fmt(memory.get("process_peak_rss_bytes")),
                            fmt(memory.get("global_all_gpus_peak_minus_baseline_MiB")),
                            str(
                                memory.get("competing_compute_pids_observed", "未提供")
                            ),
                            link(worker_path),
                        )
                    )
            for stage, values in stages.items():
                profile_rows.append(
                    (
                        profile,
                        waters,
                        stage,
                        len(values),
                        fmt(statistics.median(values)),
                        "单独同步诊断调用；不计入 throughput",
                    )
                )
    counts = {status: sum(row[0] == status for row in checks) for status in LABELS}
    lines = [
        "# MACE-POLAR 适配资格报告",
        "",
        f"生成时间（Asia/Shanghai）：{datetime.now(timezone(timedelta(hours=8))).isoformat()}。部署根目录：`{root}`。",
        "",
        f"记录状态：**通过 {counts['PASSED']}，待完成 {counts['PENDING']}，失败 {counts['FAILED']}**。这些数量指独立证据项目，不是 pytest 测试数量。",
        "",
        "本报告只读取现有文件，没有运行模型、测试或 MD。待完成表示未找到完成证据或正式协议不完整，不表示任务已经调度；失败记录不会被缺失的新记录覆盖。只有时间更新且正式协议完整通过的记录可以替代历史失败，历史证据仍单列。合约清点只证明序列化/权重归属，不证明数值正确性。",
        "",
        "## 通过、待完成与失败",
        "",
    ]
    for status in ("FAILED", "PENDING", "PASSED"):
        lines.extend(
            [
                f"### {LABELS[status]}",
                "",
                table(
                    ["项目", "实际证据说明"],
                    [
                        (name, note)
                        for current, name, note in checks
                        if current == status
                    ],
                ),
                "",
            ]
        )
    sections = [
        (
            "构建与设备",
            [
                "profile",
                "构建",
                "probe",
                "source commit",
                "backend",
                "execution space",
                "build 证据",
                "probe 证据",
            ],
            build_rows,
        ),
        (
            "实际测试",
            ["profile", "状态", "实际计数与真实模型 oracle", "证据"],
            test_rows,
        ),
        (
            "共同准备态",
            [
                "waters",
                "状态",
                "BFGS steps",
                "最终 fmax eV/Å",
                "NVT steps",
                "最终 K",
                "说明",
                "证据",
            ],
            prep_rows,
        ),
        (
            "完整物理对照",
            [
                "profile",
                "waters",
                "阶段",
                "状态",
                "max |ΔE| eV/cell",
                "max |ΔFcomp| eV/Å",
                "max |ΔQ| e",
                "max |Δspin|",
                "说明",
                "证据",
            ],
            physics_rows,
        ),
        (
            "分区域力误差",
            [
                "profile",
                "waters",
                "阶段",
                "区域",
                "max |ΔFcomp| eV/Å",
                "max |ΔFvector| eV/Å",
                "实际采样 atom counts",
            ],
            region_rows,
        ),
        (
            "NVE 漂移",
            [
                "profile",
                "waters",
                "阶段",
                "终末−初始 eV/atom",
                "max |drift| eV/atom",
                "linear eV/atom/ps",
            ],
            drift_rows,
        ),
        (
            "完整 ASE MD 性能",
            [
                "profile",
                "waters",
                "状态",
                "original repeats",
                "candidate repeats",
                "original µs/atom/step",
                "candidate µs/atom/step",
                "original/candidate speedup",
                "说明",
                "证据",
            ],
            perf_rows,
        ),
        (
            "独立诊断调用的阶段耗时",
            ["profile", "waters", "stage", "实际调用数", "median ms", "范围"],
            profile_rows,
        ),
        (
            "内存证据",
            [
                "profile",
                "waters",
                "executor",
                "repeat",
                "NVIDIA own PID MiB",
                "Torch-only peak bytes",
                "process peak RSS bytes",
                "all-GPU peak−baseline MiB",
                "其他 GPU PIDs",
                "证据",
            ],
            memory_rows,
        ),
        (
            "实际准备态与初态哈希",
            [
                "profile",
                "waters",
                "阶段",
                "positions SHA-256",
                "velocities SHA-256",
                "速度策略",
            ],
            state_rows,
        ),
    ]
    for title, headers, rows in sections:
        lines.extend(["## " + title, "", table(headers, rows), ""])
    if evidence.superseded_failures:
        lines.extend(
            [
                "## 已有更新完整通过证据的历史失败",
                "",
                table(
                    ["历史失败证据", "当时错误", "更新的完整通过证据"],
                    [
                        (link(path), note, link(newer))
                        for path, note, newer in evidence.superseded_failures
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## 判定边界",
            "",
            "正式物理阈值：总胞能量误差 ≤ 1e-4 eV，最大力分量误差 ≤ 1e-5 eV/Å，初始及两轮更新后的 |Q−1| ≤ 1e-9 e、|spin population| ≤ 1e-9。水合壳按 O–Li 最小镜像距离 ≤ 3.2 Å 选取整分子。NVE 目标为 0.25 fs、100/1000 步；漂移仅报告实际数值，没有自创漂移阈值。正式性能为 FP64、原模型/候选各 3 次独立进程、每块至少 10 秒。",
            "",
            "跳过的测试没有被当作已验证能力。原模型/候选速度只取已有汇总，区域最大值与阶段中位数由实际记录计算。NVIDIA own PID 数值包含 Torch/native 分配；Torch-only 不代表进程显存。all-GPU 数据含其他进程，不能当作模型独占显存。未提供的计数或数值保持缺失，没有填零。",
            "",
            "## 运行、checkpoint 与二进制身份",
            "",
            "以下逐证据保留实际记录的 Git、源文件、checkpoint、扩展二进制、wheel、设备和包指纹。不同提交或哈希分别列出，不假定它们来自同一构建；wheel 哈希也不等同于已加载扩展哈希。",
            "",
            table(
                ["证据文件", "字段", "实际值"],
                [(link(path), key, value) for path, key, value in identities(evidence)],
            ),
            "",
            "## 本报告读取的证据文件",
            "",
            table(
                ["文件", "SHA-256"],
                [
                    (link(Path(path)), digest)
                    for path, digest in sorted(evidence.file_inventory.items())
                ],
            ),
            "",
        ]
    )
    return "\n".join(lines), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Deployment root containing manifests/ and benchmark/",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Markdown path outside Git"
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if any((parent / ".git").exists() for parent in output.parents):
        parser.error("--output must be outside every Git checkout")
    markdown, counts = render(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(markdown)
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": file_hash(output),
                "evidence_status_counts": counts,
            }
        )
    )


if __name__ == "__main__":
    main()
