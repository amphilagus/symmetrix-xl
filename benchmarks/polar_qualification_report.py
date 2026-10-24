#!/usr/bin/env python3
"""Generate an evidence-only Markdown POLAR report without importing model code.

Read deployment manifests and experiment records; never run a model, a test,
or a benchmark. Missing and partial evidence stays pending. --output must be
outside every Git checkout, including the deployed source checkout.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


PROFILES = ("cpu-omp", "sm80", "sm90")
DIAGNOSTIC_PROFILES = ("cpu",)
UPSTREAM_COMMIT = "aee9f980eb48bbee125a500a28fa6517466a1b66"
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
        self.historical_tests = []

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


def failed_marker(
    evidence,
    directory,
    completed_path=None,
    complete=False,
    names=("failure.json", "benchmark-failure.json"),
):
    for name in names:
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
        return "PENDING", "尚无优化收敛证据；仅初始构型不计完整准备"
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


def junit_details(evidence, path):
    evidence.file_inventory[str(path)] = file_hash(path)
    try:
        tree = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        return None, str(error)
    cases = list(tree.iter("testcase"))
    counts = {
        "passed": sum(
            all(case.find(tag) is None for tag in ("failure", "error", "skipped"))
            for case in cases
        ),
        "failed": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
        "tests": len(cases),
    }
    status = (
        "FAILED"
        if counts["failed"] or counts["errors"]
        else ("PASSED" if counts["passed"] else "PENDING")
    )
    return {"status": status, "counts": counts, "cases": cases}, None


def junit_note(details):
    counts = details["counts"]
    return (
        f"{counts['passed']} passed / {counts['failed']} failed / "
        f"{counts['errors']} error / {counts['skipped']} skipped"
    )


def junit(evidence, profile):
    roots = (
        evidence.root / "tests",
        evidence.root / "manifests/tests",
        evidence.root / "manifests/build" / profile,
    )
    paths = set()
    for base in roots:
        if base.is_dir():
            if base.name == profile:
                paths.update(base.rglob("*.xml"))
            else:
                for path in base.glob("*.xml"):
                    if profile in path.name and not (
                        profile == "cpu" and "cpu-omp" in path.name
                    ):
                        paths.add(path)
                tagged = base / profile
                if tagged.is_dir():
                    paths.update(tagged.rglob("*.xml"))
    rows = []
    for path in sorted(paths):
        # Whole-suite results have dedicated current/historical sections.
        if path in {
            evidence.root / "tests/cpu-full.xml",
            evidence.root / "tests/cpu-omp-full.xml",
        }:
            continue
        details, error = junit_details(evidence, path)
        if error:
            rows.append(("FAILED", path, error, None))
            continue
        cases = details["cases"]
        if not cases:
            rows.append(("PENDING", path, "无 testcase；不推断测试已通过", None))
            continue
        status = details["status"]
        if "attempts" in path.parts:
            evidence.historical_tests.append(
                (profile, path, status, junit_note(details))
            )
            continue
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
        prefix_note = ""
        for case in cases:
            for prop in case.findall("./properties/property"):
                if prop.get("name") != "real_h512_s_native_feature_vjp_oracle":
                    continue
                try:
                    oracle = json.loads(prop.get("value", ""))
                except json.JSONDecodeError:
                    continue
                if isinstance(oracle, dict):
                    evidence.records.append((path, oracle))
                    prefix_note = "；独立前缀/VJP：" + ", ".join(
                        f"{row.get('label')} maxabs={fmt(row.get('max_abs'))}, 容差比={fmt(row.get('max_normalized_ratio'))}"
                        for row in oracle.get("comparisons", [])
                    )
        rows.append(
            (
                status,
                path,
                junit_note(details) + f"；真实 h512 S：{real_status}" + prefix_note,
                len(cases),
            )
        )
    return rows


def cpu_runtime(evidence, profile):
    directory = evidence.root / "manifests/build" / profile
    path, data, error = evidence.first(
        [
            directory / "doctor-torch-polar.json",
            evidence.root / f"tests/{profile}-doctor-torch-polar.json",
        ]
    )
    if error:
        return "FAILED", path, None, error
    if data is None:
        return "PENDING", path, None, "未找到实际 Torch/POLAR 栈的严格 doctor"
    report = data.get("diagnostics", data.get("cli_doctor", data))
    runtime = report.get("runtime", {})
    context = report.get("qualification_context", {})
    libraries = runtime.get("loaded_openmp_libraries", [])
    qualified = (
        report.get("status") == "ok"
        and report.get("ok") is True
        and report.get("strict") is True
        and runtime.get("kokkos_initialized") is True
        and runtime.get("kokkos_concurrency") == 1
        and runtime.get("worker_sentinel", {}).get("actual_team_size") == 1
        and len(libraries) == 1
        and context.get("executor", {}).get("calculator") == "SymmetrixPolar"
        and number(context.get("energy_eV"))
        and number(context.get("maximum_force_eV_A"))
    )
    issues = [
        row.get("message", row.get("name", ""))
        for row in report.get("checks", [])
        if row.get("status") in ("error", "warning")
    ]
    note = (
        "实际 Torch/POLAR 栈严格 doctor 为 ok，且仅加载一个 OpenMP runtime"
        if qualified
        else f"strict doctor={report.get('status', '未提供')}；"
        + ("；".join(issues) or "缺少已初始化、单运行库的完整通过证据")
    )
    return "PASSED" if qualified else "FAILED", path, report, note


def build_identity_status(profile, built, version, version_path):
    if not built or not version:
        return "PENDING", "构建或 probe 记录缺失"
    if version.get("probe_status") != "ok":
        return "FAILED", "native probe 未通过"
    if built.get("profile") != profile or not built.get("source_commit"):
        return "FAILED", "构建 profile 或源提交记录不完整"
    probe = version.get("probe", {})
    expected_selector = "cpu" if profile.startswith("cpu") else "cuda12-" + profile
    if probe.get("backend", {}).get("selector") != expected_selector:
        return "FAILED", "native selector 与配置不匹配"
    for identity in (probe.get("backend", {}), probe.get("build_info", {})):
        if (
            identity.get("source_commit") != built["source_commit"]
            or identity.get("source_dirty") is not False
        ):
            return "FAILED", "构建与 probe 的干净源提交不匹配"
        for key in ("source_content_sha256", "native_source_content_sha256"):
            digest = built.get("source_fingerprints", {}).get(key)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or identity.get(key) != digest
            ):
                return "FAILED", "构建与 probe 的源指纹不匹配：" + key
    if built.get("version_probe_sha256") != file_hash(version_path):
        return "FAILED", "built.json 记录的 probe 文件哈希不匹配"
    if len(built.get("extension_sha256", "")) != 64:
        return "FAILED", "缺少扩展二进制哈希"
    device = probe.get("device_environment", {})
    if profile.startswith("sm") and (
        device.get("available") is not True
        or device.get("device_memory") is not True
        or device.get("execution_space") != "Cuda"
        or device.get("compute_capability_code") != int(profile[2:])
    ):
        return "FAILED", "实际 CUDA 设备与目标架构不匹配"
    return "PASSED", "干净源提交、源指纹、probe 文件哈希和目标设备记录一致"


def full_cpu_regression(evidence):
    current_path = evidence.root / "tests/cpu-omp-full.xml"
    current, current_error = (
        junit_details(evidence, current_path)
        if current_path.is_file()
        else (None, None)
    )
    current_status = current["status"] if current else "PENDING"
    current_note = (
        junit_note(current)
        if current
        else f"新 XML 尚不可解析：{current_error}"
        if current_error
        else "未找到 cpu-omp-full.xml；旧 cpu 结果不替代当前配置"
    )
    overview = [
        (
            "cpu-omp（当前独立运行环境）",
            LABELS[current_status] + "；" + current_note,
            link(current_path) if current_path.is_file() else "未提供",
        )
    ]
    checks = [(current_status, "cpu-omp 完整回归原始结果", current_note)]
    path = evidence.root / "tests/cpu-full.xml"
    if not path.is_file():
        return overview, [], checks
    details, error = junit_details(evidence, path)
    if error:
        checks.append(("FAILED", "历史 cpu 完整回归 XML", error))
        return overview, [], checks
    overview.append(
        (
            "cpu（历史锁定运行环境；不替代 cpu-omp）",
            LABELS[details["status"]] + "；" + junit_note(details),
            link(path),
        )
    )
    baseline_dir = evidence.root / "tests/upstream-aee9f980"
    baseline_path, baseline, baseline_error = evidence.first(
        [baseline_dir / "baseline.json"]
    )
    baseline_xml = baseline_dir / "four-mh1-tests.xml"
    original, original_error = (
        junit_details(evidence, baseline_xml)
        if baseline_xml.is_file()
        else (None, None)
    )
    legacy_path, legacy, legacy_error = evidence.first(
        [evidence.root / "tests/full-cpu-9890071/legacy-environment.json"]
    )
    candidate_failures = {
        case.get("name"): case.find("failure").get("message", "")
        for case in details["cases"]
        if case.find("failure") is not None
    }
    original_failures = {
        case.get("name"): case.find("failure").get("message", "")
        for case in (original or {}).get("cases", [])
        if case.find("failure") is not None
    }
    reproduced = set()
    if (
        not baseline_error
        and not original_error
        and baseline
        and baseline.get("source_commit") == UPSTREAM_COMMIT
        and baseline.get("source_pristine") is True
        and len(original_failures) == 4
        and all(
            candidate_failures.get(name) == message
            for name, message in original_failures.items()
        )
    ):
        reproduced = set(original_failures)
    groups = {}
    for name, message in candidate_failures.items():
        reason = "尚未归因"
        if name in reproduced:
            reason = "干净上游独立构建复现同一 MH1 失败及打印误差指标"
        elif (
            name.startswith("test_initial_state_")
            or name.startswith("test_state_round_trip_")
            or name.startswith("test_worker_")
        ):
            if (
                "thermalize_momenta" in message
                and (legacy or {}).get("has_thermalize_momenta") is False
            ):
                reason = "锁定 ASE 缺少 thermalize_momenta API"
        elif name.startswith("test_ase_npt_reuses_") and "DID NOT WARN" in message:
            reason = "锁定 ASE 的 NPT 构造未发出测试期待的 FutureWarning"
        elif (
            name.startswith("test_kokkos_host_workers_match_")
            and "compact_r1_model" in message
        ):
            if (legacy or {}).get("wrapper_strips_PYTHONPATH") is True:
                reason = "隔离包装器移除 PYTHONPATH，子进程无法导入测试辅助模块"
        elif name.startswith("test_openblas_openmp_runtime_must_match_"):
            probes = (legacy or {}).get("standalone_blas_probes", [])
            if len(probes) == 2 and all(
                "GLIBCXX_3.4.29" in row.get("stderr", "") for row in probes
            ):
                reason = "独立 C++ 探针使用系统 libstdc++，缺少 GLIBCXX_3.4.29"
        elif name.startswith("test_direct_renameat2_syscall_"):
            if (legacy or {}).get("direct_renameat2_syscall", {}).get("errno") in (
                22,
                38,
                95,
            ):
                reason = "共享测试文件系统/内核不支持该直接 renameat2 发布方式"
        groups.setdefault(reason, []).append(name)
    rows = [
        (
            reason,
            len(names),
            ", ".join(names),
            link(baseline_xml)
            if reason.startswith("干净上游")
            else link(legacy_path)
            if reason != "尚未归因"
            else link(path),
        )
        for reason, names in groups.items()
    ]
    explained = len(candidate_failures) - len(groups.get("尚未归因", []))
    checks.append(
        (
            "PASSED"
            if reproduced and explained == len(candidate_failures) and not legacy_error
            else "PENDING",
            "历史 cpu 完整回归失败归因",
            f"历史 cpu 原始失败 {len(candidate_failures)} 项；同上游复现 {len(reproduced)} 项，环境/工具链解释 {explained - len(reproduced)} 项；不代表当前 cpu-omp 的测试计数或归因",
        )
    )
    if baseline:
        overview.append(
            (
                "历史 cpu 对照：干净上游 aee9f980 四项复现",
                junit_note(original) if original else "缺少 JUnit",
                link(baseline_path),
            )
        )
    return overview, rows, checks


def current_cpu_regression_attribution(evidence):
    """Bind current attribution to exact JUnit issues; never promote failures."""
    path = evidence.root / "tests/cpu-omp-full-analysis.json"
    analysis, error = evidence.read(path)
    label = "当前 cpu-omp 完整回归失败归因"

    def unavailable(status, note):
        return [(LABELS[status], "—", note, "未采用分类", link(path))], [
            (status, label, note)
        ]

    if error:
        return unavailable("FAILED", error)
    if analysis is None:
        return unavailable("PENDING", "未找到当前分析；不套用历史 cpu 的归因")
    xml_path = evidence.root / "tests/cpu-omp-full.xml"
    if not xml_path.is_file():
        return unavailable("PENDING", "未找到当前 cpu-omp-full.xml，无法绑定分析")
    details, error = junit_details(evidence, xml_path)
    if error:
        return unavailable("FAILED", "当前 JUnit 无法解析：" + error)

    def issues(cases):
        return Counter(
            (
                case.get("classname", "") + "::" + case.get("name", ""),
                child.get("message", ""),
            )
            for case in cases
            for child in case
            if child.tag in ("failure", "error")
        )

    try:
        if analysis.get("schema") != 1:
            raise ValueError("不支持的分析 schema")
        declared_junit = Path(analysis["junit"])
        if not declared_junit.is_absolute():
            raise ValueError("分析 junit 必须声明原始部署的绝对路径")
        declared_root = declared_junit.parent.parent
        if analysis.get("junit_sha256") != file_hash(xml_path):
            raise ValueError("分析 junit_sha256 与当前 XML 不一致；拒绝使用旧分类")
        if analysis.get("counts") != details["counts"]:
            raise ValueError("分析 counts 与当前原始 JUnit 计数不一致")
        categories = analysis.get("categories")
        if not isinstance(categories, dict) or any(
            not isinstance(rows, list) for rows in categories.values()
        ):
            raise ValueError("分析 categories 结构无效")
        classified = [row for rows in categories.values() for row in rows]
        if any(
            not isinstance(row, dict)
            or not isinstance(row.get("testcase"), str)
            or not isinstance(row.get("message"), str)
            for row in classified
        ):
            raise ValueError("分析的 testcase/message 必须是字符串")
        expected_count = details["counts"]["failed"] + details["counts"]["errors"]
        if len(classified) != expected_count:
            raise ValueError("分类总数不等于原始 failed + errors")
        if Counter((row["testcase"], row["message"]) for row in classified) != issues(
            details["cases"]
        ):
            raise ValueError("分类的 testcase/message 与当前原始失败逐项不一致")

        def checked_artifact(value, expected_hash):
            artifact = Path(value)
            if artifact.is_absolute():
                try:
                    artifact = artifact.relative_to(declared_root)
                except ValueError as failure:
                    raise ValueError(
                        f"分析引用的证据不在声明的部署根目录下：{value}"
                    ) from failure
            artifact = (evidence.root / artifact).resolve()
            if not artifact.is_relative_to(evidence.root.resolve()):
                raise ValueError(f"分析引用的证据映射越界：{value}")
            if not artifact.is_file():
                raise ValueError(f"分析引用的证据缺失：{artifact}")
            digest = file_hash(artifact)
            evidence.file_inventory[str(artifact)] = digest
            if digest != expected_hash:
                raise ValueError(f"分析引用的证据 SHA-256 不一致：{artifact}")
            return artifact

        environment = checked_artifact(
            analysis["compiler_environment"], analysis["compiler_environment_sha256"]
        )
        _, environment_error = evidence.read(environment)
        if environment_error:
            raise ValueError("编译环境 JSON 无法解析：" + environment_error)
        probe = analysis.get("standalone_probe", {})
        if not isinstance(probe, dict) or not isinstance(
            probe.get("elf_records", []), list
        ):
            raise ValueError("standalone_probe/elf_records 结构无效")
        elf_paths = [
            checked_artifact(row["output"], row["sha256"])
            for row in probe.get("elf_records", [])
        ]
        if probe and not elf_paths:
            raise ValueError("standalone probe 分析缺少绑定哈希的 ELF 记录")
    except (KeyError, TypeError, ValueError, OSError) as failure:
        return unavailable("FAILED", "分析未通过证据核对：" + str(failure))

    def message_excerpt(message):
        lines = [line.strip() for line in message.splitlines() if line.strip()]
        selected = [
            line
            for line in lines
            if "ImportError:" in line
            or "ModuleNotFoundError:" in line
            or line.startswith("Max absolute difference:")
        ]
        return "；".join(selected or lines[:1])

    rows = [
        (
            "原始状态保持" + LABELS[details["status"]],
            "—",
            junit_note(details) + "；分析只解释失败，不将失败转记为通过",
            "源提交 " + str(analysis.get("source_commit", "未提供")),
            link(xml_path) + "；" + link(path) + "；" + link(environment),
        )
    ]
    for category, members in categories.items():
        rows.append(
            (
                category,
                len(members),
                "；".join(
                    member["testcase"] + ": " + message_excerpt(member["message"])
                    for member in members
                ),
                "分析记录的旧 cpu 对照：同名失败 "
                + str(
                    sum(
                        member.get("same_failed_name_in_old_cpu") is True
                        for member in members
                    )
                )
                + f"/{len(members)}；message 逐字相同 "
                + str(
                    sum(
                        member.get("same_failure_message_in_old_cpu") is True
                        for member in members
                    )
                )
                + f"/{len(members)}；归因以本轮失败文本为准",
                link(path) + "；" + link(xml_path),
            )
        )
    if probe:
        rows.extend(
            (
                "standalone probe：" + kind,
                "—",
                probe.get(key, "未提供"),
                "未执行 probe；后续 loader/dlopen 行为不称为实测",
                "；".join(map(link, [path, *elf_paths])),
            )
            for kind, key in (("实证", "observed"), ("推断", "inference"))
        )
    # Raw regression status is already a check in full_cpu_regression. An
    # attribution passing its integrity checks is not a passing test result.
    return rows, []


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
        provenance = data.get("provenance", {})
        early = provenance.get("early_execution_environment", {})
        threads = provenance.get("thread_environment", {})
        if not threads or not early or not provenance.get("cpu_affinity"):
            return "PENDING", "worker 尚缺导入前 affinity 与实际线程环境证据"
        if (
            early.get("applied") is not True
            or len(early.get("allowed_after", [])) != 1
            or len(provenance["cpu_affinity"]) != 1
            or any(
                str(threads.get(name)) != "1"
                for name in (
                    "OMP_NUM_THREADS",
                    "KOKKOS_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "BLIS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            )
        ):
            return (
                "FAILED",
                "正式性能 worker 未证明固定单核和单线程；MD 并行策略不能替代此协议",
            )
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
        "qualification_source_commit",
        "qualification_only",
        "slurm_job_id",
        "slurm_job",
        "source_pristine",
        "expected_runtime_sha256",
        "loaded_runtime_sha256",
        "tensor_inventory_sha256",
        "positions_sha256",
        "velocities_sha256",
        "initial_positions_sha256",
        "input_sha256",
        "script_sha256",
        "source_git_sha",
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


def nsys_statistics(path):
    """Read concatenated Nsight CSV summaries without combining time domains."""
    reports = {"api": [], "memory_time": [], "memory_size": []}
    header, kind = None, None
    for row in csv.reader(path.read_text(errors="replace").splitlines()):
        if "Name" in row and "Total Time (ns)" in row:
            header, kind = row, "api"
        elif "Operation" in row and "Total Time (ns)" in row:
            header, kind = row, "memory_time"
        elif "Operation" in row and "Total (MB)" in row:
            header, kind = row, "memory_size"
        elif header and len(row) == len(header):
            reports[kind].append(dict(zip(header, row, strict=True)))
        else:
            header, kind = None, None
    return reports


def module_diagnostic_status(data, error):
    if error or explicit_status(data) == "FAILED":
        return "FAILED", error or str(data.get("error", "模块诊断明确失败"))
    if not data:
        return "PENDING", "未找到模块诊断"
    calls = data.get("calls", [])
    if (
        data.get("precision") != "float64"
        or "synchronized" not in data.get("scope", "")
        or not calls
        or len(calls) != data.get("diagnostic_calls")
    ):
        return "PENDING", "FP64、同步范围或完整调用数证据尚未齐全"
    parities = [row.get("instrumentation_parity", {}) for row in calls]
    parities.append(data.get("restored_instrumentation_parity", {}))
    for parity in parities:
        for key, limit in (("energy_max_abs_eV", 1e-8), ("force_max_abs_eV_A", 1e-10)):
            value = parity.get(key)
            if not number(value):
                return "PENDING", "尚缺逐次插桩及恢复后的 E/F 一致性记录"
            if value < 0 or value > limit:
                return "FAILED", f"插桩前后 {key}={value} 超出 {limit}"
    summary = data.get("summary", {})
    if not summary.get("forward_regions") or any(
        not number(summary.get(key, {}).get("median"))
        for key in ("long_range_electrostatic_forward_ms", "whole_suffix_autograd_ms")
    ):
        return "PENDING", "尚缺模块 forward、长程 forward 或整个 autograd 耗时"
    if explicit_status(data) != "PASSED":
        return "PENDING", "尚无模块诊断完整通过标记"
    return (
        "PASSED",
        f"{len(calls)} 次 FP64 同步诊断，逐次插桩及恢复后的 E/F 均在阈值内一致",
    )


def module_diagnostics(evidence):
    summaries, modules, stages, memory, synchronizations = ([] for _ in range(5))
    outcomes = {}
    directory = evidence.root / "benchmark/module-profile"
    for path in sorted(directory.glob("*/*-modules.json")):
        if path.name.endswith("-traced-modules.json"):
            continue
        data, error = evidence.read(path)
        profile = path.stem.removesuffix("-modules")
        label = f"{profile}/{path.parent.name}"
        status, note = module_diagnostic_status(data, error)
        outcomes.setdefault(profile, []).append(
            (path.stat().st_mtime_ns, path, status, note)
        )
        if not data:
            summaries.append((label, f"失败：{error}", *(["未提供"] * 9), link(path)))
            continue
        calls = data.get("calls", [])
        summary = data.get("summary", {})

        def interval(record):
            return " / ".join(
                fmt(record.get(key)) for key in ("median", "minimum", "maximum")
            )

        parities = [row.get("instrumentation_parity", {}) for row in calls]
        parities.append(data.get("restored_instrumentation_parity", {}))
        summaries.append(
            (
                label,
                LABELS[status],
                data.get("atoms", "未提供"),
                data.get("diagnostic_calls", "未提供"),
                data.get("precision", "未提供"),
                interval(summary.get("long_range_electrostatic_forward_ms", {})),
                interval(summary.get("whole_suffix_autograd_ms", {})),
                fmt(summary.get("stage_timings_ms", {}).get("total", {}).get("median")),
                fmt(maximum(parities, "energy_max_abs_eV")),
                fmt(maximum(parities, "force_max_abs_eV_A")),
                data.get("bridge", "未提供"),
                link(path),
            )
        )
        for name, record in summary.get("forward_regions", {}).items():
            counts = sorted(
                {
                    row.get("forward_regions", {}).get(name, {}).get("calls")
                    for row in calls
                    if number(row.get("forward_regions", {}).get(name, {}).get("calls"))
                }
            )
            modules.append(
                (
                    label,
                    name,
                    str(counts),
                    interval(record.get("inclusive_ms", {})),
                    interval(record.get("exclusive_ms", {})),
                )
            )
        for name, record in summary.get("stage_timings_ms", {}).items():
            stages.append((label, name, interval(record)))
        prefix = path.name.removesuffix("-modules.json")
        traced, _ = evidence.read(path.with_name(f"{prefix}-traced-modules.json"))
        trace_status = (
            LABELS[explicit_status(traced)]
            + "; cudaProfilerRange="
            + str((traced or {}).get("cuda_profiler_range", {}).get("requested"))
        )
        statistics_path = path.with_name(f"{prefix}-cuda-statistics.txt")
        for artifact in (
            statistics_path,
            path.with_name(f"{prefix}-normal-call.nsys-rep"),
            path.with_name(f"{prefix}-normal-call.sqlite"),
            path.parent / "nsys-version.txt",
        ):
            if artifact.is_file():
                evidence.file_inventory[str(artifact)] = file_hash(artifact)
        if not statistics_path.is_file():
            continue
        reports = nsys_statistics(statistics_path)
        sizes = {row["Operation"]: row for row in reports["memory_size"]}
        for row in reports["memory_time"]:
            memory.append(
                (
                    label,
                    row["Operation"],
                    row["Count"],
                    fmt(float(row["Total Time (ns)"]) / 1e6),
                    sizes.get(row["Operation"], {}).get("Total (MB)", "未提供"),
                    trace_status,
                    link(statistics_path),
                )
            )
        for row in reports["api"]:
            if "Synchronize" in row["Name"]:
                synchronizations.append(
                    (
                        label,
                        row["Name"],
                        row["Num Calls"],
                        fmt(float(row["Total Time (ns)"]) / 1e6),
                        fmt(float(row["Avg (ns)"]) / 1e3),
                        link(statistics_path),
                    )
                )
    checks = []
    for profile in PROFILES:
        candidates = outcomes.get(profile, [])
        if candidates:
            _, path, status, note = max(candidates, key=lambda row: row[0])
            note += "；" + str(path)
        else:
            status, note = (
                "PENDING",
                "未找到当前配置的同步模块诊断及前后 E/F 一致性记录",
            )
        checks.append((status, profile + " 同步模块诊断及前后 E/F 一致性", note))
    return summaries, modules, stages, memory, synchronizations, checks


def pipeline_records(evidence):
    rows = []
    paths = sorted((evidence.root / "benchmark/gpu-pair").glob("*/pipeline.json"))
    paths.extend(
        sorted((evidence.root / "manifests/qualification").glob("*/pipeline.json"))
    )
    for path in paths:
        data, error = evidence.read(path)
        records = [(path, data, error)]
        for profile in (data or {}).get("profiles", {}):
            child_path = path.with_name(f"{profile}-pipeline.json")
            child, child_error = evidence.read(child_path)
            records.append((child_path, child, child_error))
        for record_path, record, record_error in records:
            record = record or {}
            stages = record.get("stages", [])
            completed = sum(
                row.get("status") == "passed" and row.get("returncode") == 0
                for row in stages
            )
            rows.append(
                (
                    f"{record.get('profile', 'controller')}/{path.parent.name}",
                    "失败" if record_error else LABELS[explicit_status(record)],
                    f"{completed}/{len(stages)}" if stages else "—",
                    record_error
                    or "; ".join(str(row.get("output", "未提供")) for row in stages)
                    or str(record.get("profiles", "未提供")),
                    link(record_path) if record_path.is_file() else "未提供",
                )
            )
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
    runtime_rows, runtime_statuses = [], {}
    profiles = (*PROFILES, *DIAGNOSTIC_PROFILES)
    for profile in profiles:
        formal = profile in PROFILES
        role = "正式" if formal else "历史诊断；未替代 cpu-omp"
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
        build_failure_path, build_failure = failed_marker(
            evidence,
            directory,
            built_path,
            build_status == "PASSED",
            names=("failure.json", "build-failure.json"),
        )
        if build_failure:
            build_status, built_error = "FAILED", build_failure
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
        identity_status, identity_note = build_identity_status(
            profile, built, version, version_path
        )
        if built_error or version_error:
            identity_status, identity_note = "FAILED", built_error or version_error
        if formal:
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
                    (
                        identity_status,
                        profile + " 构建/probe 身份一致性",
                        identity_note,
                    ),
                )
            )
        probe = (version or {}).get("probe", {})
        compiled = probe.get("build_info", {})
        descriptor = probe.get("backend", {})
        build_rows.append(
            (
                profile,
                role,
                LABELS[build_status],
                LABELS[version_status],
                (built or {}).get("source_commit", "未提供"),
                (built or {}).get("qualification_source_commit", "未单列"),
                descriptor.get("selector", "未提供"),
                probe.get("device_environment", {}).get("execution_space", "未提供"),
                probe.get("device_environment", {}).get("device_name", "未提供"),
                LABELS[identity_status] + "；" + identity_note,
                link(built_path)
                if built_path.is_file()
                else link(build_failure_path)
                if build_failure_path
                else "—",
                link(version_path) if version_path.is_file() else "—",
            )
        )
        tests = junit(evidence, profile)
        if not tests:
            if formal:
                checks.append(
                    (
                        "PENDING",
                        profile + " JUnit 测试",
                        "未找到 JUnit，构建标记不替代独立测试计数",
                    )
                )
            test_rows.append((profile, role, "待完成", "未找到 JUnit", "—"))
        for status, path, note, _ in tests:
            if formal:
                checks.append((status, profile + " JUnit " + path.name, note))
            test_rows.append((profile, role, LABELS[status], note, link(path)))
        if profile.startswith("cpu"):
            status, path, report, note = cpu_runtime(evidence, profile)
            if status == "PASSED":
                context_info = report.get("qualification_context", {}).get(
                    "native_build_info", {}
                )
                if not built or not compiled:
                    status, note = (
                        "PENDING",
                        "doctor 已通过，等待当前构建/probe 身份记录齐全",
                    )
                elif any(
                    context_info.get(key) != compiled.get(key)
                    for key in (
                        "source_commit",
                        "source_content_sha256",
                        "native_source_content_sha256",
                    )
                ):
                    status, note = "FAILED", "doctor 与当前构建/probe 的源身份不匹配"
            runtime_statuses[profile] = status
            if formal:
                checks.append(
                    (status, profile + " 实际 Torch/POLAR 严格运行环境资格", note)
                )
            runtime_rows.append(
                (
                    profile,
                    role,
                    LABELS[status],
                    note,
                    link(path) if path.is_file() else "—",
                )
            )
    regression_rows, attribution_rows, regression_checks = full_cpu_regression(evidence)
    checks.extend(regression_checks)
    current_attribution_rows, current_attribution_checks = (
        current_cpu_regression_attribution(evidence)
    )
    checks.extend(current_attribution_checks)
    diagnostic_rows = []
    md_thread_rows = []
    diagnostic_path, diagnostic, _ = evidence.first([root / "tests/cpu-profile8.json"])
    if diagnostic:
        concurrency = diagnostic.get("openmp_runtime", {}).get(
            "kokkos_concurrency", "未提供"
        )
        for call in diagnostic.get("calls", []):
            timings = call.get("timings_ms", {})
            diagnostic_rows.append(
                (
                    call.get("label", "未提供"),
                    concurrency,
                    diagnostic.get("atoms", "未提供"),
                    fmt(call.get("wall_ms")),
                    fmt(timings.get("native_forward")),
                    fmt(timings.get("torch_suffix_forward")),
                    fmt(timings.get("torch_autograd")),
                    fmt(timings.get("native_vjp")),
                    "同步诊断；不计正式单核性能",
                    link(diagnostic_path),
                )
            )
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
                optimization.get("optimizer_class", "未提供"),
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
        for profile in profiles:
            formal = profile in PROFILES
            role = "正式" if formal else "历史诊断；OpenMP 未资格"
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
                required_phase = phase == "validation" or waters == 64
                if formal and required_phase:
                    checks.append((status, f"{profile}/{waters}/{phase}", note))
                elif data:
                    note += (
                        "；额外 MD 覆盖，不替代 64 水代表体系"
                        if formal
                        else "；仅保留数值对照，不替代 cpu-omp 正式资格"
                    )
                phase_label = LABELS[status]
                if not required_phase and not data and not error and not failure:
                    phase_label = "未执行（本轮代表体系为64水）"
                    note = "本轮只要求 64 水代表体系 MD100/MD1000；此项不计入待完成"
                rows = measurements(data)
                if phase.startswith("md") and data and profile.startswith("cpu"):
                    for executor in ("original", "candidate"):
                        provenance = data.get(executor, {})
                        early = provenance.get("early_execution_environment", {})
                        actual = provenance.get("actual_executor", {})
                        runtime = actual.get("openmp_runtime", {})
                        md_thread_rows.append(
                            (
                                profile,
                                waters,
                                phase,
                                executor,
                                early.get("policy", "未提供"),
                                str(early.get("allowed_after", "未提供")),
                                str(provenance.get("cpu_affinity", "未提供")),
                                json.dumps(
                                    provenance.get("thread_environment", {}),
                                    sort_keys=True,
                                ),
                                actual.get(
                                    "kokkos_concurrency",
                                    runtime.get("kokkos_concurrency", "未提供"),
                                ),
                                runtime.get("host_blas", {}).get(
                                    "openblas_threads", "未提供"
                                ),
                                runtime.get("omp_max_active_levels", "未提供"),
                                link(path),
                            )
                        )
                charge, spin = population_errors(rows)
                physics_rows.append(
                    (
                        profile,
                        role,
                        waters,
                        phase,
                        phase_label,
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
            if (
                profile.startswith("cpu")
                and status == "PASSED"
                and runtime_statuses.get(profile) != "PASSED"
            ):
                status, note = (
                    "PENDING",
                    "计时记录存在，但实际 Torch/POLAR 栈严格 doctor 尚未为 ok",
                )
            if formal:
                checks.append((status, f"{profile}/{waters}/performance", note))
            else:
                note += "；历史 CPU 诊断，不计作正式性能资格"
            executors = (summary or {}).get("executors", {})
            perf_rows.append(
                (
                    profile,
                    role,
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
    (
        module_rows,
        module_region_rows,
        module_stage_rows,
        copy_rows,
        sync_rows,
        module_checks,
    ) = module_diagnostics(evidence)
    checks.extend(module_checks)
    pipeline_rows = pipeline_records(evidence)
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
        "正式 CPU 优先使用独立的 cpu-omp 配置；它必须在真实 Torch/POLAR 能量和力计算后通过严格 doctor。旧 cpu 的数值、MD 和计时记录保留为历史诊断，不转记为 cpu-omp 的结果，也不计入正式配置的完成项目。GPU 编译提交与后续资格脚本提交分别记录。",
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
                "用途",
                "构建",
                "probe",
                "编译 source commit",
                "资格脚本 commit",
                "backend",
                "execution space",
                "实际设备",
                "身份一致性",
                "build 证据",
                "probe 证据",
            ],
            build_rows,
        ),
        (
            "实际测试",
            ["profile", "用途", "状态", "实际计数与真实模型 oracle", "证据"],
            test_rows,
        ),
        (
            "CPU 实际 Torch/POLAR 栈运行环境",
            ["profile", "用途", "严格资格", "实际检查", "证据"],
            runtime_rows,
        ),
        (
            "完整 CPU 回归与干净上游对照",
            ["范围", "原始测试计数", "证据"],
            regression_rows,
        ),
        (
            "当前 cpu-omp 完整回归失败归因",
            [
                "本轮分类/原始状态",
                "失败/错误数",
                "本轮 testcase 与失败文本",
                "与历史的关系/判定边界",
                "证据",
            ],
            current_attribution_rows,
        ),
        (
            "历史 cpu 完整回归失败归因（不套用到 cpu-omp）",
            ["证据支持的原因", "失败数", "实际失败 testcases", "归因证据"],
            attribution_rows,
        ),
        (
            "旧 CPU 配置单次诊断",
            [
                "调用",
                "实际 Kokkos workers",
                "atoms",
                "wall ms",
                "native forward ms",
                "Torch suffix ms",
                "Torch autograd ms",
                "native VJP ms",
                "用途",
                "证据",
            ],
            diagnostic_rows,
        ),
        (
            "实际流水线完成范围",
            [
                "profile/job",
                "完成标记",
                "通过 stage / 已记录 stage",
                "实际输出范围",
                "证据",
            ],
            pipeline_rows,
        ),
        (
            "共同准备态",
            [
                "waters",
                "状态",
                "实际 optimizer class",
                "optimizer steps",
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
                "用途",
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
            "CPU MD 实际线程策略",
            [
                "profile",
                "waters",
                "阶段",
                "executor",
                "导入前策略",
                "导入前 affinity",
                "实际 affinity",
                "实际线程环境",
                "记录的 native concurrency",
                "实际 OpenBLAS 全局线程数",
                "实际 omp_max_active_levels",
                "证据",
            ],
            md_thread_rows,
        ),
        (
            "完整 ASE MD 性能",
            [
                "profile",
                "用途",
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
            "独立模块诊断（与正式吞吐量分开）",
            [
                "profile/job",
                "状态",
                "atoms",
                "调用数",
                "精度",
                "选定长程 forward ms（median/min/max）",
                "整个 Torch autograd ms（median/min/max）",
                "插桩总阶段 median ms",
                "max |ΔE| eV",
                "max |ΔFcomp| eV/Å",
                "bridge",
                "证据",
            ],
            module_rows,
        ),
        (
            "模块 forward 明细（每次计算内多次调用取合计）",
            [
                "profile/job",
                "模块",
                "每次计算实际调用数",
                "inclusive ms（median/min/max）",
                "exclusive ms（median/min/max）",
            ],
            module_region_rows,
        ),
        (
            "同一模块插桩调用的整体阶段",
            ["profile/job", "阶段", "ms（median/min/max）"],
            module_stage_rows,
        ),
        (
            "Nsight 无 hooks 正常单次调用：GPU 内存操作",
            [
                "profile/job",
                "GPU 操作",
                "次数",
                "GPU activity 总 ms",
                "总 MB（Nsight 单位）",
                "配套调用状态",
                "证据",
            ],
            copy_rows,
        ),
        (
            "Nsight 同次调用：主机同步 API 持续时间",
            ["profile/job", "API", "次数", "主机 API 总 ms", "平均 µs", "证据"],
            sync_rows,
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
    if evidence.historical_tests:
        lines.extend(
            [
                "## 已归档的测试尝试",
                "",
                "attempts/ 下的历史 JUnit 不替代当前配置目录的测试结果；以下保留原始状态，不计入当前正式资格。",
                "",
                table(
                    ["profile", "当时状态", "原始测试计数", "证据"],
                    [
                        (profile, LABELS[status], note, link(path))
                        for profile, path, status, note in evidence.historical_tests
                    ],
                ),
                "",
            ]
        )
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
            "正式物理阈值：总胞能量误差 ≤ 1e-4 eV，最大力分量误差 ≤ 1e-5 eV/Å，初始及两轮更新后的 |Q−1| ≤ 1e-9 e、|spin population| ≤ 1e-9。水合壳按 O–Li 最小镜像距离 ≤ 3.2 Å 选取整分子。四尺寸要求 validation/benchmark；NVE 只要求 64 水代表体系，0.25 fs、100/1000 步；漂移仅报告实际数值，没有自创漂移阈值。正式性能为 FP64、原模型/候选各 3 次独立进程、每块至少 10 秒。",
            "",
            "MD 可以单独记录不同的 CPU 线程策略；实际 affinity、线程环境、native concurrency、OpenBLAS 全局线程数和 omp_max_active_levels 分列。OpenMP OpenBLAS 的实际全局线程数可以为 8，即使 OPENBLAS_NUM_THREADS=1；共享 OpenMP 且嵌套层数为 1 不等于全局 BLAS 线程数为 1，也不单独证明每次 BLAS 调用启动多少 workers。轨迹运行时间不替代严格单核的正式 benchmark，缺少实际计数时不从环境变量推断。",
            "",
            "流水线 passed 只覆盖其实际 stage 输出。非 64 水尺寸没有 MD 记录时标为未执行，不计作本轮待完成；额外 MD 记录仍按实际状态展示。模块诊断自动发现 benchmark/module-profile/<job>/<profile>-modules.json，cpu-omp/sm80/sm90 各自为必需项；每配置以最新记录判定，历史失败仍保留在诊断表。必须有完整同步耗时及逐次插桩、恢复后的 E/F 一致性记录，才能计作通过。",
            "",
            "模块计时包含 Python/hooks 和逐区域 GPU 同步开销，不属于正式吞吐量。选定长程 forward 为 descriptor geometry、两次更新×两 spin 的四次 dynamic field 以及 Coulomb energy；场更新 MLP、局部电子能、局部 readout 分列。倒空间枚举、归一化及模块外的平衡步骤仍在其余 suffix 中。整个 Torch autograd 包含所有后缀导数，未独立测出长程 backward。各项中位数分别计算，不能要求其相加严格等于总中位数。",
            "",
            "feature_export 包含 native 特征快照复制与同步，DLPack 接口目前不是零复制；native_vjp 包含 adjoint 导入/复制、桥接同步与局部反向。native_vjp 之后的力快照导出、坐标还原/归约和最终主机传输归 force_result_assembly；graph 和 native_forward 还各含图张量传输、输入 staging。",
            "",
            "Nsight 仅捕获另一次 profile=False、无模块 hooks 的正常计算及末尾完成同步。GPU memcpy/memset 是设备 activity；同步 API 是主机调用持续时间，其中包含等待 GPU kernels/复制完成的时间，两者可能重叠，不能相加，也不能把 API 持续时间称为独立的纯同步开销。全调用的 D2D 含 Torch/native 内部缓冲复制，不能全部归为特征桥接。带 Unified 标签的迁移保留为独立项。trace、SQLite、统计文本及 JSON 的 SHA-256 保存在证据清单。",
            "",
            "跳过的测试没有被当作已验证能力。原模型/候选速度只取已有汇总，区域最大值与阶段中位数由实际记录计算。NVIDIA own PID 数值包含 Torch/native 分配；Torch-only 不代表进程显存。all-GPU 数据含其他进程，不能当作模型独占显存。未提供的计数或数值保持缺失，没有填零。",
            "",
            "当前完整 CPU 回归只取 cpu-omp-full.xml；未完成前保持待完成，不从旧 cpu-full.xml 借用计数。历史 cpu 的原始 1230 passed / 17 failed / 394 skipped 及四项干净上游复现独立保留，不套用到新配置。历史上游对照通过只表示同名失败及打印误差指标复现，不表示那四项测试通过；13 项历史环境解释也不转记为 passed。JUnit 的 teardown error 可以与同一个 testcase 的 failure 同时存在，不据此减出负的 passed 数。",
            "",
            "## 按现有缺口继续",
            "",
            "1. 若上表 cpu-omp 仍有缺口，先补全其构建身份、真实 Torch/POLAR 严格 doctor 和独立 oracle；旧 cpu 的长轨迹可保留科学对照证据，但不能替代新配置的资格。",
            "2. 补全正式配置的四尺寸 validation/benchmark，以及 64 水代表体系的 100/1000 步 NVE 和各配置模块诊断。部分轨迹仅报告最后实际采样步，不将等待或已提交当作完成。",
            "3. 正式速度对比需原模型/候选各三个独立进程，每块至少十秒、FP64、单物理核、profile=False，并保留各 worker 初态、二进制和运行库身份。没有完整计时证据前不填入速度结论。",
            "4. 完整回归仍未全通过时，单独记录环境/API 限制和干净上游复现，不放宽物理或数值阈值来消除失败。",
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
