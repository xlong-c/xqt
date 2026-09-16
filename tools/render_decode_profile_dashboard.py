"""Render a self-contained HTML dashboard for the MiniCPM5-2B W4A8 decode profile.

The dashboard is a local viewing artifact: it reads the torch-profiler JSON written
by ``examples/xqt_models/minicpm5_2b_w4a8_profile.py`` plus the raw ncu counter
CSVs, and renders per-step kernel budgets, ncu counters and byte-budget arithmetic
as charts and tables, so the W4A8 vs W4A16 comparison does not require reading the
whole R-055 record in ``docs/md/explanation/operator-optimization-records.md``.

Output goes next to the profiled JSON under ``artifacts/`` (gitignored). The HTML
is self-contained: no CDN, no external CSS, no JavaScript, so it opens offline and
can be copied anywhere.

Configuration is code-level constants per workspace rules; no CLI parsing.
"""

from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROFILE_JSON = Path("artifacts/xqt/inference/minicpm5-2b/w4a8_baseline_profile.json")
OUTPUT_HTML = Path("artifacts/xqt/inference/minicpm5-2b/profile_report.html")

# First existing path per mode wins. The artifact copy is produced by running the
# profile script under ncu with ``-o``; the /tmp copies are the R-055 raw exports.
NCU_SOURCES: dict[str, tuple[Path, ...]] = {
    "w4a16": (
        Path("artifacts/xqt/inference/minicpm5-2b/ncu/w4a16.csv"),
        Path("/tmp/ncu2_w4a16.csv"),
        Path("/tmp/ncu_w4a16.csv"),
    ),
    "w4a8": (
        Path("artifacts/xqt/inference/minicpm5-2b/ncu/w4a8.csv"),
        Path("/tmp/ncu2_w4a8.csv"),
        Path("/tmp/ncu_w4a8.csv"),
    ),
}

MODE_LABELS: tuple[str, str] = ("w4a16", "w4a8")

# ``_ncu_oneshot`` replays two prefill + 3 + 3 eager decode steps, so the raw
# counters cover exactly 6 decode steps. Used only as a fallback: the real step
# count is derived from the capture itself (see ``_derive_steps``).
ONESHOT_STEPS = 6

DURATION_METRIC = "gpu__time_duration.sum"
# ``--csv`` reports raw ns and byte, while re-exporting a report with
# ``--import --csv`` switches to us/ms and scales each byte row to Kbyte/Mbyte,
# so every value has to be normalised through its own unit column.
DURATION_UNIT_TO_NS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}
BYTE_UNIT_TO_BYTES = {
    "byte": 1.0,
    "Kbyte": 1e3,
    "Mbyte": 1e6,
    "Gbyte": 1e9,
    "Tbyte": 1e12,
}
# Decode kernel used as the step-count anchor: it runs exactly once per layer per
# step, so its launch count divided by the profiler calls/step gives the steps.
ANCHOR_KERNEL = "attention partial (split-K 段1)"

# R-055 measured pure-read ceiling on this box (RTX 4070 Ti SUPER).
BANDWIDTH_CEILING_GBPS = 697.0
TARGET_SPEEDUP = 2.0

MEAN_METRICS = (
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__grid_size",
)
SUM_METRICS = (
    DURATION_METRIC,
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes.sum",
)
BYTE_METRICS = (
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes.sum",
)

# Ordered (substring, label, category) rules; first match wins.
KERNEL_RULES: tuple[tuple[str, str, str], ...] = (
    # torch profiler demangles the template args as true/false, ncu reports 1/0.
    ("awq_decode_kernel<__nv_bfloat16, 1, true>", "AWQ GEMV (残差 epilogue)", "gemv"),
    ("awq_decode_kernel<__nv_bfloat16, 1, 1>", "AWQ GEMV (残差 epilogue)", "gemv"),
    ("awq_decode_kernel<__nv_bfloat16, 1, false>", "AWQ GEMV (无 epilogue)", "gemv"),
    ("awq_decode_kernel<__nv_bfloat16, 1, 0>", "AWQ GEMV (无 epilogue)", "gemv"),
    ("_decode_attn_partial_kernel", "attention partial (split-K 段1)", "attn"),
    ("_decode_attn_merge_kernel", "attention merge (段2)", "attn"),
    ("_rope_write_qkv_kernel", "RoPE + KV scatter", "attn"),
    ("_rmsnorm_int8_kernel", "rmsnorm int8", "norm"),
    ("vectorized_layer_norm_kernel", "aten layer_norm (bf16)", "norm"),
    ("_swiglu_int8_kernel", "SwiGLU int8", "norm"),
    ("_swiglu_kernel", "SwiGLU", "norm"),
    ("ArgMaxOps", "argmax", "small"),
    ("indexSelectSmallIndex", "embedding gather (prefill)", "small"),
    ("FillFunctor<long>", "fill long (初始化)", "small"),
    ("FillFunctor<int>", "fill int (初始化)", "small"),
    ("Memcpy DtoD", "Memcpy DtoD", "other"),
    ("memcpy32_post", "memcpy32_post", "other"),
    ("bfloat16_copy_kernel", "bf16 copy", "other"),
    ("neg_kernel", "elementwise neg", "other"),
    ("direct_copy_kernel", "elementwise copy", "other"),
    ("BinaryFunctor", "elementwise binary (fp32)", "other"),
)

CATEGORY_ORDER = ("gemv", "attn", "norm", "small", "other")
CATEGORY_LABELS = {
    "gemv": "GEMV",
    "attn": "attention",
    "norm": "norm / activation",
    "small": "小 kernel",
    "other": "其他",
    "prefill": "prefill / 一次性",
}

CSS = """
:root{
  --xdl-page:#f5f6f8; --xdl-surface:#ffffff; --xdl-surface-soft:#fafbfc;
  --xdl-text:#1c2024; --xdl-muted:#5c6672; --xdl-border:#dfe3e8;
  --xdl-accent:#2563eb; --xdl-accent-soft:#e8f0fe; --xdl-code-bg:#f2f4f7;
  --bar-a:#3b82f6; --bar-b:#f59e0b; --bar-c:#a855f7; --bar-d:#10b981;
  --ok:#15803d; --warn:#b45309; --bad:#b91c1c;
}
.d-neg{color:var(--ok)}
.d-pos{color:var(--bad)}
@media (prefers-color-scheme: dark){
  :root{
    --xdl-page:#15181c; --xdl-surface:#1d2126; --xdl-surface-soft:#22272d;
    --xdl-text:#e6e9ed; --xdl-muted:#9aa4b0; --xdl-border:#313841;
    --xdl-accent:#6ea8fe; --xdl-accent-soft:#1e2b3d; --xdl-code-bg:#22272d;
    --bar-a:#60a5fa; --bar-b:#fbbf24; --bar-c:#c084fc; --bar-d:#34d399;
    --ok:#4ade80; --warn:#fbbf24; --bad:#f87171;
  }
  .d-neg{color:var(--ok)}
  .d-pos{color:var(--bad)}
}
*{box-sizing:border-box}
body{margin:0;background:var(--xdl-page);color:var(--xdl-text);
  font:15px/1.6 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans CJK SC",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 64px}
h1{font-size:26px;margin:0 0 6px}
h2{font-size:19px;margin:34px 0 6px;padding-bottom:6px;border-bottom:1px solid var(--xdl-border)}
h3{font-size:15px;margin:22px 0 6px;color:var(--xdl-muted)}
p{margin:6px 0}
.muted{color:var(--xdl-muted)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0 4px}
.card{background:var(--xdl-surface);border:1px solid var(--xdl-border);border-radius:8px;
  padding:12px 14px;min-width:158px;flex:1 1 158px}
.card .k{font-size:12px;color:var(--xdl-muted);margin-bottom:4px}
.card .v{font-size:21px;font-weight:600;letter-spacing:-.3px}
.card .s{font-size:12px;color:var(--xdl-muted);margin-top:3px}
.table-wrap{overflow-x:auto;background:var(--xdl-surface);border:1px solid var(--xdl-border);
  border-radius:8px;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--xdl-border)}
th{background:var(--xdl-surface-soft);font-weight:600;color:var(--xdl-muted);font-size:12px}
th:first-child,td:first-child{text-align:left}
th.l,td.l{text-align:left}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:var(--xdl-surface-soft)}
tfoot td{font-weight:600;background:var(--xdl-surface-soft)}
.badge{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11px;
  background:var(--xdl-accent-soft);color:var(--xdl-accent);border:1px solid var(--xdl-border)}
.badge.gemv{color:var(--bar-a);border-color:var(--bar-a)}
.badge.attn{color:var(--bar-c);border-color:var(--bar-c)}
.badge.norm{color:var(--bar-d);border-color:var(--bar-d)}
.badge.small{color:var(--bar-b);border-color:var(--bar-b)}
.badge.prefill{color:var(--xdl-muted);border-color:var(--xdl-border)}
.panel{background:var(--xdl-surface);border:1px solid var(--xdl-border);border-radius:8px;
  padding:14px;margin:12px 0;overflow-x:auto}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:var(--xdl-muted);margin-bottom:6px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;
  vertical-align:-1px}
.legend i.a{background:var(--bar-a)}
.legend i.b{background:var(--bar-b)}
svg{display:block;max-width:100%;height:auto}
.bar-label{fill:var(--xdl-text);font-size:12px}
.bar-val{fill:var(--xdl-muted);font-size:11px;font-family:ui-monospace,Menlo,Consolas,monospace}
.axis{stroke:var(--xdl-border)}
.note{background:var(--xdl-surface-soft);border-left:3px solid var(--xdl-accent);
  border-radius:0 6px 6px 0;padding:10px 14px;margin:12px 0;font-size:13.5px}
.warn{border-left-color:var(--bar-b)}
ul{margin:6px 0 6px 20px;padding:0}
li{margin:3px 0}
code{background:var(--xdl-code-bg);border:1px solid var(--xdl-border);border-radius:4px;
  padding:1px 5px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px}
"""


def _to_float(value: Any) -> float | None:
    """Parse a CSV metric value; return ``None`` when the field is empty or 'NA'."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _friendly_kernel(name: str) -> tuple[str, str]:
    """Map a mangled CUDA kernel name to ``(label, category)``."""

    for pattern, label, category in KERNEL_RULES:
        if pattern in name:
            return label, category
    return (name[:44] or "unknown"), "other"


def load_profile(path: Path) -> dict[str, Any]:
    """Load the torch-profiler JSON artifact."""

    return json.loads(path.read_text(encoding="utf-8"))


def load_ncu_rows(path: Path) -> list[dict[str, str]]:
    """Load ncu CSV rows, skipping the leading ``==PROF==`` banner lines.

    Raises ``ValueError`` when the file carries no CSV header, which is what a
    still-being-written export looks like.
    """

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.startswith('"ID"'):
            return list(csv.DictReader(lines[index:]))
    raise ValueError(f"{path} has no ncu CSV header")


def aggregate_ncu(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Aggregate raw counter rows into per-kernel sums, means and launch counts."""

    agg: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.get("Kernel Name") or ""
        metric = row.get("Metric Name") or ""
        value = _to_float(row.get("Metric Value"))
        if value is None or metric not in SUM_METRICS + MEAN_METRICS:
            continue
        if metric == DURATION_METRIC:
            value *= DURATION_UNIT_TO_NS.get(row.get("Metric Unit", "ns"), 1.0)
        elif metric in BYTE_METRICS:
            value *= BYTE_UNIT_TO_BYTES.get(row.get("Metric Unit", "byte"), 1.0)
        label, category = _friendly_kernel(name)
        slot = agg.setdefault(
            label,
            {
                "category": category,
                "sums": defaultdict(float),
                "counts": defaultdict(int),
                "grid": row.get("Grid Size") or "",
                "block": row.get("Block Size") or "",
            },
        )
        slot["sums"][metric] += value
        slot["counts"][metric] += 1
    return agg


def _profiler_calls(budget: dict[str, Any], modes: tuple[str, str]) -> dict[str, float]:
    """Decode kernels and their calls/step, taken from the torch-profiler budget.

    The profiler only traced ``decode_batch``, so any ncu kernel missing here was
    not part of a decode step (prefill or one-off setup work).
    """

    calls: dict[str, float] = {}
    for mode in modes:
        for row in budget[mode]["rows"]:
            label, _category = _friendly_kernel(row["name"])
            calls[label] = max(calls.get(label, 0.0), float(row["calls_per_step"]))
    return calls


def _derive_steps(
    ncu_agg: dict[str, dict[str, dict[str, Any]]], profiler_calls: dict[str, float]
) -> int:
    """Decode steps covered by the ncu capture.

    The capture may be decode-only (NVTX filtered) or the whole process, and ncu
    may replay kernels, so the launch count is not usable directly. Dividing the
    anchor kernel's launch count by its profiler calls/step recovers the number of
    captured steps in every case.
    """

    anchor_calls = profiler_calls.get(ANCHOR_KERNEL)
    if not anchor_calls:
        return ONESHOT_STEPS
    for agg in ncu_agg.values():
        slot = agg.get(ANCHOR_KERNEL)
        if slot and slot["counts"].get(DURATION_METRIC):
            return max(1, round(slot["counts"][DURATION_METRIC] / anchor_calls))
    return ONESHOT_STEPS


def _mean(slot: dict[str, Any], metric: str) -> float | None:
    """Mean of one metric across the launches that reported it."""

    count = slot["counts"].get(metric, 0)
    if not count:
        return None
    return slot["sums"][metric] / count


def _budget_rows(
    budget: dict[str, Any], modes: tuple[str, str]
) -> list[tuple[str, str, float, float]]:
    """Merge both torch-profiler budgets into ``(label, category, us_a, us_b)`` rows."""

    merged: dict[str, dict[str, Any]] = {}
    for mode in modes:
        for row in budget[mode]["rows"]:
            label, category = _friendly_kernel(row["name"])
            slot = merged.setdefault(
                label, {"category": category, "us": {m: 0.0 for m in modes}}
            )
            slot["us"][mode] += row["cuda_us_per_step"]
    rows = [
        (label, slot["category"], slot["us"][modes[0]], slot["us"][modes[1]])
        for label, slot in merged.items()
    ]
    rows.sort(key=lambda r: -max(r[2], r[3]))
    return rows


def _svg_budget_chart(
    rows: list[tuple[str, str, float, float]], modes: tuple[str, str]
) -> str:
    """Horizontal grouped bar chart of per-step kernel time (us/step)."""

    label_w, bar_max, row_h = 268, 430, 34
    top = 34
    width = label_w + bar_max + 96
    height = top + row_h * len(rows) + 10
    peak = max((max(r[2], r[3]) for r in rows), default=1.0) or 1.0

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="单步 kernel 预算对比">',
        f'<line class="axis" x1="{label_w}" y1="{top - 8}" x2="{label_w}" '
        f'y2="{height - 6}" stroke-width="1"/>',
    ]
    for i, (label, _category, us_a, us_b) in enumerate(rows):
        y = top + i * row_h
        parts.append(
            f'<text class="bar-label" x="{label_w - 12}" y="{y + 13}" '
            f'text-anchor="end">{html.escape(label)}</text>'
        )
        for offset, value, css in ((0, us_a, "a"), (15, us_b, "b")):
            w = max(value / peak * bar_max, 0.6)
            parts.append(
                f'<rect x="{label_w}" y="{y + offset}" width="{w:.1f}" height="12" '
                f'rx="2" fill="var(--bar-{css})"><title>{html.escape(modes[0] if css == "a" else modes[1])} '
                f"{value:.1f} us/step</title></rect>"
            )
            parts.append(
                f'<text class="bar-val" x="{label_w + w + 7:.1f}" y="{y + offset + 10}">'
                f"{value:.0f}</text>"
            )
    parts.append(
        f'<text class="bar-val" x="{label_w}" y="{top - 16}">us/step (峰值 {peak:.0f})</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _card(key: str, value: str, sub: str) -> str:
    """One KPI card."""

    return (
        f'<div class="card"><div class="k">{html.escape(key)}</div>'
        f'<div class="v">{value}</div><div class="s">{html.escape(sub)}</div></div>'
    )


def _delta(value: float, fmt: str = "{:+.1f}", tol: float = 0.05) -> str:
    """Colour a per-step delta: lower is better, so negative is good."""

    css = "d-neg" if value < -tol else ("d-pos" if value > tol else "muted")
    return f"<span class='{css}'>{fmt.format(value)}</span>"


def _budget_table(
    rows: list[tuple[str, str, float, float]], modes: tuple[str, str]
) -> str:
    """Per-kernel torch-profiler budget table with deltas."""

    total = [sum(r[2] for r in rows), sum(r[3] for r in rows)]
    head = (
        "<tr><th class='l'>内核</th><th class='l'>类别</th>"
        f"<th>{modes[0]} us/步</th><th>{modes[1]} us/步</th><th>Δ us/步</th>"
        "<th>占比 A</th><th>占比 B</th></tr>"
    )
    body = []
    for label, category, us_a, us_b in rows:
        delta = us_b - us_a
        body.append(
            f"<tr><td class='l'>{html.escape(label)}</td>"
            f"<td class='l'><span class='badge {category}'>"
            f"{CATEGORY_LABELS[category]}</span></td>"
            f"<td>{us_a:.1f}</td><td>{us_b:.1f}</td>"
            f"<td>{_delta(delta)}</td>"
            f"<td>{us_a / total[0] * 100:.1f}%</td>"
            f"<td>{us_b / total[1] * 100:.1f}%</td></tr>"
        )
    foot = (
        f"<tfoot><tr><td class='l'>合计</td><td class='l'></td>"
        f"<td>{total[0]:.1f}</td><td>{total[1]:.1f}</td>"
        f"<td>{_delta(total[1] - total[0])}</td><td>100%</td><td>100%</td></tr></tfoot>"
    )
    return f"<div class='table-wrap'><table>{head}{''.join(body)}{foot}</table></div>"


def _ncu_table(
    agg: dict[str, dict[str, dict[str, Any]]],
    modes: tuple[str, str],
    steps: int,
    decode_labels: set[str],
) -> str:
    """Merged per-kernel ncu counter table for both modes.

    Decode kernels are reported per step. Kernels that are not part of a decode
    step (prefill, one-off setup) are reported as totals and marked, because
    dividing them by the step count would invent a number.
    """

    labels: set[str] = set()
    for mode in modes:
        labels |= set(agg[mode])

    decode_rows: list[tuple[str, list[Any], list[float]]] = []
    prefill_rows: list[tuple[str, list[Any], list[float]]] = []
    for label in labels:
        slots = [agg[mode].get(label) for mode in modes]
        # A kernel belongs to the decode step when the profiler traced it (the
        # profiler only ran during decode_batch) or when its launch count is an
        # exact multiple of the step count. The second test is what keeps a
        # decode-only capture correct for kernels the profiler dropped.
        counts = [
            slot["counts"].get(DURATION_METRIC, 0) if slot else 0 for slot in slots
        ]
        is_decode = label in decode_labels or any(
            count and count % steps == 0 for count in counts
        )
        scale = 1e6 * steps if is_decode else 1e6
        ms = [
            (slot["sums"][DURATION_METRIC] / scale) if slot else 0.0 for slot in slots
        ]
        target = decode_rows if is_decode else prefill_rows
        target.append((label, slots, ms))
    decode_rows.sort(key=lambda r: -max(r[2]))
    prefill_rows.sort(key=lambda r: -max(r[2]))

    head = (
        "<tr><th class='l'>内核</th><th class='l'>类别</th>"
        f"<th>{modes[0]} ms/步</th><th>{modes[1]} ms/步</th><th>Δ ms/步</th>"
        "<th>DRAM% A / B</th>"
        "<th>读 GB/步 A / B</th>"
        "<th>写 GB/步 A / B</th>"
        "<th>L2 GB/步 A / B</th>"
        "<th>regs A / B</th><th>warps% A / B</th>"
        "<th class='l'>grid A / B</th><th>启动/步 A / B</th></tr>"
    )

    def mean_cell(slots: list[Any], metric: str, fmt: str) -> str:
        values = []
        for slot in slots:
            mean = _mean(slot, metric) if slot else None
            values.append("--" if mean is None else fmt.format(mean))
        return f"<td>{values[0]} / {values[1]}</td>"

    def bytes_cell(slots: list[Any], metric: str, decode: bool) -> str:
        values = []
        for slot in slots:
            if slot is None or not decode:
                values.append("--")
                continue
            total = slot["sums"].get(metric)
            values.append("--" if total is None else f"{total / 1e9 / steps:.3f}")
        return f"<td>{values[0]} / {values[1]}</td>"

    body = []
    for label, slots, ms in decode_rows + prefill_rows:
        counts = [
            slot["counts"].get(DURATION_METRIC, 0) if slot else 0 for slot in slots
        ]
        decode = label in decode_labels or any(
            count and count % steps == 0 for count in counts
        )
        category = (
            next((s["category"] for s in slots if s), "other") if decode else "prefill"
        )
        if decode:
            ms_cells = (
                f"<td>{ms[0]:.3f}</td><td>{ms[1]:.3f}</td>"
                f"<td>{_delta(ms[1] - ms[0], '{:+.3f}', 0.005)}</td>"
            )
            launches = "/".join(
                f"{slot['counts'][DURATION_METRIC] / steps:.0f}" if slot else "--"
                for slot in slots
            )
        else:
            ms_cells = (
                f"<td>{ms[0]:.3f}*</td><td>{ms[1]:.3f}*</td><td class='muted'>--</td>"
            )
            launches = "/".join(
                f"{slot['counts'][DURATION_METRIC]}" if slot else "--" for slot in slots
            )
        grid = next((s["grid"] for s in slots if s and s["grid"]), "--")
        body.append(
            f"<tr><td class='l'>{html.escape(label)}</td>"
            f"<td class='l'><span class='badge {category}'>"
            f"{CATEGORY_LABELS[category]}</span></td>"
            + ms_cells
            + mean_cell(
                slots, "dram__throughput.avg.pct_of_peak_sustained_elapsed", "{:.1f}"
            )
            + bytes_cell(slots, "dram__bytes_read.sum", decode)
            + bytes_cell(slots, "dram__bytes_write.sum", decode)
            + bytes_cell(slots, "lts__t_bytes.sum", decode)
            + mean_cell(slots, "launch__registers_per_thread", "{:.0f}")
            + mean_cell(
                slots, "sm__warps_active.avg.pct_of_peak_sustained_active", "{:.1f}"
            )
            + f"<td class='l mono'>{html.escape(grid)}</td>"
            f"<td>{launches}</td></tr>"
        )
    return f"<div class='table-wrap'><table>{head}{''.join(body)}</table></div>"


def _source_listing(paths: list[Path]) -> str:
    """Render the source artifact list with sizes and mtimes."""

    items = []
    for path in paths:
        if path.exists():
            stat = path.stat()
            when = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            items.append(
                f"<li><code>{html.escape(str(path))}</code> "
                f"<span class='muted'>{stat.st_size / 1024:.0f} KB, {when}</span></li>"
            )
        else:
            items.append(
                f"<li><code>{html.escape(str(path))}</code> "
                f"<span class='muted'>缺失</span></li>"
            )
    return f"<ul>{''.join(items)}</ul>"


def main() -> None:
    profile = load_profile(PROFILE_JSON)
    modes = MODE_LABELS
    budget = {mode: profile[f"kernel_budget_{mode}"] for mode in modes}
    rows = _budget_rows(budget, modes)

    ncu_agg: dict[str, dict[str, dict[str, Any]]] = {}
    ncu_paths: dict[str, Path] = {}
    for mode in modes:
        for candidate in NCU_SOURCES[mode]:
            if not candidate.exists():
                continue
            try:
                rows_ncu = aggregate_ncu(load_ncu_rows(candidate))
            except (ValueError, OSError) as exc:
                print(f"[dashboard] skip {candidate}: {exc}")
                continue
            ncu_paths[mode] = candidate
            ncu_agg[mode] = rows_ncu
            break

    steady = profile["steady"]
    ratio = profile["steady_step_ratio_w4a16_over_w4a8"]
    step_ms = {mode: steady[mode]["step_ms_median"] for mode in modes}
    tok_s = {mode: steady[mode]["steady_tok_s_median"] for mode in modes}
    prefill = {mode: profile["prefill_ms"][mode]["median"] for mode in modes}
    total_us = {
        mode: budget[mode]["total_cuda_us"] / profile["profiler_steps"]
        for mode in modes
    }

    weight_gb = None
    gemv_gbps = None
    profiler_calls = _profiler_calls(budget, modes)
    decode_labels = set(profiler_calls)
    steps = _derive_steps(ncu_agg, profiler_calls)
    if modes[1] in ncu_agg:
        gemv_slots = [
            slot for slot in ncu_agg[modes[1]].values() if slot["category"] == "gemv"
        ]
        weight_gb = (
            sum(slot["sums"]["dram__bytes_read.sum"] for slot in gemv_slots)
            / 1e9
            / steps
        )
        gemv_ms = (
            sum(slot["sums"][DURATION_METRIC] for slot in gemv_slots) / 1e6 / steps
        )
        gemv_gbps = weight_gb / (gemv_ms / 1000.0)

    cards = [
        _card(
            f"steady ({modes[0]} → {modes[1]})",
            f"{tok_s[modes[0]]:.0f} → {tok_s[modes[1]]:.0f}",
            f"tok/s, 比值 {ratio:.3f}x",
        ),
        _card(
            "step 延迟", f"{step_ms[modes[0]]:.3f} / {step_ms[modes[1]]:.3f}", "ms/step"
        ),
        _card("prefill", f"{prefill[modes[0]]:.1f} / {prefill[modes[1]]:.1f}", "ms"),
        _card(
            "单步 kernel 合计",
            f"{total_us[modes[0]]:.0f} / {total_us[modes[1]]:.0f}",
            "us/step (torch profiler)",
        ),
    ]
    if weight_gb is not None and gemv_gbps is not None:
        cards.append(
            _card("GEMV 权重读/步", f"{weight_gb:.2f} GB", "w4a16 / w4a8 完全相同")
        )
        cards.append(
            _card(
                "GEMV 读带宽",
                f"{gemv_gbps:.0f} GB/s",
                f"ncu 窗口内, 占 {BANDWIDTH_CEILING_GBPS:.0f} GB/s 上界 "
                f"{gemv_gbps / BANDWIDTH_CEILING_GBPS * 100:.0f}%",
            )
        )
    cards.append(
        _card(
            "距 2x 目标",
            f"{ratio:.3f}x / {TARGET_SPEEDUP:.1f}x",
            "同运行 steady 对照",
        )
    )

    ncu_section = ""
    if ncu_agg:
        ncu_section = (
            f"<h2>ncu 计数器 ({steps} 步 decode, 按 kernel 聚合)</h2>"
            "<p class='muted'>"
            f"步数由 <code>{ANCHOR_KERNEL}</code> 的启动次数与 torch profiler 的 "
            "calls/step 交叉推算, 因此 decode-only 与整进程捕获都能正确折算. "
            f"每步数值 = 原始计数器 / {steps}. 表内 <code>A / B</code> 即 "
            f"<code>{modes[0]} / {modes[1]}</code>. DRAM% 为逐次启动的 "
            "<code>dram__throughput.avg.pct_of_peak_sustained_elapsed</code> 均值. "
            "读/写/L2 为每步字节量. 带 <code>*</code> 的行不属于 decode step "
            "(prefill / 一次性), 显示总计而非每步. ncu 逐 kernel 串行重放, "
            "单个 kernel 的耗时不含并发重叠, 因此本列不可纵向相加 (相加会明显"
            "大于真实 step 时间); 真实单步预算以上面的 torch profiler 表为准.</p>"
            + _ncu_table(ncu_agg, modes, steps, decode_labels)
        )
    else:
        ncu_section = (
            "<h2>ncu 计数器</h2>"
            "<div class='note warn'>未找到 ncu CSV. 生成方式见本页末尾 "
            "<code>复现命令</code>.</div>"
        )

    sources = [PROFILE_JSON, *ncu_paths.values()]

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MiniCPM5-2B W4A8 decode profile 面板</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>MiniCPM5-2B · W4A8 vs W4A16 decode profile</h1>
  <p class="muted">{html.escape(profile["model_id"])} · {html.escape(profile["device"])} ·
     单请求 greedy, batch={profile["steady_batch"]}, median-of-{profile["steady_repeats"]},
     warmup={profile["warmup_steps"]}, profiler={profile["profiler_steps"]} 步</p>
  <div class="cards">{"".join(cards)}</div>
  <div class="note">
    同一进程内两个 <code>CudaGraphDecodeSession</code> 共享同一份 AWQ W4 量化模型,
    交错测量. 单次稳态读数跨运行会波动, 所以只看同一次运行内的对照.
  </div>

  <h2>单步 kernel 预算 (torch profiler)</h2>
  <div class="panel">
    <div class="legend"><span><i class="a"></i>{modes[0]}</span>
      <span><i class="b"></i>{modes[1]}</span></div>
    {_svg_budget_chart(rows, modes)}
  </div>
  {_budget_table(rows, modes)}

  {ncu_section}

  <h2>产物来源</h2>
  {_source_listing(sources)}

  <h2>复现命令</h2>
  <p class="muted">torch profiler 面板:</p>
  <p class="mono">PYTHONPATH=. python examples/xqt_models/minicpm5_2b_w4a8_profile.py</p>
  <p class="muted">ncu 计数器 (逐 mode 跑, <code>W4A8_INT8=1</code> 即 w4a8; NVTX 只圈
     <code>eager_decode</code> 窗口, 不含 prefill 噪声):</p>
  <p class="mono">W4A8_PROFILE_MODE=ncu-oneshot W4A8_INT8=0 PYTHONPATH=. ncu \\
    --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\\
sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,\\
launch__grid_size,lts__t_bytes.sum \\
    --target-processes all --nvtx --nvtx-include "eager_decode/" \\
    -o artifacts/xqt/inference/minicpm5-2b/ncu/w4a16 -f \\
    python examples/xqt_models/minicpm5_2b_w4a8_profile.py<br>
    ncu --import artifacts/xqt/inference/minicpm5-2b/ncu/w4a16.ncu-rep --csv \\
    --log-file artifacts/xqt/inference/minicpm5-2b/ncu/w4a16.csv</p>
  <p class="muted">把 <code>-o ... -f</code> 生成的 <code>.ncu-rep</code> 用
     <code>ncu-ui</code> 打开即得 GUI 视图 (WSLg 下直接跑 <code>ncu-ui</code> 会落到
     Mesa 软件渲染, 能开但滚动较慢).</p>

  <h2>判读要点</h2>
  <ul>
    <li>GEMV 是纯 DRAM 流: 每步读字节与 <code>dram%</code> 直接给出权重带宽地板.</li>
    <li>w4a8 的 int8 rmsnorm 省时间, 但 <code>swiglu int8</code> 每 tile 多做 row max 又吐回去.</li>
    <li>小 kernel (RoPE scatter / attention merge) 的 <code>warps%</code> 只有个位数, 是启动地板, 不是带宽.</li>
    <li>完整归因与已否决方案见 <code>docs/md/explanation/operator-optimization-records.md</code> 的 R-055.</li>
  </ul>
</div>
</body>
</html>
"""
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html_doc, encoding="utf-8")
    print(
        f"[dashboard] written {OUTPUT_HTML} ({OUTPUT_HTML.stat().st_size / 1024:.0f} KB)"
    )
    print(
        f"[dashboard] steady {modes[0]} {tok_s[modes[0]]:.1f} -> {modes[1]} "
        f"{tok_s[modes[1]]:.1f} tok/s, ratio {ratio:.4f}, kernel {total_us[modes[0]]:.0f}"
        f" -> {total_us[modes[1]]:.0f} us/step"
    )
    for mode in modes:
        print(f"[dashboard] ncu {mode}: {ncu_paths.get(mode, 'missing')}")


if __name__ == "__main__":
    main()
