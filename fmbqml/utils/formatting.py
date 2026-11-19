"""Formatting helpers for user-facing estimation and test summaries."""

from typing import Any, Dict

import numpy as np
import pandas as pd

from fmbqml.utils.criteria import CRITERION_NAMES


def _format_float(value: Any, digits: int = 4) -> str:
    """
    Safely format a numeric value.
    """
    if value is None:
        return "None"

    try:
        value = float(value)
    except Exception:
        return str(value)

    if not np.isfinite(value):
        return "nan"

    return f"{value:.{digits}f}"


def _get_list_item(values: Any, index: int, default: Any = "") -> Any:
    """
    Safely get one item from a list-like object.
    """
    if values is None:
        return default

    try:
        if len(values) > index:
            value = values[index]
            return default if value is None else value
    except Exception:
        pass

    return default


def _get_dict_value(data: Any, keys, default: Any = "") -> Any:
    """
    Safely get a value from a dictionary using several possible keys.
    """
    if not isinstance(data, dict):
        return default

    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    return default


def _safe_min_text(a: Any, b: Any) -> str:
    """
    Return min(a, b) as text when possible.
    """
    try:
        return str(min(int(a), int(b)))
    except Exception:
        return ""


def _format_observation_label(value: Any) -> Any:
    """Format datetime-like index labels without inferring missing dates."""
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return None
        if timestamp.time() == pd.Timestamp(0).time():
            return timestamp.strftime("%Y-%m-%d")
        return timestamp.isoformat(sep=" ")
    if value is None or (np.isscalar(value) and pd.isna(value)):
        return None
    return value


def _format_observation_labels(values: Any) -> list:
    """Return observation labels as a display-friendly independent list."""
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    else:
        try:
            values = list(values)
        except TypeError:
            values = [values]
    return [label for value in values
            if (label := _format_observation_label(value)) is not None
            and not (isinstance(label, str) and not label.strip())]


def _format_breakpoints(x: Any) -> str:
    """
    Format breakpoint candidates in a compact readable form.
    """
    if x is None:
        return "[]"

    if isinstance(x, str):
        if x.lower() == "no breakpoint":
            return "[]"
        return x

    try:
        arr = np.asarray(x).reshape(-1)
    except Exception:
        return str(x)

    if arr.size == 0:
        return "[]"

    out = []

    for v in arr:
        try:
            if np.isfinite(v):
                out.append(str(int(round(float(v)))))
        except Exception:
            pass

    return "[" + ", ".join(out) + "]" if out else "[]"


def _format_grid_text(criteria: Any) -> str:
    """
    Format merged search grid information when available.
    """
    grid = _get_dict_value(
        criteria,
        keys=[
            "merged_search_grid",
            "search_grid",
            "grid",
            "candidate_grid",
        ],
        default=None
    )

    if grid is None:
        return ""

    return f"; merged search grid={grid}."


def _format_condition_text(
        break_type: Any,
        r_left: Any,
        r_right: Any,
        r_combined: Any,
        criteria: Any
) -> str:
    """
    Format breakpoint-type classification condition.
    """
    condition = _get_dict_value(
        criteria,
        keys=[
            "condition",
            "classification_condition",
            "decision_rule",
            "rule",
        ],
        default=None
    )

    if condition is not None:
        return str(condition) + _format_grid_text(criteria)

    min_text = _safe_min_text(r_left, r_right)

    if str(break_type).startswith("Type 1"):
        if min_text:
            return (
                f"r_{{j,j+1}} ({r_combined}) > "
                f"min(r_j,r_{{j+1}}) ({min_text})"
                + _format_grid_text(criteria)
            )
        return "Type 1 condition satisfied" + _format_grid_text(criteria)

    if str(break_type).startswith("Type 2"):
        if min_text:
            return (
                f"r_{{j,j+1}} ({r_combined}) <= "
                f"min(r_j,r_{{j+1}}) ({min_text})"
                + _format_grid_text(criteria)
            )
        return "Type 2 condition satisfied" + _format_grid_text(criteria)

    return "No classification condition available" + _format_grid_text(criteria)


class Summary:
    """A structured, display-ready statistical summary.

    Sections retain semantic labels and tabular data; text, HTML, and LaTeX
    representations are generated from the same stored content.
    """
    width = 72

    def __init__(self, title):
        self.title = str(title)
        self.sections = []

    def add_pairs(self, heading, rows):
        self.sections.append((heading, "pairs", list(rows)))

    def add_fields(self, heading, rows):
        # Omit absent observation labels in every output format.
        rows = [(name, value) for name, value in rows
                if not ("breakpoint label" in name.lower()
                        and (value is None or (isinstance(value, str) and value in ("None", "N/A", ""))
                             or (isinstance(value, (list, tuple)) and not value)))]
        self.sections.append((heading, "fields", rows))

    def add_table(self, heading, headers, rows):
        self.sections.append((heading, "table", (list(headers), [list(r) for r in rows])))

    def add_notes(self, notes):
        if notes:
            self.sections.append(("Notes", "notes", list(notes)))

    def as_text(self):
        lines = [self.title, "=" * self.width]
        for heading, kind, content in self.sections:
            lines.extend(["", heading])
            if kind == "pairs":
                for row in content:
                    left, right = row
                    ll, lv = left
                    left_text = f"  {ll + ':':<17}{_display(lv)}"
                    if right is not None:
                        rl, rv = right
                        right_text = f"{rl + ':':<20}{_display(rv)}"
                        lines.append(f"{left_text:<39}{right_text}")
                    else:
                        lines.append(left_text)
            elif kind == "fields":
                label_width = max((len(str(k)) for k, _ in content), default=0) + 2
                for label, value in content:
                    lines.append(f"  {label + ':':<{label_width}}{_display(value)}")
            elif kind == "table":
                headers, rows = content
                str_rows = [[_display(value) for value in r] for r in rows]
                widths = [max(len(str(h)), *(len(r[i]) for r in str_rows))
                          if str_rows else len(str(h)) for i, h in enumerate(headers)]
                def fmt(r):
                    return "  " + "  ".join(str(x).ljust(w) for x, w in zip(r, widths)).rstrip()
                lines.append(fmt(headers))
                lines.append("  " + "-" * (sum(widths) + 2 * (len(widths) - 1)))
                for row in str_rows:
                    lines.append(fmt(row))
            else:
                lines.extend("  " + note for note in content)
        return "\n".join(lines)

    def as_html(self):
        from html import escape
        h = escape
        parts = [f"<h3>{h(self.title)}</h3>"]
        for heading, kind, content in self.sections:
            parts.append(f"<h4>{h(heading)}</h4>")
            if kind == "table":
                headers, rows = content
                parts.append("<table><thead><tr>" + "".join(f"<th>{h(str(x))}</th>" for x in headers) + "</tr></thead><tbody>")
                for row in rows:
                    parts.append("<tr>" + "".join(f"<td>{h(_display(x))}</td>" for x in row) + "</tr>")
                parts.append("</tbody></table>")
            elif kind == "notes":
                parts.append("<p>" + " ".join(h(str(x)) for x in content) + "</p>")
            else:
                parts.append("<table>")
                for row in content:
                    fields = [row] if kind == "fields" else [x for x in row if x is not None]
                    parts.append("<tr>" + "".join(f"<th>{h(str(k))}</th><td>{h(_display(v))}</td>" for k, v in fields) + "</tr>")
                parts.append("</table>")
        return "\n".join(parts)

    def as_latex(self):
        def esc(x):
            mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                       "_": r"\_", "#": r"\#", "$": r"\$",
                       "{": r"\{", "}": r"\}", "^": r"\textasciicircum{}",
                       "~": r"\textasciitilde{}"}
            return "".join(mapping.get(ch, ch) for ch in str(x))
        lines = [r"\begin{center}", r"\textbf{" + esc(self.title) + "}", r"\end{center}"]
        for heading, kind, content in self.sections:
            lines.append(r"\paragraph{" + esc(heading) + "}")
            if kind == "table":
                headers, rows = content
                lines.append(r"\begin{tabular}{" + "l" * len(headers) + "}")
                lines.append(" & ".join(esc(x) for x in headers) + " " + "\\" * 2)
                lines.append(r"\hline")
                lines.extend(" & ".join(esc(_display(x)) for x in row) + " " + "\\" * 2 for row in rows)
                lines.append(r"\end{tabular}")
            elif kind == "notes":
                lines.extend(esc(x) + r"\\" for x in content)
            else:
                lines.append(r"\begin{tabular}{ll}" if kind == "fields" else r"\begin{tabular}{llll}")
                for row in content:
                    vals = [row] if kind == "fields" else [row[0], row[1]]
                    cells = []
                    for field in vals:
                        if field is None:
                            cells.extend(["", ""])
                        else:
                            cells.extend([esc(str(field[0]) + ":"), esc(_display(field[1]))])
                    lines.append(" & ".join(cells) + " " + "\\" * 2)
                lines.append(r"\end{tabular}")
        return "\n".join(lines)

    def __str__(self):
        return self.as_text()

    def __repr__(self):
        return self.as_text()


def _display(value):
    if value is None:
        return "None"
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _rank_triplets(data, n):
    """Return recorded rank triplets without inferring missing statistics."""
    records = data.get("classification_criteria")
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, (list, tuple)):
        records = []
    reg = data.get("regime_factors")
    joint = data.get("combined_factors")
    try:
        reg = np.asarray(reg).reshape(-1).tolist() if reg is not None else []
        joint = np.asarray(joint).reshape(-1).tolist() if joint is not None else []
    except (ValueError, TypeError):
        reg, joint = [], []
    out = []
    for i in range(n):
        record = records[i] if i < len(records) and isinstance(records[i], dict) else {}
        norm = {str(k).lower().replace(" ", "").replace("-", "_"): v for k, v in record.items()}
        def pick(*keys):
            for key in keys:
                v = norm.get(key)
                if v is not None:
                    return v
            return None
        r0 = pick("r_j", "rj")
        r1 = pick("r_j+1", "r_j1", "r_j_plus_1", "rj1")
        rc = pick("r_j,j+1", "r_j_j+1", "r_jj1", "r_j_j1", "r_joint", "combined_rank")
        if r0 is None and i < len(reg):
            r0 = reg[i]
        if r1 is None and i + 1 < len(reg):
            r1 = reg[i + 1]
        if rc is None and i < len(joint):
            rc = joint[i]
        if n == 1:
            r0 = r0 if r0 is not None else data.get("r_j", data.get("rj"))
            r1 = r1 if r1 is not None else data.get("r_j+1", data.get("r_j1"))
            rc = rc if rc is not None else data.get("r_j,j+1", data.get("combined_rank"))
        out.append((r0, r1, rc))
    return out


def _types(data, n):
    values = data.get("break_types")
    if values is None:
        values = [data.get("break_type")] if n == 1 else []
    if isinstance(values, str):
        values = [values]
    try:
        values = np.asarray(values, dtype=object).reshape(-1).tolist()
    except (TypeError, ValueError):
        values = []
    return values if len(values) == n else [None] * n


def _criterion(data):
    value = data.get("factor_criterion")
    if isinstance(value, (int, np.integer)):
        return CRITERION_NAMES.get(int(value), str(value))
    return value


def _num(value):
    return "N/A" if value is None else f"{float(value):.4f}"


def _percent(value):
    return f"{float(value) * 100:g}%" if value is not None else "N/A"


def _class_table(summary, data, locations, multiple=False):
    if (not data.get("classify", True) and not data.get("classification_completed", False)) or not locations:
        return
    types = _types(data, len(locations))
    ranks = _rank_triplets(data, len(locations))
    if not any(t is not None for t in types):
        return
    headers = (["Break"] if multiple else []) + ["Location", "Type", "r_j", "r_j+1", "r_j,j+1"]
    rows = []
    for i, loc in enumerate(locations):
        rank = ["N/A" if x is None else x for x in ranks[i]]
        row = ([i + 1] if multiple else []) + [loc, types[i] or "N/A", *rank]
        rows.append(row)
    summary.add_table("Break Classification", headers, rows)


def format_single_break_result_summary(result: Dict[str, Any]) -> Summary:
    """Build a compact, structured single-break report from stored results."""
    test = result.get("test_type") == "lr_test"
    report = Summary("SingleBreakQML LR Test Results" if test
                     else "SingleBreakQML Estimation Results")
    trim = result.get("trim_ratio")
    report.add_pairs("Model Information", [
        (("Sample size", f"T = {result.get('T')}, N = {result.get('N')}"),
         ("Factor criterion", _criterion(result))),
        (("Max factors", result.get("max_factors")),
         ("Trimming ratio", f"{float(trim):.4f}" if trim is not None else "N/A")),
        (("Search range", result.get("search_range")),
         ("Estimated factors", result.get("estimated_factors"))),
    ])
    if test:
        reject = result.get("reject_null")
        candidate_breakpoint = result.get("candidate_breakpoint")
        if candidate_breakpoint is None and reject is not None and bool(reject):
            candidate_breakpoint = result.get("breakpoint")
        accepted_breakpoint = (
            result.get("breakpoint")
            if reject is not None and bool(reject)
            else None
        )
        breakpoint_fields = [
            ("Candidate breakpoint under H1",
             candidate_breakpoint if candidate_breakpoint is not None else "None"),
        ]
        candidate_label = (
            _format_observation_label(result.get("candidate_break_label"))
            if candidate_breakpoint is not None else None
        )
        if candidate_label is not None:
            breakpoint_fields.append(("Candidate breakpoint label", candidate_label))
        breakpoint_fields.append((
            "Accepted breakpoint",
            accepted_breakpoint if accepted_breakpoint is not None else "None",
        ))
        accepted_label = (
            _format_observation_label(result.get("break_label"))
            if accepted_breakpoint is not None else None
        )
        if accepted_label is not None:
            breakpoint_fields.append(("Accepted breakpoint label", accepted_label))
        report.add_fields("Breakpoint Estimation", breakpoint_fields)

        decision = ("Not available" if reject is None else
                    "Reject H0" if bool(reject) else "Do not reject H0")
        report.add_fields("Hypothesis Test", [
            ("Null hypothesis", "No structural break"),
            ("LR statistic", _num(result.get("test_statistic"))),
            (f"Critical value ({_percent(result.get('significance_level'))})",
             _num(result.get("critical_value"))),
            ("Decision", decision),
        ])
    else:
        breakpoint_fields = [
            ("Estimated breakpoint", result.get("breakpoint")),
        ]
        breakpoint_label = _format_observation_label(result.get("break_label"))
        if breakpoint_label is not None:
            breakpoint_fields.append(("Breakpoint label", breakpoint_label))
        report.add_fields("Breakpoint Estimation", breakpoint_fields)

    loc = result.get("breakpoint")
    if test and not bool(result.get("reject_null")):
        loc = None
    if loc is not None:
        _class_table(report, result, [loc])
    n_sim = result.get("monte_carlo_replications")
    if test and n_sim is not None:
        report.add_notes([
            f"Critical value computed using {n_sim} Monte Carlo replications."
        ])
    return report


def format_multi_break_result_summary(result: Dict[str, Any]) -> Summary:
    """Build a compact joint-QML or joint sup-LR report from stored results."""
    test_type = str(result.get("test_type", "")).lower()
    if "joint_sup_lr_profile" in test_type:
        raise ValueError("A joint LR profile is not a hypothesis-test result.")
    is_lr = ("joint_sup_lr" in test_type or result.get("test_statistic") is not None
             or result.get("critical_value") is not None)
    report = Summary("MultiBreakQML Joint LR Test Results" if is_lr
                     else "MultiBreakQML Joint Estimation Results")
    trim = result.get("trim_ratio")
    report.add_pairs("Model Information", [
        (("Sample size", f"T = {result.get('T')}, N = {result.get('N')}"),
         ("Factor criterion", _criterion(result))),
        (("Max factors", result.get("max_factors")),
         ("Maximum breaks", result.get("max_break"))),
        (("Trimming ratio", f"{float(trim):.4f}" if trim is not None else "N/A"),
         ("Estimated factors", result.get("estimated_factors"))),
    ])

    def _as_list(value: Any) -> list:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        try:
            return np.asarray(value, dtype=object).reshape(-1).tolist()
        except (TypeError, ValueError):
            return []

    accepted_locations = _as_list(result.get("breakpoints"))
    accepted_labels = _format_observation_labels(_as_list(result.get("break_labels")))
    n_breaks = result.get("n_breaks")

    if is_lr:
        candidate_locations = _as_list(
            result.get("candidate_breakpoints", result.get("breakpoints"))
        )
        if n_breaks is None:
            n_breaks = len(candidate_locations)
        alpha = result.get("significance_level", result.get("alpha"))
        reject = result.get("reject_null")
        decision = ("Not available" if reject is None else
                    "Reject H0" if bool(reject) else "Do not reject H0")
        report.add_fields("Hypothesis Test", [
            ("Null hypothesis", "No structural break"),
            ("Alternative hypothesis", f"Exactly {n_breaks} structural {'break' if n_breaks == 1 else 'breaks'}"),
            ("LR statistic", _num(result.get("test_statistic"))),
            (f"Critical value ({_percent(alpha)})", _num(result.get("critical_value"))),
            ("Decision", decision),
        ])
        # Candidate is the H1-optimal partition. Accepted is populated only
        # when the joint LR test rejects the null hypothesis.
        accepted_for_report = (
            accepted_locations
            if reject is not None and bool(reject)
            else []
        )
        candidate_labels = _format_observation_labels(
            _as_list(result.get("candidate_break_labels"))
        )
        accepted_labels_for_report = (
            accepted_labels if accepted_for_report else None
        )
        breakpoint_fields = [
            ("Candidate breakpoints under H1",
             candidate_locations if candidate_locations else "None"),
        ]
        if candidate_labels and candidate_locations:
            breakpoint_fields.append(("Candidate breakpoint labels", candidate_labels))
        breakpoint_fields.append((
            "Accepted breakpoints",
            accepted_for_report if accepted_for_report else "None",
        ))
        if accepted_labels_for_report:
            breakpoint_fields.append(("Accepted breakpoint labels", accepted_labels_for_report))
        report.add_fields("Breakpoint Estimation", breakpoint_fields)
        if accepted_for_report:
            _class_table(report, result, accepted_for_report, multiple=True)
        n_sim = result.get("monte_carlo_replications", result.get("n_sim"))
        if n_sim is not None:
            report.add_notes([f"Critical value computed using {n_sim} Monte Carlo replications."])
    else:
        if n_breaks is None:
            n_breaks = len(accepted_locations)
        report.add_fields("Breakpoint Estimation", [
            ("Number of breaks", n_breaks),
            ("Estimated breakpoints", accepted_locations if accepted_locations else "None"),
            ("Breakpoint labels", accepted_labels if accepted_labels else "None"),
        ])
        _class_table(report, result, accepted_locations, multiple=True)
    return report
