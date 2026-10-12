"""Result types: per-check outcomes and the aggregated QA report."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum


class Status(IntEnum):
    SKIPPED = -1
    PASS = 0
    WARN = 1
    FAIL = 2

    @property
    def label(self) -> str:
        return self.name


@dataclass
class CheckResult:
    name: str
    status: Status
    summary: str
    metrics: dict[str, float] = field(default_factory=dict)
    details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.label,
            "summary": self.summary,
            "metrics": {k: _num(v) for k, v in self.metrics.items()},
            "details": self.details,
        }


def _num(v):
    f = float(v)
    return f if f == f and abs(f) != float("inf") else None  # JSON has no NaN / inf


@dataclass
class QAReport:
    mode: str
    checks: list[CheckResult]
    meta: dict = field(default_factory=dict)

    @property
    def status(self) -> Status:
        return max((c.status for c in self.checks), default=Status.PASS)

    def check(self, name: str) -> CheckResult:
        for c in self.checks:
            if c.name == name:
                return c
        raise KeyError(name)

    @property
    def exit_code(self) -> int:
        """0 = pass, 1 = warn, 2 = fail (usable from cron / CI)."""
        return max(int(self.status), 0)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "status": self.status.label,
            "meta": self.meta,
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_text(self) -> str:
        width = max(len(c.name) for c in self.checks)
        lines = [f"GammaGuard QA report ({self.mode} mode): {self.status.label}"]
        lines += [f"  [{c.status.label:<7}] {c.name:<{width}}  {c.summary}" for c in self.checks]
        return "\n".join(lines)

    def to_markdown(self) -> str:
        out = [f"# GammaGuard QA report: **{self.status.label}**", "", f"Mode: `{self.mode}`", ""]
        out += ["| Check | Status | Summary |", "|---|---|---|"]
        out += [f"| {c.name} | {c.status.label} | {c.summary} |" for c in self.checks]
        for c in self.checks:
            if not c.details:
                continue
            keys = list(c.details[0].keys())
            out += ["", f"## {c.name}", "", "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
            for row in c.details:
                out.append("| " + " | ".join(_fmt(row.get(k)) for k in keys) + " |")
        return "\n".join(out) + "\n"


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return "" if v is None else str(v)
