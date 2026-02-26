"""
Real-time alerting logic for Volta Payment Monitor.

Evaluates hourly windows and generates alerts with severity levels.
Alert conditions:
  - P95 > 5s (warning) / P95 > 8s (critical)
  - Timeout rate > 5% (warning) / > 10% (critical)
  - Approval rate < 70% (warning) / < 60% (critical)
  - Hourly P95 spike > 2x baseline (warning) / > 3x (critical)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from src.pipeline import load_transactions, compute_timeseries, compute_psp_metrics


def detect_alerts(df: pd.DataFrame) -> list[dict]:
    """
    Run all alert rules and return a list of alert dicts.

    Each dict: psp, type, severity, message, current_value, threshold, timestamp
    """
    alerts: list[dict] = []
    psp_metrics = compute_psp_metrics(df)
    ts_df = compute_timeseries(df)

    # Rule 1: Hourly P95 spike vs PSP baseline
    for psp in psp_metrics.index:
        baseline_p95 = psp_metrics.loc[psp, "p95"]
        if baseline_p95 is None or np.isnan(baseline_p95) or baseline_p95 == 0:
            continue

        psp_ts = ts_df[ts_df["psp"] == psp]
        for _, row in psp_ts.iterrows():
            hourly_p95 = row["p95"]
            if pd.isna(hourly_p95):
                continue
            ratio = hourly_p95 / baseline_p95

            if ratio > 3.0:
                severity = "critical"
            elif ratio > 2.0:
                severity = "warning"
            else:
                continue

            alerts.append({
                "psp": psp,
                "type": "p95_spike",
                "severity": severity,
                "message": (
                    f"{psp} P95 spike at {row['timestamp_hour'].strftime('%Y-%m-%d %H:00')} "
                    f"- {hourly_p95:.0f}ms is {ratio:.1f}x baseline ({baseline_p95:.0f}ms)"
                ),
                "current_value": round(hourly_p95, 1),
                "threshold": round(baseline_p95 * (3.0 if severity == "critical" else 2.0), 1),
                "timestamp": row["timestamp_hour"],
            })

    # Rule 2: Timeout rate thresholds
    for psp in psp_metrics.index:
        timeout_rate = psp_metrics.loc[psp, "timeout_rate"]
        if timeout_rate > 0.10:
            severity, threshold = "critical", 0.10
        elif timeout_rate > 0.05:
            severity, threshold = "warning", 0.05
        else:
            continue

        alerts.append({
            "psp": psp,
            "type": "timeout_rate",
            "severity": severity,
            "message": f"{psp} timeout rate {timeout_rate*100:.1f}% exceeds {threshold*100:.0f}% threshold",
            "current_value": round(timeout_rate, 4),
            "threshold": threshold,
            "timestamp": None,
        })

    # Rule 3: Approval rate thresholds
    for psp in psp_metrics.index:
        approval_rate = psp_metrics.loc[psp, "approval_rate"]
        if approval_rate < 0.60:
            severity, threshold = "critical", 0.60
        elif approval_rate < 0.70:
            severity, threshold = "warning", 0.70
        else:
            continue

        alerts.append({
            "psp": psp,
            "type": "approval_rate",
            "severity": severity,
            "message": f"{psp} approval rate {approval_rate*100:.1f}% below {threshold*100:.0f}% threshold",
            "current_value": round(approval_rate, 4),
            "threshold": threshold,
            "timestamp": None,
        })

    # Rule 4: Absolute P95 thresholds
    for psp in psp_metrics.index:
        p95 = psp_metrics.loc[psp, "p95"]
        if p95 is None or np.isnan(p95):
            continue

        if p95 > 8000:
            severity, threshold = "critical", 8000
        elif p95 > 5000:
            severity, threshold = "warning", 5000
        else:
            continue

        alerts.append({
            "psp": psp,
            "type": "absolute_p95",
            "severity": severity,
            "message": f"{psp} overall P95 {p95:.0f}ms exceeds {threshold:,}ms threshold",
            "current_value": round(p95, 1),
            "threshold": threshold,
            "timestamp": None,
        })

    return alerts


def get_alert_summary(alerts: list[dict]) -> dict:
    """Summarize alerts by severity and type."""
    critical = [a for a in alerts if a["severity"] == "critical"]
    warnings = [a for a in alerts if a["severity"] == "warning"]

    by_type = {}
    for a in alerts:
        by_type.setdefault(a["type"], []).append(a)

    return {
        "total": len(alerts),
        "critical": len(critical),
        "warning": len(warnings),
        "by_type": {k: len(v) for k, v in by_type.items()},
        "affected_psps": list({a["psp"] for a in alerts}),
    }


if __name__ == "__main__":
    df = load_transactions()
    alerts = detect_alerts(df)
    summary = get_alert_summary(alerts)

    print(f"Alerting: {summary['total']} alerts "
          f"({summary['critical']} critical, {summary['warning']} warning)\n")

    seen: set = set()
    for a in alerts:
        key = (a["psp"], a["type"], a["severity"])
        if key not in seen:
            seen.add(key)
            tag = "CRITICAL" if a["severity"] == "critical" else "WARNING"
            print(f"[{tag}] {a['message']}")
