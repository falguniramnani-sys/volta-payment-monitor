"""
Core data pipeline for Volta Payment Monitor.

Provides loading, filtering, and aggregation functions that produce
P50/P95/P99 latency metrics, health scores, histograms, and heatmaps
consumed by the Streamlit dashboard.
"""

import pandas as pd
import numpy as np


# ── Loading ───────────────────────────────────────────────────────────────────

def load_transactions(path: str = "data/transactions.csv") -> pd.DataFrame:
    """Load transactions CSV and parse timestamp column."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


# ── Filtering ─────────────────────────────────────────────────────────────────

def apply_filters(
    df: pd.DataFrame,
    countries=None,
    methods=None,
    psps=None,
    start_time=None,
    end_time=None,
) -> pd.DataFrame:
    """Return subset of df matching all provided filter criteria."""
    mask = pd.Series(True, index=df.index)
    if countries:
        mask &= df["country"].isin(countries)
    if methods:
        mask &= df["payment_method"].isin(methods)
    if psps:
        mask &= df["psp"].isin(psps)
    if start_time is not None:
        mask &= df["timestamp"] >= pd.Timestamp(start_time)
    if end_time is not None:
        mask &= df["timestamp"] <= pd.Timestamp(end_time)
    return df[mask].copy()


# ── Health score ──────────────────────────────────────────────────────────────

def compute_health_score(
    p95: float, approval_rate: float, timeout_rate: float
) -> float:
    """
    Health score 0-100:
      40% latency component:   1 - clamp(p95 / 10_000, 0, 1)
      40% approval component:  approval_rate
      20% reliability:         1 - clamp(timeout_rate / 0.15, 0, 1)
    """
    latency_score = max(0.0, 1.0 - p95 / 10_000)
    approval_score = float(approval_rate)
    reliability_score = max(0.0, 1.0 - timeout_rate / 0.15)
    raw = 0.40 * latency_score + 0.40 * approval_score + 0.20 * reliability_score
    return round(raw * 100, 1)


# ── PSP metrics ───────────────────────────────────────────────────────────────

def compute_psp_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per PSP with: p50, p95, p99, mean, approval_rate, timeout_rate,
    decline_rate, count, health_score.

    Latency percentiles exclude timeout rows.
    """
    rows = []
    for psp, group in df.groupby("psp"):
        count = len(group)
        timeout_count = (group["status"] == "timeout").sum()
        approved_count = (group["status"] == "approved").sum()
        declined_count = (group["status"] == "declined").sum()

        timeout_rate = timeout_count / count if count else 0.0
        approval_rate = approved_count / count if count else 0.0
        decline_rate = declined_count / count if count else 0.0

        non_timeout = group[group["status"] != "timeout"]["latency_ms"]
        if len(non_timeout) > 0:
            p50 = non_timeout.quantile(0.50)
            p95 = non_timeout.quantile(0.95)
            p99 = non_timeout.quantile(0.99)
            mean = non_timeout.mean()
        else:
            p50 = p95 = p99 = mean = float("nan")

        health = compute_health_score(
            p95 if not np.isnan(p95) else 0, approval_rate, timeout_rate
        )

        rows.append({
            "psp": psp,
            "p50": round(p50, 1) if not np.isnan(p50) else None,
            "p95": round(p95, 1) if not np.isnan(p95) else None,
            "p99": round(p99, 1) if not np.isnan(p99) else None,
            "mean": round(mean, 1) if not np.isnan(mean) else None,
            "approval_rate": round(approval_rate, 4),
            "timeout_rate": round(timeout_rate, 4),
            "decline_rate": round(decline_rate, 4),
            "count": count,
            "health_score": health,
        })
    return pd.DataFrame(rows).set_index("psp")


# ── Time series ───────────────────────────────────────────────────────────────

def compute_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hourly P50/P95/P99/count per PSP (excludes timeout rows from latency).
    """
    working = df.copy()
    working["timestamp_hour"] = working["timestamp"].dt.floor("h")

    non_timeout = working[working["status"] != "timeout"]

    result = (
        non_timeout.groupby(["timestamp_hour", "psp"])["latency_ms"]
        .quantile([0.50, 0.95, 0.99])
        .unstack(level=-1)
        .reset_index()
    )
    result.columns = ["timestamp_hour", "psp", "p50", "p95", "p99"]

    counts = (
        working.groupby(["timestamp_hour", "psp"])
        .size()
        .reset_index(name="count")
    )
    result = result.merge(counts, on=["timestamp_hour", "psp"], how="left")
    return result.sort_values(["psp", "timestamp_hour"]).reset_index(drop=True)


# ── Histogram buckets ─────────────────────────────────────────────────────────

HISTOGRAM_BINS = [0, 500, 1000, 2000, 3000, 5000, 8000, 15000, float("inf")]
HISTOGRAM_LABELS = [
    "0-0.5s", "0.5-1s", "1-2s", "2-3s", "3-5s", "5-8s", "8-15s", "15s+"
]


def compute_histogram(df: pd.DataFrame) -> pd.DataFrame:
    """Transaction count per latency bucket per PSP (excludes timeouts)."""
    non_timeout = df[df["status"] != "timeout"].copy()
    non_timeout["bucket"] = pd.cut(
        non_timeout["latency_ms"],
        bins=HISTOGRAM_BINS,
        labels=HISTOGRAM_LABELS,
        right=False,
    )
    return (
        non_timeout.groupby(["psp", "bucket"], observed=True)
        .size()
        .reset_index(name="count")
    )


# ── Heatmaps ─────────────────────────────────────────────────────────────────

def compute_country_psp_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: rows=country, cols=PSP, values=P95 latency (non-timeout)."""
    non_timeout = df[df["status"] != "timeout"]
    p95 = (
        non_timeout.groupby(["country", "psp"])["latency_ms"]
        .quantile(0.95)
        .reset_index(name="p95")
    )
    pivot = p95.pivot(index="country", columns="psp", values="p95").round(1)
    pivot.columns.name = None
    return pivot


def compute_method_psp_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: rows=payment_method, cols=PSP, values=P95 latency (non-timeout)."""
    non_timeout = df[df["status"] != "timeout"]
    p95 = (
        non_timeout.groupby(["payment_method", "psp"])["latency_ms"]
        .quantile(0.95)
        .reset_index(name="p95")
    )
    pivot = p95.pivot(index="payment_method", columns="psp", values="p95").round(1)
    pivot.columns.name = None
    return pivot


# ── Worst combinations ────────────────────────────────────────────────────────

def compute_worst_combinations(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Ranked table of worst-performing (country, payment_method, PSP) combos
    by P95 latency. Includes approval_rate and timeout_rate.
    """
    non_timeout = df[df["status"] != "timeout"]
    groups = df.groupby(["country", "payment_method", "psp"])

    rows = []
    for (country, method, psp), group in groups:
        count = len(group)
        if count < 5:
            continue
        non_to = group[group["status"] != "timeout"]["latency_ms"]
        if len(non_to) == 0:
            continue
        rows.append({
            "country": country,
            "payment_method": method,
            "psp": psp,
            "p95": round(non_to.quantile(0.95), 1),
            "p50": round(non_to.quantile(0.50), 1),
            "approval_rate": round((group["status"] == "approved").mean(), 4),
            "timeout_rate": round((group["status"] == "timeout").mean(), 4),
            "count": count,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("p95", ascending=False).head(top_n).reset_index(drop=True)


# ── Payment method metrics ────────────────────────────────────────────────────

def compute_method_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """P50/P95/P99 per payment_method (excludes timeout rows)."""
    non_timeout = df[df["status"] != "timeout"]
    result = (
        non_timeout.groupby("payment_method")["latency_ms"]
        .quantile([0.50, 0.95, 0.99])
        .unstack(level=-1)
        .reset_index()
    )
    result.columns = ["payment_method", "p50", "p95", "p99"]
    counts = df.groupby("payment_method").size().reset_index(name="count")
    result = result.merge(counts, on="payment_method")
    return result.round({"p50": 1, "p95": 1, "p99": 1})


# ── Anomaly detection ─────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag anomalous hourly windows:
    - P95 > 2x rolling baseline for a PSP
    - Timeout rate > 5% in any hourly window
    - Approval rate < 70% in any hourly window
    """
    working = df.copy()
    working["timestamp_hour"] = working["timestamp"].dt.floor("h")

    alerts = []
    psp_metrics = compute_psp_metrics(df)

    for psp in df["psp"].unique():
        baseline_p95 = psp_metrics.loc[psp, "p95"] if psp in psp_metrics.index else None
        if baseline_p95 is None or np.isnan(baseline_p95):
            continue

        psp_data = working[working["psp"] == psp]
        for hour, group in psp_data.groupby("timestamp_hour"):
            non_to = group[group["status"] != "timeout"]["latency_ms"]
            if len(non_to) < 5:
                continue

            hourly_p95 = non_to.quantile(0.95)
            hourly_timeout = (group["status"] == "timeout").mean()
            hourly_approval = (group["status"] == "approved").mean()

            if hourly_p95 > 2 * baseline_p95:
                alerts.append({
                    "timestamp": hour, "psp": psp, "metric": "p95_spike",
                    "severity": "critical" if hourly_p95 > 3 * baseline_p95 else "warning",
                    "value": round(hourly_p95, 1),
                    "threshold": round(2 * baseline_p95, 1),
                    "message": f"{psp} P95 {hourly_p95:.0f}ms ({hourly_p95/baseline_p95:.1f}x baseline)",
                })

            if hourly_timeout > 0.05:
                alerts.append({
                    "timestamp": hour, "psp": psp, "metric": "timeout_rate",
                    "severity": "critical" if hourly_timeout > 0.10 else "warning",
                    "value": round(hourly_timeout, 4),
                    "threshold": 0.05,
                    "message": f"{psp} timeout rate {hourly_timeout*100:.1f}%",
                })

            if hourly_approval < 0.70:
                alerts.append({
                    "timestamp": hour, "psp": psp, "metric": "approval_rate",
                    "severity": "critical" if hourly_approval < 0.60 else "warning",
                    "value": round(hourly_approval, 4),
                    "threshold": 0.70,
                    "message": f"{psp} approval rate {hourly_approval*100:.1f}%",
                })

    return pd.DataFrame(alerts)


# ── CLI verification ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_transactions()
    print(f"Loaded {len(df):,} transactions\n")

    metrics = compute_psp_metrics(df)
    print("=== PSP Metrics ===")
    print(metrics[["p50", "p95", "p99", "approval_rate", "timeout_rate", "health_score"]].to_string())
    print()

    method_metrics = compute_method_metrics(df)
    print("=== Payment Method Metrics ===")
    print(method_metrics.to_string(index=False))
    print()

    heatmap = compute_country_psp_heatmap(df)
    print("=== Country x PSP P95 Heatmap ===")
    print(heatmap.to_string())
    print()

    worst = compute_worst_combinations(df)
    print("=== Worst Combinations ===")
    print(worst.to_string(index=False))
