"""
Smart PSP routing recommendation engine for Volta Payment Monitor.

For each (country, payment_method) pair, scores available PSPs on:
  - 40% latency component  (based on normalized P95)
  - 40% approval rate
  - 20% reliability (inverse timeout rate)

Returns routing recommendations with estimated impact.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from src.pipeline import compute_psp_metrics


def _score_psp(p95_ms: float, approval_rate: float, timeout_rate: float) -> float:
    """
    Composite routing score 0-1.
    score = 0.40 * (1 - normalized_p95) + 0.40 * approval_rate + 0.20 * (1 - timeout_rate)
    """
    latency_score = max(0.0, 1.0 - p95_ms / 10_000)
    reliability_score = max(0.0, 1.0 - timeout_rate / 0.15)
    return 0.40 * latency_score + 0.40 * approval_rate + 0.20 * reliability_score


def compute_routing_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return routing recommendations for each (country, payment_method) pair.

    Columns: country, payment_method, volume, recommended_psp, rec_score,
    current_best_psp, current_score, expected_p95, expected_approval,
    impact_p95_ms, impact_description
    """
    rows = []

    combos = (
        df.groupby(["country", "payment_method"])
        .size()
        .reset_index(name="volume")[["country", "payment_method", "volume"]]
    )

    for _, combo in combos.iterrows():
        country = combo["country"]
        method = combo["payment_method"]
        volume = combo["volume"]

        slice_df = df[(df["country"] == country) & (df["payment_method"] == method)]
        if len(slice_df) < 10:
            continue

        psp_metrics = compute_psp_metrics(slice_df)
        if psp_metrics.empty:
            continue

        # Score each PSP
        psp_scores = {}
        for psp in psp_metrics.index:
            p95 = psp_metrics.loc[psp, "p95"]
            approval = psp_metrics.loc[psp, "approval_rate"]
            timeout = psp_metrics.loc[psp, "timeout_rate"]
            if p95 is None or np.isnan(p95):
                continue
            psp_scores[psp] = {
                "score": _score_psp(p95, approval, timeout),
                "p95": p95,
                "p50": psp_metrics.loc[psp, "p50"],
                "approval_rate": approval,
                "timeout_rate": timeout,
            }

        if len(psp_scores) < 2:
            continue

        sorted_psps = sorted(psp_scores, key=lambda p: psp_scores[p]["score"], reverse=True)
        best = sorted_psps[0]
        worst = sorted_psps[-1]

        # Current "best" = highest volume PSP for this route
        psp_volumes = slice_df["psp"].value_counts()
        current_psp = psp_volumes.index[0]

        best_data = psp_scores[best]
        current_data = psp_scores.get(current_psp, best_data)

        impact_p95 = round(current_data["p95"] - best_data["p95"], 1)

        impact_desc = ""
        if impact_p95 > 0 and current_psp != best:
            impact_desc = (
                f"Switching {volume} txns from {current_psp} to {best} "
                f"would reduce P95 by {impact_p95/1000:.1f}s"
            )

        rows.append({
            "country": country,
            "payment_method": method,
            "volume": volume,
            "recommended_psp": best,
            "rec_score": round(best_data["score"], 4),
            "current_best_psp": current_psp,
            "current_score": round(current_data["score"], 4),
            "expected_p95": round(best_data["p95"], 1),
            "expected_approval": round(best_data["approval_rate"], 4),
            "impact_p95_ms": impact_p95,
            "impact_description": impact_desc,
        })

    return pd.DataFrame(rows).sort_values("impact_p95_ms", ascending=False).reset_index(drop=True)


def compute_score_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return detailed score breakdown per PSP per (country, payment_method).
    Used for the health score visualization.
    """
    rows = []
    combos = df.groupby(["country", "payment_method"]).size().reset_index()[
        ["country", "payment_method"]
    ]

    for _, combo in combos.iterrows():
        country = combo["country"]
        method = combo["payment_method"]

        slice_df = df[(df["country"] == country) & (df["payment_method"] == method)]
        psp_metrics = compute_psp_metrics(slice_df)

        for psp in psp_metrics.index:
            p95 = psp_metrics.loc[psp, "p95"]
            approval = psp_metrics.loc[psp, "approval_rate"]
            timeout = psp_metrics.loc[psp, "timeout_rate"]
            if p95 is None or np.isnan(p95):
                continue

            latency_component = max(0.0, 1.0 - p95 / 10_000) * 0.40
            approval_component = approval * 0.40
            reliability_component = max(0.0, 1.0 - timeout / 0.15) * 0.20
            total = latency_component + approval_component + reliability_component

            rows.append({
                "country": country,
                "payment_method": method,
                "psp": psp,
                "latency_component": round(latency_component, 4),
                "approval_component": round(approval_component, 4),
                "reliability_component": round(reliability_component, 4),
                "total_score": round(total, 4),
            })

    return pd.DataFrame(rows)
