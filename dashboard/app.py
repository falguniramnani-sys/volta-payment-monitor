"""
Volta Payment Monitor — Streamlit Dashboard

Multi-page interactive dashboard for monitoring payment pipeline health
across PSPs, countries, and payment methods.

Pages:
  1. Overview — KPIs and time-series trends
  2. PSP Performance — Side-by-side PSP comparison
  3. Drill-Down Analysis — Filtered deep-dive
  4. Bottleneck Finder — Heatmaps and worst combinations
  5. Routing Recommendations — Smart PSP routing with impact estimates
  6. Alerts — Active alerts and history
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.pipeline import (
    load_transactions,
    apply_filters,
    compute_psp_metrics,
    compute_timeseries,
    compute_histogram,
    compute_country_psp_heatmap,
    compute_method_psp_heatmap,
    compute_worst_combinations,
    compute_method_metrics,
    detect_anomalies,
)
from src.routing import compute_routing_recommendations, compute_score_breakdown
from src.alerting import detect_alerts, get_alert_summary


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Volta Payment Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color palette ─────────────────────────────────────────────────────────────

COLORS = {
    "PSP_Alpha": "#636EFA",
    "PSP_Beta": "#EF553B",
    "PSP_Gamma": "#00CC96",
    "PSP_Delta": "#AB63FA",
}
SEVERITY_COLORS = {"critical": "#FF4B4B", "warning": "#FFA726"}


# ── Data loading (cached) ────────────────────────────────────────────────────

@st.cache_data
def get_data():
    return load_transactions()


@st.cache_data
def get_psp_metrics(_df):
    return compute_psp_metrics(_df)


@st.cache_data
def get_timeseries(_df):
    return compute_timeseries(_df)


# ── Sidebar navigation ───────────────────────────────────────────────────────

st.sidebar.title("⚡ Volta Payment Monitor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "PSP Performance",
        "Drill-Down Analysis",
        "Bottleneck Finder",
        "Routing Recommendations",
        "Alerts",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Volta Commerce | 500K+ daily payments")
st.sidebar.caption("Brazil · Mexico · Colombia")


# ── Load data ─────────────────────────────────────────────────────────────────

df = get_data()


# ═══════════════════════════════════════════════════════════════════════════════
# Page 1: Overview
# ═══════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.title("Payment Pipeline Overview")
    st.markdown("Real-time monitoring of payment transactions across all PSPs and regions.")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    total_txns = len(df)
    avg_latency = df[df["status"] != "timeout"]["latency_ms"].mean()
    approval_rate = (df["status"] == "approved").mean()
    timeout_rate = (df["status"] == "timeout").mean()

    col1.metric("Total Transactions", f"{total_txns:,}")
    col2.metric("Avg Latency", f"{avg_latency:,.0f}ms")
    col3.metric("Approval Rate", f"{approval_rate:.1%}")
    col4.metric("Timeout Rate", f"{timeout_rate:.1%}")

    st.markdown("---")

    # Time-series: latency trends
    st.subheader("Latency Trends (48 Hours)")

    ts = get_timeseries(df)

    # Aggregate across PSPs for overall view
    ts_overall = ts.groupby("timestamp_hour").agg(
        p50=("p50", "median"),
        p95=("p95", "median"),
        p99=("p99", "median"),
        count=("count", "sum"),
    ).reset_index()

    fig_ts = go.Figure()
    for metric, color, dash in [("p50", "#636EFA", "solid"), ("p95", "#EF553B", "dash"), ("p99", "#AB63FA", "dot")]:
        fig_ts.add_trace(go.Scatter(
            x=ts_overall["timestamp_hour"],
            y=ts_overall[metric],
            mode="lines",
            name=metric.upper(),
            line=dict(color=color, dash=dash, width=2),
        ))
    fig_ts.update_layout(
        yaxis_title="Latency (ms)",
        xaxis_title="Time",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Transaction volume by hour
    st.subheader("Transaction Volume by Hour")

    hourly_vol = df.copy()
    hourly_vol["hour"] = hourly_vol["timestamp"].dt.floor("h")
    hourly_counts = hourly_vol.groupby("hour").size().reset_index(name="count")

    fig_vol = px.bar(
        hourly_counts, x="hour", y="count",
        color_discrete_sequence=["#636EFA"],
    )
    fig_vol.update_layout(
        xaxis_title="Time",
        yaxis_title="Transactions",
        height=300,
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # Quick breakdown
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("By Country")
        country_counts = df["country"].value_counts().reset_index()
        country_counts.columns = ["country", "count"]
        fig_c = px.pie(country_counts, values="count", names="country",
                       color_discrete_sequence=px.colors.qualitative.Set2)
        fig_c.update_layout(height=300)
        st.plotly_chart(fig_c, use_container_width=True)

    with col_b:
        st.subheader("By Payment Method")
        method_counts = df["payment_method"].value_counts().reset_index()
        method_counts.columns = ["payment_method", "count"]
        fig_m = px.pie(method_counts, values="count", names="payment_method",
                       color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_m.update_layout(height=300)
        st.plotly_chart(fig_m, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Page 2: PSP Performance
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "PSP Performance":
    st.title("PSP Performance Comparison")

    metrics = get_psp_metrics(df)

    # Side-by-side bar chart: P50/P95/P99
    st.subheader("Latency Percentiles by PSP")

    metrics_reset = metrics.reset_index()
    latency_long = metrics_reset.melt(
        id_vars="psp", value_vars=["p50", "p95", "p99"],
        var_name="percentile", value_name="latency_ms",
    )

    fig_lat = px.bar(
        latency_long, x="psp", y="latency_ms", color="percentile",
        barmode="group",
        color_discrete_map={"p50": "#636EFA", "p95": "#EF553B", "p99": "#AB63FA"},
    )
    fig_lat.update_layout(
        yaxis_title="Latency (ms)", xaxis_title="PSP", height=400,
        legend_title="Percentile",
    )
    st.plotly_chart(fig_lat, use_container_width=True)

    # Approval rate comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Approval Rate")
        fig_apr = px.bar(
            metrics_reset, x="approval_rate", y="psp", orientation="h",
            color="psp", color_discrete_map=COLORS,
        )
        fig_apr.update_layout(
            xaxis_title="Approval Rate", xaxis_tickformat=".0%",
            yaxis_title="", showlegend=False, height=300,
        )
        st.plotly_chart(fig_apr, use_container_width=True)

    with col2:
        st.subheader("Timeout Rate")
        fig_to = px.bar(
            metrics_reset, x="timeout_rate", y="psp", orientation="h",
            color="psp", color_discrete_map=COLORS,
        )
        fig_to.update_layout(
            xaxis_title="Timeout Rate", xaxis_tickformat=".1%",
            yaxis_title="", showlegend=False, height=300,
        )
        st.plotly_chart(fig_to, use_container_width=True)

    # Box plots showing latency distribution
    st.subheader("Latency Distribution per PSP")
    non_timeout = df[df["status"] != "timeout"]
    fig_box = px.box(
        non_timeout, x="psp", y="latency_ms", color="psp",
        color_discrete_map=COLORS,
        points=False,
    )
    fig_box.update_layout(
        yaxis_title="Latency (ms)", xaxis_title="PSP",
        showlegend=False, height=400,
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Key metrics table
    st.subheader("Key Metrics Summary")
    display_metrics = metrics[["p50", "p95", "p99", "mean", "approval_rate", "timeout_rate", "decline_rate", "count", "health_score"]].copy()
    display_metrics["approval_rate"] = display_metrics["approval_rate"].map("{:.1%}".format)
    display_metrics["timeout_rate"] = display_metrics["timeout_rate"].map("{:.1%}".format)
    display_metrics["decline_rate"] = display_metrics["decline_rate"].map("{:.1%}".format)
    st.dataframe(display_metrics, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Page 3: Drill-Down Analysis
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Drill-Down Analysis":
    st.title("Drill-Down Analysis")

    # Sidebar filters
    st.sidebar.markdown("### Filters")

    sel_countries = st.sidebar.multiselect(
        "Country", df["country"].unique().tolist(), default=df["country"].unique().tolist()
    )
    sel_methods = st.sidebar.multiselect(
        "Payment Method", df["payment_method"].unique().tolist(),
        default=df["payment_method"].unique().tolist()
    )
    sel_psps = st.sidebar.multiselect(
        "PSP", df["psp"].unique().tolist(), default=df["psp"].unique().tolist()
    )

    min_time = df["timestamp"].min().to_pydatetime()
    max_time = df["timestamp"].max().to_pydatetime()
    time_range = st.sidebar.slider(
        "Time Range",
        min_value=min_time,
        max_value=max_time,
        value=(min_time, max_time),
        format="MM/DD HH:mm",
    )

    filtered = apply_filters(
        df,
        countries=sel_countries or None,
        methods=sel_methods or None,
        psps=sel_psps or None,
        start_time=time_range[0],
        end_time=time_range[1],
    )

    st.info(f"Showing **{len(filtered):,}** of {len(df):,} transactions")

    # Latency histogram
    st.subheader("Latency Distribution")
    non_to_filtered = filtered[filtered["status"] != "timeout"]

    if len(non_to_filtered) > 0:
        fig_hist = px.histogram(
            non_to_filtered, x="latency_ms", color="psp",
            nbins=50, barmode="overlay", opacity=0.7,
            color_discrete_map=COLORS,
        )
        fig_hist.update_layout(
            xaxis_title="Latency (ms)", yaxis_title="Count", height=400,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("No non-timeout transactions match the selected filters.")

    # Time-series with applied filters
    st.subheader("Latency Trends (Filtered)")
    if len(filtered) > 0:
        ts_filtered = compute_timeseries(filtered)
        if len(ts_filtered) > 0:
            fig_ts_f = px.line(
                ts_filtered, x="timestamp_hour", y="p95", color="psp",
                color_discrete_map=COLORS,
                markers=True,
            )
            fig_ts_f.update_layout(
                yaxis_title="P95 Latency (ms)", xaxis_title="Time", height=400,
            )
            st.plotly_chart(fig_ts_f, use_container_width=True)

    # Comparison table
    st.subheader("Filtered Metrics by PSP")
    if len(filtered) > 0:
        filtered_metrics = compute_psp_metrics(filtered)
        display = filtered_metrics[["p50", "p95", "p99", "approval_rate", "timeout_rate", "count", "health_score"]].copy()
        display["approval_rate"] = display["approval_rate"].map("{:.1%}".format)
        display["timeout_rate"] = display["timeout_rate"].map("{:.1%}".format)
        st.dataframe(display, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Page 4: Bottleneck Finder
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Bottleneck Finder":
    st.title("Bottleneck Finder")

    # Worst bottleneck callout
    worst = compute_worst_combinations(df, top_n=1)
    if not worst.empty:
        w = worst.iloc[0]
        st.error(
            f"**Worst Bottleneck:** {w['country']} / {w['payment_method']} / {w['psp']} "
            f"— P95: {w['p95']:,.0f}ms | Approval: {w['approval_rate']:.1%} | "
            f"Timeout: {w['timeout_rate']:.1%} | {w['count']} transactions"
        )

    st.markdown("---")

    # Heatmap: PSP x Country
    st.subheader("P95 Latency: PSP x Country")
    heatmap_country = compute_country_psp_heatmap(df)

    fig_hm1 = px.imshow(
        heatmap_country.values,
        x=heatmap_country.columns.tolist(),
        y=heatmap_country.index.tolist(),
        color_continuous_scale="RdYlGn_r",
        text_auto=".0f",
        aspect="auto",
    )
    fig_hm1.update_layout(
        xaxis_title="PSP", yaxis_title="Country", height=350,
        coloraxis_colorbar_title="P95 (ms)",
    )
    st.plotly_chart(fig_hm1, use_container_width=True)

    # Heatmap: PSP x Payment Method
    st.subheader("P95 Latency: PSP x Payment Method")
    heatmap_method = compute_method_psp_heatmap(df)

    fig_hm2 = px.imshow(
        heatmap_method.values,
        x=heatmap_method.columns.tolist(),
        y=heatmap_method.index.tolist(),
        color_continuous_scale="RdYlGn_r",
        text_auto=".0f",
        aspect="auto",
    )
    fig_hm2.update_layout(
        xaxis_title="PSP", yaxis_title="Payment Method", height=350,
        coloraxis_colorbar_title="P95 (ms)",
    )
    st.plotly_chart(fig_hm2, use_container_width=True)

    # Ranked table of worst combinations
    st.subheader("Top 10 Worst-Performing Combinations")
    worst_10 = compute_worst_combinations(df, top_n=10)
    if not worst_10.empty:
        display_worst = worst_10.copy()
        display_worst["p95"] = display_worst["p95"].map("{:,.0f}ms".format)
        display_worst["p50"] = display_worst["p50"].map("{:,.0f}ms".format)
        display_worst["approval_rate"] = display_worst["approval_rate"].map("{:.1%}".format)
        display_worst["timeout_rate"] = display_worst["timeout_rate"].map("{:.1%}".format)
        st.dataframe(display_worst, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Page 5: Routing Recommendations
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Routing Recommendations":
    st.title("Smart Routing Recommendations")
    st.markdown(
        "Optimal PSP selection per route based on latency (40%), "
        "approval rate (40%), and reliability (20%)."
    )

    recommendations = compute_routing_recommendations(df)

    if recommendations.empty:
        st.warning("No routing recommendations available.")
    else:
        # Impact summary
        total_impact = recommendations[recommendations["impact_p95_ms"] > 0]
        if not total_impact.empty:
            total_txns_impacted = total_impact["volume"].sum()
            avg_improvement = total_impact["impact_p95_ms"].mean()
            st.success(
                f"**Potential Impact:** Optimizing {total_txns_impacted:,} transactions "
                f"could reduce average P95 by {avg_improvement/1000:.1f}s"
            )

        # Recommendations table
        st.subheader("Routing Table")
        display_rec = recommendations[[
            "country", "payment_method", "volume",
            "recommended_psp", "rec_score",
            "current_best_psp", "current_score",
            "expected_p95", "expected_approval",
            "impact_p95_ms", "impact_description",
        ]].copy()
        display_rec["expected_p95"] = display_rec["expected_p95"].map("{:,.0f}ms".format)
        display_rec["expected_approval"] = display_rec["expected_approval"].map("{:.1%}".format)
        display_rec["impact_p95_ms"] = display_rec["impact_p95_ms"].map("{:+,.0f}ms".format)
        st.dataframe(display_rec, use_container_width=True, hide_index=True)

        # Health score breakdown
        st.subheader("Health Score Breakdown by Route")
        breakdown = compute_score_breakdown(df)

        if not breakdown.empty:
            fig_breakdown = px.bar(
                breakdown,
                x="psp",
                y=["latency_component", "approval_component", "reliability_component"],
                color_discrete_map={
                    "latency_component": "#636EFA",
                    "approval_component": "#00CC96",
                    "reliability_component": "#FFA726",
                },
                facet_col="country",
                facet_row="payment_method",
                barmode="stack",
                height=600,
            )
            fig_breakdown.update_layout(
                yaxis_title="Score",
                legend_title="Component",
            )
            fig_breakdown.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(fig_breakdown, use_container_width=True)

        # Per-route details
        st.subheader("Route Impact Details")
        for _, row in recommendations.iterrows():
            if row["impact_description"]:
                st.markdown(f"- **{row['country']}/{row['payment_method']}**: {row['impact_description']}")


# ═══════════════════════════════════════════════════════════════════════════════
# Page 6: Alerts
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Alerts":
    st.title("Alert Dashboard")

    alerts = detect_alerts(df)
    summary = get_alert_summary(alerts)

    # Summary cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Alerts", summary["total"])
    col2.metric("Critical", summary["critical"])
    col3.metric("Warning", summary["warning"])

    st.markdown("---")

    # Active alerts panel
    st.subheader("Active Alerts")

    if not alerts:
        st.success("No active alerts.")
    else:
        # Separate critical and warning
        critical_alerts = [a for a in alerts if a["severity"] == "critical"]
        warning_alerts = [a for a in alerts if a["severity"] == "warning"]

        if critical_alerts:
            st.markdown("#### Critical")
            for a in critical_alerts:
                ts_str = a["timestamp"].strftime("%Y-%m-%d %H:%M") if a["timestamp"] else "Overall"
                st.error(f"🔴 **{a['type']}** | {a['psp']} | {ts_str}\n\n{a['message']}")

        if warning_alerts:
            st.markdown("#### Warning")
            for a in warning_alerts:
                ts_str = a["timestamp"].strftime("%Y-%m-%d %H:%M") if a["timestamp"] else "Overall"
                st.warning(f"🟡 **{a['type']}** | {a['psp']} | {ts_str}\n\n{a['message']}")

    st.markdown("---")

    # Alert history timeline
    st.subheader("Alert Timeline")
    timed_alerts = [a for a in alerts if a["timestamp"] is not None]
    if timed_alerts:
        alert_df = pd.DataFrame(timed_alerts)
        alert_df["hour"] = alert_df["timestamp"]

        fig_timeline = px.scatter(
            alert_df,
            x="hour",
            y="psp",
            color="severity",
            size_max=12,
            color_discrete_map=SEVERITY_COLORS,
            hover_data=["type", "message", "current_value"],
        )
        fig_timeline.update_traces(marker_size=10)
        fig_timeline.update_layout(
            xaxis_title="Time", yaxis_title="PSP", height=350,
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No time-based alerts to display.")

    # Alert configuration display
    st.subheader("Alert Configuration")
    config_data = {
        "Rule": ["P95 Spike", "P95 Spike", "Timeout Rate", "Timeout Rate",
                  "Approval Rate", "Approval Rate", "Absolute P95", "Absolute P95"],
        "Severity": ["warning", "critical", "warning", "critical",
                      "warning", "critical", "warning", "critical"],
        "Condition": [
            "Hourly P95 > 2x baseline", "Hourly P95 > 3x baseline",
            "Timeout rate > 5%", "Timeout rate > 10%",
            "Approval rate < 70%", "Approval rate < 60%",
            "P95 > 5,000ms", "P95 > 8,000ms",
        ],
    }
    st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)

    # Alerts by type
    st.subheader("Alerts by Type")
    if summary["by_type"]:
        type_df = pd.DataFrame([
            {"type": k, "count": v} for k, v in summary["by_type"].items()
        ])
        fig_type = px.bar(
            type_df, x="type", y="count",
            color_discrete_sequence=["#636EFA"],
        )
        fig_type.update_layout(
            xaxis_title="Alert Type", yaxis_title="Count", height=300,
        )
        st.plotly_chart(fig_type, use_container_width=True)
