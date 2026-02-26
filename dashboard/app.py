"""
Volta Payment Monitor — Streamlit Dashboard

Multi-page interactive dashboard for monitoring payment pipeline health
across PSPs, countries, and payment methods.

Pages:
  1. Overview — KPIs and time-series trends
  2. PSP Performance — Side-by-side PSP comparison
  3. Drill-Down Analysis — Filtered deep-dive
  4. Bottleneck Finder — Heatmaps and worst combinations
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
)


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
