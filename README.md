# Volta Payment Monitor

End-to-end monitoring system for Volta Commerce's payment pipeline, processing 500K+ daily transactions through Yuno's orchestration layer across Brazil, Mexico, and Colombia.

## Architecture

```
volta-payment-monitor/
├── src/
│   ├── data_generator.py   # Synthetic data: 15K transactions, 4 PSPs, 3 countries
│   ├── pipeline.py         # Metrics engine: P50/P95/P99, heatmaps, anomaly detection
│   ├── validation.py       # CSV schema validation, sanitization, fingerprinting
│   ├── alerting.py         # Alert rules: latency spikes, timeout/approval thresholds
│   └── routing.py          # Smart routing: health-score-based PSP recommendations
├── dashboard/
│   └── app.py              # Streamlit dashboard with 6 interactive pages
├── data/
│   ├── transactions.csv    # Generated dataset (gitignored)
│   └── sample_upload.csv   # 100-row sample for testing upload feature
└── docs/
    └── bottleneck_analysis.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python -m src.data_generator

# Launch dashboard
streamlit run dashboard/app.py
```

## PSP Profiles

| PSP | Personality | P50 | Approval | Timeout |
|-----|------------|-----|----------|---------|
| PSP_Alpha | Fast, low approval | ~800ms | ~65% | <1% |
| PSP_Beta | Slow, reliable | ~3,500ms | ~90% | ~1% |
| PSP_Gamma | Bimodal, peak-hour spikes | ~1,500ms | ~78% | ~3% |
| PSP_Delta | Moderate, high timeouts | ~2,000ms | ~75% | ~8% |

## Payment Methods by Country

- **Brazil**: Cards (Visa, Mastercard), PIX
- **Mexico**: Cards (Visa, Mastercard), OXXO
- **Colombia**: Cards (Visa, Mastercard), PSE

## Dashboard Pages

1. **Overview** — KPI cards, latency time-series, volume charts, country/method breakdowns
2. **PSP Performance** — Side-by-side latency percentiles, approval/timeout bars, box plots
3. **Drill-Down Analysis** — Multi-filter deep-dive with histogram and time-series
4. **Bottleneck Finder** — P95 heatmaps (PSP x Country, PSP x Method), worst combination ranking
5. **Routing Recommendations** — Health-score-based optimal PSP per route with impact estimates
6. **Alerts** — Active alerts with severity badges, timeline visualization, rule configuration

## Data Ingestion

The dashboard supports three ways to load transaction data:

### CSV Upload

Click the **Data Source** expander in the sidebar to upload a CSV file. The upload pipeline validates the schema, cleans the data, and pre-computes metrics before displaying results. A 4-stage processing animation shows progress.

### Expected CSV Schema

| Column | Type | Required |
|--------|------|----------|
| `id` | string | Yes |
| `timestamp` | datetime | Yes |
| `psp` | string | Yes |
| `country` | string | Yes |
| `payment_method` | string | Yes |
| `card_brand` | string | Yes |
| `latency_ms` | float | Yes |
| `status` | string (`approved`, `declined`, `timeout`) | Yes |

Minimum 10 rows required. Extra columns are ignored with a warning.

### Webhook Polling (Optional)

Inside the Data Source expander, expand **Webhook Polling** to configure:
- Enter a URL that returns CSV data
- Set the poll interval (1–60 minutes)
- Enable the toggle to start polling

The dashboard checks on each rerun whether enough time has elapsed and only updates if the data fingerprint has changed.

### Testing Upload

A sample file is included for testing:

```bash
# Upload data/sample_upload.csv through the sidebar
# 100 transactions from 2024-07-15 (different from default dataset)
```

## Metrics Pipeline

The pipeline computes vectorized aggregations using pandas:

- **Latency percentiles**: P50, P95, P99 by PSP, country, payment method, and hourly window
- **Rate metrics**: Approval, timeout, and decline rates across all dimensions
- **Cross-dimensional**: PSP x Country, PSP x Payment Method heatmaps
- **Anomaly detection**: Flags hourly P95 spikes (>2x baseline), timeout >5%, approval <70%
- **Performance**: Processes 15K records in under 2 seconds

## Routing Engine

Health score formula per PSP per route:

```
score = 0.40 * (1 - normalized_p95) + 0.40 * approval_rate + 0.20 * (1 - timeout_rate)
```

Recommends optimal PSP for each (country, payment_method) pair with estimated latency improvement.

## Alerting

Four alert rules with warning/critical severity:

| Rule | Warning | Critical |
|------|---------|----------|
| P95 Spike | >2x baseline | >3x baseline |
| Timeout Rate | >5% | >10% |
| Approval Rate | <70% | <60% |
| Absolute P95 | >5,000ms | >8,000ms |

## Design Choices

- **Synthetic data with seed**: Deterministic generation (seed=42) ensures reproducible analysis
- **Bimodal PSP_Gamma**: Realistic modeling of PSPs with intermittent degradation
- **Peak-hour targeting**: Only PSP_Gamma and PSP_Delta degrade during peak hours, matching real-world patterns where some providers handle load better
- **Timeout clustering**: Timeouts cluster at 15-20s matching typical gateway timeout configurations
- **Latency-correlated declines**: Soft declines occur more frequently with high-latency transactions

## Tech Stack

- Python 3.14
- Streamlit (dashboard framework)
- Plotly (interactive charts)
- Pandas / NumPy (data processing)
