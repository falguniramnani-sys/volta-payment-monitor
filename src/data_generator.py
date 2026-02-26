"""
Synthetic transaction data generator for Volta Payment Monitor.

Generates 15,000+ transactions over a 48-hour window across Brazil, Mexico,
and Colombia with realistic PSP latency profiles, peak-hour degradation,
and payment-method-specific latency modifiers.

Output: data/transactions.csv
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Constants ─────────────────────────────────────────────────────────────────

N_TRANSACTIONS = 15_000
WINDOW_HOURS = 48
SEED = 42

PSP_PROFILES = {
    "PSP_Alpha": {"median": 800,  "sigma": 0.4, "approval": 0.65, "timeout": 0.005},
    "PSP_Beta":  {"median": 3500, "sigma": 0.5, "approval": 0.90, "timeout": 0.010},
    "PSP_Gamma": {"median": None, "sigma": None, "approval": 0.78, "timeout": 0.030},  # bimodal
    "PSP_Delta": {"median": 2000, "sigma": 0.5, "approval": 0.75, "timeout": 0.080},
}

COUNTRY_DIST = {"BR": 0.40, "MX": 0.35, "CO": 0.25}

METHOD_DIST = {
    "BR": {"CARD": 0.50, "PIX": 0.50},
    "MX": {"CARD": 0.60, "OXXO": 0.40},
    "CO": {"CARD": 0.40, "PSE": 0.60},
}

CARD_BRANDS = {
    "BR": {"Visa": 0.55, "Mastercard": 0.45},
    "MX": {"Visa": 0.50, "Mastercard": 0.50},
    "CO": {"Visa": 0.60, "Mastercard": 0.40},
}

METHOD_MULTIPLIERS = {"PIX": 0.7, "OXXO": 0.3, "PSE": 2.5, "CARD": 1.0}

PEAK_HOURS = {12, 13, 19, 20}
PEAK_PSPS = {"PSP_Gamma", "PSP_Delta"}
PEAK_MULTIPLIER = 1.6

TIMEOUT_LATENCY_MIN = 15_000
TIMEOUT_LATENCY_MAX = 20_000

PSP_WEIGHTS = [0.28, 0.28, 0.22, 0.22]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample_psp_latency(rng: np.random.Generator, psp: str, n: int) -> np.ndarray:
    """Sample base latency (ms) for n transactions from a given PSP."""
    p = PSP_PROFILES[psp]
    if psp == "PSP_Gamma":
        # Bimodal: ~1500ms median but with high variance spikes
        slow_mask = rng.random(n) < 0.35
        fast = rng.lognormal(mean=np.log(1000), sigma=0.4, size=n)
        slow = rng.lognormal(mean=np.log(6000), sigma=0.5, size=n)
        return np.where(slow_mask, slow, fast)
    return rng.lognormal(mean=np.log(p["median"]), sigma=p["sigma"], size=n)


def _assign_status(rng: np.random.Generator, psp: str, latencies: np.ndarray) -> np.ndarray:
    """
    Assign 'approved', 'declined', or 'timeout' status.

    - Soft declines correlate with high latency
    - Hard declines are fast
    - Timeouts cluster near 15-20s threshold
    """
    p = PSP_PROFILES[psp]
    n = len(latencies)
    rand = rng.random(n)
    status = np.full(n, "approved", dtype=object)

    # Timeouts
    timeout_mask = rand < p["timeout"]
    status[timeout_mask] = "timeout"

    # Declines: base rate adjusted by latency (high latency → more soft declines)
    decline_base = 1.0 - p["approval"] - p["timeout"]
    latency_factor = np.clip(latencies / np.median(latencies), 0.5, 2.0)
    decline_probs = decline_base * latency_factor
    decline_mask = (~timeout_mask) & (rng.random(n) < decline_probs)
    status[decline_mask] = "declined"

    return status


# ── Main generator ────────────────────────────────────────────────────────────

def generate_transactions() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    psps = rng.choice(list(PSP_PROFILES.keys()), size=N_TRANSACTIONS, p=PSP_WEIGHTS)
    countries = rng.choice(
        list(COUNTRY_DIST.keys()), size=N_TRANSACTIONS, p=list(COUNTRY_DIST.values()),
    )

    # Payment methods per country
    methods = np.empty(N_TRANSACTIONS, dtype=object)
    card_brands = np.full(N_TRANSACTIONS, "", dtype=object)
    for country, dist in METHOD_DIST.items():
        mask = countries == country
        n = mask.sum()
        if n > 0:
            methods[mask] = rng.choice(list(dist.keys()), size=n, p=list(dist.values()))
            # Assign card brands
            card_mask = mask & (methods == "CARD")
            n_cards = card_mask.sum()
            if n_cards > 0:
                brands = CARD_BRANDS[country]
                card_brands[card_mask] = rng.choice(
                    list(brands.keys()), size=n_cards, p=list(brands.values())
                )

    # Timestamps (uniform over 48h, sorted)
    base_ts = datetime(2024, 6, 1, 0, 0, 0)
    offsets_sec = rng.uniform(0, WINDOW_HOURS * 3600, size=N_TRANSACTIONS)
    offsets_sec.sort()
    timestamps = [base_ts + timedelta(seconds=s) for s in offsets_sec]
    hours = np.array([ts.hour for ts in timestamps])

    # Sample latencies per PSP
    latencies = np.zeros(N_TRANSACTIONS)
    statuses = np.empty(N_TRANSACTIONS, dtype=object)
    for psp in PSP_PROFILES:
        mask = psps == psp
        n = mask.sum()
        if n == 0:
            continue

        base_lat = _sample_psp_latency(rng, psp, n)
        psp_hours = hours[mask]
        psp_methods = methods[mask]

        # Payment method multiplier
        method_mult = np.array([METHOD_MULTIPLIERS.get(m, 1.0) for m in psp_methods])

        # Peak-hour multiplier (only for PSP_Gamma and PSP_Delta)
        if psp in PEAK_PSPS:
            peak_mult = np.where(np.isin(psp_hours, list(PEAK_HOURS)), PEAK_MULTIPLIER, 1.0)
        else:
            peak_mult = 1.0

        adjusted = base_lat * method_mult * peak_mult
        psp_status = _assign_status(rng, psp, adjusted)

        # Override latency for timeouts (cluster near 15-20s)
        timeout_mask = psp_status == "timeout"
        if timeout_mask.any():
            adjusted[timeout_mask] = rng.uniform(
                TIMEOUT_LATENCY_MIN, TIMEOUT_LATENCY_MAX, size=timeout_mask.sum()
            )

        latencies[mask] = adjusted
        statuses[mask] = psp_status

    # Build DataFrame
    df = pd.DataFrame({
        "id": [f"txn_{i:06d}" for i in range(N_TRANSACTIONS)],
        "timestamp": timestamps,
        "psp": psps,
        "country": countries,
        "payment_method": methods,
        "card_brand": card_brands,
        "latency_ms": np.round(latencies, 1),
        "status": statuses,
    })

    return df


def main():
    os.makedirs("data", exist_ok=True)
    print("Generating 15,000 synthetic transactions...")
    df = generate_transactions()
    out_path = "data/transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows -> {out_path}")
    print(f"\nPSP distribution:\n{df['psp'].value_counts().to_string()}")
    print(f"\nStatus distribution:\n{df['status'].value_counts().to_string()}")
    print(f"\nCountry distribution:\n{df['country'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
