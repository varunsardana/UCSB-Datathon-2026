"""
State-Specific Disaster Selector — EDA + Combo Selection

For each US state, identifies the top N most significant disaster types
for time series modeling.

Selection criteria:
  - total_disasters >= MIN_TOTAL  : enough historical events to model
  - months_with_data >= MIN_MONTHS: events spread across time (not a one-off burst)
  - Top N by total_disasters within each state

Output:
  disaster_forecast/selected_combos.csv

Run from repo root:
    python disaster_forecast/state_disaster_selector.py
"""

import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

MIN_TOTAL  = 10   # min total disaster declarations for a combo to qualify
MIN_MONTHS = 12   # min distinct months with ≥1 declaration
TOP_N      = 3    # top N disaster types to keep per state


# ── Selection logic ───────────────────────────────────────────────────────────

def select_top_combos(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    From the full state-incident stats table, return the top N combos per state
    that pass the minimum thresholds.

    Adds a `state_rank` column (1 = most frequent disaster in that state).
    """
    eligible = stats_df[
        (stats_df["total_disasters"] >= MIN_TOTAL) &
        (stats_df["months_with_data"] >= MIN_MONTHS)
    ].copy()

    # Rank within each state by total declaration count
    eligible["state_rank"] = (
        eligible.groupby("state")["total_disasters"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    selected = (
        eligible[eligible["state_rank"] <= TOP_N]
        .sort_values(["state", "state_rank"])
        .reset_index(drop=True)
    )
    return selected


# ── Summary helpers ───────────────────────────────────────────────────────────

def print_state_summary(selected: pd.DataFrame) -> None:
    """Print one line per state showing selected disaster types."""
    print(f"\n{'STATE':<6}  {'SELECTED DISASTER TYPES (total declarations)'}")
    print("-" * 70)
    for state, grp in selected.groupby("state"):
        parts = [
            f"{row['incidentType']} ({int(row['total_disasters'])})"
            for _, row in grp.iterrows()
        ]
        print(f"  {state:<4}  {', '.join(parts)}")


def print_disaster_coverage(selected: pd.DataFrame) -> None:
    """Show how many states each disaster type appears in."""
    coverage = (
        selected.groupby("incidentType")["state"]
        .count()
        .sort_values(ascending=False)
        .rename("states_count")
    )
    print("\nDisaster type coverage across selected combos:")
    for dtype, n in coverage.items():
        bar = "█" * n
        print(f"  {dtype:<22} {bar}  ({n} states)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STATE-SPECIFIC DISASTER SELECTOR")
    print("=" * 70)

    # ── 1. Load pre-built stats (produced by original EDA script) ─────────────
    stats = pd.read_csv("disaster_forecast/fema_state_incident_stats.csv")
    print(f"\n  Loaded {len(stats):,} state-incident combinations")
    print(f"  Covering {stats['state'].nunique()} states")
    print(f"  Date range covered: 2000-01 → 2026-02\n")

    # ── 2. Overall disaster-type breakdown ────────────────────────────────────
    print("Top 10 disaster types by total declarations (all states combined):")
    top_types = (
        stats.groupby("incidentType")["total_disasters"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    for dtype, total in top_types.items():
        pct = total / stats["total_disasters"].sum() * 100
        bar = "█" * int(pct / 2)
        print(f"  {dtype:<22} {int(total):>6,}  {bar}  ({pct:.1f}%)")

    # ── 3. Per-state top-1 disaster overview ──────────────────────────────────
    top1 = (
        stats.sort_values("total_disasters", ascending=False)
        .groupby("state")
        .first()
        .reset_index()[["state", "incidentType", "total_disasters"]]
    )
    print(f"\nMost frequent disaster type per state (sample, sorted by count):")
    print(
        top1.sort_values("total_disasters", ascending=False)
        .head(15)
        .to_string(index=False)
    )

    # ── 4. Apply selection criteria ───────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"Selection criteria:")
    print(f"  • total_disasters >= {MIN_TOTAL}")
    print(f"  • months_with_data >= {MIN_MONTHS}")
    print(f"  • Top {TOP_N} disaster types per state")
    print(f"{'─' * 70}")

    selected = select_top_combos(stats)

    # States that had no qualifying combo at all
    all_states   = set(stats["state"].unique())
    sel_states   = set(selected["state"].unique())
    missing      = sorted(all_states - sel_states)

    print(f"\nResult: {len(selected)} combos across {len(sel_states)} states")
    if missing:
        print(f"  ⚠  {len(missing)} states with no qualifying combo: {', '.join(missing)}")

    print_state_summary(selected)
    print_disaster_coverage(selected)

    # ── 5. Quick sparsity check ───────────────────────────────────────────────
    # Flag combos where avg_monthly is very low — may still produce sparse forecasts
    sparse = selected[selected["avg_monthly"] < 0.5]
    if len(sparse) > 0:
        print(f"\n⚠  {len(sparse)} combos with avg_monthly < 0.5 (sparse — Prophet may show flat forecast):")
        for _, r in sparse.iterrows():
            print(f"     {r['state']} {r['incidentType']}  avg={r['avg_monthly']:.2f}/month")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    selected.to_csv("disaster_forecast/selected_combos.csv", index=False)
    print(f"\n✓ Saved → disaster_forecast/selected_combos.csv  ({len(selected)} rows)")
    print("\nNext step: python disaster_forecast/prophet_forecast.py")

    return selected


if __name__ == "__main__":
    main()
