import pandas as pd 
import numpy as np 
from datetime import timedelta
import altair as alt
import contextlib
import sys
import os
alt.data_transformers.enable("vegafusion")

def optimize_battery_schedule_df_flexible_end(
    df_day,
    start_soc,
    battery_capacity=95,
    min_soc=0.2 * 95,
    max_rate=11.5,
    delta_step=11.5
):
    df_day = df_day.reset_index(drop=True)
    prices = df_day["MW"].values / 1000.0  # Convert $/MWh → $/kWh
    T = len(prices)

    # Determine highest reachable SoC
    max_possible_charge = max_rate * T
    raw_target = start_soc + max_possible_charge
    end_soc = np.floor(raw_target / max_rate) * max_rate
    end_soc = min(end_soc, battery_capacity)
    end_soc = round(end_soc, 2)

    dp = [{} for _ in range(T + 1)]
    policy = [{} for _ in range(T + 1)]
    start_soc = round(start_soc, 2)
    dp[0][start_soc] = 0.0

    for t in range(T):
        for soc in dp[t]:
            for delta in [-max_rate, 0.0, max_rate]:  # Full charge, idle, full discharge
                next_soc = round(soc + delta, 2)
                if min_soc <= next_soc <= battery_capacity:
                    cost = dp[t][soc] + delta * prices[t]
                    if next_soc not in dp[t + 1] or cost < dp[t + 1][next_soc]:
                        dp[t + 1][next_soc] = cost
                        policy[t + 1][next_soc] = (soc, delta)

    if end_soc not in dp[T]:
        # Fallback: choose best reachable SoC
        candidates = [s for s in dp[T].keys() if s <= battery_capacity]
        if not candidates:
            raise ValueError("No feasible SoE found at all.")
        end_soc = max(candidates)

    # Backtrack for actions
    soc = end_soc
    records = []
    for t in reversed(range(T)):
        prev_soc, delta = policy[t + 1][soc]
        profit = -delta * prices[t]
        row = df_day.iloc[t]
        records.append({
            "IntervalStart": row["INTERVALSTARTTIME"],
            "Hour_Label": row["Hour_Label"],
            "Action (kWh)": delta,
            "SoE Before (kWh)": prev_soc,
            "SoE After (kWh)": soc,
            "Price ($/kWh)": prices[t],
            "Profit ($)": profit
        })
        soc = prev_soc

    records.reverse()
    df_result = pd.DataFrame(records)
    df_result["Cumulative Profit ($)"] = df_result["Profit ($)"].cumsum()
    return df_result, end_soc

def run_weekly_optimization_with_daily_usage(df, initial_soc=66.5, daily_miles=30.1, miles_per_kwh=3.83):
    results = []
    soc = initial_soc
    energy_needed = round(daily_miles / miles_per_kwh, 2)  
    total_profit = 0.0  
    
    df = df.copy()
    df["TradingWindowStart"] = df["INTERVALSTARTTIME"].apply(
        lambda x: x if x.hour >= 18 else x - timedelta(days=1)
    ).dt.normalize()

    for window_start in sorted(df["TradingWindowStart"].unique()):
        window_start = pd.to_datetime(window_start)
        mask = (df["INTERVALSTARTTIME"] >= window_start + timedelta(hours=18)) & \
               (df["INTERVALSTARTTIME"] < window_start + timedelta(days=1, hours=8))
        df_window = df[mask]

        if df_window.shape[0] < 14:
            print(f"Skipping {window_start.date()}: Incomplete data ({df_window.shape[0]} hours)")
            continue

        try:
            # Run per-night optimization
            df_result, end_soc = optimize_battery_schedule_df_flexible_end(df_window, start_soc=soc)
            df_result["TradingWindowStart"] = window_start

            # Accumulate profit across days
            df_result["Cumulative Profit ($)"] = df_result["Profit ($)"].cumsum() + total_profit
            total_profit = df_result["Cumulative Profit ($)"].iloc[-1]

            results.append(df_result)

            # Update SoC for next night, subtract daily driving usage
            soc = round(end_soc - energy_needed, 2)
            soc = max(soc, 0.2 * 95)

        except ValueError as e:
            print(f"Skipping {window_start.date()}: {e}")

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def simulate_all_nodes(full_df, initial_soc=66.5):
    profit_results = []

    # Ensure datetime columns are parsed
    full_df["INTERVALSTARTTIME"] = pd.to_datetime(full_df["INTERVALSTARTTIME"])
    full_df["INTERVALENDTIME"] = pd.to_datetime(full_df["INTERVALENDTIME"])

    # Group by NODE
    grouped = full_df.groupby("NODE")

    for node, df_node in grouped:
        try:
            df_result = run_weekly_optimization_with_daily_usage(df_node, initial_soc=initial_soc)
            total_profit = df_result["Profit ($)"].sum()
            profit_results.append({
                "NODE": node,
                "Total Profit ($)": total_profit
            })
        except Exception as e:
            print(f"Skipping node {node}: {e}")

    return pd.DataFrame(profit_results).sort_values(by="Total Profit ($)", ascending=False).reset_index(drop=True)

def prepare_visualization(monthly_result, df_prices):
    # Prepare full hourly DataFrame
    df_prices = df_prices.copy()
    df_prices["Price ($/kWh)"] = df_prices["MW"] / 1000.0  # Convert MWh → kWh
    df_prices = df_prices[["INTERVALSTARTTIME", "Price ($/kWh)"]]

    # Add SoC and cumulative profit from trading results
    df_soc = monthly_result[["IntervalStart", "SoE After (kWh)", "Cumulative Profit ($)"]].copy()
    df_soc.columns = ["INTERVALSTARTTIME", "SoE (kWh)", "Cumulative Profit ($)"]

    # Merge trading data with full hourly price data
    df_all = pd.merge(df_prices, df_soc, on="INTERVALSTARTTIME", how="left")

    # Fill non-trading hours
    df_all = df_all.sort_values("INTERVALSTARTTIME").reset_index(drop=True)

    # Fill cumulative profit forward
    df_all["Cumulative Profit ($)"] = df_all["Cumulative Profit ($)"].ffill()

    # Fill SoC linearly for 08:00–18:00 each day
    df_all["hour"] = df_all["INTERVALSTARTTIME"].dt.hour
    df_all["date"] = df_all["INTERVALSTARTTIME"].dt.normalize()

    filled_soc = []
    for _, group in df_all.groupby("date"):
        trading_soc = group["SoE (kWh)"].copy()
        soc_before = trading_soc.ffill().bfill().iloc[7]  # SoC at 08:00
        soc_after = soc_before - (30.1 / 3.83)  # Linear drop for driving

        drop_hours = (group["hour"] >= 8) & (group["hour"] < 18)
        linear_soc = np.linspace(soc_before, soc_after, drop_hours.sum())

        temp = trading_soc.copy()
        temp[drop_hours] = linear_soc
        filled_soc.append(temp)

    df_all["SoE (kWh)"] = pd.concat(filled_soc).sort_index()

    return df_all

def plot_altair_separate(df_all):
    base = alt.Chart(df_all).encode(
        x=alt.X("INTERVALSTARTTIME:T", axis=alt.Axis(title="Time"))
    )

    price_line = base.mark_line(color= '#3A7DC1', interpolate='step-after').encode(
        y=alt.Y("Price ($/kWh):Q", axis=alt.Axis(title="Price ($/kWh)"))
    ).properties(height=150)

    soc_line = base.mark_line(color="#27ae60").encode(
        y=alt.Y("SoE (kWh):Q", axis=alt.Axis(title="State of Energy (kWh)"))
    ).properties(height=150)

    profit_line = base.mark_line(color="#D4904E").encode(
        y=alt.Y("Cumulative Profit ($):Q", axis=alt.Axis(title="Cumulative Profit ($)"))
    ).properties(height=150)

    # Combine vertically and add zoom/pan
    chart = alt.vconcat(price_line, soc_line, profit_line).resolve_scale(
        y="independent"
    ).interactive()

    return chart