"""
Stage Segmentation Script
This script processes fan-model conversation logs,
assigns them to engagement stages based on purchase timing,
and saves the segmented DataFrame for downstream analysis.
"""

import pandas as pd
from pathlib import Path


def load_data():
    """
    Load raw conversation data from pickle file, clean column names, convert timestamps,
    and create fan_model_id for unique identification.

    Returns:
        pd.DataFrame: Cleaned conversation DataFrame.
    """
    df = pd.read_pickle("data/HOMEWORK_LOGS.pkl")

    df = df.rename(columns={
        'model_name': 'model_id',
        'datetime': 'timestamp',
        'purchased': 'purchase'
    })

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['fan_model_id'] = df['fan_id'].astype(str) + "_" + df['model_id'].astype(str)
    df = df.sort_values(by=['fan_model_id', 'timestamp'])
    return df


def assign_stages(df):
    """
    Assign each message to a stage (stage_1, stage_2, stage_3) based on purchase events.

    Args:
        df (pd.DataFrame): Cleaned conversation DataFrame.

    Returns:
        pd.DataFrame: DataFrame with added 'stage' column indicating engagement stage.
    """
    all_stage_rows = []

    for fan_model_id, group in df.groupby("fan_model_id"):
        group = group.sort_values(by="timestamp")
        purchase_times = group[group["purchase"] == True]["timestamp"].tolist()

        if not purchase_times:
            group["stage"] = "stage_1"
            all_stage_rows.append(group)
            continue

        first_purchase = purchase_times[0]
        last_purchase = purchase_times[-1]

        stage1 = group[group["timestamp"] <= first_purchase].copy()
        stage1["stage"] = "stage_1"

        if first_purchase != last_purchase:
            stage2 = group[(group["timestamp"] > first_purchase) & (group["timestamp"] <= last_purchase)].copy()
            stage2["stage"] = "stage_2"
        else:
            stage2 = pd.DataFrame(columns=group.columns)

        stage3 = group[group["timestamp"] > last_purchase].copy()
        stage3["stage"] = "stage_3"

        all_parts = [stage1]
        if not stage2.empty:
            all_parts.append(stage2)
        if not stage3.empty:
            all_parts.append(stage3)

        all_stage_rows.append(pd.concat(all_parts))

    return pd.concat(all_stage_rows)


def main():
    """
    Main execution: load data, assign stages, and save staged DataFrame.
    """
    df = load_data()
    staged_df = assign_stages(df)

    Path("outputs").mkdir(exist_ok=True)
    staged_df.to_pickle("outputs/staged_conversations.pkl")
    print("✅ Saved: staged_conversations.pkl")


if __name__ == "__main__":
    main()
