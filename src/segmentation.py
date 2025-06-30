"""
Conversation Feature Engineering Script
This script processes fan-model conversation logs,
assigns conversation IDs, computes various features per conversation,
and saves the output as a CSV. Intended for behavioral and purchase analysis in fan analytics pipelines.
"""

import pandas as pd
from datetime import timedelta
from pathlib import Path


def load_data(file_path):
    """
    Load data from pickle file.
    """
    return pd.read_pickle(file_path)


def preprocess(df):
    """
    Clean and prepare dataframe by renaming columns, parsing timestamps, and creating fan_model_id.
    """
    df = df.copy()

    df = df.rename(columns={
        'model_name': 'model_id',
        'datetime': 'timestamp',
        'purchased': 'purchase'
    })

    df['purchase'] = df['purchase'].astype(str).str.upper().map({'TRUE': True, 'FALSE': False, '1': True, '0': False}).fillna(False)

    if 'is_system' in df.columns:
        df = df[~df['is_system']]

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['fan_model_id'] = df['fan_id'].astype(str) + "_" + df['model_id'].astype(str)

    df = df.sort_values(by=['fan_model_id', 'timestamp'])

    return df


def assign_conversations(df):
    """
    Assign conversation IDs based on time gaps and model switches.
    """
    df = df.copy()
    df['new_convo'] = False

    df['prev_model'] = df.groupby('fan_model_id')['model_id'].shift(1)
    df['prev_time'] = df.groupby('fan_model_id')['timestamp'].shift(1)
    
    time_diff = df['timestamp'] - df['prev_time']
    model_switch = df['model_id'] != df['prev_model']
    time_gap = time_diff > timedelta(hours=4)

    df.loc[model_switch | time_gap, 'new_convo'] = True
    df['convo_id'] = df.groupby('fan_model_id')['new_convo'].cumsum()
    df['conversation_id'] = df['fan_model_id'] + "_C" + df['convo_id'].astype(str)
    return df


def compute_features(df):
    """
    Compute conversation-level features such as purchase stats, durations, and activity indicators.
    """
    features = []
    grouped = df.groupby('conversation_id')
    for convo_id, group in grouped:
        revenue = group['purchase'].sum()
        purchases = group['purchase'].sum()
        purchase_msgs = group[group['purchase']]
        purchase_rate = purchases / len(group)
        time_to_first_purchase = (
            (purchase_msgs['timestamp'].iloc[0] - group['timestamp'].iloc[0]).total_seconds() / 3600
            if not purchase_msgs.empty else None
        )
        duration = (group['timestamp'].iloc[-1] - group['timestamp'].iloc[0]).total_seconds() / 3600
        days_between = (
            (purchase_msgs['timestamp'].iloc[-1] - purchase_msgs['timestamp'].iloc[0]).days
            if len(purchase_msgs) > 1 else 0
        )
        last_msg_time = group['timestamp'].max()
        active = (df['timestamp'].max() - last_msg_time).days < 2
        msgs_before_first_purchase = (
            purchase_msgs.index[0] - group.index[0] if not purchase_msgs.empty else len(group)
        )

        features.append({
            'conversation_id': convo_id,
            'fan_model_id': group['fan_model_id'].iloc[0],
            'message_count': len(group),
            'revenue': revenue,
            'purchase_count': purchases,
            'purchase_rate': purchase_rate,
            'duration_hours': duration,
            'messages_before_first_purchase': msgs_before_first_purchase,
            'time_to_first_purchase_hrs': time_to_first_purchase,
            'days_between_first_last_purchase': days_between,
            'active': active,
            'days_since_last_message': (df['timestamp'].max() - last_msg_time).days
        })

    return pd.DataFrame(features)


def main():
    """
    Main pipeline to load, preprocess, assign conversations, compute features, and save outputs.
    """
    raw = load_data("data/HOMEWORK_LOGS.pkl")
    clean = preprocess(raw)
    convos = assign_conversations(clean)
    features = compute_features(convos)
    Path("outputs").mkdir(exist_ok=True)
    features.to_csv("outputs/conversation_features.csv", index=False)
    print("✅ Saved: conversation_features.csv")


if __name__ == "__main__":
    main()
