"""
This script processes fan-model conversation logs,
generates fan-level profiles using an LLM (OpenAI),
caches responses to avoid redundant calls,
and saves the profiles as CSV for downstream analytics.
"""

import pandas as pd
import hashlib
import os
import json
import time
from tqdm import tqdm
from pathlib import Path
import re
from openai import OpenAI

MODEL = "gpt-4o"
BATCH_SIZE = 20
CACHE_DIR = "cache/llm_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_hash(text):
    """Generate MD5 hash for caching."""
    return hashlib.md5(text.encode()).hexdigest()


def load_cache(hash_key):
    """Load cached profile if exists."""
    path = os.path.join(CACHE_DIR, hash_key + ".json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_cache(hash_key, response):
    """Save profile to cache."""
    path = os.path.join(CACHE_DIR, hash_key + ".json")
    with open(path, "w") as f:
        json.dump(response, f)


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def normalize_keys(profile_dict):
    """
    Convert keys from LLM (with spaces and capital letters) to consistent snake_case keys.
    """
    key_map = {
        "Age indicators": "age_indicators",
        "Job or career": "job_or_career",
        "Location hints": "location_hints",
        "Relationship status": "relationship_status",
        "Personality traits": "personality_traits",
        "Emotional needs": "emotional_needs",
        "Purchase motivations": "purchase_motivations",
        "Communication style": "communication_style",
        "Life events": "life_events",
    }
    return {key_map.get(k, k): v for k, v in profile_dict.items()}


def call_llm_batch(conversations_list):
    """
    Call LLM to generate profiles for a batch of fans.
    """
    batch_prompt = "You're a fan profiler for a premium chat platform.\n\n"
    batch_prompt += "For each fan conversation, extract:\n- Age indicators\n- Job or career\n- Location hints\n- Relationship status\n- Personality traits\n- Emotional needs\n- Purchase motivations\n- Communication style\n- Life events\n\n"
    batch_prompt += "Return ONLY a JSON array of objects, one object per fan, no commentary, no markdown.\n\n"

    for i, conv in enumerate(conversations_list, start=1):
        batch_prompt += f"Fan #{i} messages:\n{conv}\n\n"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert profiler. Your output must be ONLY a JSON array. No explanation or text."},
            {"role": "user", "content": batch_prompt}
        ]
    )

    content = response.choices[0].message.content

    try:
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            json_str = match.group(0)
            json_str = json_str.strip("`")
            json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
            profiles = json.loads(json_str)
        else:
            print("⚠️ No JSON array found in LLM response. Full content below:")
            print("----- RAW LLM RESPONSE -----")
            print(content)
            print("----------------------------")
            profiles = [{} for _ in conversations_list]
    except Exception as e:
        print(f"⚠️ Failed to parse JSON array: {e}")
        print("----- RAW LLM RESPONSE -----")
        print(content)
        print("----------------------------")
        profiles = [{} for _ in conversations_list]

    return profiles


def profile_fans(convos_df):
    """
    Main profiling function: loops through fans, caches results, calls LLM.
    """
    profiles = []
    fan_groups = list(convos_df.groupby("fan_model_id"))

    all_fan_texts = []
    fan_ids = []

    for fan_id, group in fan_groups:
        all_fan_text = "\n".join(group["fan_message"].dropna().astype(str).tolist())
        all_fan_texts.append(all_fan_text)
        fan_ids.append(fan_id)

    for i in tqdm(range(0, len(all_fan_texts), BATCH_SIZE), desc="Profiling fans"):
        batch_texts = all_fan_texts[i:i + BATCH_SIZE]
        batch_ids = fan_ids[i:i + BATCH_SIZE]

        to_query_texts = []
        to_query_ids = []
        final_batch_profiles = []

        for text, fan_id in zip(batch_texts, batch_ids):
            hash_key = get_hash(text)
            cached = load_cache(hash_key)
            if cached:
                cached = normalize_keys(cached)
                cached['fan_model_id'] = fan_id
                final_batch_profiles.append(cached)
            else:
                to_query_texts.append(text)
                to_query_ids.append(fan_id)

        # Query LLM for uncached
        if to_query_texts:
            try:
                new_profiles = call_llm_batch(to_query_texts)
                for prof, fan_id, text in zip(new_profiles, to_query_ids, to_query_texts):
                    prof = normalize_keys(prof)
                    prof['fan_model_id'] = fan_id
                    hash_key = get_hash(text)
                    save_cache(hash_key, prof)
                    final_batch_profiles.append(prof)
            except Exception as e:
                print(f"❌ Failed profiling batch starting at index {i}: {e}")
                for fan_id in to_query_ids:
                    final_batch_profiles.append({"fan_model_id": fan_id})

        profiles.extend(final_batch_profiles)

        # Pause
        time.sleep(2)

    return pd.DataFrame(profiles)

def main():
    """
    Main pipeline to load conversation data, profile fans, and save results.
    """
    raw = pd.read_pickle("data/HOMEWORK_LOGS.pkl")

    raw = raw.rename(columns={
        'model_name': 'model_id',
        'datetime': 'timestamp',
        'purchased': 'purchase'
    })

    raw['fan_model_id'] = raw['fan_id'].astype(str) + "_" + raw['model_id'].astype(str)

    # Profile the fans
    fan_profiles = profile_fans(raw)

    Path("outputs").mkdir(exist_ok=True)
    fan_profiles.to_csv("outputs/fan_profiles.csv", index=False)
    print("✅ Saved: fan_profiles.csv")

if __name__ == "__main__":
    main()
