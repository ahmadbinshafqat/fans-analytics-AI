"""
cluster_analysis.py

Performs clustering of segmented fan-model conversations using:
- Method A: Pure conversation-based embeddings
- Method B: Hybrid embeddings (conversation + fan profile)

Embeddings are generated via VoyageAI, clusters via KMeans, and visualized with UMAP and Plotly.
Outputs are saved as pickles and visualizations.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
import pickle
import voyageai

vo = voyageai.Client()
N_CLUSTERS = 5


def load_data():
    """
    Load staged conversations and fan profiles from outputs folder.
    """
    staged = pd.read_pickle("outputs/staged_conversations.pkl")
    profiles = pd.read_csv("outputs/fan_profiles.csv")
    return staged, profiles


def get_stage_texts(df):
    """
    Concatenate fan messages for each fan-model stage to form long texts for embedding.
    """
    grouped = df.groupby(["fan_model_id", "stage"])
    return grouped['fan_message'].apply(lambda msgs: " ".join(msgs.dropna().astype(str))).reset_index(name='text')


def generate_text_embeddings(texts):
    """
    Generate embeddings for a list of texts using VoyageAI in batches.
    """
    embeddings = []
    batch_size = 20

    clean_texts = [t for t in texts if t.strip()]

    for i in range(0, len(clean_texts), batch_size):
        batch_texts = clean_texts[i:i + batch_size]
        if not batch_texts:
            continue

        result = vo.embed(batch_texts, model="voyage-3.5")
        batch_embeddings = result.embeddings
        embeddings.extend(batch_embeddings)

    return np.array(embeddings)


def combine_with_profiles(embeddings, texts_df, profiles_df):
    """
    Combine text embeddings with fan profile features to create hybrid embeddings.
    """
    filtered_texts_df = texts_df[texts_df['text'].str.strip().astype(bool)].reset_index(drop=True)

    if len(filtered_texts_df) != len(embeddings):
        filtered_texts_df = filtered_texts_df.iloc[:len(embeddings)].reset_index(drop=True)

    merged = filtered_texts_df.merge(profiles_df, on="fan_model_id", how="left")

    drop_cols = ['fan_model_id', 'stage', 'text']
    profile_feats = merged.drop(columns=drop_cols, errors='ignore')
    profile_feats = pd.get_dummies(profile_feats)
    profile_feats = (profile_feats - profile_feats.mean()) / (profile_feats.std() + 1e-6)

    hybrid = np.hstack([embeddings, profile_feats.fillna(0).values])
    return hybrid, merged


def cluster_and_plot(embeddings, df, method="A"):
    """
    Cluster embeddings using KMeans, reduce with UMAP, and create 2D and 3D visualizations.
    """
    if len(df) != len(embeddings):
        df = df.iloc[:len(embeddings)].reset_index(drop=True)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    df["cluster"] = kmeans.fit_predict(embeddings)

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    df["x"] = reduced[:, 0]
    df["y"] = reduced[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="x", y="y", hue="cluster", palette="Set2")
    plt.title(f"UMAP Clusters - Method {method}")
    plt.savefig(f"outputs/cluster_umap_method_{method}.png")
    plt.close()

    fig3d = px.scatter_3d(
        df.assign(z=reduced[:, 0]),
        x="x", y="y", z="z" if "z" in df else "x",
        color="cluster",
        hover_data=["fan_model_id", "stage"]
    )
    fig3d.write_html(f"outputs/cluster_3d_method_{method}.html")

    df.to_pickle(f"outputs/clustered_method_{method}.pkl")

    return df, kmeans


def save_embeddings(filename, embeddings, meta_df):
    """
    Save embeddings and metadata as a pickle file.
    """
    with open(f"outputs/{filename}", "wb") as f:
        pickle.dump((embeddings, meta_df), f)


def main():
    """
    Main pipeline to process data, generate embeddings, cluster, and save outputs.
    """
    Path("outputs").mkdir(exist_ok=True)
    staged_df, profiles_df = load_data()

    texts_df = get_stage_texts(staged_df)

    print("🔹 Method A: Generating embeddings without profile info...")
    methodA_embeddings = generate_text_embeddings(texts_df["text"].tolist())
    save_embeddings("embeddings_without_profile.pkl", methodA_embeddings, texts_df)

    print("📊 Clustering Method A...")
    dfA, _ = cluster_and_plot(methodA_embeddings, texts_df.copy(), method="A")

    print("🔹 Method B: Combining with fan profiles...")
    methodB_embeddings, hybrid_df = combine_with_profiles(methodA_embeddings, texts_df, profiles_df)
    save_embeddings("embeddings_with_profile.pkl", methodB_embeddings, hybrid_df)

    print("📊 Clustering Method B...")
    dfB, _ = cluster_and_plot(methodB_embeddings, hybrid_df.copy(), method="B")

    print("✅ All clustering done. Check the outputs folder.")


if __name__ == "__main__":
    main()
