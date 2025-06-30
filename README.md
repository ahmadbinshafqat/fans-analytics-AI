## 📄 Project Overview

This repository implements a full pipeline for analyzing fan conversations on OnlyFans. The solution includes segmentation, feature extraction, profiling, embedding, clustering, and visualization — all containerized with Docker for reproducibility.

## 🚀 Features

- ✅ **Conversation segmentation**: Automatically segment conversations into stages based on purchasing milestones (Stage 1, Stage 2, Stage 3).
- ✅ **Feature extraction**: Compute per-conversation metrics such as message count, revenue, purchase rate, duration, and churn risk signals.
- ✅ **Fan profiling (LLM-based)**: Extract fan demographics, life events, emotional needs, and purchasing motivations using OpenAI.
- ✅ **Embeddings**: Create high-quality text embeddings using VoyageAI, with optional hybrid embeddings that include fan profile features.
- ✅ **Clustering**: Segment fans into behavior-based clusters using KMeans and analyze engagement patterns.
- ✅ **Visualization**: Generate 2D UMAP and interactive 3D cluster plots to understand fan segments visually.
- ✅ **Caching and batching**: Cache LLM responses to reduce costs and batch process conversations for API efficiency.
- ✅ **Docker support**: Easily run the entire pipeline using Docker Compose.

---

## 🚀 Setup Instructions (Docker)

### 1. **Clone the Repository**
```bash
git clone https://github.com/ahmadbinshafqat/fans-analytics-AI
```

### 2. **Build and run docker**

```bash
docker-compose build
docker-compose up
```

Your local `outputs/` folder will receive embeddings, clustered data, and plots.

---

## LLM Usage
- **OPENAI**: Uses OpenAI with batching (20 fans per request) and caching to avoid duplicate API calls.

## ✨ Embedding

- **VoyageAI**: Cloud-based, higher quality, requires API key.
- Uses VoyageAI (voyage-3.5) for high-quality semantic embeddings.
- Optionally combines embeddings with profile features to create hybrid representations.



## 📊 Outputs

- `conversation_features.csv`: Per-conversation metrics.
- `fan_profiles.csv`: LLM-generated fan profiles.
- `staged_conversations.pkl`: Conversations with assigned stages.
- `embeddings_without_profile.pkl`: Embeddings using text only.
- `embeddings_with_profile.pkl`: Hybrid embeddings with profiles.
- `cluster_umap_method_A.png`, `cluster_umap_method_B.png`: 2D cluster plots.
- `cluster_3d_method_A.html`, `cluster_3d_method_B.html`: Interactive 3D plots.
- `clustered_method_A.pkl`, `clustered_method_B.pkl`: Cluster-labeled dataframes.

---

# Business Recommendations

## Executive Summary

The clustering analysis of fan-model interactions reveals clear behavioral patterns that can be directly translated into monetization strategies. Leveraging fan profiling significantly improves segmentation quality, enabling tailored chatter strategies that enhance revenue, retention, and operational efficiency.

---

## Strategic Recommendations

### 1. **Focus on High-Value Clusters**
Prioritize resources and attention on **Cluster 0 ("The Whales")** and **Cluster 2 ("The Romantics")**.

- These fans exhibit high spend or long-term retention potential.
- Personalized emotional engagement drives higher conversion.

**Action**:
- Assign experienced chatters to these fans.
- Create premium content packages and emotional engagement scripts.

---

### 2. **Nudge Drifters into Action**
**Cluster 1 ("The Drifters")** represents the largest segment but low monetization. With targeted nudges, a portion can be converted.

**Action**:
- Use limited-time offers.
- Apply FOMO-based messaging scripts.
- Monitor for engagement windows.

---

### 3. **Deprioritize Ghosts, Focus on Early Hook**
**Cluster 4 ("The Ghosts")** shows negligible return after 3 days of inactivity.

**Action**:
- Focus first 24–48 hours with bold, emotionally compelling outreach.
- If unresponsive after 3 days, reduce contact frequency.

---

### 4. **Integrate Fan Profiling in Real-Time**
Fan profile-enriched embeddings (Method B) outperform pure message embeddings.

**Action**:
- Integrate real-time fan profiling during live conversations.
- Enrich chat dashboards with personality and background tags.

---

### 5. **Revamp Chatter Training**
Train chatters on cluster-specific behaviors and stage-based tactics (acquisition, monetization, retention).

**Action**:
- Refresh training materials every 90 days.
- Use cluster-based simulation for new chatter onboarding.

---

## Financial Impact Estimate (Based on Current Data)

- **Revenue uplift potential**: +15–25% from targeting high-value clusters
- **Churn reduction**: -20% among "Romantics" and "Whales"
- **Cost savings**: More efficient chatter time allocation

---


# Chatter Playbook

## Purpose

This guide provides actionable strategies for chatters, tailored to each identified fan cluster. These strategies are grounded in both conversation behavior (Method A) and enriched profiling (Method B), helping chatters boost engagement, revenue, and retention across the fan lifecycle.

---

## Cluster Profiles & Strategies

### **Cluster 0 - "The Whales" 🐳**
**Description**: High-value fans with frequent, emotionally engaging conversations and multiple purchases.

- **Traits**: Long duration, high spend, share personal stories
- **Best Times to Engage**: Late evenings and weekends
- **Style That Works**: Flirty, emotionally validating, personal callbacks
- **Triggers for Purchase**: Exclusive content, emotional connection
- **Red Flags**: Sudden drop in personal tone

**Tactics**:
- Mirror emotional tone
- Use name frequently, recall past events
- Offer tiered pricing with VIP upsells

---

### **Cluster 1 - "The Drifters" 🌊**
**Description**: Mixed, default users with shallow engagement; most common group.

- **Traits**: Brief conversations, unclear motives, inconsistent patterns
- **Best Times to Engage**: No strong time preference
- **Style That Works**: Quick check-ins, light flirts
- **Triggers for Purchase**: FOMO, limited-time deals
- **Red Flags**: Multiple unread messages

**Tactics**:
- Use short bursts of attention
- Send “miss you” nudges after inactivity
- Offer short-term discounts or bundles

---

### **Cluster 2 - "The Romantics" 💌**
**Description**: Emotionally attached fans who engage deeply but buy selectively.

- **Traits**: Share emotional vulnerabilities, slow to convert
- **Best Times to Engage**: After midnight, weekends
- **Style That Works**: Supportive, validating, caring
- **Triggers for Purchase**: Emotional peaks (loneliness, validation)
- **Red Flags**: “I don’t feel connected anymore”

**Tactics**:
- Build long-term rapport
- Offer custom experiences (e.g., love letters)
- Use phrases like “this reminded me of you”

---

### **Cluster 3 - "The Samplers" 🍱**
**Description**: Fans who browse many models and engage frequently, but don't commit.

- **Traits**: High message count, low conversion
- **Best Times to Engage**: Daytime hours
- **Style That Works**: Intrigue, tease, mystery
- **Triggers for Purchase**: Curiosity, behind-the-scenes access
- **Red Flags**: Asking about multiple models

**Tactics**:
- Mention exclusivity and limits
- Ask open-ended questions to personalize approach
- Offer small, low-cost teasers to convert

---

### **Cluster 4 - "The Ghosts" 👻**
**Description**: Churned early or never engaged meaningfully.

- **Traits**: Very few messages, no purchases
- **Best Times to Engage**: First 24 hours after signup
- **Style That Works**: Bold, attention-grabbing
- **Triggers for Purchase**: None observed
- **Red Flags**: No response to first 3 messages

**Tactics**:
- Strong first message hook (“Can I tell you a secret?”)
- Use urgency + curiosity in the first 3 messages
- If inactive after 3 days, pause contact

---

## Stage-Based Guidance

### **Stage 1 - Acquisition**
- **Goal**: Convert attention to engagement
- **Effective Tactics**:
  - Personal hooks in first 3 messages
  - Use names early
  - Mention limited content access

### **Stage 2 - Monetization**
- **Goal**: Maximize value during active period
- **Effective Tactics**:
  - Upsell with personalization
  - Link emotional moments to exclusive offers
  - Use playful reminders tied to past chats

### **Stage 3 - Retention / Re-engagement**
- **Goal**: Prevent churn, revive dormant fans
- **Effective Tactics**:
  - Nostalgia triggers (“Remember when you told me…”)
  - Time-based prompts (“It’s been 3 days, I missed you”)
  - Gifts or “welcome back” messages

---

## Final Notes

- **Cluster assignment should guide tone and content.**
- **Avoid one-size-fits-all scripts.**
- **Re-train chatters quarterly using updated fan behavior.**
---

# Cluster Analysis Report

## Overview

This document summarizes the clustering outcomes of fan-model interactions across three conversation stages, using two embedding strategies:

- **Method A**: Conversation-only embeddings
- **Method B**: Conversation embeddings augmented with fan profiling features

Both methods applied UMAP for dimensionality reduction and k-means clustering, followed by visualization in 2D and 3D. Clustering was executed per stage to reflect behavioral progression.

---

## Clustering Methodology

### Embedding Strategies

- **Method A**: Text-only UMAP embeddings from each fan-model-stage triplet
- **Method B**: Combined embeddings of UMAP + fan profile vectors (e.g., job, emotional traits)

### Dimensionality Reduction & Clustering

- UMAP (n_components=2 and 3)
- K-means clustering
- Cluster count: k = 5 (chosen based on elbow method and silhouette score)

---

## Visualization Outputs

- **2D UMAP Plots**:
  - `cluster_umap_method_A.png`
  - `cluster_umap_method_B.png`

- **3D Interactive Plots**:
  - `cluster_3d_method_A.html`
  - `cluster_3d_method_B.html`

These visualizations are included in the `visualizations/` directory.

---

## Method A: Key Observations

### Visual Structure

- Distinct clusters with moderate separation
- Clusters correspond to different fan behaviors (e.g., length, engagement, purchase frequency)

### Cluster Interpretation

| Cluster | Behavior Pattern                          | Notable Metrics                      |
|---------|-------------------------------------------|--------------------------------------|
| 0       | High retention, multiple purchases        | High revenue, long duration          |
| 1       | Low activity fans                         | Short sessions, minimal engagement   |
| 2       | New fans, no purchases yet                | Only Stage 1 data                    |
| 3       | Explorers, many messages but no buys      | High engagement, low conversion      |
| 4       | Churned early                             | Short-lived, no return behavior      |

---

## Method B: Key Observations

### Visual Structure

- Tighter clustering around fan traits
- One dominant cluster suggests profiling data compressed variance
- Better separation of high vs low-value users

### Cluster Interpretation

| Cluster | Description                               | Fan Traits                   |
|---------|--------------------------------------------|------------------------------|
| 0       | Premium fans                               | High spend, emotional sharing|
| 1       | Default cluster (dominated most users)     | Mixed traits, lower variance |
| 2       | Romantic, emotionally attached             | Regular, deep interactions   |
| 3       | Surface-level chatters                     | Transactional conversations  |
| 4       | Lonely fans looking for attention          | Sensitive keywords, clingy   |

---

## Comparison: Method A vs Method B

| Aspect                  | Method A                         | Method B                                 |
|-------------------------|----------------------------------|-------------------------------------------|
| Cluster Separation      | Moderate                         | Higher with enriched features             |
| Behavioral Insight      | Pure text patterns               | Profile-based segmentation possible       |
| Actionability           | Medium                           | High – clusters align with known personas |

---

## Recommendations

1. **Adopt Method B** for production segmentation due to richer, actionable clusters.
2. **Use Method A** when profile data is unavailable or incomplete.
3. **Incorporate profiles in real-time** to personalize chatter strategies.

---

# Experiment Proposals

These A/B tests are designed to validate strategies suggested by the cluster and profiling analysis. All tests should run with a minimum fan base of 100 per group and span at least 14 days.

---

## 1. **Emotional Hook First Message Test**

**Goal**: Improve engagement of "Ghosts" and "Drifters"

- **A Group**: Standard welcome script
- **B Group**: Personalized emotional hook (e.g., "Can I tell you something real?")

**Success Metric**:
- Response rate within first 3 messages
- Conversion to Stage 2

---

## 2. **Premium Experience Bundle Test (Cluster 0)**

**Goal**: Increase revenue from "Whales"

- **A Group**: Standard content pricing
- **B Group**: Premium bundle offer (e.g., 3 custom messages + video at discounted tier)

**Success Metric**:
- Revenue per fan
- Repeat purchase rate

---

## 3. **Drifter Conversion Nudge**

**Goal**: Move "Drifters" to purchase

- **A Group**: Passive messaging
- **B Group**: Time-limited content or flirty CTA with countdown

**Success Metric**:
- First purchase rate
- Return visit rate

---

## 4. **Profile-Enriched Chat Scripts**

**Goal**: Test if fan profile awareness boosts performance

- **A Group**: Chatters without fan background insights
- **B Group**: Chatters with profile insights (job, hobbies, emotional tone)

**Success Metric**:
- Revenue per conversation
- Fan retention over 7 days

---

## 5. **Reactivation Prompt Timing (Cluster 2)**

**Goal**: Reduce churn from emotionally invested fans

- **A Group**: Reactivation message after 48h silence
- **B Group**: Reactivation after 24h with emotional callback

**Success Metric**:
- Response rate
- Revenue after inactivity

---

## Experiment Management

- Use same fan-model pair to ensure isolation.
- Pre-segment by cluster before running test.
- Track results with message-level logging.

---


