# emotion_analysis_page.py
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils import add_bg_from_url
import re

def emotion_page():
    add_bg_from_url()
    st.title("🎭 Emotion Analysis")

    if 'df' not in st.session_state:
        st.warning("Please upload a file on the Home page first.")
        st.stop()

    df = st.session_state.df

    # HEADER & PREVIEW
    st.markdown("### 🧪 Emotion Classification Preview")
    preview_df = df[
        (df["Type"].str.lower() == "dialogue")
        & df["emotion_label"].notna()
        & df["emotion_intensity"].notna()
    ][["script content", "emotion_label", "emotion_intensity"]].head(10)
    st.dataframe(preview_df, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🧾 Total Lines", len(df))
    c2.metric(
        "🎙️ Lead Character Lines",
        f"{df['is_lead_character'].mean() * 100:.1f}%",
    )
    c3.metric("💬 Emotional Dialogue", df["emotion_intensity"].notna().sum())
    c4.metric("🙂 Avg Sentiment", f"{df['sentiment'].mean():.2f}")
    st.markdown("---")

    # HEATMAP
    st.subheader("📊 Simulated Engagement Heatmap")
    df["fake_risk_score"] = (
        1 - (df["emotion_intensity"] * df["visual_density_score"] / 100)
    ).clip(0, 1)
    fig_heat, ax_heat = plt.subplots(figsize=(12, 1.1))
    sns.heatmap(
        [df["fake_risk_score"].fillna(0).tolist()],
        cmap="coolwarm",
        yticklabels=["Risk"],
        xticklabels=False,
        cbar=True,
        ax=ax_heat,
    )
    st.pyplot(fig_heat)
    st.markdown("---")

    # EMOTION DISTRIBUTION BY SCENE
    st.subheader("🎭 Emotion Distribution by Scene")
    emo_scene = (
        df[df["is_it_a_character_line"] == 1]
        .groupby("scene_id")["emotion_label"]
        .value_counts()
        .unstack()
        .fillna(0)
    )
    st.bar_chart(emo_scene)
    st.markdown("---")

    # EMOTION ARC 
    st.subheader("📈 Emotion Arc Across Scenes")
    emo_avg = df.groupby("scene_id")["emotion_intensity"].mean().reset_index()
    fig_arc, ax_arc = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=emo_avg, x="scene_id", y="emotion_intensity", marker="o", ax=ax_arc)
    ax_arc.set_title("Emotion Arc Across Scenes")
    ax_arc.set_xlabel("Scene ID")
    ax_arc.set_ylabel("Avg Emotion Intensity")
    st.pyplot(fig_arc, use_container_width=False)

    st.markdown("---")

    # TOP EMOTIONAL SCENES
    st.subheader("🔥 Top Emotional Scenes")
    top_scenes = (
        df.groupby("scene_id")[["emotion_intensity", "visual_density_score"]]
        .mean()
        .sort_values("emotion_intensity", ascending=False)
        .reset_index()
    )
    st.dataframe(top_scenes, use_container_width=True)
