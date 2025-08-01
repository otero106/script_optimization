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
    st.pyplot(fig_arc)

    st.markdown("---")

    # CHARACTER EMOTION MAP AND RADAR CHART (Side-by-Side)
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🧍‍♂️ Character-Centric Emotion Map")
        df["Speaker_clean"] = (
            df["Speaker"]
            .str.replace(r"\s*\((CONT'D|O\.S\.|V\.O\.)\)", "", regex=True, flags=re.I)
            .str.strip()
        )
        speaker_counts = (df[df["Type"].str.lower() == "dialogue"]["Speaker_clean"].value_counts())
        valid_spk = speaker_counts[speaker_counts > 1].index
        emo_matrix = (
            df[
                df["Speaker_clean"].isin(valid_spk)
                & (df["Type"].str.lower() == "dialogue")
            ]
            .groupby(["Speaker_clean", "emotion_label"])
            .size()
            .unstack(fill_value=0)
        )
        emo_matrix["Total"] = emo_matrix.sum(axis=1)
        emo_matrix = emo_matrix.sort_values("Total", ascending=False).drop(columns="Total")
        st.dataframe(emo_matrix, use_container_width=True)
    with col4:
        st.subheader("🕸️ Emotion Distribution (Radar)")
        radar_df = emo_matrix.drop(columns=["None"], errors="ignore")
        cats = radar_df.columns.tolist()
        angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
        angles += angles[:1]
        fig_radar, ax_radar = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        ax_radar.set_theta_offset(np.pi / 2)
        ax_radar.set_theta_direction(-1)
        ax_radar.set_rlabel_position(0)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(cats)
        ax_radar.set_ylim(0, radar_df.to_numpy().max() + 1)
        for idx, row in radar_df.iterrows():
            vals = row.tolist() + [row.tolist()[0]]
            ax_radar.plot(angles, vals, linewidth=2, label=idx)
            ax_radar.fill(angles, vals, alpha=0.1)
        ax_radar.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=9)
        ax_radar.set_title("Emotion Distribution per Character", y=1.1)
        st.pyplot(fig_radar)
    st.markdown("---")
    
    # EXPLORE LINES
    st.subheader("🔍 Explore Script Lines")
    scene_filter = st.selectbox(
        "Select a Scene", sorted(df["scene_id"].unique())
    )
    st.dataframe(
        df[df["scene_id"] == scene_filter][
            [
                "line #",
                "Speaker",
                "script content",
                "emotion_label",
                "sentiment",
                "emotion_intensity",
            ]
        ],
        use_container_width=True,
    )
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
