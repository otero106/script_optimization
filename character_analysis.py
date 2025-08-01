# character_analysis.py
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils import add_bg_from_url
import re


def character_page():
    add_bg_from_url()
    st.title("👥 Character Analysis") 

    if 'df' not in st.session_state:
        st.warning("Please upload a file on the Home page first.")
        st.stop()

    df = st.session_state.df

    # LAYOUT: LEFT COLUMN (Emotion Table + Dialogue Volume) | RIGHT COLUMN (Radar)
    col3, col4 = st.columns([2, 1])

    # LEFT SIDE (Emotion Table + Dialogue Volume)
    with col3:
        st.subheader("🧍‍♂️ Character-Centric Emotion Map")

        df["Speaker_clean"] = (
            df["Speaker"]
            .str.replace(r"\s*\((CONT'D|O\.S\.|V\.O\.)\)", "", regex=True, flags=re.I)
            .str.strip()
        )
        speaker_counts = df[df["Type"].str.lower() == "dialogue"]["Speaker_clean"].value_counts()
        valid_spk = speaker_counts[speaker_counts > 1].index

        emo_matrix = (
            df[df["Speaker_clean"].isin(valid_spk) & (df["Type"].str.lower() == "dialogue")]
            .groupby(["Speaker_clean", "emotion_label"])
            .size()
            .unstack(fill_value=0)
        )
        emo_matrix["Total"] = emo_matrix.sum(axis=1)
        emo_matrix = emo_matrix.sort_values("Total", ascending=False).drop(columns="Total")

        st.dataframe(emo_matrix, use_container_width=True)

        # ↓ BELOW EMOTION TABLE: Dialogue Volume
        st.subheader("🎙️ Dialogue Volume by Character")
        char_counts = (
            df[df["Type"].str.lower() == "dialogue"]["Speaker_clean"]
            .value_counts()
            .loc[lambda s: s > 1]
            .head(10)
        )
        st.bar_chart(char_counts)

    # RIGHT SIDE: Radar Chart
    with col4:
        st.subheader("🕸️ Emotion Distribution (Radar)")
        radar_df = emo_matrix.drop(columns=["None"], errors="ignore")
        cats = radar_df.columns.tolist()
        angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
        angles += angles[:1]

        fig_radar, ax_radar = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
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
