# character_analysis.py
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils import add_bg_from_url
import re


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re
from utils import add_bg_from_url     # adjust if your path is different


def character_page() -> None:
    """Character-level emotion & dialogue dashboard."""
    add_bg_from_url()
    st.title("👥 Character Analysis")

    # ─────────────────────────────────────────
    # 1. Guard clause – make sure a DataFrame exists
    # ─────────────────────────────────────────
    if "df" not in st.session_state:
        st.warning("Please upload a file on the Home page first.")
        st.stop()

    df = st.session_state.df.copy()

    # ─────────────────────────────────────────
    # 2. 50 / 50 layout
    # ─────────────────────────────────────────
    left_col, right_col = st.columns([1, 1])

    # ───────── LEFT  – Emotion table + bar chart ─────────
    with left_col:
        st.subheader("🧍‍♂️ Character-Centric Emotion Map")

        # Clean speaker labels once
        df["Speaker_clean"] = (
            df["Speaker"]
            .str.replace(r"\s*\((CONT'D|O\.S\.|V\.O\.)\)", "", regex=True, flags=re.I)
            .str.strip()
        )

        # Filter to dialogue lines that appear at least twice
        speaker_counts = df.loc[df["Type"].str.lower() == "dialogue", "Speaker_clean"].value_counts()
        valid_speakers = speaker_counts[speaker_counts > 1].index

        emo_matrix = (
            df[
                df["Speaker_clean"].isin(valid_speakers)
                & (df["Type"].str.lower() == "dialogue")
            ]
            .groupby(["Speaker_clean", "emotion_label"])
            .size()
            .unstack(fill_value=0)
        )
        emo_matrix["Total"] = emo_matrix.sum(axis=1)
        emo_matrix = emo_matrix.sort_values("Total", ascending=False).drop(columns="Total")

        st.dataframe(emo_matrix, use_container_width=True)

        # Dialogue volume bar chart
        st.subheader("🎙️ Dialogue Volume by Character")
        char_counts = speaker_counts.loc[valid_speakers].head(10)
        st.bar_chart(char_counts)

    # ───────── RIGHT – Radar chart ─────────
    with right_col:
        st.subheader("🕸️ Emotion Distribution (Radar)")

        # Build radar dataframe (drop any “None” column quietly)
        radar_df = emo_matrix.drop(columns=["None"], errors="ignore")
        categories = radar_df.columns.tolist()

        # Prepare angles for each axis + wrap-around
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        # Bigger native canvas so the figure looks crisp when stretched
        fig_radar, ax_radar = plt.subplots(
            figsize=(7, 5),               # square figure
            subplot_kw=dict(polar=True)
        )

        ax_radar.set_theta_offset(np.pi / 2)
        ax_radar.set_theta_direction(-1)
        ax_radar.set_rlabel_position(0)

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, fontsize=10)
        ax_radar.set_ylim(0, radar_df.to_numpy().max() + 1)

        for speaker, row in radar_df.iterrows():
            values = row.tolist() + [row.iloc[0]]
            ax_radar.plot(angles, values, linewidth=2, label=speaker)
            ax_radar.fill(angles, values, alpha=0.1)

        # Slightly shrink and keep legend *inside* the figure so it scales
        ax_radar.legend(
            loc="upper right",
            bbox_to_anchor=(1.05, 1.10),
            fontsize=8,
            frameon=False,
        )
        ax_radar.set_title("Emotion Distribution per Character", y=1.15)

        # Stretch to full column width
        st.pyplot(fig_radar, use_container_width=True)

    st.markdown("---")
