# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import re
import joblib
import fitz
import shap
import wordninja
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import re

from textblob import TextBlob
from transformers import pipeline
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# 1.  MODEL LOADERS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        top_k=None,
    )


@st.cache_resource(show_spinner=False)
def load_retention_model(path: str = "toonstar_xgb_final.pkl") -> XGBRegressor:
    """Load the XGBoost model once per session."""
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        st.stop()
    return joblib.load(path)

emotion_classifier = load_emotion_model()
retention_model: XGBRegressor = load_retention_model()

# ──────────────────────────────────────────────────────────────────────────────
# 2.  HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://wallpapers.com/images/high/fade-4k-background-xra9dqc9u7bqbgqx.webp");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
    )

def is_probable_speaker(text: str) -> bool:
    text = text.strip()
    if (not text or not text.isupper()
            or re.search(r"[!?.,:;]$", text)
            or len(text.split()) > 5):
        return False
    return bool(re.match(r"^[A-Z0-9\s.\-]+(?:\([A-Z.'\s]+\))?$", text))


def classify_row(text: str) -> str:
    text_upper = text.strip().upper()
    if not text.strip() or text.strip() in {"***", "###"}:
        return "separator"
    if re.match(r"^(INT\.?|EXT\.?|EST\.?|INT/EXT)", text_upper):
        return "scene_heading"
    if re.match(r"^(CUT TO|FADE IN|FADE OUT|DISSOLVE TO|SMASH TO)", text_upper):
        return "transition"
    if text.strip().isdigit():
        return "page_number"
    if is_probable_speaker(text_upper):
        return "character"
    if re.match(r"^\(.*\)$", text.strip()):
        return "parenthetical"
    return "action"


def ends_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?…]\"?$", text.strip()))


def extract_script_id(filename: str) -> str:
    m = re.search(r"(PJ\d+)", filename, re.IGNORECASE)
    return m.group(1).upper() if m else "UNKNOWN"


def fix_spacing(text: str) -> str:
    if isinstance(text, str) and len(text.split()) == 1 and len(text) > 15:
        return " ".join(wordninja.split(text))
    return text


def get_sentiment(text: str):
    return TextBlob(text).sentiment.polarity if text and text.strip() else None


def get_emotion_info(text: str):
    """Returns (label, score)."""
    if not text or not text.strip():
        return ("None", 0.0)
    res = emotion_classifier(text)
    top = res[0][0] if isinstance(res[0], list) else res[0]
    return (top.get("label", "None"), top.get("score", 0.0))

# ──────────────────────────────────────────────────────────────────────────────
# 3.  DATAFRAME BUILDERS
# ──────────────────────────────────────────────────────────────────────────────
def assign_scene_ids(df: pd.DataFrame) -> pd.DataFrame:
    scene_id, prev_type, ids = 0, None, []
    for t in df["Type"]:
        if t == "scene_heading" and prev_type != "scene_heading":
            scene_id += 1
        ids.append(scene_id)
        prev_type = t
    df["scene_id"] = ids
    return df


def compute_dialogue_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in df.groupby("script id"):
        lines = g.reset_index()
        dlg = lines[lines["Type"].str.lower() == "dialogue"]
        total_words = dlg["script content"].str.split().str.len().sum()
        cum, n = 0, 1
        for idx, row in lines.iterrows():
            if row["Type"].lower() == "dialogue" and pd.notna(row["script content"]):
                words = len(str(row["script content"]).split())
                cum += words
                rows.append(
                    {"orig_index": row["index"], "line #": n,
                     "cumulative length": cum,
                     "total length of transcript": total_words}
                )
                n += 1
    return df.join(pd.DataFrame(rows).set_index("orig_index"))


def process_pdf_into_dataframe(file) -> pd.DataFrame:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    lines = [
        l.strip()
        for p in doc for l in p.get_text().split("\n")
        if l.strip()
    ]
    doc.close()

    sid = extract_script_id(file.name)
    ep_title = file.name.split(".")[0]
    rows = []
    i = 0
    while i < len(lines):
        line = lines[i]
        rtype = classify_row(line)

        # ---- CHARACTER block ------------------------------------------------
        if rtype == "character":
            speaker = line
            i += 1
            dlg = []
            while i < len(lines) and classify_row(lines[i]) not in {
                "character", "scene_heading", "transition"
            }:
                dlg.append(lines[i])
                if ends_sentence(lines[i]):
                    i += 1
                    break
                i += 1
            rows.append(
                {"script id": sid, "episode title": ep_title, "Type": "dialogue",
                 "Speaker": speaker, "script content": " ".join(dlg)}
            )

        # ---- ACTION block ---------------------------------------------------
        elif rtype == "action":
            block = [line]
            i += 1
            while i < len(lines) and classify_row(lines[i]) == "action":
                block.append(lines[i])
                if ends_sentence(lines[i]):
                    i += 1
                    break
                i += 1
            rows.append(
                {"script id": sid, "episode title": ep_title, "Type": "action",
                 "Speaker": None, "script content": " ".join(block)}
            )

        # ---- Other single-line types ----------------------------------------
        else:
            rows.append(
                {"script id": sid, "episode title": ep_title, "Type": rtype,
                 "Speaker": None, "script content": line}
            )
            i += 1

    df = pd.DataFrame(rows)

    # Basic NLP columns
    df["script content"] = df["script content"].apply(fix_spacing)
    df["length_of_text"] = df["script content"].str.count(r"\b\w+\b")
    df["sentiment"] = df["script content"].apply(get_sentiment)

    # Flags
    df["is_it_a_character_line"] = (df["Type"].str.lower() == "dialogue").astype("Int64")
    df["is_it_contextual_info"] = df["Type"].str.lower().isin(
        ["action", "context", "scene_heading"]
    ).astype("Int64")

    # Scene IDs + dialogue metrics
    df = assign_scene_ids(df)
    df = compute_dialogue_metrics(df)
    df["percentage_on_marker"] = (
        df["cumulative length"] / df["total length of transcript"] * 100
    ).round(2)

    # Emotion inference
    emo_tuples = df["script content"].apply(get_emotion_info)
    df[["emotion_label", "emotion_intensity"]] = pd.DataFrame(
        emo_tuples.tolist(), index=df.index
    )

    # Lead-character flag
    lead = df["Speaker"].dropna().value_counts().nlargest(1).index
    df["is_lead_character"] = df["Speaker"].isin(lead).astype("Int64")

    # Visual density per scene
    dlg_tot = df[df["Type"].str.lower() == "dialogue"].groupby("scene_id")[
        "length_of_text"
    ].transform("sum")
    df["visual_density_score"] = (df["length_of_text"] / dlg_tot * 100).round(2)

    # Clean up columns
    df['Speaker'] = df['Speaker'].str.replace(r'\(.*?\)', '', regex=True).str.strip()
    df['script content'] = df['script content'].str.replace('â', "'", regex=False)
    df['script content'] = df['script content'].str.replace('Ã©', 'é', regex=False)
    df['script content'] = df['script content'].str.replace('â', '"', regex=False)
    df['script content'] = df['script content'].str.replace('â', '"', regex=False)
    df['Speaker'] = df['Speaker'].str.replace('â', "'", regex=False)

    return df