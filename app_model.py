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

    return df

# ──────────────────────────────────────────────────────────────────────────────
# 4.  STREAMLIT UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Script Analyzer", layout="wide")
st.title("🎬 Script Analyzer Prototype")
st.markdown("Upload your annotated script **CSV** or raw **PDF** script file.")

uploaded_file = st.file_uploader("📁 Upload a script file", type=["csv", "pdf"])

# ----------------------------- Load file
if uploaded_file and uploaded_file.name.endswith(".pdf"):
    with st.spinner("Parsing PDF…"):
        df = process_pdf_into_dataframe(uploaded_file)
    st.success("✅ PDF parsed!")
elif uploaded_file and uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV loaded!")

# ----------------------------- Show UI only if file present
if uploaded_file:
    # ══════════════════════════════════════════════════════════════════
    #  HEADER & PREVIEW
    # ══════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════
    #  HEATMAP
    # ══════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════
    #  EMOTION DISTRIBUTION BY SCENE
    # ══════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════
    #  EXPLORE LINES
    # ══════════════════════════════════════════════════════════════════
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

    # ════════════════════════════════════════════════════════════════
    #  EMOTION ARC  +  DIVERSITY TABLE  side-by-side
    # ════════════════════════════════════════════════════════════════
    chart_col, tbl_col = st.columns([1.7, 0.9])          # 65 % | 35 %

    # ----- LEFT ▸ Emotion-arc line chart ----------------------------
    with chart_col:
        st.subheader("📈 Emotion Arc Across Scenes")

        emo_avg = (
            df.groupby("scene_id")["emotion_intensity"]
              .mean()
              .reset_index()
        )

        fig_arc, ax_arc = plt.subplots(figsize=(5, 3.2))   # compact
        sns.lineplot(
            data=emo_avg,
            x="scene_id", y="emotion_intensity",
            marker="o", ax=ax_arc
        )
        ax_arc.set_title("Emotion Arc Across Scenes", fontsize=10)
        ax_arc.set_xlabel("Scene ID",                 fontsize=8)
        ax_arc.set_ylabel("Avg Emotion Intensity",    fontsize=8)
        ax_arc.tick_params(labelsize=8)

        st.pyplot(fig_arc, use_container_width=False)

    # ----- RIGHT ▸ Emotion-diversity table --------------------------
    with tbl_col:
        st.subheader("🎨 Emotion Diversity by Scene")

        emo_div = (
            df.groupby("scene_id")["emotion_label"]
              .nunique()
              .reset_index(name="diversity")
              .astype(int)                # prettier ints
        )

        st.table(emo_div)                 # auto-sized, no blank rows

    st.markdown("---")


    # ══════════════════════════════════════════════════════════════════
    #  DIALOGUE VOLUME
    # ══════════════════════════════════════════════════════════════════
    st.subheader("🎙️ Dialogue Volume by Character")
    df["Speaker_clean"] = df["Speaker"].str.replace(
        r"\s*\(.*\)", "", regex=True
    ).str.strip()
    char_counts = (
        df[df["Type"].str.lower() == "dialogue"]["Speaker_clean"]
        .value_counts()
        .loc[lambda s: s > 1]
        .head(10)
    )
    st.bar_chart(char_counts)

    # ══════════════════════════════════════════════════════════════════
    #  CHARACTER MAP  +  RADAR SIDE-BY-SIDE
    # ══════════════════════════════════════════════════════════════════
    left_col, right_col = st.columns([0.9, 1.7])   # 60 % | 40 % width

    # ── LEFT  ▸ character-centric table ───────────────────────────────
    with left_col:
        st.subheader("🧍‍♂️ Character-Centric Emotion Map")

        df["Speaker_clean"] = (
            df["Speaker"]
              .str.replace(r"\s*\((CONT'D|O\.S\.|V\.O\.)\)", "", regex=True, flags=re.I)
              .str.strip()
        )

        valid = (
            df[df["Type"].str.lower() == "dialogue"]["Speaker_clean"]
              .value_counts()
              .loc[lambda s: s > 1]
              .index
        )

        emo_matrix = (
            df[
                df["Speaker_clean"].isin(valid) &
                (df["Type"].str.lower() == "dialogue")
            ]
            .groupby(["Speaker_clean", "emotion_label"])
            .size()
            .unstack(fill_value=0)
            .astype(int)
        )

        st.table(emo_matrix)        # just enough rows – no blank lines

    # ── RIGHT ▸ radar chart ────────────────────────────────────────────
    with right_col:
        st.subheader("🕸️ Emotion Distribution per Character (Radar)")

        radar_df = emo_matrix.drop(columns=["None"], errors="ignore")
        cats      = radar_df.columns.tolist()
        angles    = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist() + [0]

        SMALL = 7           # one knob for all fonts

        fig_radar, ax_radar = plt.subplots(figsize=(4.4, 4.0),
                                       subplot_kw=dict(polar=True))

        ax_radar.set_theta_offset(np.pi/2)
        ax_radar.set_theta_direction(-1)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(cats, fontsize=SMALL)
        ax_radar.tick_params(labelsize=SMALL)
        ax_radar.set_rlabel_position(0)
        ax_radar.set_ylim(0, radar_df.to_numpy().max() + 1)

        for name, row in radar_df.iterrows():
            vals = row.tolist() + [row.tolist()[0]]
            ax_radar.plot(angles, vals, linewidth=1.5, label=name)
            ax_radar.fill(angles, vals, alpha=0.08)

        ax_radar.legend(
            loc="upper left",
            bbox_to_anchor=(1.05, 1.0),
            fontsize=SMALL,
            frameon=False
        )
        st.pyplot(fig_radar, use_container_width=False)

    st.markdown("---")
  
    # ══════════════════════════════════════════════════════════════════
    #  TOP EMOTIONAL SCENES
    # ══════════════════════════════════════════════════════════════════
    st.subheader("🔥 Top Emotional Scenes")
    top_scenes = (
        df.groupby("scene_id")[["emotion_intensity", "visual_density_score"]]
        .mean()
        .sort_values("emotion_intensity", ascending=False)
        .reset_index()
    )
    st.dataframe(top_scenes, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────
    # STEP 3 – ML PRE-PROCESS & PREDICT  (absolute retention target)
    # ──────────────────────────────────────────────────────────────────
    st.markdown("## 🔧 Predictive Retention Analysis")

    has_ret = "Absolute audience retention (%)" in df.columns

    # ---- clean + one-hot ----------------------------------------------------
    work_df = df.copy()
    miss = work_df.isna().mean()
    work_df = work_df.drop(columns=miss[miss > 0.50].index)

    num_cols = work_df.select_dtypes(include="number").columns
    cat_cols = work_df.select_dtypes(include="object").columns
    work_df[num_cols] = work_df[num_cols].fillna(work_df[num_cols].mean())
    for c in cat_cols:
        work_df[c] = work_df[c].fillna(work_df[c].mode(dropna=True).iloc[0])

    work_df = pd.get_dummies(work_df, columns=cat_cols, drop_first=True)

    # ---- align to training features ----------------------------------------
    feats = retention_model.feature_names_in_
    for c in feats:
        if c not in work_df:
            work_df[c] = 0
    work_df = work_df[feats]

    # ---- predict ------------------------------------------------------------
    with st.spinner("Crunching the numbers…"):
        y_pred_abs = retention_model.predict(work_df)

    df["pred_abs"] = y_pred_abs
    if has_ret:
        df["true_abs"] = df["Absolute audience retention (%)"]

    # ---- aggregate by scene -------------------------------------------------
    pred_scene = df.groupby("scene_id")["pred_abs"].mean().sort_index()
    if has_ret:
        true_scene = df.groupby("scene_id")["true_abs"].mean().sort_index()
        r2_pct = 100 * r2_score(true_scene.values, pred_scene.values)
        st.success(
            f"📈 **{r2_pct:0.1f}% of the variation in audience retention** "
            f"is explained by scene-level script features."
        )
    else:
        st.info("🔮 Predicted retention by scene (no ground-truth column).")

    # ---- plot curve ---------------------------------------------------------
    fig_curve, ax_curve = plt.subplots(figsize=(12, 6))
    ax_curve.plot(pred_scene.index, pred_scene.values,
                  label="Predicted", marker="x")
    if has_ret:
        ax_curve.plot(true_scene.index, true_scene.values,
                      label="Actual", marker="o", linestyle="--", alpha=0.75)
    ax_curve.set_xlabel("Scene ID")
    ax_curve.set_ylabel("Audience still watching (%)")
    ax_curve.set_title("Scene-level audience retention")
    ax_curve.legend()
    st.pyplot(fig_curve,use_container_width=False)

    # ══════════════════════════════════════════════════════════════════
    #  FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════════════
    st.markdown("#### 🎯 Biggest Story Elements Driving Retention")
    imp = retention_model.feature_importances_
    idx = imp.argsort()[::-1][:12]
    fig_imp, ax_imp = plt.subplots(figsize=(6, 4))      # ⬅️ smaller
    ax_imp.barh([feats[i] for i in idx][::-1], imp[idx][::-1])
    ax_imp.set_title("Top factors that move retention")
    ax_imp.set_xlabel("Importance")
    st.pyplot(fig_imp, use_container_width=False)
    
    # ───────────────────────────────────────────
    #  Build SHAP values  (must come first)
    # ───────────────────────────────────────────
    explainer  = shap.TreeExplainer(retention_model)
    shap_vals  = explainer.shap_values(work_df, check_additivity=False)
        
    # ════════════════════════════════════════════════════════════════
    #  SHAP  – beeswarm  +  plain-English take-aways side-by-side
    # ════════════════════════════════════════════════════════════════
    # ───── context for the SHAP visuals ─────────────────────────────
    st.markdown(
        """
    <small>
    **How to read this section**  
    * Each dot in the **beeswarm** (left) is one line of your script.  
      &nbsp;&nbsp;• Dots to the **right** push predicted retention **up** (positive SHAP value).  
      &nbsp;&nbsp;• Dots to the **left** push it **down**.  
      &nbsp;&nbsp;• Colour shows whether the original feature value was high (pink) or low (blue).  

    * The bullets (right) translate the average effect into **percentage-point (pp) change**.<br>
      *“+0.12 pp” means that feature raises the retention prediction by 0.12 percentage points on average.*
    </small>
    """,
        unsafe_allow_html=True,
    )

    left_shap, right_text = st.columns([1.4, 1])        # 60 % | 40 %

    # ── LEFT ▸ beeswarm plot ─────────────────────────────────────────
    with left_shap:
        st.caption("Beeswarm – individual impact")

        fig_bee = plt.figure(figsize=(4.2, 3.8))
        shap.summary_plot(
            shap_vals,
            work_df,
            show=False,
            plot_type="dot",
            max_display=15
        )
        st.pyplot(fig_bee, use_container_width=False)

    # ── RIGHT ▸ narrative bullets ───────────────────────────────────
    with right_text:
        st.caption("What this means for your script")

        mean_shap = pd.Series(shap_vals.mean(axis=0), index=feats)

        top_pos = mean_shap.nlargest(5)
        top_neg = mean_shap.nsmallest(5)

        bullet_lines = ["##### 🔼 Elements that *raise* retention"]
        for feat, val in top_pos.items():
            bullet_lines.append(f"- **{feat}** ↑ by **{val:+.2f} pp** on average")

        bullet_lines.append("")  # blank line

        bullet_lines.append("##### 🔽 Elements that *lower* retention")
        for feat, val in top_neg.items():
            bullet_lines.append(f"- **{feat}** ↓ by **{abs(val):.2f} pp** on average")

        st.markdown("\n".join(bullet_lines))

    st.markdown("---")


