# retention_analysis_page.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import r2_score
from utils import load_retention_model

def retention_page():
    add_bg_from_url()
    
    st.title("📈 Retention Analysis")

    if 'df' not in st.session_state:
        st.warning("Please upload a file on the Home page first.")
        st.stop()

    df = st.session_state.df
    retention_model = load_retention_model()

    # ML PRE-PROCESS & PREDICT  (absolute retention target)
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
    fig_curve, ax_curve = plt.subplots(figsize=(7, 4))
    ax_curve.plot(pred_scene.index, pred_scene.values,
                  label="Predicted", marker="x")
    if has_ret:
        ax_curve.plot(true_scene.index, true_scene.values,
                      label="Actual", marker="o", linestyle="--", alpha=0.75)
    ax_curve.set_xlabel("Scene ID")
    ax_curve.set_ylabel("Audience Still Watching (%)")
    ax_curve.set_title("Scene-level Audience Retention")
    ax_curve.legend()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.pyplot(fig_curve)
    st.markdown("---")

    # ------------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------------------
    st.markdown("#### 🎯 Biggest Story Elements Driving Retention")
    imp = retention_model.feature_importances_
    idx = imp.argsort()[::-1][:12]
    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh([feats[i] for i in idx][::-1], imp[idx][::-1])
    ax_imp.set_title("Top Factors That Move Retention")
    ax_imp.set_xlabel("Importance")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig_imp)
    st.markdown("---")


    # ------------------------------------------------------------------
    # SHAP BEESWARM
    # ------------------------------------------------------------------
    st.markdown("#### 🐝 How Those Factors Nudge Viewers")
    explainer = shap.TreeExplainer(retention_model)
    shap_vals = explainer.shap_values(work_df, check_additivity=False)
    fig_bee = plt.figure()
    shap.summary_plot(
        shap_vals, work_df, show=False, plot_type="dot", max_display=15
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig_bee)
    st.markdown("---")

    # ------------------------------------------------------------------
    # PLAIN-ENGLISH TAKE-AWAYS
    # ------------------------------------------------------------------
    st.markdown("#### 📜 What this means for your script")
    mean_shap = pd.Series(shap_vals.mean(axis=0), index=feats)
    top_pos = mean_shap.nlargest(5)
    top_neg = mean_shap.nsmallest(5)

    bullets = ["##### 🔼 Elements that *raise* retention"]
    for feat, v in top_pos.items():
        bullets.append(f"- **{feat}** ↑ by **{v:+.2f} pp** on average")

    bullets.append("")
    bullets.append("##### 🔽 Elements that *lower* retention")
    for feat, v in top_neg.items():
        bullets.append(f"- **{feat}** ↓ by **{v:.2f} pp** on average")

    st.markdown("\n".join(bullets))