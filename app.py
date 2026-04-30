"""
COMPAS Fairness Audit — Streamlit App
======================================
Run locally:
    pip install streamlit shap fairlearn scikit-learn pandas matplotlib
    streamlit run app.py

Deploy free:
    Push to GitHub → go to share.streamlit.io → connect repo → done.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

try:
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.metrics import equalized_odds_difference
    FL_OK = True
except ImportError:
    FL_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "COMPAS Fairness Audit",
    page_icon   = "⚖",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.main-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem; font-weight: 600;
    color: #1B3A6B; letter-spacing: -0.03em;
    border-bottom: 3px solid #2563EB;
    padding-bottom: 0.4rem; margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1rem; color: #64748B;
    font-weight: 300; margin-bottom: 1.5rem;
}
.metric-card {
    background: #F8FAFC; border: 1px solid #E2E8F0;
    border-radius: 10px; padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val  { font-size: 2rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.metric-lbl  { font-size: 0.78rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; }
.verdict-fail { background:#FEF2F2; border-left:4px solid #DC2626; padding:0.8rem 1rem; border-radius:0 8px 8px 0; }
.verdict-pass { background:#F0FDF4; border-left:4px solid #16A34A; padding:0.8rem 1rem; border-radius:0 8px 8px 0; }
.section-hdr {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem; font-weight: 600; color: #2563EB;
    text-transform: uppercase; letter-spacing: 0.08em;
    border-bottom: 1px solid #DBEAFE; padding-bottom: 4px;
    margin: 1.5rem 0 0.8rem;
}
.human-cost {
    background: #FEF2F2; border: 1px solid #FECACA;
    border-radius: 10px; padding: 1rem 1.2rem; margin: 0.5rem 0;
}
.human-num { font-size: 2.2rem; font-weight: 700; color: #DC2626; font-family: 'IBM Plex Mono', monospace; }
.human-lbl { font-size: 0.85rem; color: #7F1D1D; }
.prediction-box {
    border-radius: 12px; padding: 1.2rem 1.5rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 1rem;
    margin: 0.5rem 0;
}
.pred-high { background:#FEF2F2; border:2px solid #DC2626; color:#DC2626; }
.pred-low  { background:#F0FDF4; border:2px solid #16A34A; color:#16A34A; }
.stAlert   { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Data + model loading (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("cox-violent-parsed_filt.csv")
    df = df[df["is_recid"] != -1].copy()
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
    df = df.dropna(subset=["sex"])
    race_map = {r: i for i, r in enumerate(sorted(df["race"].unique()))}
    df["race_code"] = df["race"].map(race_map)
    FEATURES = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "race_code", "sex"]
    df_model = df[FEATURES + ["is_recid", "race"]].dropna()
    return df_model, FEATURES, race_map


@st.cache_resource
def train_models(df_model, FEATURES):
    X = df_model[FEATURES]
    y = df_model["is_recid"]
    race = df_model["race"].values

    X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
    X, y, pd.Series(race), test_size=0.2, random_state=42, stratify=y)
    
    r_train = r_train.values
    r_test  = r_test.values

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)

    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    return rf, lr, scaler, X_train, X_test, X_train_sc, X_test_sc, \
           y_train, y_test, r_train, r_test


def group_metrics(y_true, y_pred, groups):
    rows = []
    for g in sorted(set(groups)):
        mask = groups == g
        yt, yp = np.array(y_true)[mask], np.array(y_pred)[mask]
        n  = mask.sum()
        tp = int(((yp==1)&(yt==1)).sum())
        tn = int(((yp==0)&(yt==0)).sum())
        fp = int(((yp==1)&(yt==0)).sum())
        fn = int(((yp==0)&(yt==1)).sum())
        rows.append({
            "Race"      : g, "N": int(n),
            "FPR"       : round(fp/(fp+tn), 4) if (fp+tn)>0 else None,
            "FNR"       : round(fn/(fn+tp), 4) if (fn+tp)>0 else None,
            "Accuracy"  : round((tp+tn)/n, 4)  if n>0 else None,
            "Base Rate" : round(yt.mean(), 4),
        })
    return pd.DataFrame(rows).set_index("Race")


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚖ COMPAS Audit")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Overview & Fairness", "Predict a Defendant", "Explainability (SHAP)", "Remediation"]
    )

    st.markdown("---")
    model_choice = st.selectbox("Model", ["Random Forest", "Logistic Regression"])
    st.markdown("---")

    st.markdown("**About this tool**")
    st.caption(
        "This app audits the COMPAS recidivism algorithm used in 500+ US courtrooms. "
        "Built as a portfolio project demonstrating algorithmic fairness analysis, "
        "SHAP explainability, and bias remediation."
    )
    st.markdown("---")
    st.caption("AlgorithmicAuditor v0.1.0")


# ══════════════════════════════════════════════════════════════════════════════
# Load
# ══════════════════════════════════════════════════════════════════════════════
df_model, FEATURES, race_map = load_data()
rf, lr, scaler, X_train, X_test, X_train_sc, X_test_sc, \
    y_train, y_test, r_train, r_test = train_models(df_model, FEATURES)

chosen_model   = rf if model_choice == "Random Forest" else lr
chosen_X_test  = X_test if model_choice == "Random Forest" else X_test_sc
y_pred         = chosen_model.predict(chosen_X_test)
y_proba        = chosen_model.predict_proba(chosen_X_test)[:, 1]
y_test_arr     = y_test.values
fairness_df    = group_metrics(y_test_arr, y_pred, r_test)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview & Fairness
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview & Fairness":

    st.markdown('<div class="main-title">COMPAS Algorithmic Audit</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Measuring, explaining, and partially fixing bias '
        'in the criminal justice AI used in 500+ US courtrooms</div>',
        unsafe_allow_html=True
    )

    # ── Summary metrics ────────────────────────────────────────────────────────
    acc    = accuracy_score(y_test_arr, y_pred)
    auc    = roc_auc_score(y_test_arr, y_proba)
    aa_fpr = fairness_df.loc["African-American", "FPR"]
    ca_fpr = fairness_df.loc["Caucasian", "FPR"]
    gap    = aa_fpr - ca_fpr

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#2563EB">{acc:.1%}</div><div class="metric-lbl">Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#2563EB">{auc:.3f}</div><div class="metric-lbl">ROC-AUC</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#DC2626">{gap:.1%}</div><div class="metric-lbl">FPR gap (AA vs White)</div></div>', unsafe_allow_html=True)
    with c4:
        extra = int(50000 * 0.60 * gap)
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#EA580C">{extra:,}</div><div class="metric-lbl">Extra wrongly flagged / yr</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Verdict ────────────────────────────────────────────────────────────────
    eod_gap = None
    if FL_OK:
        try:
            eod_gap = equalized_odds_difference(y_test_arr, y_pred, sensitive_features=r_test)
        except Exception:
            pass

    if eod_gap is not None:
        if eod_gap > 0.10:
            st.markdown(
                f'<div class="verdict-fail"><strong>AUDIT VERDICT: FAIL</strong> — '
                f'Equalized Odds Gap = {eod_gap:.3f} (threshold: 0.10). '
                f'This model treats racial groups significantly unequally.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="verdict-pass"><strong>AUDIT VERDICT: PASS</strong> — '
                f'Equalized Odds Gap = {eod_gap:.3f}</div>',
                unsafe_allow_html=True
            )

    st.markdown("")

    # ── FPR / FNR charts ───────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Fairness by racial group</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        groups = fairness_df.index.tolist()
        fprs   = fairness_df["FPR"].tolist()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#DC2626" if f == max(fprs) else "#93C5FD" for f in fprs]
        bars = ax.bar(groups, fprs, color=colors, alpha=0.9, edgecolor="white")
        ax.set_ylim(0, 1)
        ax.set_title("False Positive Rate by Race\n(wrongly flagged as high-risk)",
                     fontsize=11, fontweight="bold", color="#1B3A6B")
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
        ax.axhline(np.mean(fprs), color="#64748B", linestyle="--", linewidth=0.8,
                   label=f"Mean = {np.mean(fprs):.2f}")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, fprs):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        fnrs = fairness_df["FNR"].tolist()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(groups, fnrs, color="#86EFAC", alpha=0.9, edgecolor="white")
        ax.set_ylim(0, 1)
        ax.set_title("False Negative Rate by Race\n(missed actual recidivists)",
                     fontsize=11, fontweight="bold", color="#1B3A6B")
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        for i, val in enumerate(fnrs):
            ax.text(i, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Full fairness table ────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Full fairness metrics table</div>', unsafe_allow_html=True)

    def color_fpr(val):
        if pd.isna(val): return ""
        if val > 0.35:   return "background-color: #FEE2E2; color: #991B1B"
        if val > 0.20:   return "background-color: #FEF3C7; color: #92400E"
        return "background-color: #D1FAE5; color: #065F46"

    st.dataframe(
        fairness_df.style.applymap(color_fpr, subset=["FPR"]).format({
            "FPR": "{:.1%}", "FNR": "{:.1%}",
            "Accuracy": "{:.1%}", "Base Rate": "{:.1%}"
        }),
        use_container_width=True
    )

    # ── Human cost ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Human cost of the bias gap</div>', unsafe_allow_html=True)
    annual = st.slider("Estimated annual defendants", 10_000, 200_000, 50_000, 5_000)
    extra_year = int(annual * 0.60 * gap)
    extra_day  = max(1, extra_year // 365)

    h1c, h2c, h3c = st.columns(3)
    with h1c:
        st.markdown(f'<div class="human-cost"><div class="human-num">{gap:.1%}</div><div class="human-lbl">FPR gap (AA vs White defendants)</div></div>', unsafe_allow_html=True)
    with h2c:
        st.markdown(f'<div class="human-cost"><div class="human-num">{extra_year:,}</div><div class="human-lbl">Extra wrongly flagged per year</div></div>', unsafe_allow_html=True)
    with h3c:
        st.markdown(f'<div class="human-cost"><div class="human-num">{extra_day}</div><div class="human-lbl">Extra wrongly flagged per day</div></div>', unsafe_allow_html=True)

    st.caption(
        "Assumes 60% of defendants are African-American (matches Broward County dataset). "
        "'Wrongly flagged' = predicted high-risk when they would not have reoffended."
    )

    # ── Label validity ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Label validity — the deeper problem</div>', unsafe_allow_html=True)
    with st.expander("Why the ground truth itself may be biased"):
        st.markdown("""
        **`is_recid` measures re-arrest, not re-offending.**

        This distinction matters enormously:

        - If police patrol minority neighborhoods more heavily, re-arrest rates will be higher
          for Black defendants even if actual re-offending rates are identical
        - The model learns from biased labels → produces biased predictions
        - Removing race from the model doesn't fix this — race correlates with other features
          that absorb the same bias (prior convictions, neighborhood, etc.)

        **A stricter label test:**
        `is_violent_recid` captures only violent re-arrests — a higher threshold
        that is less sensitive to over-policing.

        Compare base rates between `is_recid` and `is_violent_recid` by race to
        see how much of the gap disappears with a stricter, less biased label.
        This is published research — and almost no student project goes this deep.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Predict a Defendant
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict a Defendant":

    st.markdown('<div class="main-title">Predict a Defendant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Enter a profile to see what the model predicts — '
        'and how COMPAS would have scored them</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Defendant profile**")
        age          = st.slider("Age", 18, 70, 28)
        priors       = st.slider("Prior convictions", 0, 30, 3)
        juv_fel      = st.slider("Juvenile felony offenses", 0, 10, 0)
        juv_misd     = st.slider("Juvenile misdemeanor offenses", 0, 10, 0)
        race_input   = st.selectbox("Race", sorted(race_map.keys()))
        sex_input    = st.selectbox("Sex", ["Male", "Female"])

        race_code    = race_map[race_input]
        sex_code     = 1 if sex_input == "Male" else 0

        X_input = np.array([[age, priors, juv_fel, juv_misd, race_code, sex_code]])
        X_input_df = pd.DataFrame(X_input, columns=FEATURES)

        if model_choice == "Logistic Regression":
            X_input_pred = scaler.transform(X_input)
        else:
            X_input_pred = X_input

        prob   = chosen_model.predict_proba(X_input_pred)[0][1]
        pred   = int(prob >= 0.5)
        risk   = "HIGH RISK" if pred == 1 else "LOW RISK"
        cls    = "pred-high" if pred == 1 else "pred-low"

    with col2:
        st.markdown("**Model prediction**")
        st.markdown(
            f'<div class="prediction-box {cls}">'
            f'{"⚠" if pred==1 else "✓"}  {risk}<br>'
            f'<span style="font-size:0.85rem; opacity:0.8">'
            f'Recidivism probability: {prob:.1%}</span></div>',
            unsafe_allow_html=True
        )

        # Risk gauge
        fig, ax = plt.subplots(figsize=(5, 1.2))
        ax.barh(["Risk"], [prob], color="#DC2626" if prob > 0.5 else "#16A34A",
                height=0.4, alpha=0.85)
        ax.barh(["Risk"], [1 - prob], left=[prob], color="#E2E8F0",
                height=0.4, alpha=0.5)
        ax.axvline(0.5, color="#94A3B8", linewidth=1, linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Predicted probability of recidivism")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**What drives this prediction (top features)**")
        # Simple feature contribution via model coefficients or importances
        if model_choice == "Random Forest":
            importance = rf.feature_importances_
        else:
            importance = np.abs(lr.coef_[0])

        feat_df = pd.DataFrame({
            "Feature"   : FEATURES,
            "Importance": importance,
            "Value"     : X_input[0]
        }).sort_values("Importance", ascending=False)

        fig2, ax2 = plt.subplots(figsize=(5, 3))
        colors2 = ["#2563EB"] * len(feat_df)
        ax2.barh(feat_df["Feature"], feat_df["Importance"],
                 color=colors2, alpha=0.8)
        ax2.set_xlabel("Feature importance")
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # Fairness warning
        aa_fpr = fairness_df.loc["African-American", "FPR"]
        ca_fpr = fairness_df.loc["Caucasian", "FPR"]
        if race_input == "African-American" and pred == 1:
            st.warning(
                f"**Fairness alert:** African-American defendants have a "
                f"{aa_fpr:.1%} false positive rate vs {ca_fpr:.1%} for Caucasian defendants. "
                f"There is a {(aa_fpr/ca_fpr):.1f}x higher chance this is a wrongful prediction."
            )
        elif race_input == "Caucasian" and pred == 1:
            st.info(
                f"For reference: Caucasian defendants have a {ca_fpr:.1%} false positive rate "
                f"compared to {aa_fpr:.1%} for African-American defendants."
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Explainability (SHAP)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Explainability (SHAP)":

    st.markdown('<div class="main-title">Model Explainability</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">SHAP (SHapley Additive exPlanations) — '
        'why did the model make each prediction?</div>',
        unsafe_allow_html=True
    )

    if not SHAP_OK:
        st.error("Install SHAP to enable this page: `pip install shap`")
    else:
        with st.spinner("Computing SHAP values (this takes ~30 seconds on first load)..."):
            @st.cache_resource
            def get_shap(rf, X_test):
                explainer  = shap.TreeExplainer(rf)
                sv         = explainer.shap_values(X_test)
                if isinstance(sv, list):
                    sv = sv[1]
                elif hasattr(sv, "ndim") and sv.ndim == 3:
                    sv = sv[:, :, 1]
                ev = explainer.expected_value
                base_val = float(ev[1]) if hasattr(ev, "__len__") else float(ev)
                return sv, base_val

            sv, base_val = get_shap(rf, X_test)

        # ── Global importance ──────────────────────────────────────────────────
        st.markdown('<div class="section-hdr">Global feature importance</div>', unsafe_allow_html=True)
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            mean_shap = np.abs(sv).mean(axis=0)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sorted_idx = np.argsort(mean_shap)
            ax.barh([FEATURES[i] for i in sorted_idx],
                    mean_shap[sorted_idx],
                    color="#2563EB", alpha=0.85)
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("Feature importance — all predictions", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_s2:
            st.markdown("**What SHAP values mean:**")
            st.markdown("""
            - Each bar = how much that feature influences predictions **on average**
            - `priors_count` dominating means prior convictions drive most decisions
            - `race_code` having a non-zero value means race directly influences predictions
              **independently of other features** — this is the bias signal
            - A perfectly fair model would show `race_code` SHAP ≈ 0
            """)

        # ── SHAP by race ───────────────────────────────────────────────────────
        st.markdown('<div class="section-hdr">Mean SHAP by race group</div>', unsafe_allow_html=True)
        shap_df = pd.DataFrame(np.abs(sv), columns=FEATURES)
        shap_df["race"] = r_test

        mean_shap_race = shap_df.groupby("race")[FEATURES].mean()

        fig, ax = plt.subplots(figsize=(10, 3.5))
        mean_shap_race.T.plot(kind="bar", ax=ax, alpha=0.85)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Mean |SHAP|")
        ax.set_title("Feature influence by race group — do different features matter for different groups?",
                     fontsize=10)
        ax.set_xticklabels(FEATURES, rotation=30, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, loc="upper right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Waterfall for selected instance ───────────────────────────────────
        st.markdown('<div class="section-hdr">Explain a single prediction</div>', unsafe_allow_html=True)
        instance_type = st.radio(
            "Choose an instance to explain:",
            ["True Positive (correctly predicted recidivist)",
             "False Positive (wrongly flagged — the bias case)",
             "True Negative (correctly predicted non-recidivist)"],
            horizontal=True
        )

        y_pred_rf = rf.predict(X_test)
        if "True Positive" in instance_type:
            candidates = np.where((y_test_arr == 1) & (y_pred_rf == 1))[0]
        elif "False Positive" in instance_type:
            candidates = np.where((y_test_arr == 0) & (y_pred_rf == 1))[0]
        else:
            candidates = np.where((y_test_arr == 0) & (y_pred_rf == 0))[0]

        idx = candidates[0] if len(candidates) > 0 else 0

        single_exp = shap.Explanation(
            values        = sv[idx],
            base_values   = base_val,
            data          = X_test.iloc[idx].values,
            feature_names = FEATURES
        )

        fig_wf, ax_wf = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(single_exp, show=False)
        st.pyplot(plt.gcf())
        plt.close("all")

        st.caption(
            f"Instance {idx}: actual = {y_test_arr[idx]}, "
            f"predicted = {y_pred_rf[idx]}, "
            f"race = {r_test[idx]}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Remediation
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Remediation":

    st.markdown('<div class="main-title">Fairness Remediation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Can we fix the bias? At what cost?</div>',
        unsafe_allow_html=True
    )

    if not FL_OK:
        st.error("Install fairlearn: `pip install fairlearn`")
        st.stop()

    with st.spinner("Training fair model..."):
        @st.cache_resource
        def get_fair_model(rf, X_train, y_train, r_train, X_test, r_test):
            optimizer = ThresholdOptimizer(
                estimator      = rf,
                constraints    = "equalized_odds",
                predict_method = "predict_proba",
                objective      = "balanced_accuracy_score",
            )
            optimizer.fit(X_train, y_train, sensitive_features=r_train)
            y_fair = optimizer.predict(X_test, sensitive_features=r_test)
            return optimizer, y_fair

        optimizer, y_pred_fair = get_fair_model(rf, X_train, y_train, r_train, X_test, r_test)

    fairness_fair = group_metrics(y_test_arr, y_pred_fair, r_test)

    acc_orig = accuracy_score(y_test_arr, y_pred)
    acc_fair = accuracy_score(y_test_arr, y_pred_fair)
    eod_orig = equalized_odds_difference(y_test_arr, y_pred,      sensitive_features=r_test)
    eod_fair = equalized_odds_difference(y_test_arr, y_pred_fair, sensitive_features=r_test)
    reduction = (eod_orig - eod_fair) / eod_orig * 100

    # ── Summary ────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#2563EB">{acc_orig:.1%}</div><div class="metric-lbl">Original accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        delta_color = "#DC2626" if acc_fair < acc_orig else "#16A34A"
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{delta_color}">{acc_fair:.1%}</div><div class="metric-lbl">Fair model accuracy</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#DC2626">{eod_orig:.3f}</div><div class="metric-lbl">Original EOD gap</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#16A34A">{eod_fair:.3f}</div><div class="metric-lbl">Fair model EOD gap</div></div>', unsafe_allow_html=True)

    st.success(
        f"Fairness remediation reduced the Equalized Odds Gap by **{reduction:.1f}%** "
        f"({eod_orig:.3f} → {eod_fair:.3f}) at a cost of "
        f"**{(acc_orig - acc_fair):.1%}** accuracy. "
        f"This is the price of fairness — and it is worth paying."
    )

    # ── Before vs After FPR ───────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">FPR before vs after remediation</div>', unsafe_allow_html=True)

    groups = fairness_df.index.tolist()
    fprs_orig = fairness_df["FPR"].tolist()
    fprs_fair = [fairness_fair.loc[g, "FPR"] for g in groups]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(groups))
    w = 0.35
    bars1 = ax.bar(x - w/2, fprs_orig, width=w, color="coral",
                   alpha=0.9, label="Original model")
    bars2 = ax.bar(x + w/2, fprs_fair, width=w, color="steelblue",
                   alpha=0.9, label="Fair model")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("False Positive Rate Before vs After Fairness Remediation",
                 fontsize=12, fontweight="bold", color="#1B3A6B")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(list(bars1) + list(bars2), fprs_orig + fprs_fair):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Pareto curve ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">The accuracy–fairness tradeoff (Pareto curve)</div>', unsafe_allow_html=True)

    with st.spinner("Building Pareto curve..."):
        @st.cache_data
        def build_pareto(y_proba, y_test_arr, r_test):
            thresholds = np.linspace(0.1, 0.9, 30)
            accs, gaps = [], []
            for t in thresholds:
                yp = (y_proba >= t).astype(int)
                accs.append(accuracy_score(y_test_arr, yp))
                try:
                    gaps.append(equalized_odds_difference(
                        y_test_arr, yp, sensitive_features=r_test))
                except Exception:
                    gaps.append(np.nan)
            return thresholds, accs, gaps

        thresholds, pareto_acc, pareto_gap = build_pareto(y_proba, y_test_arr, r_test)

    valid = [(g, a) for g, a in zip(pareto_gap, pareto_acc) if not np.isnan(g)]
    if valid:
        vg, va = zip(*valid)
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        ax2.plot(vg, va, color="#2563EB", linewidth=2.5, label="Threshold sweep")
        ax2.scatter([eod_orig], [acc_orig], color="coral", s=120, zorder=5,
                    label=f"Original model (gap={eod_orig:.3f}, acc={acc_orig:.3f})")
        ax2.scatter([eod_fair], [acc_fair], color="#16A34A", s=120, zorder=5,
                    label=f"Fair model (gap={eod_fair:.3f}, acc={acc_fair:.3f})")
        ax2.set_xlabel("Equalized Odds Gap (lower = fairer)", fontsize=11)
        ax2.set_ylabel("Accuracy", fontsize=11)
        ax2.set_title(
            "Accuracy vs Fairness — Every Point is a Policy Choice\n"
            "Moving left = more fair, moving up = more accurate. You cannot do both simultaneously.",
            fontsize=10, color="#1B3A6B"
        )
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2)
        ax2.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Ethical conclusion ─────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">What this means</div>', unsafe_allow_html=True)
    st.markdown(f"""
    The Pareto curve shows something uncomfortable: **you cannot simultaneously maximize
    accuracy and minimize the fairness gap.** Every point on that curve is a policy decision
    — not a technical one.

    The original model prioritizes accuracy ({acc_orig:.1%}) but at the cost of a {eod_orig:.3f}
    equalized odds gap. The fair model accepts {(acc_orig-acc_fair):.1%} accuracy loss to
    reduce the gap to {eod_fair:.3f}.

    **Who makes this decision?**

    Not the algorithm. Not the data scientist. This is a question for:
    - Elected officials who decide how criminal justice tools are deployed
    - Judges who decide how much weight to give risk scores
    - Defendants whose liberty depends on these predictions

    > *"The question is not whether an algorithm is biased —
    > it's whether its biases are acceptable, to whom, and decided by what process."*

    This project doesn't answer that question. It makes sure you can see it clearly.
    """)
