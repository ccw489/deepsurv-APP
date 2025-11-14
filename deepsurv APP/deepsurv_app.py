# deepsurv_app.py
# DeepSurv online calculator with clinical-style UI
# - Times New Roman global font
# - Compact sidebar
# - Two-row layout (titles+subtitles, then cards)
# - English management recommendations + 600 DPI PNG export
# - SHAP text + plot：右侧“Feature contributions...”与左侧副标题同一高度

import os
import io
import json
import pickle
import base64
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import shap
import matplotlib.pyplot as plt

# ===========================
#   0. Path Settings
# ===========================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE_DIR = os.path.join(APP_DIR, "model_bundle")


# ===========================
#   1. CSS for compact style
# ===========================
def set_compact_style():
    st.markdown(
        """
        <style>

        /* ========== Global font: Times New Roman (paper style) ========== */
        html, body, [class*="css"], .stMarkdown, .stText, .stButton, 
        .stSelectbox, .stNumberInput, .stTextInput {
            font-family: "Times New Roman", "Times", serif !important;
        }

        /* ========== 1) Sidebar: ultra-compact layout ========== */
        section[data-testid="stSidebar"] .block-container {
            padding-top: 0.15rem !important;
            padding-bottom: 0.15rem !important;
        }

        section[data-testid="stSidebar"] .stSelectbox,
        section[data-testid="stSidebar"] .stNumberInput,
        section[data-testid="stSidebar"] .stTextInput,
        section[data-testid="stSidebar"] .stButton {
            margin-bottom: 0.03rem !important;
            padding-bottom: 0rem !important;
        }

        section[data-testid="stSidebar"] label {
            font-size: 0.72rem !important;
            margin-bottom: -0.30rem !important;
        }

        /* ========== 2) Main container padding (top aligned with sidebar) ========== */
        .block-container {
            padding-top: 1.0rem !important;
            padding-bottom: 0.6rem !important;
        }

        /* ========== 3) Center main content horizontally ========== */
        .main-center-container {
            max-width: 1100px;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        /* ========== 4) Titles & subheading ========== */
        h1, h2, h3 {
            margin-top: 0.1rem !important;
            margin-bottom: 0.2rem !important;
        }
        h1 {
            font-size: 1.30rem !important;
            color: #23395d;
        }

        .subheading {
            font-size: 0.90rem;
            color: #4a4a4a;
            margin-bottom: 0.45rem;
        }

        /* ========== 5) Card style: shadow + compact ========== */
        .card {
            background-color: #f9fafc;
            border-radius: 0.55rem;
            padding: 0.55rem 0.8rem;
            border: 1px solid #dfe3eb;
            margin-bottom: 0.45rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
        }
        .card-title {
            font-weight: 650;
            margin-bottom: 0.25rem;
            display: flex;
            align-items: center;
            gap: 0.30rem;
            color: #2d3e50;
            font-size: 0.90rem;
        }
        .card-title-icon {
            font-size: 1.0rem;
        }

        /* ========== 6) Risk color ========== */
        .risk-high {
            color: #c0392b;
            font-weight: 700;
        }
        .risk-low {
            color: #1e8449;
            font-weight: 700;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ===========================
#   2. DeepSurv model class
# ===========================
class DeepSurvNet(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout=0.0, activation="ReLU"):
        super().__init__()
        layers = []
        in_dim = input_dim

        if activation.lower() == "relu":
            act = nn.ReLU
        else:
            act = nn.ReLU

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def infer_arch_from_state_dict(state_dict):
    keys = [
        k for k in state_dict
        if k.startswith("net.") and k.endswith(".weight") and state_dict[k].dim() == 2
    ]
    if not keys:
        raise RuntimeError("Cannot infer architecture from state_dict.")

    def idx(k): return int(k.split(".")[1])
    keys = sorted(keys, key=idx)

    dims = [state_dict[k].shape for k in keys]  # (out, in)
    input_dim = int(dims[0][1])
    hidden_sizes = [int(d[0]) for d in dims[:-1]]
    return input_dim, hidden_sizes


# ===========================
#   3. Load model bundle
# ===========================
@st.cache_resource
def load_bundle(bundle_dir):
    art_path = os.path.join(bundle_dir, "artifacts.pkl")
    if not os.path.isfile(art_path):
        raise FileNotFoundError(f"artifacts.pkl not found in {bundle_dir}")

    with open(art_path, "rb") as f:
        artifacts = pickle.load(f)

    feature_names = artifacts["feature_names"]
    scaler = artifacts["scaler"]
    background_X = artifacts["background_X"]
    train_stats = artifacts.get("train_scores_stats", {})

    # config
    activation = "ReLU"
    dropout = 0.0
    cfg_path = os.path.join(bundle_dir, "model_config.json")
    if os.path.isfile(cfg_path):
        cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
        activation = cfg.get("activation", "ReLU")
        dropout = cfg.get("dropout", 0.0)

    # state dict
    ckpt = torch.load(os.path.join(bundle_dir, "deepsurv_state.pt"), map_location="cpu")
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    input_dim, hidden_sizes = infer_arch_from_state_dict(sd)

    model = DeepSurvNet(input_dim, hidden_sizes, dropout, activation)
    model.load_state_dict(sd)
    model.eval()

    def predict_log_risk(df):
        X = df[feature_names].values.astype(np.float32)
        Xs = scaler.transform(X)
        x = torch.tensor(Xs, dtype=torch.float32)
        with torch.no_grad():
            log_risk = model(x).numpy()
        return log_risk

    background_np = background_X[feature_names].values

    return {
        "feature_names": feature_names,
        "predict_log_risk": predict_log_risk,
        "background_np": background_np,
        "background_X": background_X,
        "train_stats": train_stats,
    }


# ===========================
#   4. SHAP explainer
# ===========================
def get_shap_explainer(bundle):
    feat = bundle["feature_names"]
    pred = bundle["predict_log_risk"]
    bg = bundle["background_np"]

    def pred_fn(x):
        return pred(pd.DataFrame(x, columns=feat))

    explainer = shap.KernelExplainer(pred_fn, bg)
    return explainer


# ===========================
#   5. Risk classification + EN suggestions
# ===========================
LOW_RISK_EN_MD = """**Low-risk patient management recommendations:**
- Routine follow-up and monitoring  
- Maintain current treatment strategy if appropriate  
- Pay attention to quality of life and psychological support  
- Regular imaging and tumor marker assessments  
"""

HIGH_RISK_EN_MD = """**High-risk patient management recommendations:**
- Prompt multidisciplinary team (MDT) discussion  
- Consider treatment escalation or regimen modification  
- Close monitoring of treatment-related adverse events  
- Enhanced supportive and symptomatic care  
- Consider eligibility for clinical trials  
"""

LOW_RISK_EN_TEXT = (
    "Low-risk patient management recommendations:\n"
    "- Routine follow-up and monitoring\n"
    "- Maintain current treatment strategy if appropriate\n"
    "- Pay attention to quality of life and psychological support\n"
    "- Regular imaging and tumor marker assessments\n"
)

HIGH_RISK_EN_TEXT = (
    "High-risk patient management recommendations:\n"
    "- Prompt multidisciplinary team (MDT) discussion\n"
    "- Consider treatment escalation or regimen modification\n"
    "- Close monitoring of treatment-related adverse events\n"
    "- Enhanced supportive and symptomatic care\n"
    "- Consider eligibility for clinical trials\n"
)


def classify_risk(log_risk, train_stats):
    med = train_stats.get("percentiles", {}).get(50, 0.0)

    if log_risk >= med:
        label = "High-risk"
        desc_md = (
            "This patient is classified as **High-risk**, with a predicted risk above the median "
            "of the training cohort.\n\n" + HIGH_RISK_EN_MD
        )
        css = "risk-high"
        desc_plain = (
            "This patient is classified as High-risk, with a predicted risk above the median of "
            "the training cohort.\n\n" + HIGH_RISK_EN_TEXT
        )
    else:
        label = "Low-risk"
        desc_md = (
            "This patient is classified as **Low-risk**, with a predicted risk below the median "
            "of the training cohort.\n\n" + LOW_RISK_EN_MD
        )
        css = "risk-low"
        desc_plain = (
            "This patient is classified as Low-risk, with a predicted risk below the median of "
            "the training cohort.\n\n" + LOW_RISK_EN_TEXT
        )

    return label, desc_md, css, desc_plain


# ===========================
#   6. Feature mappings / ranges
# ===========================
CONTINUOUS_RANGES = {
    "age": (0, 100),
    "CA50": (0, 1500),
    "diameter": (0, 50),
    "Lymph_count": (0, 100),
}

binary_features_mapping: Dict[str, Dict[str, int]] = {
    "Vascular": {"No": 0, "Yes": 1},
    "Nerve": {"No": 0, "Yes": 1},
    "chemotherapy": {"No": 0, "Yes": 1},
    "targeted_therpy": {"No": 0, "Yes": 1},
    "targeted_therapy": {"No": 0, "Yes": 1},
}

multi_cat_features_mapping: Dict[str, Dict[str, int]] = {
    "T": {"T1": 1, "T2": 2, "T3": 3, "T4": 4},
    "N": {"N0": 0, "N1": 1, "N2": 2, "N3": 3},
}



# ===========================
#   7. Export 600 DPI figure
def make_export_figure(expl, risk_label: str, risk_plain_text: str) -> plt.Figure:
    """
    Create a 600 DPI figure with vertical layout:
    Title at top, SHAP plot in middle, risk description at bottom
    """
    plt.close("all")

    # 1. 调整整体画布尺寸 - 更高以适应上下布局
    fig = plt.figure(figsize=(12, 10), dpi=600)

    # 2. 创建上下三部分：标题、SHAP图、文字描述
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 6.0, 2.0])  # 上:中:下 = 0.8:6.0:2.0
    fig.subplots_adjust(hspace=0.4, left=0.08, right=0.95, top=0.94, bottom=0.06)

    # Top: 标题
    ax_title = fig.add_subplot(gs[0, 0])
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.5,
        "DeepSurv-based risk explanation for current patient",
        fontsize=14,
        fontweight="bold",
        fontfamily="Times New Roman",
        transform=ax_title.transAxes,
        va="center",
        ha="center"
    )

    # Middle: SHAP waterfall - 占据主要空间
    ax_shap = fig.add_subplot(gs[1, 0])
    plt.sca(ax_shap)

    # 设置SHAP图内部参数
    plt.rcParams['font.size'] = 10
    shap.plots.waterfall(expl, max_display=12, show=False)
    ax_shap.set_title("SHAP waterfall plot", fontsize=13, fontfamily="Times New Roman", pad=15)

    # Bottom: 风险描述
    ax_desc = fig.add_subplot(gs[2, 0])
    ax_desc.axis("off")

    # 风险组
    ax_desc.text(
        0.02, 0.85,
        f"Risk group: {risk_label}",
        fontsize=12,
        fontweight="bold",
        fontfamily="Times New Roman",
        transform=ax_desc.transAxes,
        va="top",
    )

    # 风险描述
    ax_desc.text(
        0.02, 0.65,
        risk_plain_text,
        fontsize=10,
        fontfamily="Times New Roman",
        transform=ax_desc.transAxes,
        va="top",
        wrap=True,
        linespacing=1.2
    )

    return fig

# ===========================
#   8. Streamlit App
# ===========================
def main():
    st.set_page_config(page_title="DeepSurv Calculator", layout="wide", page_icon="📊")
    set_compact_style()

    # ---------- Load model ----------
    try:
        bundle = load_bundle(DEFAULT_BUNDLE_DIR)
    except Exception as e:
        st.error(f"Model loading error:\n{e}")
        return

    feature_names = bundle["feature_names"]
    train_stats = bundle["train_stats"]
    predict_log_risk = bundle["predict_log_risk"]
    background_X = bundle["background_X"]

    # Defaults from median of background
    defaults = {}
    for col in feature_names:
        if col in background_X.columns:
            try:
                defaults[col] = float(background_X[col].median())
            except Exception:
                defaults[col] = 0.0
        else:
            defaults[col] = 0.0

    # ---------- Sidebar: patient features ----------
    st.sidebar.header("🧬 Patient Features")

    user_input: Dict[str, Any] = {}

    for feat in feature_names:
        # 1) Continuous
        if feat in CONTINUOUS_RANGES:
            vmin, vmax = CONTINUOUS_RANGES[feat]
            default_val = defaults.get(feat, (vmin + vmax) / 2.0)
            default_val = float(min(max(default_val, vmin), vmax))

            if feat in ["age", "Lymph_count"]:
                user_input[feat] = st.sidebar.number_input(
                    feat,
                    min_value=int(vmin),
                    max_value=int(vmax),
                    value=int(round(default_val)),
                    step=1,
                    key=f"feat_{feat}",
                )
            else:
                user_input[feat] = st.sidebar.number_input(
                    feat,
                    min_value=float(vmin),
                    max_value=float(vmax),
                    value=default_val,
                    step=0.1,
                    key=f"feat_{feat}",
                )

        # 2) Binary (Yes/No)
        elif feat in binary_features_mapping:
            mapping = binary_features_mapping[feat]
            options = list(mapping.keys())  # ["No", "Yes"]
            default_code = int(round(defaults.get(feat, 0.0)))
            default_label = options[0]
            for k, v in mapping.items():
                if v == default_code:
                    default_label = k
                    break
            idx = options.index(default_label)
            chosen = st.sidebar.selectbox(
                f"{feat} (binary)",
                options=options,
                index=idx,
                key=f"feat_{feat}",
            )
            user_input[feat] = mapping[chosen]

        # 3) Multi-category T/N
        elif feat in multi_cat_features_mapping:
            mapping = multi_cat_features_mapping[feat]
            options = list(mapping.keys())  # ["T1", "T2", ...]
            default_code = int(round(defaults.get(feat, 0.0)))
            default_label = options[0]
            for k, v in mapping.items():
                if v == default_code:
                    default_label = k
                    break
            idx = options.index(default_label)
            chosen = st.sidebar.selectbox(
                f"{feat} (stage)",
                options=options,
                index=idx,
                key=f"feat_{feat}",
            )
            user_input[feat] = mapping[chosen]

        # 4) Others
        else:
            default_val = defaults.get(feat, 0.0)
            user_input[feat] = st.sidebar.number_input(
                feat,
                value=float(default_val),
                key=f"feat_{feat}",
            )

    run = st.sidebar.button("▶ Run Prediction")

    # ========== Main area (right side) ==========
    st.markdown('<div class="main-center-container">', unsafe_allow_html=True)

    # 第一行：标题+副标题（左右对齐）
    title_left, title_right = st.columns([1.2, 1.0])

    # 左：主标题 + 副标题
    title_left.markdown("### 📊 DeepSurv Online Risk Calculator")
    title_left.markdown(
        '<p class="subheading">A calculator for predicting the risk of postoperative HER2 expression in gastric cancer patients.</p>',
        unsafe_allow_html=True,
    )

    # 右：SHAP 标题 + “Feature contributions...” （与左侧副标题同一高度）
    title_right.markdown("### 📉 SHAP waterfall plot")
    title_right.markdown(
        '<p class="subheading">Feature contributions to the predicted risk for this patient.</p>',
        unsafe_allow_html=True,
    )

    # 第二行：左卡片 + 右图
    left_col, right_col = st.columns([1.2, 1.0])

    left_col.markdown(
        """
        <div class="card">
            <div class="card-title">
                <span class="card-title-icon">📘</span>
                <span>Model description</span>
            </div>
            <div>
                This calculator uses a DeepSurv neural network trained on the study cohort. 
                The model outputs an individualized risk score and classifies patients into 
                high- or low-risk groups using the median risk as the cut-off.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col.markdown(
        """
        <div class="card">
            <div class="card-title">
                <span class="card-title-icon">🧭</span>
                <span>How to use</span>
            </div>
            <ul style="margin-bottom: 0;">
                <li>Enter patient-specific clinicopathological features in the left sidebar.</li>
                <li>Click <b>“Run Prediction”</b>.</li>
                <li>Inspect the risk group and the SHAP waterfall plot for feature-level interpretation.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    export_buffer = None  # will hold PNG bytes if prediction is run

    # ---------- Prediction & SHAP ----------
    if run:
        df = pd.DataFrame([user_input], columns=feature_names)
        log_risk = float(predict_log_risk(df)[0])

        risk_label, risk_desc_md, risk_css, risk_plain = classify_risk(log_risk, train_stats)

        # Risk card
        risk_html_text = risk_desc_md.replace("\n", "<br>")

        left_col.markdown(
            f"""
            <div class="card">
                <div class="card-title">
                    <span class="card-title-icon">⚠️</span>
                    <span>Risk classification</span>
                </div>
                <p class="{risk_css}" style="margin-bottom:0.25rem;">{risk_label}</p>
                <p style="font-size:0.90rem; line-height:1.3; margin-bottom:0;">
                    {risk_html_text}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # SHAP explainer
        if "explainer" not in st.session_state:
            st.session_state["explainer"] = get_shap_explainer(bundle)
        explainer = st.session_state["explainer"]

        x_np = df[feature_names].values
        shap_vals = explainer.shap_values(x_np)
        shap_vals = np.array(shap_vals).reshape(1, -1)

        expl = shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=x_np[0],
            feature_names=feature_names,
        )

        # ---- On-screen SHAP：右侧卡片内只放图，标题行在上面已对齐 ----
        plt.close("all")
        fig_inline = plt.figure(figsize=(6, 3.2))  # 比默认稍宽
        shap.plots.waterfall(expl, max_display=10, show=False)
        plt.tight_layout()

        buf_inline = io.BytesIO()
        fig_inline.savefig(buf_inline, format="png", dpi=200, bbox_inches="tight")
        buf_inline.seek(0)
        img_b64 = base64.b64encode(buf_inline.read()).decode("ascii")
        plt.close(fig_inline)

        right_col.markdown(
            f"""
            <div class="card">
              <img src="data:image/png;base64,{img_b64}" style="width:100%; display:block; margin:0 auto;"/>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- Prepare 600 DPI export：右边 waterfall 显著加宽 ----
        fig_export = make_export_figure(expl, risk_label, risk_plain)
        buf = io.BytesIO()
        fig_export.savefig(buf, format="png", dpi=600, bbox_inches="tight")
        buf.seek(0)
        export_buffer = buf

    # ---------- Export button ----------
    if run and export_buffer is not None:
        right_col.download_button(
            label="💾 Export current patient figure (600 DPI PNG)",
            data=export_buffer,
            file_name="deepsurv_patient_explanation.png",
            mime="image/png",
        )

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
