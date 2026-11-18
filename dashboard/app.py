"""
Streamlit AML Investigation, Banking & Detection Dashboard.
Integrates SQLite Database, loads PR-AUC calibrated ML models (XGBoost, GraphSAGE, and Combined XGBoost + GraphSAGE),
and features a Core Banking System with Automated AML Compliance Interception (>=50% risk blocking).
"""
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
for _mod_name in [
    "config",
    "src.database.db_manager",
    "src.models.xgboost_model",
    "src.models.graphsage_model",
    "src.models.combined_model",
    "src.banking.banking_service",
]:
    if _mod_name in sys.modules:
        importlib.reload(sys.modules[_mod_name])

import config
from config import DB_PATH, SAVED_MODELS_DIR, FEATURE_COLUMNS
FEATURE_METADATA = getattr(config, "FEATURE_METADATA", {})
format_feature_name = getattr(config, "format_feature_name", lambda x: FEATURE_METADATA.get(x, {}).get("label", x.replace("_", " ").title()))

from src.database.db_manager import AMLDatabase
from src.models.xgboost_model import AMLXGBoost
from src.models.graphsage_model import AMLGraphSAGE
from src.models.combined_model import AMLCombinedXGBoostGraphSAGE
from src.banking.banking_service import BankingService

# Page configuration
st.set_page_config(
    page_title="AML Detection & Banking System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #334155;
        color: white;
        margin-bottom: 12px;
    }
    .metric-title { font-size: 0.82rem; color: #94a3b8; margin-bottom: 4px; }
    .metric-value { font-size: 1.45rem; font-weight: 700; color: #38bdf8; }
    .tag-suspicious {
        background-color: #ef4444; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold;
    }
    .tag-legit {
        background-color: #10b981; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold;
    }
    .tag-blocked {
        background-color: #dc2626; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.85rem;
    }
    .tag-active {
        background-color: #059669; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.85rem;
    }
    .account-preview-card {
        background-color: #1e293b;
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 14px;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    .intercept-box {
        background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 18px;
        color: white;
        margin-top: 12px;
        margin-bottom: 16px;
    }
    .approved-box {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 18px;
        color: white;
        margin-top: 12px;
        margin-bottom: 16px;
    }
    .critical-alert-box {
        background: linear-gradient(135deg, #881337 0%, #4c0519 100%);
        border: 2px solid #f43f5e;
        border-radius: 10px;
        padding: 20px;
        color: white;
        margin-top: 14px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(244, 63, 94, 0.35);
    }
    /* Responsive metric typography to prevent number truncation */
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        line-height: 1.25 !important;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 1.35rem !important;
        white-space: normal !important;
        word-break: break-word !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #94a3b8 !important;
        white-space: normal !important;
        line-height: 1.2 !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
    }
    div[data-testid="stDataFrame"] {
        font-size: 0.88rem !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_db():
    return AMLDatabase(DB_PATH)

@st.cache_resource
def load_all_models(_cache_buster: float = 0.0):
    models = {}
    xgb_path = SAVED_MODELS_DIR / "xgboost.joblib"
    if xgb_path.exists():
        models["XGBoost"] = AMLXGBoost.load(xgb_path)

    sage_path = SAVED_MODELS_DIR / "graphsage.pt"
    if sage_path.exists():
        models["GraphSAGE"] = AMLGraphSAGE.load(sage_path)

    comb_path = SAVED_MODELS_DIR / "combined_model.joblib"
    if comb_path.exists():
        models["Combined (XGBoost + GraphSAGE)"] = AMLCombinedXGBoostGraphSAGE.load(comb_path)
    return models

db = get_db()
_comb_file = SAVED_MODELS_DIR / "combined_model.joblib"
_mtime = _comb_file.stat().st_mtime if _comb_file.exists() else 0.0
models = load_all_models(_mtime)

# Initialize Banking Service directly (not cached, uses cached db and models)
banking = BankingService(db, models)

def reset_demo_account(account_id: str, balance: float = 10000.0):
    """Safely reset a demo account's balance and unblock status."""
    if hasattr(banking, "reset_demo_account"):
        try:
            banking.reset_demo_account(account_id, balance)
            return
        except Exception:
            pass
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE accounts SET is_blocked = 0, block_reason = '', balance = ? WHERE account_id = ?", (float(balance), account_id))
        conn.commit()

def reset_all_demo_data(balance: float = 10000.0):
    """Reset all accounts, balances, and demo transfers back to their original state."""
    if hasattr(banking, "reset_all_demo_data"):
        banking.reset_all_demo_data(balance)
    elif hasattr(db, "reset_all_demo_data"):
        db.reset_all_demo_data(balance)
    else:
        for acc_id in ["800042CB0", "8000FFF70", "1004286A8", "800045000"]:
            reset_demo_account(acc_id, balance)

@st.cache_data
def get_banking_dropdown_catalog(_cache_buster: float = 0.0):
    """Load all 206,239 accounts formatted for fast dropdown selection."""
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT account_id,
                   account_id || ' | ' || 
                   CASE WHEN is_suspicious = 1 THEN '🚨 Laundering' ELSE '🟢 Legitimate' END || 
                   ' (' || CAST(ROUND(risk_score * 100, 1) AS TEXT) || '% Risk) | ' || 
                   COALESCE(pattern_type, 'NONE') || ' | Txs: ' || CAST(CAST(total_tx_count AS INT) AS TEXT),
                   is_suspicious
            FROM accounts
            ORDER BY is_suspicious DESC, total_tx_count DESC, risk_score DESC
        """)
        rows = cursor.fetchall()
    laund_opts = [r[1] for r in rows if r[2] == 1]
    legit_opts = [r[1] for r in rows if r[2] == 0]
    all_sender_opts = laund_opts + legit_opts
    all_recipient_opts = legit_opts + laund_opts
    id_to_label = {r[0]: r[1] for r in rows}
    return {
        "all_sender": all_sender_opts,
        "all_recipient": all_recipient_opts,
        "laundering": laund_opts,
        "legitimate": legit_opts,
        "id_to_label": id_to_label
    }

def format_compact_currency(amount: float) -> str:
    """Format large currency values cleanly ($86.37B, $4.00M, $1.5K) to prevent UI truncation."""
    val = float(amount or 0.0)
    abs_val = abs(val)
    if abs_val >= 1e9:
        return f"${val / 1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"${val / 1e6:.2f}M"
    elif abs_val >= 1e3:
        return f"${val / 1e3:.1f}K"
    else:
        return f"${val:,.2f}"

# Cached EDA Data Fetchers (cache-busted by model/DB update timestamp)
@st.cache_data
def get_eda_overview_stats(_cache_buster: float = 0.0):
    with db.get_connection() as conn:
        acc_df = pd.read_sql("SELECT is_suspicious, COUNT(*) as count FROM accounts GROUP BY is_suspicious", conn)
        tx_df = pd.read_sql("SELECT is_laundering, COUNT(*) as count FROM transactions GROUP BY is_laundering", conn)
    return acc_df, tx_df

@st.cache_data
def get_eda_amount_stats(_cache_buster: float = 0.0):
    with db.get_connection() as conn:
        df = pd.read_sql("SELECT is_laundering, amount_paid FROM transactions", conn)
    return df

@st.cache_data
def get_eda_payment_stats(_cache_buster: float = 0.0):
    with db.get_connection() as conn:
        df = pd.read_sql("""
            SELECT payment_format, 
                   is_laundering, 
                   COUNT(*) as count, 
                   SUM(amount_paid) as volume 
            FROM transactions 
            GROUP BY payment_format, is_laundering
        """, conn)
    return df

@st.cache_data
def get_eda_typologies(_cache_buster: float = 0.0):
    with db.get_connection() as conn:
        df = pd.read_sql("""
            SELECT pattern_type, 
                   COUNT(*) as tx_count, 
                   SUM(amount_paid) as total_volume, 
                   AVG(amount_paid) as avg_volume,
                   MIN(amount_paid) as min_volume,
                   MAX(amount_paid) as max_volume
            FROM transactions 
            WHERE is_laundering = 1 AND pattern_type != 'NONE'
            GROUP BY pattern_type 
            ORDER BY tx_count DESC
        """, conn)
    return df

@st.cache_data
def get_eda_correlations(_cache_buster: float = 0.0):
    with db.get_connection() as conn:
        df = pd.read_sql("""
            SELECT in_degree, out_degree, degree_centrality, total_tx_in, total_tx_out,
                   avg_amount_in, avg_amount_out, unique_senders, unique_receivers,
                   ach_tx_count, wire_tx_count, in_out_tx_ratio, is_suspicious
            FROM accounts
        """, conn)
    return df

# Sidebar
st.sidebar.title("🛡️ AML & Banking System")
st.sidebar.caption("Financial Crime Surveillance & Core Ledger Platform")

menu = st.sidebar.radio(
    "Navigation",
    [
        "📊 System Overview",
        "🔬 Exploratory Data Analysis (EDA)",
        "🏦 Core Banking & Transfers",
        "🔍 Suspicious Accounts",
        "💳 Transaction Explorer",
        "🤖 Live ML Inference",
        "📈 Model Benchmarks",
        "🧠 Explainability & Typologies"
    ]
)

# ----------------------------------------------------
# 1. SYSTEM OVERVIEW
# ----------------------------------------------------
if menu == "📊 System Overview":
    st.title("📊 System Overview & Operational Metrics")
    st.write("Real-time summary of core accounts, banking volume, AML risk surveillance, and model performance.")

    stats = db.get_summary_stats()
    b_stats = db.get_banking_stats()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Accounts</div>
            <div class="metric-value">{stats['total_accounts']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Suspicious Accounts</div>
            <div class="metric-value" style="color: #f87171;">{stats['suspicious_accounts']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">AML Blocked Accounts</div>
            <div class="metric-value" style="color: #ef4444;">{stats.get('blocked_accounts', 0):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Indexed Transactions</div>
            <div class="metric-value">{stats['total_transactions']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Verified Laundering Txs</div>
            <div class="metric-value" style="color: #f87171;">{stats['laundering_transactions']:,}</div>
        </div>
        """, unsafe_allow_html=True)

    # Banking System Performance Row
    st.subheader("🏦 Live Banking Activity & AML Gatekeeper Stats")
    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        st.metric("Total Interbank Transfers", f"{b_stats['total_transfers']:,}")
    with bc2:
        st.metric("Approved & Settled Transfers", f"{b_stats['approved_transfers']:,}")
    with bc3:
        st.metric("AML Blocked Transfers", f"{b_stats['blocked_transfers']:,}", delta=f"{b_stats['blocked_transfers']} blocked", delta_color="inverse")
    with bc4:
        st.metric("Total Settled Volume", f"${b_stats['total_volume']:,.2f}")

    st.divider()
    st.subheader("Model Performance Summary")
    metrics_df = db.get_model_metrics()
    if not metrics_df.empty:
        metrics_overview = metrics_df.sort_values(by="pr_auc", ascending=False).reset_index(drop=True)
        st.dataframe(metrics_overview[["model_name", "roc_auc", "pr_auc"]], use_container_width=True)
    
    st.subheader("Laundering Pattern Distribution (6,361 Suspicious Accounts)")
    with db.get_connection() as conn:
        pattern_df = pd.read_sql(
            "SELECT pattern_type, COUNT(*) as count FROM accounts WHERE is_suspicious = 1 AND pattern_type != 'NONE' GROUP BY pattern_type ORDER BY count DESC",
            conn
        )
    if not pattern_df.empty:
        col_pat_chart, col_pat_tbl = st.columns([3, 2])
        with col_pat_chart:
            st.bar_chart(pattern_df.set_index("pattern_type")["count"])
        with col_pat_tbl:
            pat_tbl = pattern_df.copy()
            pat_tbl["Share (%)"] = (pat_tbl["count"] / pat_tbl["count"].sum() * 100.0).map(lambda x: f"{x:.1f}%")
            pat_tbl.rename(columns={"pattern_type": "Laundering Typology", "count": "Suspicious Accounts"}, inplace=True)
            st.dataframe(pat_tbl, use_container_width=True, hide_index=True)


# ----------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------
elif menu == "🔬 Exploratory Data Analysis (EDA)":
    st.title("🔬 Exploratory Data Analysis (AML Forensic Telemetry)")
    st.markdown(
        "Empirical investigative exploration across **206,242 accounts** and multi-million transaction graphs. "
        "This suite explains **ROC-AUC and PR-AUC** (how they work, what they calculate, and how they are calculated), "
        "details our **5-Stage Skewed-Data Balancing Architecture**, "
        "uncovers the **sub-\\$10,000 structuring ('smurfing') signature**, analyzes payment channel risk concentration, "
        "catalogs 8 distinct network laundering typologies, and maps graph topological correlations."
    )

    eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
        "⚖️ Class Imbalance, ROC-AUC & PR-AUC",
        "💵 Financial Distribution & Smurfing Patterns",
        "💳 Payment Channels & Velocity",
        "🕸️ Typology Taxonomy & Ring Behavior",
        "🔗 Graph Topology & Feature Correlations"
    ])

    # ----------------------------------------------------
    # TAB 1: CLASS IMBALANCE, ROC-AUC & PR-AUC
    # ----------------------------------------------------
    with eda_tab1:
        st.subheader("⚖️ Extreme Class Imbalance & Why Point-Threshold Metrics Are Excluded")
        st.markdown(
            "In commercial banking surveillance, financial crime is a rare-event anomaly. "
            "Across real-world interbank settlement systems, illicit flows constitute **0.1% or less** of all transfers (and **3.08%** at the account level). "
            "Single-threshold metrics like raw **Accuracy** are fatally deceptive in this regime, which is why our system evaluates models strictly using "
            "threshold-independent ranking curves: **ROC-AUC** and **PR-AUC**."
        )

        acc_counts, tx_counts = get_eda_overview_stats(_mtime)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Surveillance Accounts", "206,242")
        with c2:
            st.metric("Suspicious Accounts", "6,361", delta="3.08% of total")
        with c3:
            st.metric("Full Dataset Laundering Rate", "0.103%", delta="~1 in 970 txs")
        with c4:
            st.metric("Sampled Laundering Rate", "9.38%", delta="5,177 / 55,193 txs")

        st.markdown("---")
        
        col_chart, col_math = st.columns([1, 1])
        with col_chart:
            st.markdown("#### **Class Distribution Comparison**")
            acc_data = pd.DataFrame({
                "Entity": ["Legitimate Accounts (96.92%)", "Suspicious Accounts (3.08%)"],
                "Count": [199881, 6361]
            }).set_index("Entity")
            st.bar_chart(acc_data)
            st.caption("Distribution of labeled accounts across the surveillance graph.")

            tx_data = pd.DataFrame({
                "Transaction Class": ["Legitimate Transfers (90.62%)", "Laundering Transfers (9.38%)"],
                "Count": [50016, 5177]
            }).set_index("Transaction Class")
            st.bar_chart(tx_data)
            st.caption("Sampled transactions in the active operational database.")

        with col_math:
            st.markdown("""
            <div class="critical-alert-box">
                <h4 style="margin-top:0; color:#fecdd3;">⚠️ The "Accuracy Fallacy" in AML Surveillance</h4>
                <p>Consider a naive classifier that predicts <b>EVERY</b> transaction as <code>Legitimate (0)</code>:</p>
                <ul>
                    <li><b>Raw Accuracy:</b> <code>99.897%</code> (Appears mathematically "near perfect")</li>
                    <li><b>Laundering Recall:</b> <code>0.00%</code> (100% of money laundering escapes undetected)</li>
                    <li><b>PR-AUC:</b> <code>0.0010</code> (Immediately exposes the model as useless)</li>
                </ul>
                <p>Because single-threshold metrics (Accuracy, Precision, Recall, F1) depend on an arbitrary cutoff and accuracy is dominated by True Negatives ($TN$), we evaluate all models strictly using <b>ROC-AUC</b> and <b>PR-AUC</b>.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### **Summary: ROC-AUC vs. PR-AUC in AML**")
            st.markdown("""
            | Metric | What It Measures | Formula / Axes | Role in This AML System |
            | :--- | :--- | :--- | :--- |
            | **ROC-AUC** | Global ranking separation between all suspicious and legitimate accounts | Area under **TPR (Recall)** vs. **FPR** | ✅ **Global Discrimination Metric** |
            | **PR-AUC** | Positive-class detection purity vs. coverage across all thresholds (ignores $TN$) | Area under **Precision** vs. **Recall** | 🏆 **Primary Decision & Calibration Metric** |
            """)

        st.markdown("---")
        st.subheader("📐 Deep Dive: How ROC-AUC and PR-AUC Work, What They Calculate & How They Are Calculated")

        st.markdown("""
        Every ML model in this platform (**XGBoost**, **GraphSAGE**, and **Combined**) outputs a continuous laundering probability $\\hat{p}_i \\in [0, 1]$ for each account $i$.
        For any candidate decision threshold $t \\in [0, 1]$, accounts with $\\hat{p}_i \\ge t$ are flagged as **Suspicious ($1$)**, and accounts with $\\hat{p}_i < t$ are cleared as **Legitimate ($0$)**.
        At threshold $t$, predictions partition into four confusion-matrix counts:
        - **True Positives ($TP(t)$):** Confirmed launderers correctly flagged ($\\hat{p}_i \\ge t$ and $y_i = 1$).
        - **False Positives ($FP(t)$):** Legitimate accounts falsely flagged ($\\hat{p}_i \\ge t$ and $y_i = 0$).
        - **True Negatives ($TN(t)$):** Legitimate accounts correctly cleared ($\\hat{p}_i < t$ and $y_i = 0$).
        - **False Negatives ($FN(t)$):** Confirmed launderers missed by the model ($\\hat{p}_i < t$ and $y_i = 1$).
        """)

        col_roc, col_pr = st.columns(2)
        with col_roc:
            st.markdown("### 📈 1. ROC-AUC (Receiver Operating Characteristic AUC)")
            st.markdown("""
            **What It Calculates:**
            - ROC-AUC measures the probability that a randomly chosen **money-laundering account** ($y=1$) is assigned a strictly higher risk score $\\hat{p}$ than a randomly chosen **legitimate account** ($y=0$):
            $$\\text{ROC-AUC} = P(\\hat{p}_{\\text{laundering}} > \\hat{p}_{\\text{legitimate}})$$

            **How It Works:**
            - By sweeping the threshold $t$ continuously from $1.0$ down to $0.0$, we plot the **True Positive Rate (TPR / Sensitivity)** on the vertical axis against the **False Positive Rate (FPR / Fall-out)** on the horizontal axis:
            $$\\text{TPR}(t) = \\frac{TP(t)}{TP(t) + FN(t)}, \\qquad \\text{FPR}(t) = \\frac{FP(t)}{FP(t) + TN(t)}$$

            **How It Is Calculated Mathematically:**
            - Sorting all $K$ unique probability thresholds so that $0 = \\text{FPR}_0 \\le \\text{FPR}_1 \\le \\dots \\le \\text{FPR}_K = 1$, the area under the ROC curve is computed via trapezoidal numerical integration:
            $$\\text{ROC-AUC} = \\int_{0}^{1} \\text{TPR}\\!\\left(\\text{FPR}^{-1}(x)\\right) dx \\approx \\sum_{k=1}^{K} \\left(\\text{FPR}_k - \\text{FPR}_{k-1}\\right) \\cdot \\frac{\\text{TPR}_k + \\text{TPR}_{k-1}}{2}$$
            - **Why ROC-AUC Alone Is Not Enough in AML:** Because $TN$ is massive ($199,881 legitimate accounts), the denominator $FP + TN$ makes $\\text{FPR}$ look tiny even when thousands of legitimate customers are falsely blocked. That is why we pair ROC-AUC with **PR-AUC**.
            """)

        with col_pr:
            st.markdown("### 🎯 2. PR-AUC (Precision-Recall AUC / Average Precision)")
            st.markdown("""
            **What It Calculates:**
            - PR-AUC measures the trade-off between **alert purity (Precision)** and **criminal capture rate (Recall)** across all possible thresholds, **completely excluding True Negatives ($TN$)** from the calculation.

            **How It Works:**
            - As threshold $t$ sweeps from $1.0$ down to $0.0$, we plot **Precision** on the vertical axis against **Recall** on the horizontal axis:
            $$\\text{Precision}(t) = \\frac{TP(t)}{TP(t) + FP(t)}, \\qquad \\text{Recall}(t) = \\frac{TP(t)}{TP(t) + FN(t)}$$
            - Notice that $TN(t)$ appears **nowhere** in either formula. Every false alarm ($FP$) directly reduces Precision, and every missed launderer ($FN$) directly reduces Recall.

            **How It Is Calculated Mathematically:**
            - Ordered across all ranked thresholds $n = 1, \\dots, N$ from highest probability to lowest, PR-AUC (Average Precision) is calculated as the weighted sum of precisions at each threshold, weighted by the incremental gain in recall:
            $$\\text{PR-AUC} = \\int_{0}^{1} P(R)\\,dR = \\sum_{n=1}^{N} \\left(\\text{Recall}_n - \\text{Recall}_{n-1}\\right) \\times \\text{Precision}_n$$
            - A random guesser achieves a PR-AUC equal only to the base positive rate ($0.0308$ on accounts, $0.0010$ on transactions), whereas our **Combined (XGBoost + GraphSAGE)** model achieves **>0.88 PR-AUC**.
            """)

        st.info(
            "🎯 **How PR-AUC Decides Which Account Is Laundering vs. Legitimate:**\n"
            "1. **Training Objective:** XGBoost (`eval_metric='aucpr'`) and GraphSAGE are trained and selected to maximize area under the Precision-Recall curve.\n"
            "2. **PR-Curve Threshold Selection & Probability Calibration:** For each model, we compute the full Precision-Recall curve on validation logits $z(x)$ to find the optimal threshold $z^*$ that maximizes the precision-recall balance. We then apply a monotonic logit calibration $P_{\\text{cal}}(x) = \\sigma((z(x) - b)/T)$ so that $z^*$ maps directly to the **50.0% (`0.50`) operational blocking boundary** while preserving 100% of ROC-AUC and PR-AUC.\n"
            "3. **Balanced Ensemble Decision:** Because both `XGBoost` and `GraphSAGE` are calibrated onto the exact same PR-curve probability scale, the **Combined (XGBoost + GraphSAGE)** model blends them (`0.50 XGBoost + 0.50 GraphSAGE`) to achieve the highest overall **PR-AUC** and **ROC-AUC**, blocking any account with calibrated risk $\\ge 50\\%$."
        )

        st.markdown("---")
        st.subheader("🛡️ How This Platform Handles Skewed Data (5-Stage Balancing Pipeline)")
        st.markdown("""
        Because naive oversampling (like SMOTE) creates disconnected fake accounts that break transaction graph topology, our pipeline handles **both Class Skew (97% vs. 3%)** and **Feature Skew (power-law dollar amounts)** using five integrated stages on the training split (while keeping the test split 100% untouched at real-world skew):
        1. **Topology-Preserving Hard-Negative Undersampling:** Retains **100% of suspicious accounts** (5,089 in train), **100% of 1-hop & 2-hop ring neighbors**, and **100% of Hard-Negative legitimate accounts** (60,686 active/high-volume accounts that resemble commercial mules), while downsampling 99,216 trivial 1-transaction isolated accounts during GNN training.
        2. **Signed-Log1p Feature Normalization & Scale-Invariant Ratios:** Applies $\\text{sign}(x) \\cdot \\ln(1 + |x|)$ to compress 10-order-of-magnitude dollar amounts and pairs them with scale-invariant ratios (`in_out_tx_ratio`, `smurfing_tx_count` share, `ach_tx_count` share).
        3. **Focal Loss ($\\gamma = 1.5$) & Cost-Sensitive `aucpr` Splitting:** GraphSAGE optimizes $\\mathcal{L}_{\\text{Focal}} = -(1 - p_t)^{\\gamma} w_y \\log(p_t)$, automatically silencing gradients from easy negatives so neural weights focus on complex laundering rings.
        4. **Multi-Hop Graph Neighborhood Aggregation:** Propagates 1-hop and 2-hop neighbor risk indicators (`neighbor_ach_avg`, `neighbor_smurf_avg`, `hop2_ach_avg`) so low-volume mule accounts inherit the topological signature of their connected ring.
        5. **Precision-Recall Curve Logit Calibration:** Re-centers post-training log-odds onto the PR-curve optimal threshold so final probabilities are never distorted by class weighting or undersampling.
        """)

    # ----------------------------------------------------
    # TAB 2: FINANCIAL DISTRIBUTION & SMURFING PATTERNS
    # ----------------------------------------------------
    with eda_tab2:
        st.subheader("💵 Financial Distribution & Structuring ('Smurfing') Signature")
        st.markdown(
            "Under the US Bank Secrecy Act (BSA, 31 CFR § 1010.311) and international FATF recommendations, "
            "financial institutions must file a **Currency Transaction Report (CTR)** for transactions exceeding **\\$10,000**. "
            "Criminal organizations employ **structuring** (smurfing)—splitting large sums into sub-\\$10,000 tranches—to bypass detection."
        )

        amount_df = get_eda_amount_stats(_mtime)
        legit_amounts = amount_df[amount_df["is_laundering"] == 0]["amount_paid"]
        laund_amounts = amount_df[amount_df["is_laundering"] == 1]["amount_paid"]

        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        with c_m1:
            st.metric("Legitimate Median", f"${legit_amounts.median():,.2f}")
        with c_m2:
            delta_val = laund_amounts.median() - legit_amounts.median()
            st.metric("Laundering Median", f"${laund_amounts.median():,.2f}", delta=f"+${delta_val:,.0f}")
        with c_m3:
            st.metric("Laundering 25th Pct", f"${laund_amounts.quantile(0.25):,.2f}")
        with c_m4:
            st.metric("Laundering 75th Pct", f"${laund_amounts.quantile(0.75):,.2f}")

        st.markdown("---")
        
        st.markdown("#### **Percentile Comparison: Legitimate vs. Laundering Transfers**")
        percentiles = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        pct_data = []
        for p in percentiles:
            pct_data.append({
                "Percentile Rank": f"{int(p*100)}th Pct",
                "Legitimate Transfer": f"${legit_amounts.quantile(p):,.2f}",
                "Laundering Transfer": f"${laund_amounts.quantile(p):,.2f}",
                "Observation": "Smurfing / Structuring Clustered" if p == 0.50 else ("Micro-transfers" if p <= 0.25 else "Institutional / Layering")
            })
        st.dataframe(pd.DataFrame(pct_data), use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="intercept-box">
            <h4 style="margin-top:0;">🔍 Empirical Smurfing Detection Insight</h4>
            <p>Notice the median value for laundering transfers: <b>&dollar;8,667.21</b> compared to <b>&dollar;2,324.14</b> for legitimate transfers. 
            Furthermore, 50% of all laundering transfers fall precisely between <b>&dollar;2,634.97 and &dollar;18,832.27</b>. 
            This confirms classic smurfing behavior: transactions are concentrated immediately below the &dollar;10,000 regulatory radar.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### **Transaction Size Tranche Distribution**")
        bins = [0, 1000, 5000, 10000, 50000, 250000, 1e12]
        labels = ["< $1K", "$1K - $5K", "$5K - $10K (Smurfing Zone)", "$10K - $50K", "$50K - $250K", "> $250K"]
        
        amount_df["Tranche"] = pd.cut(amount_df["amount_paid"], bins=bins, labels=labels, right=False)
        tranche_breakdown = amount_df.groupby(["Tranche", "is_laundering"], observed=False).size().unstack(fill_value=0)
        tranche_breakdown.columns = ["Legitimate", "Laundering"]
        tranche_breakdown["Laundering Ratio (%)"] = (tranche_breakdown["Laundering"] / (tranche_breakdown["Legitimate"] + tranche_breakdown["Laundering"]) * 100).round(2)
        
        tranche_display = tranche_breakdown.copy()
        tranche_display["Legitimate"] = tranche_display["Legitimate"].map(lambda x: f"{int(x):,}")
        tranche_display["Laundering"] = tranche_display["Laundering"].map(lambda x: f"{int(x):,}")
        tranche_display["Laundering Ratio"] = tranche_display["Laundering Ratio (%)"].map(lambda x: f"{x:.2f}%")
        st.dataframe(tranche_display[["Legitimate", "Laundering", "Laundering Ratio"]], use_container_width=True)
        st.bar_chart(tranche_breakdown[["Legitimate", "Laundering"]])

    # ----------------------------------------------------
    # TAB 3: PAYMENT CHANNELS & VELOCITY
    # ----------------------------------------------------
    with eda_tab3:
        st.subheader("💳 Payment Channels & Exploitation Rates")
        st.markdown(
            "Criminal rings select specific payment mechanisms depending on execution speed, counterparty vetting, "
            "and anonymity. Below is the empirical breakdown of laundering concentration across payment channels."
        )

        pay_df = get_eda_payment_stats(_mtime)
        pivot_pay = pay_df.pivot(index="payment_format", columns="is_laundering", values="count").fillna(0)
        pivot_pay.columns = ["Legitimate_Count", "Laundering_Count"]
        pivot_pay["Total_Transfers"] = pivot_pay["Legitimate_Count"] + pivot_pay["Laundering_Count"]
        pivot_pay["Laundering_Rate (%)"] = ((pivot_pay["Laundering_Count"] / pivot_pay["Total_Transfers"]) * 100).round(2)
        
        vol_pivot = pay_df.pivot(index="payment_format", columns="is_laundering", values="volume").fillna(0)
        pivot_pay["Laundering_Volume ($)"] = vol_pivot.get(1, 0.0)
        
        risk_map = {
            "Bitcoin": "🔴 CRITICAL (73.7% Laundering Rate)",
            "ACH": "🔴 HIGH (67.5% Laundering Rate - Primary Mule Highway)",
            "Cash": "🟡 MODERATE (7.3% Laundering Rate)",
            "Cheque": "🟡 MODERATE (6.5% Laundering Rate)",
            "Credit Card": "🟢 LOW (4.4% Laundering Rate)",
            "Reinvestment": "🟢 MINIMAL (0.0% Laundering Rate)",
            "Wire": "🟢 MINIMAL (0.0% Laundering Rate)"
        }
        pivot_pay["Channel Risk Level"] = pivot_pay.index.map(lambda x: risk_map.get(x, "Normal"))
        
        display_pay = pivot_pay.sort_values(by="Laundering_Count", ascending=False).reset_index()
        display_pay.rename(columns={"payment_format": "Payment Channel"}, inplace=True)

        display_pay_view = pd.DataFrame({
            "Payment Channel": display_pay["Payment Channel"],
            "Laundering Transfers": display_pay["Laundering_Count"].map(lambda x: f"{int(x):,}"),
            "Total Transfers": display_pay["Total_Transfers"].map(lambda x: f"{int(x):,}"),
            "Laundering Rate": display_pay["Laundering_Rate (%)"].map(lambda x: f"{x:.1f}%"),
            "Laundering Volume": display_pay["Laundering_Volume ($)"].map(format_compact_currency),
            "Risk Level": display_pay["Channel Risk Level"]
        })
        st.dataframe(display_pay_view, use_container_width=True, hide_index=True)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("#### **Laundering Transfers by Payment Format**")
            st.bar_chart(display_pay.set_index("Payment Channel")["Laundering_Count"])
        with col_p2:
            st.markdown("#### **Channel Laundering Share (%)**")
            st.bar_chart(display_pay.set_index("Payment Channel")["Laundering_Rate (%)"])

        st.markdown("""
        <div class="account-preview-card">
            <h4>💡 Forensic Channel Insights</h4>
            <ul>
                <li><b>Automated Clearing House (ACH):</b> Represents <b>4,483 out of 5,177 (86.6%)</b> of all laundering transactions in the operational sample. 
                Because ACH transfers are processed in electronic batches, smurfing rings use automated scripts to disperse money through ACH mule networks without teller intervention.</li>
                <li><b>Bitcoin:</b> Exhibits the highest percentage concentration of laundering activity (<b>73.7%</b>). Criminal networks utilize cryptocurrency to bridge international boundaries and obfuscate source entity identities before off-ramping into fiat currency.</li>
                <li><b>Cheque & Cash:</b> Serve as traditional placement methods where illicit cash is introduced into retail accounts.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAB 4: TYPOLOGY TAXONOMY & RING BEHAVIOR
    # ----------------------------------------------------
    with eda_tab4:
        st.subheader("🕸️ Typology Taxonomy & Ring Mechanics")
        st.markdown(
            "Money laundering is fundamentally a collaborative network crime. Individual transfers often appear benign in isolation; "
            "the criminality only becomes evident when mapping multi-hop transaction topologies across interconnected accounts."
        )

        typo_df = get_eda_typologies(_mtime)
        typo_df_display = typo_df[typo_df["pattern_type"] != "NONE"].copy()
        typo_tbl = pd.DataFrame({
            "Typology Name": typo_df_display["pattern_type"],
            "Identified Transfers": typo_df_display["tx_count"].map(lambda x: f"{int(x):,}"),
            "Total Volume": typo_df_display["total_volume"].map(format_compact_currency),
            "Avg Volume": typo_df_display["avg_volume"].map(format_compact_currency)
        })
        st.dataframe(typo_tbl, use_container_width=True, hide_index=True)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("#### **Transaction Count by Typology**")
            st.bar_chart(typo_df_display.set_index("pattern_type")["tx_count"])
        with col_t2:
            st.markdown("#### **Total Volume by Typology ($ Billions)**")
            vol_chart_df = typo_df_display.copy()
            vol_chart_df["Volume ($B)"] = (vol_chart_df["total_volume"] / 1e9).round(2)
            st.bar_chart(vol_chart_df.set_index("pattern_type")["Volume ($B)"])

        st.markdown("---")
        st.subheader("🔍 Interactive Typology Forensic Dossier")
        
        typo_stats_map = typo_df.set_index("pattern_type").to_dict(orient="index") if not typo_df.empty else {}
        typology_catalog = {
            "GATHER-SCATTER": {
                "diagram": "Sources [A1, A2, A3] ──▶ Central Mule Aggregator [M] ──▶ Destinations [B1, B2, B3]",
                "stage": "Layering & Rapid Integration",
                "tx_count": int(typo_stats_map.get("GATHER-SCATTER", {}).get("tx_count", 1233)),
                "vol": format_compact_currency(typo_stats_map.get("GATHER-SCATTER", {}).get("total_volume", 4003134000)),
                "anatomy": "Multiple subordinate accounts funnel structured amounts into a single aggregator account, which immediately breaks down and scatters the funds to secondary recipient accounts. The central mule maintains near-zero ending balance.",
                "red_flags": "High in-degree and high out-degree within a compressed timestamp window; rapid fund velocity; in/out flow ratio near 1.0.",
                "gnn_advantage": "GraphSAGE 2-hop aggregation captures both incoming feeder nodes and downstream receivers simultaneously, tagging the hub node even if individual transfers are below reporting thresholds."
            },
            "SCATTER-GATHER": {
                "diagram": "Primary Origin [O] ──▶ Intermediary Layer [I1, I2, I3] ──▶ Final Collector [C]",
                "stage": "Layering (Obfuscation)",
                "tx_count": int(typo_stats_map.get("SCATTER-GATHER", {}).get("tx_count", 799)),
                "vol": format_compact_currency(typo_stats_map.get("SCATTER-GATHER", {}).get("total_volume", 86370720000)),
                "anatomy": "Capital origin disperses funds into multiple intermediate buffer accounts to fragment the audit trail. These intermediate nodes subsequently send funds inward to a unified collection entity.",
                "red_flags": "Symmetrical fan-out followed by convergent fan-in; temporal alignment across disconnected intermediary accounts.",
                "gnn_advantage": "Graph Neural Networks preserve path connectivity and topological flow symmetry that tabular single-row feature sets are blind to."
            },
            "STACK": {
                "diagram": "Origin [A] ──▶ Hop 1 [B] ──▶ Hop 2 [C] ──▶ Hop 3 [D] ──▶ Hop 4 [E] ...",
                "stage": "Layering (Jurisdiction & Format Hopping)",
                "tx_count": int(typo_stats_map.get("STACK", {}).get("tx_count", 533)),
                "vol": format_compact_currency(typo_stats_map.get("STACK", {}).get("total_volume", 1191305000)),
                "anatomy": "Sequential pass-through hops across accounts, often crossing banking institutions and payment formats (e.g., Wire -> ACH -> Cheque) to maximize investigative friction for law enforcement subpoenas.",
                "red_flags": "Low holding duration per account; in_degree = 1 and out_degree = 1; pass-through ratio ~ 1.0 with minor deductions for laundering commission fees.",
                "gnn_advantage": "Random-walk neighbor sampling in GraphSAGE unmasks extended linear chains that tabular models treat as isolated, unrelated transfers."
            },
            "FAN-OUT": {
                "diagram": "Master Account [M] ──▶ [Mule 1, Mule 2, Mule 3, Mule 4, Mule 5...]",
                "stage": "Placement / Structuring Dispersal",
                "tx_count": int(typo_stats_map.get("FAN-OUT", {}).get("tx_count", 785)),
                "vol": format_compact_currency(typo_stats_map.get("FAN-OUT", {}).get("total_volume", 620636400)),
                "anatomy": "A single funding account rapidly distributes capital into tens or hundreds of distinct recipient accounts. Often employed to disperse payroll fraud, ransomware bounties, or drug trafficking cash.",
                "red_flags": "Extreme `unique_receivers` count; `out_degree` heavily exceeds `in_degree`; bursts of uniform or near-uniform transfer sizes.",
                "gnn_advantage": "Degree centrality and out-degree feature embeddings explicitly flag high-divergence nodes."
            },
            "BIPARTITE": {
                "diagram": "Source Cluster {A1, A2, A3} ════▶ Recipient Cluster {B1, B2, B3}",
                "stage": "Layering (Clearinghouse Disconnect)",
                "tx_count": int(typo_stats_map.get("BIPARTITE", {}).get("tx_count", 703)),
                "vol": format_compact_currency(typo_stats_map.get("BIPARTITE", {}).get("total_volume", 23842660000)),
                "anatomy": "A set of origin accounts transfers funds exclusively into a separate set of recipient accounts with dense cross-connections, frequently simulating legitimate business supplier invoices.",
                "red_flags": "Dense bipartite graph density without reciprocal feedback transfers; common IP or timestamp clustering across disparate entity registrations.",
                "gnn_advantage": "GraphSAGE embeddings cluster bipartite partitions based on structural neighborhood equivalence."
            },
            "FAN-IN": {
                "diagram": "[Mule 1, Mule 2, Mule 3, Mule 4] ──▶ Master Collector Account [C]",
                "stage": "Integration (Consolidation)",
                "tx_count": int(typo_stats_map.get("FAN-IN", {}).get("tx_count", 516)),
                "vol": format_compact_currency(typo_stats_map.get("FAN-IN", {}).get("total_volume", 66927870000)),
                "anatomy": "Dozens of small smurfing deposits made by foot mules are consolidated into a single master account ready for real estate purchases or luxury asset acquisitions.",
                "red_flags": "`in_degree` dramatically exceeds `out_degree`; high `unique_senders` count; structured sub-$10,000 inputs.",
                "gnn_advantage": "GraphSAGE neighbor aggregation identifies convergence nodes before final integration occurs."
            },
            "CYCLE": {
                "diagram": "Account A ──▶ Account B ──▶ Account C ──▶ Account D ──▶ Account A (Loop)",
                "stage": "Layering (Round-tripping)",
                "tx_count": int(typo_stats_map.get("CYCLE", {}).get("tx_count", 314)),
                "vol": format_compact_currency(typo_stats_map.get("CYCLE", {}).get("total_volume", 3548274000)),
                "anatomy": "Funds move in a closed loop through shell companies, offshore accounts, and multiple banks, returning to the origin disguised as foreign investment capital or consulting fees.",
                "red_flags": "Closed circular path; zero net value creation; high cumulative transaction velocity with static retained equity.",
                "gnn_advantage": "Cyclic subgraphs produce distinct spectral graph signatures detected by multi-layer GraphSAGE message passing."
            },
            "RANDOM": {
                "diagram": "Stochastic Mesh Network (Decentralized Hop Matrix)",
                "stage": "Layering (Noise Generation)",
                "tx_count": int(typo_stats_map.get("RANDOM", {}).get("tx_count", 294)),
                "vol": format_compact_currency(typo_stats_map.get("RANDOM", {}).get("total_volume", 567908900)),
                "anatomy": "Transactions generated with stochastic timing, randomized dollar amounts, and variable recipient choices designed to defeat simple rule-based threshold filters.",
                "red_flags": "High variance in transaction frequency; unusual inter-bank routing paths lacking commercial rationale.",
                "gnn_advantage": "The combined XGBoost + GraphSAGE ensemble combines tabular entropy metrics with topological node representations to detect deliberate randomness."
            }
        }

        selected_typo = st.selectbox("Select Typology to inspect forensic architecture:", list(typology_catalog.keys()))
        t_meta = typology_catalog[selected_typo]

        st.markdown(f"### **Typology: {selected_typo}**")
        st.code(t_meta["diagram"], language="text")
        
        c_i1, c_i2, c_i3 = st.columns(3)
        with c_i1:
            st.metric("AML Lifecycle Stage", t_meta["stage"])
        with c_i2:
            st.metric("Sampled Transfers", f"{t_meta['tx_count']:,} txs")
        with c_i3:
            st.metric("Sampled Volume", t_meta["vol"])

        st.markdown(f"**Modus Operandi:** {t_meta['anatomy']}")
        st.markdown(f"**Forensic Red Flags:** `{t_meta['red_flags']}`")
        st.info(f"🛡️ **Why GraphSAGE GNN Outperforms Tabular Rules:** {t_meta['gnn_advantage']}")

    # ----------------------------------------------------
    # TAB 5: GRAPH TOPOLOGY & FEATURE CORRELATIONS
    # ----------------------------------------------------
    with eda_tab5:
        st.subheader("🔗 Graph Topology & Feature Correlations")
        st.markdown(
            "Money laundering leaves distinct structural footprints in transaction networks. "
            "Below is the empirical correlation analysis between account features and confirmed laundering status across all **206,242 accounts**."
        )

        corr_df = get_eda_correlations(_mtime)
        corr_series = corr_df.corr()["is_suspicious"].sort_values(ascending=False).drop("is_suspicious")
        
        corr_display = pd.DataFrame({
            "Technical Key": corr_series.index,
            "Understandable Name": [format_feature_name(k) for k in corr_series.index],
            "Correlation with Suspicion (r)": corr_series.values.round(4)
        })

        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.markdown("#### **Top Indicators Correlated with Laundering**")
            st.dataframe(corr_display, use_container_width=True)

        with col_c2:
            st.markdown("#### **Correlation Magnitude Ranking**")
            st.bar_chart(corr_display.set_index("Understandable Name")["Correlation with Suspicion (r)"])

        st.markdown("---")
        st.subheader("🔥 Comprehensive Feature Correlation Heatmap")
        
        # Matplotlib + Seaborn Heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sub_corr = corr_df.corr()
        sns.heatmap(
            sub_corr, 
            annot=True, 
            fmt=".2f", 
            cmap="vlag", 
            center=0, 
            linewidths=0.5, 
            ax=ax,
            annot_kws={"size": 7.5},
            cbar_kws={'label': 'Pearson Correlation (r)'}
        )
        plt.title("Correlation Matrix of Topological & Behavioral AML Features", fontsize=11, fontweight='bold', pad=10)
        plt.xticks(fontsize=8, rotation=45, ha='right')
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        <div class="account-preview-card">
            <h4>📊 Data Completeness & Schema Audit (206,242 Accounts)</h4>
            <ul>
                <li><b>Missing Values / Null Count:</b> <code>0 (0.00%)</code> — 100% complete across all 30 engineered features.</li>
                <li><b>Leading Predictive Signal:</b> <code>ach_tx_count</code> (Pearson r = +0.3614) confirms that ACH channel automation is the predominant vehicle for structuring rings.</li>
                <li><b>Topological Network Signals:</b> Graph features (<code>in_degree</code>, <code>unique_senders</code>, <code>degree_centrality</code>) correlate significantly with criminal status, outperforming raw dollar averages (<code>avg_amount_in</code>: r = 0.0107).</li>
                <li><b>Conclusion:</b> Network topology is vastly more informative than raw transaction amounts for detecting organized financial crime.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ----------------------------------------------------
# 3. CORE BANKING & TRANSFERS
# ----------------------------------------------------
elif menu == "🏦 Core Banking & Transfers":
    st.title("🏦 Core Banking & Automated AML Compliance Gatekeeper")
    st.write(
        "Execute inter-account transfers with real-time ML risk screening. "
        "**Rule Enforcement:** If an account's predicted laundering probability exceeds **50%**, "
        "the transaction is intercepted and the account is automatically locked down from further operations."
    )

    # Load fast cached account dropdown catalog (all 206,239 accounts)
    catalog = get_banking_dropdown_catalog(_mtime)
    with db.get_connection() as conn:
        _blk_rows = conn.execute("SELECT account_id FROM accounts WHERE is_blocked = 1").fetchall()
    blocked_ids = [r[0] for r in _blk_rows]
    blocked_opts = [catalog["id_to_label"].get(aid, aid) for aid in blocked_ids]

    # Persistent session state across page navigation (entering EDA and returning)
    preset_options = [
        "🌟 Normal Flow (Approved Legitimate Transaction, Low Risk <20%)",
        "⛔ High Laundering Risk (50% - 70% -> Elevated ACH Velocity Block)",
        "🚨 Critical Laundering Threat (>70% -> High Outgoing Transactions Block)",
        "🔍 Custom Search (Select Any Accounts)"
    ]
    if "saved_cb_preset" not in st.session_state or st.session_state["saved_cb_preset"] not in preset_options:
        st.session_state["saved_cb_preset"] = preset_options[0]
    if "saved_cb_custom_sender" not in st.session_state:
        st.session_state["saved_cb_custom_sender"] = "100428660"
    if "saved_cb_custom_receiver" not in st.session_state:
        st.session_state["saved_cb_custom_receiver"] = "800045000"
    if "saved_cb_sender_filter" not in st.session_state:
        st.session_state["saved_cb_sender_filter"] = "All Accounts (206,239)"
    if "saved_cb_receiver_filter" not in st.session_state:
        st.session_state["saved_cb_receiver_filter"] = "All Accounts (206,239)"
    if "saved_cb_amount" not in st.session_state:
        st.session_state["saved_cb_amount"] = 250.0
    if "saved_cb_format" not in st.session_state:
        st.session_state["saved_cb_format"] = "ACH"
    if "saved_cb_currency" not in st.session_state:
        st.session_state["saved_cb_currency"] = "US Dollar"
    if "saved_cb_threshold" not in st.session_state:
        st.session_state["saved_cb_threshold"] = 50
    if "saved_cb_desk_filter" not in st.session_state:
        st.session_state["saved_cb_desk_filter"] = "All Accounts (206,239)"
    if "saved_cb_desk_acc" not in st.session_state:
        st.session_state["saved_cb_desk_acc"] = "1004286A8"

    b_tabs = st.tabs(["💸 Send Transactions", "📜 Banking Audit Ledger", "⚖️ Account & Compliance Desk"])

    # ---------------- TAB 1: SEND TRANSACTIONS ----------------
    with b_tabs[0]:
        st.subheader("Initiate Interbank Transfer")
        
        # Test preset selector (index backed by persistent session state so leaving page never resets it)
        preset_mode = st.radio(
            "Demo Testing Quick Presets:",
            preset_options,
            index=preset_options.index(st.session_state["saved_cb_preset"]),
            horizontal=True
        )

        # Only trigger preset_changed when the user actively switches presets on this page
        preset_changed = (st.session_state["saved_cb_preset"] != preset_mode)
        if preset_changed:
            st.session_state["saved_cb_preset"] = preset_mode
            if "Normal Flow" in preset_mode:
                st.session_state["saved_cb_amount"] = 250.0
                st.session_state["saved_cb_format"] = "ACH"
            elif "High Laundering Risk" in preset_mode:
                st.session_state["saved_cb_amount"] = 500.0
                st.session_state["saved_cb_format"] = "ACH"
            elif "Critical Laundering Threat" in preset_mode:
                st.session_state["saved_cb_amount"] = 1000.0
                st.session_state["saved_cb_format"] = "Wire"
            else:
                st.session_state["saved_cb_amount"] = 500.0
                st.session_state["saved_cb_format"] = "ACH"
            if "last_transfer_result" in st.session_state:
                del st.session_state["last_transfer_result"]

        if "Normal Flow" in preset_mode:
            sender_default = "800042CB0"
            receiver_default = "800045000"
            col_cap1, col_cap2 = st.columns([4, 1])
            with col_cap1:
                st.caption("🌟 **Preset Loaded:** Sender `800042CB0` has clean, normal transactional history (~10.4% risk). When executed, this transfer will pass compliance and settle. Changes and balances remain saved across tabs unless you click **Reset Demo**.")
            with col_cap2:
                if st.button("🔄 Reset Demo", key="btn_rst_norm", use_container_width=True):
                    reset_all_demo_data(balance=10000.0)
                    st.session_state["saved_cb_amount"] = 250.0
                    st.session_state["saved_cb_format"] = "ACH"
                    if "last_transfer_result" in st.session_state:
                        del st.session_state["last_transfer_result"]
                    st.toast("🔄 Demo data reset to original state!", icon="✅")
                    st.rerun()
        elif "High Laundering Risk" in preset_mode:
            sender_default = "8000FFF70"
            receiver_default = "800045000"
            col_cap1, col_cap2 = st.columns([4, 1])
            with col_cap1:
                st.caption("⛔ **Preset Loaded:** Sender `8000FFF70` (FAN-IN mule gathering: elevated ACH velocity & outflow drain). When executed, it blocks the transfer citing `'Elevated ACH electronic velocity'` and locks down the account until you unblock it or click **Reset Demo**.")
            with col_cap2:
                if st.button("🔄 Reset Demo", key="btn_rst_high", use_container_width=True):
                    reset_all_demo_data(balance=10000.0)
                    st.session_state["saved_cb_amount"] = 500.0
                    st.session_state["saved_cb_format"] = "ACH"
                    if "last_transfer_result" in st.session_state:
                        del st.session_state["last_transfer_result"]
                    st.toast("🔄 Demo data reset to original state!", icon="✅")
                    st.rerun()
        elif "Critical Laundering Threat" in preset_mode:
            sender_default = "1004286A8"
            receiver_default = "800045000"
            col_cap1, col_cap2 = st.columns([4, 1])
            with col_cap1:
                st.caption("🚨 **Preset Loaded:** Sender `1004286A8` (Severe laundering profile: 1,659 outgoing transactions, 404 structured sub-\\$10,000 smurfing records to 853 recipients). When executed, it blocks the transfer citing `'High outgoing transactions'` and keeps the account blocked until you unblock it or click **Reset Demo**.")
            with col_cap2:
                if st.button("🔄 Reset Demo", key="btn_rst_crit", use_container_width=True):
                    reset_all_demo_data(balance=10000.0)
                    st.session_state["saved_cb_amount"] = 1000.0
                    st.session_state["saved_cb_format"] = "Wire"
                    if "last_transfer_result" in st.session_state:
                        del st.session_state["last_transfer_result"]
                    st.toast("🔄 Demo data reset to original state!", icon="✅")
                    st.rerun()
        else:
            sender_default = st.session_state["saved_cb_custom_sender"]
            receiver_default = st.session_state["saved_cb_custom_receiver"]
            col_cap1, col_cap2 = st.columns([4, 1])
            with col_cap1:
                st.caption("🔍 **Custom Search Loaded:** Pick any of the **206,239 accounts** directly from the Sender and Recipient dropdowns below (type to search by Account ID, risk level, or typology). All blocks and balance changes remain saved until you unblock the account or click **Reset Demo**.")
            with col_cap2:
                if st.button("🔄 Reset Demo", key="btn_rst_cust", use_container_width=True):
                    reset_all_demo_data(balance=10000.0)
                    st.session_state["saved_cb_amount"] = 500.0
                    st.session_state["saved_cb_format"] = "ACH"
                    if "last_transfer_result" in st.session_state:
                        del st.session_state["last_transfer_result"]
                    st.toast("🔄 Demo data reset to original state!", icon="✅")
                    st.rerun()

        col_s, col_r = st.columns(2)

        # Sender Column
        with col_s:
            st.markdown("### 📤 Sender Account (From)")
            if "Custom Search" in preset_mode:
                s_filter_choices = [
                    "All Accounts (206,239)",
                    "🚨 Laundering / Suspicious Only (6,361)",
                    "🟢 Legitimate / Clean Only (199,878)",
                    "🔴 Currently Blocked Accounts"
                ]
                s_flt_idx = s_filter_choices.index(st.session_state["saved_cb_sender_filter"]) if st.session_state["saved_cb_sender_filter"] in s_filter_choices else 0
                s_filter = st.selectbox("Filter Sender Dropdown:", s_filter_choices, index=s_flt_idx)
                st.session_state["saved_cb_sender_filter"] = s_filter

                if "Laundering" in s_filter:
                    s_opts = catalog["laundering"]
                elif "Legitimate" in s_filter:
                    s_opts = catalog["legitimate"]
                elif "Blocked" in s_filter:
                    s_opts = blocked_opts if blocked_opts else catalog["all_sender"]
                    if not blocked_opts:
                        st.caption("ℹ️ No accounts are currently blocked. Showing all accounts.")
                else:
                    s_opts = catalog["all_sender"]

                cur_s_label = catalog["id_to_label"].get(st.session_state["saved_cb_custom_sender"])
                try:
                    s_idx = s_opts.index(cur_s_label) if cur_s_label else 0
                except ValueError:
                    s_idx = 0

                sel_s_opt = st.selectbox(
                    "Select Sender Account (Type ID or Typology to Search):",
                    s_opts,
                    index=s_idx
                )
                sender_input = sel_s_opt.split(" | ")[0].strip()
                st.session_state["saved_cb_custom_sender"] = sender_input
            else:
                sender_input = st.text_input("Sender Account ID:", value=sender_default, disabled=True)

            sender_acc = banking.get_account(sender_input.strip()) if sender_input.strip() else None

            if sender_acc:
                is_blk = int(sender_acc.get("is_blocked", 0) or 0) == 1
                status_html = '<span class="tag-blocked">🔴 BLOCKED</span>' if is_blk else '<span class="tag-active">🟢 ACTIVE</span>'
                bal = float(sender_acc.get("balance", 0.0) or 0.0)
                risk_pct = float(sender_acc.get("risk_score", 0.0) or 0.0) * 100.0
                notice_html = f'<div style="color:#f87171; font-size:0.85rem; margin-top:4px;"><b>Restriction Notice:</b> {sender_acc.get("block_reason")}</div>' if is_blk else ''

                st.html(f"""
                <div class="account-preview-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <strong>ID: {sender_acc['account_id']}</strong>
                        {status_html}
                    </div>
                    <div style="margin-top:6px; font-size:0.9rem; color:#cbd5e1;">
                        <div>Bank: <b>{sender_acc['bank_name']}</b> | Entity: <b>{sender_acc['entity_name']}</b></div>
                        <div>Available Balance: <b style="color:#38bdf8; font-size:1.1rem;">${bal:,.2f}</b></div>
                        <div>Historical Risk Score: <b>{risk_pct:.1f}%</b> | Typology: <b>{sender_acc.get('pattern_type', 'NONE')}</b></div>
                        {notice_html}
                    </div>
                </div>
                """)
            else:
                st.warning("Please provide a valid sender account ID.")

        # Recipient Column
        with col_r:
            st.markdown("### 📥 Recipient Account (To)")
            if "Custom Search" in preset_mode:
                r_filter_choices = [
                    "All Accounts (206,239)",
                    "🟢 Legitimate / Clean Only (199,878)",
                    "🚨 Laundering / Suspicious Only (6,361)",
                    "🔴 Currently Blocked Accounts"
                ]
                r_flt_idx = r_filter_choices.index(st.session_state["saved_cb_receiver_filter"]) if st.session_state["saved_cb_receiver_filter"] in r_filter_choices else 0
                r_filter = st.selectbox("Filter Recipient Dropdown:", r_filter_choices, index=r_flt_idx)
                st.session_state["saved_cb_receiver_filter"] = r_filter

                if "Laundering" in r_filter:
                    r_opts = catalog["laundering"]
                elif "Legitimate" in r_filter:
                    r_opts = catalog["legitimate"]
                elif "Blocked" in r_filter:
                    r_opts = blocked_opts if blocked_opts else catalog["all_recipient"]
                    if not blocked_opts:
                        st.caption("ℹ️ No accounts are currently blocked. Showing all accounts.")
                else:
                    r_opts = catalog["all_recipient"]

                cur_r_label = catalog["id_to_label"].get(st.session_state["saved_cb_custom_receiver"])
                try:
                    r_idx = r_opts.index(cur_r_label) if cur_r_label else 0
                except ValueError:
                    r_idx = 0

                sel_r_opt = st.selectbox(
                    "Select Recipient Account (Type ID or Typology to Search):",
                    r_opts,
                    index=r_idx
                )
                recipient_input = sel_r_opt.split(" | ")[0].strip()
                st.session_state["saved_cb_custom_receiver"] = recipient_input
            else:
                recipient_input = st.text_input("Recipient Account ID:", value=receiver_default, disabled=True)

            recipient_acc = banking.get_account(recipient_input.strip()) if recipient_input.strip() else None

            if recipient_acc:
                is_r_blk = int(recipient_acc.get("is_blocked", 0) or 0) == 1
                r_status_html = '<span class="tag-blocked">🔴 BLOCKED</span>' if is_r_blk else '<span class="tag-active">🟢 ACTIVE</span>'
                r_bal = float(recipient_acc.get("balance", 0.0) or 0.0)
                r_notice_html = f'<div style="color:#f87171; font-size:0.85rem; margin-top:4px;"><b>Restriction Notice:</b> {recipient_acc.get("block_reason")}</div>' if is_r_blk else ''

                st.html(f"""
                <div class="account-preview-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <strong>ID: {recipient_acc['account_id']}</strong>
                        {r_status_html}
                    </div>
                    <div style="margin-top:6px; font-size:0.9rem; color:#cbd5e1;">
                        <div>Bank: <b>{recipient_acc['bank_name']}</b> | Entity: <b>{recipient_acc['entity_name']}</b></div>
                        <div>Available Balance: <b style="color:#38bdf8; font-size:1.1rem;">${r_bal:,.2f}</b></div>
                        <div>Entity Type: <b>{recipient_acc.get('pattern_type', 'Legitimate')}</b></div>
                        {r_notice_html}
                    </div>
                </div>
                """)
            else:
                st.warning("Please provide a valid recipient account ID.")

        # Transfer Parameters (persisted across page navigation)
        st.markdown("### 💳 Transaction Details & AML Parameters")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            tx_amount = st.number_input("Transfer Amount ($):", min_value=1.0, value=float(st.session_state["saved_cb_amount"]), step=50.0)
            st.session_state["saved_cb_amount"] = float(tx_amount)
        with p2:
            fmt_choices = ["ACH", "Wire", "Cheque"]
            fmt_idx = fmt_choices.index(st.session_state["saved_cb_format"]) if st.session_state["saved_cb_format"] in fmt_choices else 0
            tx_format = st.selectbox("Payment Channel / Format:", fmt_choices, index=fmt_idx)
            st.session_state["saved_cb_format"] = tx_format
        with p3:
            curr_choices = ["US Dollar", "Euro", "British Pound"]
            curr_idx = curr_choices.index(st.session_state["saved_cb_currency"]) if st.session_state["saved_cb_currency"] in curr_choices else 0
            tx_currency = st.selectbox("Currency:", curr_choices, index=curr_idx)
            st.session_state["saved_cb_currency"] = tx_currency
        with p4:
            aml_threshold_pct = st.slider("AML Blocking Threshold (%):", min_value=10, max_value=90, value=int(st.session_state["saved_cb_threshold"]), step=5, help="Accounts with laundering probability equal to or exceeding this threshold are blocked.")
            st.session_state["saved_cb_threshold"] = int(aml_threshold_pct)
            aml_threshold = aml_threshold_pct / 100.0

        if st.button("⚡ Execute Interbank Transfer", type="primary", use_container_width=True):
            if not sender_acc or not recipient_acc:
                st.error("Both valid Sender and Recipient accounts must be provided.")
            elif sender_input.strip() == recipient_input.strip():
                st.error("Sender and Recipient accounts must be distinct.")
            else:
                with st.spinner("Screening transaction against calibrated ML models (XGBoost, GraphSAGE & Combined Ensemble)..."):
                    result = banking.execute_transfer(
                        from_account_id=sender_input.strip(),
                        to_account_id=recipient_input.strip(),
                        amount=float(tx_amount),
                        payment_format=tx_format,
                        payment_currency=tx_currency,
                        blocking_threshold=aml_threshold,
                        warning_threshold=0.70
                    )
                st.session_state["last_transfer_result"] = result
                if result.get("status") == "BLOCKED":
                    st.session_state["saved_cb_desk_acc"] = sender_input.strip()
                st.rerun()

        # Display result from session state
        if "last_transfer_result" in st.session_state:
            result = st.session_state["last_transfer_result"]
            st.divider()
            st.subheader("🛡️ Real-Time AML Compliance Radar & Decision")

            # Show Risk Metrics
            risk_eval = result.get("risk_eval", {})
            highest_prob = result.get("laundering_probability", 0.0)

            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                st.metric("⚡ XGBoost Tabular Risk", f"{risk_eval.get('xgb_prob', 0.0)*100:.1f}%")
            with rc2:
                st.metric("🕸️ GraphSAGE Graph Risk", f"{risk_eval.get('sage_prob', 0.0)*100:.1f}%")
            with rc3:
                comb_val = risk_eval.get('combined_prob', 0.0)
                st.metric("🔮 Combined (XGB + SAGE)", f"{comb_val*100:.1f}%" if comb_val > 0 else f"{highest_prob*100:.1f}%")
            with rc4:
                delta_text = f"+{(highest_prob - aml_threshold)*100:.1f}% over threshold" if highest_prob >= aml_threshold else f"{(aml_threshold - highest_prob)*100:.1f}% below threshold"
                delta_color = "inverse" if highest_prob >= aml_threshold else "normal"
                st.metric("🚨 Final Laundering Risk", f"{highest_prob*100:.1f}%", delta=delta_text, delta_color=delta_color)

            st.caption("🎯 **Gatekeeper Intelligence:** Decision powered by PR-AUC optimized **Combined (XGBoost + GraphSAGE)** ensemble model.")

            # Show Result Card
            if result["status"] == "BLOCKED":
                is_crit = result.get("is_critical_warning") or (highest_prob >= 0.70)
                forensic_reasons = result.get("forensic_reasons") or []
                primary_reason = result.get("primary_reason") or (forensic_reasons[0] if forensic_reasons else "Elevated Laundering Risk")

                # Build HTML list of forensic reasons
                reasons_items = "".join([f"<li style='margin-bottom:6px; line-height:1.4;'><b>{r.split('(')[0].strip()}:</b> {r.split('(', 1)[1].rstrip(')') if '(' in r else r}</li>" for r in forensic_reasons])
                reasons_card = f"""
                <div style="background: rgba(220, 38, 38, 0.15); border-left: 4px solid #ef4444; border-radius: 4px; padding: 12px 16px; margin: 12px 0;">
                    <div style="font-weight: 700; color: #fca5a5; font-size: 0.95rem; margin-bottom: 8px;">🔍 Confirmed Forensic AML Block Reasons:</div>
                    <ul style="margin: 0 0 0 18px; padding: 0; color: #fecdd3; font-size: 0.92rem;">
                        {reasons_items}
                    </ul>
                </div>
                """ if forensic_reasons else ""

                if is_crit:
                    st.toast(f"🚨 AML Block: {primary_reason.split('(')[0].strip()}!", icon="⚠️")
                    st.warning(f"⚠️ **Urgent Compliance Block ({primary_reason.split('(')[0].strip()}):** Account **{result.get('from_account', sender_input.strip())}** was blocked. Model Risk: **{highest_prob*100:.1f}%**.")
                    st.html(f"""
                    <div class="critical-alert-box">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h3 style="margin-top:0; margin-bottom:0; color:#fca5a5;">🚨 URGENT AML COMPLIANCE WARNING & INTERCEPT</h3>
                            <span style="background-color:#be123c; color:white; padding:4px 12px; border-radius:6px; font-weight:bold; font-size:0.85rem;">REASON: {primary_reason.split('(')[0].strip().upper()}</span>
                        </div>
                        <p style="font-size:1.1rem; margin-top:10px; margin-bottom:8px;"><b>Transfer Reference:</b> {result['transfer_id']}</p>
                        <p style="font-size:1.05rem;">{result['message']}</p>
                        {reasons_card}
                        <hr style="border-color:#f43f5e; margin:12px 0;">
                        <div style="font-size:0.95rem; color:#fecdd3;">
                            <b>🚨 Regulatory Enforcement Action:</b> Sender account <code>{result.get('from_account', sender_input.strip())}</code> was locked down due to <b>{primary_reason}</b>.
                            The transaction was intercepted, and the account status remains saved as <b><code>is_blocked = 1 (🔴 BLOCKED)</code></b> until unblocked in the <b>⚖️ Account & Compliance Desk</b> or via <b>🔄 Reset Demo</b>.
                        </div>
                    </div>
                    """)
                else:
                    st.toast(f"⛔ AML Block: {primary_reason.split('(')[0].strip()}!", icon="⚠️")
                    st.html(f"""
                    <div class="intercept-box">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h3 style="margin-top:0; margin-bottom:0; color:#fca5a5;">⛔ TRANSACTION BLOCKED BY AML SYSTEM</h3>
                            <span style="background-color:#991b1b; color:white; padding:4px 12px; border-radius:6px; font-weight:bold; font-size:0.85rem;">REASON: {primary_reason.split('(')[0].strip().upper()}</span>
                        </div>
                        <p style="font-size:1.1rem; margin-top:10px; margin-bottom:8px;"><b>Transfer Reference:</b> {result['transfer_id']}</p>
                        <p style="font-size:1.05rem;">{result['message']}</p>
                        {reasons_card}
                        <hr style="border-color:#ef4444; margin:12px 0;">
                        <div style="font-size:0.95rem;">
                            <b>Compliance Action:</b> Sender account <code>{result.get('from_account', sender_input.strip())}</code> was locked down due to <b>{primary_reason}</b> (Risk: {highest_prob*100:.1f}%, threshold: {aml_threshold*100:.0f}%).
                            The account status remains saved as <b><code>is_blocked = 1 (🔴 BLOCKED)</code></b> until unblocked in the <b>⚖️ Account & Compliance Desk</b> or via <b>🔄 Reset Demo</b>.
                        </div>
                    </div>
                    """)
            elif result["status"] == "APPROVED":
                st.toast("✅ Transfer approved and settled successfully!", icon="💸")
                st.html(f"""
                <div class="approved-box">
                    <h3 style="margin-top:0; color:#6ee7b7;">✅ TRANSACTION APPROVED & SETTLED</h3>
                    <p style="font-size:1.1rem; margin-bottom:8px;"><b>Transfer Reference:</b> {result['transfer_id']}</p>
                    <p style="font-size:1.05rem;">{result['message']}</p>
                    <hr style="border-color:#10b981; margin:12px 0;">
                    <div style="display:flex; justify-content:space-between; font-size:1.0rem;">
                        <div><b>Sender New Balance:</b> <span style="color:#6ee7b7;">${result['sender_new_balance']:,.2f}</span></div>
                        <div><b>Recipient New Balance:</b> <span style="color:#6ee7b7;">${result['recipient_new_balance']:,.2f}</span></div>
                        <div><b>Status:</b> Settled in Ledger</div>
                    </div>
                </div>
                """)
            else:
                st.error(f"Transfer Failed: {result.get('message')}")

    # ---------------- TAB 2: AUDIT LEDGER ----------------
    with b_tabs[1]:
        st.subheader("📜 Banking Transfers Audit Ledger")
        st.write("Complete audit log of all interbank transfers processed by the system, tracking AML risk scores and automated block decisions.")

        audit_limit = st.slider("Display records:", min_value=10, max_value=200, value=50, step=10)
        df_ledger = db.get_banking_transfers(limit=audit_limit)

        if not df_ledger.empty:
            if "rf_prob" in df_ledger.columns:
                df_ledger = df_ledger.drop(columns=["rf_prob"])
            st.dataframe(df_ledger, use_container_width=True)
        else:
            st.info("No transfers recorded in the audit ledger yet. Execute a transfer to see it logged here.")

    # ---------------- TAB 3: ACCOUNT & COMPLIANCE DESK ----------------
    with b_tabs[2]:
        st.subheader("⚖️ AML Compliance & Account Management Desk")
        st.write("Select any account from the dropdown to inspect balances, deposit funds, or unblock/block accounts during investigations.")

        d_col1, d_col2 = st.columns([1, 2])
        with d_col1:
            desk_filter_choices = [
                "All Accounts (206,239)",
                "🔴 Currently Blocked Accounts",
                "🚨 Laundering / Suspicious Only (6,361)",
                "🟢 Legitimate / Clean Only (199,878)"
            ]
            d_flt_idx = desk_filter_choices.index(st.session_state["saved_cb_desk_filter"]) if st.session_state["saved_cb_desk_filter"] in desk_filter_choices else 0
            desk_filter = st.selectbox("Filter Accounts:", desk_filter_choices, index=d_flt_idx)
            st.session_state["saved_cb_desk_filter"] = desk_filter

        with d_col2:
            if "Blocked" in desk_filter:
                d_opts = blocked_opts if blocked_opts else catalog["all_sender"]
                if not blocked_opts:
                    st.caption("ℹ️ No accounts are currently blocked. Showing all accounts.")
            elif "Laundering" in desk_filter:
                d_opts = catalog["laundering"]
            elif "Legitimate" in desk_filter:
                d_opts = catalog["legitimate"]
            else:
                # Put currently blocked accounts right at the top of All Accounts for quick unblocking
                d_opts = blocked_opts + [opt for opt in catalog["all_sender"] if opt not in set(blocked_opts)] if blocked_opts else catalog["all_sender"]

            cur_d_label = catalog["id_to_label"].get(st.session_state["saved_cb_desk_acc"])
            try:
                d_idx = d_opts.index(cur_d_label) if cur_d_label else 0
            except ValueError:
                d_idx = 0

            sel_desk_opt = st.selectbox(
                "Select Account to Inspect / Manage (Type ID to Search):",
                d_opts,
                index=d_idx
            )
            desk_acc_id = sel_desk_opt.split(" | ")[0].strip()
            st.session_state["saved_cb_desk_acc"] = desk_acc_id

        if desk_acc_id.strip():
            acc_record = banking.get_account(desk_acc_id.strip())
            if acc_record:
                is_curr_blocked = int(acc_record.get("is_blocked", 0) or 0) == 1
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Account ID", acc_record["account_id"])
                m2.metric("Available Balance", f"${float(acc_record.get('balance', 0.0) or 0.0):,.2f}")
                m3.metric("Laundering Risk", f"{float(acc_record.get('risk_score', 0.0) or 0.0)*100:.1f}%")
                m4.metric("Status", "🔴 BLOCKED (Restricted)" if is_curr_blocked else "🟢 ACTIVE (Permitted)")

                st.write(f"**Bank:** {acc_record.get('bank_name')} | **Entity:** {acc_record.get('entity_name')} | **Block Reason:** {acc_record.get('block_reason') or 'None'}")

                st.divider()
                col_act1, col_act2 = st.columns(2)

                # Deposit funds
                with col_act1:
                    st.markdown("#### 💰 Deposit / Fund Account")
                    dep_amt = st.number_input("Deposit Amount ($):", min_value=10.0, value=1000.0, step=100.0)
                    if st.button("➕ Deposit Funds"):
                        res_dep = banking.deposit(desk_acc_id.strip(), dep_amt)
                        if res_dep["success"]:
                            st.success(res_dep["message"])
                            st.rerun()
                        else:
                            st.error(res_dep["message"])

                # Override block status
                with col_act2:
                    st.markdown("#### 🛡️ Compliance Override (Block / Unblock)")
                    override_reason = st.text_input("Reason for Action:", value="Manual compliance review by investigator")
                    if is_curr_blocked:
                        if st.button("🔓 Unblock Account (Restore Transfers)", type="primary"):
                            res_unb = banking.set_block_status(desk_acc_id.strip(), is_blocked=False, reason="")
                            if "last_transfer_result" in st.session_state and st.session_state["last_transfer_result"].get("from_account") == desk_acc_id.strip():
                                del st.session_state["last_transfer_result"]
                            st.success(res_unb["message"])
                            st.rerun()
                    else:
                        if st.button("🔒 Block Account (Restrict Transfers)", type="secondary"):
                            res_blk = banking.set_block_status(desk_acc_id.strip(), is_blocked=True, reason=f"Blocked: {override_reason}")
                            st.warning(res_blk["message"])
                            st.rerun()
            else:
                st.error(f"Account '{desk_acc_id.strip()}' was not found.")


# ----------------------------------------------------
# 3. SUSPICIOUS ACCOUNTS
# ----------------------------------------------------
elif menu == "🔍 Suspicious Accounts":
    st.title("🔍 Suspicious Accounts Explorer")
    st.write("Browse and filter accounts flagged by the AML detection pipeline, inspect balances, and review block status.")

    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        filter_status = st.selectbox(
            "Status Filter",
            [
                "Suspicious Only (is_suspicious=1)",
                "AML Blocked Only (is_blocked=1)",
                "All Accounts",
                "Legitimate Only"
            ]
        )
    with col_f2:
        search_query = st.text_input("Search Account ID or Entity Name", "")

    query = "SELECT account_id, bank_name, entity_name, balance, is_blocked, block_reason, is_suspicious, suspicion_category, risk_score, total_tx_in, total_tx_out, net_flow, pattern_type FROM accounts WHERE 1=1"
    if "Suspicious Only" in filter_status:
        query += " AND is_suspicious = 1"
    elif "AML Blocked Only" in filter_status:
        query += " AND is_blocked = 1"
    elif "Legitimate Only" in filter_status:
        query += " AND is_suspicious = 0"
    
    if search_query:
        query += f" AND (account_id LIKE '%{search_query}%' OR entity_name LIKE '%{search_query}%')"

    query += " ORDER BY risk_score DESC, total_tx_in + total_tx_out DESC LIMIT 200"

    with db.get_connection() as conn:
        df_acc = pd.read_sql(query, conn)

    st.write(f"Showing **{len(df_acc)}** matching accounts:")
    st.dataframe(df_acc, use_container_width=True)

    # Deep-Dive Card
    if not df_acc.empty:
        st.divider()
        st.subheader("Account Deep-Dive & Compliance Status")
        selected_id = st.selectbox("Select Account ID to Inspect", df_acc["account_id"].tolist())
        acc_info = df_acc[df_acc["account_id"] == selected_id].iloc[0]

        is_acc_blk = int(acc_info.get("is_blocked", 0) or 0) == 1
        blk_badge = '<span class="tag-blocked">🔴 BLOCKED FROM TRANSFERS</span>' if is_acc_blk else '<span class="tag-active">🟢 TRANSFERS ACTIVE</span>'

        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Risk Score", f"{acc_info['risk_score']*100:.1f}%")
        d2.metric("Category", acc_info['suspicion_category'])
        d3.metric("Pattern Typology", acc_info['pattern_type'])
        d4.metric("Net Flow", f"${acc_info['net_flow']:,.2f}")
        d5.metric("Available Balance", f"${float(acc_info.get('balance', 0.0) or 0.0):,.2f}")

        st.markdown(f"**Entity:** {acc_info['entity_name']} | **Bank:** {acc_info['bank_name']} | **Status:** {blk_badge}", unsafe_allow_html=True)
        if is_acc_blk:
            st.error(f"Restriction Notice: {acc_info.get('block_reason') or 'Blocked by AML policy'}")

        # Quick compliance override button
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if is_acc_blk:
                if st.button("🔓 Unblock Account", key="unblk_deep"):
                    banking.set_block_status(selected_id, False, "Investigator clearance")
                    st.success("Account unblocked!")
                    st.rerun()
            else:
                if st.button("🔒 Block Account", key="blk_deep"):
                    banking.set_block_status(selected_id, True, "Investigator restriction")
                    st.warning("Account blocked!")
                    st.rerun()

        # Associated Transactions
        with db.get_connection() as conn:
            tx_df = pd.read_sql("""
            SELECT timestamp, from_account, to_account, amount_paid, payment_currency, payment_format, is_laundering, pattern_type
            FROM transactions
            WHERE from_account = ? OR to_account = ?
            ORDER BY timestamp DESC LIMIT 50
            """, conn, params=(selected_id, selected_id))
        
        st.write("Recent Transaction History:")
        st.dataframe(tx_df, use_container_width=True)


# ----------------------------------------------------
# 4. TRANSACTION EXPLORER
# ----------------------------------------------------
elif menu == "💳 Transaction Explorer":
    st.title("💳 Transaction Explorer")
    st.write("Examine indexed interbank transactions, currency flows, and laundering markers.")

    c1, c2 = st.columns(2)
    with c1:
        tx_filter = st.selectbox("Transaction Type", ["All Indexed Transactions", "Laundering Only (is_laundering=1)", "Legitimate Banking Only"])
    with c2:
        format_filter = st.selectbox("Payment Format", ["All Formats", "ACH", "Wire", "Cheque", "Reinvestment"])

    q = "SELECT rowid AS id, timestamp, from_bank, from_account, to_bank, to_account, amount_paid, payment_currency, payment_format, is_laundering, pattern_type FROM transactions WHERE 1=1"
    if "Laundering Only" in tx_filter:
        q += " AND is_laundering = 1"
    elif "Legitimate Banking Only" in tx_filter:
        q += " AND is_laundering = 0"

    if format_filter != "All Formats":
        q += f" AND payment_format = '{format_filter}'"
    
    q += " ORDER BY rowid DESC LIMIT 100"

    with db.get_connection() as conn:
        df_tx_view = pd.read_sql(q, conn)

    st.write(f"Displaying **{len(df_tx_view)}** transactions:")
    st.dataframe(df_tx_view, use_container_width=True)


# ----------------------------------------------------
# 5. LIVE ML INFERENCE
# ----------------------------------------------------
elif menu == "🤖 Live ML Inference":
    st.title("🤖 Live ML Models Prediction & Inference")
    st.write("Direct internal Python calls to PR-AUC calibrated **XGBoost**, **GraphSAGE**, and **Combined (XGBoost + GraphSAGE)** models.")

    c_flt1, c_flt2 = st.columns([1, 2])
    with c_flt1:
        acc_type_filter = st.selectbox(
            "Account Category Filter:",
            [
                "Typical Legitimate Accounts (Normal Flow)",
                "Suspicious Accounts (Flagged)",
                "High-Volume Legitimate (False Positive Edge Cases)",
                "Balanced Mix (Suspicious + Legitimate)",
                "Manual Account Search"
            ]
        )
    with c_flt2:
        if acc_type_filter == "Manual Account Search":
            search_acc = st.text_input("Enter exact or partial Account ID:", "")
        else:
            search_acc = st.text_input("Filter list by Account ID:", "")

    # Query accounts based on user selection
    if acc_type_filter == "Typical Legitimate Accounts (Normal Flow)":
        base_query = "SELECT account_id, is_suspicious, risk_score, pattern_type, total_tx_count, balance, is_blocked FROM accounts WHERE is_suspicious = 0 AND total_tx_count BETWEEN 2 AND 8"
    elif acc_type_filter == "Suspicious Accounts (Flagged)":
        base_query = "SELECT account_id, is_suspicious, risk_score, pattern_type, total_tx_count, balance, is_blocked FROM accounts WHERE is_suspicious = 1"
    elif acc_type_filter == "High-Volume Legitimate (False Positive Edge Cases)":
        base_query = "SELECT account_id, is_suspicious, risk_score, pattern_type, total_tx_count, balance, is_blocked FROM accounts WHERE is_suspicious = 0 AND total_tx_count >= 15"
    elif acc_type_filter == "Balanced Mix (Suspicious + Legitimate)" and not search_acc.strip():
        q_susp = "SELECT account_id, is_suspicious, risk_score, pattern_type, total_tx_count, balance, is_blocked FROM accounts WHERE is_suspicious = 1 ORDER BY total_tx_count DESC LIMIT 25"
        q_legit = "SELECT account_id, is_suspicious, risk_score, pattern_type, total_tx_count, balance, is_blocked FROM accounts WHERE is_suspicious = 0 AND total_tx_count BETWEEN 2 AND 8 LIMIT 25"
        with db.get_connection() as conn:
            df_s = pd.read_sql(q_susp, conn)
            df_l = pd.read_sql(q_legit, conn)
            sample_accs = pd.concat([df_s, df_l], ignore_index=True)
        base_query = None
    else:
        base_query = "SELECT account_id, is_suspicious, risk_score, pattern_type, total_tx_count, balance, is_blocked FROM accounts WHERE 1=1"

    if base_query:
        if search_acc.strip():
            base_query += f" AND account_id LIKE '%{search_acc.strip()}%'"
        base_query += " ORDER BY total_tx_count DESC LIMIT 100"
        with db.get_connection() as conn:
            sample_accs = pd.read_sql(base_query, conn)

    if sample_accs.empty:
        st.warning("No matching accounts found for the current filter/search.")
        st.stop()

    account_options = sample_accs["account_id"].tolist()
    
    def format_acc_label(acc_id):
        row = sample_accs[sample_accs["account_id"] == acc_id].iloc[0]
        status = "SUSPICIOUS" if row["is_suspicious"] == 1 else "LEGITIMATE"
        blk = " [🔴 BLOCKED]" if row.get("is_blocked") == 1 else ""
        tx_cnt = int(row["total_tx_count"]) if pd.notnull(row["total_tx_count"]) else 0
        return f"{acc_id}  |  [{status}]{blk}  |  Txs: {tx_cnt:,}  |  Typology: {row['pattern_type']}"

    selected_account = st.selectbox(
        "Select Account to Predict:",
        account_options,
        format_func=format_acc_label
    )
    row_info = sample_accs[sample_accs["account_id"] == selected_account].iloc[0]

    blk_text = "🔴 BLOCKED" if row_info.get("is_blocked") == 1 else "🟢 ACTIVE"
    st.info(f"Target Account: **{selected_account}** | Ground Truth: **{'SUSPICIOUS' if row_info['is_suspicious'] == 1 else 'LEGITIMATE'}** | Status: **{blk_text}** | Pattern: **{row_info['pattern_type']}**")

    # High-volume edge case advisory
    if row_info['is_suspicious'] == 0 and pd.notnull(row_info['total_tx_count']) and row_info['total_tx_count'] >= 15:
        st.warning("⚠️ **Compliance Anomaly Advisory:** This account is labeled Legitimate in ground truth, but has an exceptionally high volume of incoming transactions / unique senders (outlier activity). In financial crime surveillance, such high-volume commercial accounts frequently trigger **False Positive** alerts across ML models due to topology resemblance with fan-in / mule gathering rings.")

    # Fetch account features
    with db.get_connection() as conn:
        feat_row = pd.read_sql("SELECT * FROM accounts WHERE account_id = ?", conn, params=(selected_account,))

    # Construct input feature vector
    input_df = pd.DataFrame([{col: feat_row[col].iloc[0] if col in feat_row.columns else 0.0 for col in FEATURE_COLUMNS}])

    with st.expander("🔍 View Account Features Used for Inference (Understandable Breakdown)"):
        feat_display = []
        for col in FEATURE_COLUMNS:
            val = feat_row[col].iloc[0] if col in feat_row.columns else 0.0
            meta = FEATURE_METADATA.get(col, {})
            if "amount" in col or "flow" in col:
                val_str = f"${val:,.2f}"
            elif "count" in col or "degree" in col or "senders" in col or "receivers" in col:
                val_str = f"{val:,.0f}"
            else:
                val_str = f"{val:.4f}"

            feat_display.append({
                "Understandable Feature Name": meta.get("label", format_feature_name(col)),
                "Category": meta.get("category", "General"),
                "Account Value": val_str,
                "Technical Key": col,
                "Investigative Meaning": meta.get("description", "")
            })
        st.dataframe(pd.DataFrame(feat_display), use_container_width=True)

    if st.button("🚀 Run Live ML Inference (Combined XGBoost + GraphSAGE)", type="primary"):
        # 1. Primary Flagship Model: Combined XGBoost + GraphSAGE
        comb_model = models.get("Combined (XGBoost + GraphSAGE)")
        xgb_model = models.get("XGBoost")
        sage_model = models.get("GraphSAGE")

        prob_comb = 0.0
        th_comb = 0.50
        if comb_model:
            prob_comb = float(comb_model.predict_proba(input_df)[0])
            th_comb = float(comb_model.optimal_threshold)
            pred_comb = int(prob_comb >= th_comb)
            
            st.markdown("### 🔮 Flagship Model: Combined (XGBoost + GraphSAGE)")
            st.caption("✨ Combines XGBoost tabular flow analysis with GraphSAGE neighborhood graph embeddings, calibrated on PR-AUC.")

            cf1, cf2, cf3, cf4 = st.columns(4)
            with cf1:
                st.metric("Combined Laundering Risk", f"{prob_comb*100:.2f}%")
            with cf2:
                status_str = "🚨 SUSPICIOUS" if pred_comb == 1 else "✅ LEGITIMATE"
                st.metric("Ensemble Classification", status_str)
            with cf3:
                st.metric("PR-AUC Calibrated Threshold", f"{th_comb*100:.1f}%")
            with cf4:
                st.metric("Component Blend", f"{comb_model.xgb_weight*100:.0f}% XGB + {comb_model.sage_weight*100:.0f}% SAGE")
            st.divider()

        # 2. Individual Component Breakdown
        st.markdown("### 🔬 Sub-Model Detection Breakdown")
        res_cols = st.columns(2)
        probs_list = []
        if prob_comb > 0:
            probs_list.append(prob_comb)

        # Standalone XGBoost
        if xgb_model:
            prob_xgb = float(xgb_model.predict_proba(input_df)[0])
            pred_xgb = int(prob_xgb >= 0.5)
            probs_list.append(prob_xgb)
            with res_cols[0]:
                st.subheader("⚡ XGBoost (Tabular Flows)")
                st.metric("Classification", "SUSPICIOUS" if pred_xgb == 1 else "LEGITIMATE")
                st.metric("Laundering Probability", f"{prob_xgb*100:.2f}%")

        # Standalone GraphSAGE
        if sage_model:
            prob_sage = float(sage_model.predict_proba(input_df)[0])
            pred_sage = int(prob_sage >= 0.5)
            probs_list.append(prob_sage)
            with res_cols[1]:
                st.subheader("🕸️ GraphSAGE (GNN Network)")
                st.metric("Classification", "SUSPICIOUS" if pred_sage == 1 else "LEGITIMATE")
                st.metric("Laundering Probability", f"{prob_sage*100:.2f}%")

        # Threshold compliance rule summary
        eval_p = prob_comb if prob_comb > 0 else (max(probs_list) if probs_list else 0.0)
        st.divider()
        if eval_p >= 0.50:
            st.error(f"🚨 **Automated Banking Rule Triggered:** Laundering probability is **{eval_p*100:.2f}%** (≥ 50.0%). In the Banking System, transfers from this account will be intercepted and restricted.")
        else:
            st.success(f"✅ **Automated Banking Rule Passed:** Laundering probability is **{eval_p*100:.2f}%** (< 50.0%). This account is permitted to transact in the Banking System.")


# ----------------------------------------------------
# 6. MODEL BENCHMARKS
# ----------------------------------------------------
elif menu == "📈 Model Benchmarks":
    st.title("📈 Machine Learning Model Benchmarks")
    st.write("Threshold-independent ranking evaluation metrics (**ROC-AUC** and **PR-AUC**) stored in SQLite (`model_metrics` table).")

    # PR-AUC Focus Rationale Banner
    st.info("""
    🎯 **Primary Evaluation Focus: Precision-Recall AUC (PR-AUC) & ROC-AUC**
    - **Why Point Metrics Are Excluded:** In AML surveillance where laundering accounts for **~0.1% of transactions**, raw accuracy is dominated by True Negatives ($TN$) and single-threshold metrics depend on an arbitrary cutoff.
    - **PR-AUC (Primary Decision Metric):** Evaluates **Precision** (avoiding false alarms) against **Recall** (catching real money launderers) across all thresholds while completely ignoring $TN$.
    - **ROC-AUC (Global Ranking Metric):** Evaluates the probability that a randomly selected laundering account receives a higher risk score than a randomly selected legitimate account ($P(\\hat{p}_1 > \\hat{p}_0)$).
    """)

    metrics_df = db.get_model_metrics()
    if not metrics_df.empty:
        # Sort by PR-AUC descending to put the best imbalanced detector at top
        metrics_sorted = metrics_df.sort_values(by="pr_auc", ascending=False).reset_index(drop=True)
        st.dataframe(metrics_sorted[["model_name", "roc_auc", "pr_auc"]], use_container_width=True)

        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            st.subheader("🎯 PR-AUC (Precision-Recall AUC) Comparison")
            st.bar_chart(metrics_sorted.set_index("model_name")["pr_auc"])
            st.caption("Primary metric: measures positive-class precision-recall trade-off under extreme class imbalance.")

        with col_ch2:
            st.subheader("📈 ROC-AUC (Global Ranking Discrimination)")
            st.bar_chart(metrics_sorted.set_index("model_name")["roc_auc"])
            st.caption("Secondary metric: measures global pairwise ranking separation between suspicious and legitimate accounts.")

        st.subheader("ROC-AUC vs. PR-AUC Comparison Across Models")
        chart_data = metrics_sorted[["model_name", "pr_auc", "roc_auc"]].set_index("model_name")
        st.bar_chart(chart_data)
    else:
        st.warning("No model metrics found. Please run the training pipeline first.")


# ----------------------------------------------------
# 7. EXPLAINABILITY & TYPOLOGIES
# ----------------------------------------------------
elif menu == "🧠 Explainability & Typologies":
    st.title("🧠 Model Explainability & AML Typologies")
    st.write("Human-understandable feature importance rankings, plain-English definitions, and documented laundering typology breakdown.")

    selected_model = st.selectbox(
        "Select Model for Feature Importances:",
        ["Combined (XGBoost + GraphSAGE)", "XGBoost", "GraphSAGE"]
    )
    fi_df = db.get_top_features(selected_model, top_n=15)
    
    if not fi_df.empty:
        # Enrich with human-understandable labels, categories, and descriptions
        fi_df["Readable Name"] = fi_df["feature_name"].map(lambda x: FEATURE_METADATA.get(x, {}).get("label", format_feature_name(x)))
        fi_df["Category"] = fi_df["feature_name"].map(lambda x: FEATURE_METADATA.get(x, {}).get("category", "General Metric"))
        fi_df["Importance (%)"] = fi_df["importance"].map(lambda x: f"{x*100:.2f}%")
        fi_df["Investigative Significance"] = fi_df["feature_name"].map(lambda x: FEATURE_METADATA.get(x, {}).get("description", "Predictive behavioral signal."))

        # Highlight Top 3 Key Decision Drivers
        st.subheader("🏆 Top Decision Drivers (Human-Understandable Signals)")
        top3_cols = st.columns(3)
        for i in range(min(3, len(fi_df))):
            row_i = fi_df.iloc[i]
            with top3_cols[i]:
                st.metric(
                    label=f"Rank #{row_i['rank']}: {row_i['Category']}",
                    value=row_i['Readable Name'],
                    delta=f"{row_i['importance']*100:.2f}% influence",
                    delta_color="normal"
                )
                st.caption(f"**Variable:** `{row_i['feature_name']}`\n\n{row_i['Investigative Significance']}")

        st.divider()

        # Bar Chart with Understandable Feature Names
        st.subheader(f"📊 Feature Importance Distribution: {selected_model}")
        chart_df = fi_df[["Readable Name", "importance"]].set_index("Readable Name")
        st.bar_chart(chart_df)
        st.caption("Ordered by relative contribution to the model's laundering prediction.")

        # Comprehensive Table with Human Labels and Plain English Explanations
        st.subheader("📋 Comprehensive Feature Breakdown & AML Significance")
        display_cols = ["rank", "Readable Name", "Category", "Importance (%)", "feature_name", "Investigative Significance"]
        renamed_cols = {
            "rank": "Rank",
            "Readable Name": "Understandable Feature Name",
            "Category": "Analytical Category",
            "Importance (%)": "Weight Contribution",
            "feature_name": "Technical Key (Code)",
            "Investigative Significance": "Why This Signal Matters in AML"
        }
        st.dataframe(fi_df[display_cols].rename(columns=renamed_cols), use_container_width=True)

        # Feature Glossary Deep-Dive
        with st.expander("📖 Interactive Feature Glossary (What each signal measures)"):
            glossary_feat = st.selectbox(
                "Select Feature to inspect investigative details:",
                fi_df["feature_name"].tolist(),
                format_func=lambda x: f"{FEATURE_METADATA.get(x, {}).get('label', x)} ({x})"
            )
            meta = FEATURE_METADATA.get(glossary_feat, {})
            st.markdown(f"### **{meta.get('label', glossary_feat)}**")
            st.markdown(f"- **Technical Name in Code:** `{glossary_feat}`")
            st.markdown(f"- **Analytical Category:** **{meta.get('category', 'General')}**")
            st.markdown(f"- **Investigative Meaning:** {meta.get('description', 'No description available.')}")
            
            if glossary_feat == "ach_tx_count":
                st.info("💡 **Why is ACH Count important in AML?** Automated Clearing House (ACH) transfers are processed in electronic batches. Money launderers frequently exploit ACH channels because single transfers blend in with legitimate payroll or merchant debits, using mule gathering accounts to collect dispersed structuring deposits.")
            elif "degree" in glossary_feat or "centrality" in glossary_feat:
                st.info("💡 **Why is Graph Topology important?** Criminal organizations rely on networks (mule rings, cycles, fan-outs). Graph metrics identify accounts that act as central bottlenecks or high-volume distributors regardless of transaction dollar amounts.")
            elif "flow" in glossary_feat or "ratio" in glossary_feat:
                st.info("💡 **Why are Flow Ratios important?** Legitimate businesses maintain working capital reserves. In contrast, laundering 'mule' accounts typically pass money through with near-zero retained balance, resulting in in-to-out ratios hovering close to 1.0.")

    st.subheader("Documented Typologies in Synthetic AML Dataset")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("""
        **1. Fan-Out (Structuring / Layering)**
        - One master account rapidly disburses funds across numerous secondary accounts to avoid triggering single-transaction threshold reporting.
        - High `unique_receivers` and `total_tx_out` asymmetry.
        
        **2. Cycle (Circular Routing)**
        - Funds transfer in a closed loop (Account A -> B -> C -> D -> A) across different banks and currencies to obscure fund origins.
        """)
    with t2:
        st.markdown("""
        **3. Gather-Scatter (Pass-through Mule Hub)**
        - Multiple source accounts funnel money into a central aggregation node, which subsequently scatters the consolidated capital to destinations.
        
        **4. Stack (Chain Layering)**
        - Sequential hops between accounts across jurisdictions and payment formats (ACH -> Wire -> Cheque).
        """)