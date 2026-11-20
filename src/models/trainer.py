"""
AML Model Trainer & Evaluation Orchestrator.
Trains XGBoost, GraphSAGE, and the Combined Ensemble on an 80/20 train/test split,
evaluates them using ROC-AUC and PR-AUC, and saves results to SQLite and disk.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
import sys

# Add project root folder so imports work from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import FEATURE_COLUMNS, RANDOM_STATE, TEST_SIZE, SAVED_MODELS_DIR
from src.database.db_manager import AMLDatabase
from src.models.xgboost_model import AMLXGBoost
from src.models.graphsage_model import AMLGraphSAGE
from src.models.combined_model import AMLCombinedXGBoostGraphSAGE


def train_and_evaluate_all(
    df_features: pd.DataFrame,
    df_acc_profile: pd.DataFrame = None,
    df_tx: pd.DataFrame = None,
    db: AMLDatabase = None
) -> Dict[str, Any]:
    """Train all 3 AML models, evaluate ROC-AUC and PR-AUC, and save everything to SQLite."""
    print("\n" + "=" * 60)
    print("STARTING AML MODEL TRAINING & PR-AUC CALIBRATION PIPELINE")
    print("=" * 60)

    if db is None:
        db = AMLDatabase()

    # Remove duplicate account rows if any exist
    if df_features["account_id"].nunique() < len(df_features):
        df_features = (
            df_features.sort_values("is_suspicious", ascending=False)
            .drop_duplicates(subset=["account_id"], keep="first")
            .reset_index(drop=True)
        )

    # Assign every suspicious account and laundering transaction to one of the 8 laundering patterns
    from src.data.loader import resolve_laundering_typologies, parse_patterns
    with db.get_connection() as conn:
        tx_for_pats = df_tx if df_tx is not None else pd.read_sql("SELECT * FROM transactions", conn)
        resolved_pats = resolve_laundering_typologies(df_features, tx_for_pats, parse_patterns())
        df_features["pattern_type"] = np.where(
            df_features["is_suspicious"] == 1,
            df_features["account_id"].map(resolved_pats).fillna("STACK"),
            "NONE"
        )
        tx_db = pd.read_sql("SELECT * FROM transactions", conn)
        tx_db["pattern_type"] = np.where(
            tx_db["is_laundering"] == 1,
            tx_db["from_account"].map(resolved_pats)
            .fillna(tx_db["to_account"].map(resolved_pats))
            .fillna("STACK"),
            tx_db["pattern_type"]
        )
        tx_db.to_sql("transactions", conn, if_exists="replace", index=False)
        conn.commit()

    # Build the 30-column feature table (X) and target label array (y)
    for col in FEATURE_COLUMNS:
        if col not in df_features.columns:
            df_features[col] = 0.0

    X = df_features[FEATURE_COLUMNS].copy().fillna(0.0)
    y = df_features["is_suspicious"].values
    account_ids = df_features["account_id"].values

    print(f"Total dataset: {len(X):,} accounts ({np.sum(y == 1):,} suspicious, {np.sum(y == 0):,} legitimate)")

    # Split data: 80% for training, 20% for testing (stratify=y keeps the exact same 3.08% laundering ratio in both)
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_accs = account_ids[test_idx]

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_records = []
    explanations_records = []
    preds_list = []

    # 1. Train XGBoost (260 trees, depth 7, learning rate 0.05)
    print("\n[1/3] Training & Calibrating XGBoost (Tabular & Flow Model)...")
    xgb_model = AMLXGBoost(n_estimators=260, max_depth=7, learning_rate=0.05)
    xgb_model.fit(X_train, y_train)
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    xgb_metrics["trained_at"] = timestamp_str
    xgb_metrics["dataset_name"] = "HI-Small"
    metrics_records.append(xgb_metrics)
    xgb_model.save(SAVED_MODELS_DIR / "xgboost.joblib")

    xgb_fi = xgb_model.get_feature_importances()
    for _, row in xgb_fi.iterrows():
        explanations_records.append({
            "model_name": "XGBoost",
            "feature_name": row["feature_name"],
            "importance": float(row["importance"]),
            "rank": int(row["rank"])
        })

    xgb_probs = xgb_model.predict_proba(X_test)
    xgb_preds = (xgb_probs >= 0.50).astype(int)
    for acc, t, p, prob in zip(test_accs, y_test, xgb_preds, xgb_probs):
        preds_list.append({
            "model_name": "XGBoost",
            "account_id": acc,
            "true_label": int(t),
            "predicted_label": int(p),
            "predicted_prob": round(float(prob), 4),
            "created_at": timestamp_str
        })
    print(f"XGBoost Metrics -> ROC-AUC: {xgb_metrics['roc_auc']:.4f} | PR-AUC: {xgb_metrics['pr_auc']:.4f}")

    # 2. Train GraphSAGE (128 hidden units, 0.005 learning rate, 16 epochs)
    print("\n[2/3] Training & Calibrating GraphSAGE (Inductive Multi-Hop GNN)...")
    sage = AMLGraphSAGE(hidden_dim=128, lr=0.005, epochs=16)
    sage.fit(X_train, y_train)
    sage_metrics = sage.evaluate(X_test, y_test)
    sage_metrics["trained_at"] = timestamp_str
    sage_metrics["dataset_name"] = "HI-Small"
    metrics_records.append(sage_metrics)
    sage.save(SAVED_MODELS_DIR / "graphsage.pt")

    sage_fi = sage.get_feature_importances()
    for _, row in sage_fi.iterrows():
        explanations_records.append({
            "model_name": "GraphSAGE",
            "feature_name": row["feature_name"],
            "importance": float(row["importance"]),
            "rank": int(row["rank"])
        })

    sage_probs = sage.predict_proba(X_test)
    sage_preds = (sage_probs >= 0.50).astype(int)
    for acc, t, p, prob in zip(test_accs, y_test, sage_preds, sage_probs):
        preds_list.append({
            "model_name": "GraphSAGE",
            "account_id": acc,
            "true_label": int(t),
            "predicted_label": int(p),
            "predicted_prob": round(float(prob), 4),
            "created_at": timestamp_str
        })
    print(f"GraphSAGE Metrics -> ROC-AUC: {sage_metrics['roc_auc']:.4f} | PR-AUC: {sage_metrics['pr_auc']:.4f}")

    # 3. Build Combined Model (50% XGBoost + 50% GraphSAGE, 50% risk cutoff)
    print("\n[3/3] Constructing & Calibrating Combined (XGBoost + GraphSAGE) Ensemble...")
    combined = AMLCombinedXGBoostGraphSAGE(xgb_weight=0.50, sage_weight=0.50, optimal_threshold=0.50)
    combined.xgb_model = xgb_model
    combined.sage_model = sage
    combined.feature_names = list(X.columns)

    comb_metrics = combined.evaluate(X_test, y_test)
    comb_metrics["trained_at"] = timestamp_str
    comb_metrics["dataset_name"] = "HI-Small"
    metrics_records.append(comb_metrics)
    combined.save(SAVED_MODELS_DIR / "combined_model.joblib")

    comb_fi = combined.get_feature_importances()
    for _, row in comb_fi.iterrows():
        explanations_records.append({
            "model_name": "Combined (XGBoost + GraphSAGE)",
            "feature_name": row["feature_name"],
            "importance": float(row["importance"]),
            "rank": int(row["rank"])
        })

    comb_probs = combined.predict_proba(X_test)
    comb_preds = (comb_probs >= 0.50).astype(int)
    for acc, t, p, prob in zip(test_accs, y_test, comb_preds, comb_probs):
        preds_list.append({
            "model_name": "Combined (XGBoost + GraphSAGE)",
            "account_id": acc,
            "true_label": int(t),
            "predicted_label": int(p),
            "predicted_prob": round(float(prob), 4),
            "created_at": timestamp_str
        })
    print(f"Combined (XGBoost + GraphSAGE) Metrics -> ROC-AUC: {comb_metrics['roc_auc']:.4f} | PR-AUC: {comb_metrics['pr_auc']:.4f}")

    # Save calibrated Combined risk score on every account in SQLite
    print("\nUpdating account risk scores in SQLite with calibrated Combined probabilities...")
    all_comb_probs = np.round(combined.predict_proba(X), 4)
    df_features["risk_score"] = all_comb_probs
    with db.get_connection() as conn:
        df_features.to_sql("accounts", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_acc_id ON accounts(account_id)")
        conn.commit()

    # Save ROC-AUC / PR-AUC metrics, feature importances, and sample predictions in SQLite
    print("Persisting benchmark artifacts into SQLite Database...")
    db.save_model_metrics(metrics_records)
    db.save_feature_explanations(explanations_records)

    df_preds_all = pd.DataFrame(preds_list)
    db.save_predictions(df_preds_all.iloc[:20000])

    print("Pipeline training & calibration completed successfully!")
    return {
        "metrics": metrics_records,
        "models": ["XGBoost", "GraphSAGE", "Combined (XGBoost + GraphSAGE)"]
    }
