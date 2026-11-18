"""
End-to-end AML Pipeline Execution Script.
Runs Data Processing -> Feature Engineering -> Model Training -> Database Storage.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_SAMPLE_SIZE
from src.database.db_manager import AMLDatabase
from src.data.loader import load_accounts, load_transactions_sample, parse_patterns
from src.features.feature_engineering import extract_account_features
from src.models.trainer import train_and_evaluate_all

def run_pipeline(sample_size: int = DEFAULT_SAMPLE_SIZE):
    start_time = time.time()
    print("="*70)
    print(" ANTI-MONEY LAUNDERING (AML) DATA & ML PIPELINE")
    print(f" Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Configured Transaction Sample Size: {sample_size:,}")
    print("="*70)

    # 1. Initialize SQLite Database
    print("\n[Step 1/5] Initializing SQLite Database...")
    db = AMLDatabase()
    print(f"Database ready at: {db.db_path}")

    # 2. Load Raw AML Dataset & Patterns
    print("\n[Step 2/5] Loading AML Dataset & Typology Patterns...")
    patterns = parse_patterns()
    print(f"Extracted {len(patterns):,} account-pattern associations.")

    df_accounts = load_accounts()
    print(f"Loaded {len(df_accounts):,} total account profiles.")

    df_tx = load_transactions_sample(sample_size=sample_size, include_all_laundering=True)
    df_tx["pattern_type"] = df_tx["from_account"].map(patterns).fillna(df_tx["to_account"].map(patterns)).fillna("NONE")

    # 3. Feature Engineering & Graph Analytics
    print("\n[Step 3/5] Extracting Financial, Flow & Graph Topological Features...")
    df_features, df_acc_profile = extract_account_features(df_tx, df_accounts, patterns)

    resolved_acc_pats = dict(
        zip(
            df_acc_profile[df_acc_profile["is_suspicious"] == 1]["account_id"],
            df_acc_profile[df_acc_profile["is_suspicious"] == 1]["pattern_type"]
        )
    )
    df_tx["pattern_type"] = np.where(
        df_tx["is_laundering"] == 1,
        df_tx["from_account"].map(resolved_acc_pats).fillna(df_tx["to_account"].map(resolved_acc_pats)).fillna("STACK"),
        "NONE"
    )

    # 4. Ingest Accounts and Transactions into SQLite
    print("\n[Step 4/5] Storing Accounts and Transactions in SQLite Database...")
    # Store accounts profile
    db.insert_accounts(df_acc_profile, if_exists="replace")
    print(f"Stored {len(df_acc_profile):,} account profiles in SQLite.")

    # Map raw numeric bank IDs to actual human-readable bank names
    acc_to_bank = dict(zip(df_accounts["account_id"].astype(str).str.strip(), df_accounts["bank_name"].str.strip()))
    id_to_bank = dict(zip(df_accounts["bank_id"].dropna().astype(str).str.strip(), df_accounts["bank_name"].str.strip()))

    def resolve_name(raw_b, acc_id):
        b_str = str(raw_b).strip() if raw_b is not None else ""
        if any(c.isalpha() for c in b_str):
            return b_str
        if acc_id in acc_to_bank and any(c.isalpha() for c in acc_to_bank[acc_id]):
            return acc_to_bank[acc_id]
        if b_str in id_to_bank and any(c.isalpha() for c in id_to_bank[b_str]):
            return id_to_bank[b_str]
        return f"Bank #{b_str}" if b_str else "Unknown Bank"

    # Store transactions sample in SQLite
    tx_to_store = df_tx[[
        "timestamp", "from_bank", "from_account", "to_bank", "to_account",
        "amount_received", "receiving_currency", "amount_paid", "payment_currency",
        "payment_format", "is_laundering", "pattern_type"
    ]].copy()

    tx_to_store["from_bank"] = [resolve_name(b, a) for b, a in zip(tx_to_store["from_bank"], tx_to_store["from_account"])]
    tx_to_store["to_bank"] = [resolve_name(b, a) for b, a in zip(tx_to_store["to_bank"], tx_to_store["to_account"])]

    # Store all laundering transactions + sample of legitimate transactions
    laund_tx = tx_to_store[tx_to_store["is_laundering"] == 1]
    legit_sample = tx_to_store[tx_to_store["is_laundering"] == 0].head(50000)
    db_tx_sample = pd.concat([laund_tx, legit_sample], ignore_index=True)
    db.insert_transactions(db_tx_sample, if_exists="replace")
    print(f"Stored {len(db_tx_sample):,} transactions ({len(laund_tx):,} laundering) with resolved bank names in SQLite.")

    # 5. Train & Evaluate Models (XGBoost, GraphSAGE, and Combined Ensemble)
    print("\n[Step 5/5] Training ML Models & Computing Metrics...")
    results = train_and_evaluate_all(df_features, df_acc_profile, df_tx, db)

    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print(" PIPELINE EXECUTION COMPLETED")
    print(f" Total Elapsed Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print("="*70)

    # Print summary metrics table
    metrics_df = db.get_model_metrics()
    print("\nEVALUATION BENCHMARK METRICS (SQLite model_metrics):")
    print(metrics_df[["model_name", "roc_auc", "pr_auc"]].to_string(index=False))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Data & ML Pipeline")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Transaction sample limit")
    args = parser.parse_args()
    run_pipeline(sample_size=args.sample_size)