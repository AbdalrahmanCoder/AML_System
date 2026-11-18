"""
Feature Engineering for AML Detection.
Computes account-level financial, flow, behavioral, and graph topological features.
"""
import pandas as pd
import numpy as np
import networkx as nx
from typing import Tuple, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import FEATURE_COLUMNS

def extract_account_features(
    df_tx: pd.DataFrame,
    df_acc: pd.DataFrame,
    account_patterns: Dict[str, str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes behavioral and network features per account based on transactions.
    Returns:
        df_features: DataFrame of engineered features per account for ML models
        df_acc_profile: DataFrame of enriched accounts with suspicion labels, risk metrics & typologies
    """
    account_patterns = account_patterns or {}
    print("Building account transaction aggregations...")

    # Group incoming transactions
    in_agg = df_tx.groupby("to_account").agg(
        total_tx_in=("amount_received", "count"),
        total_amount_in=("amount_received", "sum"),
        avg_amount_in=("amount_received", "mean"),
        max_amount_in=("amount_received", "max"),
        std_amount_in=("amount_received", "std"),
        unique_senders=("from_account", "nunique"),
        laundering_in=("is_laundering", "sum")
    ).reset_index().rename(columns={"to_account": "account_id"})

    # Group outgoing transactions
    out_agg = df_tx.groupby("from_account").agg(
        total_tx_out=("amount_paid", "count"),
        total_amount_out=("amount_paid", "sum"),
        avg_amount_out=("amount_paid", "mean"),
        max_amount_out=("amount_paid", "max"),
        std_amount_out=("amount_paid", "std"),
        unique_receivers=("to_account", "nunique"),
        laundering_out=("is_laundering", "sum")
    ).reset_index().rename(columns={"from_account": "account_id"})

    # Payment formats per account (both in and out)
    fmt_in = df_tx.pivot_table(index="to_account", columns="payment_format", values="amount_received", aggfunc="count", fill_value=0)
    fmt_out = df_tx.pivot_table(index="from_account", columns="payment_format", values="amount_paid", aggfunc="count", fill_value=0)
    
    # Merge aggregations
    df_features = pd.merge(in_agg, out_agg, on="account_id", how="outer").fillna(0.0)

    # Fill basic flow features
    df_features["total_tx_count"] = df_features["total_tx_in"] + df_features["total_tx_out"]
    df_features["net_flow"] = df_features["total_amount_in"] - df_features["total_amount_out"]
    
    # Ratios
    df_features["in_out_tx_ratio"] = (df_features["total_tx_in"] + 1) / (df_features["total_tx_out"] + 1)
    df_features["in_out_amount_ratio"] = (df_features["total_amount_in"] + 1) / (df_features["total_amount_out"] + 1)

    # Standardize format counts
    fmt_combined = fmt_in.add(fmt_out, fill_value=0)
    df_features["ach_tx_count"] = df_features["account_id"].map(fmt_combined.get("ACH", pd.Series(dtype=float))).fillna(0)
    df_features["wire_tx_count"] = df_features["account_id"].map(fmt_combined.get("Wire", pd.Series(dtype=float))).fillna(0)
    df_features["cheque_tx_count"] = df_features["account_id"].map(fmt_combined.get("Cheque", pd.Series(dtype=float))).fillna(0)
    df_features["reinvestment_tx_count"] = df_features["account_id"].map(fmt_combined.get("Reinvestment", pd.Series(dtype=float))).fillna(0)

    # Sub-$10k structuring ("smurfing") transaction counts ($2,500 - $9,999.99)
    smurf_tx = df_tx[(df_tx["amount_paid"] >= 2500.0) & (df_tx["amount_paid"] < 10000.0)]
    smurf_tot = smurf_tx.groupby("to_account").size().add(smurf_tx.groupby("from_account").size(), fill_value=0)
    df_features["smurfing_tx_count"] = df_features["account_id"].map(smurf_tot).fillna(0.0)

    # Build transaction network graph for topological features
    print("Computing graph network topology features...")
    G = nx.DiGraph()
    # Add edges with weights
    edge_weights = df_tx.groupby(["from_account", "to_account"])["amount_paid"].sum().reset_index()
    for _, row in edge_weights.iterrows():
        G.add_edge(row["from_account"], row["to_account"], weight=row["amount_paid"])

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    deg_cent = nx.degree_centrality(G)

    df_features["in_degree"] = df_features["account_id"].map(in_deg).fillna(0).astype(int)
    df_features["out_degree"] = df_features["account_id"].map(out_deg).fillna(0).astype(int)
    df_features["degree_centrality"] = df_features["account_id"].map(deg_cent).fillna(0.0).astype(float)

    # Compute 1-hop and 2-hop sparse neighborhood aggregations
    import scipy.sparse as sp
    acc_map = {a: i for i, a in enumerate(df_features["account_id"])}
    valid_tx = df_tx[df_tx["from_account"].isin(acc_map) & df_tx["to_account"].isin(acc_map)]
    u = valid_tx["from_account"].map(acc_map).values
    v = valid_tx["to_account"].map(acc_map).values
    n_accs = len(df_features)
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    vals = np.ones(len(rows), dtype=np.float32)
    adj_no_self = sp.coo_matrix((vals, (rows, cols)), shape=(n_accs, n_accs)).tocsr()
    deg_arr = np.array(adj_no_self.sum(axis=1)).flatten()
    deg_safe = np.where(deg_arr == 0, 1.0, deg_arr)
    norm_adj = sp.diags(1.0 / deg_safe).dot(adj_no_self)

    df_features["neighbor_ach_avg"] = norm_adj.dot(df_features["ach_tx_count"].values.astype(np.float32))
    df_features["neighbor_reinv_avg"] = norm_adj.dot(df_features["reinvestment_tx_count"].values.astype(np.float32))
    df_features["neighbor_smurf_avg"] = norm_adj.dot(df_features["smurfing_tx_count"].values.astype(np.float32))
    df_features["neighbor_degree_avg"] = norm_adj.dot((df_features["in_degree"] + df_features["out_degree"]).values.astype(np.float32))
    df_features["neighbor_tx_avg"] = norm_adj.dot(df_features["total_tx_count"].values.astype(np.float32))
    df_features["hop2_ach_avg"] = norm_adj.dot(df_features["neighbor_ach_avg"].values.astype(np.float32))

    # Ground truth labels
    total_laundering_tx = df_features["laundering_in"] + df_features["laundering_out"]
    df_features["is_suspicious"] = (total_laundering_tx > 0).astype(int)
    df_features["laundering_ratio"] = np.where(
        df_features["total_tx_count"] > 0,
        total_laundering_tx / df_features["total_tx_count"],
        0.0
    )

    # Derive suspicion category
    def get_category(row):
        ratio = row["laundering_ratio"]
        if ratio == 0.0:
            return "Legitimate"
        elif ratio >= 1.0:
            return "Highly Suspicious (100%)"
        else:
            return f"Partially Suspicious ({int(ratio * 100)}%)"

    df_features["suspicion_category"] = df_features.apply(get_category, axis=1)
    df_features["pattern_type"] = df_features["account_id"].map(account_patterns).fillna("NONE")

    # Ensure all target columns in FEATURE_COLUMNS exist
    for col in FEATURE_COLUMNS:
        if col not in df_features.columns:
            df_features[col] = 0.0

    # Merge with accounts catalog (bank name, entity name) and preserve all ML feature columns
    cols_to_profile = ["account_id", "is_suspicious", "suspicion_category", "laundering_ratio", "pattern_type"]
    for col in FEATURE_COLUMNS:
        if col not in cols_to_profile:
            cols_to_profile.append(col)

    df_acc_unique = df_acc.drop_duplicates(subset=["account_id"], keep="first")
    df_acc_profile = pd.merge(
        df_features[cols_to_profile],
        df_acc_unique,
        on="account_id",
        how="left"
    )
    df_acc_profile["risk_score"] = df_features["laundering_ratio"].round(4)
    df_acc_profile["bank_name"] = df_acc_profile["bank_name"].fillna("Unknown Bank")
    df_acc_profile["entity_name"] = df_acc_profile["entity_name"].fillna("Individual Account")
    df_acc_profile["bank_id"] = df_acc_profile["bank_id"].fillna("000")
    df_acc_profile["entity_id"] = df_acc_profile["entity_id"].fillna("000")

    from src.data.loader import resolve_laundering_typologies
    resolved_pats = resolve_laundering_typologies(df_acc_profile, df_tx, account_patterns)
    df_acc_profile["pattern_type"] = np.where(
        df_acc_profile["is_suspicious"] == 1,
        df_acc_profile["account_id"].map(resolved_pats).fillna("STACK"),
        "NONE"
    )
    df_features["pattern_type"] = df_acc_profile["pattern_type"].values

    print(f"Engineered features for {len(df_features):,} active accounts.")
    print(f"Suspicious accounts identified: {df_features['is_suspicious'].sum():,} ({df_features['is_suspicious'].mean()*100:.2f}%)")
    
    return df_features, df_acc_profile