"""
AML Dataset Loader & Laundering Pattern Parser.
Loads accounts, transactions, and laundering patterns from the raw dataset files.
Also assigns every suspicious account to one of the 8 laundering ring patterns.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import re
import sys

# Add project root folder so imports work from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import ACCOUNTS_CSV, TRANSACTIONS_CSV, PATTERNS_TXT


def parse_patterns(patterns_path: Path = PATTERNS_TXT) -> Dict[str, str]:
    """
    Read the pattern text file and map each account ID to its laundering pattern
    (such as FAN-OUT, FAN-IN, CYCLE, GATHER-SCATTER, STACK, BIPARTITE, or RANDOM).
    """
    account_patterns = {}
    if not patterns_path.exists():
        return account_patterns

    current_pattern = None
    with open(patterns_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Start of a laundering group block, e.g. "BEGIN LAUNDERING ATTEMPT - FAN-OUT"
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                match = re.search(r"BEGIN LAUNDERING ATTEMPT\s*-\s*([A-Za-z\-]+)", line)
                if match:
                    current_pattern = match.group(1).upper()
            # End of the current laundering block
            elif line.startswith("END LAUNDERING ATTEMPT"):
                current_pattern = None
            # Transaction row inside a block: save both sender (column 2) and receiver (column 4)
            elif current_pattern and "," in line:
                parts = line.split(",")
                if len(parts) >= 5:
                    sender = parts[2].strip()
                    receiver = parts[4].strip()
                    if sender:
                        account_patterns[sender] = current_pattern
                    if receiver:
                        account_patterns[receiver] = current_pattern

    return account_patterns


def load_accounts(accounts_path: Path = ACCOUNTS_CSV) -> pd.DataFrame:
    """Load the accounts CSV file and rename columns to simple snake_case names."""
    df_acc = pd.read_csv(accounts_path, dtype=str)
    rename_map = {
        "Bank Name": "bank_name",
        "Bank ID": "bank_id",
        "Account Number": "account_id",
        "Entity ID": "entity_id",
        "Entity Name": "entity_name"
    }
    df_acc = df_acc.rename(columns=rename_map)
    df_acc["account_id"] = df_acc["account_id"].str.strip()
    return df_acc


def load_transactions_sample(
    trans_path: Path = TRANSACTIONS_CSV,
    sample_size: int = 250000,
    include_all_laundering: bool = True
) -> pd.DataFrame:
    """
    Load transactions in chunks of 250,000 rows so we don't run out of RAM.
    Why sample_size=250000 and include_all_laundering=True:
      - Keeps 100% of the 5,177 laundering transactions so no criminal rings are broken.
      - Pairs them with 250,000 legitimate transactions so training stays fast on any laptop.
    """
    col_names = [
        "timestamp", "from_bank", "from_account", "to_bank", "to_account",
        "amount_received", "receiving_currency", "amount_paid", "payment_currency",
        "payment_format", "is_laundering"
    ]

    laundering_dfs = []
    legit_dfs = []
    legit_needed = sample_size
    chunk_size = 250000

    print(f"Loading transactions from {trans_path.name}...")
    for chunk in pd.read_csv(trans_path, names=col_names, header=0, chunksize=chunk_size, low_memory=False):
        # Clean numbers and strip extra spaces from account IDs
        chunk["amount_received"] = pd.to_numeric(chunk["amount_received"], errors="coerce").fillna(0.0)
        chunk["amount_paid"] = pd.to_numeric(chunk["amount_paid"], errors="coerce").fillna(0.0)
        chunk["is_laundering"] = pd.to_numeric(chunk["is_laundering"], errors="coerce").fillna(0).astype(int)
        chunk["from_account"] = chunk["from_account"].astype(str).str.strip()
        chunk["to_account"] = chunk["to_account"].astype(str).str.strip()

        # Keep every laundering row
        laund_chunk = chunk[chunk["is_laundering"] == 1]
        if not laund_chunk.empty and include_all_laundering:
            laundering_dfs.append(laund_chunk)

        # Keep legitimate rows until we reach sample_size (250,000)
        if legit_needed > 0:
            legit_chunk = chunk[chunk["is_laundering"] == 0]
            take_count = min(len(legit_chunk), legit_needed)
            legit_dfs.append(legit_chunk.iloc[:take_count])
            legit_needed -= take_count

        if not include_all_laundering and legit_needed <= 0:
            break

    df_laund = pd.concat(laundering_dfs, ignore_index=True) if laundering_dfs else pd.DataFrame(columns=col_names)
    df_legit = pd.concat(legit_dfs, ignore_index=True) if legit_dfs else pd.DataFrame(columns=col_names)

    combined = pd.concat([df_laund, df_legit], ignore_index=True).drop_duplicates()
    print(
        f"Loaded {len(combined):,} transactions: {len(df_laund):,} laundering "
        f"({len(df_laund)/len(combined)*100:.2f}%), {len(df_legit):,} legitimate."
    )
    return combined


def resolve_laundering_typologies(
    df_acc: pd.DataFrame,
    df_tx: pd.DataFrame,
    base_patterns: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Make sure 100% of suspicious accounts get an accurate laundering pattern (no 'NONE' labels).
    Uses 3 simple steps:
      1. Use the pattern from the text file if already listed.
      2. If an account is connected to a laundering group in the graph, share the group's pattern
         (or detect loops as CYCLE, 1-to-many as FAN-OUT, many-to-1 as FAN-IN, chains as STACK).
      3. For remaining 2-account pairs, check their payment format and connections (e.g., Bitcoin/Cash
         hops -> RANDOM, same-bank/Cheque -> CYCLE, multi-source -> GATHER-SCATTER / SCATTER-GATHER).
    """
    import networkx as nx

    acc_lookup = df_acc.sort_values("is_suspicious", ascending=False).drop_duplicates("account_id").set_index("account_id")
    known_acc_pat: Dict[str, str] = {}

    # Step 1: Load known patterns from the text file
    if base_patterns:
        for k, v in base_patterns.items():
            if v and v != "NONE":
                known_acc_pat[k] = v

    if "pattern_type" in acc_lookup.columns:
        for acc_id, pat in acc_lookup[acc_lookup["is_suspicious"] == 1]["pattern_type"].items():
            if pat and pat != "NONE" and acc_id not in known_acc_pat:
                known_acc_pat[acc_id] = str(pat)

    # Build a graph of only the laundering transactions
    tx_laund = df_tx[df_tx["is_laundering"] == 1]
    G = nx.DiGraph()
    for _, r in tx_laund.iterrows():
        G.add_edge(
            str(r["from_account"]),
            str(r["to_account"]),
            fmt=str(r.get("payment_format", "ACH")),
            amt=float(r.get("amount_paid", 0.0) or 0.0)
        )

    # Step 2: Check each connected group of laundering accounts
    for comp in nx.weakly_connected_components(G):
        pats = [known_acc_pat[n] for n in comp if n in known_acc_pat]
        if pats:
            # Spread the most common known pattern across the connected group
            dom_pat = pd.Series(pats).mode().iloc[0]
            for n in comp:
                known_acc_pat.setdefault(n, dom_pat)
        else:
            # Otherwise identify the shape of the group
            sub = G.subgraph(comp)
            max_in = max(d for _, d in sub.in_degree())
            max_out = max(d for _, d in sub.out_degree())
            if not nx.is_directed_acyclic_graph(sub):
                dom_pat = "CYCLE"      # Has a closed loop
            elif max_out >= 3 and max_out > max_in * 1.5:
                dom_pat = "FAN-OUT"    # One sender splitting money to many receivers
            elif max_in >= 3 and max_in > max_out * 1.5:
                dom_pat = "FAN-IN"     # Many senders collecting into one receiver
            elif len(comp) >= 3:
                dom_pat = "STACK"      # Multi-step chain
            else:
                dom_pat = None
            if dom_pat:
                for n in comp:
                    known_acc_pat[n] = dom_pat

    # Step 3: Classify remaining small pairs by their full-graph degrees and payment rail
    def _classify_pair(u: str, v: str, fmt: str) -> str:
        ru = acc_lookup.loc[u] if u in acc_lookup.index else None
        rv = acc_lookup.loc[v] if v in acc_lookup.index else None
        u_in = float(ru["in_degree"]) if ru is not None and "in_degree" in ru else 0.0
        u_out = float(ru["out_degree"]) if ru is not None and "out_degree" in ru else 0.0
        v_in = float(rv["in_degree"]) if rv is not None and "in_degree" in rv else 0.0
        v_out = float(rv["out_degree"]) if rv is not None and "out_degree" in rv else 0.0
        u_same_bank = (
            ru["bank_name"] == rv["bank_name"]
            if (ru is not None and rv is not None and "bank_name" in ru and "bank_name" in rv)
            else False
        )

        if fmt in ("Bitcoin", "Cash", "Credit Card") and u_out <= 3 and v_in <= 3:
            return "RANDOM"
        if u_same_bank or fmt == "Cheque":
            return "CYCLE"
        if u_out >= 3 and u_out > u_in * 1.5:
            return "FAN-OUT"
        if v_in >= 2 and u_out == 1 and u_in == 0:
            return "FAN-IN"
        if (u_in >= 2 and u_out >= 2) or (v_in >= 2 and v_out >= 1):
            return "GATHER-SCATTER"
        if u_out >= 2 and (v_in >= 2 or v_out >= 1 or u_in >= 1):
            return "SCATTER-GATHER"
        if (u_in == 1 and u_out == 1) or (v_in == 1 and v_out == 1):
            return "STACK"
        return "BIPARTITE"

    for comp in nx.weakly_connected_components(G):
        if any(n in known_acc_pat for n in comp):
            continue
        sub = G.subgraph(comp)
        edges = list(sub.edges(data=True))
        if edges:
            u, v, d = edges[0]
            pat = _classify_pair(u, v, d.get("fmt", "ACH"))
            for n in comp:
                known_acc_pat[n] = pat

    # Final fallback for any isolated suspicious account
    for acc_id, row in acc_lookup[acc_lookup["is_suspicious"] == 1].iterrows():
        if acc_id not in known_acc_pat:
            if float(row.get("out_degree", 0) or 0) >= 3:
                known_acc_pat[acc_id] = "FAN-OUT"
            elif float(row.get("in_degree", 0) or 0) >= 2:
                known_acc_pat[acc_id] = "FAN-IN"
            else:
                known_acc_pat[acc_id] = "STACK"

    return known_acc_pat
