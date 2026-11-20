"""
AML System Central Configuration Module.
Defines portable file-system paths, model training constants, the 30 engineered
financial and graph-topological feature names, and human-readable Explainable AI (XAI) metadata.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Portable Base Directory & Dataset Path Resolution
# ---------------------------------------------------------------------------
# Resolves PROJECT_DIR dynamically relative to this file so the repository
# works out-of-the-box on any machine or directory location after cloning.
PROJECT_DIR = Path(__file__).resolve().parent

# Locate the raw CSV dataset folder if present locally or in parent directories
if (PROJECT_DIR / "Data_Backup").exists():
    RAW_DATA_DIR = PROJECT_DIR / "Data_Backup"
elif (PROJECT_DIR.parent / "Data_Backup").exists():
    RAW_DATA_DIR = PROJECT_DIR.parent / "Data_Backup"
elif (PROJECT_DIR.parent.parent / "Data_Backup").exists():
    RAW_DATA_DIR = PROJECT_DIR.parent.parent / "Data_Backup"
else:
    RAW_DATA_DIR = PROJECT_DIR / "data"

# Raw IBM HI-Small AML Dataset Files (used when running scripts/run_pipeline.py)
ACCOUNTS_CSV = RAW_DATA_DIR / "HI-Small_accounts.csv"
TRANSACTIONS_CSV = RAW_DATA_DIR / "HI-Small_Trans.csv"
PATTERNS_TXT = RAW_DATA_DIR / "New Text Document.txt"

# ---------------------------------------------------------------------------
# 2. SQLite Database & Serialized Model Artifact Paths
# ---------------------------------------------------------------------------
DB_PATH = PROJECT_DIR / "aml_system.db"
SAVED_MODELS_DIR = PROJECT_DIR / "saved_models"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 3. Model Training & Sampling Hyperparameters
# ---------------------------------------------------------------------------
RANDOM_STATE = 42               # Fixed random seed for reproducible splits and GNN initialization
TEST_SIZE = 0.2                 # 80% training / 20% unseen stratified test split
DEFAULT_SAMPLE_SIZE = 250000    # Number of legitimate transactions sampled alongside all laundering txs

# ---------------------------------------------------------------------------
# 4. Engineered Feature Vector Schema (30 Behavioral & Graph Features)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    # Inbound/Outbound Transaction Counts & Volume Totals
    "total_tx_in", "total_tx_out", "total_tx_count",
    "total_amount_in", "total_amount_out", "net_flow",
    # Statistical Dispersion & Extremes per Account
    "avg_amount_in", "avg_amount_out", "max_amount_in", "max_amount_out",
    "std_amount_in", "std_amount_out",
    # Counterparty Diversity & Pass-Through Flow Ratios
    "unique_senders", "unique_receivers",
    "in_out_tx_ratio", "in_out_amount_ratio",
    # Payment Rail Usage & Sub-$10K BSA Structuring ("Smurfing") Counts
    "ach_tx_count", "wire_tx_count", "cheque_tx_count",
    "in_degree", "out_degree", "degree_centrality",
    "reinvestment_tx_count", "smurfing_tx_count",
    # 1-Hop & 2-Hop Graph Neighborhood Message-Passing Aggregations
    "neighbor_ach_avg", "neighbor_reinv_avg", "neighbor_smurf_avg",
    "neighbor_degree_avg", "neighbor_tx_avg", "hop2_ach_avg"
]

# ---------------------------------------------------------------------------
# 5. Human-Readable Feature Metadata for Explainable AI (XAI) & Dashboard UI
# ---------------------------------------------------------------------------
FEATURE_METADATA = {
    "ach_tx_count": {
        "label": "ACH Electronic Transfers Count",
        "category": "Payment Channel Distribution",
        "description": "Number of Automated Clearing House electronic bank transfers (e.g., payroll, recurring deposits, or automated batch debits)."
    },
    "wire_tx_count": {
        "label": "Wire Transfers Count",
        "category": "Payment Channel Distribution",
        "description": "Number of urgent, real-time gross settlement wire transfers (favored for high-value rapid cross-border layering)."
    },
    "cheque_tx_count": {
        "label": "Paper Cheque Transactions Count",
        "category": "Payment Channel Distribution",
        "description": "Number of physical bank draft / paper cheque payments processed."
    },
    "reinvestment_tx_count": {
        "label": "Institutional Reinvestment Transfers Count",
        "category": "Payment Channel Distribution",
        "description": "Number of automated commercial reinvestment transactions (strongly associated with legitimate institutional portfolios)."
    },
    "smurfing_tx_count": {
        "label": "Sub-$10K Structuring ('Smurfing') Transfers Count",
        "category": "Outlier & Threshold Screening",
        "description": "Count of transactions between $2,500 and $9,999.99 positioned directly below the $10,000 BSA Currency Transaction Report (CTR) threshold."
    },
    "neighbor_ach_avg": {
        "label": "1-Hop Counterparty Avg ACH Transfers",
        "category": "Network Graph Topology",
        "description": "Average number of ACH electronic transfers executed by immediate 1-hop counterparties connected to this account."
    },
    "neighbor_reinv_avg": {
        "label": "1-Hop Counterparty Avg Reinvestment Transfers",
        "category": "Network Graph Topology",
        "description": "Average number of institutional reinvestment transfers across immediate 1-hop connected counterparties."
    },
    "neighbor_smurf_avg": {
        "label": "1-Hop Counterparty Avg Structuring Transfers",
        "category": "Network Graph Topology",
        "description": "Average number of sub-$10,000 smurfing transfers across immediate 1-hop connected counterparties."
    },
    "neighbor_degree_avg": {
        "label": "1-Hop Counterparty Avg Network Degree",
        "category": "Network Graph Topology",
        "description": "Average graph degree (number of connections) of immediate counterparties, identifying connections to central mule hubs."
    },
    "neighbor_tx_avg": {
        "label": "1-Hop Counterparty Avg Transaction Volume",
        "category": "Network Graph Topology",
        "description": "Average transaction count of immediate 1-hop counterparties in the payment graph."
    },
    "hop2_ach_avg": {
        "label": "2-Hop Ring Network Avg ACH Activity",
        "category": "Network Graph Topology",
        "description": "Multi-hop (2-hop) neighborhood average of ACH transfers, capturing extended Gather-Scatter, Stack, and Cycle laundering rings."
    },
    "net_flow": {
        "label": "Net Cash Flow Balance ($)",
        "category": "Financial Flow & Balance",
        "description": "Difference between total funds received and disbursed (Funds In - Funds Out). Extreme imbalance indicates gathering hubs or pass-through draining."
    },
    "in_out_tx_ratio": {
        "label": "Inbound / Outbound Tx Velocity Ratio",
        "category": "Transaction Velocity & Asymmetry",
        "description": "Ratio of incoming to outgoing transaction counts. Asymmetry reveals funnel accounts (fan-in) or structuring disbursements (fan-out)."
    },
    "in_out_amount_ratio": {
        "label": "Inbound / Outbound Dollar Flow Ratio",
        "category": "Financial Flow & Balance",
        "description": "Ratio of incoming to outgoing dollar volume. Mule accounts typically exhibit ratios close to 1.0 (draining ~100% of received money)."
    },
    "degree_centrality": {
        "label": "Network Graph Centrality",
        "category": "Network Graph Topology",
        "description": "Topological importance and interconnectedness of the account node within the banking graph. Highlights key hubs in laundering networks."
    },
    "in_degree": {
        "label": "Graph In-Degree (Source Accounts)",
        "category": "Network Graph Topology",
        "description": "Number of distinct upstream accounts feeding capital into this account across the transaction network."
    },
    "out_degree": {
        "label": "Graph Out-Degree (Destination Accounts)",
        "category": "Network Graph Topology",
        "description": "Number of distinct downstream accounts receiving capital from this account across the transaction network."
    },
    "unique_senders": {
        "label": "Unique Sender Counterparties",
        "category": "Counterparty Diversity",
        "description": "Total count of unique individuals or entities that transferred funds to this account."
    },
    "unique_receivers": {
        "label": "Unique Receiver Counterparties",
        "category": "Counterparty Diversity",
        "description": "Total count of unique individuals or entities that received funds from this account."
    },
    "total_tx_count": {
        "label": "Total Completed Transactions",
        "category": "Volume & Velocity",
        "description": "Aggregate count of all completed transactions (both incoming and outgoing)."
    },
    "total_tx_in": {
        "label": "Total Inbound Transfers",
        "category": "Volume & Velocity",
        "description": "Total number of credits received into the account."
    },
    "total_tx_out": {
        "label": "Total Outbound Transfers",
        "category": "Volume & Velocity",
        "description": "Total number of debits sent from the account."
    },
    "total_amount_in": {
        "label": "Total Amount Received ($)",
        "category": "Financial Flow & Balance",
        "description": "Cumulative dollar volume credited to the account."
    },
    "total_amount_out": {
        "label": "Total Amount Disbursed ($)",
        "category": "Financial Flow & Balance",
        "description": "Cumulative dollar volume sent from the account."
    },
    "avg_amount_in": {
        "label": "Average Inbound Transfer Size ($)",
        "category": "Financial Flow & Balance",
        "description": "Mean dollar value per incoming transfer."
    },
    "avg_amount_out": {
        "label": "Average Outbound Transfer Size ($)",
        "category": "Financial Flow & Balance",
        "description": "Mean dollar value per outgoing transfer."
    },
    "max_amount_in": {
        "label": "Maximum Single Inbound Transfer ($)",
        "category": "Outlier & Threshold Screening",
        "description": "Largest single deposit or incoming credit received."
    },
    "max_amount_out": {
        "label": "Maximum Single Outbound Transfer ($)",
        "category": "Outlier & Threshold Screening",
        "description": "Largest single payment or outgoing transfer sent."
    },
    "std_amount_in": {
        "label": "Inbound Amount Volatility (Std Dev)",
        "category": "Statistical Dispersion",
        "description": "Standard deviation of incoming payment sizes (very low volatility indicates fixed-amount structured smurfing)."
    },
    "std_amount_out": {
        "label": "Outbound Amount Volatility (Std Dev)",
        "category": "Statistical Dispersion",
        "description": "Standard deviation of outgoing payment sizes."
    }
}


def format_feature_name(feat_name: str) -> str:
    """Return a human-friendly display label for a technical feature column name."""
    if feat_name in FEATURE_METADATA:
        return FEATURE_METADATA[feat_name]["label"]
    return feat_name.replace("_", " ").title()
