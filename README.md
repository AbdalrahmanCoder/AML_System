# AI-Based Anti-Money Laundering (AML) Detection & Investigation System

**Faculty of Computers and Information Systems — Egyptian Chinese University**

An end-to-end **Anti-Money Laundering (AML) Detection, Graph Analytics, and Core Banking Interception System** built on the IBM Synthetic AML dataset (`HI-Small` accounts, transactions, and laundering typologies).

---

## 🏗️ System Architecture & Core Components

| Layer / Component | Technology | Purpose | Status |
|---|---|---|---|
| **Interactive AML & Banking Dashboard** | Streamlit (`dashboard/app.py`) | Executive Overview, EDA, Core Banking & Transfers, Account Inspector, Live Predictor, and XAI Insights. | ✅ Implemented & Verified |
| **Hybrid ML & Graph Neural Network Engine** | XGBoost + PyTorch GraphSAGE | Combines tabular financial velocity/ratio decision trees with inductive multi-hop GraphSAGE message passing. | ✅ Implemented & Calibrated |
| **Core Banking & Real-Time AML Gatekeeper** | `BankingService` + SQLite | Screens interbank transfers in real time, blocks laundering accounts with factual forensic reasons, and persists compliance state. | ✅ Implemented & Verified |
| **Relational & Graph Feature Pipeline** | Pandas / NetworkX / SciPy | Extracts 30 behavioral, payment-rail, sub-$10K BSA structuring, and 1-hop/2-hop neighborhood graph features. | ✅ Implemented (`feature_engineering.py`) |
| **Persistent Storage Layer** | SQLite (`aml_system.db`) | Stores 206,239 account profiles, 55,177 transactions, model metrics, XAI explanations, and banking audit logs. | ✅ Implemented (`db_manager.py`) |

---

## 🎯 Evaluation Methodology: ROC-AUC & PR-AUC Under Extreme Class Imbalance

In real-world financial surveillance, true money laundering represents a tiny minority of overall activity (**~3.08%** of accounts / **~0.1%** of total raw transactions):
- **Why Single-Threshold Metrics (Accuracy, Precision, Recall, F1) Are Excluded:** A trivial classifier predicting *Legitimate* for every account achieves ~97%–99.9% accuracy while detecting **zero** money launderers. Single-threshold metrics also depend on an arbitrary cutoff and slow down evaluation without measuring full ranking separation.
- **ROC-AUC (Receiver Operating Characteristic AUC):** Measures global rank-ordering capability across all classification thresholds by plotting **True Positive Rate** against **False Positive Rate**.
- **PR-AUC (Precision-Recall AUC / Average Precision):** The primary operational decision metric. It integrates **Precision** against **Recall** across the full probability spectrum without being inflated by the massive volume of legitimate True Negatives.

---

## 📊 Model Evaluation Benchmark (Unseen Test Split)

| Model Name | ROC-AUC | PR-AUC (Primary Decision Metric) |
|---|---|---|
| 🔮 **Combined (XGBoost + GraphSAGE)** | **0.9937** | **0.8807** |
| ⚡ **XGBoost (Tabular Financial Flow)** | **0.9936** | **0.8788** |
| 🕸️ **GraphSAGE (Graph Neural Network)** | **0.9935** | **0.8787** |

---

## 🏦 Core Banking & Real-Time AML Interception Features

- **Searchable Full-Catalog Account Dropdowns:** Browse or filter all **206,239 accounts** directly inside **Custom Search** and the **Account & Compliance Desk** (with quick filters for *All Accounts*, *Laundering / Suspicious Only*, *Legitimate / Clean Only*, and *Currently Blocked Accounts*).
- **Data-Backed Forensic Block Reasons:** When a transaction is intercepted, the system inspects the sender's actual financial and topological profile and reports specific forensic causes—such as *High outgoing transactions*, *Structuring / Smurfing below the $10,000 BSA CTR threshold*, *Elevated ACH / Wire / Cheque velocity*, *High incoming transactions (Fan-In concentration)*, *Near-1:1 rapid pass-through layering*, and *Network Typology (GATHER-SCATTER, FAN-IN, FAN-OUT, CYCLE, STACK, BIPARTITE, RANDOM)*.
- **Persistent Compliance State & Reset Demo:** Account blocks (`is_blocked = 1`), forensic block reasons, and balance updates persist across dashboard page navigation and sessions until manually unblocked in the **Account & Compliance Desk** or restored to baseline via the **🔄 Reset Demo** button.

---

## 📁 Project Structure

```text
AML_System/
├── run.bat                                    # One-click Windows setup (.venv + requirements) & dashboard launcher
├── config.py                                  # Portable path resolution, 30 feature definitions & XAI metadata
├── aml_system.db                              # Pre-populated SQLite database (206,239 accounts, transactions & metrics)
├── requirements.txt                           # Python package dependencies
├── saved_models/                              # Serialized PR-AUC calibrated model artifacts
│   ├── combined_model.joblib                  # Calibrated XGBoost + GraphSAGE ensemble
│   ├── xgboost.joblib                         # Calibrated XGBoost classifier
│   └── graphsage.pt                           # PyTorch Inductive GraphSAGE checkpoint
├── dashboard/
│   └── app.py                                 # Streamlit interactive AML surveillance & core banking dashboard
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb     # Interactive Jupyter Notebook for Exploratory Data Analysis
├── scripts/
│   └── run_pipeline.py                        # End-to-end data ingestion, feature engineering & model training pipeline
└── src/
    ├── banking/
    │   └── banking_service.py                 # Core Banking ledger, real-time AML gatekeeper & forensic reason engine
    ├── data/
    │   └── loader.py                          # Dataset loader & laundering typology parser
    ├── database/
    │   └── db_manager.py                      # SQLite database manager & demo reset state handler
    ├── features/
    │   └── feature_engineering.py             # Account flow, BSA structuring & multi-hop graph feature extraction
    └── models/
        ├── combined_model.py                  # PR-AUC calibrated XGBoost + GraphSAGE ensemble
        ├── graphsage_model.py                 # Inductive GraphSAGE neural network
        ├── xgboost_model.py                   # Calibrated XGBoost model
        └── trainer.py                         # Training, calibration & evaluation orchestrator
```

---

## 🚀 Quick Start (Setting Up on Any Computer)

### Option A: One-Click Windows Launcher (`run.bat`)
Double-click **`run.bat`** (or run `.\run.bat` in PowerShell / Command Prompt). It will automatically:
1. Create a local virtual environment (`.venv`) for that computer if one does not exist.
2. Install all packages from `requirements.txt`.
3. Launch the Streamlit dashboard at **http://localhost:8501**.

### Option B: Manual Virtual Environment Setup (`.venv`)
```powershell
# 1. Create a local virtual environment on the target machine
python -m venv .venv

# 2. Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
# source .venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit Dashboard
python -m streamlit run dashboard/app.py
```

### (Optional) Re-run the Full Data & Training Pipeline
If raw dataset files (`HI-Small_accounts.csv`, `HI-Small_Trans.csv`, `New Text Document.txt`) are placed in `Data_Backup/` or `data/`:
```powershell
python scripts/run_pipeline.py --sample-size 250000
```
