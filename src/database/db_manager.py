"""
Database Manager for AML SQLite Database.
Stores accounts, transactions, model metrics, predictions, and explanations.
"""
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional
import os
import sys

from pathlib import Path

# Add project root to sys.path if not present
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import DB_PATH

class AMLDatabase:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DB_PATH)
        self.init_database()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self):
        """Create tables if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Accounts Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                account_id TEXT PRIMARY KEY,
                bank_id TEXT,
                bank_name TEXT,
                entity_id TEXT,
                entity_name TEXT,
                is_suspicious INTEGER DEFAULT 0,
                suspicion_category TEXT DEFAULT 'Legitimate',
                risk_score REAL DEFAULT 0.0,
                total_tx_in INTEGER DEFAULT 0,
                total_tx_out INTEGER DEFAULT 0,
                total_amount_in REAL DEFAULT 0.0,
                total_amount_out REAL DEFAULT 0.0,
                net_flow REAL DEFAULT 0.0,
                pattern_type TEXT,
                total_tx_count INTEGER DEFAULT 0,
                avg_amount_in REAL DEFAULT 0.0,
                avg_amount_out REAL DEFAULT 0.0,
                max_amount_in REAL DEFAULT 0.0,
                max_amount_out REAL DEFAULT 0.0,
                std_amount_in REAL DEFAULT 0.0,
                std_amount_out REAL DEFAULT 0.0,
                unique_senders INTEGER DEFAULT 0,
                unique_receivers INTEGER DEFAULT 0,
                in_out_tx_ratio REAL DEFAULT 0.0,
                in_out_amount_ratio REAL DEFAULT 0.0,
                ach_tx_count REAL DEFAULT 0.0,
                wire_tx_count REAL DEFAULT 0.0,
                cheque_tx_count REAL DEFAULT 0.0,
                in_degree INTEGER DEFAULT 0,
                out_degree INTEGER DEFAULT 0,
                degree_centrality REAL DEFAULT 0.0,
                balance REAL DEFAULT 10000.0,
                is_blocked INTEGER DEFAULT 0,
                block_reason TEXT DEFAULT ''
            )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_acc_suspicious ON accounts(is_suspicious)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_acc_risk ON accounts(risk_score)")

            # Auto-migration for existing accounts table
            cursor.execute("PRAGMA table_info(accounts)")
            existing_cols = [c[1] for c in cursor.fetchall()]
            if existing_cols:
                if "balance" not in existing_cols:
                    cursor.execute("ALTER TABLE accounts ADD COLUMN balance REAL DEFAULT 10000.0")
                    cursor.execute("UPDATE accounts SET balance = 10000.0 WHERE balance IS NULL")
                if "is_blocked" not in existing_cols:
                    cursor.execute("ALTER TABLE accounts ADD COLUMN is_blocked INTEGER DEFAULT 0")
                if "block_reason" not in existing_cols:
                    cursor.execute("ALTER TABLE accounts ADD COLUMN block_reason TEXT DEFAULT ''")

                cursor.execute("UPDATE accounts SET balance = 10000.0 WHERE balance IS NULL")
                cursor.execute("UPDATE accounts SET is_blocked = 0 WHERE is_blocked IS NULL")
                cursor.execute("UPDATE accounts SET block_reason = '' WHERE block_reason IS NULL")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_acc_blocked ON accounts(is_blocked)")

            # Transactions Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                from_bank TEXT,
                from_account TEXT,
                to_bank TEXT,
                to_account TEXT,
                amount_received REAL,
                receiving_currency TEXT,
                amount_paid REAL,
                payment_currency TEXT,
                payment_format TEXT,
                is_laundering INTEGER,
                pattern_type TEXT
            )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_from ON transactions(from_account)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_to ON transactions(to_account)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_laundering ON transactions(is_laundering)")

            # Banking Transfers Audit Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS banking_transfers (
                transfer_id TEXT PRIMARY KEY,
                timestamp TEXT,
                from_account TEXT,
                to_account TEXT,
                amount REAL,
                payment_currency TEXT DEFAULT 'US Dollar',
                payment_format TEXT DEFAULT 'ACH',
                status TEXT,
                laundering_probability REAL DEFAULT 0.0,
                rf_prob REAL DEFAULT 0.0,
                xgb_prob REAL DEFAULT 0.0,
                sage_prob REAL DEFAULT 0.0,
                combined_prob REAL DEFAULT 0.0,
                risk_decision TEXT,
                reason TEXT
            )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_btransfer_from ON banking_transfers(from_account)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_btransfer_to ON banking_transfers(to_account)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_btransfer_status ON banking_transfers(status)")

            # Auto-migration for banking_transfers table
            cursor.execute("PRAGMA table_info(banking_transfers)")
            bt_cols = [c[1] for c in cursor.fetchall()]
            if bt_cols and "combined_prob" not in bt_cols:
                cursor.execute("ALTER TABLE banking_transfers ADD COLUMN combined_prob REAL DEFAULT 0.0")

            # Model Metrics Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                dataset_name TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                roc_auc REAL,
                pr_auc REAL,
                trained_at TEXT
            )
            """)

            # Model Predictions Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                account_id TEXT,
                true_label INTEGER,
                predicted_label INTEGER,
                predicted_prob REAL,
                created_at TEXT
            )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_model ON model_predictions(model_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_acc ON model_predictions(account_id)")

            # Explanations / Feature Importances Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                feature_name TEXT,
                importance REAL,
                rank INTEGER
            )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feat_model ON feature_explanations(model_name)")

            conn.commit()

    def insert_accounts(self, df_accounts: pd.DataFrame, if_exists: str = "replace"):
        """Save account profiles into accounts table."""
        with self.get_connection() as conn:
            df_accounts.to_sql("accounts", conn, if_exists=if_exists, index=False)

    def insert_transactions(self, df_tx: pd.DataFrame, if_exists: str = "append"):
        """Save transactions into transactions table."""
        with self.get_connection() as conn:
            df_tx.to_sql("transactions", conn, if_exists=if_exists, index=False)

    def save_model_metrics(self, metrics_list: List[Dict[str, Any]]):
        """Save evaluated metrics for trained models (replacing prior benchmark runs)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM model_metrics")
            for m in metrics_list:
                cursor.execute("""
                INSERT INTO model_metrics 
                (model_name, dataset_name, accuracy, precision, recall, f1_score, roc_auc, pr_auc, trained_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    m.get("model_name"),
                    m.get("dataset_name", "HI-Small"),
                    m.get("accuracy"),
                    m.get("precision"),
                    m.get("recall"),
                    m.get("f1_score"),
                    m.get("roc_auc"),
                    m.get("pr_auc"),
                    m.get("trained_at")
                ))
            conn.commit()

    def save_predictions(self, df_preds: pd.DataFrame):
        """Save model predictions table."""
        with self.get_connection() as conn:
            df_preds.to_sql("model_predictions", conn, if_exists="replace", index=False)

    def save_feature_explanations(self, explanations_list: List[Dict[str, Any]]):
        """Save feature importances / explanation ranks."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM feature_explanations")
            for exp in explanations_list:
                cursor.execute("""
                INSERT INTO feature_explanations (model_name, feature_name, importance, rank)
                VALUES (?, ?, ?, ?)
                """, (exp.get("model_name"), exp.get("feature_name"), exp.get("importance"), exp.get("rank")))
            conn.commit()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Query high-level dataset statistics for the dashboard."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM accounts")
            total_acc = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM accounts WHERE is_suspicious = 1")
            suspicious_acc = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM accounts WHERE is_blocked = 1")
            blocked_acc = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM transactions")
            total_tx = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_laundering = 1")
            laundering_tx = cursor.fetchone()[0]

            return {
                "total_accounts": total_acc,
                "suspicious_accounts": suspicious_acc,
                "blocked_accounts": blocked_acc,
                "total_transactions": total_tx,
                "laundering_transactions": laundering_tx
            }

    def get_model_metrics(self) -> pd.DataFrame:
        """Fetch all model evaluation records."""
        with self.get_connection() as conn:
            return pd.read_sql("SELECT * FROM model_metrics ORDER BY id DESC", conn)

    def get_top_features(self, model_name: str, top_n: int = 10) -> pd.DataFrame:
        """Fetch feature importances for a given model."""
        with self.get_connection() as conn:
            return pd.read_sql("""
            SELECT feature_name, importance, rank 
            FROM feature_explanations 
            WHERE model_name = ? 
            ORDER BY rank ASC LIMIT ?
            """, conn, params=(model_name, top_n))

    def get_account_by_id(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single account profile by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM accounts WHERE account_id = ?", (account_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def update_account_balance(self, account_id: str, new_balance: float):
        """Update account balance."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE accounts SET balance = ? WHERE account_id = ?", (float(new_balance), account_id))
            conn.commit()

    def set_account_block_status(self, account_id: str, is_blocked: int, reason: str = ""):
        """Update account block status and reason."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE accounts 
            SET is_blocked = ?, block_reason = ?, is_suspicious = CASE WHEN ? = 1 THEN 1 ELSE is_suspicious END
            WHERE account_id = ?
            """, (int(is_blocked), reason, int(is_blocked), account_id))
            conn.commit()

    def record_banking_transfer(self, transfer_data: Dict[str, Any]):
        """Record a banking transfer attempt in the audit table."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO banking_transfers (
                transfer_id, timestamp, from_account, to_account, amount,
                payment_currency, payment_format, status,
                laundering_probability, rf_prob, xgb_prob, sage_prob, combined_prob, risk_decision, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transfer_data.get("transfer_id"),
                transfer_data.get("timestamp"),
                transfer_data.get("from_account"),
                transfer_data.get("to_account"),
                transfer_data.get("amount"),
                transfer_data.get("payment_currency", "US Dollar"),
                transfer_data.get("payment_format", "ACH"),
                transfer_data.get("status"),
                transfer_data.get("laundering_probability", 0.0),
                transfer_data.get("rf_prob", 0.0),
                transfer_data.get("xgb_prob", 0.0),
                transfer_data.get("sage_prob", 0.0),
                transfer_data.get("combined_prob", 0.0),
                transfer_data.get("risk_decision", ""),
                transfer_data.get("reason", "")
            ))
            conn.commit()

    def get_banking_transfers(self, limit: int = 50) -> pd.DataFrame:
        """Fetch recent banking transfer logs."""
        with self.get_connection() as conn:
            return pd.read_sql("""
            SELECT transfer_id, timestamp, from_account, to_account, amount, payment_currency,
                   payment_format, status, laundering_probability, combined_prob, xgb_prob, sage_prob, rf_prob, risk_decision, reason
            FROM banking_transfers 
            ORDER BY rowid DESC LIMIT ?
            """, conn, params=(limit,))

    def get_banking_stats(self) -> Dict[str, Any]:
        """Fetch summary statistics for the banking transfer module."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM banking_transfers")
            total_transfers = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM banking_transfers WHERE status = 'APPROVED'")
            approved_transfers = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM banking_transfers WHERE status = 'BLOCKED'")
            blocked_transfers = cursor.fetchone()[0]

            cursor.execute("SELECT COALESCE(SUM(amount), 0.0) FROM banking_transfers WHERE status = 'APPROVED'")
            total_volume = cursor.fetchone()[0]

            return {
                "total_transfers": total_transfers,
                "approved_transfers": approved_transfers,
                "blocked_transfers": blocked_transfers,
                "total_volume": total_volume
            }

    def reset_demo_account(self, account_id: str, balance: float = 10000.0):
        """Reset a demo account to unblocked with fresh balance for repeatable testing."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE accounts 
            SET is_blocked = 0, block_reason = '', balance = ?
            WHERE account_id = ?
            """, (float(balance), account_id))
            conn.commit()

    def reset_all_demo_data(self, balance: float = 10000.0):
        """Reset all banking demo modifications back to original clean state."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Revert any live demo transfers recorded in transactions
            cursor.execute("""
                SELECT rowid, from_account, to_account, amount_paid 
                FROM transactions 
                WHERE pattern_type = 'Legitimate Banking'
            """)
            demo_txs = cursor.fetchall()
            for row in demo_txs:
                rid, f_acc, t_acc, amt = row[0], row[1], row[2], float(row[3] or 0.0)
                cursor.execute("""
                    UPDATE accounts 
                    SET total_tx_out = MAX(0, total_tx_out - 1),
                        total_tx_count = MAX(0, total_tx_count - 1),
                        total_amount_out = MAX(0.0, total_amount_out - ?),
                        net_flow = net_flow + ?
                    WHERE account_id = ?
                """, (amt, amt, f_acc))
                cursor.execute("""
                    UPDATE accounts 
                    SET total_tx_in = MAX(0, total_tx_in - 1),
                        total_tx_count = MAX(0, total_tx_count - 1),
                        total_amount_in = MAX(0.0, total_amount_in - ?),
                        net_flow = net_flow - ?
                    WHERE account_id = ?
                """, (amt, amt, t_acc))
            if demo_txs:
                cursor.execute("DELETE FROM transactions WHERE pattern_type = 'Legitimate Banking'")

            # Restore all accounts to original unblocked status and starting balance
            cursor.execute("""
                UPDATE accounts 
                SET is_blocked = 0, block_reason = '', balance = ?
                WHERE is_blocked != 0 OR block_reason != '' OR balance != ?
            """, (float(balance), float(balance)))

            # Clear banking transfer audit ledger
            cursor.execute("DELETE FROM banking_transfers")
            conn.commit()
