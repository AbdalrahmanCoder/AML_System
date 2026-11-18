"""
Core Banking Service with Automated AML Risk Interception.
Manages account balances, ledger transfers, and real-time AML gatekeeping.
If an account exceeds the laundering probability threshold (default 50%),
it is automatically blocked from completing transactions and restricted from future transfers.
"""
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import FEATURE_COLUMNS
from src.database.db_manager import AMLDatabase


class BankingService:
    def __init__(self, db: AMLDatabase, models: Optional[Dict[str, Any]] = None):
        self.db = db
        self.models = models or {}

    def set_models(self, models: Dict[str, Any]):
        """Update or inject trained ML models (RF, XGBoost, GraphSAGE)."""
        self.models = models

    def get_account(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full account details including balance and block status."""
        return self.db.get_account_by_id(account_id)

    def search_accounts(
        self,
        query: str = "",
        filter_type: str = "all",
        limit: int = 50
    ) -> pd.DataFrame:
        """
        Search accounts for UI selection.
        Filters:
          - 'all': Any accounts
          - 'active': Unblocked accounts (is_blocked = 0)
          - 'blocked': Blocked accounts (is_blocked = 1)
          - 'clean': Low risk legitimate accounts (is_suspicious = 0, risk_score < 0.1)
          - 'high_risk': Suspicious accounts (risk_score >= 0.5 or is_suspicious = 1)
        """
        sql = "SELECT account_id, bank_name, entity_name, balance, is_blocked, block_reason, risk_score, is_suspicious, pattern_type, total_tx_count FROM accounts WHERE 1=1"
        params = []

        order_by = " ORDER BY total_tx_count DESC LIMIT ?"

        if filter_type == "active":
            sql += " AND is_blocked = 0"
        elif filter_type == "blocked":
            sql += " AND is_blocked = 1"
            order_by = " ORDER BY risk_score DESC LIMIT ?"
        elif filter_type == "clean":
            sql += " AND is_suspicious = 0 AND risk_score < 0.05 AND is_blocked = 0 AND total_tx_count BETWEEN 2 AND 10"
            order_by = " ORDER BY total_tx_count ASC LIMIT ?"
        elif filter_type == "high_risk":
            sql += " AND (risk_score >= 0.5 OR is_suspicious = 1)"
            order_by = " ORDER BY risk_score DESC, total_tx_count DESC LIMIT ?"

        if query.strip():
            sql += " AND (account_id LIKE ? OR entity_name LIKE ? OR bank_name LIKE ?)"
            term = f"%{query.strip()}%"
            params.extend([term, term, term])

        sql += order_by
        params.append(limit)

        with self.db.get_connection() as conn:
            return pd.read_sql(sql, conn, params=params)

    def get_preset_accounts(self) -> Dict[str, Dict[str, Any]]:
        """Provides verified testing presets for the banking interface."""
        return {
            "normal": {
                "sender_id": "800042CB0",
                "receiver_id": "800045000",
                "amount": 250.0,
                "format": "ACH",
                "label": "🌟 Normal Flow (Approved Legitimate Transaction, Low Risk <20%)",
                "expected": "APPROVED",
                "description": "Legitimate sender with balanced flow. Evaluated risk is ~15% (<50%), so the transaction settles immediately."
            },
            "high_risk": {
                "sender_id": "8000FFF70",
                "receiver_id": "800045000",
                "amount": 500.0,
                "format": "ACH",
                "label": "⛔ High Laundering Risk (50% - 70% -> Elevated ACH Velocity Block)",
                "expected": "BLOCKED",
                "description": "Sender evaluated at ~60.4% laundering risk (FAN-IN mule gathering with elevated ACH velocity). Exceeds the 50% threshold, blocking the transfer and citing factual forensic causes."
            },
            "critical_warning": {
                "sender_id": "1004286A8",
                "receiver_id": "800045000",
                "amount": 1000.0,
                "format": "Wire",
                "label": "🚨 Critical Laundering Threat (>70% -> High Outgoing Transactions Block)",
                "expected": "BLOCKED_WITH_WARNING",
                "description": "Sender evaluated at ~90.6% laundering risk with 1,659 outgoing transactions and 404 smurfing transfers. Triggers instant transaction block, locks the account, and displays factual forensic causes."
            }
        }

    def reset_demo_account(self, account_id: str, balance: float = 10000.0):
        """Reset a demo account to unblocked with fresh balance for repeatable testing."""
        self.db.reset_demo_account(account_id, balance)

    def reset_all_demo_data(self, balance: float = 10000.0):
        """Reset all accounts, balances, and demo transfers back to their original state."""
        if hasattr(self.db, "reset_all_demo_data"):
            self.db.reset_all_demo_data(balance)
        else:
            for acc_id in ["800042CB0", "8000FFF70", "1004286A8", "800045000"]:
                self.reset_demo_account(acc_id, balance)

    def deposit(self, account_id: str, amount: float) -> Dict[str, Any]:
        """Deposit funds into an account balance."""
        if amount <= 0:
            return {"success": False, "message": "Deposit amount must be strictly positive."}
        acc = self.get_account(account_id)
        if not acc:
            return {"success": False, "message": f"Account '{account_id}' does not exist."}

        new_balance = float(acc.get("balance", 0.0) or 0.0) + float(amount)
        self.db.update_account_balance(account_id, new_balance)
        return {
            "success": True,
            "message": f"Successfully deposited ${amount:,.2f} into account {account_id}.",
            "new_balance": new_balance
        }

    def set_block_status(self, account_id: str, is_blocked: bool, reason: str = "") -> Dict[str, Any]:
        """Manual compliance toggle to block or unblock an account."""
        acc = self.get_account(account_id)
        if not acc:
            return {"success": False, "message": f"Account '{account_id}' does not exist."}

        self.db.set_account_block_status(account_id, 1 if is_blocked else 0, reason)
        status_text = "BLOCKED" if is_blocked else "UNBLOCKED"
        return {
            "success": True,
            "message": f"Account {account_id} has been {status_text}. Reason: {reason}",
            "is_blocked": bool(is_blocked)
        }

    def simulate_sender_features(
        self,
        account: Dict[str, Any],
        amount: float,
        payment_format: str
    ) -> Dict[str, Any]:
        """
        Simulate updated account features reflecting the outgoing transfer
        to run real-time predictive AML inference.
        """
        feat = dict(account)
        cur_tx_out = int(feat.get("total_tx_out", 0) or 0) + 1
        cur_amt_out = float(feat.get("total_amount_out", 0.0) or 0.0) + amount
        cur_amt_in = float(feat.get("total_amount_in", 0.0) or 0.0)
        cur_tx_in = int(feat.get("total_tx_in", 0) or 0)

        feat["total_tx_out"] = cur_tx_out
        feat["total_amount_out"] = cur_amt_out
        feat["total_tx_count"] = cur_tx_in + cur_tx_out
        feat["net_flow"] = cur_amt_in - cur_amt_out
        feat["avg_amount_out"] = cur_amt_out / max(1, cur_tx_out)
        feat["max_amount_out"] = max(float(feat.get("max_amount_out", 0.0) or 0.0), amount)
        feat["in_out_tx_ratio"] = (cur_tx_in + 1.0) / (cur_tx_out + 1.0)
        feat["in_out_amount_ratio"] = (cur_amt_in + 1.0) / (cur_amt_out + 1.0)

        fmt_lower = payment_format.lower()
        if "ach" in fmt_lower:
            prev_ach = float(feat.get("ach_tx_count", 0.0) or 0.0)
            if prev_ach > 0 or amount >= 1000.0:
                feat["ach_tx_count"] = prev_ach + 1.0
        elif "wire" in fmt_lower:
            feat["wire_tx_count"] = float(feat.get("wire_tx_count", 0.0) or 0.0) + 1.0
        elif "cheque" in fmt_lower:
            feat["cheque_tx_count"] = float(feat.get("cheque_tx_count", 0.0) or 0.0) + 1.0
        elif "reinvest" in fmt_lower:
            feat["reinvestment_tx_count"] = float(feat.get("reinvestment_tx_count", 0.0) or 0.0) + 1.0

        if 2500.0 <= amount < 10000.0:
            feat["smurfing_tx_count"] = float(feat.get("smurfing_tx_count", 0.0) or 0.0) + 1.0

        feat["out_degree"] = int(feat.get("out_degree", 0) or 0) + 1
        feat["unique_receivers"] = int(feat.get("unique_receivers", 0) or 0) + 1
        return feat

    def evaluate_aml_risk(
        self,
        account: Dict[str, Any],
        amount: float,
        payment_format: str
    ) -> Dict[str, Any]:
        """
        Runs ML models (XGBoost, GraphSAGE, and Combined XGBoost + GraphSAGE) on account features.
        Evaluates both baseline sender risk and post-transfer simulated risk so outgoing transfers
        never artificially dilute an existing laundering profile.
        """
        base_df = pd.DataFrame([{col: account.get(col, 0.0) for col in FEATURE_COLUMNS}])
        sim_feat = self.simulate_sender_features(account, amount, payment_format)
        sim_df = pd.DataFrame([{col: sim_feat.get(col, 0.0) for col in FEATURE_COLUMNS}])

        results = {
            "rf_prob": 0.0,
            "xgb_prob": 0.0,
            "sage_prob": 0.0,
            "combined_prob": 0.0,
            "highest_prob": 0.0,
            "ensemble_prob": 0.0,
            "models_evaluated": []
        }

        probs = []
        if "XGBoost" in self.models:
            try:
                xgb_base = float(self.models["XGBoost"].predict_proba(base_df)[0])
                xgb_sim = float(self.models["XGBoost"].predict_proba(sim_df)[0])
                xgb_p = max(xgb_base, xgb_sim)
                results["xgb_prob"] = xgb_p
                probs.append(xgb_p)
                results["models_evaluated"].append("XGBoost")
            except Exception as e:
                print(f"Error predicting with XGBoost: {e}")

        if "GraphSAGE" in self.models:
            try:
                sage_base = float(self.models["GraphSAGE"].predict_proba(base_df)[0])
                sage_sim = float(self.models["GraphSAGE"].predict_proba(sim_df)[0])
                sage_p = max(sage_base, sage_sim)
                results["sage_prob"] = sage_p
                probs.append(sage_p)
                results["models_evaluated"].append("GraphSAGE")
            except Exception as e:
                print(f"Error predicting with GraphSAGE: {e}")

        if "Combined (XGBoost + GraphSAGE)" in self.models:
            try:
                comb_model = self.models["Combined (XGBoost + GraphSAGE)"]
                if "XGBoost" in results["models_evaluated"] and "GraphSAGE" in results["models_evaluated"]:
                    comb_p = float(comb_model.xgb_weight * results["xgb_prob"] + comb_model.sage_weight * results["sage_prob"])
                else:
                    comb_base = float(comb_model.predict_proba(base_df)[0])
                    comb_sim = float(comb_model.predict_proba(sim_df)[0])
                    comb_p = max(comb_base, comb_sim)
                results["combined_prob"] = comb_p
                probs.append(comb_p)
                results["models_evaluated"].append("Combined (XGBoost + GraphSAGE)")
            except Exception as e:
                print(f"Error predicting with Combined Model: {e}")

        if "Combined (XGBoost + GraphSAGE)" in results["models_evaluated"]:
            results["highest_prob"] = results["combined_prob"]
        else:
            results["highest_prob"] = float(np.mean(probs)) if probs else 0.0

        results["ensemble_prob"] = results["highest_prob"]
        return results

    def generate_forensic_reasons(
        self,
        account: Dict[str, Any],
        amount: float = 0.0,
        payment_format: str = ""
    ) -> List[str]:
        """
        Derives factual, empirically verified forensic money-laundering indicators
        from the account's historical profile, network graph features, and current transaction context.
        Returns a prioritized list of clear, human-understandable forensic reasons.
        """
        reasons = []

        tx_out = float(account.get("total_tx_out", 0) or 0)
        tx_in = float(account.get("total_tx_in", 0) or 0)
        uniq_recv = float(account.get("unique_receivers", 0) or 0)
        uniq_send = float(account.get("unique_senders", 0) or 0)
        amt_out = float(account.get("total_amount_out", 0) or 0)
        amt_in = float(account.get("total_amount_in", 0) or 0)
        net_flow = float(account.get("net_flow", 0) or 0)
        smurf_cnt = float(account.get("smurfing_tx_count", 0) or 0)
        ach_cnt = float(account.get("ach_tx_count", 0) or 0)
        wire_cnt = float(account.get("wire_tx_count", 0) or 0)
        cheque_cnt = float(account.get("cheque_tx_count", 0) or 0)
        reinv_cnt = float(account.get("reinvestment_tx_count", 0) or 0)
        n_smurf = float(account.get("neighbor_smurf_avg", 0) or 0)
        n_ach = float(account.get("neighbor_ach_avg", 0) or 0)
        hop2_ach = float(account.get("hop2_ach_avg", 0) or 0)
        pt = str(account.get("pattern_type") or "").strip()

        topo_descriptions = {
            "GATHER-SCATTER": "Network Typology: GATHER-SCATTER (Multi-source gathering followed by high-velocity layered dispersal)",
            "FAN-IN": "Network Typology: FAN-IN (Mule gathering ring converging funds into a single collector)",
            "FAN-OUT": "Network Typology: FAN-OUT (Rapid fan-out distribution from a master source to multiple sub-accounts)",
            "CYCLE": "Network Typology: CYCLE (Circular pass-through loop / round-tripping to fabricate legitimate origin)",
            "SCATTER-GATHER": "Network Typology: SCATTER-GATHER (Layered pass-through dispersal and secondary re-aggregation)",
            "BIPARTITE": "Network Typology: BIPARTITE (Dense bipartite mule web transferring across separate clusters)",
            "STACK": "Network Typology: STACK (Multi-tier intermediary pass-through layering)",
            "RANDOM": "Network Typology: RANDOM (Unstructured multi-hop pass-through obfuscation chain)"
        }

        # 1. High Outgoing Transactions / Mass Dispersal
        if tx_out >= 50 or (tx_out >= 5 and tx_out > tx_in * 3) or uniq_recv >= 20:
            recv_text = f" to {int(uniq_recv):,} distinct recipients" if uniq_recv > 0 else ""
            reasons.append(f"High outgoing transactions ({int(tx_out):,} outgoing transfers{recv_text})")
        elif tx_out > tx_in and tx_in == 0 and tx_out >= 1:
            reasons.append(f"One-way outbound drain ({int(tx_out):,} outbound transfers with zero inbound history)")

        # 2. High Inbound Mule Gathering / Fan-In Aggregation
        if (tx_in >= 5 and tx_in > tx_out * 2) or (uniq_send >= 5 and tx_in >= 3):
            send_text = f" from {int(uniq_send):,} unique senders" if uniq_send > 0 else ""
            reasons.append(f"High inbound mule consolidation ({int(tx_in):,} deposits aggregated{send_text})")
        elif tx_in > tx_out and tx_out == 0 and tx_in >= 1:
            reasons.append(f"One-way inbound accumulation ({int(tx_in):,} inbound deposits with zero outbound settlements)")

        # 3. Structuring / Smurfing (Sub-$10,000 CTR Evasion)
        if smurf_cnt > 0:
            reasons.append(f"Structuring / Smurfing detected ({int(smurf_cnt):,} transfers structured below $10,000 regulatory limit)")
        elif 5000 <= amount < 10000:
            reasons.append(f"Current transfer (${amount:,.2f}) structured immediately below $10,000 CTR reporting threshold")

        # 4. Elevated Electronic Batch Clearing (ACH) Channel Abuse
        if ach_cnt >= 2:
            reasons.append(f"Elevated ACH electronic velocity ({int(ach_cnt):,} automated clearing house transfers)")
        elif payment_format == "ACH" and ach_cnt >= 1:
            reasons.append("High-velocity ACH payment channel usage")

        # 5. Wire Transfer Velocity / Cheque Volume
        if wire_cnt >= 5:
            reasons.append(f"Rapid wire transfer velocity ({int(wire_cnt):,} high-value wire movements)")
        if cheque_cnt >= 10:
            reasons.append(f"High cheque settlement volume ({int(cheque_cnt):,} physical cheque instruments)")

        # 6. Reinvestment / Round-tripping Loops
        if reinv_cnt > 0:
            reasons.append(f"Circular reinvestment loop ({int(reinv_cnt):,} round-tripping transfers detected)")

        # 7. Extreme Capital Drain / Balance Asymmetry
        if amt_out > 1e6 and amt_in > 0 and (amt_out / amt_in > 20):
            reasons.append(f"Extreme capital drain asymmetry (Outflow ${amt_out:,.2f} heavily exceeds Inflow ${amt_in:,.2f})")
        elif net_flow < -1_000_000:
            reasons.append(f"Heavy cumulative capital liquidation (${abs(net_flow):,.2f} net drain)")

        # 8. Network Laundering Typology
        if pt in topo_descriptions:
            reasons.append(topo_descriptions[pt])

        # 9. Graph Multi-Hop Contagion
        if n_smurf >= 1.0 or n_ach >= 5.0 or hop2_ach >= 5.0:
            reasons.append(f"Graph neighbor contagion (connected counterparties average {n_smurf:.1f} smurfing and {n_ach:.1f} ACH velocity)")

        if not reasons:
            reasons.append("Multivariate model-detected laundering risk across transaction network features")

        return reasons

    def execute_transfer(
        self,
        from_account_id: str,
        to_account_id: str,
        amount: float,
        payment_format: str = "ACH",
        payment_currency: str = "US Dollar",
        blocking_threshold: float = 0.50,
        warning_threshold: float = 0.70
    ) -> Dict[str, Any]:
        """
        Executes an inter-account transfer with real-time AML compliance screening.
        - If laundering probability >= blocking_threshold (default 50%), transaction is BLOCKED.
        - If laundering probability >= warning_threshold (default 70%), transaction is BLOCKED and
          an urgent regulatory compliance warning notification is dispatched.
        """
        transfer_id = f"TX-{uuid.uuid4().hex[:10].upper()}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Basic validation
        if amount <= 0:
            return {
                "success": False,
                "status": "FAILED",
                "transfer_id": transfer_id,
                "message": "Transfer amount must be strictly greater than $0.00."
            }

        if from_account_id == to_account_id:
            return {
                "success": False,
                "status": "FAILED",
                "transfer_id": transfer_id,
                "message": "Sender and recipient accounts cannot be the same account."
            }

        sender = self.get_account(from_account_id)
        if not sender:
            return {
                "success": False,
                "status": "FAILED",
                "transfer_id": transfer_id,
                "message": f"Sender account '{from_account_id}' was not found in the banking registry."
            }

        recipient = self.get_account(to_account_id)
        if not recipient:
            return {
                "success": False,
                "status": "FAILED",
                "transfer_id": transfer_id,
                "message": f"Recipient account '{to_account_id}' was not found in the banking registry."
            }

        # Evaluate real-time AML risk across all calibrated models
        risk_eval = self.evaluate_aml_risk(sender, amount, payment_format)
        laundering_probability = risk_eval["highest_prob"]

        # 2. Check if sender is ALREADY blocked
        if int(sender.get("is_blocked", 0) or 0) == 1:
            reason = sender.get("block_reason") or "AML compliance restriction"
            forensic_reasons = self.generate_forensic_reasons(sender, amount, payment_format)
            primary_reason = forensic_reasons[0] if forensic_reasons else "Prior AML compliance lockdown"
            self.db.record_banking_transfer({
                "transfer_id": transfer_id,
                "timestamp": timestamp,
                "from_account": from_account_id,
                "to_account": to_account_id,
                "amount": amount,
                "payment_currency": payment_currency,
                "payment_format": payment_format,
                "status": "BLOCKED",
                "laundering_probability": laundering_probability,
                "rf_prob": 0.0,
                "xgb_prob": risk_eval["xgb_prob"],
                "sage_prob": risk_eval["sage_prob"],
                "combined_prob": risk_eval["combined_prob"],
                "risk_decision": "SENDER_ALREADY_BLOCKED",
                "reason": f"Sender account blocked: {primary_reason}"
            })
            return {
                "success": False,
                "status": "BLOCKED",
                "transfer_id": transfer_id,
                "message": f"Transfer rejected: Sender account {from_account_id} is currently BLOCKED by AML compliance. Cause: {primary_reason}.",
                "laundering_probability": laundering_probability,
                "blocking_threshold": blocking_threshold,
                "warning_threshold": warning_threshold,
                "risk_eval": risk_eval,
                "account_blocked": True,
                "is_critical_warning": laundering_probability >= warning_threshold,
                "risk_decision": "SENDER_ALREADY_BLOCKED",
                "block_reason": reason,
                "primary_reason": primary_reason,
                "forensic_reasons": forensic_reasons,
                "from_account": from_account_id,
                "to_account": to_account_id
            }

        # 3. Check if recipient is blocked
        if int(recipient.get("is_blocked", 0) or 0) == 1:
            reason = recipient.get("block_reason") or "AML compliance restriction"
            r_forensic = self.generate_forensic_reasons(recipient)
            r_primary = r_forensic[0] if r_forensic else "AML compliance restriction"
            self.db.record_banking_transfer({
                "transfer_id": transfer_id,
                "timestamp": timestamp,
                "from_account": from_account_id,
                "to_account": to_account_id,
                "amount": amount,
                "payment_currency": payment_currency,
                "payment_format": payment_format,
                "status": "BLOCKED",
                "laundering_probability": laundering_probability,
                "rf_prob": 0.0,
                "xgb_prob": risk_eval["xgb_prob"],
                "sage_prob": risk_eval["sage_prob"],
                "combined_prob": risk_eval["combined_prob"],
                "risk_decision": "RECIPIENT_BLOCKED",
                "reason": f"Recipient account blocked: {r_primary}"
            })
            return {
                "success": False,
                "status": "BLOCKED",
                "transfer_id": transfer_id,
                "message": f"Transfer rejected: Recipient account {to_account_id} is currently BLOCKED by AML compliance ({r_primary}).",
                "laundering_probability": laundering_probability,
                "blocking_threshold": blocking_threshold,
                "warning_threshold": warning_threshold,
                "risk_eval": risk_eval,
                "account_blocked": True,
                "is_critical_warning": False,
                "risk_decision": "RECIPIENT_BLOCKED",
                "block_reason": reason,
                "primary_reason": r_primary,
                "forensic_reasons": r_forensic,
                "from_account": from_account_id,
                "to_account": to_account_id
            }

        # 4. Check sender balance
        sender_balance = float(sender.get("balance", 0.0) or 0.0)
        if sender_balance < amount:
            return {
                "success": False,
                "status": "FAILED",
                "transfer_id": transfer_id,
                "message": f"Insufficient funds: Available balance is ${sender_balance:,.2f}, required ${amount:,.2f}."
            }

        # 5. Apply compliance rule: Block if >= blocking_threshold (default 50%)
        if laundering_probability >= blocking_threshold:
            is_critical = laundering_probability >= warning_threshold
            forensic_reasons = self.generate_forensic_reasons(sender, amount, payment_format)
            primary_reason = forensic_reasons[0] if forensic_reasons else "Elevated Multivariate Laundering Risk"
            reasons_summary = " | ".join(forensic_reasons[:3])

            if is_critical:
                decision_code = "CRITICAL_WARNING_INTERCEPT"
                block_msg = f"AML Intercept: {reasons_summary} [Risk: {laundering_probability*100:.1f}%]"
                ret_msg = (
                    f"[CRITICAL AML ALERT] Transaction Intercepted & Blocked! "
                    f"Primary Forensic Cause: {primary_reason}. "
                    f"Sender account {from_account_id} evaluated at {laundering_probability*100:.1f}% laundering risk and locked down."
                )
            else:
                decision_code = "INTERCEPTED_AND_BLOCKED"
                block_msg = f"AML Intercept: {reasons_summary} [Risk: {laundering_probability*100:.1f}%]"
                ret_msg = (
                    f"[AML INTERCEPT] Transaction Intercepted & Blocked! "
                    f"Primary Forensic Cause: {primary_reason}. "
                    f"Sender account {from_account_id} evaluated at {laundering_probability*100:.1f}% laundering risk (threshold: {blocking_threshold*100:.0f}%)."
                )

            # Lockdown sender account immediately
            self.db.set_account_block_status(from_account_id, is_blocked=1, reason=block_msg)

            # Record blocked attempt
            self.db.record_banking_transfer({
                "transfer_id": transfer_id,
                "timestamp": timestamp,
                "from_account": from_account_id,
                "to_account": to_account_id,
                "amount": amount,
                "payment_currency": payment_currency,
                "payment_format": payment_format,
                "status": "BLOCKED",
                "laundering_probability": laundering_probability,
                "rf_prob": risk_eval.get("rf_prob", 0.0),
                "xgb_prob": risk_eval["xgb_prob"],
                "sage_prob": risk_eval["sage_prob"],
                "combined_prob": risk_eval["combined_prob"],
                "risk_decision": decision_code,
                "reason": block_msg
            })

            return {
                "success": False,
                "status": "BLOCKED",
                "transfer_id": transfer_id,
                "message": ret_msg,
                "laundering_probability": laundering_probability,
                "blocking_threshold": blocking_threshold,
                "warning_threshold": warning_threshold,
                "is_critical_warning": is_critical,
                "risk_eval": risk_eval,
                "account_blocked": True,
                "block_reason": block_msg,
                "primary_reason": primary_reason,
                "forensic_reasons": forensic_reasons,
                "from_account": from_account_id,
                "to_account": to_account_id
            }

        # 7. APPROVED: Execute ledger transfer in SQLite
        new_sender_balance = sender_balance - amount
        recipient_balance = float(recipient.get("balance", 0.0) or 0.0)
        new_recipient_balance = recipient_balance + amount

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Update balances and metrics on sender
            cursor.execute("""
            UPDATE accounts 
            SET balance = ?,
                total_tx_out = total_tx_out + 1,
                total_tx_count = total_tx_count + 1,
                total_amount_out = total_amount_out + ?,
                net_flow = net_flow - ?
            WHERE account_id = ?
            """, (new_sender_balance, amount, amount, from_account_id))

            # Update balances and metrics on recipient
            cursor.execute("""
            UPDATE accounts 
            SET balance = ?,
                total_tx_in = total_tx_in + 1,
                total_tx_count = total_tx_count + 1,
                total_amount_in = total_amount_in + ?,
                net_flow = net_flow + ?
            WHERE account_id = ?
            """, (new_recipient_balance, amount, amount, to_account_id))

            # Insert into core transactions table
            cursor.execute("""
            INSERT INTO transactions (
                timestamp, from_bank, from_account, to_bank, to_account,
                amount_received, receiving_currency, amount_paid, payment_currency,
                payment_format, is_laundering, pattern_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'Legitimate Banking')
            """, (
                timestamp,
                sender.get("bank_name", "Bank"),
                from_account_id,
                recipient.get("bank_name", "Bank"),
                to_account_id,
                amount,
                payment_currency,
                amount,
                payment_currency,
                payment_format
            ))

            # Insert into banking audit ledger
            cursor.execute("""
            INSERT INTO banking_transfers (
                transfer_id, timestamp, from_account, to_account, amount,
                payment_currency, payment_format, status,
                laundering_probability, rf_prob, xgb_prob, sage_prob, combined_prob, risk_decision, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'APPROVED', ?, ?, ?, ?, ?, 'PASSED_COMPLIANCE', 'Transaction within acceptable AML risk limits')
            """, (
                transfer_id,
                timestamp,
                from_account_id,
                to_account_id,
                amount,
                payment_currency,
                payment_format,
                laundering_probability,
                risk_eval["rf_prob"],
                risk_eval["xgb_prob"],
                risk_eval["sage_prob"],
                risk_eval["combined_prob"]
            ))

            conn.commit()

        return {
            "success": True,
            "status": "APPROVED",
            "transfer_id": transfer_id,
            "timestamp": timestamp,
            "message": f"[APPROVED] Transfer of ${amount:,.2f} {payment_currency} from {from_account_id} to {to_account_id} was successfully processed.",
            "from_account": from_account_id,
            "to_account": to_account_id,
            "amount": amount,
            "payment_format": payment_format,
            "sender_new_balance": new_sender_balance,
            "recipient_new_balance": new_recipient_balance,
            "laundering_probability": laundering_probability,
            "blocking_threshold": blocking_threshold,
            "risk_eval": risk_eval,
            "account_blocked": False
        }
