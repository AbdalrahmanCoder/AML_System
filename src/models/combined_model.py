"""
Combined (XGBoost + GraphSAGE) Ensemble Model.
Blends XGBoost (tabular transaction flows) and GraphSAGE (network graph connections)
into one final laundering risk score.
Evaluated using ROC-AUC and PR-AUC only (no accuracy, precision, recall, or F1).
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)
from sklearn.model_selection import train_test_split
import sys

# Add project root folder so imports work from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import SAVED_MODELS_DIR, RANDOM_STATE
from src.models.xgboost_model import AMLXGBoost
from src.models.graphsage_model import AMLGraphSAGE, InductiveGraphSAGE


class AMLCombinedXGBoostGraphSAGE:
    """
    Combined Ensemble Model.
    Why these weights and threshold are used:
      - xgb_weight=0.50 & sage_weight=0.50: Gives equal 50/50 importance to individual account
        behavior (XGBoost) and surrounding network ring connections (GraphSAGE), preventing either
        model from overpowering the final score.
      - optimal_threshold=0.50: Since both sub-models are calibrated so their PR-curve cutoff sits
        at 50%, any combined score >= 50% (0.50) indicates laundering.
    """

    def __init__(
        self,
        xgb_weight: float = 0.50,
        sage_weight: float = 0.50,
        optimal_threshold: float = 0.50
    ):
        self.xgb_weight = xgb_weight
        self.sage_weight = sage_weight
        self.optimal_threshold = optimal_threshold
        self.xgb_model: Optional[AMLXGBoost] = None
        self.sage_model: Optional[AMLGraphSAGE] = None
        self.feature_names: List[str] = []
        self.calibration_metrics: Dict[str, Any] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        adj_coo: Optional[sp.coo_matrix] = None,
        val_size: float = 0.20,
        optimize_weights: bool = True
    ):
        """Train both sub-models and pick the best blend weight around 50/50 using validation PR-AUC."""
        self.feature_names = list(X.columns)
        n_samples = len(X)

        indices = np.arange(n_samples)
        train_idx, val_idx = train_test_split(
            indices, test_size=val_size, random_state=RANDOM_STATE, stratify=y
        )

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        print("  -> Training GraphSAGE component on network structure...")
        self.sage_model = AMLGraphSAGE(hidden_dim=128, lr=0.005, epochs=16)
        self.sage_model.fit(X_tr, y_tr)

        print("  -> Training XGBoost component with PR-AUC objective (eval_metric='aucpr')...")
        self.xgb_model = AMLXGBoost(n_estimators=260, max_depth=7, learning_rate=0.05)
        self.xgb_model.fit(X_tr, y_tr)

        p_val_xgb = self.xgb_model.predict_proba(X_val)
        p_val_sage = self.sage_model.predict_proba(X_val)

        pr_val_xgb = average_precision_score(y_val, p_val_xgb)
        pr_val_sage = average_precision_score(y_val, p_val_sage)

        # Test blend weights between 0.45 and 0.55 so both models always contribute almost equally
        if optimize_weights:
            best_weight = 0.50
            best_pr_auc = 0.0
            for w in np.linspace(0.45, 0.55, 21):
                p_comb = w * p_val_xgb + (1.0 - w) * p_val_sage
                score = average_precision_score(y_val, p_comb)
                if score > best_pr_auc:
                    best_pr_auc = score
                    best_weight = float(w)

            self.xgb_weight = round(best_weight, 4)
            self.sage_weight = round(1.0 - best_weight, 4)

        self.optimal_threshold = 0.50
        p_val_comb = self.xgb_weight * p_val_xgb + self.sage_weight * p_val_sage
        self.calibration_metrics = {
            "val_xgb_pr_auc": round(float(pr_val_xgb), 4),
            "val_sage_pr_auc": round(float(pr_val_sage), 4),
            "val_combined_pr_auc": round(float(average_precision_score(y_val, p_val_comb)), 4),
            "xgb_weight": self.xgb_weight,
            "sage_weight": self.sage_weight,
            "optimal_threshold": self.optimal_threshold
        }
        return self

    def predict_proba(
        self,
        X: pd.DataFrame,
        adj_coo: Optional[sp.coo_matrix] = None
    ) -> np.ndarray:
        """Calculate the weighted average laundering probability: w_xgb * P_xgb + w_sage * P_sage."""
        p_xgb = self.xgb_model.predict_proba(X)
        p_sage = self.sage_model.predict_proba(X, adj_coo)
        return self.xgb_weight * p_xgb + self.sage_weight * p_sage

    def predict(
        self,
        X: pd.DataFrame,
        adj_coo: Optional[sp.coo_matrix] = None,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """Return 1 (Laundering) if combined probability >= threshold (default 50%), else 0."""
        th = threshold if threshold is not None else self.optimal_threshold
        probs = self.predict_proba(X, adj_coo)
        return (probs >= th).astype(int)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        adj_test: Optional[sp.coo_matrix] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculate only ROC-AUC and PR-AUC on the test set."""
        probs = self.predict_proba(X_test, adj_test)
        th = threshold if threshold is not None else self.optimal_threshold

        return {
            "model_name": "Combined (XGBoost + GraphSAGE)",
            "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
            "pr_auc": round(float(average_precision_score(y_test, probs)), 4),
            "xgb_weight": self.xgb_weight,
            "sage_weight": self.sage_weight,
            "decision_threshold": th
        }

    def get_feature_importances(self) -> pd.DataFrame:
        """Blend feature importance scores from XGBoost and GraphSAGE using their ensemble weights."""
        fi_xgb = self.xgb_model.get_feature_importances().set_index("feature_name")["importance"]
        fi_sage = self.sage_model.get_feature_importances().set_index("feature_name")["importance"]

        norm_xgb = fi_xgb / fi_xgb.sum() if fi_xgb.sum() > 0 else fi_xgb
        norm_sage = fi_sage / fi_sage.sum() if fi_sage.sum() > 0 else fi_sage

        combined_fi = self.xgb_weight * norm_xgb + self.sage_weight * norm_sage
        df = pd.DataFrame({
            "feature_name": combined_fi.index,
            "importance": combined_fi.values
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        df["model_name"] = "Combined (XGBoost + GraphSAGE)"
        return df

    def save(self, path: Path = SAVED_MODELS_DIR / "combined_model.joblib"):
        """Save both sub-models, ensemble weights, and plain-English documentation into a single file."""
        bundle = {
            "xgb_weight": self.xgb_weight,
            "sage_weight": self.sage_weight,
            "optimal_threshold": self.optimal_threshold,
            "feature_names": self.feature_names,
            "calibration_metrics": self.calibration_metrics,
            "documentation": {
                "summary": "Combined (XGBoost + GraphSAGE) Flagship AML Ensemble Model.",
                "xgb_weight": f"{self.xgb_weight} (50% weight given to individual account tabular flows)",
                "sage_weight": f"{self.sage_weight} (50% weight given to 1-hop and 2-hop graph network connections)",
                "optimal_threshold": f"{self.optimal_threshold} (accounts with >= 50% risk are blocked as laundering)"
            },
            "xgb_state": {
                "model": self.xgb_model.model,
                "features": self.xgb_model.feature_names,
                "logit_shift": self.xgb_model.logit_shift,
                "logit_temp": self.xgb_model.logit_temp
            },
            "sage_state": {
                "state_dict": self.sage_model.model.state_dict(),
                "in_dim": self.sage_model.model.fc_self1.in_features,
                "hidden_dim": self.sage_model.hidden_dim,
                "features": self.sage_model.feature_names,
                "mean": self.sage_model.mean,
                "std": self.sage_model.std,
                "topo_model": self.sage_model.topo_model,
                "logit_shift": self.sage_model.logit_shift,
                "logit_temp": self.sage_model.logit_temp
            }
        }
        joblib.dump(bundle, path)
        print(f"Saved Combined (XGBoost + GraphSAGE) ensemble to {path}")

    @classmethod
    def load(cls, path: Path = SAVED_MODELS_DIR / "combined_model.joblib"):
        """Load the saved Combined (XGBoost + GraphSAGE) model from disk."""
        bundle = joblib.load(path)
        inst = cls(
            xgb_weight=bundle.get("xgb_weight", 0.50),
            sage_weight=bundle.get("sage_weight", 0.50),
            optimal_threshold=bundle.get("optimal_threshold", 0.50)
        )
        inst.feature_names = bundle.get("feature_names", [])
        inst.calibration_metrics = bundle.get("calibration_metrics", {})

        # Rebuild XGBoost sub-model
        inst.xgb_model = AMLXGBoost()
        inst.xgb_model.model = bundle["xgb_state"]["model"]
        inst.xgb_model.feature_names = bundle["xgb_state"]["features"]
        inst.xgb_model.logit_shift = float(bundle["xgb_state"].get("logit_shift", 0.0))
        inst.xgb_model.logit_temp = float(bundle["xgb_state"].get("logit_temp", 1.55))

        # Rebuild GraphSAGE sub-model
        sage_data = bundle["sage_state"]
        inst.sage_model = AMLGraphSAGE(hidden_dim=sage_data["hidden_dim"])
        inst.sage_model.feature_names = sage_data["features"]
        inst.sage_model.mean = sage_data["mean"]
        inst.sage_model.std = sage_data["std"]
        inst.sage_model.topo_model = sage_data.get("topo_model", None)
        inst.sage_model.logit_shift = float(sage_data.get("logit_shift", 0.0))
        inst.sage_model.logit_temp = float(sage_data.get("logit_temp", 1.75))
        inst.sage_model.model = InductiveGraphSAGE(
            in_dim=sage_data["in_dim"],
            hidden_dim=sage_data["hidden_dim"],
            out_dim=1
        )
        inst.sage_model.model.load_state_dict(sage_data["state_dict"])
        inst.sage_model.model.eval()

        return inst
