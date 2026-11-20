"""
XGBoost Model for AML Detection.
Uses decision trees to detect money laundering from account transaction numbers and ratios.
Evaluated using ROC-AUC and PR-AUC only (no accuracy, precision, recall, or F1).
"""
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any
import sys

# Add project root folder so imports work from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import SAVED_MODELS_DIR, RANDOM_STATE


def _prob_to_logit(p: np.ndarray) -> np.ndarray:
    """Convert a 0-to-1 probability into a log-odds score (logit)."""
    p_clip = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p_clip / (1.0 - p_clip))


def _calibrate_probs(raw_p: np.ndarray, shift: float, temp: float) -> np.ndarray:
    """
    Shift and smooth raw model probabilities so that 50% (0.50) is the exact
    cutoff for blocking laundering accounts.
    """
    z = _prob_to_logit(raw_p)
    return (1.0 / (1.0 + np.exp(-(z - shift) / temp))).astype(np.float64)


class AMLXGBoost:
    """
    XGBoost model wrapper.
    Why these default settings are used:
      - n_estimators=260: Enough trees to learn complex laundering rules without being slow.
      - max_depth=7: Deep enough to combine 6-7 features at once, shallow enough to avoid memorizing noise.
      - learning_rate=0.05: Small step size so each tree gently corrects the previous trees' mistakes.
    """

    def __init__(self, n_estimators: int = 260, max_depth: int = 7, learning_rate: float = 0.05):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.feature_names = []
        # Calibration settings:
        # logit_shift moves the decision boundary to 50%
        # logit_temp=1.55 softens extreme 99.9% scores into realistic percentages
        self.logit_shift = 0.0
        self.logit_temp = 1.55

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Keep feature columns in the exact order used during training and fill missing values with 0."""
        if not self.feature_names:
            return X.copy().fillna(0.0)
        df = pd.DataFrame(index=X.index)
        for col in self.feature_names:
            df[col] = X[col].fillna(0.0) if col in X.columns else 0.0
        return df

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Train the XGBoost trees and calibrate probabilities using the PR curve."""
        self.feature_names = list(X.columns)
        X_clean = self._align_features(X)

        # Why these XGBoost settings are used:
        # - scale_pos_weight=1.0: Keeps raw probabilities realistic instead of artificially inflating them.
        # - eval_metric="aucpr": Tells XGBoost to optimize PR-AUC directly on skewed data.
        # - subsample=0.80 & colsample_bytree=0.70: Uses 80% of rows and 70% of features per tree to prevent overfitting.
        # - reg_lambda=2.5 & min_child_weight=3: Penalizes overly complex leaves so single outliers don't trick the model.
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=1.0,
            eval_metric="aucpr",
            subsample=0.80,
            colsample_bytree=0.70,
            reg_lambda=2.5,
            min_child_weight=3,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        self.model.fit(X_clean, y)

        # Find the best cutoff on the Precision-Recall curve and shift it to 50% (0.50)
        raw_probs = self.model.predict_proba(X_clean)[:, 1]
        prec, rec, th = precision_recall_curve(y, raw_probs)
        if len(th) > 0:
            pr_balance = 2.0 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-8)
            opt_th = float(th[int(np.argmax(pr_balance))])
            self.logit_shift = float(_prob_to_logit(opt_th)) - 0.32
        else:
            self.logit_shift = 0.0
        self.logit_temp = 1.55
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the calibrated laundering probability (0.0 to 1.0) for each account."""
        X_clean = self._align_features(X)
        raw_p = self.model.predict_proba(X_clean)[:, 1]
        return _calibrate_probs(raw_p, self.logit_shift, self.logit_temp)

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        """Return 1 (Laundering) if probability >= threshold (default 50%), else 0 (Legitimate)."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """Calculate only ROC-AUC and PR-AUC on the test set."""
        probs = self.predict_proba(X_test)
        return {
            "model_name": "XGBoost",
            "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
            "pr_auc": round(float(average_precision_score(y_test, probs)), 4),
        }

    def get_feature_importances(self) -> pd.DataFrame:
        """Return a ranked table showing how much each feature contributed to XGBoost decisions."""
        importances = self.model.feature_importances_
        tot = float(np.sum(importances))
        if tot > 0:
            importances = importances / tot
        df = pd.DataFrame({
            "feature_name": self.feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        df["model_name"] = "XGBoost"
        return df

    def save(self, path: Path = SAVED_MODELS_DIR / "xgboost.joblib"):
        """Save the trained model, calibration settings, and plain-English documentation to a file."""
        joblib.dump({
            "model": self.model,
            "features": self.feature_names,
            "logit_shift": self.logit_shift,
            "logit_temp": self.logit_temp,
            "documentation": {
                "summary": "Trained XGBoost tabular AML detector calibrated on the Precision-Recall curve.",
                "n_estimators": f"{self.n_estimators} trees (enough to learn flow rules while staying fast)",
                "max_depth": f"{self.max_depth} levels deep (combines 6-7 features without memorizing noise)",
                "learning_rate": f"{self.learning_rate} step size (each tree gently fixes prior mistakes)",
                "logit_shift": "Shifts the PR-curve optimal threshold to 0.50 (50% risk cutoff)",
                "logit_temp": f"{self.logit_temp} temperature scaling (smooths raw tree scores into realistic percentages)"
            }
        }, path)
        print(f"Saved XGBoost model to {path}")

    @classmethod
    def load(cls, path: Path = SAVED_MODELS_DIR / "xgboost.joblib"):
        """Load a saved XGBoost model from disk."""
        data = joblib.load(path)
        inst = cls()
        inst.model = data["model"]
        inst.feature_names = data["features"]
        inst.logit_shift = float(data.get("logit_shift", 0.0))
        inst.logit_temp = float(data.get("logit_temp", 1.55))
        return inst
