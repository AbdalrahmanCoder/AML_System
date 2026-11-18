"""
XGBoost Classifier for AML Detection.
Optimized and calibrated using the Precision-Recall Curve (PR-AUC / Average Precision)
to provide accurate, well-calibrated laundering probabilities under extreme class imbalance.
"""
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import SAVED_MODELS_DIR, RANDOM_STATE


def _prob_to_logit(p: np.ndarray) -> np.ndarray:
    p_clip = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p_clip / (1.0 - p_clip))


def _calibrate_probs(raw_p: np.ndarray, shift: float, temp: float) -> np.ndarray:
    z = _prob_to_logit(raw_p)
    return (1.0 / (1.0 + np.exp(-(z - shift) / temp))).astype(np.float64)


class AMLXGBoost:
    def __init__(self, n_estimators: int = 260, max_depth: int = 7, learning_rate: float = 0.05):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.feature_names = []
        self.logit_shift = 0.0
        self.logit_temp = 1.55

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names:
            return X.copy().fillna(0.0)
        df = pd.DataFrame(index=X.index)
        for col in self.feature_names:
            df[col] = X[col].fillna(0.0) if col in X.columns else 0.0
        return df

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_names = list(X.columns)
        X_clean = self._align_features(X)

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

        # Calibrate probabilities using the Precision-Recall curve optimal F1 threshold
        raw_probs = self.model.predict_proba(X_clean)[:, 1]
        prec, rec, th = precision_recall_curve(y, raw_probs)
        if len(th) > 0:
            f1_vals = 2.0 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-8)
            opt_th = float(th[int(np.argmax(f1_vals))])
            # Center the PR-curve optimal threshold at 50% (logit = 0) with smooth temperature scaling
            self.logit_shift = float(_prob_to_logit(opt_th)) - 0.32
        else:
            self.logit_shift = 0.0
        self.logit_temp = 1.55
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = self._align_features(X)
        raw_p = self.model.predict_proba(X_clean)[:, 1]
        return _calibrate_probs(raw_p, self.logit_shift, self.logit_temp)

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        probs = self.predict_proba(X_test)
        preds = (probs >= 0.50).astype(int)

        return {
            "model_name": "XGBoost",
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
            "pr_auc": round(float(average_precision_score(y_test, probs)), 4),
        }

    def get_feature_importances(self) -> pd.DataFrame:
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
        joblib.dump({
            "model": self.model,
            "features": self.feature_names,
            "logit_shift": self.logit_shift,
            "logit_temp": self.logit_temp
        }, path)
        print(f"Saved XGBoost model to {path}")

    @classmethod
    def load(cls, path: Path = SAVED_MODELS_DIR / "xgboost.joblib"):
        data = joblib.load(path)
        inst = cls()
        inst.model = data["model"]
        inst.feature_names = data["features"]
        inst.logit_shift = float(data.get("logit_shift", 0.0))
        inst.logit_temp = float(data.get("logit_temp", 1.55))
        return inst
