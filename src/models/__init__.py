"""
AML System Models Package.
Exports PR-AUC calibrated AML detection algorithms:
- AMLXGBoost: Gradient-boosted decision trees (optimized and calibrated on PR-AUC)
- AMLGraphSAGE: Inductive Graph Neural Network with multi-hop neighborhood aggregation
- AMLCombinedXGBoostGraphSAGE: Balanced PR-AUC hybrid ensemble of XGBoost + GraphSAGE
"""
from src.models.xgboost_model import AMLXGBoost
from src.models.graphsage_model import AMLGraphSAGE
from src.models.combined_model import AMLCombinedXGBoostGraphSAGE

__all__ = [
    "AMLXGBoost",
    "AMLGraphSAGE",
    "AMLCombinedXGBoostGraphSAGE"
]
