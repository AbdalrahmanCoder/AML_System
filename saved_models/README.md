# Saved Model Files (`saved_models/`)

Because `.joblib` and `.pt` files are binary model checkpoints, this guide (along with the embedded `"documentation"` dictionary stored inside each file) explains what each file contains and why its settings were chosen:

---

## 1. `xgboost.joblib` (Tabular Financial Flow Model)
- **What it is:** A serialized `joblib` dictionary containing the trained **XGBoost** classifier and its probability calibration settings.
- **Internal Keys:**
  - `model`: Trained `XGBClassifier` (`n_estimators=260` trees, `max_depth=7`, `learning_rate=0.05`, `eval_metric="aucpr"`).
  - `features`: List of the 30 account features in exact training order.
  - `logit_shift`: Shifts the Precision-Recall curve optimal threshold to `0.50` (`50%` risk).
  - `logit_temp` (`1.55`): Temperature scaling factor that smooths extreme raw tree log-odds into realistic percentages.
  - `documentation`: Built-in plain-English dictionary explaining every key and hyperparameter inside the file.

---

## 2. `graphsage.pt` (Graph Neural Network Model)
- **What it is:** A PyTorch checkpoint containing the trained **InductiveGraphSAGE** neural network and its companion topological tree interaction booster.
- **Internal Keys:**
  - `state_dict`: Trained PyTorch neural weights (`hidden_dim=128`, `32` structural ring prototypes, `dropout=0.15`, trained for `16 epochs` with `lr=0.005`, `batch_size=4096`, and `Focal Loss gamma=1.5`).
  - `in_dim` (`36`): 30 signed-log1p account features + 6 payment-rail and degree ratios.
  - `mean` & `std`: Training split normalization statistics.
  - `topo_model`: Companion depth-9 topological tree booster blended at `55%` with the `45%` GNN neural output.
  - `logit_shift` & `logit_temp` (`1.75`): PR-curve calibration parameters aligning GraphSAGE probabilities with the 50% blocking cutoff.
  - `documentation`: Built-in plain-English dictionary explaining every key and hyperparameter inside the file.

---

## 3. `combined_model.joblib` (Flagship Ensemble: XGBoost + GraphSAGE)
- **What it is:** A single self-contained `joblib` bundle combining both **XGBoost** and **GraphSAGE**.
- **Internal Keys:**
  - `xgb_weight` (`0.50`) & `sage_weight` (`0.50`): Equal 50/50 blend weights combining individual account behavior with network ring topology.
  - `optimal_threshold` (`0.50`): The 50% laundering cutoff used by the Core Banking system to block suspicious transfers.
  - `xgb_state` & `sage_state`: Complete sub-model weights and calibration parameters.
  - `documentation`: Built-in plain-English dictionary explaining every key and hyperparameter inside the file.

---

## How to Inspect the Embedded Comments in Python
```python
import joblib, torch

comb = joblib.load("saved_models/combined_model.joblib")
print(comb["documentation"])
```
