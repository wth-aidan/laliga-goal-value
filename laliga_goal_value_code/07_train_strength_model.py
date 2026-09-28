import pandas as pd
import numpy as np
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # folder containing this code folder and the data
from sklearn.model_selection import GroupKFold
import xgboost as xgb
from sklearn.metrics import log_loss

OUT_DIR = BASE + "/output"
os.makedirs(OUT_DIR, exist_ok=True)
states = pd.read_pickle(os.path.join(OUT_DIR, "states_v3_with_strength.pkl"))

label_map = {"loss": 0, "draw": 1, "win": 2}
states["y"] = states["outcome"].map(label_map)

FEATURES = ["goals_for", "goals_against", "minutes_remaining", "home_flag",
            "pre_match_win_prob", "pre_match_draw_prob"]
X = states[FEATURES].values
y = states["y"].values

# Reuse the SAME test match split as the raw model for apples-to-apples comparison
import pickle
raw_meta = pickle.load(open(os.path.join(OUT_DIR, "model_meta_v2.pkl"), "rb"))
test_matches = raw_meta["test_matches"]
train_mask = ~states["match_id"].isin(test_matches)
test_mask = states["match_id"].isin(test_matches)

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
groups_train = states.loc[train_mask, "match_id"].values

print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

param_grid = [
    {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 200},
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 200},
    {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 400},
    {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 300},
]
gkf = GroupKFold(n_splits=5)
results = []
for params in param_grid:
    fold_losses = []
    for tr_idx, val_idx in gkf.split(X_train, y_train, groups=groups_train):
        model = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3,
            max_depth=params["max_depth"], learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"], min_child_weight=20,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            n_jobs=4, verbosity=0,
        )
        model.fit(X_train[tr_idx], y_train[tr_idx])
        pred = model.predict_proba(X_train[val_idx])
        fold_losses.append(log_loss(y_train[val_idx], pred, labels=[0, 1, 2]))
    avg_loss = np.mean(fold_losses)
    results.append((params, avg_loss))
    print(f"{params} -> CV log-loss: {avg_loss:.4f}")

best_params, best_loss = min(results, key=lambda r: r[1])
print(f"\nBest params: {best_params} (CV log-loss {best_loss:.4f})")

final_model = xgb.XGBClassifier(
    objective="multi:softprob", num_class=3,
    max_depth=best_params["max_depth"], learning_rate=best_params["learning_rate"],
    n_estimators=best_params["n_estimators"], min_child_weight=20,
    subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
    n_jobs=4, verbosity=0,
)
final_model.fit(X_train, y_train)

test_pred = final_model.predict_proba(X_test)
test_loss = log_loss(y_test, test_pred, labels=[0, 1, 2])
print(f"\nHeld-out test log-loss (strength-adjusted): {test_loss:.4f}")
print("(raw model test log-loss was 0.7740, for comparison)")

final_model.save_model(os.path.join(OUT_DIR, "strength_model.json"))
pickle.dump({"test_matches": test_matches, "features": FEATURES, "label_map": label_map},
            open(os.path.join(OUT_DIR, "strength_model_meta.pkl"), "wb"))
print("Saved strength_model.json and strength_model_meta.pkl")
