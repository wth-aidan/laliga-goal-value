import pandas as pd
import numpy as np
import pickle
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # folder containing this code folder and the data
import xgboost as xgb

OUT_DIR = BASE + "/output"
os.makedirs(OUT_DIR, exist_ok=True)
states = pd.read_pickle(os.path.join(OUT_DIR, "states_v2.pkl"))
meta = pickle.load(open(os.path.join(OUT_DIR, "model_meta_v2.pkl"), "rb"))
FEATURES, test_matches, label_map = meta["features"], meta["test_matches"], meta["label_map"]

model = xgb.XGBClassifier()
model.load_model(os.path.join(OUT_DIR, "raw_model_v2.json"))

test = states[states["match_id"].isin(test_matches)].copy()
X_test = test[FEATURES].values
y_test = test["outcome"].map(label_map).values
probs = model.predict_proba(X_test)  # columns: loss, draw, win
test["p_loss"], test["p_draw"], test["p_win"] = probs[:, 0], probs[:, 1], probs[:, 2]

# --- Calibration: bucket predicted P(win) into deciles, compare to actual win rate ---
test["win_actual"] = (test["outcome"] == "win").astype(int)
test["p_win_bucket"] = pd.cut(test["p_win"], bins=np.arange(0, 1.01, 0.1), include_lowest=True)

calib = test.groupby("p_win_bucket", observed=True).agg(
    n=("win_actual", "size"),
    predicted_mean=("p_win", "mean"),
    actual_rate=("win_actual", "mean"),
).reset_index()
print("=== Calibration: predicted P(win) bucket vs actual win rate ===")
print(calib.to_string(index=False))

# --- Sanity checks on specific known states ---
def predict_state(goals_for, goals_against, minutes_remaining, home_flag):
    x = np.array([[goals_for, goals_against, minutes_remaining, home_flag]])
    p = model.predict_proba(x)[0]
    return {"p_loss": p[0], "p_draw": p[1], "p_win": p[2]}

print("\n=== Sanity checks ===")
checks = [
    ("0-0, kickoff, home", 0, 0, 90, 1),
    ("0-0, kickoff, away", 0, 0, 90, 0),
    ("+3 goals, 5 min left, home", 3, 0, 5, 1),
    ("+1 goal, 1 min left, home", 1, 0, 1, 1),
    ("-1 goal, 1 min left, home", 0, 1, 1, 1),
    ("0-0, 1 min left, home", 0, 0, 1, 1),
    ("+1 goal, 89 min left (early), home", 1, 0, 89, 1),
    ("-3 goals, 5 min left, home", 0, 3, 5, 1),
]
for label, gf, ga, mr, hf in checks:
    r = predict_state(gf, ga, mr, hf)
    print(f"{label:40s} -> P(win)={r['p_win']:.3f} P(draw)={r['p_draw']:.3f} P(loss)={r['p_loss']:.3f}")

# --- Expected points at each check state ---
print("\n=== Expected points ===")
for label, gf, ga, mr, hf in checks:
    r = predict_state(gf, ga, mr, hf)
    ep = 3 * r["p_win"] + 1 * r["p_draw"]
    print(f"{label:40s} -> EP = {ep:.3f}")
