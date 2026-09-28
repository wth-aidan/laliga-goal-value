"""
Compute goal_value = EP_after - EP_before for every goal in the dataset,
using both the raw model and the strength-adjusted model, plus the
environment_gap = raw_value - adjusted_value.
"""
import pandas as pd
import numpy as np
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # folder containing this code folder and the data
import pickle
import xgboost as xgb

OUT_DIR = BASE + "/output"
os.makedirs(OUT_DIR, exist_ok=True)

match_info = pd.read_pickle(os.path.join(OUT_DIR, "match_info_with_odds.pkl"))
goal_events = pd.read_pickle(os.path.join(OUT_DIR, "goal_events.pkl"))

# drop the 2 glitched matches, same as before
recon = goal_events.groupby("match_id")["scoring_side"].apply(
    lambda s: pd.Series({"h": (s == "h").sum(), "a": (s == "a").sum()})
).unstack()
bad = match_info.merge(recon, on="match_id", how="left").fillna(0)
bad_ids = bad[(bad["h_goals_final"] != bad["h"]) | (bad["a_goals_final"] != bad["a"])]["match_id"]
match_info = match_info[~match_info["match_id"].isin(bad_ids)].copy()
goal_events = goal_events[~goal_events["match_id"].isin(bad_ids)].copy()

raw_model = xgb.XGBClassifier()
raw_model.load_model(os.path.join(OUT_DIR, "raw_model_v2.json"))
strength_model = xgb.XGBClassifier()
strength_model.load_model(os.path.join(OUT_DIR, "strength_model.json"))

def ep_raw(gf, ga, mr, hf):
    x = np.array([[gf, ga, mr, hf]])
    p = raw_model.predict_proba(x)[0]  # loss, draw, win
    return 3 * p[2] + p[1]

def ep_strength(gf, ga, mr, hf, pmw, pmd):
    x = np.array([[gf, ga, mr, hf, pmw, pmd]])
    p = strength_model.predict_proba(x)[0]
    return 3 * p[2] + p[1]

results = []
for mid, mgoals in goal_events.groupby("match_id"):
    m = match_info[match_info["match_id"] == mid]
    if len(m) == 0:
        continue
    m = m.iloc[0]
    pmw_h, pmd, pmw_a = m["pre_match_p_home_win"], m["pre_match_p_draw"], m["pre_match_p_away_win"]

    mgoals = mgoals.sort_values("minute")
    h_run, a_run = 0, 0
    for _, g in mgoals.iterrows():
        minute = g["minute"]
        mins_remaining_before = max(0.0, 90.0 - minute)
        mins_remaining_after = mins_remaining_before  # instantaneous state change at same minute

        h_before, a_before = h_run, a_run
        if g["scoring_side"] == "h":
            h_run += 1
        else:
            a_run += 1
        h_after, a_after = h_run, a_run

        # scoring team's perspective
        if g["scoring_side"] == "h":
            gf_before, ga_before = h_before, a_before
            gf_after, ga_after = h_after, a_after
            home_flag = 1
            pmw, pmd_ = pmw_h, pmd
            conceding_home_flag = 0
            conceding_pmw = pmw_a
        else:
            gf_before, ga_before = a_before, h_before
            gf_after, ga_after = a_after, h_after
            home_flag = 0
            pmw, pmd_ = pmw_a, pmd
            conceding_home_flag = 1
            conceding_pmw = pmw_h

        # scoring team EP before/after
        ep_r_before = ep_raw(gf_before, ga_before, mins_remaining_before, home_flag)
        ep_r_after = ep_raw(gf_after, ga_after, mins_remaining_after, home_flag)
        ep_s_before = ep_strength(gf_before, ga_before, mins_remaining_before, home_flag, pmw, pmd_)
        ep_s_after = ep_strength(gf_after, ga_after, mins_remaining_after, home_flag, pmw, pmd_)

        raw_value = ep_r_after - ep_r_before
        adjusted_value = ep_s_after - ep_s_before

        results.append({
            "match_id": mid,
            "minute": minute,
            "player": g["player"],
            "scoring_team": m["h_team"] if g["scoring_side"] == "h" else m["a_team"],
            "conceding_team": m["a_team"] if g["scoring_side"] == "h" else m["h_team"],
            "is_own_goal": g["result"] == "OwnGoal",
            "score_before": f"{gf_before}-{ga_before}",
            "score_after": f"{gf_after}-{ga_after}",
            "pre_match_win_prob": pmw,
            "raw_value": raw_value,
            "adjusted_value": adjusted_value,
            "environment_gap": raw_value - adjusted_value,
        })

goal_values = pd.DataFrame(results)
print(f"Total goals valued: {len(goal_values)}")
print(goal_values[["raw_value", "adjusted_value", "environment_gap"]].describe())

goal_values.to_pickle(os.path.join(OUT_DIR, "goal_values.pkl"))
goal_values.to_csv(os.path.join(OUT_DIR, "goal_values.csv"), index=False)
print("\nSaved goal_values.pkl / goal_values.csv")
