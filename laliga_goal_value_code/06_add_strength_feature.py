import pandas as pd
import numpy as np
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # folder containing this code folder and the data

OUT_DIR = BASE + "/output"
os.makedirs(OUT_DIR, exist_ok=True)

states = pd.read_pickle(os.path.join(OUT_DIR, "states_v2.pkl"))
odds_info = pd.read_pickle(os.path.join(OUT_DIR, "match_info_with_odds.pkl"))

odds_small = odds_info[["match_id", "pre_match_p_home_win", "pre_match_p_draw", "pre_match_p_away_win"]]
states = states.merge(odds_small, on="match_id", how="left")

# pre_match_win_prob from the perspective of the team the row belongs to
states["pre_match_win_prob"] = np.where(
    states["home_flag"] == 1,
    states["pre_match_p_home_win"],
    states["pre_match_p_away_win"],
)
states["pre_match_draw_prob"] = states["pre_match_p_draw"]

print(f"Missing pre_match_win_prob: {states['pre_match_win_prob'].isna().sum()} / {len(states)}")

states.to_pickle(os.path.join(OUT_DIR, "states_v3_with_strength.pkl"))
print("Saved states_v3_with_strength.pkl")
print(states[["team", "opponent", "home_flag", "goals_for", "goals_against",
              "minutes_remaining", "pre_match_win_prob", "pre_match_draw_prob", "outcome"]].head(6))
