"""
Per-season goalscorer leaderboard using goal_value instead of raw goal count.
Ranks by total raw_value (primary) and shows adjusted_value + environment_gap
alongside, plus traditional goal count for comparison.
"""
import pandas as pd
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # folder containing this code folder and the data

OUT_DIR = BASE + "/output"
os.makedirs(OUT_DIR, exist_ok=True)

goal_values = pd.read_pickle(os.path.join(OUT_DIR, "goal_values.pkl"))
match_info = pd.read_pickle(os.path.join(OUT_DIR, "match_info_with_odds.pkl"))[["match_id", "season"]]
goal_values = goal_values.merge(match_info, on="match_id", how="left")

# season is stored as e.g. 2015.0 meaning the 2015/16 season
goal_values["season_label"] = goal_values["season"].astype(int).astype(str) + "/" + \
    (goal_values["season"].astype(int) + 1).astype(str).str[-2:]

# Exclude own goals from player credit (own goals aren't attributed to a scorer)
scoring = goal_values[~goal_values["is_own_goal"]].copy()

leaderboard = scoring.groupby(["season_label", "player"]).agg(
    goals=("raw_value", "size"),
    total_raw_value=("raw_value", "sum"),
    total_adjusted_value=("adjusted_value", "sum"),
    total_environment_gap=("environment_gap", "sum"),
    avg_raw_value_per_goal=("raw_value", "mean"),
).reset_index()

leaderboard = leaderboard.sort_values(["season_label", "total_raw_value"], ascending=[True, False])

leaderboard.to_csv(os.path.join(OUT_DIR, "leaderboard_full.csv"), index=False)
print(f"Full leaderboard saved: {len(leaderboard)} player-seasons")

# Print top 10 per season
for season in sorted(scoring["season_label"].unique()):
    print(f"\n=== {season} — Top 10 by total goal value (raw_value) ===")
    top = leaderboard[leaderboard["season_label"] == season].head(10)
    print(top[["player", "goals", "total_raw_value", "total_adjusted_value",
               "avg_raw_value_per_goal"]].round(3).to_string(index=False))
