"""
Parse Understat shot-level CSVs (8 La Liga seasons) into match-state snapshots.

For every match we build a timeline of goal events (Goal + OwnGoal, own goals
credited to the opposing team), then generate a snapshot row for each team at:
  - kickoff (0-0, 90 minutes remaining)
  - immediately after every goal event

Each row records: goals_for, goals_against, minutes_remaining, home_flag,
and the match's final outcome (win/draw/loss) from that team's perspective.
"""
import pandas as pd
import numpy as np
import glob
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # folder containing this code folder and the data

DATA_DIR = BASE + "/scraping-understat-dataset-main/datasets/la_liga"
OUT_DIR = BASE + "/output"
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(DATA_DIR, "shots_la_liga_*.csv")))
print(f"Found {len(files)} season files")

all_shots = []
for f in files:
    df = pd.read_csv(f)
    all_shots.append(df)
shots = pd.concat(all_shots, ignore_index=True)
print(f"Total shot rows: {len(shots)}")
print(f"Total matches: {shots['match_id'].nunique()}")

# --- Build per-match info: teams + final score ---
match_info = shots.groupby("match_id").agg(
    h_team=("h_team", "first"),
    a_team=("a_team", "first"),
    h_goals_final=("h_goals", "first"),
    a_goals_final=("a_goals", "first"),
    season=("season", "first"),
).reset_index()
print(f"Matches with info: {len(match_info)}")

# --- Build goal event timelines ---
goal_events = shots[shots["result"].isin(["Goal", "OwnGoal"])].copy()

def scoring_side(row):
    # For a normal Goal, the scoring team is whichever side (h_a) took the shot.
    # For an OwnGoal, h_a records the *unlucky* team, so the goal is credited
    # to the OTHER side.
    if row["result"] == "Goal":
        return row["h_a"]
    else:  # OwnGoal
        return "a" if row["h_a"] == "h" else "h"

goal_events["scoring_side"] = goal_events.apply(scoring_side, axis=1)
goal_events = goal_events.sort_values(["match_id", "minute"])

print("\nSample goal timeline (one match):")
sample_mid = goal_events["match_id"].iloc[0]
print(goal_events[goal_events["match_id"] == sample_mid][
    ["minute", "result", "h_a", "scoring_side", "player"]
].to_string())

# --- Sanity check: does reconstructed final score match h_goals_final/a_goals_final? ---
def reconstruct_final(g):
    h = (g["scoring_side"] == "h").sum()
    a = (g["scoring_side"] == "a").sum()
    return pd.Series({"h_goals_recon": h, "a_goals_recon": a})

recon = goal_events.groupby("match_id").apply(reconstruct_final, include_groups=False).reset_index()
check = match_info.merge(recon, on="match_id", how="left").fillna(0)
mismatches = check[
    (check["h_goals_final"] != check["h_goals_recon"]) |
    (check["a_goals_final"] != check["a_goals_recon"])
]
print(f"\nMatches where reconstructed score matches final score: "
      f"{len(check) - len(mismatches)} / {len(check)}")
print(f"Mismatches: {len(mismatches)}")
if len(mismatches) > 0:
    print(mismatches.head(10).to_string())

match_info.to_pickle(os.path.join(OUT_DIR, "match_info.pkl"))
goal_events.to_pickle(os.path.join(OUT_DIR, "goal_events.pkl"))
print("\nSaved match_info.pkl and goal_events.pkl")
