"""
Build state snapshots v2: kickoff + after every goal + periodic snapshots
every 5 minutes throughout the match (using whatever the score was at that
point). This fixes the gap where "no goal yet" states (e.g. still 0-0 late
in the match) were never represented in training data.
"""
import pandas as pd
import numpy as np
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # folder containing this code folder and the data

OUT_DIR = BASE + "/output"
os.makedirs(OUT_DIR, exist_ok=True)

match_info = pd.read_pickle(os.path.join(OUT_DIR, "match_info.pkl"))
goal_events = pd.read_pickle(os.path.join(OUT_DIR, "goal_events.pkl"))

recon = goal_events.groupby("match_id")["scoring_side"].apply(
    lambda s: pd.Series({"h": (s == "h").sum(), "a": (s == "a").sum()})
).unstack()
bad = match_info.merge(recon, on="match_id", how="left").fillna(0)
bad_ids = bad[(bad["h_goals_final"] != bad["h"]) | (bad["a_goals_final"] != bad["a"])]["match_id"]
match_info = match_info[~match_info["match_id"].isin(bad_ids)].copy()
goal_events = goal_events[~goal_events["match_id"].isin(bad_ids)].copy()

PERIODIC_MINUTES = list(range(0, 91, 5))  # 0, 5, 10, ..., 90

rows = []
for _, m in match_info.iterrows():
    mid = m["match_id"]
    h_final, a_final = m["h_goals_final"], m["a_goals_final"]

    if h_final > a_final:
        h_outcome, a_outcome = "win", "loss"
    elif h_final < a_final:
        h_outcome, a_outcome = "loss", "win"
    else:
        h_outcome, a_outcome = "draw", "draw"

    mgoals = goal_events[goal_events["match_id"] == mid].sort_values("minute")

    # full goal timeline (minute, h_score_after, a_score_after)
    goal_timeline = [(0.0, 0, 0)]
    h_run, a_run = 0, 0
    for _, g in mgoals.iterrows():
        if g["scoring_side"] == "h":
            h_run += 1
        else:
            a_run += 1
        goal_timeline.append((g["minute"], h_run, a_run))

    # snapshot points = goal event minutes UNION periodic minutes, deduped
    goal_minutes = [t[0] for t in goal_timeline]
    all_minutes = sorted(set(goal_minutes) | set(PERIODIC_MINUTES))

    def score_at(minute):
        # score just after all goals with goal_minute <= minute
        h, a = 0, 0
        for gm, h_after, a_after in goal_timeline:
            if gm <= minute:
                h, a = h_after, a_after
            else:
                break
        return h, a

    is_kickoff_or_goal = set(goal_minutes)

    for minute in all_minutes:
        h_score, a_score = score_at(minute)
        mins_remaining = max(0.0, 90.0 - minute)
        source = "goal_or_kickoff" if minute in is_kickoff_or_goal else "periodic"
        rows.append({
            "match_id": mid, "season": m["season"], "team": m["h_team"],
            "opponent": m["a_team"], "home_flag": 1,
            "goals_for": h_score, "goals_against": a_score,
            "minute": minute, "minutes_remaining": mins_remaining,
            "outcome": h_outcome, "source": source,
        })
        rows.append({
            "match_id": mid, "season": m["season"], "team": m["a_team"],
            "opponent": m["h_team"], "home_flag": 0,
            "goals_for": a_score, "goals_against": h_score,
            "minute": minute, "minutes_remaining": mins_remaining,
            "outcome": a_outcome, "source": source,
        })

states = pd.DataFrame(rows)
print(f"Total snapshot rows: {len(states)} (was 20588 in v1)")
print(f"Rows per match (avg): {len(states) / states['match_id'].nunique():.2f}")
print(states["source"].value_counts())

near = states[(states["goals_for"] == 0) & (states["goals_against"] == 0) & (states["minutes_remaining"] < 10)]
print(f"\n0-0 rows with <10 min remaining: {len(near)} (was 0 in v1)")

states.to_pickle(os.path.join(OUT_DIR, "states_v2.pkl"))
print("\nSaved states_v2.pkl")
