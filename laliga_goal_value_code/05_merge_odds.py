"""
Build pre_match_win_prob feature from Football-Data.co.uk bookmaker odds
(via the xgabora/Club-Football-Match-Data-2000-2025 repo), matched to our
Understat-based La Liga matches by date + team name.
"""
import pandas as pd
import numpy as np
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # folder containing this code folder and the data

OUT_DIR = BASE + "/output"
os.makedirs(OUT_DIR, exist_ok=True)
ODDS_PATH = BASE + "/repo2/Club-Football-Match-Data-main/data/Matches.csv"

# Team name crosswalk: Understat name -> Football-Data name
NAME_MAP = {
    "Alaves": "Alaves", "Almeria": "Almeria", "Athletic Club": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid", "Barcelona": "Barcelona", "Cadiz": "Cadiz",
    "Celta Vigo": "Celta", "Cordoba": "Cordoba", "Deportivo La Coruna": "La Coruna",
    "Eibar": "Eibar", "Elche": "Elche", "Espanyol": "Espanol", "Getafe": "Getafe",
    "Girona": "Girona", "Granada": "Granada", "Las Palmas": "Las Palmas",
    "Leganes": "Leganes", "Levante": "Levante", "Malaga": "Malaga",
    "Mallorca": "Mallorca", "Osasuna": "Osasuna", "Rayo Vallecano": "Vallecano",
    "Real Betis": "Betis", "Real Madrid": "Real Madrid", "Real Sociedad": "Sociedad",
    "Real Valladolid": "Valladolid", "SD Huesca": "Huesca", "Sevilla": "Sevilla",
    "Sporting Gijon": "Sp Gijon", "Valencia": "Valencia", "Villarreal": "Villarreal",
}

match_info = pd.read_pickle(os.path.join(OUT_DIR, "match_info.pkl"))

odds = pd.read_csv(ODDS_PATH, low_memory=False)
odds = odds[odds["Division"] == "SP1"].copy()
odds["MatchDate"] = pd.to_datetime(odds["MatchDate"])
odds["HomeTeam_mapped"] = odds["HomeTeam"]
odds["AwayTeam_mapped"] = odds["AwayTeam"]

match_info["h_team_mapped"] = match_info["h_team"].map(NAME_MAP)
match_info["a_team_mapped"] = match_info["a_team"].map(NAME_MAP)

unmapped = match_info[match_info["h_team_mapped"].isna() | match_info["a_team_mapped"].isna()]
print(f"Unmapped teams: {unmapped[['h_team','a_team']].drop_duplicates()}")

# We don't have exact kickoff dates in match_info (understat CSV date is in shots file).
# Pull match dates from the shots data instead.
shots_sample = pd.concat([
    pd.read_csv(f) for f in __import__("glob").glob(
        BASE + "/scraping-understat-dataset-main/datasets/la_liga/shots_la_liga_*.csv")
], ignore_index=True)
match_dates = shots_sample.groupby("match_id")["date"].first().reset_index()
match_dates["date"] = pd.to_datetime(match_dates["date"]).dt.normalize()
match_info = match_info.merge(match_dates, on="match_id", how="left")

# Merge on (date, home_team_mapped, away_team_mapped). Allow +/-1 day slack for timezone issues.
odds_small = odds[["MatchDate", "HomeTeam_mapped", "AwayTeam_mapped", "OddHome", "OddDraw", "OddAway"]].copy()

merged = match_info.merge(
    odds_small,
    left_on=["date", "h_team_mapped", "a_team_mapped"],
    right_on=["MatchDate", "HomeTeam_mapped", "AwayTeam_mapped"],
    how="left",
)
print(f"Matched directly: {merged['OddHome'].notna().sum()} / {len(merged)}")

# retry unmatched with +-1 day
unmatched = merged[merged["OddHome"].isna()].copy()
if len(unmatched) > 0:
    for delta in [-1, 1, -2, 2]:
        still_unmatched = unmatched[unmatched["OddHome"].isna()] if "OddHome" in unmatched else unmatched
        odds_shift = odds_small.copy()
        odds_shift["MatchDate"] = odds_shift["MatchDate"] + pd.Timedelta(days=delta)
        retry = match_info[match_info["match_id"].isin(unmatched["match_id"])].merge(
            odds_shift,
            left_on=["date", "h_team_mapped", "a_team_mapped"],
            right_on=["MatchDate", "HomeTeam_mapped", "AwayTeam_mapped"],
            how="left",
        )
        found = retry[retry["OddHome"].notna()]
        for _, row in found.iterrows():
            idx = merged[merged["match_id"] == row["match_id"]].index
            merged.loc[idx, ["OddHome", "OddDraw", "OddAway"]] = row[["OddHome", "OddDraw", "OddAway"]].values
        unmatched = merged[merged["OddHome"].isna()]

print(f"Matched after date-slack retries: {merged['OddHome'].notna().sum()} / {len(merged)}")
still_missing = merged[merged["OddHome"].isna()]
print(f"Still missing: {len(still_missing)}")
if len(still_missing) > 0:
    print(still_missing[["match_id", "h_team", "a_team", "date"]].head(10))

# --- Convert odds to probabilities, removing the bookmaker overround ---
merged["raw_p_home"] = 1 / merged["OddHome"]
merged["raw_p_draw"] = 1 / merged["OddDraw"]
merged["raw_p_away"] = 1 / merged["OddAway"]
overround = merged["raw_p_home"] + merged["raw_p_draw"] + merged["raw_p_away"]
merged["pre_match_p_home_win"] = merged["raw_p_home"] / overround
merged["pre_match_p_draw"] = merged["raw_p_draw"] / overround
merged["pre_match_p_away_win"] = merged["raw_p_away"] / overround

merged.to_pickle(os.path.join(OUT_DIR, "match_info_with_odds.pkl"))
print("\nSaved match_info_with_odds.pkl")
print(merged[["h_team", "a_team", "date", "OddHome", "OddDraw", "OddAway",
              "pre_match_p_home_win", "pre_match_p_draw", "pre_match_p_away_win"]].head(10))
