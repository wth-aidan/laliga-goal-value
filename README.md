# How much was that goal worth? Valuing La Liga goals by expected points added

"It's not about how many goals you score, it's when you score them." This project turns that claim into a number.

A goal's value is how much it changed the scoring team's **expected points** (3 × P(win) + 1 × P(draw)) at the moment it went in. Going 1–0 up in the 89th minute of a level game is worth a lot. Making it 6–2 is worth almost nothing. Summing these values over a season rewards players whose goals moved results, not just players who scored often.

This is the same idea as Win Probability Added in baseball analytics, adapted to football's three outcomes.

## Key findings

**Griezmann's goals mattered more than Suárez's in 2018/19, despite six fewer of them.**

| 2018/19 | Goals | Expected points added (raw) | Strength-adjusted |
|---|---|---|---|
| Antoine Griezmann | 15 | 13.31 | 9.55 |
| Luis Suárez | 21 | 11.61 | 7.60 |

The gap holds after adjusting for team strength. So it isn't just "Barcelona were winning anyway". Griezmann's goals came at more decisive moments.

**Stuani (Girona, 19 goals) came within 5% of Messi (36 goals) once adjusted for team strength** in 2018/19: 14.26 vs 14.93 strength-adjusted expected points added. Goals for a weaker team move results more, because that team is rarely already cruising.

**Suárez's move from Barcelona to Atlético.** At Barcelona (2014/15–2019/20), each Suárez goal added 0.37–0.61 expected points. At Atlético in 2020/21 that jumped to 0.80, and his 20.7 total topped the league, above Messi's 30 goals. Same player, different team context: close to a natural experiment on how much team dominance dilutes a goal's marginal value.

**Messi and Ronaldo rank lower per goal than volume suggests** (Messi: 0.43–0.65 per goal across seasons). This isn't the metric penalising great players. Even the raw model, which knows nothing about team strength, values those goals lower, because so many came when the match was already won. That is the question the metric answers. It is not a player-quality ranking.

## Method

1. **Match states.** From Understat shot-level data, every match is rebuilt as a timeline of states: goals for, goals against, minutes remaining and home/away. Snapshots are taken at kickoff, after every goal, and every 5 minutes, so late 0–0 states are represented.
2. **Win/draw/loss model.** An XGBoost multiclass model estimates P(win), P(draw) and P(loss) from any state. Exact states (say, "2–1 up with 23 minutes left") are too sparse to look up directly, so the model smooths across neighbouring states. Hyperparameters were chosen by 5-fold cross-validation grouped by match, so snapshots from one match never appear in both training and validation.
3. **Strength adjustment.** A second model adds pre-match win and draw probabilities from bookmaker odds, with the overround removed. The raw model answers "how much did this goal change the match?". The adjusted model answers "how much did it change the match relative to what was expected?". Both are reported, and their difference is shown as `environment_gap`.
4. **Goal value.** For each goal, value = expected points after − expected points before, evaluated at the same minute.

## Validation

Held-out test set: 20% of matches (564 matches, 23,796 snapshots), never seen in training or tuning.

| Model | Test log-loss |
|---|---|
| Baseline (always predict historical outcome frequencies) | 1.077 |
| Raw model (score + time) | 0.774 |
| Strength-adjusted model | 0.714 |

Calibration of the raw model on the test set:

| Predicted P(win) bucket | States | Mean predicted | Actual win rate |
|---|---|---|---|
| 0.0–0.1 | 5,440 | 0.04 | 0.04 |
| 0.1–0.2 | 2,263 | 0.16 | 0.16 |
| 0.2–0.3 | 5,277 | 0.27 | 0.27 |
| 0.3–0.4 | 1,683 | 0.36 | 0.40 |
| 0.4–0.5 | 2,627 | 0.43 | 0.46 |
| 0.5–0.6 | 541 | 0.57 | 0.52 |
| 0.6–0.7 | 881 | 0.64 | 0.63 |
| 0.7–0.8 | 2,273 | 0.74 | 0.77 |
| 0.8–0.9 | 984 | 0.84 | 0.86 |
| 0.9–1.0 | 1,827 | 0.96 | 0.97 |

Predicted probabilities are within about 5 percentage points of observed outcomes in every bucket.

## Limitations

- **Stoppage time.** The source data has no exact added-time length, so minutes remaining is clipped at 0 from minute 90. As a result, a +1 lead with "1 minute left" shows P(win) ≈ 0.86, because several minutes of added time are still to play. This slightly undervalues very late goals.
- **In-sample goal values.** Goal values are computed with models trained on about 80% of the same matches. With shallow trees on four to six features, overfitting is limited, but out-of-fold predictions would be cleaner.
- **Log-loss is measured on snapshots.** Many snapshots are late-game states that are close to certain, which makes the improvement over the baseline look larger than it would at kickoff alone.
- **Penalties** are not yet separated from open-play goals. Own goals are excluded from player credit.
- **2021/22 is partial** in the source data (389 goals, against 920–1,100 for other seasons).

## Data

- Shots and match results: [douglasbc/scraping-understat-dataset](https://github.com/douglasbc/scraping-understat-dataset), La Liga 2014/15–2021/22
- Bookmaker odds: [xgabora/Club-Football-Match-Data-2000-2025](https://github.com/xgabora/Club-Football-Match-Data-2000-2025) (Football-Data.co.uk)

All 2,826 matches were matched to pre-match odds. Two matches whose shot data didn't reconstruct the final score were dropped. In total, 7,470 goals were valued.

## Outputs

- `results/goal_values.csv`: every goal, with the score before and after, pre-match win probability, raw value, adjusted value and environment gap.
- `results/leaderboard_full.csv`: per-season player totals.

## How to run

Requires Python 3.10+ with `pandas>=2.2`, `scikit-learn` and `xgboost`. On macOS, XGBoost also needs `brew install libomp`.

1. Download both datasets as ZIPs from GitHub and arrange them like this:
   ```
   laliga_goal_value/
   ├── laliga_goal_value_code/
   ├── scraping-understat-dataset-main/
   └── repo2/
       └── Club-Football-Match-Data-main/
   ```
2. Run the scripts in order:
   ```
   cd laliga_goal_value_code
   for f in 01 02 03 04 05 06 07 08 09; do python3 ${f}_*.py || break; done
   ```

Outputs are written to `output/`. The full pipeline runs in a few minutes on a laptop.

| Script | Step |
|---|---|
| 01 | Parse shots into per-match goal timelines, and check them against final scores |
| 02 | Build state snapshots (kickoff, after each goal, every 5 minutes) |
| 03 | Train and cross-validate the raw model; test log-loss and baseline |
| 04 | Calibration table and sanity checks |
| 05 | Merge bookmaker odds by date and team; remove the overround |
| 06 | Add pre-match strength features to states |
| 07 | Train the strength-adjusted model |
| 08 | Compute raw and adjusted value for every goal |
| 09 | Per-season player leaderboards |

---

*Methodology and analysis by Aidan Wong. Implementation written with AI coding assistance.*
