# MLDS Hackathon — Mode Choice Prediction

## Overview
A two-session hackathon hosted on Kaggle as part of CEGE0004: Machine Learning for Data Science at UCL. The task was to predict the probability of each transport mode (DRIVE, WALK, CYCLE, PASSENGER, PUBLICTRANSPORT, OTHER) for trips recorded in the Victorian Integrated Survey of Travel and Activity (VISTA), a household travel survey from Melbourne, Australia. Submissions were evaluated on multiclass log-loss.

**Final result: 3rd place**

[Kaggle Page for CEGE0004 Hackathon 2026](https://www.kaggle.com/competitions/cege0004-hackathon-2026/overview)

---

## Data
Three relational tables were provided — households, persons, and trips — reflecting the natural hierarchy of the survey. Trips were merged as the base unit of prediction, with person and household attributes joined in via `persid` and `hhid` respectively. The merged dataset contained 13,641 trips across 63 features.

---

## Approach

### Feature Engineering
Domain knowledge about transport behaviour drove most of the feature engineering decisions:

- **Speed proxy** (`cumdist / travtime`) — different modes have characteristic speeds. A trip at high km/min is almost certainly a car; slow trips are likely walking or cycling
- **Distance thresholds** — binary flags for walkable (<1km), cycleable (<5km), short (<2km), and long (>20km) trips, encoding the physical feasibility of each mode
- **Vehicle and bike access** — whether the household owns any cars or bikes, since mode choice is constrained by what is available
- **No licence flag and interaction** — a person without a car licence cannot drive; combined with long trip distance this strongly predicts public transport usage
- **Vehicles per person** — a household with 3 cars and 2 people has very different car availability than one with 1 car and 5 people
- **Trip purpose flags** — work and education trips have distinct mode distributions compared to recreational trips
- **Trip duration** — derived from arrival and start times as an independent signal from the recorded travel time

Redundant columns were removed — the seven individual WFH day columns were dropped in favour of the summary `anywfh` flag, and `homeregion_ASGS` was dropped as it was fully captured by the more granular `homesubregion_ASGS`.

### Model
XGBoost was chosen for its suitability to heterogeneous tabular data — handling mixed feature types, missing values, and class imbalance without requiring feature scaling. The following regularisation parameters were tuned to address overfitting:

- `max_depth=4` — shallower trees to prevent memorisation
- `learning_rate=0.05` — cautious updates to reduce variance
- `subsample=0.8` and `colsample_bytree=0.8` — stochastic sampling of rows and features per tree
- `min_child_weight=5` — minimum samples required to create a leaf node
- `reg_lambda=2` — L2 regularisation penalty

Once hyperparameters were established, the final model was retrained on the full training dataset to maximise exposure to rare classes (CYCLE: 250 samples, OTHER: 78 samples).

### Encoding
All categorical features were label-encoded using saved encoders fitted on the training data, with unseen test values mapped to the most frequent known class to avoid transform errors.

---

## Reflection
The main limitation was the use of a random train/validation split rather than a grouped split by household. Because multiple trips from the same person appeared in both the training and validation sets during hyperparameter tuning, the validation logloss (0.22) was overly optimistic. The final model was retrained on the full dataset without a validation split, which is why the public leaderboard score (0.498) reflects true out-of-sample performance. A `GroupShuffleSplit` on `hhid` would have produced a more honest validation signal and likely improved hyperparameter tuning. This is a key learning point for future work with hierarchical panel data.

---

## Results

| Submission | Public Score |
|---|---|
| XGBoost baseline | 0.631 |
| XGBoost + regularisation | 0.517 |
| XGBoost + feature engineering | 0.504 |
| XGBoost + full dataset | 0.499 |
| XGBoost + full dataset final score | 0.457 |

**Final leaderboard position: 3rd**
