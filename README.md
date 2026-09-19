# Freight Rate Prediction Challenge

Predicts `posted_rate` for freight loads from lane, equipment, weight, date
and market-signal features, using historical loads in `data/train_test.csv`

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:
   
## Setup
```bash
python -m pip install -r requirements.txt
```
## Run

1. Open `freight_rate_ml_assessment.ipynb` in Jupyter or Google Colab and run all cells in order.
   It trains the model, prints holdout validation metrics, and writes:
   - `validation_predictions.csv` — predictions for all 12,000 loads in `data/validation.csv`
   - `data/december_chart_inputs.csv` — the fixed December lane with `predicted_rate` filled in

2. Generate the required chart with the provided scorer:
```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

## Approach summary

- **Validation split:** time-based, not random — trained on the first 80% of dates
  (Jan–Aug 2025), tested on the last 20% (Sep–Oct 2025), since the real validation
  set is out-of-time (Nov–Dec 2025). Result: MAPE ≈ 5.45%, MAE ≈ $126.
- **Model:** HistGradientBoostingRegressor (scikit-learn), trained on log(posted_rate),
  selected after comparing against linear regression and XGBoost on the same holdout.
- **Features:** distance (raw + log), weight, market_index, quote_signal,
  pickup/delivery coordinates, equipment (one-hot), and date-derived features
  (month, day-of-week, cyclical day-of-year sin/cos).
