# Formula 1 Race Prediction System Documentation

## 1. Introduction

This project predicts Formula 1 race finishing positions using machine learning.

The system is trained on historical Formula 1 race data from multiple seasons and uses qualifying results from an upcoming race weekend to forecast the likely race outcome.

---

## 2. System Architecture

```text
FastF1 Data
     │
     ▼
Season Dataset Creation
     │
     ▼
Feature Engineering
     │
     ▼
Training Matrix
     │
     ▼
Random Forest Model
     │
     ▼
Race Prediction
```

---

## 3. Dataset

### Training Seasons

| Season |
|----------|
| 2022 |
| 2023 |
| 2024 |
| 2025 |

### Prediction Season

| Season |
|----------|
| 2026 |

---

## 4. Data Collection

Historical race data is collected using the FastF1 API.

Information collected:

- Driver
- Team
- Grid Position
- Finishing Position
- Points
- Race Results

The collected data is stored as season CSV files.

Example:

```text
2022_season.csv
2023_season.csv
2024_season.csv
2025_season.csv
```

---

## 5. Feature Engineering

The system generates machine learning features from historical race results.

Features include:

### Grid Position

Starting position obtained from qualifying.

Example:

```text
Driver: VER
Grid Position: 2
```

### Driver Average Finish

Average finishing position across historical races.

### Team Average Finish

Average team finishing position.

### Recent Driver Form

Driver performance in recent races.

### Recent Team Form

Constructor performance in recent races.

---

## 6. Model Training

### Algorithm

Random Forest Regressor

Reasons for selection:

- Handles non-linear relationships
- Works well with tabular data
- Robust against overfitting
- Interpretable feature importance

Training process:

```text
Historical Data
      │
      ▼
Feature Matrix
      │
      ▼
Random Forest Training
      │
      ▼
Saved Model
```

Output:

```text
f1_model.pkl
```

---

## 7. Prediction Workflow

### Step 1

Wait for qualifying session to finish.

### Step 2

Update qualifying positions in:

```text
predict_2026.py
```

### Step 3

Run prediction script.

```bash
python predict_2026.py
```

### Step 4

Model generates predicted finishing positions.

### Step 5

Display:

- Predicted winner
- Predicted podium
- Full predicted classification

---

## 8. Example Prediction

### Australian Grand Prix 2026

Predicted Winner:

```text
RUS
```

Predicted Podium:

```text
1. RUS
2. ANT
3. LEC
```

---

## 9. Evaluation

The project includes:

### Backtesting

```bash
python backtest.py
```

Used to evaluate prediction accuracy using historical races.

### Forecast Validation

```bash
python forecast_validation.py
```

Compares predicted race outcomes against actual race results.

---

## 10. Known Limitations

The model does not currently account for:

- Weather changes
- Tire strategy
- Safety cars
- Mechanical failures
- Driver penalties
- Race incidents

These factors can significantly affect race outcomes.

---

## 11. Future Work

Planned improvements:

1. Dynamic driver ratings
2. Dynamic constructor ratings
3. Circuit-specific performance models
4. Weather integration
5. Monte Carlo simulation engine
6. Championship prediction model

---

## 12. Conclusion

This project demonstrates how machine learning can be applied to Formula 1 race forecasting using historical performance data and qualifying results.

While race outcomes remain inherently uncertain, the model provides a data-driven estimate of likely finishing positions and race winners.
