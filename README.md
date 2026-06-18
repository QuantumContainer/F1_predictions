# 🏎️ Formula 1 Race Prediction System

A machine learning-based Formula 1 race prediction system that uses historical F1 race data and actual qualifying results to forecast race outcomes.

The model is trained on Formula 1 seasons from 2022 to 2025 and predicts finishing positions for future races using driver performance, team performance, and qualifying position.

---

## Project Overview

This project aims to answer a simple question:

> Given the qualifying results of an upcoming Formula 1 race, who is most likely to win?

The model learns patterns from historical race weekends and generates race finish predictions for future Grand Prix events.

---

## Features

- Historical F1 data collection using FastF1
- Automated season dataset generation
- Feature engineering for drivers and teams
- Machine learning-based race prediction
- Future race forecasting
- Model validation and backtesting
- Predicted winner and podium generation

---

## Training Data

The model is trained using:

- 2022 Formula 1 Season
- 2023 Formula 1 Season
- 2024 Formula 1 Season
- 2025 Formula 1 Season

Prediction target:

- 2026 Formula 1 Season

---

## Project Structure

```text
f1_predictor/
│
├── data/
│   └── seasons/
│       ├── 2022_season.csv
│       ├── 2023_season.csv
│       ├── 2024_season.csv
│       └── 2025_season.csv
│
├── feature_engineering.py
├── model.py
├── train_production_model.py
├── predict_2026.py
├── forecast_validation.py
├── backtest.py
│
├── requirements.txt
├── README.md
├── DOCUMENTATION.md
└── .gitignore
```

---

## Machine Learning Pipeline

```text
Historical Race Data
          │
          ▼
 Feature Engineering
          │
          ▼
 Training Dataset
          │
          ▼
 Random Forest Model
          │
          ▼
 Future Race Prediction
```

---

## How Race Prediction Works

The model requires the actual qualifying results for the race weekend.

Workflow:

1. Obtain qualifying results after the qualifying session.
2. Update the qualifying positions in `predict_2026.py`.
3. Run the prediction script.
4. The model combines:
   - Historical driver performance
   - Historical team performance
   - Grid position
   - Recent form
5. The model predicts the expected finishing order.

### Important

Predictions cannot be generated before qualifying because qualifying position is one of the strongest predictors of race performance.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/f1_predictor.git
cd f1_predictor
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How To Train The Model

Run:

```bash
python train_production_model.py
```

The script:

- Loads historical data
- Builds the feature matrix
- Trains the machine learning model
- Saves the trained model

Output:

```text
Model saved as f1_model.pkl
```

---

## How To Predict A Race

### Step 1

Open:

```text
predict_2026.py
```

### Step 2

Enter the actual qualifying positions.

Example:

```python
qualifying_positions = {
    "RUS": 1,
    "ANT": 2,
    "LEC": 3,
    "PIA": 4,
    ...
}
```

### Step 3

Run:

```bash
python predict_2026.py
```

The script will:

- Load historical data
- Train the model
- Generate race predictions
- Display the winner prediction
- Display the podium prediction

---

## Example Prediction

### 2026 Australian Grand Prix

Output:

```text
============================================================
2026 AUSTRALIAN GP PREDICTION
============================================================

Driver  PredictedFinish

RUS     3.47
ANT     3.48
LEC     5.70
PIA     7.18
HAD     7.24
NOR     8.71
VER     9.17
LAW     9.63
HAM    10.31
OCO    11.57
LIN    11.68
HUL    12.20
ALB    12.30
PER    12.61
BOR    12.89
BEA    13.10
GAS    13.40
COL    14.76
STR    14.95
BOT    15.00
ALO    15.20
SAI    15.86
```

### Predicted Winner

```text
RUS
```

### Predicted Podium

```text
1. RUS
2. ANT
3. LEC
```

Generated using:

- Historical seasons (2022–2025)
- Actual Australian GP qualifying positions
- Random Forest prediction model

---

## Technologies Used

- Python
- FastF1
- Pandas
- NumPy
- Scikit-Learn
- Joblib

---

## Future Improvements

- Circuit-specific driver ratings
- Weather-aware predictions
- Safety car probability modelling
- Tire strategy modelling
- Monte Carlo race simulations
- Championship prediction system

---

## Limitations

Formula 1 races are influenced by many unpredictable factors:

- Safety Cars
- Red Flags
- Weather
- Mechanical failures
- Strategy decisions
- Driver incidents

Predictions should therefore be treated as probabilistic forecasts rather than guaranteed results.

---

## Author

Bhavesh Sahu

B.Tech Electronics and Communication Engineering (ECE)

The LNM Institute of Information Technology (LNMIIT)
