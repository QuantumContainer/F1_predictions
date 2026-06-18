# 🏎️ Formula 1 Race Prediction System

A machine learning-based Formula 1 race prediction system built using historical race data from FastF1.

The model is trained on previous Formula 1 seasons and uses qualifying performance, team strength, driver performance, and race history to predict finishing positions for upcoming races.

---

## Features

- Train on multiple Formula 1 seasons
- Automatic data collection using FastF1
- Feature engineering from race and qualifying data
- Predict future race outcomes
- Evaluate model performance using historical backtesting
- Generate race winner predictions before race weekend

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
├── f1_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

The model uses historical Formula 1 race data obtained through FastF1.

Training seasons:

- 2022
- 2023
- 2024
- 2025

Prediction season:

- 2026

---

## Machine Learning Pipeline

### Data Collection

Race results and qualifying data are collected using FastF1.

### Feature Engineering

Features include:

- Grid Position
- Driver Average Finish
- Team Average Finish
- Driver Championship Standing
- Team Championship Standing
- Recent Driver Form
- Recent Team Form

### Model Training

The model learns relationships between:

```text
Qualifying Performance
+
Driver Performance
+
Constructor Performance
↓
Predicted Race Finish
```

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

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How To Use

### 1. Train Model

Train using historical seasons:

```bash
python train_production_model.py
```

This creates:

```text
f1_model.pkl
```

---

### 2. Predict Upcoming Race

Example:

```bash
python predict_2026.py
```

Output:

```text
Predicted Winner:
1. Verstappen
2. Norris
3. Leclerc
4. Piastri
...
```

---

### 3. Backtest Model

Evaluate how well the model performs on historical races.

```bash
python backtest.py
```

Example Output:

```text
Winner Accuracy: 72%
Podium Accuracy: 64%
Top 10 Accuracy: 58%
```

---

### 4. Validate Forecasts

```bash
python forecast_validation.py
```

Compares model predictions against actual race results.

---

## Example Prediction

Japanese Grand Prix

```text
1. Max Verstappen
2. Lando Norris
3. Charles Leclerc
4. Oscar Piastri
5. George Russell
```

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

- Weather-aware predictions
- Safety car probability modelling
- Circuit-specific driver performance
- Tire degradation modelling
- Monte Carlo race simulations
- Neural network-based ranking models

---

## Disclaimer

Formula 1 contains many unpredictable variables such as:

- Weather
- Safety Cars
- Mechanical Failures
- Strategy Decisions
- Driver Errors

Predictions should be considered probabilistic rather than exact outcomes.

---

## Author

Bhavesh Sahu

B.Tech Electronics and Communication Engineering (ECE)
The LNM Institute of Information Technology (LNMIIT)
