# ⚡ GridGuardian AI
### AI-Powered Smart Grid Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.3-green)
![ML](https://img.shields.io/badge/ML-RandomForest%20%7C%20XGBoost%20%7C%20LightGBM-purple)
![License](https://img.shields.io/badge/License-MIT-cyan)

---

## 🎯 Overview

GridGuardian AI is a production-ready intelligent platform that predicts electricity demand,
forecasts solar generation, detects grid anomalies, and generates AI-driven recommendations —
all through a cyberpunk-themed, futuristic dashboard.

---

## 🚀 Features

| Module | Description |
|---|---|
| ⚡ Demand Forecast | Predicts next-hour MW demand using RF/XGBoost/LightGBM |
| ☀️ Solar Forecast | Forecasts solar AC output from inverter generation patterns |
| 🚨 Anomaly Detection | Isolation Forest detects spikes and grid instabilities |
| 💡 Recommendations | AI-generated actionable grid management advice |
| 🧪 Simulation Lab | What-if scenario testing with sliders |
| 📊 Analytics | Full model performance comparison and leaderboard |
| 📜 History | Complete audit trail of all predictions and events |

---

## 🗂 Datasets Used

| Dataset | Rows | Purpose |
|---|---|---|
| AEP_hourly.csv | 121,273 | Demand forecasting (AEP power company, 2004–2018) |
| Plant_1_Generation_Data.csv | 68,778 | Solar energy generation |
| smart_grid_stability_augmented.csv | 60,000 | Grid stability classification |
| DailyDelhiClimateTest.csv | 114 | Weather context features |

---

## 🛠 Tech Stack

- **Frontend:** HTML5, CSS3, Bootstrap 5, Chart.js, Font Awesome
- **Backend:** Flask 3, Python 3.11
- **ML:** Random Forest, XGBoost, LightGBM, Isolation Forest
- **Database:** SQLite3
- **Visualisation:** Matplotlib
- **Deployment:** Render (free tier)

---

## ⚙️ Local Setup

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/GridGuardianAI.git
cd GridGuardianAI
pip install -r requirements.txt
```

### 2. Add Datasets

Place the following CSV files in the project root:

```
AEP_hourly.csv
Plant_1_Generation_Data.csv
smart_grid_stability_augmented.csv
DailyDelhiClimateTest.csv
```

### 3. Train Models

```bash
python train_model.py
```

This will:
- Load and preprocess all 4 datasets
- Engineer 15+ features per dataset
- Train Random Forest, XGBoost, LightGBM for each task
- Auto-select the best model per task
- Save models to `models/`
- Generate 8 charts to `static/charts/`

Expected output:
```
★ Best demand model: RandomForest (RMSE=21.9)
★ Best solar model: RandomForest (RMSE=1153.6)
Grid stability classifier accuracy: 1.0000
✅ TRAINING COMPLETE
```

### 4. Run the App

```bash
python app.py
```

Visit: **http://localhost:5000**

Register a new account and start using the dashboard.

---

## 🌐 Render Deployment

### Option A: render.yaml (recommended)

1. Push your repo to GitHub (including CSVs and trained `.pkl` files)
2. Go to [render.com](https://render.com) → New Web Service
3. Connect GitHub repo → Render auto-detects `render.yaml`
4. Click **Deploy**

The build command runs `python train_model.py` automatically.

### Option B: Manual

- **Build command:** `pip install -r requirements.txt && python train_model.py`
- **Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
- **Runtime:** Python 3.11

> **Note:** On Render's free tier, models are re-trained on every deploy.
> For faster deploys, commit your `models/*.pkl` files to Git.

---

## 📁 Project Structure

```
GridGuardianAI/
├── static/
│   ├── css/
│   │   ├── style.css          # Core cyberpunk theme
│   │   ├── dashboard.css      # Dashboard-specific styles
│   │   └── animations.css     # All animations
│   ├── js/
│   │   ├── script.js          # Clock, counters, sidebar
│   │   ├── dashboard.js       # Chart.js dashboard charts
│   │   └── simulation.js      # Simulation lab sliders/gauge
│   ├── charts/                # Matplotlib-generated PNGs
│   └── images/
├── templates/
│   ├── base.html              # Master layout + sidebar
│   ├── login.html             # Auth
│   ├── register.html          # Auth
│   ├── dashboard.html         # Main KPI dashboard
│   ├── demand_forecast.html   # Demand prediction
│   ├── renewable_forecast.html# Solar prediction
│   ├── anomaly_detection.html # Anomaly scanner
│   ├── recommendation_center.html
│   ├── simulation_lab.html    # What-if scenarios
│   ├── analytics.html         # Model leaderboards
│   └── history.html           # Audit trail
├── models/                    # Trained .pkl files
├── database/                  # SQLite users.db
├── train_model.py             # Full ML pipeline
├── app.py                     # Flask application
├── requirements.txt
├── Procfile
├── runtime.txt
├── render.yaml
└── README.md
```

---

## 🧠 ML Model Performance

### Demand Forecasting (120,842 training samples)

| Model | MAE (MW) | RMSE (MW) | R² |
|---|---|---|---|
| **RandomForest ★** | 9.4 | 21.9 | **0.9999** |
| XGBoost | 46.4 | 64.2 | 0.9993 |
| LightGBM | 47.4 | 65.8 | 0.9993 |

### Solar Forecasting (3,134 training samples)

| Model | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|
| **RandomForest ★** | 400.8 | 1153.6 | **0.9791** |
| XGBoost | 414.7 | 1180.3 | 0.9782 |
| LightGBM | 463.6 | 1197.0 | 0.9775 |

### Grid Stability Classifier

| Model | Accuracy |
|---|---|
| Random Forest | **100%** |

---

## 🔒 Security

- Passwords are SHA-256 hashed before storage
- Session management via Flask sessions
- All routes protected with `@login_required`
- Input validation on all forms

---

## 👤 Author

Built for AI/ML Hackathon — GridGuardian AI  
Solo Developer Submission

---

## 📄 License

MIT License — free to use, modify, and deploy.
