
import os
import json
import warnings
import hashlib
import sqlite3
from datetime import datetime, timedelta
from functools import wraps

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify)


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, 'database', 'users.db')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CHARTS_DIR = os.path.join(BASE_DIR, 'static', 'charts')

os.makedirs(os.path.join(BASE_DIR, 'database'), exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'gridguardian_secret_2024_xk9!@#'
app.permanent_session_lifetime = timedelta(hours=8)


def load_model(name):
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        return joblib.load(path)
    return None

demand_model      = load_model('demand_model.pkl')
demand_scaler     = load_model('demand_scaler.pkl')
demand_features   = load_model('demand_features.pkl')
demand_metrics    = load_model('demand_metrics.pkl')
demand_stats      = load_model('demand_stats.pkl')
best_demand_name  = load_model('best_demand_name.pkl')

renewable_model   = load_model('renewable_model.pkl')
renewable_scaler  = load_model('renewable_scaler.pkl')
solar_features    = load_model('solar_features.pkl')
solar_metrics     = load_model('solar_metrics.pkl')
solar_stats       = load_model('solar_stats.pkl')
best_solar_name   = load_model('best_solar_name.pkl')

anomaly_model     = load_model('anomaly_model.pkl')
anomaly_scaler    = load_model('anomaly_scaler.pkl')
anomaly_features  = load_model('anomaly_features.pkl')
anomaly_stats_data= load_model('anomaly_stats.pkl')

grid_model        = load_model('grid_stability_model.pkl')
grid_scaler       = load_model('grid_stability_scaler.pkl')
grid_feat_cols    = load_model('grid_features.pkl')


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS forecast_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            forecast_type TEXT,
            input_data TEXT,
            prediction REAL,
            model_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS renewable_forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            solar_forecast REAL,
            wind_estimate REAL,
            renewable_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            anomaly_type TEXT,
            severity TEXT,
            description TEXT,
            demand_value REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            grid_score REAL,
            recommendation_text TEXT,
            priority TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

init_db()


def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def log_action(user_id, action, details=''):
    try:
        conn = get_db()
        conn.execute("INSERT INTO system_logs (user_id,action,details) VALUES (?,?,?)",
                     (user_id, action, details))
        conn.commit()
        conn.close()
    except Exception:
        pass

def compute_grid_health(num_anomalies, demand_val, forecast_val, solar_val):
    score = 100.0
    anom_rate = min(num_anomalies / 500, 1.0)
    score -= anom_rate * 30
    if demand_val > 0 and forecast_val > 0:
        error_pct = abs(demand_val - forecast_val) / demand_val * 100
        score -= min(error_pct * 0.5, 20)
    solar_max = solar_stats.get('max_solar', 5000) if solar_stats else 5000
    renewable_ratio = min(solar_val / (solar_max + 1e-6), 1.0)
    score += renewable_ratio * 10
    score = max(0, min(100, score))
    return round(score, 1)

def grid_status_label(score):
    if score >= 90: return 'Excellent', 'success'
    if score >= 75: return 'Good', 'info'
    if score >= 60: return 'Moderate', 'warning'
    if score >= 40: return 'Degraded', 'danger'
    return 'Critical', 'critical'

def generate_recommendations(grid_score, demand_val, forecast_val, solar_val, num_anomalies):
    recs = []
    hour = datetime.now().hour

    if forecast_val > demand_val * 1.1:
        recs.append({'priority': 'HIGH', 'icon': '⚡',
                     'text': f'High demand surge expected (+{((forecast_val-demand_val)/demand_val*100):.1f}%). '
                             'Activate reserve generation units immediately.',
                     'action': 'Activate Reserve'})
    if 17 <= hour <= 21:
        recs.append({'priority': 'HIGH', 'icon': '🕐',
                     'text': 'Peak demand window (5 PM–9 PM) is active. '
                             'Deploy battery storage and defer non-critical loads.',
                     'action': 'Shift Loads'})
    if num_anomalies > 10:
        recs.append({'priority': 'CRITICAL', 'icon': '🚨',
                     'text': f'{num_anomalies} grid anomalies detected in recent data. '
                             'Inspect transmission lines and switchgear immediately.',
                     'action': 'Inspect Grid'})
    if solar_val < (solar_stats.get('avg_solar', 2000) if solar_stats else 2000) * 0.5:
        recs.append({'priority': 'MEDIUM', 'icon': '☁️',
                     'text': 'Solar generation is below 50% of average. '
                             'Increase reliance on backup generation or grid imports.',
                     'action': 'Adjust Mix'})
    if grid_score < 70:
        recs.append({'priority': 'HIGH', 'icon': '🔧',
                     'text': f'Grid health score is {grid_score}/100. '
                             'Run full diagnostic and schedule preventive maintenance.',
                     'action': 'Run Diagnostic'})
    if grid_score >= 85:
        recs.append({'priority': 'LOW', 'icon': '✅',
                     'text': 'Grid is operating optimally. '
                             'Continue monitoring and consider increasing renewable storage buffer.',
                     'action': 'Monitor'})
    if 22 <= hour or hour <= 5:
        recs.append({'priority': 'LOW', 'icon': '🌙',
                     'text': 'Off-peak hours active. '
                             'Ideal window for grid maintenance and battery charging cycles.',
                     'action': 'Schedule Maintenance'})
    if not recs:
        recs.append({'priority': 'LOW', 'icon': '📊',
                     'text': 'No immediate action required. All systems within normal parameters.',
                     'action': 'Monitor'})
    return recs

def make_demand_prediction(hour, day, weekday, month, quarter, season,
                            is_weekend, is_peak, lag_1h, lag_24h, lag_168h,
                            roll_mean_6h, roll_mean_24h, roll_std_24h, demand_change):
    if demand_model is None: return None
    row = np.array([[hour, day, weekday, month, quarter, season, is_weekend,
                     is_peak, lag_1h, lag_24h, lag_168h,
                     roll_mean_6h, roll_mean_24h, roll_std_24h, demand_change]])
    row_sc = demand_scaler.transform(row)
    return float(demand_model.predict(row_sc)[0])

def make_solar_prediction(hour, day, month, weekday, is_daylight,
                           efficiency, lag_1h, lag_24h, roll_mean_3h):
    if renewable_model is None: return None
    row = np.array([[hour, day, month, weekday, is_daylight,
                     efficiency, lag_1h, lag_24h, roll_mean_3h]])
    row_sc = renewable_scaler.transform(row)
    return max(0.0, float(renewable_model.predict(row_sc)[0]))


@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            flash('Please fill in all fields.', 'danger')
            return render_template('login.html')
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        conn.close()
        if user and user['password_hash'] == hash_password(password):
            session.permanent = True
            session['user_id'] = user['id']
            session['username'] = user['username']
            log_action(user['id'], 'LOGIN', f'User {username} logged in')
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm', '')
        if not all([username, email, password, confirm]):
            flash('Please fill in all fields.', 'danger')
            return render_template('register.html')
        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('register.html')
        try:
            conn = get_db()
            conn.execute("INSERT INTO users (username,email,password_hash) VALUES (?,?,?)",
                         (username, email, hash_password(password)))
            conn.commit()
            conn.close()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists.', 'danger')
    return render_template('register.html')

@app.route('/logout')
def logout():
    uid = session.get('user_id')
    if uid: log_action(uid, 'LOGOUT')
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    now = datetime.now()
    # Use realistic values from training data
    if demand_stats:
        current_demand  = demand_stats['current_demand']
        forecast_demand = demand_stats['forecast_demand']
    else:
        current_demand  = 15000.0
        forecast_demand = 15500.0

    if solar_stats:
        current_solar  = solar_stats['current_solar']
        forecast_solar = solar_stats['forecast_solar']
    else:
        current_solar  = 2500.0
        forecast_solar = 2800.0

    num_anomalies = anomaly_stats_data.get('num_anomalies', 5) if anomaly_stats_data else 5

    grid_score = compute_grid_health(num_anomalies, current_demand, forecast_demand, current_solar)
    status_label, status_class = grid_status_label(grid_score)
    recs = generate_recommendations(grid_score, current_demand, forecast_demand, current_solar, num_anomalies)[:3]

    demand_change_pct = ((forecast_demand - current_demand) / current_demand * 100) if current_demand else 0
    solar_change_pct  = ((forecast_solar - current_solar) / current_solar * 100) if current_solar else 0

    # 24-hour forecast trend data (for JS chart)
    trend_hours  = list(range(24))
    trend_demand = [current_demand * (0.85 + 0.15 * np.sin(h / 24 * 2 * np.pi + 0.5)) + np.random.normal(0, 200)
                    for h in range(24)]
    trend_solar  = [max(0, current_solar * np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0
                    for h in range(24)]

    # Store prediction in DB
    conn = get_db()
    conn.execute("INSERT INTO forecast_history (user_id,forecast_type,input_data,prediction,model_used) "
                 "VALUES (?,?,?,?,?)",
                 (session['user_id'], 'DASHBOARD',
                  json.dumps({'hour': now.hour, 'demand': current_demand}),
                  forecast_demand, str(best_demand_name)))
    conn.commit()
    conn.close()

    return render_template('dashboard.html',
        now=now, grid_score=grid_score,
        status_label=status_label, status_class=status_class,
        current_demand=current_demand, forecast_demand=forecast_demand,
        current_solar=current_solar, forecast_solar=forecast_solar,
        num_anomalies=num_anomalies, recs=recs,
        demand_change_pct=round(demand_change_pct, 1),
        solar_change_pct=round(solar_change_pct, 1),
        trend_hours=json.dumps(trend_hours),
        trend_demand=json.dumps([round(v, 1) for v in trend_demand]),
        trend_solar=json.dumps([round(v, 1) for v in trend_solar]),
        best_demand_model=best_demand_name, best_solar_model=best_solar_name)


@app.route('/demand_forecast', methods=['GET', 'POST'])
@login_required
def demand_forecast():
    now = datetime.now()
    prediction = None
    inputs = {}
    if request.method == 'POST':
        try:
            hour       = int(request.form.get('hour', now.hour))
            day        = int(request.form.get('day', now.day))
            weekday    = int(request.form.get('weekday', now.weekday()))
            month      = int(request.form.get('month', now.month))
            quarter    = (month - 1) // 3 + 1
            season     = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[month]
            is_weekend = int(weekday >= 5)
            is_peak    = int(17 <= hour <= 21)
            lag_1h     = float(request.form.get('lag_1h', demand_stats['current_demand'] if demand_stats else 14000))
            lag_24h    = float(request.form.get('lag_24h', demand_stats['current_demand'] if demand_stats else 14000))
            lag_168h   = float(request.form.get('lag_168h', demand_stats['current_demand'] if demand_stats else 14000))
            roll_mean_6h  = float(request.form.get('roll_mean_6h', lag_1h))
            roll_mean_24h = float(request.form.get('roll_mean_24h', lag_1h))
            roll_std_24h  = float(request.form.get('roll_std_24h', 300))
            demand_change = float(request.form.get('demand_change', 0))

            prediction = make_demand_prediction(hour, day, weekday, month, quarter, season,
                                                 is_weekend, is_peak, lag_1h, lag_24h, lag_168h,
                                                 roll_mean_6h, roll_mean_24h, roll_std_24h, demand_change)
            inputs = dict(request.form)

            if prediction:
                conn = get_db()
                conn.execute("INSERT INTO forecast_history (user_id,forecast_type,input_data,prediction,model_used) "
                             "VALUES (?,?,?,?,?)",
                             (session['user_id'], 'DEMAND', json.dumps(inputs), prediction, str(best_demand_name)))
                conn.commit()
                conn.close()
                log_action(session['user_id'], 'DEMAND_FORECAST', f'Prediction: {prediction:.1f} MW')
        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'danger')

    metrics = demand_metrics or []
    return render_template('demand_forecast.html',
        now=now, prediction=prediction, inputs=inputs,
        metrics=metrics, best_model=best_demand_name,
        demand_stats=demand_stats)


@app.route('/renewable_forecast', methods=['GET', 'POST'])
@login_required
def renewable_forecast():
    now = datetime.now()
    prediction = None
    inputs = {}
    if request.method == 'POST':
        try:
            hour       = int(request.form.get('hour', now.hour))
            day        = int(request.form.get('day', now.day))
            month      = int(request.form.get('month', now.month))
            weekday    = int(request.form.get('weekday', now.weekday()))
            is_daylight = int(6 <= hour <= 18)
            lag_1h     = float(request.form.get('lag_1h', solar_stats['current_solar'] if solar_stats else 2000))
            lag_24h    = float(request.form.get('lag_24h', solar_stats['current_solar'] if solar_stats else 2000))
            roll_mean_3h = float(request.form.get('roll_mean_3h', lag_1h))
            efficiency = float(request.form.get('efficiency', 0.85))

            prediction = make_solar_prediction(hour, day, month, weekday, is_daylight,
                                                efficiency, lag_1h, lag_24h, roll_mean_3h)
            inputs = dict(request.form)

            if prediction is not None:
                conn = get_db()
                conn.execute("INSERT INTO renewable_forecasts (user_id,solar_forecast,wind_estimate,renewable_score) "
                             "VALUES (?,?,?,?)",
                             (session['user_id'], prediction,
                              prediction * 0.3,
                              min(100, prediction / (solar_stats.get('max_solar', 5000)+1e-6) * 100)))
                conn.commit()
                conn.close()
                log_action(session['user_id'], 'SOLAR_FORECAST', f'Prediction: {prediction:.1f} kW')
        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'danger')

    metrics = solar_metrics or []
    return render_template('renewable_forecast.html',
        now=now, prediction=prediction, inputs=inputs,
        metrics=metrics, best_model=best_solar_name,
        solar_stats=solar_stats)


@app.route('/anomaly_detection', methods=['GET', 'POST'])
@login_required
def anomaly_detection():
    result = None
    if request.method == 'POST':
        try:
            demand_mw      = float(request.form.get('demand_mw', 14000))
            lag_1h         = float(request.form.get('lag_1h', 14000))
            lag_24h        = float(request.form.get('lag_24h', 14000))
            roll_mean_24h  = float(request.form.get('roll_mean_24h', 14000))
            roll_std_24h   = float(request.form.get('roll_std_24h', 300))
            demand_change  = float(request.form.get('demand_change', 0))

            row = np.array([[demand_mw, lag_1h, lag_24h, roll_mean_24h, roll_std_24h, demand_change]])
            row_sc = anomaly_scaler.transform(row)
            pred  = anomaly_model.predict(row_sc)[0]
            score = float(anomaly_model.score_samples(row_sc)[0])

            is_anomaly = pred == -1
            severity   = 'Normal'
            if is_anomaly:
                if score < -0.3: severity = 'Critical'
                elif score < -0.1: severity = 'High'
                else: severity = 'Medium'

            result = {
                'is_anomaly': is_anomaly,
                'score': round(score, 4),
                'severity': severity,
                'demand_mw': demand_mw
            }

            if is_anomaly:
                conn = get_db()
                conn.execute("INSERT INTO anomalies (user_id,anomaly_type,severity,description,demand_value) "
                             "VALUES (?,?,?,?,?)",
                             (session['user_id'], 'DEMAND_SPIKE', severity,
                              f'Anomaly detected at {demand_mw:.0f} MW (score={score:.3f})', demand_mw))
                conn.commit()
                conn.close()
                log_action(session['user_id'], 'ANOMALY_DETECTED', f'Severity: {severity}, Demand: {demand_mw}')
        except Exception as e:
            flash(f'Detection error: {str(e)}', 'danger')

    anomaly_stats = anomaly_stats_data or {'num_anomalies': 5, 'total': 500, 'anomaly_rate': 0.01}
    conn = get_db()
    recent_anomalies = conn.execute(
        "SELECT * FROM anomalies WHERE user_id=? ORDER BY created_at DESC LIMIT 10",
        (session['user_id'],)).fetchall()
    conn.close()

    return render_template('anomaly_detection.html',
        result=result, anomaly_stats=anomaly_stats,
        recent_anomalies=recent_anomalies)


@app.route('/recommendation_center')
@login_required
def recommendation_center():
    if demand_stats: current_demand = demand_stats['current_demand']; forecast_demand = demand_stats['forecast_demand']
    else: current_demand = forecast_demand = 14000.0
    if solar_stats: current_solar = solar_stats['current_solar']
    else: current_solar = 2000.0
    num_anomalies = anomaly_stats_data.get('num_anomalies', 5) if anomaly_stats_data else 5
    grid_score = compute_grid_health(num_anomalies, current_demand, forecast_demand, current_solar)
    recs = generate_recommendations(grid_score, current_demand, forecast_demand, current_solar, num_anomalies)
    status_label, status_class = grid_status_label(grid_score)

    conn = get_db()
    for rec in recs:
        conn.execute("INSERT INTO recommendations (user_id,grid_score,recommendation_text,priority) "
                     "VALUES (?,?,?,?)",
                     (session['user_id'], grid_score, rec['text'], rec['priority']))
    past_recs = conn.execute(
        "SELECT * FROM recommendations WHERE user_id=? ORDER BY created_at DESC LIMIT 20",
        (session['user_id'],)).fetchall()
    conn.commit()
    conn.close()

    return render_template('recommendation_center.html',
        recs=recs, past_recs=past_recs, grid_score=grid_score,
        status_label=status_label, status_class=status_class,
        current_demand=current_demand, forecast_demand=forecast_demand,
        current_solar=current_solar)


@app.route('/simulation_lab', methods=['GET', 'POST'])
@login_required
def simulation_lab():
    sim_result = None
    if request.method == 'POST':
        try:
            base_demand     = float(request.form.get('base_demand', demand_stats['current_demand'] if demand_stats else 14000))
            demand_growth   = float(request.form.get('demand_growth', 0))
            temperature     = float(request.form.get('temperature', 25))
            renewable_pct   = float(request.form.get('renewable_pct', 30))
            hour            = int(request.form.get('hour', datetime.now().hour))

            sim_demand = base_demand * (1 + demand_growth / 100)
            temp_factor = 1 + max(0, (temperature - 20)) * 0.005
            sim_demand *= temp_factor

            solar_base = solar_stats.get('avg_solar', 2000) if solar_stats else 2000
            sim_solar  = solar_base * (renewable_pct / 100) * (1.2 if 9 <= hour <= 15 else 0.4)
            sim_solar  = max(0, sim_solar)

            sim_anomalies = anomaly_stats_data.get('num_anomalies', 5) if anomaly_stats_data else 5
            if demand_growth > 20: sim_anomalies += int(demand_growth / 5)

            sim_score = compute_grid_health(sim_anomalies, base_demand, sim_demand, sim_solar)
            sim_status, sim_class = grid_status_label(sim_score)
            sim_recs  = generate_recommendations(sim_score, base_demand, sim_demand, sim_solar, sim_anomalies)

            sim_result = {
                'sim_demand': round(sim_demand, 1),
                'sim_solar': round(sim_solar, 1),
                'sim_score': sim_score, 'sim_status': sim_status,
                'sim_class': sim_class, 'sim_recs': sim_recs,
                'sim_anomalies': sim_anomalies,
                'demand_growth': demand_growth,
                'temperature': temperature,
                'renewable_pct': renewable_pct,
            }
            log_action(session['user_id'], 'SIMULATION',
                       f'Demand growth={demand_growth}%, Temp={temperature}°C, Solar={renewable_pct}%')
        except Exception as e:
            flash(f'Simulation error: {str(e)}', 'danger')

    base_vals = {
        'base_demand': demand_stats['current_demand'] if demand_stats else 14000,
        'solar_avg': solar_stats['avg_solar'] if solar_stats else 2000,
    }
    return render_template('simulation_lab.html', sim_result=sim_result, base_vals=base_vals)


@app.route('/analytics')
@login_required
def analytics():
    conn = get_db()
    total_forecasts  = conn.execute("SELECT COUNT(*) FROM forecast_history WHERE user_id=?",
                                    (session['user_id'],)).fetchone()[0]
    total_anomalies  = conn.execute("SELECT COUNT(*) FROM anomalies WHERE user_id=?",
                                    (session['user_id'],)).fetchone()[0]
    total_recs       = conn.execute("SELECT COUNT(*) FROM recommendations WHERE user_id=?",
                                    (session['user_id'],)).fetchone()[0]
    recent_forecasts = conn.execute(
        "SELECT * FROM forecast_history WHERE user_id=? ORDER BY created_at DESC LIMIT 10",
        (session['user_id'],)).fetchall()
    conn.close()

    dm = demand_metrics or []
    sm = solar_metrics or []

    # Build chart data for JS
    if dm:
        d_names = [r['name'] for r in dm]
        d_rmse  = [r['rmse']  for r in dm]
        d_r2    = [r['r2']    for r in dm]
        d_mae   = [r['mae']   for r in dm]
    else:
        d_names = d_rmse = d_r2 = d_mae = []

    if sm:
        s_names = [r['name'] for r in sm]
        s_rmse  = [r['rmse']  for r in sm]
        s_r2    = [r['r2']    for r in sm]
    else:
        s_names = s_rmse = s_r2 = []

    return render_template('analytics.html',
        total_forecasts=total_forecasts,
        total_anomalies=total_anomalies,
        total_recs=total_recs,
        recent_forecasts=recent_forecasts,
        demand_metrics=dm, solar_metrics=sm,
        d_names=json.dumps(d_names), d_rmse=json.dumps(d_rmse),
        d_r2=json.dumps(d_r2), d_mae=json.dumps(d_mae),
        s_names=json.dumps(s_names), s_rmse=json.dumps(s_rmse),
        s_r2=json.dumps(s_r2),
        best_demand=best_demand_name, best_solar=best_solar_name)


@app.route('/history')
@login_required
def history():
    conn = get_db()
    forecasts  = conn.execute(
        "SELECT * FROM forecast_history WHERE user_id=? ORDER BY created_at DESC LIMIT 50",
        (session['user_id'],)).fetchall()
    anomalies  = conn.execute(
        "SELECT * FROM anomalies WHERE user_id=? ORDER BY created_at DESC LIMIT 20",
        (session['user_id'],)).fetchall()
    renewables = conn.execute(
        "SELECT * FROM renewable_forecasts WHERE user_id=? ORDER BY created_at DESC LIMIT 20",
        (session['user_id'],)).fetchall()
    logs       = conn.execute(
        "SELECT * FROM system_logs WHERE user_id=? ORDER BY created_at DESC LIMIT 30",
        (session['user_id'],)).fetchall()
    conn.close()
    return render_template('history.html',
        forecasts=forecasts, anomalies=anomalies,
        renewables=renewables, logs=logs)


@app.route('/api/live_stats')
@login_required
def api_live_stats():
    now = datetime.now()
    jitter = lambda v, pct: v * (1 + np.random.uniform(-pct, pct))
    d = demand_stats.get('current_demand', 14000) if demand_stats else 14000
    s = solar_stats.get('current_solar', 2000) if solar_stats else 2000
    n = anomaly_stats_data.get('num_anomalies', 5) if anomaly_stats_data else 5
    fd = demand_stats.get('forecast_demand', 14500) if demand_stats else 14500
    score = compute_grid_health(n, d, fd, s)
    return jsonify({
        'demand': round(jitter(d, 0.01), 1),
        'solar': round(max(0, jitter(s, 0.02)), 1),
        'grid_score': score,
        'anomalies': n,
        'timestamp': now.strftime('%H:%M:%S')
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
