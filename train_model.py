"""
GridGuardian AI - Complete ML Training Pipeline
Handles: Data Analysis, Preprocessing, Feature Engineering, Model Training
Datasets: AEP Hourly Demand, Plant Solar Generation, Smart Grid Stability, Delhi Weather
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, accuracy_score, classification_report)
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(BASE_DIR, 'static', 'charts')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

plt.style.use('dark_background')
NEON_CYAN   = '#00E5FF'
NEON_PURPLE = '#8B5CF6'
NEON_GREEN  = '#22D3EE'
NEON_ORANGE = '#F59E0B'
NEON_PINK   = '#EC4899'
BG_COLOR    = '#050816'
CARD_COLOR  = '#0B1120'

def styled_fig(figsize=(14, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(CARD_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1E293B')
    ax.tick_params(colors='#94A3B8', labelsize=9)
    ax.xaxis.label.set_color('#94A3B8')
    ax.yaxis.label.set_color('#94A3B8')
    ax.title.set_color('#E2E8F0')
    ax.grid(True, color='#1E293B', alpha=0.5, linewidth=0.5)
    return fig, ax

def save_chart(fig, name):
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  ✓ Saved {name}")

print("\n" + "="*60)
print("  GRIDGUARDIAN AI — TRAINING PIPELINE")
print("="*60)

print("\n[1/8] Loading datasets...")

aep = pd.read_csv(os.path.join(BASE_DIR, 'AEP_hourly.csv'), parse_dates=['Datetime'])
aep.rename(columns={'Datetime': 'timestamp', 'AEP_MW': 'demand_mw'}, inplace=True)
aep.sort_values('timestamp', inplace=True)
aep.drop_duplicates('timestamp', inplace=True)
print(f"  AEP Demand  : {aep.shape[0]:,} rows")


plant = pd.read_csv(os.path.join(BASE_DIR, 'Plant_1_Generation_Data.csv'),
                    parse_dates=['DATE_TIME'], dayfirst=True)
plant.rename(columns={'DATE_TIME': 'timestamp'}, inplace=True)
plant.sort_values('timestamp', inplace=True)

plant_agg = plant.groupby('timestamp').agg(
    dc_power=('DC_POWER', 'sum'),
    ac_power=('AC_POWER', 'sum'),
    daily_yield=('DAILY_YIELD', 'sum')
).reset_index()
print(f"  Solar Plant : {plant_agg.shape[0]:,} rows (aggregated)")

grid = pd.read_csv(os.path.join(BASE_DIR, 'smart_grid_stability_augmented.csv'))
print(f"  Grid Stability: {grid.shape[0]:,} rows")

delhi = pd.read_csv(os.path.join(BASE_DIR, 'DailyDelhiClimateTest.csv'),
                    parse_dates=['date'])
delhi.rename(columns={'date': 'timestamp'}, inplace=True)
print(f"  Delhi Weather: {delhi.shape[0]:,} rows")

print("\n[2/8] Engineering demand features...")

df_demand = aep.copy()
df_demand['demand_mw'].fillna(df_demand['demand_mw'].interpolate(), inplace=True)

mu, sigma = df_demand['demand_mw'].mean(), df_demand['demand_mw'].std()
df_demand = df_demand[(df_demand['demand_mw'] > mu - 3*sigma) &
                       (df_demand['demand_mw'] < mu + 3*sigma)]

df_demand['hour']      = df_demand['timestamp'].dt.hour
df_demand['day']       = df_demand['timestamp'].dt.day
df_demand['weekday']   = df_demand['timestamp'].dt.weekday
df_demand['month']     = df_demand['timestamp'].dt.month
df_demand['quarter']   = df_demand['timestamp'].dt.quarter
df_demand['is_weekend']= (df_demand['weekday'] >= 5).astype(int)
df_demand['season']    = df_demand['month'].map(
    {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3})
df_demand['is_peak']   = ((df_demand['hour'] >= 17) &
                           (df_demand['hour'] <= 21)).astype(int)

df_demand.sort_values('timestamp', inplace=True)
df_demand['lag_1h']   = df_demand['demand_mw'].shift(1)
df_demand['lag_24h']  = df_demand['demand_mw'].shift(24)
df_demand['lag_168h'] = df_demand['demand_mw'].shift(168)

df_demand['roll_mean_6h']  = df_demand['demand_mw'].shift(1).rolling(6).mean()
df_demand['roll_mean_24h'] = df_demand['demand_mw'].shift(1).rolling(24).mean()
df_demand['roll_std_24h']  = df_demand['demand_mw'].shift(1).rolling(24).std()
df_demand['demand_change'] = df_demand['demand_mw'].diff()
df_demand.dropna(inplace=True)
print(f"  Demand features: {df_demand.shape[1]} columns, {df_demand.shape[0]:,} rows")

print("\n[3/8] Engineering solar features...")

df_solar = plant_agg.copy()
df_solar['ac_power'].fillna(df_solar['ac_power'].interpolate(), inplace=True)
df_solar['dc_power'].fillna(df_solar['dc_power'].interpolate(), inplace=True)

df_solar['hour']       = df_solar['timestamp'].dt.hour
df_solar['day']        = df_solar['timestamp'].dt.day
df_solar['month']      = df_solar['timestamp'].dt.month
df_solar['weekday']    = df_solar['timestamp'].dt.weekday
df_solar['is_daylight']= ((df_solar['hour'] >= 6) & (df_solar['hour'] <= 18)).astype(int)
df_solar['efficiency'] = np.where(df_solar['dc_power'] > 0,
                                   df_solar['ac_power'] / (df_solar['dc_power'] + 1e-6), 0)
df_solar['lag_1h']     = df_solar['ac_power'].shift(1)
df_solar['lag_24h']    = df_solar['ac_power'].shift(24)
df_solar['roll_mean_3h'] = df_solar['ac_power'].shift(1).rolling(3).mean()
df_solar.dropna(inplace=True)
print(f"  Solar features: {df_solar.shape[1]} columns, {df_solar.shape[0]:,} rows")

print("\n[4/8] Processing grid stability data...")

df_grid = grid.copy()
le = LabelEncoder()
df_grid['stab_label'] = le.fit_transform(df_grid['stabf'])  # stable=1, unstable=0
joblib.dump(le, os.path.join(MODELS_DIR, 'stability_encoder.pkl'))

print("\n[5/8] Training demand forecasting models...")

demand_features = ['hour','day','weekday','month','quarter','season',
                   'is_weekend','is_peak','lag_1h','lag_24h','lag_168h',
                   'roll_mean_6h','roll_mean_24h','roll_std_24h','demand_change']
target_demand = 'demand_mw'

X_d = df_demand[demand_features]
y_d = df_demand[target_demand]

X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(X_d, y_d, test_size=0.2, shuffle=False)

scaler_demand = StandardScaler()
X_tr_d_sc = scaler_demand.fit_transform(X_tr_d)
X_te_d_sc  = scaler_demand.transform(X_te_d)

def eval_metrics(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100
    r2   = r2_score(y_true, y_pred)
    print(f"    {name:12s} → MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.2f}%  R²={r2:.4f}")
    return {'name': name, 'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}

demand_results = []
demand_models  = {}

# Random Forest
rf_d = RandomForestRegressor(n_estimators=150, max_depth=12, n_jobs=-1, random_state=42)
rf_d.fit(X_tr_d_sc, y_tr_d)
demand_models['RandomForest'] = rf_d
demand_results.append(eval_metrics(y_te_d, rf_d.predict(X_te_d_sc), 'RandomForest'))

# XGBoost
xgb_d = xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=42, verbosity=0)
xgb_d.fit(X_tr_d_sc, y_tr_d)
demand_models['XGBoost'] = xgb_d
demand_results.append(eval_metrics(y_te_d, xgb_d.predict(X_te_d_sc), 'XGBoost'))

# LightGBM
lgb_d = lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, verbose=-1)
lgb_d.fit(X_tr_d_sc, y_tr_d)
demand_models['LightGBM'] = lgb_d
demand_results.append(eval_metrics(y_te_d, lgb_d.predict(X_te_d_sc), 'LightGBM'))

best_d = min(demand_results, key=lambda x: x['rmse'])
best_demand_model = demand_models[best_d['name']]
print(f"\n  ★ Best demand model: {best_d['name']} (RMSE={best_d['rmse']:.1f})")

joblib.dump(best_demand_model, os.path.join(MODELS_DIR, 'demand_model.pkl'))
joblib.dump(scaler_demand,     os.path.join(MODELS_DIR, 'demand_scaler.pkl'))
joblib.dump(demand_features,   os.path.join(MODELS_DIR, 'demand_features.pkl'))
joblib.dump(demand_results,    os.path.join(MODELS_DIR, 'demand_metrics.pkl'))
joblib.dump(best_d['name'],    os.path.join(MODELS_DIR, 'best_demand_name.pkl'))

last_ts   = df_demand['timestamp'].values[-48:]
actual_48 = y_d.values[-48:]
pred_48   = best_demand_model.predict(scaler_demand.transform(X_d.iloc[-48:]))

fig, ax = styled_fig((14, 5))
ax.plot(range(48), actual_48, color=NEON_CYAN,   linewidth=2,   label='Actual Demand')
ax.plot(range(48), pred_48,   color=NEON_PURPLE, linewidth=2,   label='Predicted Demand', linestyle='--')
ax.fill_between(range(48), pred_48 * 0.97, pred_48 * 1.03, color=NEON_PURPLE, alpha=0.15)
ax.set_title('Electricity Demand Forecast — Last 48 Hours', fontsize=13, pad=12)
ax.set_xlabel('Hour Index')
ax.set_ylabel('Demand (MW)')
ax.legend(facecolor='#0B1120', edgecolor='#1E293B', labelcolor='#E2E8F0')
save_chart(fig, 'demand_forecast.png')

joblib.dump({
    'current_demand': float(actual_48[-1]),
    'forecast_demand': float(pred_48[-1]),
    'max_demand': float(actual_48.max()),
    'min_demand': float(actual_48.min()),
}, os.path.join(MODELS_DIR, 'demand_stats.pkl'))

print("\n[6/8] Training renewable (solar) forecasting models...")

solar_features = ['hour','day','month','weekday','is_daylight',
                  'efficiency','lag_1h','lag_24h','roll_mean_3h']
target_solar = 'ac_power'

X_s = df_solar[solar_features]
y_s = df_solar[target_solar]

X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_s, y_s, test_size=0.2, shuffle=False)

scaler_solar = StandardScaler()
X_tr_s_sc = scaler_solar.fit_transform(X_tr_s)
X_te_s_sc  = scaler_solar.transform(X_te_s)

solar_results = []
solar_models  = {}

rf_s = RandomForestRegressor(n_estimators=150, max_depth=10, n_jobs=-1, random_state=42)
rf_s.fit(X_tr_s_sc, y_tr_s)
solar_models['RandomForest'] = rf_s
solar_results.append(eval_metrics(y_te_s, rf_s.predict(X_te_s_sc), 'RandomForest'))

xgb_s = xgb.XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.05,
                           random_state=42, verbosity=0)
xgb_s.fit(X_tr_s_sc, y_tr_s)
solar_models['XGBoost'] = xgb_s
solar_results.append(eval_metrics(y_te_s, xgb_s.predict(X_te_s_sc), 'XGBoost'))

lgb_s = lgb.LGBMRegressor(n_estimators=200, max_depth=7, learning_rate=0.05,
                            random_state=42, verbose=-1)
lgb_s.fit(X_tr_s_sc, y_tr_s)
solar_models['LightGBM'] = lgb_s
solar_results.append(eval_metrics(y_te_s, lgb_s.predict(X_te_s_sc), 'LightGBM'))

best_s = min(solar_results, key=lambda x: x['rmse'])
best_solar_model = solar_models[best_s['name']]
print(f"\n  ★ Best solar model: {best_s['name']} (RMSE={best_s['rmse']:.1f})")

joblib.dump(best_solar_model, os.path.join(MODELS_DIR, 'renewable_model.pkl'))
joblib.dump(scaler_solar,     os.path.join(MODELS_DIR, 'renewable_scaler.pkl'))
joblib.dump(solar_features,   os.path.join(MODELS_DIR, 'solar_features.pkl'))
joblib.dump(solar_results,    os.path.join(MODELS_DIR, 'solar_metrics.pkl'))
joblib.dump(best_s['name'],   os.path.join(MODELS_DIR, 'best_solar_name.pkl'))

actual_s_48 = y_s.values[-48:]
pred_s_48   = best_solar_model.predict(scaler_solar.transform(X_s.iloc[-48:]))
pred_s_48   = np.clip(pred_s_48, 0, None)

fig, ax = styled_fig((14, 5))
ax.fill_between(range(48), actual_s_48, color=NEON_ORANGE, alpha=0.3, label='Actual Solar')
ax.plot(range(48), actual_s_48, color=NEON_ORANGE, linewidth=2)
ax.fill_between(range(48), pred_s_48, color=NEON_GREEN, alpha=0.2, label='Forecasted Solar')
ax.plot(range(48), pred_s_48, color=NEON_GREEN, linewidth=2, linestyle='--')
ax.set_title('Solar Energy Generation Forecast — Last 48 Periods', fontsize=13, pad=12)
ax.set_xlabel('Period Index')
ax.set_ylabel('AC Power (kW)')
ax.legend(facecolor='#0B1120', edgecolor='#1E293B', labelcolor='#E2E8F0')
save_chart(fig, 'renewable_forecast.png')

joblib.dump({
    'current_solar': float(actual_s_48[-1]),
    'forecast_solar': float(pred_s_48[-1]),
    'max_solar': float(actual_s_48.max()),
    'avg_solar': float(actual_s_48.mean()),
}, os.path.join(MODELS_DIR, 'solar_stats.pkl'))

print("\n[7/8] Training anomaly detection model...")


anomaly_features_cols = ['demand_mw','lag_1h','lag_24h','roll_mean_24h','roll_std_24h','demand_change']
X_anom = df_demand[anomaly_features_cols].dropna().sample(min(30000, len(df_demand)), random_state=42)

scaler_anom = StandardScaler()
X_anom_sc   = scaler_anom.fit_transform(X_anom)

iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
iso.fit(X_anom_sc)


recent_data  = df_demand[anomaly_features_cols].dropna().tail(500)
recent_sc    = scaler_anom.transform(recent_data)
anomaly_pred = iso.predict(recent_sc)          # -1 = anomaly, 1 = normal
anomaly_scores = iso.score_samples(recent_sc)  # lower = more anomalous

num_anomalies = (anomaly_pred == -1).sum()
print(f"  Anomalies detected (recent 500): {num_anomalies}")

# Anomaly chart
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), facecolor=BG_COLOR)
for ax in [ax1, ax2]:
    ax.set_facecolor(CARD_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1E293B')
    ax.tick_params(colors='#94A3B8', labelsize=9)
    ax.grid(True, color='#1E293B', alpha=0.5, linewidth=0.5)

demand_vals = recent_data['demand_mw'].values
normal_idx  = anomaly_pred == 1
anom_idx    = anomaly_pred == -1

ax1.plot(demand_vals, color=NEON_CYAN, linewidth=1.5, label='Demand (MW)', alpha=0.8)
ax1.scatter(np.where(anom_idx)[0], demand_vals[anom_idx],
            color='#EF4444', zorder=5, s=40, label=f'Anomalies ({num_anomalies})')
ax1.set_title('Anomaly Detection — Recent Demand Signal', fontsize=12, color='#E2E8F0', pad=10)
ax1.set_ylabel('Demand (MW)', color='#94A3B8')
ax1.legend(facecolor='#0B1120', edgecolor='#1E293B', labelcolor='#E2E8F0')
ax1.title.set_color('#E2E8F0')

ax2.fill_between(range(len(anomaly_scores)), anomaly_scores,
                  color=NEON_PURPLE, alpha=0.4)
ax2.plot(anomaly_scores, color=NEON_PURPLE, linewidth=1.2)
ax2.axhline(y=np.percentile(anomaly_scores, 10), color='#EF4444',
            linestyle='--', linewidth=1.5, label='Anomaly Threshold')
ax2.set_title('Anomaly Scores (lower = more anomalous)', fontsize=12, color='#E2E8F0', pad=10)
ax2.set_xlabel('Sample Index', color='#94A3B8')
ax2.set_ylabel('Score', color='#94A3B8')
ax2.legend(facecolor='#0B1120', edgecolor='#1E293B', labelcolor='#E2E8F0')
ax2.title.set_color('#E2E8F0')

plt.tight_layout(pad=2)
save_chart(fig, 'anomaly_chart.png')

joblib.dump(iso,          os.path.join(MODELS_DIR, 'anomaly_model.pkl'))
joblib.dump(scaler_anom,  os.path.join(MODELS_DIR, 'anomaly_scaler.pkl'))
joblib.dump(anomaly_features_cols, os.path.join(MODELS_DIR, 'anomaly_features.pkl'))
joblib.dump({'num_anomalies': int(num_anomalies), 'total': 500,
             'anomaly_rate': float(num_anomalies/500)},
            os.path.join(MODELS_DIR, 'anomaly_stats.pkl'))


print("\n[8/8] Generating extra charts & grid stability model...")

# Model comparison chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG_COLOR)
models_names = ['RandomForest', 'XGBoost', 'LightGBM']
colors = [NEON_CYAN, NEON_PURPLE, NEON_GREEN]

d_rmse = [r['rmse'] for r in demand_results]
ax = axes[0]
ax.set_facecolor(CARD_COLOR)
for s in ax.spines.values(): s.set_edgecolor('#1E293B')
ax.tick_params(colors='#94A3B8')
ax.grid(True, color='#1E293B', alpha=0.5, linewidth=0.5, axis='y')
bars = ax.bar(models_names, d_rmse, color=colors, alpha=0.85, width=0.5)
best_idx = d_rmse.index(min(d_rmse))
bars[best_idx].set_alpha(1.0)
bars[best_idx].set_edgecolor('#FFD700')
bars[best_idx].set_linewidth(2)
ax.set_title('Demand Forecast — Model RMSE Comparison', fontsize=11, color='#E2E8F0', pad=10)
ax.set_ylabel('RMSE (MW)', color='#94A3B8')
for bar, val in zip(bars, d_rmse):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
            f'{val:.0f}', ha='center', va='bottom', color='#E2E8F0', fontsize=10)

s_rmse = [r['rmse'] for r in solar_results]
ax = axes[1]
ax.set_facecolor(CARD_COLOR)
for s in ax.spines.values(): s.set_edgecolor('#1E293B')
ax.tick_params(colors='#94A3B8')
ax.grid(True, color='#1E293B', alpha=0.5, linewidth=0.5, axis='y')
bars2 = ax.bar(models_names, s_rmse, color=colors, alpha=0.85, width=0.5)
best_idx2 = s_rmse.index(min(s_rmse))
bars2[best_idx2].set_alpha(1.0)
bars2[best_idx2].set_edgecolor('#FFD700')
bars2[best_idx2].set_linewidth(2)
ax.set_title('Solar Forecast — Model RMSE Comparison', fontsize=11, color='#E2E8F0', pad=10)
ax.set_ylabel('RMSE (kW)', color='#94A3B8')
for bar, val in zip(bars2, s_rmse):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
            f'{val:.1f}', ha='center', va='bottom', color='#E2E8F0', fontsize=10)

plt.tight_layout(pad=2)
save_chart(fig, 'model_comparison.png')

fig, ax = styled_fig((12, 4))
ax.hist(df_demand['demand_mw'], bins=80, color=NEON_CYAN, alpha=0.7, edgecolor='#0B1120')
ax.axvline(df_demand['demand_mw'].mean(), color=NEON_ORANGE, linewidth=2, linestyle='--', label='Mean')
ax.axvline(df_demand['demand_mw'].median(), color=NEON_PURPLE, linewidth=2, linestyle='--', label='Median')
ax.set_title('Demand Distribution (Historical)', fontsize=12, pad=10)
ax.set_xlabel('Demand (MW)')
ax.set_ylabel('Frequency')
ax.legend(facecolor='#0B1120', edgecolor='#1E293B', labelcolor='#E2E8F0')
save_chart(fig, 'demand_distribution.png')


hourly_avg = df_demand.groupby('hour')['demand_mw'].mean()
fig, ax = styled_fig((12, 4))
ax.fill_between(hourly_avg.index, hourly_avg.values, color=NEON_CYAN, alpha=0.3)
ax.plot(hourly_avg.index, hourly_avg.values, color=NEON_CYAN, linewidth=2.5, marker='o', markersize=4)
ax.set_title('Average Demand by Hour of Day', fontsize=12, pad=10)
ax.set_xlabel('Hour')
ax.set_ylabel('Average Demand (MW)')
ax.set_xticks(range(24))
save_chart(fig, 'hourly_profile.png')

solar_hourly = df_solar.groupby('hour')['ac_power'].mean()
fig, ax = styled_fig((12, 4))
ax.fill_between(solar_hourly.index, solar_hourly.values, color=NEON_ORANGE, alpha=0.4)
ax.plot(solar_hourly.index, solar_hourly.values, color=NEON_ORANGE, linewidth=2.5, marker='o', markersize=4)
ax.set_title('Average Solar Generation by Hour', fontsize=12, pad=10)
ax.set_xlabel('Hour')
ax.set_ylabel('AC Power (kW)')
ax.set_xticks(range(24))
save_chart(fig, 'solar_profile.png')

grid_feat_cols = [c for c in df_grid.columns if c not in ['stabf', 'stab_label']]
X_g = df_grid[grid_feat_cols]
y_g = df_grid['stab_label']
X_tr_g, X_te_g, y_tr_g, y_te_g = train_test_split(X_g, y_g, test_size=0.2, random_state=42)
scaler_grid = StandardScaler()
X_tr_g_sc = scaler_grid.fit_transform(X_tr_g)
X_te_g_sc  = scaler_grid.transform(X_te_g)

rf_g = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
rf_g.fit(X_tr_g_sc, y_tr_g)
g_acc = accuracy_score(y_te_g, rf_g.predict(X_te_g_sc))
print(f"  Grid stability classifier accuracy: {g_acc:.4f}")

joblib.dump(rf_g,         os.path.join(MODELS_DIR, 'grid_stability_model.pkl'))
joblib.dump(scaler_grid,  os.path.join(MODELS_DIR, 'grid_stability_scaler.pkl'))
joblib.dump(grid_feat_cols, os.path.join(MODELS_DIR, 'grid_features.pkl'))

# Feature importance chart
fi = pd.Series(best_demand_model.feature_importances_
               if hasattr(best_demand_model, 'feature_importances_') else
               [0]*len(demand_features), index=demand_features).sort_values(ascending=True)
fig, ax = styled_fig((10, 6))
fi.plot(kind='barh', ax=ax, color=NEON_PURPLE, alpha=0.85)
ax.set_title('Feature Importance — Demand Model', fontsize=12, pad=10)
ax.set_xlabel('Importance Score')
save_chart(fig, 'feature_importance.png')

print("\n" + "="*60)
print("  ✅  TRAINING COMPLETE")
print("="*60)
print(f"  Models saved  : {MODELS_DIR}")
print(f"  Charts saved  : {CHARTS_DIR}")
print("  Run: python app.py")
print("="*60 + "\n")
