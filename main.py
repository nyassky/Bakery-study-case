import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────

data = pd.read_csv('data/data_bakery.csv')

print("=== BEFORE CLEANING ===")
print(data.info())
print(data.head())

# ─────────────────────────────────────────
# DATA CLEANING
# ─────────────────────────────────────────

data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
data = data.drop(columns=['year', 'month', 'day', 'hour', 'day of week'], errors='ignore')

data['year']       = data['datetime'].dt.year
data['month']      = data['datetime'].dt.month
data['day']        = data['datetime'].dt.day
data['hour']       = data['datetime'].dt.hour
data['day of week'] = data['datetime'].dt.weekday

ignore = ['datetime', 'total', 'place', 'year', 'month', 'day', 'hour', 'day of week']
product_cols = [c for c in data.columns if c not in ignore]

data[product_cols] = data[product_cols].fillna(0)
data['place'] = data['place'].fillna('Unknown')
data = data.dropna(axis=1, how='all')

print("\n=== AFTER CLEANING ===")
print(data.info())

data.to_csv('data_bakery_clean.csv', index=False)
print("Saved -> data_bakery_clean.csv")

# ─────────────────────────────────────────
# OUTLIER ANALYSIS & REMOVAL
# Transactions after 22:00 with anomalously
# high per-transaction values are flagged.
# We compare mean daily sales with and without
# them to quantify their effect on forecasts.
# ─────────────────────────────────────────

data_clean = pd.read_csv('data_bakery_clean.csv')
data_clean['datetime'] = pd.to_datetime(data_clean['datetime'])

late_night = data_clean[data_clean['hour'] >= 22]
q75 = data_clean['total'].quantile(0.75)
iqr = data_clean['total'].quantile(0.75) - data_clean['total'].quantile(0.25)
threshold = q75 + 3 * iqr

outlier_mask = (data_clean['hour'] >= 22) & (data_clean['total'] > threshold)
n_outliers = outlier_mask.sum()
n_days_affected = data_clean.loc[outlier_mask, 'datetime'].dt.date.nunique()
total_days = data_clean['datetime'].dt.date.nunique()

print(f"\n=== OUTLIER REPORT ===")
print(f"Threshold (Q75 + 3*IQR): {threshold:.0f}")
print(f"Outlier transactions:     {n_outliers}")
print(f"Days affected:            {n_days_affected} / {total_days} ({100*n_days_affected/total_days:.1f}%)")
print(f"Mean total (with):        {data_clean['total'].mean():.1f}")
print(f"Mean total (without):     {data_clean.loc[~outlier_mask, 'total'].mean():.1f}")

# Dataset without outliers — used for SARIMA & RF
data_filtered = data_clean[~outlier_mask].copy()
data_filtered.to_csv('data_bakery_filtered.csv', index=False)

# ─────────────────────────────────────────
# TASK 1 — Exploratory: sales by hour & day
# ─────────────────────────────────────────

df = data_filtered.copy()
df = df.set_index('datetime')

sales_by_hour = df.groupby('hour')['total'].mean()
plt.figure(figsize=(8, 4))
sales_by_hour.plot(kind='bar', color='steelblue')
plt.title("Average sales by hour of day")
plt.xlabel("Hour")
plt.ylabel("Average revenue (₩)")
plt.tight_layout()
plt.savefig("sales_by_hour.png", dpi=150)
plt.show()

sales_by_dow = df.groupby('day of week')['total'].mean()
dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
plt.figure(figsize=(8, 4))
sales_by_dow.plot(kind='bar', color='steelblue')
plt.title("Average sales by day of week")
plt.xticks(range(7), dow_labels, rotation=0)
plt.xlabel("Day of week")
plt.ylabel("Average revenue (₩)")
plt.tight_layout()
plt.savefig("sales_by_dow.png", dpi=150)
plt.show()

# ─────────────────────────────────────────
# TASK 2 — SARIMA forecast (total revenue)
#
# Fix: use seasonal_order=(1,0,1,7) — only one
# level of seasonal differencing. With a short
# daily series, D=1 + d=1 causes over-differencing
# and collapses the forecast to a flat periodic wave.
# Keeping D=0 preserves the seasonal shape while
# allowing the trend to carry through.
# ─────────────────────────────────────────

from statsmodels.tsa.statespace.sarimax import SARIMAX

daily_sales = df['total'].resample('D').sum()

model_sarima = SARIMAX(daily_sales, order=(2, 1, 2), seasonal_order=(1, 0, 1, 7))
fit_sarima = model_sarima.fit(disp=False)
forecast_sarima = fit_sarima.forecast(30)

plt.figure(figsize=(10, 4))
plt.plot(daily_sales[-100:], label='History', color='steelblue')
plt.plot(forecast_sarima, label='Forecast', color='darkorange')
plt.legend()
plt.title("30-day sales forecast (SARIMA)")
plt.tight_layout()
plt.savefig("sarima_forecast.png", dpi=150)
plt.show()

# ─────────────────────────────────────────
# TASK 3 — Random Forest: predict TOTAL revenue
#
# Fix: aggregate to daily level before train/test
# split. Transaction-level data gives thousands of
# points on the x-axis and makes the forecast plot
# unreadable. Daily aggregation also matches the
# granularity of the SARIMA series.
# ─────────────────────────────────────────

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df2 = data_filtered.copy()
df2['datetime'] = pd.to_datetime(df2['datetime'])
df2['place_code'] = df2['place'].astype('category').cat.codes

# Aggregate to daily totals
numeric_cols = df2.select_dtypes(include='number').columns.tolist()
daily_rf = df2.groupby(df2['datetime'].dt.date)[numeric_cols].sum()
daily_rf.index = pd.to_datetime(daily_rf.index)

target = 'total'
ignore_cols = ['total']
feature_cols = [c for c in daily_rf.columns if c not in ignore_cols]

X = daily_rf[feature_cols]
y = daily_rf[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

r2  = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"\n=== RANDOM FOREST (total revenue, daily) ===")
print(f"R²:  {r2:.2f}")
print(f"MSE: {mse:.2f}")

# Forecast plot — daily total revenue, readable x-axis
plt.figure(figsize=(10, 4))
plt.plot(y_test.index, y_test.values, label='Actual Sales',  color='steelblue')
plt.plot(y_test.index, y_pred,        label='Predicted',     color='darkorange', linestyle='--')
plt.title("Forecast — total daily revenue (Random Forest)")
plt.xlabel("Date")
plt.ylabel("Revenue (₩)")
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.savefig("random_forest_forecast.png", dpi=150)
plt.show()

# Feature importance — total revenue
# Run across 5 seeds to confirm stability (mentioned in paper)
importances_runs = []
for seed in range(5):
    m = RandomForestRegressor(n_estimators=200, random_state=seed)
    m.fit(X_train, y_train)
    importances_runs.append(m.feature_importances_)

mean_importance = pd.Series(
    np.mean(importances_runs, axis=0), index=X.columns
).sort_values()

plt.figure(figsize=(8, 6))
mean_importance.plot(kind='barh', color='steelblue')
plt.title("Feature importance — what drives total revenue\n(mean over 5 random seeds)")
plt.xlabel("Mean impurity decrease (normalized)")
plt.tight_layout()
plt.savefig("feauture_importance1.png", dpi=150)   # typo kept to match \includegraphics in .tex
plt.show()

print("\nTop 5 features:")
print(mean_importance.sort_values(ascending=False).head(5))