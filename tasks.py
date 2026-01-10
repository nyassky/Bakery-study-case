import pandas as pd
import matplotlib.pyplot as plt
# CSV upload
data = pd.read_csv('data_bakery.csv')

# DATA BEFORE CLEANING

print(data.info())
print(data.head())
print('-----------------------------------------------------------------------------------------------------------')
print(data.tail())

# DATA CLEANING 

data['datetime'] = pd.to_datetime(data['datetime'], errors="coerce")
data = data.drop(columns=['year', 'month', 'day', 'hour', 'day of week'], errors='ignore')

data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.day
data['hour'] = data['datetime'].dt.hour
data['day of week'] = data['datetime'].dt.weekday

ignore = ['datetime','total','place','year','month','day','hour','day of week']
product_cols = [c for c in data.columns if c not in ignore]

data[product_cols] = data[product_cols].fillna(0)
data['place'] = data['place'].fillna('Unknown')

data = data.dropna(axis=1, how='all')

# DATA AFTER CLEANING

print(data.info())
print(data.head())
print('-----------------------------------------------------------------------------------------------------------')
print(data.tail())

data.to_csv('data_bakery_clean.csv', index=False)
print("File is successfully saved -> data_bakery_clean.csv")

# TASK 1

data = pd.read_csv('data_bakery_clean.csv')

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

sales_by_hour = data.groupby('hour')['total'].mean()

plt.figure(figsize=(8,4))
sales_by_hour.plot(kind='bar')
plt.title("Average sales by hour")
plt.xlabel("Hour")
plt.ylabel("Average income (₩)")
plt.show()

sales_by_dow = data.groupby('day of week')['total'].mean()

plt.figure(figsize=(8,4))
sales_by_dow.plot(kind='bar')
plt.title("Average sales by day of the week")
plt.xlabel("Day (0=Mon, 6=Sun)")
plt.ylabel("Average income")
plt.show()

from statsmodels.tsa.arima.model import ARIMA

daily_sales = data['total'].resample('D').sum()

model = ARIMA(daily_sales, order=(3,1,2))
model_fit = model.fit()
forecast = model_fit.forecast(30)

plt.figure(figsize=(10,4))
plt.plot(daily_sales[-100:], label="History")
plt.plot(forecast, label="Forecast")
plt.legend()
plt.title("30-day sales forecast (ARIMA)")
plt.show()

from statsmodels.tsa.statespace.sarimax import SARIMAX

model2 = SARIMAX(daily_sales, order=(2,1,2), seasonal_order=(1,1,1,7))
model_fit2 = model2.fit()
forecast2 = model_fit2.forecast(30)

plt.figure(figsize=(10,4))
plt.plot(daily_sales[-100:], label="History")
plt.plot(forecast2, label="Forecast")
plt.legend()
plt.title("30-day sales forecast (SARIMA)")
plt.show()

# TASK 3

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load clean data
data_3 = pd.read_csv("data_bakery_clean.csv")
data_3["datetime"] = pd.to_datetime(data_3["datetime"])
data_3 = data_3.set_index("datetime")

# Remove non-numeric column
data_3 = data_3.drop(columns=["place"])

# Resample to daily totals
daily_3 = data_3.resample("D").sum()

# Select product target
target = "croissant"
daily_3[target] = daily_3[target].fillna(0)

# Input features (all except target)
X = daily_3.drop(columns=[target])
y = daily_3[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Random Forest Results:")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2:  {r2_score(y_test, y_pred):.2f}")

# Plot results
plt.figure(figsize=(10,4))
plt.plot(y_test.index, y_test, label="Actual Sales")
plt.plot(y_test.index, y_pred, label="Predicted", linestyle='--')
plt.title(f"Forecast — {target} sales (Random Forest)")
plt.legend()
plt.show()

# Feature importance
importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values().plot(kind="barh", figsize=(6,5))
plt.title("Feature Importance — Which variables affect Croissant sales")
plt.show()


# TASK 5

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load clean dataset
data_5 = pd.read_csv("data_bakery_clean.csv")
data_5["datetime"] = pd.to_datetime(data_5["datetime"])

# Select features/target
target = "total"
ignore = ['datetime', 'total', 'place']
product_cols = [c for c in data_5.columns if c not in ignore]

# Convert categorical columns
data_5["place"] = data_5["place"].astype("category").cat.codes

# Features
X = data_5[product_cols + ['hour', 'day of week', 'place']]
y = data_5[target]

# Train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Predict
pred = rf.predict(X_test)
print("R²:", r2_score(y_test, pred))
print("MSE:", mean_squared_error(y_test, pred))

# Feature importance
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8,5))
importance.plot(kind='bar')
plt.title("Feature Importance – What affects total sales most?")
plt.show()
