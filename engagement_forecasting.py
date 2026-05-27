from pymongo import MongoClient
import pandas as pd

# ==============================
# DATABASE CONNECTION
# ==============================
client = MongoClient("mongodb://localhost:27017/")
db = client["mall_analytics"]
col = db["engagement_logs"]

print("✓ Connected to MongoDB")

# ==============================
# LOAD COMPLETED SESSIONS
# ==============================
docs = list(col.find(
    {"active": False},
    {
        "_id": 0,
        "first_seen": 1,
        "engaged": 1
    }
))

if not docs:
    raise RuntimeError("No data available for forecasting")

df = pd.DataFrame(docs)

#Timezone handling
df["ts_utc"] = pd.to_datetime(df["first_seen"], utc=True)
df["ts_local"] = df["ts_utc"].dt.tz_convert("Asia/Colombo")

#Create time-series signal
# Keep only engaged sessions
df["engaged_flag"] = df["engaged"].astype(int)

# Hourly aggregation
hourly_series = (
    df.set_index("ts_local")
        .resample("h")["engaged_flag"]
      .sum()
)

print("\n=== HISTORICAL HOURLY ENGAGEMENT ===")
print(hourly_series.tail())

#Forecast using ARIMA
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(hourly_series, order=(1, 1, 1))
model_fit = model.fit()

# Forecast next 6 hours
forecast_steps = 6
forecast = model_fit.forecast(steps=forecast_steps)

print("\n🔮 ENGAGEMENT FORECAST (NEXT HOURS)")
print(forecast)

#Visualize prediction
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(hourly_series[-24:], label="Historical")
plt.plot(forecast, label="Forecast", linestyle="--")
plt.title("Hourly Engagement Forecast")
plt.xlabel("Time")
plt.ylabel("Engaged People Count")
plt.legend()
plt.grid()
plt.show()
