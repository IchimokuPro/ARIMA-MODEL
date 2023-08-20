import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load time series data with full path (replace with your data path)
data = pd.read_csv('C:/Users/Priya/Documents/PythonLab/time_series_data.csv', parse_dates=['Date'], dayfirst=True)
data.set_index('Date', inplace=True)
ts = data['Value']

# Plot the time series data
plt.figure(figsize=(10, 6))
plt.plot(ts)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series Data')
plt.show()

# Plot ACF and PACF to determine p and q values
plot_acf(ts, lags=30)
plot_pacf(ts, lags=30)
plt.show()

# ARIMA model parameters (replace p, d, and q with appropriate values)
p = 1
d = 1
q = 1

# Create ARIMA model
model = ARIMA(ts, order=(p, d, q))
model_fit = model.fit()

# Get model summary
print(model_fit.summary())

# Forecast next steps (adjust the steps parameter as needed)
forecast_steps = 10
forecast, stderr, conf_int = model_fit.forecast(steps=forecast_steps)

# Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(ts, label='Original Data')
plt.plot(forecast.index, forecast, label='Forecasted Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Forecasted Time Series Data')
plt.legend()
plt.show()
