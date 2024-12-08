import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# program parameters
np.random.seed(42)
n_points = 500 # 500 days
linear_slope = 0.05
seasonal_scale = 5
noise_weight = 1

# create time points
t = np.arange(n_points)

# create components
linear_trend = linear_slope * t
seasonal = seasonal_scale * np.sin(2 * np.pi * t / 365)  # Yearly seasonality
noise = noise_weight * np.random.normal(0, 1, n_points)

# combine components
y = linear_trend + seasonal + noise

# train our linear regression model
X = t.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# make plots
plt.figure(figsize=(15, 8))
plt.plot(t, y, 'b.', alpha=0.5, label='Original Data')
plt.plot(t, y_pred, 'r-', label=f'Trend Line (slope={model.coef_[0]:.4f})')
plt.xlabel('Time (days)')
plt.ylabel('Value')
plt.title('Synthetic Time Series with Trend, Seasonality, and Noise')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()