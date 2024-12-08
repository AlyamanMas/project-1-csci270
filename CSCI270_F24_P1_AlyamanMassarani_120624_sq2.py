import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# parameters (can be used to customize the program):
min_lin = 0
max_lin = 10
num_samples = 500
noise_average = 0
noise_std = 1.5
np.random.seed(42)

X = np.linspace(min_lin, max_lin, num_samples).reshape(-1, 1)
# noise following random distribution
noise = np.random.normal(noise_average, noise_std, num_samples).reshape(-1, 1)
y = 3 * X + 7 + noise

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the linear regression model on the train data
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# start the plotting process
plt.figure(figsize=(10, 6))
# plot training and testing data points
plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training Data')
plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Testing Data')
# plot the regression line
X_line = np.linspace(min_lin, max_lin, num_samples).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color='red', label='Regression Line')

# add labels and title
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression on Synthetic Dataset\n' +
          f'y = {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f}\n' +
          f'MSE = {mse:.2f}')
plt.legend()
plt.grid(True, alpha=0.3)

# print model parameters
print(f"Coefficient (slope): {model.coef_[0][0]:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

plt.show()