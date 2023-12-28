# Import necessary libraries for this section
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import Ridge

# Load the data from the CSV file
file_path = 'data_reg.csv'
data = pd.read_csv(file_path)

# Split the dataset into training, validation, and testing sets
train, remain = train_test_split(data, train_size=120, shuffle=False)
validation, test = train_test_split(remain, train_size=40, shuffle=False)

# Plot the data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for training data
ax.scatter(train['x1'], train['x2'], train['y'], color='r', label='Training set')

# Scatter plot for validation data
ax.scatter(validation['x1'], validation['x2'], validation['y'], color='g', label='Validation set')

# Scatter plot for testing data
ax.scatter(test['x1'], test['x2'], test['y'], color='b', label='Testing set')

# Adding labels for axises
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.legend()

# Show plot
plt.show()


# Apply polynomial regression for each degree from 1 to 10
degrees = range(1, 11)
validation_errors = []
models = []

X_train, y_train = train[['x1', 'x2']], train['y']
X_validation, y_validation = validation[['x1', 'x2']], validation['y']

for degree in degrees:
    # Create polynomial features
    poly = PolynomialFeatures(degree)
    X_train_polynomial = poly.fit_transform(X_train)
    X_validation_polynomial = poly.transform(X_validation)

    # Train the model
    model = LinearRegression()
    model.fit(X_train_polynomial, y_train)
    models.append(model)

    # Predict on the validation set
    y_validation_pred = model.predict(X_validation_polynomial)

    # Calculate and store the validation error
    mse = mean_squared_error(y_validation, y_validation_pred)
    validation_errors.append(mse)

# Plot the validation error vs polynomial degree curve
plt.figure(figsize=(10, 6))
plt.plot(degrees, validation_errors, marker='o')
plt.title('Validation Error vs Polynomial Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.xticks(degrees)
plt.grid(True)
plt.show()
