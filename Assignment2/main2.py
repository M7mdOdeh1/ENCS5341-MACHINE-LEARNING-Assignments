import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

# Load data
data = pd.read_csv('data_reg.csv')

# Split the data
train_data = data.iloc[:120]
validation_data = data.iloc[120:160]
test_data = data.iloc[160:]

# Preparing the data
X_train = train_data[['x1', 'x2']]
y_train = train_data['y']
X_val = validation_data[['x1', 'x2']]
y_val = validation_data['y']

# Range of polynomial degrees to consider
degrees = range(1, 11)

# Lists to store validation errors and models
validation_errors = []
models = []

# Perform polynomial regression for each degree
for degree in degrees:
    # Create polynomial features
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    # Fit the model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    models.append(model)

    # Predict and calculate validation error
    y_val_pred = model.predict(X_val_poly)
    error = mean_squared_error(y_val, y_val_pred)
    validation_errors.append(error)

# Plot validation error vs polynomial degree
plt.figure()
plt.plot(degrees, validation_errors, marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('Validation Error')
plt.title('Validation Error vs. Polynomial Degree')
plt.show()

# Plotting the surfaces of the learned functions
for i, degree in enumerate(degrees):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_data['x1'], train_data['x2'], train_data['y'], c='r', marker='o', label='Training Set')

    # Creating a meshgrid for plotting
    x1_range = np.linspace(X_train['x1'].min(), X_train['x1'].max(), 100)
    x2_range = np.linspace(X_train['x2'].min(), X_train['x2'].max(), 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    X_grid = np.c_[X1.ravel(), X2.ravel()]
    Y_grid = models[i].predict(PolynomialFeatures(degree).fit_transform(X_grid)).reshape(X1.shape)

    # Plotting the surface
    ax.plot_surface(X1, X2, Y_grid, alpha=0.3, color='blue')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title(f'Polynomial Degree: {degree}')
    plt.show()