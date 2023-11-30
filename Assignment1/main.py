import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


print('----------------------------------------------------------------------')
############################## part 1 ########################################

path = 'Assignment1/cars.csv'
# check if the file path exists and exit if it doesn't
try:
    with open(path) as f:
        pass
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()

# Load the dataset into a pandas dataframe
cars_data = pd.read_csv(path)

# Examine the dataset for features (columns) and examples (rows)
num_features = cars_data.shape[1]
num_examples = cars_data.shape[0]

print("Number of features: ", num_features)
print("Number of examples: ", num_examples)


print('----------------------------------------------------------------------')
############################## part 2 ########################################

# Check for missing values in each feature
temp = cars_data.isnull().sum()
missing_values = temp[temp > 0]
if missing_values.empty:
    print("No, there are no missing values in the dataset.")
else:
    print("Yes, there are missing values in the following features:")
    print(missing_values.to_string())

print('----------------------------------------------------------------------')

############################## part 3 ########################################

for column in cars_data.columns:
    if pd.api.types.is_numeric_dtype(cars_data[column]):
        # Filling missing values for numeric columns with median
        median_value = cars_data[column].median()
        cars_data[column].fillna(median_value, inplace=True)
    else:
        # Filling missing values for non-numeric column 'origin' with mode
        mode_value = cars_data[column].mode()[0]
        cars_data[column].fillna(mode_value, inplace=True)

# Check if any missing values remain
print("Number of missing values after filling:")
print(cars_data.isnull().sum().to_string())


print('----------------------------------------------------------------------')

############################## part 4 ########################################
best_fuel_economy_country = cars_data.groupby('origin')['mpg'].mean().idxmax()
print("Country with the best fuel economy: ", best_fuel_economy_country)

plt.figure(figsize=(11, 7))
sns.boxplot(x='origin', y='mpg', data=cars_data)

plt.title('Fuel Economy (MPG) by Country of Origin')
plt.xlabel('Origin')
plt.ylabel('MPG')
plt.show()



print('----------------------------------------------------------------------')

############################## part 5 ########################################

# Plotting histograms for 'acceleration', 'horsepower', and 'mpg' to compare their distributions
plt.figure(figsize=(15, 5))

# Histogram for 'acceleration'
plt.subplot(1, 3, 1)
sns.histplot(cars_data['acceleration'], kde=True)
plt.title('Acceleration Distribution')

# Histogram for 'horsepower'
plt.subplot(1, 3, 2)
sns.histplot(cars_data['horsepower'], kde=True)
plt.title('Horsepower Distribution')

# Histogram for 'mpg'
plt.subplot(1, 3, 3)
sns.histplot(cars_data['mpg'], kde=True)
plt.title('MPG Distribution')

plt.tight_layout()
plt.show()

print("The distributions of 'acceleration' are similar to Gaussian distribution.")

print('----------------------------------------------------------------------')

############################## part 6 ########################################

def median_skewness(data):
    median = data.median()
    mean = data.mean()
    std_dev = data.std()
    skewness = 3 * (mean - median) / std_dev
    return skewness

skewness_acceleration = median_skewness(cars_data['acceleration'])
skewness_horsepower = median_skewness(cars_data['horsepower'])
skewness_mpg = median_skewness(cars_data['mpg'])

print("Skewness of 'acceleration': ", skewness_acceleration)
print("Skewness of 'horsepower': ", skewness_horsepower)
print("Skewness of 'mpg': ", skewness_mpg)

"""
Interpretation:
Acceleration: A skewness close to 0 indicates a distribution close to symmetric. Acceleration's distribution is the closest to symmetric among the three.
Horsepower and MPG: Both have higher skewness values, indicating a more pronounced deviation from symmetry. Horsepower, in particular, shows a substantial skew.
Conclusion:
Based on the Pearson's second skewness coefficient, 'acceleration' has a distribution most similar to a Gaussian (normal) distribution, as it exhibits the least skewness among the three features. ​
"""
print('----------------------------------------------------------------------')

############################## part 7 ########################################

# Creating a scatter plot to show the relationship between 'horsepower' and 'mpg'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='horsepower', y='mpg', data=cars_data)

plt.title('Scatter Plot of Horsepower vs MPG')
plt.xlabel('Horsepower')
plt.ylabel('Miles Per Gallon (MPG)')
plt.show()

"""
Observation:
There appears to be a negative correlation between horsepower and mpg. This means as the horsepower increases, the mpg tends to decrease, and vice versa.
Interpretation:
Vehicles with higher horsepower typically consume more fuel, resulting in lower mpg. Conversely, vehicles with lower horsepower are generally more fuel-efficient, reflected in higher mpg values.
"""

print('----------------------------------------------------------------------')

############################## part 8 ########################################

# Preparing the data for linear regression
X = cars_data['horsepower'].values.reshape(-1, 1)  # Feature
X = np.hstack((np.ones((X.shape[0], 1)), X))  # Adding x0 = 1 for the intercept
y = cars_data['mpg'].values.reshape(-1, 1)  # Target variable

# Closed form solution of linear regression: (X'X)^-1 X'y
XT = X.T
w = np.linalg.inv(XT.dot(X)) @ XT @ y  # w (parameters)

# Predicting 'mpg' from 'horsepower' using the learned model
y_pred = X @ w

# Plotting the scatter plot and the learned linear regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(x='horsepower', y='mpg', data=cars_data, label='Data points')
plt.plot(cars_data['horsepower'], y_pred, color='red', label='Regression Line')

plt.title('Horsepower vs MPG with Linear Regression Line')
plt.xlabel('Horsepower')
plt.ylabel('Miles Per Gallon (MPG)')
plt.legend()
plt.show()

print('----------------------------------------------------------------------')

############################## part 9 ########################################

# Generating horsepower values for the quadratic curve
horsepower_range = np.linspace(cars_data['horsepower'].min(), cars_data['horsepower'].max(), 500)
horsepower_range_reshaped = horsepower_range.reshape(-1, 1)
# Creates a matrix for the quadratic model with three columns: [1, horsepower, horsepower^2]
z_range = np.hstack((np.ones(horsepower_range_reshaped.shape), horsepower_range_reshaped, horsepower_range_reshaped ** 2))

# Create z matrix from X matrix for the quadratic model with three columns: [1, horsepower, horsepower^2]
z = np.hstack((np.ones((X.shape[0], 1)), X[:, 1].reshape(-1, 1), (X[:, 1] ** 2).reshape(-1, 1)))

# Calculating w_quad for the quadratic model
zT= z.T
w_quadratic = np.linalg.inv(zT.dot(z)) @ zT @ y

# Predicting 'mpg' for the generated horsepower range using the quadratic model
ypred_quadratic = z_range @ w_quadratic

# Plotting the scatter plot with the quadratic regression curve
plt.figure(figsize=(10, 6))
plt.scatter(cars_data['horsepower'], cars_data['mpg'], label='Data points')
plt.plot(horsepower_range, ypred_quadratic, color='green', label='Quadratic Regression')

plt.title('Horsepower vs MPG with Quadratic Regression Curve')
plt.xlabel('Horsepower')
plt.ylabel('Miles Per Gallon (MPG)')
plt.legend()
plt.show()

print('----------------------------------------------------------------------')

############################## part 10 ########################################


# Function to perform gradient descent
def gradient_descent(X, y, w, learning_rate, num_iterations):
    m = len(y)

    for i in range(num_iterations):
        predictions = X.dot(w)
        w -= (1/m) * learning_rate * X.T.dot(predictions - y)
    return w


# Extracting features and target variable
X = cars_data['horsepower'].values.reshape(-1, 1)
y = cars_data['mpg'].values.reshape(-1, 1)

# Normalize the 'horsepower' feature using z-score normalization
mean_horsepower = np.mean(X)
std_horsepower = np.std(X)
X = (X - mean_horsepower) / std_horsepower

# Add intercept term to feature matrix
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Initializing parameters
w = np.zeros((2, 1))  # 2 parameters (intercept and slope)
learning_rate = 0.01  # Adjust the learning rate
num_iterations = 1000  # Adjust the number of iterations


# Performing gradient descent
w = gradient_descent(X, y, w, learning_rate, num_iterations)

# Print the final w values
print("w after Gradient Descent:", w)


# Generate predictions using the learned model
predictions = X.dot(w)

# Plot the data and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 1], y, label='Data Points')  # Original data
plt.plot(X[:, 1], predictions, color='red', label='Regression Line')  # Regression line
plt.title('Horsepower vs MPG')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.legend()
plt.show()




