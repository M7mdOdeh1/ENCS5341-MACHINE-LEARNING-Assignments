# Normalize the 'horsepower' feature using z-score normalization
mean_horsepower = np.mean(X)
std_horsepower = np.std(X)
X = (X - mean_horsepower) / std_horsepower