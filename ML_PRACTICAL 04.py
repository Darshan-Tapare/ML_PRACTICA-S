import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = load_diabetes()

# Use one feature for simple visualization
X = data.data[:, [2]]
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Try different polynomial degrees
degrees = [1, 2, 3, 5, 10]

for degree in degrees:

    # Create polynomial regression model
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate errors
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)

    print("Degree:", degree)
    print("Training MSE:", round(train_error, 2))
    print("Testing MSE:", round(test_error, 2))
    print("R2 Score:", round(r2_score(y_test, y_test_pred), 2))
    print()

# Plot data
plt.scatter(X_test, y_test)

# Smooth curve for prediction
x_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

for degree in degrees:

    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    y_plot = model.predict(x_plot)

    plt.plot(x_plot, y_plot, label="Degree " + str(degree))

plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Polynomial Regression with Different Degrees")
plt.legend()
plt.show()