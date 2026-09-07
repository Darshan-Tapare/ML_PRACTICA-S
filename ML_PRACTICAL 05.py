import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Ridge Regression (L2)
ridge = LogisticRegression(
    penalty='l2',
    C=1.0,
    max_iter=5000
)

ridge.fit(X_train, y_train)

ridge_pred = ridge.predict(X_test)

print("Ridge Accuracy:",
      accuracy_score(y_test, ridge_pred))


# 5. Lasso Regression (L1)
lasso = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    C=1.0,
    max_iter=5000
)

lasso.fit(X_train, y_train)

lasso_pred = lasso.predict(X_test)

print("Lasso Accuracy:",
      accuracy_score(y_test, lasso_pred))


# 6. Compare coefficients
print("\nRidge coefficients:")
print(ridge.coef_)

print("\nLasso coefficients:")
print(lasso.coef_)