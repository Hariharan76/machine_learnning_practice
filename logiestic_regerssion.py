import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data (replace this with your own dataset)
# X should be a 2D array where each row represents a data point, and each column represents a feature.
# y should be a 1D array representing the target labels (binary classes: 0 or 1).
X = np.array([[2.3, 3.4], [1.2, 5.0], [4.1, 2.5], [3.7, 1.8], [2.8, 3.9]])
y = np.array([0, 1, 0, 1, 0])

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# You can also access the model's coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
