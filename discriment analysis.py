import pandas as pd
#https://www.geeksforgeeks.org/ml-linear-discriminant-analysis/
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Prepare your data
# Assuming you have your data in a pandas DataFrame
# Make sure to replace 'X' with your features and 'y' with your target variable

# For example:
# X = data.drop('target_variable', axis=1)
# y = data['target_variable']

# For this example, I'll generate some random data as an illustration
import numpy as np
np.random.seed(42)
data = pd.DataFrame({
    'Feature1': np.random.rand(100),
    'Feature2': np.random.rand(100),
    'Feature3': np.random.rand(100),
    'Target': np.random.choice([0, 1], size=100)
})

X = data.drop('Target', axis=1)
y = data['Target']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Perform Discriminant Analysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Step 4: Make predictions on the test set
y_pred = lda.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
