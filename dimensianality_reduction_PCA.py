import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Prepare your data
# Assuming you have your data in a pandas DataFrame or numpy array
# Make sure to replace 'X' with your dataset

# For example:
# X = pd.read_csv('your_data.csv')

# For this example, I'll generate some random data as an illustration
import numpy as np
np.random.seed(42)
X = pd.DataFrame({
    'Feature1': np.random.rand(100),
    'Feature2': np.random.rand(100),
    'Feature3': np.random.rand(100),
})

# Step 2: Perform PCA
pca = PCA(n_components=2)  # You can adjust the number of components as needed
X_pca = pca.fit_transform(X)

# Step 3: Visualize the reduced data
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - Dimensionality Reduction')
plt.show()
