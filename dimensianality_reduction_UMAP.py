import pandas as pd
import umap
import matplotlib.pyplot as plt
#https://www.geeksforgeeks.org/principal-component-analysis-pca/?ref=lbp

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

# Step 2: Perform UMAP
umap_model = umap.UMAP(n_components=2)  # You can adjust the number of components and other parameters as needed
X_umap = umap_model.fit_transform(X)

# Step 3: Visualize the reduced data
plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title('UMAP - Dimensionality Reduction')
plt.show()
