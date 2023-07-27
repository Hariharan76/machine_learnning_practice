import pandas as pd
from sklearn.manifold import TSNE
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

# Step 2: Perform t-SNE
tsne = TSNE(n_components=2)  # You can adjust the perplexity and other parameters as needed
X_tsne = tsne.fit_transform(X)

# Step 3: Visualize the reduced data
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE - Dimensionality Reduction')
plt.show()
