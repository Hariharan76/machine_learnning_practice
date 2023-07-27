import pandas as pd
from sklearn.cross_decomposition import CCA
#https://cmdlinetips.com/2020/12/canonical-correlation-analysis-in-python/

# Step 1: Prepare your data
# Assuming you have your data in two pandas DataFrames or numpy arrays, X1 and X2
# Make sure that both X1 and X2 have the same number of samples

# For example:
# X1 = pd.read_csv('data_set1.csv')
# X2 = pd.read_csv('data_set2.csv')

# For this example, I'll generate some random data as an illustration
import numpy as np
np.random.seed(42)
X1 = pd.DataFrame({
    'Var1': np.random.rand(100),
    'Var2': np.random.rand(100),
    'Var3': np.random.rand(100),
})

X2 = pd.DataFrame({
    'Var4': np.random.rand(100),
    'Var5': np.random.rand(100),
    'Var6': np.random.rand(100),
})

# Step 2: Perform Canonical Analysis
cca = CCA(n_components=2)  # You can set the number of canonical components as needed
cca.fit(X1, X2)

# Step 3: Transform the original datasets into canonical variates
X1_canonical, X2_canonical = cca.transform(X1, X2)

# Step 4: Access the canonical correlation coefficients
canonical_correlations = cca.corr_

# Step 5: Print the canonical correlation coefficients
print("Canonical Correlation Coefficients:")
print(canonical_correlations)

# Step 6: Visualize the canonical variates (first two components) if you used n_components=2
import matplotlib.pyplot as plt

plt.scatter(X1_canonical[:, 0], X1_canonical[:, 1], label='Set 1')
plt.scatter(X2_canonical[:, 0], X2_canonical[:, 1], label='Set 2')
plt.xlabel('Canonical Variate 1')
plt.ylabel('Canonical Variate 2')
plt.title('Canonical Analysis')
plt.legend()
plt.show()
