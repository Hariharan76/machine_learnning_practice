# # https://www.geeksforgeeks.org/how-to-perform-a-one-way-anova-in-python/
# import pandas as pd
# import statsmodels.api as sm
# from statsmodels.formula.api import ols

# # Step 1: Prepare your data
# # Assuming you have your data in a pandas DataFrame
# # Make sure to replace 'data' with your actual dataset

# # For example:
# # data = pd.read_csv('your_data.csv')

# # For this example, I'll generate some random data as an illustration
# import numpy as np
# np.random.seed(42)
# data = pd.DataFrame({
#     'Group': np.random.choice(['A', 'B', 'C'], size=100),
#     'DependentVariable': np.random.rand(100),
# })

# # Step 2: Perform one-way ANOVA
# model = ols('DependentVariable ~ Group', data=data).fit()
# anova_table = sm.stats.anova_lm(model)

# # Step 3: Print the ANOVA results
# print(anova_table)
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Step 1: Prepare your data
# Assuming you have your data in a pandas DataFrame
# Make sure to replace 'data' with your actual dataset

# For example:
# data = pd.read_csv('your_data.csv')

# For this example, I'll generate some random data as an illustration
import numpy as np
np.random.seed(42)
data = pd.DataFrame({
    'Group': np.random.choice(['A', 'B', 'C'], size=100),
    'DependentVariable': np.random.rand(100),
})

# Step 2: Perform one-way ANOVA
model = ols('DependentVariable ~ Group', data=data).fit()
anova_table = sm.stats.anova_lm(model)

# Step 3: Print the ANOVA results
print(anova_table)

