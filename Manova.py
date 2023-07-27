# import pandas as pd
# import statsmodels.api as sm
# from statsmodels.multivariate.manova import MANOVA
#https://www.reneshbedre.com/blog/manova-python.html

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
#     'Var1': np.random.rand(100),
#     'Var2': np.random.rand(100),
#     'Var3': np.random.rand(100),
# })

# # Step 2: Perform MANOVA
# dependent_vars = ['Var1', 'Var2', 'Var3']
# independent_var = 'Group'
# maov = MANOVA.from_formula(f"{', '.join(dependent_vars)} ~ {independent_var}", data=data)
# result = maov.mv_test()

# # Step 3: Print the MANOVA results
# print(result)

# # Additional: If you want to obtain univariate ANOVA results for each dependent variable
# univariate_anova = sm.stats.anova_lm(maov.mv_test())
# print(univariate_anova)

import pandas as pd
import pingouin as pg

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
    'Var1': np.random.rand(100),
    'Var2': np.random.rand(100),
    'Var3': np.random.rand(100),
})

# Step 2: Perform MANOVA
dependent_vars = ['Var1', 'Var2', 'Var3']
independent_var = 'Group'
result = pg.multivariate_anova(data=data, dv=dependent_vars, between=independent_var)

# Step 3: Print the MANOVA results
print(result)
