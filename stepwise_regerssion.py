import pandas as pd
import numpy as np
import statsmodels.api as sm

# Step 1: Prepare your data
# Assuming you have your data in a pandas DataFrame or numpy array
# Make sure to replace 'X' and 'y' with your predictors and target variable, respectively

# For example:
# X = data.drop('target_variable', axis=1)
# y = data['target_variable']

# For this example, I'll generate some random data as an illustration
np.random.seed(42)
data = pd.DataFrame({
    'X1': np.random.rand(100),
    'X2': np.random.rand(100),
    'X3': np.random.rand(100),
    'y': 2 + 3 * np.random.rand(100) + 0.5 * np.random.randn(100)
})

X = data.drop('y', axis=1)
y = data['y']

# Step 2: Perform stepwise regression using the AIC criterion
def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        # Forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype='float64')
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Add  {best_feature} with p-value {best_pval:.6f}')
        
        # Backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # excluding the constant term
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f'Drop {worst_feature} with p-value {worst_pval:.6f}')
        
        if not changed:
            break
    return included

# Step 3: Perform stepwise selection
selected_features = stepwise_selection(X, y)

# Step 4: Fit the final linear regression model using the selected features
final_model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[selected_features]))).fit()

# Step 5: Print the final model summary
print(final_model.summary())
