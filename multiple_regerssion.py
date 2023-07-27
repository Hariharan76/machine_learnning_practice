import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Create the data
data = {
    'x1': [1, 2, 3, 4, 5],
    'x2': [1, 2, 3, 4, 5],
    'y': [6, 8, 10, 12, 14]
}
df = pd.DataFrame(data)

# Create the model
model = LinearRegression()

# Fit the model
model.fit(df[['x1', 'x2']], df['y'])

# Evaluate the model
print('R-squared:', model.score(df[['x1', 'x2']], df['y']))
print('MSE:', np.mean((model.predict(df[['x1', 'x2']]) - df['y'])**2))
print('RMSE:', np.sqrt(np.mean((model.predict(df[['x1', 'x2']]) - df['y'])**2)))
print('MAE:', np.mean(np.abs(model.predict(df[['x1', 'x2']]) - df['y'])))
