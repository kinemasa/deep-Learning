import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 ##import dataset
dataset = pd.read_csv('../data/Position_Salaries1.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
##n_estimators  = number of tree
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X, y)
##predict
regressor.predict([[6.5]])

## show result
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()