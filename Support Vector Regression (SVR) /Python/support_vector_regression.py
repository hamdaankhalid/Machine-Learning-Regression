# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
# make y into a 2d array like x to match x, because we need them in that
# array format for our feature scaling library
# reshape(rows, cols)
y = y.reshape(len(y), 1)
print(y)

# Feature Scaling
""" Here we feature scale y as well because our dependent variable takes
 on more values than just 1 or 0, it is a salary range so it must be 
 normalized as well.
  We also use feature scaling for x independent var because our feature "level"
 may be ignored because of how small it is compared to our Y """
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# diff standard scalar obj because x and y have different mean and stddev
# so to fit and transform them accordingly, we must make two objects
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
print(X)
print(y)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
# using gaussian rbf kernel
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
# we will pass predict_for (position level 6.5) as a 2d array later
predict_for = 6.5
# transform the predict_for var with sc_X object because predict_for
# is our independent var, use predict function to find independent var reg_prediction
reg_prediction = regressor.predict(sc_X.transform( [[predict_for]]) )
# inverse transform the resultant y with y scaler object
sc_y.inverse_transform(reg_prediction)

# Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()