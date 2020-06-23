#The Boston Housing dataset was obtained from the mlbench R package, which was loaded using the following commands:
library(mlbench)
data(BostonHousing)

#Dataframe
import pandas as pd

#The last column is the Y variable, all other columns are X variables
BostonHousing = pd.read_csv("BostonHousing.csv")
BostonHousing









#Data Split into X^Y variables
Y = BostonHousing.medv
Y

#Axis=1 combines dataframes 
X = BostonHousing.drop(['medv'], axis=1)
X

#Now we split the data by importing necessary library
from sklearn.model_selection import train_test_split

#Assign 80/20 split to 4 variablkes
#80 to train, 20 to test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#Dimensions
X_train.shape, Y_train.shape
X_test.shape, Y_test.shape









#Library import necessary for model creation
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#Define Linear Regression Model
model = linear_model.LinearRegression()

#Define input for variables
model.fit(X_train, Y_train)

#Applying trained model makes a prediction
Y_pred = model.predict(X_test)

#Prediction results
#Model Performance (utilizing %.f to allow for string formatting of 2 or 3 decimal places)
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred))








#Necessary import for Scatter Plot
import seaborn as sns

#Data formatting
Y_test
import numpy as np
np.array(Y_test)
Y_pred

#Scatter Plot
sns.scatterplot(Y_test, Y_pred)

#Marker modification
sns.scatterplot(Y_test, Y_pred, marker="+")

#Alpha change from 1 -> 0.5 allows for more translucent data points
sns.scatterplot(Y_test, Y_pred, alpha=0.5)
