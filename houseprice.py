#Dependencies

import pandas as pd 
import numpy as np 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('train.csv')
# print(df.head())

# print(df.columns)
y = df.SalePrice
features = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = df[features]

# print(X.describe)

iowa_model = DecisionTreeRegressor(random_state=2)
iowa_model.fit(X,y)

predictions = iowa_model.predict(X)
# print(predictions)

# print(df.head().SalePrice)
# print(predictions[:5])


#The model prints the same sale price as that of the real and the data that we are using to test has already been used by the model to 
# learn and predict hence it already knows the value of the house with the given parameters .


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state =1)
new_model = DecisionTreeRegressor(random_state=1)
new_model.fit(X_train,y_train)
new_predicitions = new_model.predict(X_test)
# print(new_predicitions[:5])
# print(y_test[:5])


mean_error = mean_absolute_error(y_test,new_predicitions)
print("The mean absolute error for the test sample is ",mean_error)
