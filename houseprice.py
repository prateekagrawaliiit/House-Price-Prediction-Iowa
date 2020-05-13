# Dependencies

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('train.csv')
# print(df.head())

# print(df.columns)
y = df.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF',
            '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = df[features]

# print(X.describe)

iowa_model = DecisionTreeRegressor(random_state=2)
iowa_model.fit(X, y)

predictions = iowa_model.predict(X)
# print(predictions)

# print(df.head().SalePrice)
# print(predictions[:5])


# The model prints the same sale price as that of the real and the data that we are using to test has already been used by the model to
# learn and predict hence it already knows the value of the house with the given parameters .


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
new_model = DecisionTreeRegressor(random_state=1)
new_model.fit(X_train, y_train)
new_predicitions = new_model.predict(X_test)
# print(new_predicitions[:5])
# print(y_test[:5])


mean_error = mean_absolute_error(y_test, new_predicitions)
# print("The mean absolute error for the test sample is ", mean_error)


def get_mae(nodes, X_train, y_train, X_test, y_test):
    model = DecisionTreeRegressor(max_leaf_nodes=nodes, random_state=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(predictions, y_test)
    return mae


scores = {}

for nodes in [10, 50, 100, 200, 300, 400, 500, 1000, 5000, 10000]:
    mae1 = get_mae(nodes, X_train, y_train, X_test, y_test)
    # print(mae1)
    scores[nodes] = mae1

key_min = min(scores, key=(lambda k: scores[k]))

best_tree_size = key_min
# print(best_tree_size)

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size,random_state=1)
final_model.fit(X_train,y_train)
predi = final_model.predict(X_test)

mean_error = mean_absolute_error(y_test, predi)
print("The mean absolute error for the test sample is ", mean_error)
