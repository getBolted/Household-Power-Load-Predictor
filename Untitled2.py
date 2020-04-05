import xgboost as xgb 
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

X = pd.read_csv(r'MayX.csv')
X = X.set_index('Time')
X.index = X.index.astype('datetime64[ns]')

y = pd.read_csv(r'MayY.csv')
y = y.set_index('Time')
y.index = y.index.astype('datetime64[ns]')

y['Predict'] = -1
y = y.filter(['Aggregate', 'Predict'])

loaded_model = pickle.load(open("pima.pickle.dat", "rb"))

predict = loaded_model.predict(X)

y['Predict'] = predict

print('MAE is - ', mean_absolute_error(y_true=y['Aggregate'],
                   y_pred=y['Predict']), 'W')
print('MAPE is - ', mean_absolute_percentage_error(y_true=y['Aggregate'],
                   y_pred=y['Predict']), ' %')


