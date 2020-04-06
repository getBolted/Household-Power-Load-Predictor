import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
import pickle

def code_mean(data, cat_feature, real_feature):
	"""
	Возвращает словарь, где ключами являются уникальные категории признака cat_feature, 
	а значениями - средние по real_feature
	"""
	return dict(data.groupby(cat_feature)[real_feature].mean())


split_date = '2015-04-29 23:50'

name = 'E:/КурсачИИ/week_top500.csv'
X = pd.read_csv(name)
y = pd.read_csv(r'E:/КурсачИИ/target.csv')

y = y.set_index('Time')
y.index = y.index.astype('datetime64[ns]')
X = X.set_index('Time')
X.index = X.index.astype('datetime64[ns]')

data = X.join(y)

result=list(set(['weekday','hour', 'day_part']) & set(list(data.columns)))
for j in result:
	newName = j + '_average'
	data[newName] = list(map(code_mean(data[data.index <= split_date], j, "Aggregate").get, data[j]))
	data.drop(j, axis = 1, inplace = True)

data_train = data.loc[data.index <= split_date].copy()
data_test = data.loc[data.index > split_date].copy()

X_train = data_train.drop('Aggregate', axis = 1)
y_train = data_train['Aggregate']
X_test = data_test.drop('Aggregate', axis = 1)
y_test = data_test['Aggregate']

# param_test1= {
#  'reg_alpha':[45e-7, 46e-7, 47e-7,48e-7, 49e-7, 50e-7,51e-7, 52e-7, 53e-7,54e-7]
# }
# gsearch = GridSearchCV(estimator =
#                       XGBRegressor(
# # tree_method='gpu_hist', 
#  #gpu_id=0,
#  max_depth = 10, 
#  gamma=0,
#  subsample = 0.6606, 
#  colsample_bytree = 0.859,
#  min_child_weight = 2,                         
#  learning_rate = 0.1,
#  n_estimators=120,
#  objective= 'reg:gamma',
#  scale_pos_weight=1,
#  seed=17, 
#  verbocity = 2), param_grid=param_test1, scoring='neg_mean_absolute_error', verbose=4, n_jobs=-1)
# gsearch.fit(X_test, y_test)


# print(gsearch.best_params_ , gsearch.best_score_)
reg = XGBRegressor(
 # tree_method='gpu_hist', 
 # gpu_id=0,
 max_depth = 10, 
 gamma=0,
 subsample = 0.6606, 
 colsample_bytree = 0.859,
 min_child_weight = 2,                         
 objective= 'reg:gamma',
 scale_pos_weight=1,
 reg_alpha = 5e-6,
 seed=17, 
 learning_rate = 0.1, #0.001
 n_estimators=300,  #30000
 verbocity = 2)

reg.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_test, y_test)],early_stopping_rounds=50,verbose=True)

data_test['Predict'] = reg.predict(X_test)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print('MAE is - ', mean_absolute_error(y_true=data_test['Aggregate'],
                   y_pred=data_test['Predict']), 'W')
print('MAPE is - ', mean_absolute_percentage_error(y_true=data_test['Aggregate'],
                   y_pred=data_test['Predict']), ' %')

pickle.dump(reg, open("500-300-32bit.pickle.dat", "wb"))
reg.save_model('500-300-32bit.model')