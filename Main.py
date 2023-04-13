import pandas as pd
df = pd.read_csv('YOUR PATH\optionschain.csv')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
import time
from collections import Counter

def average_percentage_error(a,b):
  average_error = 0
  predictions = np.array(a)
  ytest = np.array(b)
  for g in range(len(predictions)):
    if (ytest[g] < 2):
      continue
    err = predictions[g] - ytest[g]
    err = (abs(predictions[g]-ytest[g])/((predictions[g]+ytest[g])/2))*100
    average_error += err
  average_error = average_error/(len(predictions))
  return(average_error)

df.drop("symbol", axis = 1, inplace=True)
df.drop("contractSymbol", axis = 1, inplace=True)
df.drop("currency", axis = 1, inplace=True)
df.drop("contractSize", axis=1, inplace=True)
df.drop('lastTradeDate', axis=1, inplace=True)


le = LabelEncoder()

optionType_categorical = df['optionType']
le.fit(optionType_categorical)
optionType_numerical = le.transform(optionType_categorical)

inTheMoney_categorical = df['inTheMoney']
le.fit(inTheMoney_categorical)
inTheMoney_numerical = le.transform(inTheMoney_categorical)

#symbol_categorical = df["symbol"]
#le.fit(symbol_categorical)
#symbol_num = le.transform(symbol_categorical)


df.drop("inTheMoney", axis=1, inplace=True)
df.drop('optionType', axis=1, inplace=True )
##df.drop("symbol",axis = 1, inplace=True)
df['inTheMoney'] = inTheMoney_numerical
df['optionType'] = optionType_numerical
##df['symbol'] = symbol_num
##0 = call, 1 = put.
##1 = in the money, 0  = out of the money

expiration_list = df['expiration'].values.tolist()
new_date = list()
for expire_date in expiration_list:
  date_change = datetime.strptime(expire_date, '%Y-%m-%d').date()
  new_date.append(date_change)


today_l = '2023-04-10'
today = datetime.strptime(today_l, '%Y-%m-%d').date()

time_to_expiration = list()
for date in new_date:
  difference = (date-today)
  time_to_expiration.append(difference.days)
df['TimeToExpiration'] = time_to_expiration
df.drop('expiration', axis=1, inplace=True)

df = df.dropna()
np.set_printoptions(precision=5, suppress=True)

print('To call data use df.head(). df is the name of the dataset when calling')

def ada_boost(a,b,c):
  w1 = 0.5
  w2 = 0.5
  error1 = 0
  error2 = 0
  weights = 0
  prediction = 0
  pred1 = np.array(a)
  pred2 = np.array(b)
  actualv = np.array(c)
  predictions = []
  for j in range(len(actualv)):
    prediction = pred1[j]*w1 + pred2[j]*w2
    error1 = abs(pred1[j]-actualv[j])
    error1 = min(1,error1)
    error2 = abs(pred2[j]-actualv[j])
    error2 = min(1,error2)
    w1 = w1/(2**error1)
    w2 = w2/(2**error2)
    weights = w1+w2
    w1 = w1/weights
    w2 = w2/weights
    predictions.append(prediction)
  return prediction
  return w1
  return w2

y = df['bid']
X = df.drop(columns = ['ask','bid'])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
predictionslr = regression_model.predict(X_test)
APElr = average_percentage_error(predictionslr,y_test)


forest_model = RandomForestRegressor(max_depth=40, n_estimators = 100)
forest_model.fit(X_train, y_train)
predictionsfr = forest_model.predict(X_test)
APEfr = average_percentage_error(predictionsfr,y_test)


NN_model = MLPRegressor(hidden_layer_sizes=(200,200,200), max_iter=600)
NN_model.fit(X_train, y_train)
predictionsmr = NN_model.predict(X_test)
APEmr = average_percentage_error(predictionsmr,y_test)


array_y_test = np.array(y_test)
array_predictionslr = np.array(predictionslr)
array_predictionsfr = np.array(predictionsfr)
array_predictionsmr = np.array(predictionsmr)
print(array_predictionslr)
print(array_predictionsfr)
print(array_predictionsmr)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

train_df = df.sample(frac=0.8, random_state = 1)
test_df = df.drop(train_df.index)
trainfs = train_df.copy()
testfs = test_df.copy()
train_labels = trainfs.pop("strike")
test_labels = testfs.pop("strike")
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(trainfs))

first = np.array(trainfs[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(128, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_percentage_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    trainfs,
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)
test_results = {}
test_results['dnn_model'] = dnn_model.evaluate(testfs, test_labels, verbose=0)


pd.DataFrame(test_results, index=['Mean absolute percentage error [strike]']).T

test_predictions = dnn_model.predict(testfs).flatten()

tenker_APE = average_percentage_error(test_predictions,test_labels)
print(tenker_APE)
