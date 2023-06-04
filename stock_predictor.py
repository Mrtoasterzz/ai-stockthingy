import twelvedata as td
from twelvedata import TDClient

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Normalization, CuDNNLSTM
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import adam
from keras.losses import binary_crossentropy, mean_squared_error 
from keras.metrics import MeanAbsoluteError
import keras 
from keras import layers 
from keras.utils.vis_utils import plot_model
import pydot

from sklearn.model_selection import train_test_split

import pandas_datareader
import DateTime
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing as pre

import os
import json


td = TDClient(apikey="5af0b4c282db4d2bbb41b55526559a08")
Stock_ticker ="spy"
size =5000
date ="2003-07-24"

#pd.set_option('display.max_row', None)

# gets stock data from twelve date
spy_data_first_half=td.time_series( symbol=Stock_ticker, interval="1day", start_date=date ,order="desc", outputsize=size)
spy_data_second_half=td.time_series( symbol=Stock_ticker, interval="1day", end_date=date ,order="desc", outputsize=size)

#makes data into a panda dataframe
spy_first = spy_data_first_half.as_pandas()
spy_second = spy_data_second_half.as_pandas() 

#combines both datasets into one dataframe
spy_all = spy_first.combine_first(spy_second) 
spy = pd.DataFrame(data=spy_all)

# sets all variables   
open_price = spy.open
price_high = spy.high
price_low = spy.low
close_price = spy.close
day_mean = (open_price+ close_price)/2

tensor_mean = tf.constant(day_mean)
tensor_opening = tf.constant(open_price)

# set vaules to a range from -1,1 
scaler = pre.MinMaxScaler(feature_range=(-1,1), copy= False, clip=False)
tensor_open = scaler.fit_transform(np.array(tensor_opening).reshape(-1,1))

#print(scaler.inverse_transform(tensor_open))
#print(scaler.inverse_transform(tensor_open)==spy)
#splitting it up to testing and training data 
training , test = train_test_split(tensor_open, train_size=0.9, shuffle= False)
X_train_data=[]
X_test_data=[]

Y_train_data=[]
Y_test_data=[]

train_len = len(training)
test_len = len(test)

# Create the training dataset
for i in range(train_len-101):
    a = training[i:(i+100), 0]
    X_train_data.append(a)
    Y_train_data.append(training[i + 100, 0])

# Create the test dataset
for j in range(test_len-101):
    b = test[j:(j+100), 0]
    X_test_data.append(a)
    Y_test_data.append(test[j + 100, 0])


X_train_data = np.array(X_train_data)
Y_train_data = np.array(Y_train_data)
X_test_data = np.array(X_test_data)
Y_test_data = np.array(Y_test_data)

model = Sequential()

model.add(LSTM(100, return_sequences=True,input_shape=(100,1),activation='sigmoid'))
    # 20% of the layers will be 
model.add(Dropout(0.2))
    # 2nd LSTM layer
    # * units = add 50 neurons is the dimensionality of the output space
    # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
model.add(LSTM(50, return_sequences=True, activation='sigmoid'))
    # 20% of the layers will be dropped
model.add(Dropout(0.2))
    # 3rd LSTM layer
    # * units = add 50 neurons is the dimensionality of the output space
    # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
model.add(LSTM(50, return_sequences=True,activation='sigmoid'))
    # 50% of the layers will be dropped
model.add(Dropout(0.5))
    # 4th LSTM layer
    # * units = add 50 neurons is the dimensionality of the output space
model.add(LSTM(50))
    # 50% of the layers will be dropped
model.add(Dropout(0.5))
    # Dense layer that specifies an output of one unit
model.add(Dense(1))


checkpoint = ModelCheckpoint("checkpoint1.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

logdir='logs1'
tensorboard_Visualization = TensorBoard(log_dir=logdir)

model.compile(loss='mean_squared_error', optimizer='adam')
# Training The Model

model.fit(X_train_data, 
          Y_train_data, 
          validation_data=(X_test_data, Y_test_data), 
          epochs=len(spy), 
          batch_size=16, 
          verbose=1,
          callbacks=[checkpoint, tensorboard_Visualization])



train_predict = model.predict(X_train_data)
test_predict = model.predict(X_test_data)

# Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# prints the prediction im like 90% sure
print("Train MSE = ", math.sqrt(mean_squared_error(Y_train_data, train_predict)))
print("Test MSE = ", math.sqrt(mean_squared_error(Y_test_data, test_predict)))

#test accuraracy hoppefully 
score = model.evaluate(X_test_data, Y_test_data, batch_size = 32) 


acc = model.evaluate(X_test_data, Y_test_data, batch_size = 32) 

print('Test score:', score) 
print('Test accuracy:', acc)
