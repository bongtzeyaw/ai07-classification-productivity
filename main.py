import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 1
EPOCHS = 50
BATCH_SIZE = 32

# Load data
df = pd.read_csv('garments_worker_productivity.csv')
df.loc[ df["actual_productivity"]>0.5, "actual_productivity"] = 1
df.loc[ df["actual_productivity"]<=0.5, "actual_productivity"] = 0
df = df.drop('date',axis=1)
df = df.fillna(0.0)
df = df.replace('finishing ','finishing')
category_columns = ['quarter','department','day']
df = pd.get_dummies(df,prefix=category_columns,columns=category_columns)

# Preprocess data
y = df.actual_productivity
X = df.drop('actual_productivity',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=SEED)

sd = StandardScaler()
sd.fit(X_train)
X_train = sd.transform(X_train)
X_test = sd.transform(X_test)

#Create model
nIn = X_test.shape[1]
nClass = len(np.unique(y_test))

l1 = keras.regularizers.L1(l1=0.001)

model = keras.Sequential()
model.add(layers.InputLayer(input_shape=(nIn,)))
model.add(layers.Dense(64,activation='relu',kernel_regularizer=l1))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(64,activation='relu',kernel_regularizer=l1))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(nClass,activation='softmax')) # Don't apply regularizer to output layer

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test,y_test),batch_size=BATCH_SIZE,epochs=EPOCHS)

# Result
# With 50 epochs and a batch size of 32, our model attains a training accuracy of 90.39% and validation accuracy of 88.75%