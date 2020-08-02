import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

dataset_train = pd.read_csv(r'C:\Users\Varun Boya\Desktop\cotton\train.csv')
train = dataset_train.iloc[:, 1:2].values

type(train)

#print(train)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(train)

X_train = []
y_train = []
for i in range(30,202):
    X_train.append(training_set_scaled[i-30:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

model = Sequential()

model.add(LSTM(units = 100,return_sequences = True, input_shape = (X_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 100, return_sequences = True,activation='tanh', recurrent_activation='sigmoid' ))
model.add(Dropout(0.2))

model.add(LSTM(units = 100, return_sequences = True,activation='tanh', recurrent_activation='sigmoid'))
model.add(Dropout(0.2))

model.add(LSTM(units = 100, return_sequences = False,activation='tanh', recurrent_activation='sigmoid'))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train, y_train, epochs = 50, batch_size = 32)

dataset_test = pd.read_csv(r'C:\Users\Varun Boya\Desktop\cotton\test.csv')
real_price = dataset_test.iloc[:, 1:2].values


dataset_total = pd.concat((dataset_train['Price'], dataset_test['Price']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 30:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(30, 68):
    X_test.append(inputs[i-30:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_price = model.predict(X_test)
predicted_price = sc.inverse_transform(predicted_price)


plt.plot(real_price, color = 'red', label = 'Real Cotton Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Cotton Price')
plt.title('Prediction')
plt.xlabel('Time')
plt.ylabel('CottonPrice')
plt.legend()
plt.show()

pickle.dump(predicted_price,open('model1.pkl','wb'))
model1=pickle.load(open('model1.pkl','rb'))