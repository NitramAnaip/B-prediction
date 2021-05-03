import numpy as np
import pandas as pd
from sklearn import metrics

import torch
import torch.nn as nn

from keras.utils import to_categorical

from utils import preprocess, split_multi_seq, get_unblaced_freq
from model import LSTM


#opening cvs
df = pd.read_csv("Binance_BTCUSDT_1h.csv")

df, num_classes = preprocess(df)

labels = [ 0, 1, 2]
get_unblaced_freq(df, labels)

# How many periods looking back to train
n_per_in  = 30
# How many periods ahead to predict
n_per_out = 1




# Splitting the data into appropriate sequences 

close = np.array(df['groups'])
vol = list(df['Volume USDT'])
evolution_group = np.array(to_categorical(close,num_classes), dtype=np.float32)
split = list(df['split'])
close_evolution = list(df["evolution"])



X, y = split_multi_seq(close_evolution, vol, split, evolution_group, n_per_in, n_per_out)




y = y.reshape(y.shape[0], y.shape[2])
#y = tf.convert_to_tensor(y)
# Reshaping the X variable from 2D to 3D
#X = tf.convert_to_tensor(X)
# Reshaping the X variable from 2D to 3D




X_val, X = X[:int(X.shape[0]/10)], X[int(X.shape[0]/10):]
y_val, y = y[:int(y.shape[0]/10)], y[int(y.shape[0]/10):]


x_train = torch.from_numpy(X).type(torch.Tensor)
y_train = torch.from_numpy(y).type(torch.Tensor)
X_val = torch.from_numpy(X_val).type(torch.Tensor)


input_dim = 3
hidden_dim = 32
num_layers = 6
output_dim = 3
num_epochs = 40

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

import time
hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []
for t in range(num_epochs):
    y_train_pred = model(x_train)
    loss = criterion(y_train_pred, y_train)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))

print(y_train_pred)



y_pred = model(X_val).detach().numpy()
if y_pred is not None:
    b = np.zeros_like(y_pred)
    b[np.arange(len(y_pred)), y_pred.argmax(1)] = 1

    print(metrics.accuracy_score(y_val, b))
    a=metrics.multilabel_confusion_matrix(y_val, b)
    print(a)
    for i in range (len(a)):
        TP = a[i][1][1]
        FP = a[i][0][1]
        print("cat {} TP/(TP+FP): {}".format(i, TP/(FP+TP)))
    