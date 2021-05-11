import numpy as np
import pandas as pd
import time
from sklearn import metrics

import torch
import torch.nn as nn
import xgboost as xgb

from keras.utils import to_categorical

from utils import preprocess, split_multi_seq, get_unblaced_freq
from model import LSTM, GRU


#opening cvs
df = pd.read_csv("Binance_LINKUSDT_1h.csv")
df_btc = pd.read_csv("Binance_BTCUSDT_1h.csv")
df, num_classes = preprocess(df, df_btc)


xgb_bool  = True
labels = [ 0, 1, 2]

# How many periods looking back to train
n_per_in  = 30
# How many periods ahead to predict
n_per_out = 1
nbr_dt = 3




# Splitting the data into appropriate sequences 

close = np.array(df['groups'])
vol = list(df['Volume LINK'])
evolution_group = list(close)#np.array(to_categorical(close,num_classes), dtype=np.float32)
split = list(df['split'])
close_evolution = list(df["evolution"])
btc_evol = list(df["btc_evol"])
btc_vol = list(df["Volume BTC"])



X, y = split_multi_seq(close_evolution, vol, split, btc_evol, btc_vol, evolution_group, n_per_in, n_per_out, nbr_dt)

print(X.shape)
print(y.shape)
# Shuffling the sequences:
p = np.random.permutation(len(X))
X, y = X[p], y[p]
import pdb
pdb.set_trace()

y = y.reshape(y.shape[0])

#y = tf.convert_to_tensor(y)
# Reshaping the X variable from 2D to 3D
#X = tf.convert_to_tensor(X)
# Reshaping the X variable from 2D to 3D




X_val, X = X[:int(X.shape[0]/10)], X[int(X.shape[0]/10):]
y_val, y = y[:int(y.shape[0]/10)], y[int(y.shape[0]/10):]





input_dim = 5
hidden_dim = 32
num_layers = 6
output_dim = 3
num_epochs = 100

if xgb_bool:

    X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1]*X_val.shape[2]))


    dtrain = xgb.DMatrix(data=X, label=y)
    dtest = xgb.DMatrix(data=X_val, label=y_val)
    # Parameter dictionary
    params = {'max_depth':4, 'objective':'multi:softprob',
              'n_estimators':100, 'booster':'gbtree', 'num_class':3} 

    # Train the model with train data sets
    xgb_clf = xgb.train(params=params, dtrain=dtrain)

    y_pred = xgb_clf.predict(dtest)
    y_train_pred = xgb_clf.predict(dtrain)

else :

    x_train = torch.from_numpy(X).type(torch.Tensor)
    y_train = torch.from_numpy(y).type(torch.Tensor)
    X_val = torch.from_numpy(X_val).type(torch.Tensor)


    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.CrossEntropyLoss(weight = torch.from_numpy(np.array([0.3, 0.4, 0.3])).type(torch.Tensor))
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)


    hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []
    for t in range(num_epochs):
        y_train_pred = model(x_train)

        #print(y_train_pred.shape())
        loss = criterion(y_train_pred, y_train.type(torch.LongTensor))
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
    training_time = time.time()-start_time
    print("Training time: {}".format(training_time))




    y_pred = model(X_val).detach().numpy()


# this section was only in order to try and get a threshold
"""
means = [0,0,0]
y_train_pred = y_train_pred.detach().numpy()
for i in range (y_train_pred.shape[0]):
    for j in range (len(means)):
        means[j] += y_train_pred[i][j]
for j in range (len(means)):
    means[j] = means[j] / y_train_pred.shape[0]

means = np.array(means)
means = list(means)
means = [means]*y_pred.shape[0]
means = np.array(means)
y_train = y_train.detach().numpy()
"""




if y_pred is not None:
    b = np.zeros_like(y_pred)
    b[np.arange(len(y_pred)), y_pred.argmax(1)] = 1

    
    y_val = to_categorical(y_val)

    """
    y_pred = y_pred>means
    for i in range (means.shape[0]-1, -1, -1):

        
        if y_pred[i].sum()>1:
            y_pred = np.delete(y_pred, i, 0)
            y_val = np.delete(y_val, i, 0)
    """

    print(y_val.shape)
    print(metrics.accuracy_score(y_val, b))
    a=metrics.multilabel_confusion_matrix(y_val, b)
    print(a)
    for i in range (len(a)):
        TP = a[i][1][1]
        FP = a[i][0][1]
        print("cat {} TP/(TP+FP): {}".format(i, TP/(FP+TP)))
