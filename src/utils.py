# Library Imports
from datetime import datetime
from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
plt.style.use("ggplot")
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout




def preprocess(df, df_btc):

    df_new = pd.DataFrame({"unix": [], "open": [], "high": [], "low": [], "close": [], "Volume USDT": []})

    #transforming into humanly readable time
    for index in df.index:
        if df['unix'][index]>1000000000000:
            #getting rid of ms
            df['unix'][index] = df['unix'][index]/1000
        df['unix'][index] = datetime.utcfromtimestamp(df['unix'][index]).strftime('%Y-%m-%d %H:%M:%S')

    df_base = df.drop_duplicates(subset=['unix'], ignore_index=True)
    df_base.head()



    #Creating the % evolution
    evol = []
    evol_btc = []
    for i in range(df.shape[0]-1):
        evol.append((df['close'][i]-df['close'][i+1])/df['close'][i+1])
        evol_btc.append((df_btc['close'][i]-df_btc['close'][i+1])/df_btc['close'][i+1])
        


    evol.append(0)
    evol_btc.append(0)
    df['evolution'] = evol

    df["split"] = (df["high"] - df["low"])/df["low"]
    df_btc["split"] = (df_btc["high"] - df_btc["low"])/df_btc["low"]

    labels = [ 0, 1, 2]
    num_classes=len(labels)
    df["groups"] = pd.cut(df["evolution"], bins=[-100,-0.01,0.01, 100], labels = labels)
    df["btc_evol"] = evol_btc


    #Choice of features
    df = df.set_index("unix")[["evolution", 'groups', 'Volume USDT', "split", "btc_evol"]].head(32000)
    return df, num_classes

def get_unblaced_freq(df, labels):
    for label in labels:
        print(df[df["groups"]==label].shape[0])


def split_multi_seq(close_evolution, volume, split, btc_evol, evolution_group, n_steps_in, n_steps_out):
    """
    Here seq is of the form [[prices], [volumes], ...]
    """
    X, y = [], []
    for i in range(len(volume)):
        seq_x, seq_y = [], []
        end = i + n_steps_in
        out_end = end + n_steps_out
              
        if out_end > len(volume):
            break
        
        scaler = MinMaxScaler()
        scaled_vol = np.array(volume[max(i-240, 0):end]) # the max part is to have a volume over 10 days
        #print(volume)
        scaled_vol = scaled_vol.reshape(-1, 1)
        
        scaler.fit(scaled_vol) 
        scaled_vol = scaler.transform(scaled_vol)
        
        scaled_vol = list(scaled_vol.reshape(scaled_vol.shape[0]))
        scaled_vol = scaled_vol[-n_steps_in:]


        #print(volume)
        
        # This is a trick I'm trying: I can't have my training data of the for [ [0,0,1], [volume]]
        #as it poses a pb when I transform it to a tensor. I'm therefore concatenating it in the form
        #[ [0,0,1,volume] ]
        for k in range (i, end): 
            inputs = [close_evolution[k]] #evolution
            inputs.append(scaled_vol[k-i])
            inputs.append(split[k]) #difference between high and low
            inputs.append(btc_evol[k])
            seq_x.append(inputs)


        for k in range(end, out_end):
            seq_y.append(evolution_group[k]) #category
        y.append(seq_y)
        X.append(seq_x)
    #print(X[:3])
    return np.array(X), np.array(y)



