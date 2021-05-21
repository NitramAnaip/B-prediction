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
    """
    Preprocessing of the data: gets timestamps into human readable, gets the evolution of the price in percentage of the price from one timeframe to another, 
    calculates the difference between high and low
    """

    #df_new = pd.DataFrame({"unix": [], "open": [], "high": [], "low": [], "close": [], "Volume USDT": []})

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

    labels = [ 0, 1]
    num_classes=len(labels)
    df["groups"] = pd.cut(df["evolution"], bins=[-100,0.01, 100], labels = labels)
    df["btc_evol"] = evol_btc
    df["Volume BTC"] = df_btc["Volume BTC"].head(df.shape[0])


    #Choice of features
    df = df['date','close', 'Volume LINK', 'tradecount', 'open', 'high', "Volume BTC"].head(32000)
    
    #df = df.set_index("unix")[["evolution", 'groups', 'Volume LINK', "split", "btc_evol", "Volume BTC"]].head(32000)
    return df, num_classes

def get_unblaced_freq(df, labels):
    for label in labels:
        print(df[df["groups"]==label].shape[0])


def scale(volume, i, end, n_steps_in, nbr_dt):
    """
    Function that scales the list (volume) given over the time frame specified by nbr_dt
    """
    scaler = MinMaxScaler()
    scaled_vol = np.array(volume[max(i-nbr_dt, 0):end]) # the max part is to have a volume over 10 days
    #print(volume)
    scaled_vol = scaled_vol.reshape(-1, 1)
    
    scaler.fit(scaled_vol) 
    scaled_vol = scaler.transform(scaled_vol)
    
    scaled_vol = list(scaled_vol.reshape(scaled_vol.shape[0]))
    scaled_vol = scaled_vol[-n_steps_in:]
    return scaled_vol

def split_multi_seq(close_evolution, volume, split, btc_evol, btc_vol, evolution_group, n_steps_in, n_steps_out, nbr_dt):
    """
    ARGS:
         - nbr_dt is the number of time frames between the ast info and the moment we want to predict. For instance if nbr_dt=3 
        we want to predict the changes 3dt after the time of the last input)
    """
    X, y = [], []
    for i in range(len(volume)):
        seq_x, seq_y = [], []
        end = i + n_steps_in
        out_end = end + n_steps_out
              
        if out_end > len(volume)-(nbr_dt-1):
            break
        
        scaled_btc_vol = scale(btc_vol, i, end, n_steps_in, 240)
        scaled_vol = scale(volume, i, end, n_steps_in, 240)
        scaled_split = scale(split, i, end, n_steps_in, 240)


        #print(volume)
        
        # This is a trick I'm trying: I can't have my training data of the for [ [0,0,1], [volume]]
        #as it poses a pb when I transform it to a tensor. I'm therefore concatenating it in the form
        #[ [0,0,1,volume] ]
        for k in range (i, end): 
            inputs = [close_evolution[k]] #evolution
            inputs.append(scaled_vol[k-i])
            inputs.append(scaled_split[k-i]) #difference between high and low
            inputs.append(btc_evol[k])
            inputs.append(scaled_btc_vol[k-i])
 
            seq_x.append(inputs)


        for k in range(end, out_end):
            
            seq_y.append(evolution_group[k+nbr_dt-1]) #category
        y.append(seq_y)
        X.append(seq_x)
    #print(X[:3])
    return np.array(X), np.array(y)


def preprocess_unix(df):
    for index in df.index:
        if df['unix'][index]>1000000000000:
            #getting rid of ms
            df['unix'][index] = df['unix'][index]/1000
        df['unix'][index] = datetime.utcfromtimestamp(df['unix'][index]).strftime('%Y-%m-%d %H:%M:%S')
    return df

def create_new_df(df, dt, crypto_name):
    """
    df: fataframe with 1 minute interval in data
    dt: time interval between every data line in new dataframe (in minutes)

    retuens a dataframe with the new data but with different dt between data
    NOTE THAT THIS IS TO BE RUN ONCE THE UNIX TIMESTAMPS ARE ORDERED IN ASCENDING ORDER (ie earliest has index 0)
    """

    df_new = pd.DataFrame({"unix": [], "open": [], "high": [], "low": [], "close": [], "Volume "+crypto_name: [], "Volume USDT": [], "tradecount": []})
    for i in range (0, df.shape[0]-dt, dt):
        new = []
        new.append(df["unix"][i])
        new.append(df["open"][i])
        highs = list(df["high"][i:i+dt])
        new.append(max(highs))
        lows = list(df["low"][i:i+dt])
        new.append(min(lows))
        new.append(df["close"][i+dt])
        new.append(sum(list(df["Volume USDT"][i:i+dt])))
        new.append(sum(list(df["Volume "+crypto_name][i:i+dt])))
        new.append(sum(list(df["tradecount"][i:i+dt])))
        df_new = df_new.append(pd.Series(new, index=df_new.columns), ignore_index=True)

    return df_new


def close_data(df, key, index, m, m_out, s):
    close_list=[]
    start = index+m_out
    end = start+s*m

    for i in range (start, end, m):
        try:
            close_list.append(df[key][i])
        except:
            import pdb
            pdb.set_trace()
    return close_list