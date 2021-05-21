import os
import torch
import random
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils import preprocess_unix, close_data


class crypto_dataset(Dataset):
    def __init__(self, crypto_name, dataframe, dataframe_BTC, s, m0, m1, m2, m_out, close,volume_crypto, tradecount=False, volume_USDT=False, open_=False, high=False, low=False):
        """
        dataframe is the dataframe as we have it fresh out from the csv: unix in descending order etc
        s is the number of data points per time scale (we have three time scales: m0, m1, m2)
        m0 is the number of minutes seperating every data point in first time scale
        open is bool. If open == True we keep the open data both for btc and crypto
        """
        self.df_BTC = preprocess_unix(dataframe_BTC)
        self.df = preprocess_unix(dataframe)
        self.s = s
        self.m0 = m0
        self.m1=m1
        self.m2 = m2
        self.m_out = m_out
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume_crypto = volume_crypto
        self.volume_USDT = volume_USDT
        self.tradecount = tradecount

        #dropping unwanted columns:
        self.df = self.df.drop(columns=["date", "symbol"])
        self.df_BTC = self.df_BTC.drop(columns=["date", "symbol"])


        if self.volume_USDT == False:
            self.df = self.df.drop(columns=["Volume USDT"])
            self.df_BTC = self.df_BTC.drop(columns=["Volume USDT"])
        if self.open == False:
            self.df = self.df.drop(columns=["open"])
            self.df_BTC = self.df_BTC.drop(columns=["open"])
        if self.high == False:   
            self.df = self.df.drop(columns=["high"])
            self.df_BTC = self.df_BTC.drop(columns=["high"])
        if self.low == False:   
            self.df = self.df.drop(columns=["low"])
            self.df_BTC = self.df_BTC.drop(columns=["low"])
        if self.tradecount == False:   
            self.df = self.df.drop(columns=["tradecount"])
            self.df_BTC = self.df_BTC.drop(columns=["tradecount"])


        #merging BTC and other crypto df

        self.df = self.df.merge(self.df_BTC, how='inner', on='unix')
        self.df = self.df.drop_duplicates(subset=['unix'], ignore_index=True)
        print(self.df.head())

    def __len__(self) :
        return self.df[self.m_out:-(self.m2*self.s)+1].shape[0]
        

    def __getitem__(self, index: int):
        """
        the index will be the index of the data from which we compute the label (ie if the input stops at 10 am on the 5/02 
        and we want to predict for the next day then the index will be the index of the data of 6/02 at 10 am
        """

        label = self.df["close_x"][index]/self.df["close_x"][index+self.m_out]
        if label>0.01:
            label = np.array([1])
        else:
            label = np.array([0])


        close_BTC_m0 = close_data(self.df, "close_y",index, self.m0, self.m_out, self.s)
        close_BTC_m1 = close_data(self.df, "close_y",index, self.m1, self.m_out, self.s)
        close_BTC_m2 = close_data(self.df, "close_y",index, self.m2, self.m_out, self.s)
        close_BTC = close_BTC_m0 + close_BTC_m1 + close_BTC_m2

        close_m0 = close_data(self.df, "close_x",index, self.m0, self.m_out, self.s)
        close_m1 = close_data(self.df, "close_x",index, self.m1, self.m_out, self.s)
        close_m2 = close_data(self.df, "close_x",index, self.m2, self.m_out, self.s)
        close = close_m0 + close_m1 + close_m2


        input_data = np.array([close_BTC, close])

        
        return input_data, label
