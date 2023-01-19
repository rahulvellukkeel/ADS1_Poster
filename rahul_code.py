# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:23:31 2023

@author: rak29
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_file(file_name):
        df = pd.read_excel(file_name)
        df_changed = df.drop(columns=["Series Name","Country Name","Country Code"])
        #print(df)
        df_changed = df_changed.replace(np.nan,0)
        df_transposed = np.transpose(df_changed)
        #print(df_transposed)
        df_transposed = df_transposed.reset_index()
        df_transposed = df_transposed.rename(columns={"index":"year", 0:"INDIA", 1:"UK"})

        df_transposed = df_transposed.iloc[1:]
        df_transposed = df_transposed.dropna()
        #print(df_transposed)

        df_transposed["year"] = df_transposed["year"].str[:4]
        df_transposed["year"] = pd.to_numeric(df_transposed["year"])
        df_transposed["INDIA"] = pd.to_numeric(df_transposed["INDIA"])
        df_transposed["UK"] = pd.to_numeric(df_transposed["UK"])
        print(df_transposed)
        #print(df_transposed["INDIA"])



read_file("E:/Herts/ADS1/Assignment 3/Rahul/co2_emission.xlsx")
read_file("E:/Herts/ADS1/Assignment 3/Rahul/gdp.xlsx")
read_file("E:/Herts/ADS1/Assignment 3/Rahul/Renewable.xlsx")
