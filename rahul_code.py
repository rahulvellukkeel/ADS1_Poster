# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:23:31 2023

@author: rak29
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import seaborn as sns
from sklearn import cluster
import err_ranges as err

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
        return df_changed, df_transposed

def curve_fun(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    c = scale * np.exp(growth * (t-1970))
    return c


df_co2, df_co2t = read_file("E:/Herts/ADS1/Assignment 3/Rahul/co2_emission.xlsx")
df_gdp, df_gdpt = read_file("E:/Herts/ADS1/Assignment 3/Rahul/gdp.xlsx")
df_renew, df_renewt = read_file("E:/Herts/ADS1/Assignment 3/Rahul/Renewable.xlsx")


param, cov = opt.curve_fit(curve_fun,df_co2t["year"],df_co2t["INDIA"],p0=[4e8, 0.1])
sigma = np.sqrt(np.diag(cov))
print(*param)
low,up = err.err_ranges(df_co2t["year"],curve_fun,param,sigma)






