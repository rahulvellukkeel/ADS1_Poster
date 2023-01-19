# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:23:31 2023

@author: rak29
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import scipy.optimize as opt
#import seaborn as sns
#from sklearn import cluster
#import err_ranges as err
import numpy as np

def read_file(file_name):
        df = pd.read_excel(file_name)
        df_changed = df.drop(columns=["Series Name","Country Name","Country Code"])
        print(df)

read_file("E:/Herts/ADS1/Assignment 3/Rahul/co2_emission.xlsx")