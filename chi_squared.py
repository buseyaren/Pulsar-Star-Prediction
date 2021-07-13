# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:32:49 2019

@author: Buse
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2

file_data = pd.read_csv('data/pulsar_stars.csv')
group_ids=file_data.iloc[:,0]
raw_data=file_data.iloc[:,1:-1]
result_data=file_data.iloc[:,-1]

min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(raw_data)
chi_vals, p_vals = chi2(X,result_data)

n=100
max_val = np.argsort(chi_vals)[::-1][:n]


