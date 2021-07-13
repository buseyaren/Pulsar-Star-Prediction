# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:40:47 2019

@author: Win7
"""
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def chi2_ozellik_cikar(giris, cikis, ozellik_sayisi):

    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(giris)
    chi_vals, p_vals = chi2(X,cikis)
    ozellikler  = np.argsort(chi_vals)[::-1][:ozellik_sayisi]
    return ozellikler