
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier  

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.tree import DecisionTreeClassifier

#Dosyayi Yukle

veri = pd.read_csv('data/pulsar_stars.csv')

ozellik_sayisi = 9





#giris cikis belirle

giris_verileri = veri.iloc[:,1:ozellik_sayisi+1]

cikis = veri.iloc[:,-1]



#Egitim ve test verilerini ayir

egitim_giris, test_giris,egitim_cikis, test_cikis = train_test_split(giris_verileri,cikis, test_size=0.15, random_state=0)



#Standardizasyon

scaler = preprocessing.StandardScaler()

stdGiris = scaler.fit_transform(egitim_giris)

stdTest = scaler.transform(test_giris)

siniflandiricilar=[KNeighborsClassifier(n_neighbors=3), LogisticRegression(random_state=0), GaussianNB(), DecisionTreeClassifier()]



basari=list()

fSkor = list()

for i in range(4):

    siniflandiricilar[i].fit(stdGiris, egitim_cikis)

    cikis_tahmin = siniflandiricilar[i].predict(stdTest)

    basari.append(accuracy_score(test_cikis, cikis_tahmin))

    fSkor.append( f1_score(test_cikis, cikis_tahmin, labels=None, pos_label=1, average='binary', sample_weight=None))
