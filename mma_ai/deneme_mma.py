#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 04:18:20 2023

@author: datadata0001
"""
#YapaySinirAğları
#################################################################################
#################################################################################
#################################################################################

#################################################################################
#################################################################################
#################################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score

veri_mma=pd.read_csv('guncel_ufc_veri.txt')
#Veriler

sig_str_def = veri_mma['sig_str_absorbed'] / veri_mma['sig_str_landed']

takedown_defense_avg = veri_mma['takedown_def']/veri_mma['total_fights']

sig_str_def_pd = pd.DataFrame(sig_str_def,columns=['sig_str_def_a'])

takedown_defense_avg_pd = pd.DataFrame(takedown_defense_avg,columns=['takedown_defense_avg'])

savunma = pd.concat([sig_str_def_pd,takedown_defense_avg_pd],axis = 1)

veri = pd.concat([veri_mma,savunma], axis = 1)

grap_str = ((veri['str_clinch'] + veri['str_ground'] )/2 ) / veri['avg_fight_time']/100

grap_str_pd = pd.DataFrame(grap_str,columns=['grap_str'])

veri = pd.concat([veri,grap_str_pd], axis = 1)

only_str = ((veri['str_standing'] + veri['str_head'] + veri['str_body'] + veri['str_leg'])/4)/veri['avg_fight_time']/100

only_str_pd = pd.DataFrame(grap_str,columns=['only_str'])

veri = pd.concat([veri,only_str_pd], axis = 1)


veri.set_index('name', inplace=True)  # 'name' sütununu indeks olarak belirleyin
veri.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]
[0,1,2,3,4,5,6,7,12,13,14,15,16,17,18,19,26,27,28,29,30]#soyutlanmış veriler

#[1,2,3,4,5,6,7,8,13,14,15,16,17,18,19,20,27,28,29] soyutlanmış veriler
#Win_Rate
x = veri.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30]]
y = veri.iloc[:,[26]] # win_rate



#Grappling
grap_str = veri.iloc[:,[29]]*10
t = veri.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,30]]
z = veri.iloc[:,[14,28]]
z = pd.concat((z,grap_str),axis=1)





#Striking
sig_str_def = veri.iloc[:,[27]]*5
only_str= veri.iloc[:,[30]]*10
a = veri.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,29]]
b = veri.iloc[:,[12]]
b = pd.concat((b,sig_str_def),axis=1)
b = pd.concat((b,only_str),axis=1)


#Control
controlx = veri.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,19,20,21,22,23,24,25,26,27,28,29,30]]
controly = veri.iloc[:,[15,18]]


#Durability
c = veri.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30]]
d = veri.iloc[:,[13,19]]


#Mentality

finishes = veri.iloc[:,[2]]/10
win_streak = veri.iloc[:,[6]]*1.5
e = veri.iloc[:,[0,1,3,4,5,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]
g = veri.iloc[:,[7]]
g = pd.concat((g,finishes),axis=1)
g = pd.concat((g,win_streak),axis=1)

########################################################
#Grafik ile İnceleme
plt.plot(veri)
plt.show()

#Win_Rate
plt.plot(y)
plt.show()
plt.plot(x)
plt.show()

#Grappling
plt.plot(z)
plt.show()
plt.plot(t)
plt.show()

#Striking
plt.plot(b)
plt.show()
plt.plot(a)
plt.show()

#Control
plt.plot(controly)
plt.show()
plt.plot(controlx)
plt.show()

#Durability
plt.plot(d)
plt.show()
plt.plot(c)
plt.show()

#Mentality
plt.plot(g)
plt.show()
plt.plot(e)
plt.show()
########################################################


rs=17898

#Win_Rate

#Standard_Scaler

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X= sc.fit_transform(x)





#Yapay_Sinir_Aglari(win_rate)


import keras

from keras.models import Sequential
from keras.layers import Dense



classifier = Sequential()

classifier.add(Dense(1024, kernel_initializer ='uniform', activation = 'tanh',input_dim = 30))

classifier.add(Dense(1, kernel_initializer ='uniform', activation = 'linear'))

classifier.compile(optimizer = "adam" , loss = 'mse' , metrics = ['mae'])

history = classifier.fit(X,y, epochs =300 ,validation_data=(X, y))


pred_win_rateD = classifier.predict(X)

evaluation = classifier.evaluate(X, y)
print(f'Mean Absolute Error on Test Set: {evaluation[1]}')





#################################################################################
#################################################################################
#################################################################################

#################################################################################
#################################################################################
#################################################################################



#Grappling



#Standard_Scaler

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

T = sc.fit_transform(t)



#Yapay_Sinir_Aglari(grappling)

import keras

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(13, kernel_initializer ='uniform', activation = 'relu',input_dim = 28))

classifier.add(Dense(838, kernel_initializer ='uniform', activation = 'relu'))

classifier.add(Dense(3, kernel_initializer ='uniform', activation = 'linear'))

classifier.compile(optimizer = 'adam' , loss = 'mse' , metrics = ['mse'])

classifier.fit(T,z, epochs =500,validation_data=(T, z))

pred_grapplingD = classifier.predict(T)


evaluation = classifier.evaluate(T, z)
print(f'Mean Absolute Error on Test Set: {evaluation[1]}')






#################################################################################
#################################################################################
#################################################################################

#################################################################################
#################################################################################
#################################################################################












#Striking



#Standard_Scaler

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

A = sc.fit_transform(a)



#Yapay_Sinir_Aglari(striking)

import keras

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(512, kernel_initializer ='uniform', activation = 'relu',input_dim = 28))

classifier.add(Dense(52, kernel_initializer ='uniform', activation = 'relu'))

classifier.add(Dense(3, kernel_initializer ='uniform', activation = 'linear'))

classifier.compile(optimizer = 'adam' , loss =  "mse" , metrics = ['mse'])

classifier.fit(A,b, epochs =300,validation_data=(A, b))

pred_strikingD = classifier.predict(A)

evaluation = classifier.evaluate(A, b)
print(f'Mean Absolute Error on Test Set: {evaluation[1]}')






#################################################################################
#################################################################################
#################################################################################

#################################################################################
#################################################################################
#################################################################################





#Control



#Standard_Scaler

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

CX = sc.fit_transform(controlx)



#Yapay_Sinir_Aglari(control)

import keras

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(200, kernel_initializer ='uniform', activation = "relu",input_dim = 29))

classifier.add(Dense(200, kernel_initializer ='uniform', activation = "relu"))

classifier.add(Dense(2, kernel_initializer ='uniform', activation = 'linear'))

classifier.compile(optimizer = 'sgd' , loss = 'mse' , metrics = ["mae"])

classifier.fit(CX,controly, epochs =500,validation_data=(CX, controly))

pred_controlD = classifier.predict(CX)

evaluation = classifier.evaluate(CX, controly)
print(f'Mean Absolute Error on Test Set: {evaluation[1]}')

#################################################################################
#################################################################################
#################################################################################

#################################################################################
#################################################################################
#################################################################################



#Durability



#Standard_Scaler

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

C = sc.fit_transform(c)



#Yapay_Sinir_Aglari(durability)

import keras

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(45, kernel_initializer ='uniform', activation = 'leaky_relu',input_dim = 29))

classifier.add(Dense(45, kernel_initializer ='uniform', activation = 'leaky_relu'))

classifier.add(Dense(2, kernel_initializer ='uniform', activation = 'linear'))

classifier.compile(optimizer = 'adam' , loss = 'mse' , metrics = ["mae"])

classifier.fit(C,d, epochs =300,validation_data=(C, d))

pred_durabilityD = classifier.predict(C)

evaluation = classifier.evaluate(C, d)
print(f'Mean Absolute Error on Test Set: {evaluation[1]}')



#################################################################################
#################################################################################
#################################################################################

#################################################################################
#################################################################################
#################################################################################




#Mentality



#Standard_Scaler

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

E = sc.fit_transform(e)



#Yapay_Sinir_Aglari(mentality)

import keras

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(45, kernel_initializer ='uniform', activation = 'leaky_relu',input_dim = 28))

classifier.add(Dense(45, kernel_initializer ='uniform', activation = 'leaky_relu'))

classifier.add(Dense(3, kernel_initializer ='uniform', activation = 'linear'))

classifier.compile(optimizer = 'sgd' , loss = 'mse' , metrics = ['mae'])

classifier.fit(E,g, epochs =300,validation_data=(E, g))

evaluation = classifier.evaluate(E, g)

pred_mentalityD = classifier.predict(E)

print(f'Mean Absolute Error on Test Set: {evaluation[1]}')

fighters=[]
for i in range(144):
    fighter = (
        (pred_win_rateD[i].mean() / 10) +
        (pred_grapplingD[i].mean() * 3) +
        (pred_strikingD[i].mean() * 2) +
        (pred_controlD[i].mean() * 30) +
        (pred_durabilityD[i].mean() / 5) +
        (pred_mentalityD[i].mean() * 4.25)
    )
    print(f"{fighter}",y.index[i])
    fighters.append((fighter,y.index[i]))
    print(fighters)

fightersRate = pd.DataFrame(fighters,columns=["rating","name"])  
print(fightersRate)
   



# Kullanıcıdan dövüşçü isimlerini al
print('İsimleri lütfen örnekte gösterildiği gibi giriniz. Ör : Valentina Shevchenko')
fighter_first = input('Birinci dövüşçünün ismini giriniz: ').lower()
fighter_second = input('Ikinci dövüşçünün ismini giriniz: ').lower()

# İsimlere göre indeksleri bul
fighter1_index = fightersRate[fightersRate['name'].str.lower() == fighter_first].index
fighter2_index = fightersRate[fightersRate['name'].str.lower() == fighter_second].index

# Dövüşçü 1 ve dövüşçü 2'nin performansını değerlendirme
fighter1_score = (  
    pred_win_rateD[fighter1_index[0]].mean()/10 +
    pred_grapplingD[fighter1_index[0]].mean()*3  + 
    pred_strikingD[fighter1_index[0]].mean()*3/2  +
    pred_controlD[fighter1_index[0]].mean()*5  +
    pred_durabilityD[fighter1_index[0]].mean()/2 +
    pred_mentalityD[fighter1_index[0]].mean() 
)

fighter2_score = (
    pred_win_rateD[fighter2_index[0]].mean()/10 +
    pred_grapplingD[fighter2_index[0]].mean()*3  + 
    pred_strikingD[fighter2_index[0]].mean() *3/2  +
    pred_controlD[fighter2_index[0]].mean()*5   +
    pred_durabilityD[fighter2_index[0]].mean()/2 +
    pred_mentalityD[fighter2_index[0]].mean()  
)

# Değerleri yazdır
print(fightersRate.loc[fighter1_index[0],'name'], "dövüşçü puanı:", fighter1_score)
print(fightersRate.loc[fighter2_index[0],'name'], "dövüşçü puanı:", fighter2_score)

if fighter1_score > fighter2_score:
    print(fightersRate.loc[fighter1_index[0],'name',], 'Muhtemelen Kazanacaktir.')
elif fighter2_score > fighter1_score:
    print(fightersRate.loc[fighter2_index[0],'name'], 'Muhtemelen Kazanacaktir.')
else:
    print("Berabere")

print('Unutmayiniz ki bunlar yalnizca tahmini verilerdir.')




        



        
###Dövüşçü girmek isterseniz kullanabilirsiniz.

name = str(input('Name: '))
total_fights = float(input('Total Fights: '))
wins = float(input('Wins: '))
finishes = float(input('Finishes: '))
ko_tko_wins = float(input('KO/TKO Wins: '))
submission_wins = float(input('Submission Wins: '))
decision_wins = float(input('Decision Wins: '))
win_streak = float(input('Win Streak: '))
title_fight_wins = float(input('Title Fight Wins: '))
sig_strikes_landed = float(input('Sig Strikes Landed: '))
sig_strikes_attempted = float(input('Sig Strikes Attempted: '))
takedowns_landed = float(input('Takedowns Landed: '))
takedown_attempted = float(input('Takedown Attempted: '))
sig_str_landed = float(input('Sig Str Landed Avg: '))
sig_str_absorbed = float(input('Sig Str Absorbed: '))
takedown_avg = float(input('Takedown Avg: '))
submission_avg = float(input('Submission Avg: '))
sig_str_def = float(input('Sig Str Defense to %: '))
takedown_def = float(input('Takedown Defense to %: '))
knockdown_avg = float(input('Knockdown Avg: '))
avg_fight_time = float(input('Average Fight Time: '))
str_standing = float(input('Striking Standing: '))
str_clinch = float(input('Striking Clinch: '))
str_ground = float(input('Striking Ground: '))
str_head = float(input('Striking Head: '))
str_body = float(input('Striking Body: '))
str_leg = float(input('Striking Leg: '))
win_rate = float(input('Win Rate: '))
sig_str_def_avg = float(input('Sig Strikes Defense Avg: '))
takedown_defense_avg = float(input('Takedown Defense Avg: '))


append_name = np.append(veri.iloc[:,0:0],[name])
append_total_fight = np.append(veri.iloc[:, 0:1], [total_fights])
append_wins = np.append(veri.iloc[:, 1:2], [wins])
append_finishes = np.append(veri.iloc[:, 2:3], [finishes])
append_ko_tko_wins = np.append(veri.iloc[:, 3:4], [ko_tko_wins])
append_submission_wins = np.append(veri.iloc[:, 4:5], [submission_wins])
append_decision_wins = np.append(veri.iloc[:, 5:6], [decision_wins])
append_win_streak = np.append(veri.iloc[:, 6:7], [win_streak])
append_title_fight_wins = np.append(veri.iloc[:, 7:8], [title_fight_wins])
append_sig_strikes_landed = np.append(veri.iloc[:, 8:9], [sig_strikes_landed])
append_sig_strikes_attempted = np.append(veri.iloc[:, 9:10], [sig_strikes_attempted])
append_takedowns_landed = np.append(veri.iloc[:, 10:11], [takedowns_landed])
append_takedown_attempted = np.append(veri.iloc[:, 11:12], [takedown_attempted])
append_sig_str_landed = np.append(veri.iloc[:, 12:13], [sig_str_landed])
append_sig_str_absorbed = np.append(veri.iloc[:, 13:14], [sig_str_absorbed])
append_takedown_avg = np.append(veri.iloc[:, 14:15], [takedown_avg])
append_submission_avg = np.append(veri.iloc[:, 15:16], [submission_avg])
append_sig_str_def = np.append(veri.iloc[:, 16:17], [sig_str_def])
append_takedown_def = np.append(veri.iloc[:, 17:18], [takedown_def])
append_knockdown_avg = np.append(veri.iloc[:, 18:19], [knockdown_avg])
append_avg_fight_time = np.append(veri.iloc[:, 19:20], [avg_fight_time])
append_str_standing = np.append(veri.iloc[:, 20:21], [str_standing])
append_str_clinch = np.append(veri.iloc[:, 21:22], [str_clinch])
append_str_ground = np.append(veri.iloc[:, 22:23], [str_ground])
append_str_head = np.append(veri.iloc[:, 23:24], [str_head])
append_str_body = np.append(veri.iloc[:, 24:25], [str_body])
append_str_leg = np.append(veri.iloc[:, 25:26], [str_leg])
append_win_rate = np.append(veri.iloc[:, 26:27], [win_rate])
append_sig_str_def_avg = np.append(veri.iloc[:, 27:28], [sig_str_def_avg])
append_takedown_def_avg = np.append(veri.iloc[:,28:29],[takedown_defense_avg])



veri_ekleme = pd.DataFrame({
    'name':append_name,
    'total_fights': append_total_fight,
    'wins': append_wins,
    'finishes': append_finishes,
    'ko/tko': append_ko_tko_wins,
    'submission_wins': append_submission_wins,
    'decision_wins': append_decision_wins,
    'win_streak': append_win_streak,
    'title_fight_wins': append_title_fight_wins,
    'sig_strikes_landed': append_sig_strikes_landed,
    'sig_strikes_attempted': append_sig_strikes_attempted,
    'takedowns_landed': append_takedowns_landed,
    'takedown_attempted': append_takedown_attempted,
    'sig_str_landed': append_sig_str_landed,
    'sig_str_absorbed': append_sig_str_absorbed,
    'takedown_avg': append_takedown_avg,
    'submission_avg': append_submission_avg,
    'sig_str_def': append_sig_str_def,
    'takedown_def': append_takedown_def,
    'knockdown_avg': append_knockdown_avg,
    'avg_fight_time': append_avg_fight_time,
    'str_standing': append_str_standing,
    'str_clinch': append_str_clinch,
    'str_ground': append_str_ground,
    'str_head': append_str_head,
    'str_body': append_str_body,
    'str_leg': append_str_leg,
    'win_rate': append_win_rate,
    'sig_str_def_avg': append_sig_str_def_avg,
    'takedown_def_avg': append_takedown_def_avg
})




