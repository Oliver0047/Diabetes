#coding:utf-8
#autor:Oliver
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
import random
import tst_yu4chu4li2_04
import train
import pandas as pd
from sklearn.neighbors import NearestNeighbors 
random.seed(2018)

def split_none(data):
    nonone = []
    none = []
    for elm in data:
        flag_none = 0
        for elm1 in elm:
            if elm1 is None:
                flag_none = 1
                break
        if flag_none is 0:
            nonone += [elm]
        if flag_none is 1:
            none += [elm]
    return (np.array(nonone),np.array(none))

file_train = "E://diabetes/data/diabetes_regression.db"
conn = sqlite3.connect(file_train)
curs = conn.cursor()
query = "select * from test"
curs.execute(query)
test_data = curs.fetchall()
conn.close()


#用训练集和测试集完整的数据的均值填充测试集缺漏值
x_nan = []
x_nv = []
for elm in test_data:
    if elm[1]=="男":
        x_nan += [[elm[0]]+[elm[2]]+list(elm[4:])]
    if elm[1]=="女":
        x_nv += [[elm[0]]+[elm[2]]+list(elm[4:])]

x_nan=np.array(x_nan)
x_nv=np.array(x_nv)

data=[x_nan,x_nv]
  
mean_nan=tst_yu4chu4li2_04.mean_nan.reshape(1,38)
std_nan=tst_yu4chu4li2_04.std_nan.reshape(1,38)

mean_nv=tst_yu4chu4li2_04.mean_nv.reshape(1,38)
std_nv=tst_yu4chu4li2_04.std_nv.reshape(1,38)

for i in range(len(data)):
    r,c=data[i].shape
    nonone,none=split_none(data[i])
    nonone_mean=np.mean(nonone,0)[1:]
    if i==0:
        both_mean=(nonone_mean+mean_nan)/2
    else:
        both_mean=(nonone_mean+mean_nv)/2
    for j in range(1,c):
        data[i][np.argwhere(data[i][:,j]==None),j]=both_mean[0][j-1]
data1=[[],[]]
data1[0]=(data[0][:,1:]-mean_nan)/std_nan
data1[1]=(data[1][:,1:]-mean_nv)/std_nv
both=np.concatenate(data,0)
both1=np.concatenate(data1,0)
both1=both1[both[:,0].argsort(),:]
'''
mean_nan=tst_yu4chu4li2_04.mean_nan.reshape(1,38)
std_nan=tst_yu4chu4li2_04.std_nan.reshape(1,38)

mean_nv=tst_yu4chu4li2_04.mean_nv.reshape(1,38)
std_nv=tst_yu4chu4li2_04.std_nv.reshape(1,38)

def kneighbor_index(miss,x):
    if x[0]=='男':
        group=np.array(tst_yu4chu4li2_04.x_nan2_flt)
    elif x[0]=='女':
        group=np.array(tst_yu4chu4li2_04.x_nv2_flt)
    x=np.array(x[1:]).astype(np.float64)
    x=x.reshape(1,len(x))
    miss=[i-1 for i in miss]
    le=list(np.arange(group.shape[1]))
    le=[i for i in le if i not in miss]
    x=x[:,le]
    group1=group[:,le]
    nbrs = NearestNeighbors(n_neighbors=10, algorithm="auto").fit(group1)
    distances, indices = nbrs.kneighbors(x)
    indices=list(indices.reshape(10))
    s=np.sum(distances)
    weights=distances/s
    #s=np.exp(-distances)
    #weights=(s-np.mean(s))/(np.std(s))
    choice=group[indices]
    re=choice*(weights.T)
    re=list(np.sum(re,0))
    fin=[]
    for i in miss:
        fin.append(re[i])
    return fin

X=[]
for elm in test_data:   
    X += [list(elm[1:3])+list(elm[4:])]

for i in range(len(X)):
    d=X[i]
    miss=[]
    for j in range(len(d)):
        if d[j] is None:
            miss.append(j)
    if len(miss)!=0:
        res=kneighbor_index(miss,d)
        n=0
        for k in miss:
            X[i][k]=res[n]
            n=n+1
    X[i]=X[i][1:]
X=np.array(X).astype(np.float64)
'''
pred=[]
model_nv=train.model
model_nan=train.model1

for i in range(1000):
    x=both1[i].reshape(1,38)
    if test_data[i][1]=='男':        
        #x=(x-mean_nan)/std_nan
        re=model_nan.predict(x)
    else:
        #x=(x-mean_nv)/std_nv
        re=model_nv.predict(x)
    if re<3:
        re=np.array([3.0]) 
    elif re>30:
        re=np.array([30.0]) 
    pred.append(re)
df=pd.DataFrame(pred)
df.to_csv('submit.csv',index=False,header=False)