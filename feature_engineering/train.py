#coding:utf-8
#autor:Oliver
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.neighbors import NearestNeighbors  
#get none-clustered-data
import tst_yu4chu4li2_04
xi_nv=tst_yu4chu4li2_04.x_nv2
labeli_nv=np.array(tst_yu4chu4li2_04.y_nv2_flt)

#define split ratio and etc.
split=1
max_iter1=10000
alpha1=0.001
'''
#get clustered-data
import GMM
x=GMM.X
label=GMM.label
err=0
num_all=0
for i in range(len(x)): 
    xi=x[i]
    labeli=label[i]
    #split data
    num=int(len(xi)*0.8)
    
    xi_train=xi[:num]
    yi_train=labeli[:num]
    xi_valid=xi[num:]
    yi_valid=labeli[num:] 
    num_all+=len(xi_valid)
    #define model
    model=Lasso(max_iter=max_iter1,alpha=alpha1)
    model.fit(xi_train,yi_train)
    y_pred=model.predict(xi_valid)
    #calculate cost error
    err+=np.sum((y_pred-yi_valid)**2)    

f=err/((num_all)*2)
print('女性均方误差：',f)
'''
num=int(len(xi_nv)*split)
xi_train_nv=xi_nv[:num]
yi_train_nv=labeli_nv[:num]
#xi_valid_nv=xi_nv[num:]
#yi_valid_nv=labeli_nv[num:] 
#define model
model=Lasso(max_iter=max_iter1,alpha=alpha1)
model.fit(xi_train_nv,yi_train_nv)
y_pred_nv=model.predict(xi_train_nv)
#calculate cost error
err=np.sum((y_pred_nv-yi_train_nv)**2)  
f=err/(len(y_pred_nv)*2)
print('女性均方误差：',f)  

'''
x1=GMM.X1
label1=GMM.label1
err1=0
num_all1=0

for i in range(len(x1)): 
    xi=x1[i]
    labeli=label1[i]
    #split data
    num=int(len(xi)*0.8)   
    xi_train=xi[:num]
    yi_train=labeli[:num]
    xi_valid=xi[num:]
    yi_valid=labeli[num:] 
    num_all1+=len(xi_valid)
    #define model
    model1=Lasso(max_iter=max_iter1,alpha=alpha1)
    model1.fit(xi_train,yi_train)
    y_pred=model.predict(xi_valid)
    #calculate cost error
    err1+=np.sum((y_pred-yi_valid)**2)    

f1=err1/((num_all1)*2)
'''
#get none-clustered-data
xi_nan=tst_yu4chu4li2_04.x_nan2
labeli_nan=np.array(tst_yu4chu4li2_04.y_nan2_flt)
num=int(len(xi_nan)*split)
xi_train_nan=xi_nan[:num]
yi_train_nan=labeli_nan[:num]
#xi_valid_nan=xi_nan[num:]
#yi_valid_nan=labeli_nan[num:] 
#define model
model1=Lasso(max_iter=max_iter1,alpha=alpha1)
model1.fit(xi_train_nan,yi_train_nan)
y_pred_nan=model1.predict(xi_train_nan)
#calculate cost error
err1=np.sum((y_pred_nan-yi_train_nan)**2)  
f1=err1/(len(yi_train_nan)*2)
print('男性均方误差：',f1)