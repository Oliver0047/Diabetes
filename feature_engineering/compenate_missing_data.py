#coding:utf-8
#autor:Oliver
import sqlite3
import numpy as np
import random
from sklearn.linear_model import Lasso
random.seed(2018)

#read data from sqlite3
original='E://diabetes/diabetes_original.db'
now = "E://diabetes/data/diabetes.db"
conn = sqlite3.connect(now)
curs = conn.cursor()
query = "select * from train"
curs.execute(query)
lst_dt = curs.fetchall()

#split data by col(none or no-none)
col=23
index=[]
data_miss=[]
data_nomiss=[]
for i in range(len(lst_dt)):
    d=lst_dt[i]
    if d[col] is None:
        index.append(d[0])
        data_miss.append(d)
    else:
        data_nomiss.append(d)

#select data have features with none data from no-none data------------->train data
left=24
right=41
left1=4
right1=18
nomiss=[]
for i in range(len(data_nomiss)):
    d=data_nomiss[i]
    label=0
    for j in range(left,right+1):
        if d[j] is None:
            label=1
            break
    for j in range(left1,right1+1):
        if d[j] is None:
            label=1
            break
    if label==0:
        nomiss.append(d)
        
#split by sex ------------->train data          
nomiss_nan_x=[]
nomiss_nan_y=[]
nomiss_nv_x=[] 
nomiss_nv_y=[]  
for elm in nomiss:
    if elm[1]=='男':
        nomiss_nan_x+=[[elm[2]]+list(elm[left1:(right1+1)])+list(elm[left:(right+1)])]
        nomiss_nan_y+=[[elm[col]]]
    elif elm[1]=='女':
        nomiss_nv_x+=[[elm[2]]+list(elm[left1:(right1+1)])+list(elm[left:(right+1)])]
        nomiss_nv_y+=[[elm[col]]]

#Package to array------------->train data
nomiss_nan_x=np.array(nomiss_nan_x)
nomiss_nan_y=np.array(nomiss_nan_y)
nomiss_nv_x=np.array(nomiss_nv_x)
nomiss_nv_y=np.array(nomiss_nv_y)

#data standardization------------->train data
col_num=34
mean_nan=np.mean(nomiss_nan_x,0).reshape(1,col_num)
std_nan=np.std(nomiss_nan_x,0).reshape(1,col_num)
nomiss_nan_x=(nomiss_nan_x-mean_nan)/std_nan

mean_nv=np.mean(nomiss_nv_x,0).reshape(1,col_num)
std_nv=np.std(nomiss_nv_x,0).reshape(1,col_num)
nomiss_nv_x=(nomiss_nv_x-mean_nv)/std_nv

#sex split and data standardization------------->test data
miss_nan_x=[]
index_nan=[]
miss_nv_x=[]
index_nv=[]
for i in range(len(data_miss)):
    elm=data_miss[i]
    j=index[i]
    if elm[1]=='男':
        miss_nan_x+=[[elm[2]]+list(elm[left1:(right1+1)])+list(elm[left:(right+1)])]
        index_nan.append(j)
    elif elm[1]=='女':
        miss_nv_x+=[[elm[2]]+list(elm[left1:(right1+1)])+list(elm[left:(right+1)])]  
        index_nv.append(j)
miss_nan_x=(np.array(miss_nan_x)-mean_nan)/std_nan
miss_nv_x=(np.array(miss_nv_x)-mean_nv)/std_nv

#train model
model_nan=Lasso(max_iter=10000,alpha=0.01)
model_nan.fit(nomiss_nan_x,nomiss_nan_y)
pred_nan=model_nan.predict(miss_nan_x)
pred_nan[np.argwhere(pred_nan<0)]=0

model_nv=Lasso(max_iter=10000,alpha=0.01)
model_nv.fit(nomiss_nv_x,nomiss_nv_y)
pred_nv=model_nv.predict(miss_nv_x)
pred_nv[np.argwhere(pred_nv<0)]=0
#update data in sqlite3
col_name='乙肝核心抗体'
for i in range(len(pred_nan)):
    query = "update train set %s=%f where id=%d"%(col_name,pred_nan[i],index_nan[i])
    curs.execute(query)
for i in range(len(pred_nv)):
    query = "update train set %s=%f where id=%d"%(col_name,pred_nv[i],index_nv[i])
    curs.execute(query)
conn.commit()#提交结果，不然修改无效
conn.close()

#缺漏值预测误差评估

pred=model_nan.predict(nomiss_nan_x).reshape(len(nomiss_nan_y),1)
err=np.sum((pred-nomiss_nan_y)**2)
pred1=model_nv.predict(nomiss_nv_x).reshape(len(nomiss_nv_y),1)
err1=np.sum((pred1-nomiss_nv_y)**2)
f=(err+err1)/((len(pred)+len(pred1))*2)
print(f)