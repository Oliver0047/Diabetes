#coding:utf-8
#autor:Oliver
import sqlite3
import random
import numpy as np
import matplotlib.pyplot as plt
random.seed(2018)

def missing_count(data):
    num=[]
    kind=[]
    for d in data:
        miss=[]
        for i in range(len(d)):
            if d[i] is None:
                miss.append(i)
        if miss not in kind and miss!=[]:
            kind.append(miss)
            num.append(1)
        elif miss in kind and miss!=[]:
            num[kind.index(miss)]+=1
    return (num,kind)    
    
#全部训练数据
file_train = "E://diabetes_original.db"
conn = sqlite3.connect(file_train)
curs = conn.cursor()
query = "select * from train"
curs.execute(query)
lst_dt = curs.fetchall()
conn.close()

#男女分类
data_nan = []
data_nv = []
for elm in lst_dt:
    if elm[1]=="男":
        data_nan.append(elm)
    if elm[1]=="女":
        data_nv.append(elm)

num1,kind1=missing_count(lst_dt)
num2,kind2=missing_count(data_nan)
num3,kind3=missing_count(data_nv)

for i in kind3:
    print(i)
plt.bar(np.arange(len(num3)),np.array(num3))
for x, y in zip(np.arange(len(num3)), np.array(num3)):
    plt.text(x, y, y, ha='center', va='bottom')
plt.show()
print('共',len(num3),'种情况')



