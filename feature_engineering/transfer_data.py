#coding:utf-8
import pandas as pd
import sqlite3
conn=sqlite3.connect('diabetes.db')

df=pd.read_csv(u'd_train_20180102.csv',encoding='gbk')
df.to_sql('train', conn,index=False)

df=pd.read_csv(u'd_test_A_20180102.csv',encoding='gbk')
df.to_sql('test', conn,index=False)

conn.close()