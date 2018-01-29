#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 13:09:40 2018

@author: ross
"""

import sqlite3
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors  
random.seed(2018)
###############################################################################
# read from sqlite3

file_train = "E://diabetes/data/diabetes_regression.db"
conn = sqlite3.connect(file_train)
curs = conn.cursor()
query = "select * from train"
curs.execute(query)
lst_dt = curs.fetchall()
conn.close()

###############################################################################
# divide two parts: no none data and data including none

lst_dt_nonone = []
lst_dt_none = []
for elm in lst_dt:
    flag_none = 0
    for elm1 in elm:
        if elm1 is None:
            flag_none = 1
            break
    if flag_none is 0:
        lst_dt_nonone += [elm]
    if flag_none is 1:
        lst_dt_none += [elm]
        
random.shuffle(lst_dt_nonone)
random.shuffle(lst_dt_none)

###############################################################################
# divide two parts: man and woman
only=list(lst_dt_none[0])
only[1]='女'
only[19:24]=[ 0.71667448,  3.76816773,  0.04696159,  1.84564698,  1.88477636]

x_nonone_nan2 = []
x_nonone_nv2 = []
for elm in lst_dt_nonone:
    if elm[1]=="男":
        x_nonone_nan2 += [[elm[2]]+list(elm[4:])]
    if elm[1]=="女":
        x_nonone_nv2 += [[elm[2]]+list(elm[4:])]

x_nonone_nv2 += [[only[2]]+list(only[4:])]
###############################################################################
#use KNN to divide the none-sex data
#import matplotlib.pyplot as plt
#X=[x_nonone_nan2,x_nonone_nv2]
#X=np.concatenate(X,0)
#only= [lst_dt_none[0][2]]+list(lst_dt_none[0][4:])
#only=np.array(only).reshape(1,39).astype(np.float64)
#nmiss=list(np.arange(39))
#for i in range(16,21):
#    nmiss.remove(i)
#X1=X[:,nmiss]
#only1=only[:,nmiss]
#nbrs = NearestNeighbors(n_neighbors=10, algorithm="auto").fit(X1)
#distances, indices = nbrs.kneighbors(only1)  
#sel=list(indices[0][:2])
#weights=list(distances[0][:2])
#s=weights[0]+weights[1]
#weights=list(weights/s)
#fill=X[sel[0],16:21]*weights[0]+X[sel[1],16:21]*weights[1]
#only[:,16:21]=fill[0:5]
#plt.plot(np.arange(0,10),distances.reshape(10))
###############################################################################
# divide two parts: man and woman       
#import matplotlib.pyplot as plt
# 
##x = [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]
#
#x_nan2 = [elm[0] for elm in x_nonone_nan2]
#x_nv2 = [elm[0] for elm in x_nonone_nv2]
#num_bins = 100
#
#plt.figure(figsize=(20, 10))
#
#plt.subplot(211)
#n, bins, patches = plt.hist(x_nan2, num_bins, facecolor='blue', alpha=0.5)
#plt.xlim(0, 100)
#
#plt.subplot(212)
#n, bins, patches = plt.hist(x_nv2, num_bins, facecolor='blue', alpha=0.5)
#plt.xlim(0, 100)
#
#plt.savefig('../test/hist_nan2.png')

###############################################################################
# filter age: 20 - 100
#x_nv2_flt = []
#y_nv2_flt = []
#len_x_nv2 = len(x_nonone_nv2)
#for i in range(len_x_nv2):
#    if x_nonone_nv2[i][0]>=20:
#        x_nv2_flt[i] += [x_nonone_nv2[i]]

xy_nv2_flt = [elm for elm in x_nonone_nv2 if elm[0]>=20]
xy_nan2_flt = [elm for elm in x_nonone_nan2 if elm[0]>=20]
###############################################################################
# filter the odd value from pca graph

del xy_nv2_flt[1508]
del xy_nv2_flt[602]
del xy_nv2_flt[1956]#乙肝e抗原特别大
del xy_nv2_flt[1442]#血糖特别大

del xy_nan2_flt[506]

###############################################################################
# normalize every column
x_nv2_flt = [elm[:-1] for elm in xy_nv2_flt]
y_nv2_flt = [elm[-1] for elm in xy_nv2_flt]
x_nv2 = np.array(x_nv2_flt)
rw_x, cl_x = x_nv2.shape
mean_nv=np.mean(x_nv2,0)
std_nv=np.std(x_nv2,0)
for i in range(cl_x):
    x_nv2[:, i] = (x_nv2[:, i]-np.mean(x_nv2[:, i]))/np.std(x_nv2[:, i])

x_nan2_flt = [elm[:-1] for elm in xy_nan2_flt]
y_nan2_flt = [elm[-1] for elm in xy_nan2_flt]
x_nan2 = np.array(x_nan2_flt)
rw_x, cl_x = x_nan2.shape
mean_nan=np.mean(x_nan2,0)
std_nan=np.std(x_nan2,0)
for i in range(cl_x):
    x_nan2[:, i] = (x_nan2[:, i]-np.mean(x_nan2[:, i]))/np.std(x_nan2[:, i])
###############################################################################
# pca imshow
'''
from sklearn import decomposition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1, figsize=(20, 10))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=0, azim=0)#48 134

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(x_nan2)
x_dim3 = pca.transform(x_nan2)
y = np.array(y_nan2_flt)

ax.scatter(x_dim3[:, 0], x_dim3[:, 1], x_dim3[:, 2], s=y*60, alpha=0.6, 
           cmap=plt.cm.spectral, edgecolor='k')

plt.savefig('try.png')
'''

