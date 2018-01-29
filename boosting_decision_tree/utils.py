#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 19:42:31 2018

@author: ross
"""

import sqlite3
import numpy as np
#import random
from sklearn import decomposition

def load_data(name_file, name_table):
    #file_train = "./data/diabetes.db"
    conn = sqlite3.connect(name_file)
    curs = conn.cursor()
    
    query = "select * from %s"%name_table
    curs.execute(query)
    lst_dt = curs.fetchall()
    
    conn.close()
    
    lst_dt_no_none = []
    lst_dt_none = []
    for elm in lst_dt:
        flag_none = 0
        for elm1 in elm:
            if elm1 is None:
                flag_none = 1
                break
        if flag_none is 0:
            lst_dt_no_none += [elm]
        if flag_none is 1:
            lst_dt_none += [elm]
            
#    random.shuffle(lst_dt_no_none)
#    random.shuffle(lst_dt_none)
    
    return lst_dt_no_none, lst_dt_none

def preproc(lst_dt):
    dt_male = [[d[2]]+list(d[4:]) for d in lst_dt if d[1]=="男"]
    dt_fema = [[d[2]]+list(d[4:]) for d in lst_dt if d[1]=="女"]

    dt_male = [elm for elm in dt_male if elm[0]>=20]
    dt_fema = [elm for elm in dt_fema if elm[0]>=20]

    ###########################################################################
    # filter 1 isolated point:    
    x_male = [elm[:-1] for elm in dt_male]
    x_fema = [elm[:-1] for elm in dt_fema]
    
    x_male = norm_column(x_male, "male")
    x_fema = norm_column(x_fema, "female") 

    idx = pca_filter(x_male)
    if idx>0:
        del dt_male[idx]    
    idx = pca_filter(x_fema)
    if idx>0:
        del dt_fema[idx]
   
    return dt_male, dt_fema

def make_column_norm_param(xs, tag_gender):
    num_col = len(xs[0])
    params_nrm = []
    for i in range(num_col):
        dtcl = [x[i] for x in xs if x[i]!=None]
        param = [np.mean(dtcl), np.std(dtcl), np.max(dtcl), np.min(dtcl)]
    #    param = [round(p, 4) for p in param]
        params_nrm += [param]
    np.save("../result/params_nrm_%s.npy"%tag_gender, params_nrm)
    print("%s normalization parameters is established !"%(tag_gender))
    
def norm_column(xs, tag_gender):
    xs_arr = np.array(xs)
    rw_x, cl_x = xs_arr.shape
    xns = np.empty((rw_x, cl_x))
#    xns[:] = np.NAN
    params_nrm = np.load("../result/params_nrm_%s.npy"%tag_gender)
    for i in range(cl_x):
        xns[:, i] = (xs_arr[:, i]-params_nrm[i][0])/params_nrm[i][1] 
    return xns.tolist()

def pca_filter(xs): 
    thresh_distance_isolated = 8
    pca = decomposition.PCA(n_components=3)
    pca.fit(xs)
    # x3: x's dimension will become 3.
    x3 = pca.transform(xs)
    dst_x3 = [sum(abs(elm)) for elm in x3]
    idx_max = np.argmax(dst_x3)
    val_max = np.max(dst_x3)
    if val_max/np.mean(dst_x3)>thresh_distance_isolated:
        print("del 1 isolated point %d: %f"%(idx_max, val_max))
        return idx_max
    else:
        print("del 0 isolated point!")
        return -1

def sort_2column(xys):
    xysr = np.array(xys)
    idxs_sort = np.argsort(xysr[0, :])
    return xysr[:, idxs_sort].tolist()

def verify_data(xns, ys, tag_gender):   
    num_flts = 32
    ys_pred = test_data(xns, num_flts, tag_gender)
    mse = np.mean([(y-y_pred)**2 for y,y_pred in zip(ys, ys_pred)])
    print("%6s: filter number:%4d MSE: %.4f"%(tag_gender, num_flts, mse))
    
def test_data(xns, num_flts, tag_gender):    
    Txsm = np.load("../result/Txsm_%d_%s.npy"%(num_flts, tag_gender)).tolist()
    ys_pred = []
    for xn in xns:
        parts = 0
        for tx in Txsm:
            idxc = round(tx[5])
            parts += tx[0]+tx[1]*xn[idxc] if xn[idxc]<=tx[4] \
                    else tx[2]+tx[3]*xn[idxc] 
        ys_pred += [parts]
    return ys_pred
    
  