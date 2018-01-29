#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 16:36:13 2018

@author: ross
"""

import numpy as np
import utils
import boost_tree as bt
    
def build_gender_tree(xns, ys, num_stump, tag):    
    cols_dt = len(xns[0])
    rsdu = [y for y in ys]
    Txsm = []
    err_min0 = sum([abs(y) for y in rsdu])
    print("begin to plant tree: ...")
    for k in range(num_stump):
        err_min = err_min0
        Tx_min = []
        rsdu_min = []
        idx_err_min = 0
        ys_cur = [y for y in rsdu]
        for i in range(cols_dt):
            x1s = [x[i] for x in xns]
            Tx, err, rsdu = bt.boost_tree(x1s, ys_cur)
            if err<err_min:
                err_min = err
                Tx_min = [t for t in Tx]
                rsdu_min = [r for r in rsdu]
                idx_err_min = i
        Txsm += [Tx_min+[idx_err_min]]
        rsdu = [r for r in rsdu_min]
        print("%6s%4d: col%3d: %.4f (%.4f, %.4f, %.4f, %.4f, %.4f)"%( \
              tag, k, idx_err_min, err_min, \
              Tx[0], Tx[1], Tx[2], Tx[3], Tx[4]))
    np.save("../result/Txsm_%d_%s.npy"%(num_stump, tag), Txsm)  

###############################################################################
# load and preproc data:
dict_dt_filled = np.load("./result/dict_dt_filled.npy").tolist()

num_stump = 256

xs_male = dict_dt_filled["male"]
ys_male = dict_dt_filled["ys_male"]
tag_gender = "male"
build_gender_tree(xs_male, ys_male, num_stump, tag_gender)

print("################################  #  #############################")
print("################################  #  #############################")
print("################################  #  #############################")

xs_fema = dict_dt_filled["female"]
ys_fema = dict_dt_filled["ys_fema"]
tag_gender = "female"
build_gender_tree(xs_fema, ys_fema, num_stump, tag_gender)
