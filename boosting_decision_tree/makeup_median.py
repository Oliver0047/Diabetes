#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 16:57:31 2018

@author: ross
"""

import numpy as np
import statistics
import utils

def get_filled_values_median(xs, tag_gender): 
    params_nrm = np.load("./result/params_nrm_%s.npy"%tag_gender)
    xns = [[(l-param[0])/param[1] if l!=None else None \
               for param,l in zip(params_nrm, x)] for x in xs]
    num_col = len(xns[0])
    xns_rvs = [[x[i] for x in xns if x[i]!=None] for i in range(num_col)]
    vals_fill_med = [statistics.median(x) for x in xns_rvs]
    return (vals_fill_med,xns)

def fill_with_median_vals(xs, tag_gender):
    vals_filled,xns = get_filled_values_median(xs, tag_gender)
    xs_filled = []
    for lk in xns:
        lk_filled = [vals_filled[i] if l==None else l for i,l in enumerate(lk)]
        xs_filled += [lk_filled]
    return xs_filled
    
name_file = "E://diabetes/data/diabetes_original.db"
name_table = "train"
dt_full, dt_lack = utils.load_data(name_file, name_table)
dt = dt_full+dt_lack

dt_male = [[d[2]]+list(d[4:]) for d in dt if d[1]=="男"]
xs_male = [d[:-1] for d in dt_male]
ys_male = [d[-1] for d in dt_male]

dt_fema = [[d[2]]+list(d[4:]) for d in dt if d[1]=="女"]
xs_fema = [d[:-1] for d in dt_fema]
ys_fema = [d[-1] for d in dt_fema]

dict_dt_filled = {}
dict_dt_filled["ys_male"] = ys_male
dict_dt_filled["ys_fema"] = ys_fema

xs = xs_male
tag_gender = "male"
xs_filled = fill_with_median_vals(xs, tag_gender)
dict_dt_filled[tag_gender] = xs_filled

xs = xs_fema
tag_gender = "female"
xs_filled = fill_with_median_vals(xs, tag_gender)
dict_dt_filled[tag_gender] = xs_filled

np.save("./result/dict_dt_filled.npy", dict_dt_filled)

    

