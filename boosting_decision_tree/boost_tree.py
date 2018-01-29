#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 22:38:35 2018

@author: ross
"""
import numpy as np

def boost_tree(xs, ys):
# xys: input data, nX2
# depth: the number of the requested classifiers
    ys_org = [r for r in ys]
    len_xys = len(xs)
    
    fxs = [0]*len_xys
###############################################################################
# boost:
    cs, idx_min = find_stump(xs, ys)
    Tx = (cs[0], cs[1], cs[2], cs[3], xs[idx_min])
    if Tx[2]==None:
        fx1s = [Tx[0]+Tx[1]*x for x in xs]
    else:
        fx1s = [Tx[0]+Tx[1]*x if x<=Tx[4] else Tx[2]+Tx[3]*x for x in xs]
    fxs = [fx1+fx for fx1, fx in zip(fx1s, fxs)]
    # rsdu: residual
    rsdu = [y-f for y,f in zip(ys_org, fxs)]
    err = np.mean([r**2 for r in rsdu])
#    print("error: %f"%err)
        
    return Tx, err, rsdu

def find_stump(xs, rsdu):
    mss = []
    cs = []
    for xi in xs:
        xs1 = [x for x in xs if x<=xi]
        ys1 = [y for x, y in zip(xs, rsdu) if x<=xi]       
        xs2 = [x for x in xs if x> xi]
        ys2 = [y for x, y in zip(xs, rsdu) if x> xi]
        
        a1, b1, err1 = least_square_estimator(xs1, ys1) 
        a2, b2, err2 = least_square_estimator(xs2, ys2)         
        mss += [err1+err2]
        cs += [(a1, b1, a2, b2)]
        
    idx_min = np.argmin(mss)
    return cs[idx_min], idx_min

def least_square_estimator(xs, ys):
    thrsh_same = 0.0001
    if ys==[]:
        a = b = None
        err = 0
        return a, b, err
    elif np.std(xs)<thrsh_same:
        a = np.mean(ys)
        b = 0
        err = 0
        return a, b, err

    sum_x2 = sum([x*x for x in xs])
    sum_x = sum([x for x in xs])
    sum_y = sum([y for y in ys])
    sum_xy = sum([x*y for x, y in zip(xs, ys)])
    n = len(xs)
    
    # estimator: y = a+bx
    try:
        a = (sum_x2*sum_y-sum_x*sum_xy)/(n*sum_x2-sum_x**2)
        b = (n*sum_xy-sum_x*sum_y)/(n*sum_x2-sum_x**2)
        errs = [y-(a+b*x) for x, y in zip(xs, ys)]
    except ZeroDivisionError:
        print("LSE ZeroDivisionError ! ")
#        print("xs: %d, %.4f, %.4f"%(len(xs), np.min(xs), np.max(xs)))
        print(" ".join("%.4f"%x for x in xs))        
        a = np.mean(ys)
        b = 0
        errs = [y-a for y in ys]

    err = sum([er**2 for er in errs])
    return a, b, err


