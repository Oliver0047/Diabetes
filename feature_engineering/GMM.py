#coding:utf-8
#autor:Oliver
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import tst_yu4chu4li2_04
import random
from sklearn.metrics import silhouette_score
random.seed(2018)
best_score=0
ori_X=tst_yu4chu4li2_04.x_nv2
ori_label=np.array(tst_yu4chu4li2_04.y_nv2_flt)
for cluster in range(2,33):
    # Fit a Gaussian mixture with EM using five components
    gmm = mixture.GaussianMixture(n_components=cluster, covariance_type='full',random_state=2018).fit(ori_X)
    re = gmm.predict(ori_X)
    '''
    # Fit a Dirichlet process Gaussian mixture using five components
    dpgmm = mixture.BayesianGaussianMixture(n_components=5,
                                           covariance_type='full').fit(X)
    '''
    silhouette_avg = silhouette_score(ori_X,re)
    if silhouette_avg > best_score:
        best_clusterer = gmm
        best_score = silhouette_avg
        best_cluster = cluster
#n, bins, patches = plt.hist(best_clusterer.predict(X), best_cluster, facecolor='blue', alpha=0.5)
Y=best_clusterer.predict(ori_X)
kind=set(Y)
X=[]
label=[]
for i in kind:
    X.append(ori_X[np.argwhere(Y==i)][:,0,:])
    label.append(ori_label[np.argwhere(Y==i)][:,0])
    
best_score=0
ori_X=tst_yu4chu4li2_04.x_nan2
ori_label=np.array(tst_yu4chu4li2_04.y_nan2_flt)
for cluster in range(2,33):
    # Fit a Gaussian mixture with EM using five components
    gmm = mixture.GaussianMixture(n_components=cluster, covariance_type='full',random_state=2018).fit(ori_X)
    re = gmm.predict(ori_X)
    '''
    # Fit a Dirichlet process Gaussian mixture using five components
    dpgmm = mixture.BayesianGaussianMixture(n_components=5,
                                           covariance_type='full').fit(X)
    '''
    silhouette_avg = silhouette_score(ori_X,re)
    if silhouette_avg > best_score:
        best_clusterer1 = gmm
        best_score = silhouette_avg
        best_cluster1 = cluster
#n, bins, patches = plt.hist(best_clusterer.predict(X), best_cluster, facecolor='blue', alpha=0.5)
Y=best_clusterer1.predict(ori_X)
kind=set(Y)
X1=[]
label1=[]
for i in kind:
    X1.append(ori_X[np.argwhere(Y==i)][:,0,:])
    label1.append(ori_label[np.argwhere(Y==i)][:,0])