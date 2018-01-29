#coding:utf-8
#autor:Oliver
import tensorflow as tf
import numpy as np

import tst_yu4chu4li2_04
xi_nv=tst_yu4chu4li2_04.x_nv2.astype(np.float32)
labeli_nv=np.array(tst_yu4chu4li2_04.y_nv2_flt).astype(np.float32)
labeli_nv=labeli_nv.reshape(len(labeli_nv),1)
xi_nan=tst_yu4chu4li2_04.x_nan2.astype(np.float32)
labeli_nan=np.array(tst_yu4chu4li2_04.y_nan2_flt).astype(np.float32)
labeli_nan=labeli_nan.reshape(len(labeli_nan),1)
batch_size = 50

def data_iter(X,y):
    num=X.shape[0]
    idx=list(range(num))
    for i in range(0, num, batch_size):
        j = np.array(idx[i:min(i+batch_size,num)])
        yield np.take(X,j,axis=0), np.take(y,j,axis=0)

X = tf.placeholder(dtype=tf.float32,shape=[None, 38])  
Y = tf.placeholder(dtype=tf.float32,shape=[None, 1])    
def model(X, w, b):  
    return tf.matmul(X, w)+b   
  
w = tf.Variable(tf.zeros([38,1]))  
b = tf.Variable(tf.zeros([1]))  
y_pred = model(X, w, b)  
  
cost = tf.reduce_mean(tf.square(Y - y_pred))/2  
  
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)  
  
with tf.Session() as sess:  
    init = tf.global_variables_initializer()
    sess.run(init)  
    step=0
    epoch=100
    for i in range(epoch):
        for xi,yi in data_iter(xi_nan,labeli_nan):
            sess.run(train_op, feed_dict={X: xi, Y: yi})  
            step+=1
            print('step ',step,'cost: ',sess.run(cost,feed_dict={X: xi, Y: yi}))
    w_train=sess.run(w)
    b_train=sess.run(b)
    
with tf.Session() as sess:       
    W_test = tf.placeholder(tf.float32)  
    B_test = tf.placeholder(tf.float32)  
    X_test = tf.placeholder(tf.float32)  
    Y_test = tf.placeholder(tf.float32)  
    pred = model(X_test,W_test,B_test)
    loss = tf.reduce_mean(tf.pow(pred-Y_test,2))/2 
    loss = sess.run(loss,{X_test:xi_nan,Y_test:labeli_nan,W_test:w_train,B_test:b_train})  
    print('最终误差:',loss)
