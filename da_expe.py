#!/usr/bin/env python
# coding: utf-8


# In[1]:

import os
os.environ["OMP_NUM_THREADS"] = "1"
from time import process_time as time
import ot
import ot.plot
import  da_screenkhorn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import argparse

def toy(n_samples_source,n_samples_target,nz=0.75,random_state=None):
    Xs, ys = ot.datasets.make_data_classif('3gauss', n_samples_source,nz=nz,random_state=random_state)
    Xt, yt = ot.datasets.make_data_classif('3gauss2', n_samples_target,nz=nz, random_state=random_state)
    return Xs, ys, Xt, yt

def subsample(x,y,n, nb_class=10):
    ind_rand = np.random.permutation(x.shape[0])
    x = x[ind_rand]
    y = y[ind_rand]
    x_r = np.empty((0,x.shape[1]))
    y_r = np.empty(0)
    for i in range(nb_class):
        ind = np.where(y==i)[0]
        ind_sub = ind[0:n]
        x_r = np.vstack((x_r,x[ind_sub]))
        y_r = np.hstack((y_r,y[ind_sub]))
    return x_r, y_r

parser = argparse.ArgumentParser()
parser.add_argument('-n', action='store', dest='n', default = 2000, type=int,
                        help='number of samples ')
parser.add_argument('-d', action='store', dest='d', default = 2, type=int,
                        help='dataset type ')
arguments = parser.parse_args()
n = arguments.n # number of samples per class


if arguments.d == 1:
    data = 'toy'
else:
    data = 'mnist'
    

n_samples_source = n
n_samples_target = n

nb_iter = 1
reg_cl = 10
K = 1 # K of KNN

pathres='./resultat/'
p_vec = [1.5,2,5,10,20,50,100]
n_pvec=  len(p_vec)
filename = 'da_{:}_n{:d}_regcl{:d}.npz'.format(data,n,reg_cl)

bc_none = np.zeros(nb_iter)
bc_sink = np.zeros(nb_iter)
bc_screen = np.zeros((nb_iter,n_pvec))
time_sink = np.zeros(nb_iter)   
time_screen = np.zeros((nb_iter,n_pvec))
print(filename)
for i in range(nb_iter):

    print('iter:',i)
    np.random.seed(i)

    if data =='toy':
        Xs,ys,Xt,yt = toy(n_samples_source,n_samples_target,nz=1,random_state=i)
    else:
        data = np.load('mnist_usps_feat10.npz')
        Xs,ys = subsample(data['X_s'], data['y_s'],n//10)
        Xt,yt = subsample(data['X_t'], data['y_t'],n//10)
        Xs=Xs
        Xt=Xt

    #%%
    clf_screened = KNeighborsClassifier(K)
    clf_screened.fit(Xs, ys)
    y_pred = clf_screened.predict(Xt)
    bc_none[i] = np.mean(y_pred==yt)


    #%% 
    # Sinkhorn Transport
    tic = time()
    ot_sinkhorn = ot.da.SinkhornLpl1Transport(reg_e=1,reg_cl=reg_cl)
    ot_sinkhorn.fit(Xs=Xs,ys= ys, Xt=Xt)
    time_sink[i] = time() - tic
    transp_Xs = ot_sinkhorn.transform(Xs=Xs)
    
    clf_screened = KNeighborsClassifier(K)
    clf_screened.fit(transp_Xs, ys)
    y_pred = clf_screened.predict(Xt)
    bc_sink[i] = np.mean(y_pred==yt)
    
    #%%
    
    for j,p_n in enumerate(p_vec):
        print(p_n)
        # Screenkhorn Transport
        tic = time()
        ot_screenkhorn = da_screenkhorn.ScreenkhornLpl1Transport(reg_e=1e0,reg_cl=reg_cl)
        ot_screenkhorn.fit(Xs=Xs,ys=ys, Xt=Xt, p_n=p_n, p_m=p_n)
       
        time_screen[i,j] = time() - tic
        # transport source samples onto target samples
        transp_Xs_screenkhorn = ot_screenkhorn.transform(Xs=Xs)
        
        clf_screened = KNeighborsClassifier(K)
        clf_screened.fit(transp_Xs_screenkhorn, ys)
        y_pred = clf_screened.predict(Xt)
        bc_screen[i,j] = np.mean(y_pred==yt)
    
#%%
    
    np.savez(pathres + filename, 
             bc_sink= bc_sink,  bc_screen= bc_screen,
             time_sink = time_sink,  time_screen = time_screen)
    
#%%
mean_screen_time = time_screen.mean(axis=0)
for i,p in enumerate(p_vec):
    print('gain dec {:1.2f}: {:2.2f}'.format(p,time_sink.mean()/mean_screen_time[i]))
        
print('perf none : {:2.2f} perf sink : {:2.2f}, perf screen {:2.2f}'.format(bc_none.mean()*100,bc_sink.mean()*100,bc_screen.mean()*100))
