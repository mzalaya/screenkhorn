#!/usr/bin/env python
# coding: utf-8

__author__ = 'Alain Rakotomamonjy'

import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import qr
sns.set_context("poster")
sns.set_style("ticks")
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from time import process_time as time
import argparse
import wda_screenkhorn as wda_screen
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

#%% parameters

def circle_data(n,nz = 0.2,nbnoise = 8):

# generate circle dataset
    t = np.random.rand(n) * 2 * np.pi
    ys = np.floor((np.arange(n) * 1.0 / n * 3)) + 1
    xs = np.concatenate(
        (np.cos(t).reshape((-1, 1)), np.sin(t).reshape((-1, 1))), 1)
    xs = xs * ys.reshape(-1, 1) + nz * np.random.randn(n, 2)
    
    t = np.random.rand(n) * 2 * np.pi
    yt = np.floor((np.arange(n) * 1.0 / n * 3)) + 1
    xt = np.concatenate(
        (np.cos(t).reshape((-1, 1)), np.sin(t).reshape((-1, 1))), 1)
    xt = xt * yt.reshape(-1, 1) + nz * np.random.randn(n, 2)
    
    
    xs = np.hstack((xs, np.random.randn(n, nbnoise)))
    xt = np.hstack((xt, np.random.randn(n, nbnoise)))

    return xs,xt,ys,yt

#%%
def multi_gauss(n,nb_noise=8,s_noise=0.1):
    centers = [(1, 0), (-1, 0), (0.4, 0.8), (-0.4, -0.8),(-0.4, 0.8), (0.4, -0.8)]
    x, y = make_blobs(n_samples=n, n_features=200, cluster_std= s_noise,
                  centers=centers, shuffle=False)
    y[y==0] = 1
    y[y==3] = 2
    y[ y==4] = 3
    y[ y==5] = 3
    x  = np.hstack((x, s_noise*np.random.randn(n, nb_noise)))
    return x, y

def balanced_mnist(n=100 , train = True):
    data = loadmat('mnist.mat')
    n = n // 10
    if train:
        x = data['xapp']
        y = np.array(data['yapp']).squeeze(axis=1)
        arr = np.arange(x.shape[0])
        x = x[arr]
        y[y==10]=0
        y = y[arr]
        x_r = np.empty((0,x.shape[1]))
        y_r = np.empty(0)
        for i in range(10):
            ind = np.where(y==i)[0]
            ind_sub = ind[0:n]
            x_r = np.vstack((x_r,x[ind_sub]))
            y_r = np.hstack((y_r,y[ind_sub]))
    
    else:
         x_r = data['xtest']
         y_r = np.array(data['ytest']).squeeze(axis=1)
    return x_r.astype(float), y_r.astype(float)

#%%


parser = argparse.ArgumentParser()
parser.add_argument('-n', action='store', dest='n', default = 1000, type=int,
                        help='number of samples ')
parser.add_argument('-d', action='store', dest='d', default = 1, type=int,
                        help='dataset type ')
arguments = parser.parse_args()
n = arguments.n # number of samples per class

if arguments.d == 1:
    data = 'toy'
else:
    data = 'mnist'
    

reg = 1  # regularizer weight
k = 10  # nb of sinkhorn iteration
maxiter = 1000 # max iter in WDA
nb_iter = 30
K = 5 # K in KNN
p_vec = [1.5,2,5,10,20,50,100]
pathres='./resultat/'



n_pvec=  len(p_vec)
time_wda = np.zeros((nb_iter))
time_swda = np.zeros((nb_iter,n_pvec))
bc_wda = np.zeros((nb_iter))
bc_swda = np.zeros((nb_iter,n_pvec))
for i in range(nb_iter):
    print('iter:',i)
    if data == 'toy':
        p = 2    # relevant dimensions
        nb_noise = 8
        xs, ys = multi_gauss(n,nb_noise = nb_noise)
        xt, yt = multi_gauss(n,nb_noise = nb_noise)
        dim = p + nb_noise
        filename = 'wda_{:}_n{:d}_p{:d}'.format(data,n,p)
        print(filename)

    else:
        xs, ys = balanced_mnist(n)
        xt, yt = balanced_mnist(train=False)
        scaler = StandardScaler()
        xs = scaler.fit_transform(xs)
        xt = scaler.transform(xt)
        dim = 784
        p = 20
        filename = 'wda_{:}_n{:d}_p{:d}'.format(data,n,p)
        print(filename)

    P_init = qr(np.random.randn(dim,p))[0]
    
    #%%
#    clf_wda = KNeighborsClassifier(n_neighbors = 3, metric='euclidean')
#    clf_wda.fit(xs, ys)
#    y_pred = clf_wda.predict(xt)
#    print('knn : ',np.mean(y_pred==yt))
    #%%
    tic = time()
    Pwda_sink, projwda_sink = wda_screen.wda_sinkhorn(xs, ys, p, reg, k, maxiter=maxiter,P0=P_init)
    time_wda[i] = time() -tic
    xtpw = projwda_sink(xt)                                                     
    xspw = projwda_sink(xs)  
    clf_wda = KNeighborsClassifier(K)
    clf_wda.fit(xspw, ys)
    y_pred = clf_wda.predict(xtpw)
    print('wda knn : ',np.mean(y_pred==yt))
    bc_wda[i]= np.mean(y_pred==yt)
    
    
    for j,p_n in enumerate(p_vec):
        tic = time()
        Pwda_screen, projwda_screen = wda_screen.wda_screenkhorn(xs, ys, p, reg, k, solver=None, maxiter=maxiter,
                                                                p_n=p_n, p_m=p_n,P0=P_init)
        time_swda[i,j] = time() -tic
    
        xspw_screen = projwda_screen(xs)
        xtpw_screen = projwda_screen(xt)
        clf_screened = KNeighborsClassifier(K)
        clf_screened.fit(xspw_screen, ys)
        y_pred = clf_screened.predict(xtpw_screen)
        print('s-wda knn : ',np.mean(y_pred==yt))
        bc_swda[i,j]= np.mean(y_pred==yt)

    np.savez(pathres + filename, 
             bc_wda = bc_wda,  bc_swda = bc_swda,
             time_wda = time_wda,  time_swda = time_swda)
    


