#!/usr/bin/env python
# coding: utf-8


# In[1]:


from time import process_time as time
import matplotlib.pylab as plt
import ot
import ot.plot
import  da_screenkhorn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import argparse


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
    

n_samples_source = n
n_samples_target = n

nb_iter = 50
pathres='./resultat/'
p_vec = [1.5,2,5,10]
n_pvec=  len(p_vec)

bc_sink = np.zeros(nb_iter)
bc_screen = np.zeros((nb_iter,n_pvec))
time_sink = np.zeros(nb_iter)
time_screen = np.zeros((nb_iter,n_pvec))
for i in range(nb_iter):

    print('iter:',i)

    Xs, ys = ot.datasets.make_data_classif('3gauss', n_samples_source,nz=0.75,random_state=i)
    Xt, yt = ot.datasets.make_data_classif('3gauss2', n_samples_target,nz=0.75, random_state=i)
    M = ot.dist(Xs, Xt, metric='sqeuclidean')
    
    filename = 'da_{:}_{:d}'.format(data,n)
    #%% 
    # Sinkhorn Transport
    tic = time()
    ot_sinkhorn = ot.da.SinkhornLpl1Transport(reg_e=1e0,reg_cl=10)
    ot_sinkhorn.fit(Xs=Xs,ys= ys, Xt=Xt)
    time_sink[i] = time() - tic
    transp_Xs = ot_sinkhorn.transform(Xs=Xs)
    
    K = 3
    clf_screened = KNeighborsClassifier(K)
    clf_screened.fit(transp_Xs, ys)
    y_pred = clf_screened.predict(Xt)
    bc_sink[i] = np.mean(y_pred==yt)
    
    #%%
    
    for j,p_n in enumerate(p_vec):
        # Screenkhorn Transport
        tic = time()
        ot_screenkhorn = da_screenkhorn.ScreenkhornLpl1Transport(reg_e=1e0,reg_cl=10)
        ot_screenkhorn.fit(Xs=Xs,ys=ys, Xt=Xt, n_b=p_n, m_b=p_n)
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
        
print('perf sink : {:2.2f}, perf screen {:2.2f}'.format(bc_sink.mean(),bc_screen.mean()))
