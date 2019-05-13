#!/usr/bin/env python
# coding: utf-8


# In[1]:


from time import process_time as time
import matplotlib.pylab as pl
import ot
import ot.plot

import  da_screenkhorn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


n_samples_source = 2000
n_samples_target = 2000

nb_iter = 10

bc_sink = np.zeros(nb_iter)
bc_screen = np.zeros(nb_iter)
time_sink = np.zeros(nb_iter)
time_screen = np.zeros(nb_iter)
for i in range(nb_iter):

    
    Xs, ys = ot.datasets.make_data_classif('3gauss', n_samples_source,nz=1,random_state=None)
    Xt, yt = ot.datasets.make_data_classif('3gauss2', n_samples_target,nz=1, random_state=None)
    M = ot.dist(Xs, Xt, metric='sqeuclidean')
    
    
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
    
    
    # Screenkhorn Transport
    tic = time()
    ot_screenkhorn = da_screenkhorn.ScreenkhornLpl1Transport(reg_e=1e0,reg_cl=10)
    ot_screenkhorn.fit(Xs=Xs,ys=ys, Xt=Xt, n_b=5, m_b=5)
    time_screen[i] = time() - tic
    # transport source samples onto target samples
    transp_Xs_screenkhorn = ot_screenkhorn.transform(Xs=Xs)
    
    clf_screened = KNeighborsClassifier(K)
    clf_screened.fit(transp_Xs_screenkhorn, ys)
    y_pred = clf_screened.predict(Xt)
    bc_screen[i] = np.mean(y_pred==yt)
    


