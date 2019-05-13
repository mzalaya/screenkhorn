#!/usr/bin/env python
# coding: utf-8

# 
# # OT for domain adaptation on empirical distributions
# 
# 
# This example introduces a domain adaptation in a 2D setting. It explicits
# the problem of domain adaptation and introduces some optimal transport
# approaches to solve it.
# 
# Quantities such as optimal couplings, greater coupling coefficients and
# transported samples are represented in order to give a visual understanding
# of what the transport methods are doing.
# 
# 

# In[1]:


# Authors: Remi Flamary <remi.flamary@unice.fr>
#          Stanislas Chambon <stan.chambon@gmail.com>
#
# License: MIT License

from time import process_time as time
import matplotlib.pylab as pl
import ot
import ot.plot

import domain_adaptation_maxime as da_screenkhorn
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


# NUMPY
import numpy as np
import matplotlib.pyplot as plt


# SEABORN 
import seaborn as sns


# generate data
# -------------


n_samples_source = 2000
n_samples_target = 2000

Xs, ys = ot.datasets.make_data_classif('3gauss', n_samples_source,nz=1,random_state=None)
Xt, yt = ot.datasets.make_data_classif('3gauss2', n_samples_target,nz=1, random_state=None)
M = ot.dist(Xs, Xt, metric='sqeuclidean')


#%% 
# Sinkhorn Transport
tic = time()
ot_sinkhorn = ot.da.SinkhornLpl1Transport(reg_e=1e0,reg_cl=10)
ot_sinkhorn.fit(Xs=Xs,ys= ys, Xt=Xt)
time_sink = time() - tic
transp_Xs = ot_sinkhorn.transform(Xs=Xs)

K = 3
clf_screened = KNeighborsClassifier(K)
clf_screened.fit(transp_Xs, ys)
y_pred = clf_screened.predict(Xt)
print('sink knn : ',np.mean(y_pred==yt))

# Sinkhorn Transport with Group lasso regularization
#ot_lpl1_sinkhorn = ot.da.SinkhornLpl1Transport(reg_e=1e0, reg_cl=1e1)
#ot_lpl1_sinkhorn.fit(Xs=Xs, ys=ys, Xt=Xt)

# transport source samples onto target samples
#transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=Xs)
#transp_Xs_lpl1_sinkhorn = ot_lpl1_sinkhorn.transform(Xs=Xs)

#%%


# Screenkhorn Transport
tic = time()
ot_screenkhorn = da_screenkhorn.ScreenkhornLpl1Transport(reg_e=1e0,reg_cl=10)
ot_screenkhorn.fit(Xs=Xs,ys=ys, Xt=Xt, n_b=5, m_b=5)
time_screen = time() - tic
# transport source samples onto target samples
transp_Xs_screenkhorn = ot_screenkhorn.transform(Xs=Xs)

clf_screened = KNeighborsClassifier(K)
clf_screened.fit(transp_Xs_screenkhorn, ys)
y_pred = clf_screened.predict(Xt)
print('s-wda knn : ',np.mean(y_pred==yt))


