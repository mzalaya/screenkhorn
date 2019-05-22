#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:12:46 2019

@author: GGilles
"""

# NUMPY
import numpy as np
from numpy.linalg import norm as norm
import scipy.stats as stats

np.random.seed(3946)

# MATPLOTLIB
import matplotlib.pyplot as plt
#from matplotlib import rc
#rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
#params = {'axes.labelsize': 34, # 12
#          'font.size': 24, # 12
#          'legend.fontsize': 22, # 12
#          'xtick.labelsize': 32, # 10
#          'ytick.labelsize': 32, # 10
#          #'text.usetex': True,
#          #'figure.figsize': (8, 6)
#          }
#plt.rcParams.update(params)
plt.close('all')

# SEABORN
import seaborn as sns

color_pal_t = sns.color_palette("colorblind", 11).as_hex()
color_pal = color_pal_t.copy()

colors = ["black", "salmon pink", "neon pink", "cornflower","cobalt blue"
          ,"blue green", "aquamarine", "bright yellow", "golden yellow", "reddish pink" , "reddish purple"]
color_pal = sns.xkcd_palette(colors)

# WARNINGS
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# TIME
from time import time 

# POT
from ot.datasets import make_data_classif
from ot import dist, sinkhorn

# SCREENKHORN
from screenkhorn import Screenkhorn

#%% Datasets
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

#%%
def toy(n_samples_source,n_samples_target,nz=0.75,random_state=None):
    Xs, ys = make_data_classif('3gauss', n_samples_source,nz=nz,random_state=random_state)
    Xt, yt = make_data_classif('3gauss2', n_samples_target,nz=nz, random_state=random_state)
    return Xs, ys, Xt, yt

#%%
def plot_mean_and_CI(pvect, mean, margin, color_mean=None, color_shading=None, label=''):
    plt.fill_between(pvect, mean+margin, mean-margin, 
                     color=color_shading, alpha=.15)
    plt.semilogx(pvect, mean, color_mean, marker='s', markersize=8, label=label)
    #plt.xlim(max(pvect), min(pvect))
    plt.xticks(pvect, pvect, rotation='vertical')

#%%
def compare_marginals(a, b, M, reg, decvect = [50, 10, 5, 2, 1.1]):
    n_decvect = len(decvect)
    rel_time_vect = np.empty(n_decvect)
    diff_a_vect   = np.empty(n_decvect)
    diff_b_vect   = np.empty(n_decvect)
    rel_cost_vect = np.empty(n_decvect)
    
    # sinkhorn 
    tic = time()
    P_sink = sinkhorn(a, b, M, reg)
    time_sink = time() - tic
    Pstar = P_sink[0]
    
    for j,p in enumerate(decvect):
        print('p:', p)
        # screenkhorn
        n_budget = int(np.ceil(a.shape[0]/p))
        m_budget = int(np.ceil(b.shape[0]/p))
        
        tic = time()
        screenkhorn = Screenkhorn(a, b, M, reg, n_budget, m_budget, verbose=False)
        lbfgsb = screenkhorn.lbfgsb()
        time_bfgs = time() - tic
        P_sc= lbfgsb[2]
        
        # screenkhorn marginals
        a_sc = P_sc @ np.ones(b.shape)
        b_sc = P_sc.T @ np.ones(a.shape)
        
        # comparisons
        rel_time_vect[j] = time_sink/time_bfgs
        diff_a_vect[j]   = norm(a - a_sc, ord=1)
        diff_b_vect[j]   = norm(b - b_sc, ord=1)
        rel_cost_vect[j] = np.abs(np.sum(M*(P_sc - Pstar)))/np.sum(M*Pstar)
        
    return diff_a_vect, diff_b_vect, rel_time_vect, rel_cost_vect

#%%
pathres= './resultat/'
nvect = [1000]
#decvect = [1.1, 1.25, 1.5, 2, 2.5, 5, 10, 20, 50, 100]
decvect = [1.1, 1.25, 2, 5, 10, 20, 50, 100]
regvect = [1e-1, 5e-1, 1, 10]
datatype = 'toy' # change to mnist to run on Mnist dataset
n_iter = 30 # we repeat n_iter times 
normalize = True

#%%
for n in nvect:
    print('n samples :',n)
    # filename
    if normalize:
        filename = 'norm_M_marginals_{:}_n{:d}'.format(datatype,n)
    else:
        filename = 'marginals_{:}_n{:d}'.format(datatype,n)
    
    M_diff_a = np.empty((n_iter, len(regvect), len(decvect)))
    M_diff_b = np.empty((n_iter, len(regvect), len(decvect)))
    M_time   = np.empty((n_iter, len(regvect), len(decvect)))
    M_cost   = np.empty((n_iter, len(regvect), len(decvect)))
    
    for i in range(n_iter):
        #np.random.seed(i)
        # gen data
        print('iter = ', i)
        if datatype =='toy':
            Xs,ys,Xt,yt = toy(n_samples_source=n, n_samples_target=n, nz=1, 
                              random_state=i)
        else:
            data = np.load('./data/mnist_usps_feat10.npz')
            Xs,ys = subsample(data['X_s'], data['y_s'],n//10)
            Xt,yt = subsample(data['X_t'], data['y_t'],n//10)
            
        # cost matrix
        M = dist(Xs, Xt)
        if normalize:
            M /= M.max()
        
        # Marginals
        a = np.ones(n)/n
        b = np.ones(n)/n
        
        for j, reg in enumerate(regvect):
            print('reg:', reg)
            d_av, d_bv, rel_timev, rel_costv = compare_marginals(a, b, M, reg, decvect)
            M_diff_a[i, j, :] = d_av
            M_diff_b[i, j, :] = d_bv
            M_time[i, j, :]   = rel_timev
            M_cost[i,j,:] = rel_costv

    np.savez(pathres + filename, 
             M_diff_a = M_diff_a,  M_diff_b = M_diff_b,
             M_time = M_time, M_cost = M_cost)


#%% Plots
pathfig = './figure/'
z_crit = stats.norm.ppf(q = 0.975)
markert = ['o','p','s','d','h','o','p','<','>','8','P']
colort = color_pal
for n in nvect:
    print('n samples :',n)
    # filename
    if normalize:
        filename = 'norm_M_marginals_{:}_n{:d}.npz'.format(datatype,n)
    else:            
        filename = 'marginals_{:}_n{:d}.npz'.format(datatype,n)

    np.load(pathres + filename)
    
    coeff = z_crit/np.sqrt(n_iter)    
    plt.close('all')
    
    plt.figure(1, figsize=(14, 8))
    plt.figure(2, figsize=(14, 8))
    plt.figure(3, figsize=(14, 8))
    plt.figure(4, figsize=(14, 8))
    
    for j, reg in enumerate(regvect):  
        # -------
        diff_a      = M_diff_a[:, j, :]
        diff_a_mean = diff_a.mean(axis=0)
        diff_a_std  = diff_a.std(axis=0)
        
        plt.figure(1)
        #plt.semilogx(decvect, diff_a_mean, marker='s', markersize=8, linewidth=3.5,
         #            label='$\eta = ${:}'.format(reg))
        plt.semilogx(decvect, diff_a_mean, lw = 2, marker = markert[j], 
                     markersize=12, c=colort[j], label='$\eta = ${:}'.format(reg))
        plt.fill_between(decvect, diff_a_mean+diff_a_std*coeff, 
                         diff_a_mean-diff_a_std*coeff, 
                         facecolor=colort[j], alpha=.15)
        plt.yscale('log')
        plt.xlabel(r'Decimation factor $n/n_b$', fontsize = 30)
        plt.ylabel(r'$\|\|\, \mu - \mu^{sc} \, \|\|_1$', fontsize = 30)
        plt.xticks(decvect, decvect, rotation='vertical', fontsize = 26)
        plt.yticks(fontsize=26)
        plt.title('$n=m=${:d}'.format(n), fontsize = 30)
        
        
        # -------
        diff_b      = M_diff_b[:, j, :]
        diff_b_mean = diff_b.mean(axis=0)
        diff_b_std  = diff_b.std(axis=0)
        
        plt.figure(2)
        plt.semilogx(decvect, diff_b_mean, lw = 2, marker = markert[j], 
                     markersize=12, c=colort[j], label='$\eta = ${:}'.format(reg))
        plt.fill_between(decvect, diff_b_mean+diff_b_std*coeff, 
                         diff_b_mean-diff_b_std*coeff, 
                         facecolor=colort[j], alpha=.15)
        
        plt.yscale('log')
        plt.xlabel(r'Decimation factor $m_b/m$', fontsize = 30)
        plt.ylabel(r'$\||\, \nu - \nu^{sc} \, \|\|_1$', fontsize = 30)
        plt.xticks(decvect, decvect, rotation='vertical', fontsize = 26)
        plt.yticks(fontsize=26)
        plt.title('$n=m=${:d}'.format(n), fontsize = 30)
        
        
        # -------
        rel_time      = M_time[:,j,:]
        rel_time_mean = rel_time.mean(axis=0)
        rel_time_std  = rel_time.std(axis=0)
        
        plt.figure(3)
        plt.semilogx(decvect, rel_time_mean, lw = 2, marker = markert[j], 
                     markersize=12, c=colort[j], label='$\eta = ${:}'.format(reg))
        plt.fill_between(decvect, rel_time_mean+rel_time_std*coeff, 
                         rel_time_mean-rel_time_std*coeff, 
                         facecolor=colort[j], alpha=.15)
        #plt.yscale('log')
        plt.xlabel(r'Decimation factor $n/n_b$', fontsize = 30)
        plt.ylabel('Running Time Gain', fontsize = 30)
        plt.xticks(decvect, decvect, rotation='vertical', fontsize = 26)
        plt.yticks(fontsize=26)
        plt.title('$n=m=${:d}'.format(n), fontsize = 30)
        
        # -------
        rel_cost     = M_cost[:,j,:]
        rel_cost_mean = rel_cost.mean(axis=0)
        rel_cost_std  = rel_cost.std(axis=0)
        
        plt.figure(4)
        plt.semilogx(decvect, rel_cost_mean, lw = 2, marker = markert[j], 
                     markersize=12, c=colort[j], label='$\eta = ${:}'.format(reg))
        plt.fill_between(decvect, rel_cost_mean+rel_cost_std*coeff, 
                         rel_cost_mean-rel_cost_std*coeff, 
                         facecolor=colort[j], alpha=.15)
        plt.yscale('log')
        plt.xlabel(r'Decimation factor $n/n_b$', fontsize = 30)
        plt.ylabel('Relative Divergence Variation', fontsize = 30)
        plt.xticks(decvect, decvect, rotation='vertical', fontsize = 26)
        plt.yticks(fontsize=26)
        plt.title('$n=m=${:d}'.format(n), fontsize = 30)
        
        
    plt.figure(1)
    plt.legend(fontsize=26), #plt.xticks(decvect, decvect, rotation='vertical')
    plt.grid(color='k', linestyle=':', linewidth=1,alpha=0.5)
    plt.figure(2)
    plt.legend(fontsize=26), #plt.xticks(decvect, decvect, rotation='vertical')
    plt.grid(color='k', linestyle=':', linewidth=1,alpha=0.5)
    plt.figure(3), plt.legend(fontsize=26), #plt.xticks(decvect, decvect, rotation='vertical')
    plt.grid(color='k', linestyle=':', linewidth=1,alpha=0.5)
    plt.figure(4), plt.legend(fontsize=26), #plt.xticks(decvect, decvect, rotation='vertical')
    plt.grid(color='k', linestyle=':', linewidth=1,alpha=0.5)

    if normalize:
        filename_fig_mu = 'norm_M_Mu_marginals_{:}_n{:d}.pdf'.format(datatype, n)
        filename_fig_nu = 'norm_M_Nu_marginals_{:}_n{:d}.pdf'.format(datatype, n)
        filename_fig_time = 'norm_M_time_{:}_n{:d}.pdf'.format(datatype,n)
        filename_fig_div = 'norm_M_div_{:}_n{:d}.pdf'.format(datatype,n)
    else:
        filename_fig_mu = 'Mu_marginals_{:}_n{:d}.pdf'.format(datatype, n)
        filename_fig_nu = 'Nu_marginals_{:}_n{:d}.pdf'.format(datatype, n)
        filename_fig_time = 'time_{:}_n{:d}.pdf'.format(datatype,n)
        filename_fig_div = 'div_{:}_n{:d}.pdf'.format(datatype,n)

    plt.figure(1)
    plt.savefig(pathfig+filename_fig_mu,   dpi=600, bbox_inches='tight')
    plt.figure(2)
    plt.savefig(pathfig+filename_fig_nu,   dpi=600, bbox_inches='tight')
    plt.figure(3)
    plt.savefig(pathfig+filename_fig_time, dpi=600, bbox_inches='tight')
    plt.figure(4)
    plt.savefig(pathfig+filename_fig_div,  dpi=600, bbox_inches='tight')
