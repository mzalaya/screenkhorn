#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:11:50 2019

@author: alain
"""


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

color_pal_t = sns.color_palette("colorblind", 11).as_hex()
color_pal = color_pal_t.copy()


colors = ["black", "salmon pink", "neon pink", "cornflower","cobalt blue"
          ,"blue green", "aquamarine", "dark yellow", "golden yellow", "reddish pink" , "reddish purple"]

color_pal = sns.xkcd_palette(colors)

plt.close("all")

pathres = './resultat/da/'

data = 'mnist'
if data == 'toy':
    n_vec = [200,500,1000,2000,3000,5000]
    nb_p_vec = 7
    p = 2 
    regcl = 100
else:
    data = 'mnist'
    n_vec = [200,500,1000,2000,3000,4000]
    nb_p_vec = 7
    p = 20
    regcl = 10

method_vec = ['wda', 'swda']
legend_vec_accur = ['Sinkhorn','dec=1.5','dec=2','dec=5','dec=10','dec=20','dec=50','dec=100','No Adapt']
legend_vec_time = ['dec=1.5','dec=2','dec=5','dec=10','dec=20','dec=50','dec=100']
t = 1
in_sample = 0

Mres = np.zeros((len(n_vec), nb_p_vec+2))
Sres = np.zeros((len(n_vec), nb_p_vec+2))

Mtime = np.zeros((len(n_vec), nb_p_vec+1))
Stime = np.zeros((len(n_vec), nb_p_vec+1))

Mgain = np.zeros((len(n_vec), nb_p_vec+1))
Sgain = np.zeros((len(n_vec), nb_p_vec+1))

for i_k, n in enumerate(n_vec):                
    if data == 'toy':
        filename = 'da_{:}_n{:d}_regcl{:2.2f}'.format(data,n,regcl)
    else:
        filename = 'da_{:}_n{:d}_regcl{:d}'.format(data,n,regcl)

    # reading files and performance
    res = np.load(pathres + filename + '.npz')
    
    
    
    
    aux = res['bc_sink']
    nb_computed = np.where(aux)[0].shape[0]
    print(nb_computed)
    print(' \t\t {:2.2f} \t\t{:d}'.format( aux[np.where(aux)].sum()/nb_computed*100, nb_computed ))
    Mres[i_k,0] = aux[np.where(aux)].mean()
    Sres[i_k,0] = aux[np.where(aux)].std()
    
    aux = res['bc_screen']
    nb_computed = np.where(np.sum(aux,axis=1))[0].shape[0]
    print(nb_computed) 
    mean_perf = aux[np.where(np.sum(aux,axis=1))[0]].mean(axis=0)
    std_perf = aux[np.where(np.sum(aux,axis=1))[0]].std(axis=0)
    Mres[i_k,1:nb_p_vec+1] = mean_perf # stocking average perf for all decimation
    Sres[i_k,1:nb_p_vec+1] = std_perf
    
        
    aux = res['bc_none']
    nb_computed = np.where(aux)[0].shape[0]
    print(nb_computed)
    print(' \t\t {:2.2f} \t\t{:d}'.format( aux[np.where(aux)].sum()/nb_computed*100, nb_computed ))
    Mres[i_k,nb_p_vec+1] = aux[np.where(aux)].mean()
    Sres[i_k,nb_p_vec+1] = aux[np.where(aux)].std()
    
    
    # reading computation time
    aux = res['time_sink']
    nb_computed = np.where(aux)[0].shape[0]
    print(nb_computed)
    print(' \t\t {:2.2f} \t\t{:d}'.format( aux[np.where(aux)].sum()/nb_computed*100, nb_computed ))
    Mtime[i_k,0] = aux[np.where(aux)].mean()
    Stime[i_k,0] = aux[np.where(aux)].std()
    
    
    
    # computing gain 
    aux2 = res['time_screen']
    aux1 = aux.reshape(-1,1)/res['time_screen']
    
    
    nb_computed = np.where(np.isnan(np.sum(aux1,axis=1)) == False)[0].shape[0]
    print(nb_computed) 
    mean_perf = aux1[ np.where(np.isnan(np.sum(aux1,axis=1)) == False)[0]].mean(axis=0)
    std_perf = aux1[np.where(np.isnan(np.sum(aux1,axis=1)) == False)[0]].std(axis=0)
    # keeping gain for all p 
    Mtime[i_k,1:] = aux2[np.where(np.sum(aux2,axis=1))].mean(axis=0)
    Stime[i_k,1:] =aux2[np.where(np.sum(aux2,axis=1))].std(axis=0)
    
    Mgain[i_k,1:] = mean_perf
    Sgain[i_k,1:] = std_perf





#%% figure for accuracy
    
plt.figure(0)
ax = []
markert = ['o','p','s','d','h','o','p','<','>','8','P']
colort = color_pal
for i in range(nb_p_vec+2):
    ax1, = plt.plot(n_vec, Mres[:,i],label=str(i), lw = 2, marker = markert[i], markersize=10,
                    c=colort[i])
    #error=Sres[:,i]
    #plt.fill_between(n_vec, Mres[:,i]-error, Mres[:,i]+error, color=colort[i],alpha = 0.1)
    ax.append(ax1)

if t==1:
    plt.ylim((0.4,1))
else:
    plt.ylim((0.4,1))




plt.xlabel('Number of samples', fontsize = 16)
plt.ylabel('Accuracy', fontsize = 16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(legend_vec_accur)
plt.title('OTDA+Knn on {:}'.format(data))
plt.grid(color='k', linestyle=':', linewidth=1,alpha=0.5)
filename = 'da_accur_{:}_regcl{:d}.pdf'.format(data,regcl)
plt.savefig('figure/' + filename,dpi=600)


# %% figure for gain


plt.figure(1)

for i in range(1,nb_p_vec+1):
    ax1, = plt.plot(n_vec, Mgain[:,i],label=str(i), lw = 2, marker = markert[i], markersize=12,
                    c=colort[i])
    #error=Stime[:,i]
    #plt.fill_between(n_vec, Mtime[:,i]-error, Mtime[:,i]+error, color=colort[i],alpha = 0.1)
    ax.append(ax1)



plt.xlabel('Number of samples', fontsize = 16)
plt.ylabel('Running Time Gain', fontsize = 16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(legend_vec_time)
plt.title('Screened OTDA on {:}'.format(data))
plt.grid(color='k', linestyle=':', linewidth=1,alpha=0.5)
filename = 'da_gain_{:}_regcl{:d}.pdf'.format(data,regcl)
plt.savefig('figure/' + filename,dpi=600)



# %% figure for timing


plt.figure(2)

for i in range(0,nb_p_vec+1):
    ax1, = plt.semilogy(n_vec, Mtime[:,i],label=str(i), lw = 2, marker = markert[i], markersize=12,
                    c=colort[i])
    #error=Stime[:,i]
    #plt.fill_between(n_vec, Mtime[:,i]-error, Mtime[:,i]+error, color=colort[i],alpha = 0.1)
    ax.append(ax1)



plt.xlabel('Number of samples', fontsize = 16)
plt.ylabel('Running Time (s)', fontsize = 16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(legend_vec_accur)
plt.title('Screened OTDA on {:}'.format(data))
plt.grid(color='k', linestyle=':', linewidth=1,alpha=0.5)
filename = 'da_time_{:}_regcl{:d}.pdf'.format(data,regcl)
plt.savefig('figure/' + filename,dpi=600)