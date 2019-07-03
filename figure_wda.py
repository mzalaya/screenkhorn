#!/usr/bin/env python
# coding: utf-8

__author__ = 'Alain Rakotomamonjy'


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

color_pal_t = sns.color_palette("colorblind", 11).as_hex()
color_pal = color_pal_t.copy()


colors = ["black", "salmon pink", "neon pink", "cornflower","cobalt blue"
          ,"blue green", "aquamarine", "dark yellow", "golden yellow", "reddish pink" , "reddish purple"]

color_pal = sns.xkcd_palette(colors)

plt.close("all")

pathres = './resultat/wda/'

data = 'toy'
if data == 'toy':
    n_vec = [200,500,1000,2000,3500,5000]
    nb_p_vec = 7
    p = 2
else:
    data = 'mnist'
    n_vec = [200,500,1000,2000,3000,4000]
    nb_p_vec = 7
    p = 20
    

method_vec = ['wda', 'swda']
legend_vec_accur = ['Sinkhorn','dec=1.5','dec=2','dec=5','dec=10','dec=20','dec=50','dec=100']
legend_vec_time = ['dec=1.5','dec=2','dec=5','dec=10','dec=20','dec=50','dec=100']

t = 1
in_sample = 0

Mres = np.zeros((len(n_vec), nb_p_vec+1))
Sres = np.zeros((len(n_vec), nb_p_vec+1))

Mtime = np.zeros((len(n_vec), nb_p_vec+1))
Stime = np.zeros((len(n_vec), nb_p_vec+1))

for i_k, n in enumerate(n_vec):                
    filename = 'wda_{:}_n{:d}_p{:d}'.format(data,n,p)
    
    
    # reading files and performance
    res = np.load(pathres + filename + '.npz')
    aux = res['bc_wda']
    nb_computed = np.where(aux)[0].shape[0]
    print(nb_computed)
    print(' \t\t {:2.2f} \t\t{:d}'.format( aux[np.where(aux)].sum()/nb_computed*100, nb_computed ))
    Mres[i_k,0] = aux[np.where(aux)].mean()
    Sres[i_k,0] = aux[np.where(aux)].std()
    
    aux = res['bc_swda']
    nb_computed = np.where(np.sum(aux,axis=1))[0].shape[0]
    print(nb_computed) 
    mean_perf = aux[np.where(np.sum(aux,axis=1))[0]].mean(axis=0)
    std_perf = aux[np.where(np.sum(aux,axis=1))[0]].std(axis=0)
    Mres[i_k,1:] = mean_perf # stocking average perf for all decimation
    Sres[i_k,1:] = std_perf
    
    # reading computation time
    aux = res['time_wda']
    nb_computed = np.where(aux)[0].shape[0]
    print(nb_computed)
    print(' \t\t {:2.2f} \t\t{:d}'.format( aux[np.where(aux)].sum()/nb_computed*100, nb_computed ))
    Mtime[i_k,0] = aux[np.where(aux)].mean()
    Stime[i_k,0] = aux[np.where(aux)].std()
    
    # computing gain 
    aux1 = aux.reshape(-1,1)/res['time_swda']
    nb_computed = np.where(np.isnan(np.sum(aux1,axis=1)) == False)[0].shape[0]
    print(nb_computed) 
    mean_perf = aux1[ np.where(np.isnan(np.sum(aux1,axis=1)) == False)[0]].mean(axis=0)
    std_perf = aux1[np.where(np.isnan(np.sum(aux1,axis=1)) == False)[0]].std(axis=0)
    # keeping gain for all p 
    Mtime[i_k,1:] = mean_perf
    Stime[i_k,1:] = std_perf

#%% figure for accuracy
    
plt.figure(0)
ax = []
markert = ['o','p','s','d','h','o','p','<','>','8','P']
colort = color_pal
for i in range(nb_p_vec+1):
    ax1, = plt.plot(n_vec, Mres[:,i],label=str(i), lw = 2, marker = markert[i], markersize=12,
                    c=colort[i])
    #error=Sres[:,i]
    #plt.fill_between(n_vec, Mres[:,i]-error, Mres[:,i]+error, color=colort[i],alpha = 0.1)
    ax.append(ax1)

if data ==  'mnist':
    plt.ylim([0.4,0.8])
    plt.xlim([0,4200])
else:
    plt.ylim([0.6,1])
    plt.xlim([0,4200])


plt.xlabel('Number of samples', fontsize = 16)
plt.ylabel('Accuracy', fontsize = 16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(legend_vec_accur)
plt.title('WDA+Knn on {:}'.format(data))
plt.grid(color='k', linestyle=':', linewidth=1,alpha=0.5)
filename = 'wda_accur_{:}.pdf'.format(data)
plt.savefig('figure/' + filename,dpi=600,bbox_inches='tight')


# %% figure for timing


plt.figure(1)

for i in range(1,nb_p_vec+1):
    ax1, = plt.plot(n_vec, Mtime[:,i],label=str(i), lw = 2, marker = markert[i], markersize=12,
                    c=colort[i])
    #error=Stime[:,i]
    #plt.fill_between(n_vec, Mtime[:,i]-error, Mtime[:,i]+error, color=colort[i],alpha = 0.1)
    ax.append(ax1)



plt.xlabel('Number of samples', fontsize = 16)
plt.ylabel('Running Time Gain', fontsize = 16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(legend_vec_time)
plt.title('Screened WDA on {:}'.format(data))
plt.grid(color='k', linestyle=':', linewidth=1,alpha=0.5)
filename = 'wda_gain_{:}.pdf'.format(data)
plt.savefig('figure/' + filename,dpi=600,bbox_inches='tight')
