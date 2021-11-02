#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 19:29:44 2020

@author: mmolina
"""


import numpy as np
import os
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import davies_bouldin_score
import pandas
import matplotlib.pyplot as plt
from config import cfg
import umap

sns.set(rc={'figure.figsize':(11.7,8.27)})
# Read and organize the data
data = pandas.read_csv('../data/extracted_cell_data.csv',header=None).to_numpy()
X=data[:,:cfg.N_features]
group=data[:,cfg.N_features].astype('int')
group_ids=['Group '+str(int(i+1)) for i in range(group.max())]
behavior=data[:,cfg.N_features+1].astype(int)

palette = sns.color_palette("bright", int(behavior.max()))
palette_group = sns.color_palette("colorblind", int(group.max()+1))


# Results directory
result_dir=os.path.join('../data')

# Analyze the Davies-Bouldin score for all the parameter combinations
DB_score=np.zeros((cfg.param1.shape[0],cfg.param2.shape[0]),dtype='float')
for i in range(len(cfg.param1)):
    for j in range(len(cfg.param2)):
        print('---'+str(i)+'---'+str(j))
        if (cfg.alg=='tsne'):
            X_embedded = TSNE(n_components=2,perplexity=cfg.param1[i],early_exaggeration=cfg.param2[j],learning_rate=200.0, n_iter=1000,
                              n_iter_without_progress=300,metric='euclidean',random_state=1).fit_transform(X)
        else:
            X_embedded=umap.UMAP(n_neighbors=int(cfg.param1[i]),
                      min_dist=int(cfg.param2[j]),
                      metric='euclidean',random_state=1).fit_transform(X)
        
        # Lower value, better clustering
        DB_score[i,j]=davies_bouldin_score(X_embedded, behavior.ravel())

# Graphical representation of the behavior proportion in the groups
histograms=np.zeros((group.max(),behavior.max()),dtype='float')
for i in range(1,int(group.max()+1)):
    cluster=behavior[np.where(group==i)]
    for j in range(1,int(behavior.max()+1)):
        histograms[int(i-1),int(j-1)]=np.sum(cluster==j)/cluster.shape[0]
ind=np.arange(start=1,stop=int(group.max()+1))*0.8
width=0.5
p=[]
leg=[]
for i in range(behavior.max()):
    leg.append('behavior '+str(i+1))
    if (i==0):
        bottom=np.zeros((group.max(),),dtype='float')
    else:
        bottom=np.zeros((group.max(),),dtype='float')
        for j in range(i):
            bottom+=histograms[:,j]
    p.append(plt.bar(ind, histograms[:,i], width, bottom=bottom, color=palette[i], edgecolor="gray"))
plt.ylabel('Behavior proportion in groups',fontsize=12)
plt.xticks(ind, group_ids,fontsize=12)
plt.legend(tuple(p), tuple(leg),prop={'size': 12})
plt.savefig(os.path.join(result_dir,'Stacked_behavior_proportion.png'),dpi=600)
plt.clf()

width=1.0/(behavior.max()+2)
rr=[]
for i in range(behavior.max()):
    if (i==0):
        rr.append(np.arange(len(histograms[:,0]))+width/2+0.1*width)
    else:
        aux=[x + width+0.1*width for x in rr[-1]]
        rr.append(aux)

# Make the plot
p=[]
for i in range(behavior.max()):
    p.append(plt.bar(rr[i], histograms[:,i], color=palette[i], width=width, edgecolor='gray', label=leg[i]))

# Add xticks on the middle of the group bars
plt.ylabel('Behavior proportion in groups',fontsize=12)
plt.legend(tuple(p), tuple(leg),prop={'size': 12})
plt.xticks([i+0.22+group.max()/2*0.11 for i in range(group.max())], group_ids,fontsize=12)
plt.savefig(os.path.join(result_dir,'Independent_behavior_proportion.png'),dpi=600)
plt.clf()
# For the best cases of Davies-Bouldin values, obtain the t-SNE or UMAP graphs
for m in range(cfg.graphs_number):
    # Davies Bouldin
    ind = np.unravel_index(np.argmin(DB_score, axis=None), DB_score.shape)
    DB_score[ind[0],ind[1]]=np.inf # Delete for the next iteration
    if (cfg.alg=='tsne'):
        # t-SNE
        X_embedded = TSNE(n_components=2,perplexity=int(cfg.param1[ind[0]]),early_exaggeration=int(cfg.param2[ind[1]]),learning_rate=200.0, n_iter=1000,
                                  n_iter_without_progress=300,metric='euclidean',random_state=1).fit_transform(X)
    else:
        # UMAP
        X_embedded = umap.UMAP(n_neighbors=int(cfg.param1[ind[0]]),min_dist=int(cfg.param2[ind[1]]),metric='euclidean',random_state=1).fit_transform(X)
    
    sns_plot = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=behavior.ravel(), s=cfg.POINT_SIZE, legend='full', alpha=cfg.ALPHA, palette=palette)
    new_labels=[]
    for i in range (1,int(behavior.max()+1)):
        # replace labels
        new_labels.append('behavior '+str(i))
    for t, l in zip(sns_plot.get_legend().texts, new_labels):  t.set_text(l)
    sns_plot.figure.savefig(os.path.join(result_dir,cfg.alg+'_behaviors'+str(m)+'.png'))
    plt.clf()
    
    