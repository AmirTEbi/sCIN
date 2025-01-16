import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt 
from sCIN.models.ConAAE.conAAE import mink
from typing import Dict, List
import os


def compute_metrics(mod1_embs:np.array, mod2_embs:np.array, labels:np.array):

    cell_num = mod1_embs.shape[0]
    pairlist=[]
    classlist=[]
    distlist=[]
    flag=np.zeros(cell_num)
    kright10=0
    kright20=0
    kright30=0
    kright40=0
    kright50=0
      
    for index,i in enumerate(mod1_embs):
      mini=999999
      minindex=0
      distlist.clear()
      for idx,j in enumerate(mod2_embs):
        dist=np.linalg.norm(i-j)
        distlist.append(dist)
          #print(dist)
        if(dist<mini and flag[idx]==0):
          mini=dist
          minindex=idx
      kindex10=mink(distlist,10)
      kindex20=mink(distlist,20)
      kindex30=mink(distlist,30)
      kindex40=mink(distlist,40)
      kindex50=mink(distlist,50)
      if(index in kindex10):
        kright10+=1
      if(index in kindex20):
        kright20+=1
      if(index in kindex30):
        kright30+=1
      if(index in kindex40):
        kright40+=1
      if(index in kindex50):
        kright50+=1
      flag[minindex]=1
      pairlist.append(minindex)
      classlist.append(labels[minindex])
      
    ATAC_seq=np.arange(0,cell_num,1)
      
    pairlist=np.array(pairlist)
            
    classlist=np.array(classlist)

    recall_at_k = {10: 0, 20: 0, 30: 0, 40: 0, 50: 0}
    recall_at_k[10] = float(kright10)/cell_num
    recall_at_k[20] = float(kright20)/cell_num
    recall_at_k[30] = float(kright30)/cell_num
    recall_at_k[40] = float(kright40)/cell_num
    recall_at_k[50] = float(kright50)/cell_num
    num_pairs = float(np.sum(pairlist==ATAC_seq))
    cell_type_acc = float(np.sum(labels==classlist)/cell_num)

    return recall_at_k, num_pairs, cell_type_acc


def assess(mod1_embs, mod2_embs, labels_test, seed=None):

    joint_emb = np.column_stack((mod1_embs, mod2_embs))

    recall_at_k, num_pairs, cell_type_acc = compute_metrics(mod2_embs, mod1_embs, labels_test)
    asw = (silhouette_score(joint_emb, labels_test, random_state=seed) + 1) / 2

    return recall_at_k, num_pairs, cell_type_acc, asw