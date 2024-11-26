import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt 
from sc_cool.models.ConAAE.conAAE import mink
from typing import Dict, List


# Assessment functions
def ct_recall(ATAC_latent, RNA_latent, ATAC_class_label):

    cell_num = ATAC_latent.shape[0]
    recall_at_k = {10: 0, 20: 0, 30: 0, 40: 0, 50: 0}
    pairlist = []
    classlist = []
    visited = np.zeros(cell_num)

    rna_tree = KDTree(RNA_latent)

    for index, atac_emb in enumerate(ATAC_latent):
        
        nearest_idx = None
        k = 1

        print("Finding nearest ...")
        while nearest_idx is None:
            dist_nearest, nearest_indices = rna_tree.query(atac_emb.reshape(1, -1), k=k)
            nearest_indices = nearest_indices[0]

            for idx in nearest_indices:
                if visited[idx] == 0:
                    nearest_idx = idx
                    break

            k += 1 # find another nearest if the current has been visited
        
        print("Nearest was found ...")
    
        for k in [10, 20, 30, 40, 50]:
            print(f"Computing recall for k = {k}")
            
            dist_k, k_indices = rna_tree.query(atac_emb.reshape(1, -1), k=k)
            if index in k_indices[0]:  
                recall_at_k[k] += 1
    
        visited[nearest_idx] = 1
        pairlist.append(nearest_idx)
        classlist.append(ATAC_class_label[nearest_idx])

    pairlist = np.array(pairlist)
    classlist = np.array(classlist)
    
    ATAC_seq = np.arange(0, cell_num)

    # Padding
    if len(pairlist) < cell_num:
        pairlist = np.pad(pairlist, (0, cell_num - len(pairlist)), 'constant', constant_values=-1)

    # Compute recall and accuracy
    print(f"length classlist is: {np.shape(classlist)}")
    for k in [10, 20, 30, 40, 50]:
        recall_at_k[k] /= cell_num

    # Handle -1 padding by excluding from accuracy calculation
    valid_pairs = (pairlist != -1)
    number_of_pairs = float(np.sum(pairlist[valid_pairs] == ATAC_seq[valid_pairs]))
    class_lbl_acc = float(np.sum(ATAC_class_label[valid_pairs] == classlist)) / np.sum(valid_pairs)

    return recall_at_k, number_of_pairs, class_lbl_acc

####################################################################

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
      
      #print(len(pairlist))
    pairlist=np.array(pairlist)
      
      #print(flag)
      
    classlist=np.array(classlist)
      #print(pairlist)
      #print(classlist)

    recall_at_k = {10: 0, 20: 0, 30: 0, 40: 0, 50: 0}
    recall_at_k[10] = float(kright10)/cell_num
    recall_at_k[20] = float(kright20)/cell_num
    recall_at_k[30] = float(kright30)/cell_num
    recall_at_k[40] = float(kright40)/cell_num
    recall_at_k[50] = float(kright50)/cell_num
    number_of_pairs = float(np.sum(pairlist==ATAC_seq))
    class_lbl_acc = float(np.sum(labels==classlist)/cell_num)

    return recall_at_k, number_of_pairs, class_lbl_acc


def assess(rna_emb, atac_emb, labels_test, n_pc=10, 
           save_path=None, seed=None):

    # Define jonit embedding and PCA plot
    joint_emb = np.column_stack((rna_emb, atac_emb))
    pca = PCA(n_components=n_pc)
    joint_emb_pca = pca.fit_transform(joint_emb)
    cmap = plt.get_cmap('viridis', len(np.unique(labels_test)))
    plt.figure(figsize=(8, 6))
    plt.scatter(joint_emb_pca[:, 0], joint_emb[:, 1], c=labels_test, cmap=cmap)
    plt.title("PCA plot of embeddings for replicate" + "" + str(seed))
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.savefig(save_path + "/rep" + str(seed) + ".png")

    # Compute metrics
    recall_at_k, num_pairs, class_lbl_acc = ct_recall_v1(atac_emb, rna_emb, labels_test)

    # Compute ASW
    asw = (silhouette_score(joint_emb, labels_test) + 1) / 2

    # Save embddings
    np.save(save_path + "/rna_emb" + str(seed), rna_emb)
    np.save(save_path + "/atac_emb" + str(seed), atac_emb)

    return recall_at_k, num_pairs, class_lbl_acc, asw