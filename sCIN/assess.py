import numpy as np
from sklearn.metrics import silhouette_score
from models.ConAAE.conAAE import mink
from scipy.spatial.distance import cdist


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

########################################################################

def compute_distance(mod1_embs:np.array, mod2_embs:np.array) -> np.array:
       
    return cdist(mod1_embs, mod2_embs, metric="euclidean")
   

def sort_match_cells(dist_matrix:np.array) -> tuple[np.array, np.array, np.array]:
   
   num_cells = dist_matrix.shape[0]
   paired_cells = np.full(num_cells, -1)
   is_seen = np.zeros(num_cells, dtype=bool)
   for i in range(num_cells):
       sorted_indices = np.argsort(dist_matrix[i])

       for j in sorted_indices:
          if not is_seen[j]:
             paired_cells[i] = j
             is_seen[j] = True
             break
          
   sorted_indices_all = np.argsort(dist_matrix, axis=1)
          
   return paired_cells, sorted_indices_all


def compute_joint_distance(joint_embs:np.array) -> np.array:
   
   return cdist(joint_embs, joint_embs, metric="euclidean")


def select_k_closest(dist_matrix:np.ndarray, k:int) -> np.ndarray:
   
   np.fill_diagonal(dist_matrix, np.inf)

   return np.argsort(dist_matrix, axis=1)[:, :k]

   
def compute_recall_at_k(sorted_indices_all:np.array) -> dict:
   
   num_cells = sorted_indices_all.shape[0]
   recall_at_k = {}
   for k in [10, 20, 30, 40, 50]:
       recall_at_k[k] = np.mean([i in sorted_indices_all[i, :k] for i in range(num_cells)])
    
   return recall_at_k
   

def compute_num_pairs(paired_cells:np.array, num_cells:int) -> float:
   
   return float(np.sum(paired_cells == np.arange(num_cells)))
   

def compute_cell_type_accuracy(paired_cells:np.array, labels:np.array) -> float:
   
   return float(np.mean(labels[paired_cells] == labels))


def compute_cell_type_accuracy_joint(closest_cells:np.ndarray, labels:np.ndarray) -> float:
   
   num_cells = closest_cells.shape[0]

   is_seen = np.zeros(num_cells, dtype=bool)
   correct_matches = 0
   total_matches = 0

   for i in range(num_cells):
      count_correct = 0
      count_total = 0

      for j in closest_cells[i]:
         if not is_seen[j]:
            count_total += 1
            if labels[i] == labels[j]:
               count_correct += 1  
            is_seen[j] = True

         if count_total >= closest_cells.shape[1]:
            break
    
      correct_matches += count_correct
      total_matches += count_total

   return correct_matches / total_matches if total_matches > 0 else 0.0


def compute_norm_ASW(joint_embs:np.array, labels:np.array, seed=None) -> float:
   
   return (silhouette_score(joint_embs, labels, random_state=seed) + 1) / 2


def assess(mod1_embs:np.array, mod2_embs:np.array, 
           labels:np.array, seed:int=None) -> tuple[dict, int, float, float]:
   
   dist_matrix = compute_distance(mod1_embs, mod2_embs)
   paired_cells, sorted_indices_all = sort_match_cells(dist_matrix)
   recall_at_k = compute_recall_at_k(sorted_indices_all)
   num_pairs = compute_num_pairs(paired_cells, num_cells=dist_matrix.shape[0])
   cell_type_acc = compute_cell_type_accuracy(paired_cells, labels)

   joint_embs = np.column_stack((mod1_embs, mod2_embs))
   asw = compute_norm_ASW(joint_embs, labels, seed=seed)

   return recall_at_k, num_pairs, cell_type_acc, asw


def assess_joint(joint_embs:np.ndarray, labels:np.ndarray, 
                 seed:int=None, k:int=1) -> tuple[float, float]:
   
   dist_matrix = compute_joint_distance(joint_embs)
   closest_cells = select_k_closest(dist_matrix, k)
   cell_type_acc = compute_cell_type_accuracy_joint(closest_cells, labels)
   asw = compute_norm_ASW(joint_embs, labels, seed=seed)

   return cell_type_acc, asw