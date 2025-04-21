import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, csr_matrix  
from scipy.sparse.csgraph import connected_components
from typing import *


def compute_distance(mod1_embs:np.array, mod2_embs:np.array) -> np.array:
    """
    
    """
       
    return cdist(mod1_embs, mod2_embs, metric="euclidean")
   

def sort_match_cells(dist_matrix:np.array) -> Tuple[np.array, np.array, np.array]:
   """
   
   """
   
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
   """
   
   """
   
   return cdist(joint_embs, joint_embs, metric="euclidean")


def select_k_closest(dist_matrix:np.ndarray, k:int) -> np.ndarray:
   """
   
   """
   
   np.fill_diagonal(dist_matrix, np.inf)

   return np.argsort(dist_matrix, axis=1)[:, :k]

   
def compute_recall_at_k(sorted_indices_all:np.array) -> dict:
   """
   
   """
   
   num_cells = sorted_indices_all.shape[0]
   recall_at_k = {}
   for k in [10, 20, 30, 40, 50]:
       recall_at_k[k] = np.mean([i in sorted_indices_all[i, :k] for i in range(num_cells)])
    
   return recall_at_k
   

def compute_num_pairs(paired_cells:np.array, num_cells:int) -> float:
   """
   
   """
   
   return float(np.sum(paired_cells == np.arange(num_cells)))
   

def compute_cell_type_accuracy(paired_cells:np.array, labels:np.array) -> float:
   """
   
   """
   
   return float(np.mean(labels[paired_cells] == labels))


def compute_cell_type_accuracy_joint(closest_cells:np.ndarray, labels:np.ndarray) -> float:
   """
   
   """
   
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
   """
   
   """
   
   return (silhouette_score(joint_embs, labels, random_state=seed) + 1) / 2


def compute_MedR(mod1_embs:np.ndarray, mod2_embs:np.ndarray) -> float:
   """
   Compute Median Rank metric between two embedding matrices.

   Parameters
   ----------

   """
   sum_embs1 = np.sum(mod1_embs**2, axis=1)[:, np.newaxis]
   sum_embs2 = np.sum(mod2_embs**2, axis=1)[:, np.newaxis]
   dists = sum_embs1 + sum_embs2 - 2 * np.dot(mod1_embs, mod2_embs.T)
   diag_dists = np.diag(dists)
   ranks = np.sum(dists < diag_dists[:, np.newaxis], axis=1)

   return np.median(ranks)


def assess(mod1_embs:np.array, mod2_embs:np.array, 
           labels:np.array, seed:int=None) -> Tuple[dict, int, float, float]:
   
   dist_matrix = compute_distance(mod1_embs, mod2_embs)
   paired_cells, sorted_indices_all = sort_match_cells(dist_matrix)
   recall_at_k = compute_recall_at_k(sorted_indices_all)
   num_pairs = compute_num_pairs(paired_cells, num_cells=dist_matrix.shape[0])
   cell_type_acc = compute_cell_type_accuracy(paired_cells, labels)

   joint_embs = np.column_stack((mod1_embs, mod2_embs))
   asw = compute_norm_ASW(joint_embs, labels, seed=seed)
   medr = compute_MedR(mod1_embs=mod1_embs, mod2_embs=mod2_embs)

   return recall_at_k, num_pairs, cell_type_acc, asw, medr


def assess_joint(joint_embs:np.ndarray, 
                 labels:np.ndarray, 
                 seed:int=None, k:int=1) -> Tuple[float, float]:
   
   """
   
   """
   
   dist_matrix = compute_joint_distance(joint_embs)
   closest_cells = select_k_closest(dist_matrix, k)
   cell_type_acc = compute_cell_type_accuracy_joint(closest_cells, labels)
   asw = compute_norm_ASW(joint_embs, labels, seed=seed)

   return cell_type_acc, asw


def assess_joint_from_separate_embs(mod1_embs: np.ndarray,
                                    mod2_embs: np.ndarray,
                                    labels: np.ndarray,
                                    seed: int = None,
                                    k: int = 1) -> Tuple[float, float]:
   """
   Compute metrics for joint embeddings when the model returns separate embeddings.

   Parameters
   ----------
   mod1_embs: np.ndarray
      Embeddings for the modality 1.

   mod2_embs: np.ndarray
      Embeddings for the modality 2.

   labels: np.ndarray
      Cell type labels.

   seed: int
      Random seed for the current replication.

   k: int
      The number of nearest neighbors.

   Returns
   -------
   Tuple[float, float]
      Cell type accuracy and Average Silhouette Width (ASW)
   """
   
   joint_embs = np.column_stack([mod1_embs, mod2_embs])
   dist_matrix = compute_joint_distance(joint_embs)
   closest_cells = select_k_closest(dist_matrix, k)
   cell_type_acc = compute_cell_type_accuracy_joint(closest_cells, labels)
   asw = compute_norm_ASW(joint_embs, labels, seed=seed)

   return cell_type_acc, asw

######## Unpaired ########

def _build_knn_adj_graph(embs: np.ndarray, k: int) -> csr_matrix:
   """
    Build a symmetric k-NN adjacency matrix for all cells.

    Parameters
    ----------
    embs : shape (n_cells, n_features)
    k : number of neighbors (excluding self)

    Returns
    -------
    csr_matrix : sparse adjacency of shape (n_cells, n_cells)
    """
   num_cells = embs.shape[0]
   nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(embs)
   _, indices = nbrs.kneighbors(embs)
   indices = indices[:, 1:]  # Remove self

   row = np.repeat(np.arange(num_cells), k)
   col = indices.flatten()  
   data = np.ones(len(row), dtype=bool)
   adj = coo_matrix((data, (row, col)), shape=(num_cells, num_cells))
   adj = adj + adj.T  # Make undirected
   adj.data = np.clip(adj.data, 0, 1)

   return adj.tocsr() 


def compute_graph_connectivity(embs: np.ndarray, 
                               labels: np.ndarray, 
                               k: int = 15) -> float:
   
   """
    Compute graph connectivity (GC) for a single modality embedding.

    GC = (1 / M) * sum_j (LCC_j / N_j)
    where LCC_j = size of largest connected component among cells of type j.
    """
   num_cells = embs.shape[0]
   if labels.shape[0] != num_cells:
      raise ValueError("embeddings and labels must have the same length")
   
   adj = _build_knn_adj_graph(embs, k)
   unique_labels, _ = np.unique(labels, return_counts=True)

   scores = []
   for lbl in unique_labels:
      idx = np.where(labels == lbl)[0]
      sub = adj[idx][:, idx]
      n_comp, comp_labels = connected_components(sub, directed=False)
      sizes = np.bincount(comp_labels)
      if idx.size > 0:
         lcc_ratio = sizes.max() / idx.size 
      
      else: 
         lcc_ratio = 0.0
      
      scores.append(lcc_ratio)
   
   return float(np.mean(scores))


def compute_graph_connectivity_multimodal(embs_dict: Dict[str, np.ndarray], 
                                          labels_dict: Dict[str, np.ndarray], 
                                          k: int = 15, 
                                          mode: str = "union") -> Union[float, Dict[str, float]]:

   if mode == "uni-modal":
        return {
            m: compute_graph_connectivity(X, labels_dict[m], k)
            for m, X in embs_dict.items()
        }
   
   if mode == "joint":
      joint_embs = np.concatenate(list(embs_dict.values()), axis=0)
      joint_lbls = np.concatenate(list(labels_dict.values()), axis=0)
      
      return compute_graph_connectivity(joint_embs, joint_lbls, k)
   
   if mode == "union":
      adjs = []
      label_splits = []
      offset = 0
      for m in embs_dict:
          embs = embs_dict[m]
          lbls = labels_dict[m]
          adj = _build_knn_adj_graph(embs, k)
          adjs.append(adj)
          label_splits.append((lbls, offset))
          offset += embs.shape[0]
      
      combined = sum(adjs)
      combined.data = np.clip(combined.data, 0, 1)
      lbls_all = np.concatenate([y for y, _ in label_splits], axis=0)

      unique_labels, _ = np.unique(lbls_all, return_counts=True)
      scores = []
      for lbl in unique_labels:
         idx = np.where(lbls_all == lbl)[0]
         sub = combined[idx][:, idx]
         n_comp, comp_labels = connected_components(sub, directed=False)
         sizes = np.bincount(comp_labels)
         scores.append(sizes.max() / idx.size if idx.size > 0 else 0.0)
      
      return float(np.mean(scores))
   
   raise ValueError("mode must be one of 'individual', 'concat', or 'union'")


def cell_type_at_k_unpaired(mod1_embs: np.ndarray,
                            mod2_embs: np.ndarray,
                            mod1_labels: np.ndarray,
                            mod2_labels: np.ndarray,
                            num_nbrs: List[int] = [10, 20, 30, 40, 50]) -> Dict[int, float]:
   
   ct_at_k = {}
   for k in num_nbrs:
      nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(mod2_embs)
      dists, indices = nbrs.kneighbors(mod1_embs)
      nbrs_labels = mod2_labels[indices]
      matches = (nbrs_labels == mod1_labels[:, None])
      ct_at_k_all_cells = matches.sum(axis=1) / k 
      mean_ct_at_k = ct_at_k_all_cells.mean()
      ct_at_k[k] = mean_ct_at_k
   
   return ct_at_k


def assess_unpaired(mod1_embs: np.ndarray, 
                    mod2_embs: np.ndarray,
                    mod1_labels: np.ndarray,
                    mod2_labels: np.ndarray,
                    seed: int) -> Tuple[Dict[int, float], int, float, float]:
   
   ct_at_k = cell_type_at_k_unpaired(mod1_embs, 
                                     mod2_embs,
                                     mod1_labels,
                                     mod2_labels)
   
   joint_embs = np.concatenate((mod1_embs, mod2_embs), axis=0)
   joint_lbls = np.concatenate((mod1_labels, mod2_labels), axis=0)
   asw = compute_norm_ASW(joint_embs, joint_lbls, seed=seed)
   embs_dict = {"Mod1": mod1_embs, "Mod2": mod2_embs}
   labels_dict = {"Mod1": mod1_labels, "Mod2": mod2_labels}
   GC_uni = compute_graph_connectivity_multimodal(embs_dict, labels_dict, mode="uni-modal")
   GC_joint = compute_graph_connectivity_multimodal(embs_dict, labels_dict, mode="joint")
   GC_union = compute_graph_connectivity_multimodal(embs_dict, labels_dict, mode="union")

   return ct_at_k, asw, GC_uni, GC_joint, GC_union