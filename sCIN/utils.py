"""Utility functions"""

import numpy as np
import anndata as ad
import pandas as pd
import pyranges as pr
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
from sCIN.assess import assess, assess_joint, assess_joint_from_separate_embs
import colorcet as cc
import json
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
from typing import Dict, Tuple, List, Optional, Union, Any
import logging
import re
import os


###########################################
#                                         #
#  Functions for computing gene activity  #
#                                         #
###########################################

def _remove_dot_in_gene_ids(data: pd.DataFrame) -> pd.DataFrame:
     """
     Remove '.' in gene ids.

     Parameters
     ----------
     file: pd.DataFrame
          A GTF data frame or .var.gene_id of an AnnData object.
     
     Return
     ------
     pd.DataFrame
          Corrected gene_ids.
     """
     data_ = data.copy()
     data_["gene_id_corr"] = data_["gene_id"].str.extract(r"^([^\.]+)")

     return data_


def extend_gene_regions(gtf_df: pd.DataFrame, extension: int = 2000) -> pd.DataFrame:
     """
     Extend the gene body to cover the promoter region.

     Parameters
     ----------
     gtf_df: pd.DataFrame
          The Gene Transfer Format file as a data frame.

     extension: int
          The number of basepairs the gene body extend to. 
     
     Returns
     -------
     pd.DataFrame
          Updated gtf_df with necessary columns for downstream tasks.
     """
     plus_strand = gtf_df["Strand"] == "+"
     minus_strand = ~plus_strand

     gtf_df.loc[plus_strand, "Start"] = np.maximum(0, gtf_df.loc[plus_strand, "Start"] - extension)
     gtf_df.loc[minus_strand, "End"] = gtf_df.loc[minus_strand, "End"] + extension

     return gtf_df[["Chromosome", "Start", "End", "gene_id"]]


def process_ATAC_peaks_for_gene_act(atac_ad: ad.AnnData,
                                    columns_map: Dict[str,str],
                                    std_chrs: Dict[str, Any]) -> pr.PyRanges:
     """
     Process the ATAC-seq peaks for computing gene activity.

     Parameters
     ----------
     atac_ad: ad.AnnData
          ATAC-seq data
     
     Returns
     -------
     pr.pyranges.PyRanges
          ATAC-seq peaks characteristics as a PyRanges object.
     """
     peaks_df = atac_ad.var.reset_index()
     peaks_df.rename(columns={k: v for k, v in columns_map.items() if k in peaks_df.columns}, inplace=True)
     if "peak_id" not in peaks_df.columns:
          peaks_df["peak_id"] = peaks_df.index.astype(str)
     
     if "Chromosome" in peaks_df.columns:
          peaks_df = peaks_df.loc[peaks_df["Chromosome"].isin(std_chrs), ["peak_id", "Chromosome", "Start", "End"]].copy()
     else:
        raise KeyError("Missing 'Chromosome' column in ATAC peak data.")

     return pr.PyRanges(peaks_df)


def process_GTF_for_gene_act(gtf_df: pd.DataFrame,
                             std_chrs: Dict[str, Any],
                             extension: int = 2000,
                             feature_type: str = "transcript") -> pr.PyRanges:
     """
     Process GTF file for computing gene activity.

     Parameters
     ----------
     gtf_df: pd.DataFrame
          If you used pyranges.read_gtf() to parse the GTF file, simply use the **df** attribute of the output.
     
     extention: int
          The number of basepairs the gene body extend to. 
     
     Return
     ------
     pr.pyranges.PyRanges
     """
     gtf_df["Chromosome"] = gtf_df["Chromosome"].apply(lambda x: x if x.startswith("chr") else "chr" + x)
     gtf_df = gtf_df[(gtf_df["Feature"] == feature_type) & (gtf_df["Chromosome"].isin(std_chrs))].copy()
     gtf_extended = extend_gene_regions(gtf_df, extension)

     return pr.PyRanges(gtf_extended)


def compute_gene_activity(atac_ad: ad.AnnData, 
                          gtf_file_path: str,
                          save_dir: str, 
                          extension: int = 2000,
                          feature_type: str = "transcript") -> None :
     
     """
     Compute the gene activity matrix from the given ATAC-seq data.

     Parameters
     ----------
     atac_ad: ad.AnnData
          ATAC-seq data.

     rna_ad: ad.AnnData
          RNA-seq data.

     gtf_file_path: str
          The path to the Gene Tranfer Format (GTF) file.

     extension: int
          The number of basepairs the gene body extend to. 
     
     Returns
     -------
     Tuple[np.ndarray, List[str]]
          gene_activity (n_cells, n_genes) containing summed ATAC counts.
          gene_ids corresponding to the columns.
     """
     # Read GTF file
     gtf = pr.read_gtf(gtf_file_path)
     gtf_df = gtf.df
     print("Head of the GTF annotations:")
     print(gtf_df.head())

     # Correct gene_id 
     gtf_df = _remove_dot_in_gene_ids(gtf_df)

     std_chrs = {f'chr{i}' for i in range(1, 23)} | {"chrX", "chrY", "chrM"}

     # Process ATAC peaks
     rename_dict = {
        "index": "peak_id",
        "chrom": "Chromosome",
        "chromStart": "Start",
        "chromEnd": "End"
        }
     
     peaks_ranges = process_ATAC_peaks_for_gene_act(atac_ad, rename_dict, std_chrs)

     # Process GTF
     gene_ranges = process_GTF_for_gene_act(gtf_df, std_chrs, feature_type="transcript")

     # Overlap ATAC peaks with gene regions
     overlap_df = peaks_ranges.join(gene_ranges, how="left").df
     peaks_df = peaks_ranges.df
     peaks_df.set_index(["Chromosome", "Start", "End"], inplace=True)
     overlap_df["peak_id"] = peaks_df.loc[overlap_df.set_index(["Chromosome", "Start", "End"]).index, "peak_id"].values

     gene2peak = overlap_df.groupby("gene_id")["peak_id"].apply(list)

     # Build gene activity matrix
     num_cells = atac_ad.n_obs
     genes = list(gene2peak.index)
     gene_activity = np.zeros((num_cells, len(genes)))

     for i, gene in enumerate(genes):
          peak_ids = gene2peak[gene]
          valid_indices = []
          for p in peak_ids:
            try:
                idx = int(p)
                if idx < atac_ad.shape[1]:
                    valid_indices.append(idx)
            except ValueError:
                continue
          if not valid_indices:
               continue

          peak_names = atac_ad.var_names[valid_indices]
          if hasattr(atac_ad.X, "A"):
               gene_activity[:, i] = atac_ad[:, peak_names].X.sum(axis=1).A.ravel()
          
          else:
               gene_activity[:, i] = atac_ad[:, peak_names].X.sum(axis=1)
          
     # Save the ouput
     gene_act_ad = ad.AnnData(X=gene_activity)
     genes_str = [str(g) for g in genes]
     # print(pd.Series(genes).fillna("").astype(str))
     gene_ids = pd.Series(genes).fillna("").astype(str).apply(lambda x: str(x).strip())
     gene_act_ad.var["gene_id"] = gene_ids.values
     print(gene_act_ad.var["gene_id"].head())
     print(gene_act_ad.var["gene_id"].dtype)
     print(type(gene_act_ad.var["gene_id"].iloc[0]))

     if "cell_type" in atac_ad.obs.columns:
          gene_act_ad.obs["CellType"] = atac_ad.obs["cell_type"]
     else:
          print("Warning: 'cell_type' column not found in ATAC AnnData.obs.")
     gene_act_ad.write(os.path.join(save_dir, "gene_activity.h5ad"))
     
     return None


def find_ovelap_genes(gene_activity_ad: ad.AnnData, rna_ad: ad.AnnData) -> Tuple[ad.AnnData, ad.AnnData]:
     """
     Find overlap genes betwen ATAC gene activity and the RNA data.

     Parameters
     ----------
     gene_activity_mat: np.ndarray
          The gene activity matrix computed from the ATAC-seq data.

     rna_ad: ad.AnnData
          The RNA-seq data.
     
     Returns
     -------
     Tuple[ad.AnnData, ad.AnnData]
          Updated AnnData objects with overlap genes.
     """
     if "gene_id" not in rna_ad.var.columns:
          raise KeyError("RNA AnnData must have a 'gene_id' column in rna.var.")
     
     rna_ad.var = rna_ad.var.set_index("gene_id", drop=False)
     if not rna_ad.var.index.is_unique:
          print("Warning: Duplicate gene_ids found in RNA data. Removing duplicates.")
          rna = rna[:, ~rna.var.index.duplicated()].copy()
     
     if "gene_id" not in gene_activity_ad.var.columns:
          raise KeyError("Gene activity AnnData must have a 'gene_id' column in .var.")
     
     atac_genes = gene_activity_ad.var["gene_id"].tolist()
     rna_genes = rna_ad.var.index.tolist()

     shared_genes = set(atac_genes).intersection(rna_genes)
     print(f"Number of shared genes: {len(shared_genes)}")

     shared_genes_sorted = sorted(shared_genes)

     filtered_gene_activity_ad = gene_activity_ad[:, shared_genes_sorted].copy()
     filtered_rna_ad = rna_ad[:, shared_genes_sorted].copy()

     return filtered_gene_activity_ad, filtered_rna_ad


###########################################
#                                         #
#  Functions for computing metrics from   #
#               embeddings                #
#                                         #
###########################################

def read_embs_from_np(file_list: List[str]) -> List[np.ndarray]:
     """
     Read embeddings' files with .npy extention.

     Parameters
     ----------
     file_list: List[str]
          A list of embeddings files.
     
     Returns
     -------
     List[np.ndarray]
          A list of embeddings.
     """
     out = []
     for f in file_list:
          f_np = np.load(f)
          out.append(f_np)
     
     return out 


def read_embs_from_csv(file_list: List[str]) -> List[np.ndarray]:
     """
     Read embeddings' files with .csv extension.

     Parameters
     ----------
     file_list: List[str]
          A list of embeddings files.

     Returns
     -------
     List[np.ndarray]
          A list of embeddings.
     """

     out = []
     for f in file_list:
          f_df = pd.read_csv(f_df)
          f_np = f_df.values
          out.append(f_np)

     return out


def find_matched_files(embs_file: str, 
                       other_embs_files: List[str], 
                       labels_files: Optional[List[str]] = None) -> Union[Tuple[str,str], str, None]:
     """
     Find the matching files to the given embeddinf file based on the replication.

     Parameters
     ----------
     embs_file: string
          Name of the embedding file you want to find the matching files for.
     
     other_embs_files: List[str]
          A list of embedding files you want to find the matching files from.

     labels_files:  Optional[List[str]]
          A list of labels files you want to find the matching files from.

     Returns
     -------
     Union[Tuple[str,str], str, None]
          A tuple of the names of the matched files (embedding and label),
          or only the name of the matched file, or None if no match is found.  
     """

     match = re.search(r"\d+", embs_file)
     if not match:
          return None
     
     rep = match.group()
     
     if labels_files is not None:
          for f, f_ in zip(other_embs_files, labels_files):
               if rep in f and rep in f_:   
                    return (f, f_)
          raise FileNotFoundError(f"No matching embedding and label files found for rep '{rep}'")

     else:
          for f in other_embs_files:
               if rep in f:
                    return f
          raise FileNotFoundError(f"No matching embedding file found for rep '{rep}'")


def extract_file_extension(file_path: str) -> str:
    """
    Extract the file extension from the file path.
    
    Parameters
    ----------
    file_path: string
		The path to the file or the file name.
          
    Returns
    -------
    string 
		The file extenstion.
    """
    _, extenstion = os.path.splitext(file_path)

    return extenstion


def find_seed_from_filename(filename: str) -> int:
     """
     Find the random seed from the filename.

     Parameters
     ----------
     filename: str
          The name of the file.
     
     Returns
     -------
     int
          The random seed.
     """

     match = re.search(r"\d+", filename)
     if not match:
          return None
     
     seeds = [seed for seed in range(0, 100, 10)]
     reps = [rep for rep in range(1, 11)]
     found = match.group()
     if found in reps:
          return int(found)
     
     elif found in seeds:
          return (found / 10) + 1 
     

def find_rep_from_filename(filename: str, seed_to_rep_map: Optional[Dict[int, int]] = None) -> int:
     """
     Extracts the replication number from a file name.

     Parameters
     ----------
     filename : str
        The name of the file.
     seed_to_rep_map : Optional[Dict[int, int]]
        A dictionary mapping seed values to replication numbers (e.g., {40: 5, 70: 6}).

     Returns
     -------
     int
        The replication number.
    
     Raises
     ------
     ValueError
        If the replication number cannot be determined.
     """
     if "rep" in filename.lower():
          match = re.search(r"(\d+)", filename, re.IGNORECASE)
          if match:
               return int(match.group())
     
     match = re.search(r"(\d+)", filename, re.IGNORECASE)
     if match:
          seed = int(match.group())
          if seed_to_rep_map is not None and seed in seed_to_rep_map:
               return seed_to_rep_map[seed]
          
          return seed

     raise ValueError(f"Cannot parse replication from filename: {filename}")


def compute_metrics_from_embs_for_all_models(res_dir: str) -> None:
     """
     Compute metrics for all models' embeddings.

     Parameters
     ----------
     res_dir: string
          The directory to save all the results.
     """


def compute_metrics_from_embs_for_one_model(model_res_dir: str, compute_metrics_inversly: bool = True) -> None:

     """
     Compute metrics for the given model.

     Parameters
     ----------
     model_res_dir: string
          The directory to save the results of the given model.
     
     compute_metrics_inversly: bool
          Whether the function should compute metrics from the modality 2 to moality 1, if applicable.
     
     Returns
     -------
     None
          Saves the resulting CSV file in the 'outs' directory in the given directory.
     """
     embs_dir = os.path.join(model_res_dir, "embs")
     if not os.path.isdir(embs_dir):
          raise FileNotFoundError(f"Directory not found: {embs_dir}")
     
     rna_embs_files = [f for f in os.listdir(embs_dir) if "rna" in f]
     atac_embs_files = [f for f in os.listdir(embs_dir) if "atac" in f]
     labels_files = [f for f in os.listdir(embs_dir) if "label" in f] 
     file_extension = extract_file_extension(rna_embs_files[0])
     if file_extension == ".npy":
          read_embs = read_embs_from_np

     elif file_extension == ".csv":
          read_embs = read_embs_from_csv
     
     else:
          raise ValueError("Unsupported file extension. Embeddings should be stored as .npy or .csv files.")

     seed_to_rep_map = {
          0: 1,
          10: 2,
          20: 3, 
          30: 4, 
          40: 5, 
          50: 6,
          60: 7,
          70: 8,
          80: 9,
          90: 10
     }
     
     model_name = os.path.basename(model_res_dir)
     res = []
     for f_rna in rna_embs_files:
          f_atac, f_lbls = find_matched_files(f_rna, atac_embs_files, labels_files)
          seed = find_seed_from_filename(f_rna)
          rep = find_rep_from_filename(f_rna, seed_to_rep_map)
          rna_embs, atac_embs, lbls = read_embs([f_rna, f_atac, f_lbls])
          cell_type_acc_joint, asw = assess_joint_from_separate_embs(atac_embs, rna_embs, lbls, seed)
          recall_at_k_a2r, num_pairs_a2r, cell_type_acc_a2r, _, medr_a2r = assess(atac_embs, rna_embs, lbls, seed)

          if compute_metrics_inversly:
               recall_at_k_r2a, num_pairs_r2a, cell_type_acc_r2a, _, medr_r2a = assess(rna_embs, atac_embs, lbls, seed)
               for k, v_a2r in recall_at_k_a2r.items():
                    v_r2a = recall_at_k_r2a.get(k, 0)
                    res.append({
                         "Models": model_name,
                         "Replicates":rep,
                         "k": k,
                         "Recall_at_k_a2r": v_a2r,
                         "Recall_at_k_r2a": v_r2a if v_r2a is not None else 0.0,
                         "num_pairs_a2r": num_pairs_a2r,
                         "num_pairs_r2a": num_pairs_r2a if num_pairs_r2a is not None else 0.0,
                         "cell_type_acc_a2r": cell_type_acc_a2r,
                         "cell_type_acc_r2a": cell_type_acc_r2a if cell_type_acc_r2a is not None else 0.0,
                         "cell_type_ASW": asw,
                         "MedR_a2r": medr_a2r,
                         "MedR_r2a": medr_r2a if medr_r2a is not None else 0.0,
                         "cell_type_acc_joint": cell_type_acc_joint
                    })
          
          else:
               for k, v_a2r in recall_at_k_a2r.items():
                    res.append({
                        "Models": model_name,
                         "Replicates": rep,
                         "k": k,
                         "Recall_at_k_a2r": v_a2r,
                         "num_pairs_a2r": num_pairs_a2r,
                         "cell_type_acc_a2r": cell_type_acc_a2r,
                         "cell_type_ASW": asw,
                         "MedR_a2r": medr_a2r
                    })
          
     results = pd.DataFrame(res)
     res_save_dir = os.path.join(model_res_dir, "outs")
     os.makedirs(res_save_dir, exist_ok=True)
     results.to_csv(os.path.join(res_save_dir, f"Metrics_{model_name}_10_reps_from_embs.csv"), index=False)

                               
def setup_logging(level:str, 
                  log_dir:str,
                  model_name:str) -> object:
     
     os.makedirs(log_dir, exist_ok=True)
     log_file = os.path.join(log_dir, f"{model_name}.log")

     if level == "info":
          logging.basicConfig(level=logging.INFO,
                              format="%(asctime)s - %(levelname)s - %(message)s",
                              handlers=[logging.FileHandler(log_file),
                                        logging.StreamHandler()])
     elif level == "debug":
          logging.basicConfig(level=logging.DEBUG,
                              format="%(asctime)s - %(levelname)s - %(message)s",
                              handlers=[logging.FileHandler(log_file),
                                        logging.StreamHandler()])
     elif level == "warning":
          logging.basicConfig(level=logging.WARNING,
                              format="%(asctime)s - %(levelname)s - %(message)s",
                              handlers=[logging.FileHandler(log_file),
                                        logging.StreamHandler()])
     
     elif level == "error":
          logging.basicConfig(level=logging.ERROR,
                              format="%(asctime)s - %(levelname)s - %(message)s",
                              handlers=[logging.FileHandler(log_file),
                                        logging.StreamHandler()])
          
     logging.info(f"Logs will be saved to {log_file}.")

###########################################
#                                         #
#   Helper functions for runnig models    #
#                                         #
###########################################

def  impute_cells(labels):
    """Make same-size label arrays

    Parameters
    ----------
    labels : numpy.ndarray
        Array of the encoded cell types as integers.

    Returns
    -------

    
    """
    indices_by_type = {label: np.where(labels == label)[0] for label in np.unique(labels)}
    max_length = max(len(indices) for indices in indices_by_type.values())
    imputed_indices_by_type = {}
    for label, indices in indices_by_type.items():
        current_length = len(indices)
        if current_length < max_length:
            add_indices = np.random.choice(indices, max_length - \
                                           current_length, replace=True)
            imputed_indices = np.concatenate([indices, add_indices])
        else:
            imputed_indices = indices
        
        imputed_indices_by_type[label] = imputed_indices
    
    return imputed_indices_by_type


def  impute_cells_v1(labels: np.ndarray, shared_labels: List[Union[str, int]]):
    """Make same-size label arrays

    Parameters
    ----------
    labels : numpy.ndarray
        Array of the encoded cell types as integers.

    Returns
    -------

    
    """
    indices_by_type = {label: np.where(labels == label)[0] for label in np.unique(shared_labels)}
    max_length = max(len(indices) for indices in indices_by_type.values())
    imputed_indices_by_type = {}
    for label, indices in indices_by_type.items():
        current_length = len(indices)
        if current_length < max_length:
            add_indices = np.random.choice(indices, max_length - \
                                           current_length, replace=True)
            imputed_indices = np.concatenate([indices, add_indices])
        else:
            imputed_indices = indices
        
        imputed_indices_by_type[label] = imputed_indices
    
    return imputed_indices_by_type


def extract_counts(rna, atac):
     
     rna_count = rna.layers["norm_raw_counts"]
     atac_count = atac.layers["norm_raw_counts"]

     return rna_count, atac_count


def split_partial_data(rna, atac, proportion=None, seed=None):
     
     # Extract counts from anndata objects
     rna_count, atac_count = extract_counts(rna, atac)
     labels = rna.obs["cell_type_encoded"].values
     
     # Randomly select cells from count matrices 
     num_cells_to_select = int(proportion * rna.shape[0])
     selected_cells = np.random.choice(rna.shape[0], size=num_cells_to_select, replace=False)
     rna_selected = rna_count[selected_cells, :]
     atac_selected = atac_count[selected_cells, :]
     labels_selected = labels[selected_cells, :]
     
     # Split the data
     rna_train, rna_test, atac_train, atac_test, labels_train, labels_test = \
        train_test_split(rna_selected, atac_selected, labels_selected, test_size=0.3, 
                         random_state=seed, stratify=labels)
     
     print(f"RNA train set shape: {rna_train.shape} and RNA test set shape: {rna_test.shape}")
     print(f"ATAC train set shape: {atac_train.shape} and ATAC test set shape: {atac_test.shape}")
     print(f"The shape of the train labels: {labels_train.shape} and shape of the test labels: {labels_test.shape}")
     
     return rna_train, rna_test, atac_train, atac_test, labels_train, labels_test
     

def split_full_data(rna, atac, seed=None):


    rna_count, atac_count = extract_counts(rna, atac)
    labels = rna.obs["cell_type_encoded"].values

    rna_train, rna_test, atac_train, atac_test, labels_train, labels_test = \
        train_test_split(rna_count, atac_count, labels, test_size=0.3, 
                         random_state=seed, stratify=labels)

    print(f"RNA train set shape: {rna_train.shape} and RNA test set shape: {rna_test.shape}")
    print(f"ATAC train set shape: {atac_train.shape} and ATAC test set shape: {atac_test.shape}")
    print(f"The shape of the train labels: {labels_train.shape} and shape of the test labels: {labels_test.shape}")

    return rna_train, rna_test, atac_train, atac_test, labels_train, labels_test


# Helper functions for training Con-AAE
def compute_KL_loss(mu, logvar):

        KLloss = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return 0.00000001 * KLloss

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

        batch_size = int(source.size()[0])
        kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss


def accuracy(output, target):
        pred = output.argmax(dim=1).view(-1)
        correct = pred.eq(target.view(-1)).float().sum().item()
        return correct


def dis_accuracy(output,target):
        zero=torch.zeros_like(output)
        one=torch.ones_like(output)
        output=torch.where(output<0.5,zero,output)
        output=torch.where(output>0.5,one,output)
        accuracy=(output==target).sum().item()
        return accuracy


def train_autoencoders(rna_inputs, atac_inputs, rna_model, atac_model, 
                       simple_classifier, fc_classifier, recon_loss, 
                       dis_loss, triplet_loss, cond_loss, rna_opt, atac_opt, 
                       simple_clf_opt, rna_class_labels=None, atac_class_labels=None):
    
    if rna_class_labels is not None:
        rna_class_labels = rna_class_labels.long()
    if atac_class_labels is not None:
        atac_class_labels = atac_class_labels.long()

    rna_model.train()
    atac_model.train()
    fc_classifier.eval()
    simple_classifier.train()

    rna_model.zero_grad()
    atac_model.zero_grad()
    simple_classifier.zero_grad()

    #print(rna_class_labels)
    #print(atac_class_labels)

    rna_latents, rna_recon = rna_model(rna_inputs)
    atac_latents, atac_recon = atac_model(atac_inputs)
    rna_scores = fc_classifier(rna_latents)
    atac_scores = fc_classifier(atac_latents)
    rna_scores = torch.squeeze(rna_scores, dim=1)
    atac_scores = torch.squeeze(atac_scores, dim=1)
    rna_labels = torch.zeros(rna_scores.size(0), ).float()
    rna_labels = rna_labels.to(device="cuda")
    atac_labels = torch.ones(atac_scores.size(0), ).float()
    atac_labels = atac_labels.to(device="cuda")
    rna_class_scores = simple_classifier(rna_latents)
    atac_class_scores = simple_classifier(atac_latents)

    #print(rna_class_scores)
    #print(atac_class_scores)

    # compute losses
    rna_recon_loss = recon_loss(rna_inputs, rna_recon)
    atac_recon_loss = recon_loss(atac_inputs, atac_recon)
    loss = 10.0 * (rna_recon_loss + atac_recon_loss)

    clf_loss = 0.5 * dis_loss(rna_scores, atac_labels) + 0.5 * dis_loss(atac_scores, rna_labels)
    loss += 10.0 * clf_loss

    clf_class_loss = 0.5 * cond_loss(rna_class_scores, rna_class_labels) + \
                    0.5 * cond_loss(atac_class_scores, atac_class_labels)
    loss += 1.0 * clf_class_loss

    inputs = torch.cat((atac_latents, rna_latents), 0)
    labels = torch.cat((atac_class_labels, rna_class_labels), 0)
    tri_loss = triplet_loss(inputs, labels)

    anchor_loss = recon_loss(rna_latents, atac_latents)
    loss += 0.1 * anchor_loss

    atac_latents_recon = rna_model.encoder(rna_model.decoder(atac_latents))
    rna_latents_recon = atac_model.encoder(atac_model.decoder(rna_latents))
    atac_latents_recon_loss = recon_loss(atac_latents, atac_latents_recon)
    rna_latents_recon_loss = recon_loss(rna_latents, rna_latents_recon)
    loss += 10.0 * (atac_latents_recon_loss + rna_latents_recon_loss)

    MMD_loss = mmd_rbf(atac_latents, rna_latents)
    loss += 10 * MMD_loss

    # Update models' parameters
    loss.backward()
    rna_opt.step()
    atac_opt.step()
    simple_clf_opt.step()

    summary_stats = {'rna_recon_loss': rna_recon_loss.item() * rna_latents.size(0),
                     'atac_recon_loss': atac_recon_loss.item() * atac_latents.size(0),
                     'clf_class_loss': clf_class_loss.item() * (rna_latents.size(0) + atac_latents.size(0)),
                     'triplet_loss': tri_loss * rna_latents.size(0),
                     'anchor_loss': anchor_loss * rna_latents.size(0),
                     'atac_latents_recon_loss': atac_latents_recon_loss * atac_latents.size(0),
                     'rna_latents_recon_loss': rna_latents_recon_loss * rna_latents.size(0),
                     'MMD_loss': MMD_loss * rna_latents.size(0),
                     'clf_loss': clf_loss.item() * (rna_latents.size(0) + atac_latents.size(0))}
    
    return summary_stats


def train_classifier(rna_inputs, atac_inputs, rna_model, atac_model,
                     fc_classifier, dis_loss, fc_clf_opt, rna_class_labels=None, 
                    atac_class_labels=None):
    
    rna_model.eval()
    atac_model.eval()
    fc_classifier.train()

    fc_classifier.zero_grad()

    rna_latents, _ = rna_model(rna_inputs)
    atac_latents, _ = atac_model(atac_inputs)

    rna_scores = fc_classifier(rna_latents)
    atac_scores = fc_classifier(atac_latents)

    rna_labels = torch.zeros(rna_scores.size(0),).float()
    rna_labels = rna_labels.to(device="cuda")
    atac_labels = torch.ones(atac_scores.size(0),).float()
    atac_labels = atac_labels.to(device="cuda")
    #print(f"rna_labels dim:{rna_labels.shape}, atac_labels dim:{atac_labels.shape}")

    rna_scores = torch.squeeze(rna_scores,dim=1)
    atac_scores = torch.squeeze(atac_scores,dim=1)
    #print(f"rna_scores dim:{rna_scores.shape}, atac_scores dim:{atac_scores.shape}")

    # Compute losses:

    clf_loss = 0.5 * dis_loss(rna_scores, rna_labels) + \
        0.5 * dis_loss(atac_scores, atac_labels)
    loss = clf_loss

    # Update model
    loss.backward()
    fc_clf_opt.step()
    summary_stats = {'clf_loss': \
                     clf_loss * (rna_scores.size(0)+atac_scores.size(0))}
    
    return summary_stats


# Helper functions for training models

def get_func_name(model_name):
     
     if model_name == "sCIN":
          return "train_sCIN", "get_emb_sCIN"
     
     elif model_name == "AE":
          return "train_ae", "get_emb_ae"
     
     elif model_name == "Con-AAE":
          return "train_con", "get_emb_con"
     
     elif model_name == "Harmony":
          return "train_hm", "get_emb_hm"
     
     elif model_name == "MOFA":
          return "train_mofa", "get_mofa_emb"
     
def get_func_name_unpaired(model_name):
     
     if model_name == "scCOOL":
          return "train_sccool_unpaired", "get_emb_sccool"
     
     elif model_name == "AE":
          return "train_ae_unpaired", "get_emb_ae"
     
     elif model_name == "Con-AAE":
          return "train_con_unpaired", "get_emb_con"


def read_config(file_path):
     
    with open(file_path, "r") as file:
         
         config = json.load(file)

    return config


def load_data(config: Any) -> Tuple:
    
    rna = ad.read_h5ad(config["RNA_DIR"])
    atac = ad.read_h5ad(config["ATAC_DIR"])
   
    return rna, atac


def make_plots(results_df, save_dir):
     
    # Plot Recall@K scores by Model and Replication
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results_df, x='Replicates', y='Recall_at_k', hue='Models', style='Models',
                 markers=True, dashes=False)
    plt.title('Line Plot of Recall@k by Model and Replication')
    plt.xlabel('Replicates')
    plt.ylabel('Recall@k')
    plt.legend(title='Model')
    plt.xticks(ticks=range(results_df['Replicates'].min(), results_df['Replicates'].max() + 1))
    plt.savefig(save_dir + "/model_rep_recall.png")

    # Plot ASW_score by Model and Replication
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results_df, x='Replicates', y='cell_type_ASW', hue='Models', style='Models', 
                 markers=True, dashes=False)
    plt.title('Line Plot of ASW score by Model and Replication')
    plt.xlabel('Replicates')
    plt.ylabel('ASW_score')
    plt.legend(title='Model')
    plt.xticks(ticks=range(results_df['Replicates'].min(), results_df['Replicates'].max() + 1))
    plt.savefig(save_dir + "/model_rep_asw.png")

    # Plot Integration score by Model and Replication
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results_df, x='Replicates', y='cell_type_acc', hue='Models', style='Models', 
                 markers=True, dashes=False)
    plt.title('Line Plot of Cell type accuracy by Model and Replication')
    plt.xlabel('Replicates')
    plt.ylabel('Cell type accuracy')
    plt.legend(title='Model')
    plt.xticks(ticks=range(results_df['Replicates'].min(), results_df['Replicates'].max() + 1))
    plt.savefig(save_dir + "/model_rep_ct_acc.png")

    grouped = results_df.groupby(['Models', 'Replicates']).agg({'Recall_at_k': ['mean', 'std'], 
                                                                'cell_type_ASW': ['mean', 'std'], 
                                                                'cell_type_acc':['mean', 'std']}).reset_index()
    grouped.columns = ['Models', 'Replicates', 'Recall_mean', 'Recall_std', 'ASW_mean', 'ASW_std', 
                       'ct_acc_mean', 'ct_acc__std']

    plt.figure(figsize=(12, 6))
    sns.barplot(data=grouped, x='Models', y='Recall_mean', hue='Replicates', ci=None)
    plt.savefig(save_dir + "/bar_model_recall.png")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=grouped, x='Models', y='ASW_mean', hue='Replicates', ci=None)
    plt.savefig(save_dir + "/bar_model_asw.png")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=grouped, x='Models', y='Recall_mean', hue='Replicates', ci=None)
    plt.savefig(save_dir + "/bar_model_intgn.png")

    models = grouped['Models'].unique()
    data = [grouped[grouped['Models'] == model]['Recall_mean'].values for model in models]
    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=models)
    plt.title('Box Plot of Recall@k by Model')
    plt.xlabel('Models')
    plt.ylabel('Mean Recall@k')
    plt.savefig(save_dir + "/box_model_Recall.png")


    models = grouped['Models'].unique()
    data = [grouped[grouped['Models'] == model]['Recall_mean'].values for model in models]
    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=models)
    plt.title('Box Plot of Cell type accuracy by Model')
    plt.xlabel('Models')
    plt.ylabel('Mean Cell type accuracy')
    plt.savefig(save_dir + "/box_model_intgn.png")

    models = grouped['Models'].unique()
    data = [grouped[grouped['Models'] == model]['ASW_mean'].values for model in models]
    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=models)
    plt.title('Box Plot of ASW score by Model')
    plt.xlabel('Models')
    plt.ylabel('Mean ASW')
    plt.savefig(save_dir + "/box_model_asw.png")


def plot_tsne_original(data: np.array, labels: np.array, 
                       is_show=False, save_dir=None) -> None:
     """
     Plot t-SNE embeddings of the original multi-omics dataset.

     Parameters
     ----------
     data: Original, concatenated, paired multi-omics data.
           It has the shape (num_cells, (num_features1 + num_features2))

     labels: Cell type labels. It has the shape (num_cells,)

     is_show: Whether to show the plot or save it.

     save_dir: save directory

     Return
     ----------
     None
     """

     tsne_embs = TSNE(n_components=2, learning_rate='auto',
                  init='random').fit_transform(data)
     cell_types = np.unique(labels)
     colors = cc.glasbey[:len(cell_types)]
     for i, cell_type in enumerate(cell_types):
          mask = labels == cell_type
          plt.scatter(tsne_embs[mask, 0], tsne_embs[mask, 1], 
                         s=0.5, label=f" {cell_type}", color=colors[i])
     
     plt.tick_params(axis='both', which='major', labelsize=8)
     ax = plt.gca()  
     ax.spines['top'].set_visible(False)
     ax.spines['right'].set_visible(False)
     plt.legend(
     title="Cell Types", 
     bbox_to_anchor=(0.5, -0.1),  
     loc='upper center',          
     fontsize=8, 
     ncol=5,                     
     frameon=False,
     handleheight=1,           
     markerscale=6               
     )

     if is_show:
          plt.show()
     else:
          plt.savefig(os.path.join(save_dir, "tsne_original.png"))


def plot_tsne_embs(joint_embs:np.array, labels:np.array, 
                   is_show=False, save_dir=None) -> None:
     """
     Plot t-SNE embeddings of the joint embeddings from a model.

     Parameters
     ----------
     data: Concatenated joint embeddings.
           It has the shape (num_cells, 2 * emb_dim). Note that num_cells is 
           the number of **test cells**.

     labels: Cell type labels. It has the shape (num_cells,)

     is_show: Whether to show the plot or save it.

     save_dir: save directory

     Return
     ----------
     None
     """

     tsne_embs = TSNE(n_components=2, learning_rate='auto',
                  init='random').fit_transform(joint_embs)
     cell_types = np.unique(labels)
     colors = cc.glasbey[:len(cell_types)]
     for i, cell_type in enumerate(cell_types):
          mask = labels == cell_type
          plt.scatter(tsne_embs[mask, 0], tsne_embs[mask, 1], 
                         s=0.5, label=f" {cell_type}", color=colors[i])
     
     plt.tick_params(axis='both', which='major', labelsize=8)
     ax = plt.gca()  
     ax.spines['top'].set_visible(False)
     ax.spines['right'].set_visible(False)
     plt.legend(
     title="Cell Types", 
     bbox_to_anchor=(0.5, -0.1),  
     loc='upper center',          
     fontsize=8, 
     ncol=5,                     
     frameon=False,
     handleheight=1,           
     markerscale=6               
     )

     if is_show:
          plt.show()
     else:
          plt.savefig(os.path.join(save_dir, "tsne_joint_embs.png"))


def map_cell_types(path:str) -> dict:

     """
     Map encoded cell types to the cell type names.

     Parameters
     ----------
     path: Path to the mapping file.

     Return
     ----------
     ct_mapping: A dictionary containing mapped cell types. 
     """
     
     with open(path, "r") as f:
          ct_mapping = {}
          for line in f:
               k, v = line.strip().split(":", 1)
               ct_mapping[int(k)] = v.strip()
     
     return ct_mapping


def list_embs(embs_dir:str, seeds:list, is_ct=False) -> list:

     embs_paths = []
     for seed in seeds:
          path1 = os.path.join(embs_dir, f"rna_emb{seed}.noy")
          path2 = os.path.join(embs_dir, f"atac_emb{seed}.npy")
          if is_ct:
               path_ct = os.path.join(embs_dir, f"labels_test_{seed}.npy")
               embs_paths.append((path1, path2, path_ct))
          else:
               embs_paths.append((path1, path2))
     
     return embs_paths


def make_unpaired(mod1, mod2, labels, seed, p=0.1):

     np.random.seed(seed)

     mod1_new = np.copy(mod1).astype(float)
     mod2_new = np.full_like(mod2, np.nan, dtype=float)

     mod1_lbls_new = np.copy(labels)
     mod2_lbls_new = np.full_like(labels, -1, dtype=int)

     for lbl in np.unique(labels):

          indices = np.where(labels == lbl)[0]
          mod2_cells = np.random.choice(indices, 
                                      max(1, round(p * len(indices))), 
                                      replace=False)
          print(f"Mod2 cells is: {mod2_cells}")
          mod1_cells = np.setdiff1d(indices, mod2_cells, 
                                    assume_unique=True)
          
          mod1_new[mod2_cells] = np.nan
          mod1_lbls_new[mod2_cells] = -1

          mod2_new[mod2_cells] = mod2[mod2_cells]
          mod2_lbls_new[mod2_cells] = labels[mod2_cells]

     return mod1_new, mod1_lbls_new, mod2_new, mod2_lbls_new


def make_unpaired_random(mod1, mod2, labels, seed, num_mod2_ct=5):

     np.random.seed(seed)

     mod1_new = np.copy(mod1).astype(float)
     mod1_lbls_new = np.copy(labels)

     mod2_new = np.full_like(mod2, np.nan, dtype=float)
     mod2_lbls_new = np.full_like(labels, -1, dtype=int)

     num_cells_total = len(mod1)
     #num_ct = len(np.unique(labels))

     for lbl in np.unique(labels):
          
          mod2_cells = np.random.choice(num_cells_total, num_mod2_ct,
                                   replace=False)
          
          mod2_new[mod2_cells] = mod2[mod2_cells]
          mod2_lbls_new[mod2_cells] = lbl

          mod1_new[mod2_cells] = np.nan
          mod1_lbls_new[mod2_cells] = -1

     return mod1_new, mod1_lbls_new, mod2_new, 


def encode_cell_types(anndata: ad.AnnData) -> ad.AnnData:
     """
     Encode cell type labels to integers in (0, num_cell_types).

     Parameters
     ----------
     adata: ad.AnnData
          Input data
     
     Returns
     -------
     ad.AnnData
          Modified anndata with 'cell_type_encoded' in 'obs' layer
          and 'cell_type_mapping' in 'uns' layer.
     """
     if not isinstance(anndata.obs["cell_type"].dtype, pd.CategoricalDtype):
          anndata.obs["cell_type"] = anndata.obs["cell_type"].astype("category")

     anndata.obs["cell_type_encoded"] = anndata.obs["cell_type"].cat.codes.to_numpy()
     cell_type_mapping = dict(enumerate(anndata.obs["cell_type"].cat.categories))
     anndata.uns["cell_type_mapping"] = cell_type_mapping

     return anndata