"""Utility functions"""

import numpy as np
import anndata as ad
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, IncrementalPCA
import torch
import networkx as nx
import json
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
from typing import Any, Tuple

def impute_cells(labels):
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
     
     if model_name == "scCOOL":
          return "train_sccool", "get_emb_sccool"
     
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


def select_unpaired_cells_by_type(label_indices, bob, rng):

    idx_rna = []
    idx_atac = []
    for j in range(bob):
        for k in label_indices:
            idx_rna.append(k[rng.integers(0, len(k))])
            idx_atac.append(k[rng.integers(0, len(k))])

    return idx_rna, idx_atac


def select_cell_types(adata:ad.AnnData, save_dir:str) -> None:
     
     cell_types = adata.obs["cell_type"]
     ct_arr = np.array(cell_types.values)
     print(ct_arr.shape)

     _, ct_test = train_test_split(ct_arr, test_size=0.3, random_state=0)
     print(ct_test.shape)

     np.save(save_dir + "/test_cell_types_seed_0.npy", ct_test)


def shuffle_per_cell_type(data: np.array, labels: np.array, seed: int) -> np.array:
     """
     Shuffle cells within each cell type category.

     Parameters
     ----------
     data: np.array
        Input **training** dataset.

     labels: np.array
        Cell type labels for each cell.

     seed: int
        Random seed for reproducibility.

    Return
    ----------
    shuffled_data: np.array
    
     """
     
     if seed is not None:
          np.random.seed(seed)

     shuffled_data = np.empty_like(data)
     
     df = pd.DataFrame(data)
     df["labels"] = labels

     for lbl in np.unique(labels):
          cell_type_indices = df[df["labels"] == lbl].index
          shuffled_indices = np.random.permutation(cell_type_indices)
          shuffled_data[cell_type_indices, :] = data[shuffled_indices, :]

     return shuffled_data


def random_shuffle(data: np.array, seed: int) -> np.array:
     
     np.random.seed(seed)

     return data[np.random.permutation(data.shape[0])]


def remove_rows(data: np.array, rm_frac: float
                , seed: int) -> np.array:
     """
     Remove random rows from a 2-D numpy array.

     Parameters
     ----------
     data: np.array

     rm_frac: float
     Fraction of rows to randomly remove from the second modality 
        (default: 0.1)

     seed: int
        Random seed for reproducibility.

     Return
     ----------
     np.array
     """
     np.random.seed(seed)
     num_to_remove = int(len(data) * rm_frac)
     indices_to_keep = np.random.choice(len(data),
                                        len(data) - num_to_remove,
                                        replace=False)
     return data[indices_to_keep]


def make_extreme_unpaired(data1: np.array, 
                          data2:np.array,
                          cell_types:np.array,
                          seed:int,
                          rm_frac: float = 0.1) -> Tuple[np.array, np.array]:
     """
     In each modality, shuffle cells per cell type and select half
     of them for one modality and another half for the other modality.

     Parameters
     ----------
     data1: np.array
        First modality's count matrix.

     data2: np.array
        Second modality's count matrix.

     cell_types: np.array
        Encoded cell types.
    
     seed: int
        Random seed for reproducibility.

     rm_frac: float
        Fraction of rows to randomly remove from the second modality 
        (default: 0.1)
    
     Return
     ----------
     data1_unpaired, data2_unpaired: Tuple
        unpaired datasets for both modalities.
     """
     data1_unpaired_l = []
     data2_unpaired_l = []
     ct_unpaired_l = []

     for ct in np.unique(cell_types):
          indices = np.where(cell_types == ct)[0]

          np.random.seed(seed)  

          midpoint = len(indices) // 2
          data1_indices = indices[:midpoint]
          data2_indices = indices[midpoint:]

          data1_unpaired_l.append(data1[data1_indices])
          data2_unpaired_l.append(data2[data2_indices])
          ct_unpaired_l.append(cell_types[data1_indices])

     data1_unpaired = np.vstack(data1_unpaired_l)
     data2_unpaired = np.vstack(data2_unpaired_l)
     ct_unpaired = np.concatenate(ct_unpaired_l)

     shuffled_cells = np.random.permutation(len(data1_unpaired))
     shuff_data1_unpaired = data1_unpaired[shuffled_cells]
     shuff_ct_unpaired = ct_unpaired[shuffled_cells]

     if rm_frac > 0:
          data2_unpaired_ = remove_rows(data2_unpaired, 
                                        rm_frac,
                                        seed=seed)
     

     return shuff_data1_unpaired, data2_unpaired_, shuff_ct_unpaired