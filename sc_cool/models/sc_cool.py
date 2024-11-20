import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.decomposition import PCA
from utils import impute_cells


class RNAEncoder(nn.Module):
    """ An encoder with three layers"""
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(RNAEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)
    
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """Forward methods

        Parameters
        ----------
        x : torch.tensor
            RNA count tensor
            

        Returns
        -------
        z_rna : torch.tensor
            RNA embeddings (Dimension: (x.shape[0], 128))
        """
        h = self.linear1(x)
        h1 = self.bn(h)
        h2 = self.relu(h1)
        z_rna = self.linear2(h2)

        return z_rna


class ATACEncoder(nn.Module):
    """ An encoder with three layers"""
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(ATACEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """Forward method

        Parameters
        ----------
        x : torch.tensor
            ATAC count tensor

        Returns
        -------
        z_atac : torch.tensor
            ATAC embeddings (Dimension: (x.shape[0], 128))
        """
        h = self.linear1(x)
        h1 = self.bn(h)  # BatchNorm added before relu as Deep Learning (Bishop, 2023) suggests
        h2 = self.relu(h1)
        z_atac = self.linear2(h2)

        return z_atac
    
class scCOOL(nn.Module):
    """ Contrastive learning model inspired by CLIP"""
    def __init__(self, rna_encoder, atac_encoder, t): 
        super(scCOOL, self).__init__()
        
        self.rna_encoder = rna_encoder
        self.atac_encoder = atac_encoder
        self.W_rna = nn.Parameter(torch.randn((self.rna_encoder.latent_dim, 
                                               self.rna_encoder.latent_dim)))
        self.W_atac = nn.Parameter(torch.randn((self.atac_encoder.latent_dim, 
                                                self.atac_encoder.latent_dim)))
        self.t = nn.Parameter(torch.tensor(t))  # make 't' a learnable parameter

    def forward(self, rna, atac):
        """Forward method

        Parameters
        ----------
        rna : torch.tensor
            RNA count tensor
        atac : torch.tensor
            ATAC count tensor

        Returns
        -------
        logits : torch.tensor
            Similarity matrix between RNA and ATAC embeddings
        rna_emb : torch.tensor
            RNA embeddings (Dimension: (rna.shape[0], 128))
        atac_emb : torch.tensor
            ATAC embeddings (Dimension: (atac.shape[0], 128))
        """
        rna_f = self.rna_encoder(rna)
        atac_f = self.atac_encoder(atac)
        rna_emb = F.normalize(torch.matmul(rna_f, self.W_rna), p=2, dim=1)
        atac_emb = F.normalize(torch.matmul(atac_f, self.W_atac), p=2, dim=1)
        logits = torch.matmul(rna_emb, atac_emb.t()) * torch.exp(self.t)

        return logits, rna_emb, atac_emb


def train_sccool(rna_train, atac_train, labels_train, epochs, settings,
               device=None):
    
    t = settings["t"]
    lr = settings["lr"]
    hidden_dim = settings["hidden_dim"]
    latent_dim = settings["latent_dim"]
    device = device
    bob = settings["bob"]

    # PCA transformations
    print("PCA transformation ...")
    pca_rna = PCA(n_components=settings["PCs"])
    pca_atac =PCA(n_components=settings["PCs"])
    pca_rna.fit(rna_train)
    pca_atac.fit(atac_train)
    rna_train = pca_rna.transform(rna_train)
    atac_train = pca_atac.transform(atac_train)
    print("PCA finished.")

    # Arrays to tensors
    rna_train_t = torch.from_numpy(rna_train)
    rna_train_t = rna_train_t.to(torch.float32)
    rna_train_t = rna_train_t.to(device)
    atac_train_t = torch.from_numpy(atac_train).to(torch.float32)
    atac_train_t = atac_train_t.to(torch.float32)
    atac_train_t = atac_train_t.to(device)
    #labels_train_t = torch.from_numpy(labels_train)
    #labels_train_t = labels_train_t.to(device)

    num_classes = len(np.unique(labels_train))
    target = torch.arange(num_classes)
    target = target.to(device)
    imputed_indices_by_type = impute_cells(labels_train)
    ##### DEBUG
    imputed_indices_by_type[0] == imputed_indices_by_type[1]
    print(len(imputed_indices_by_type[0]))
    #####
    label_indices = []
    for key in imputed_indices_by_type.keys():
        label_indices.append(imputed_indices_by_type[key])

    rna_encoder = RNAEncoder(rna_train_t.shape[1], hidden_dim, latent_dim)
    rna_encoder.to(device)
    atac_encoder = ATACEncoder(atac_train_t.shape[1], hidden_dim, latent_dim)
    atac_encoder.to(device)
    cool = scCOOL(rna_encoder, atac_encoder, t)
    cool.to(device)
    optimizer = Adam(cool.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):

        cool.train()
        epoch_loss = 0.0
        total_samples = 0

        # Select one cell per cell type in order
        for i in range(0, len(label_indices[0]), bob):
            idx = []
            for j in range(i, i+bob):
                for k in label_indices:
                    idx.append(k[j])

            # Create mini-batch
            rna_l = [rna_train_t[l, :] for l in idx]
            atac_l = [atac_train_t[m, :] for m in idx]
            rna_batch = torch.vstack(rna_l)
            atac_batch = torch.vstack(atac_l)

            # Compute similarity logits
            logits, _, _ = cool(rna_batch, atac_batch)

            # Select diagonal blocks of logits, compute loss and 
            # sum losses for all blocks 
            batch_logits = []
            block_tensor_shape = (num_classes, num_classes)
            for l in range(bob):
                start_row = l * block_tensor_shape[0]
                end_row = start_row + block_tensor_shape[0]
                start_col = l * block_tensor_shape[1]
                end_col = start_col + block_tensor_shape[1]
                sub_logits = logits[start_row:end_row, start_col:end_col]
                batch_logits.append(sub_logits)
                losses = [F.cross_entropy(logit, target) + F.cross_entropy(logit.T, target) \
                          for logit in batch_logits]
                batch_loss = sum(losses)

            # Update model parameters
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item() * num_classes * bob
            total_samples += num_classes * bob

        # Compute epoch loss
        epoch_loss /= total_samples
        print(f"Epoch: {epoch} | Loss: {epoch_loss:.4f}")

    return [cool, pca_rna, pca_atac]


def get_emb_sccool(rna_test, atac_test, labels_test, obj_list, save_dir=None, seed=None, device=None):

    model = obj_list[0]
    pca_rna = obj_list[1]
    pca_atac = obj_list[2]

    print("PCA transformation of test data ...")
    rna_test = pca_rna.transform(rna_test)
    atac_test = pca_atac.transform(atac_test)
    print("PCA finished.")

    # Arrays to tensors
    rna_test_t = torch.from_numpy(rna_test)
    rna_test_t = rna_test_t.to(torch.float32)
    rna_test_t = rna_test_t.to(device)
    atac_test_t = torch.from_numpy(atac_test).to(torch.float32)
    atac_test_t = atac_test_t.to(torch.float32)
    atac_test_t = atac_test_t.to(device)

    with torch.no_grad():
        logits_test, rna_emb, atac_emb = model(rna_test_t, atac_test_t)

    print("Embeddings were generated.")

    rna_emb_np = rna_emb.cpu().numpy()
    atac_emb_np = atac_emb.cpu().numpy()
    logits_test = logits_test.cpu().numpy()
    np.save(save_dir + f"/labels_test_{seed}.npy", labels_test)
    np.save(save_dir + f"/logits_test_{seed}.npy", logits_test)

    return rna_emb_np, atac_emb_np