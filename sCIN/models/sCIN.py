import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.decomposition import PCA
from sCIN.utils.utils import impute_cells
from itertools import cycle
import os


class Mod1Encoder(nn.Module):
    """ An encoder with three layers"""
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(Mod1Encoder, self).__init__()
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
        z_mod1 = self.linear2(h2)

        return z_mod1


class Mod2Encoder(nn.Module):
    """ An encoder with three layers"""
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(Mod2Encoder, self).__init__()
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
        z_mod2 = self.linear2(h2)

        return z_mod2
    
class scCOOL(nn.Module):
    """ Contrastive learning model inspired by CLIP"""
    def __init__(self, mod1_encoder, mod2_encoder, t): 
        super(scCOOL, self).__init__()
        
        self.mod1_encoder = mod1_encoder
        self.mod2_encoder = mod2_encoder
        self.W_mod1 = nn.Parameter(torch.randn((self.mod1_encoder.latent_dim, 
                                               self.mod1_encoder.latent_dim)))
        self.W_mod2 = nn.Parameter(torch.randn((self.mod2_encoder.latent_dim, 
                                                self.mod2_encoder.latent_dim)))
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
        ----------
        logits : torch.tensor
            Similarity matrix between RNA and ATAC embeddings
        rna_emb : torch.tensor
            RNA embeddings (Dimension: (rna.shape[0], 128))
        atac_emb : torch.tensor
            ATAC embeddings (Dimension: (atac.shape[0], 128))
        """
        mod1_f = self.mod1_encoder(rna)
        mod2_f = self.mod2_encoder(atac)
        mod1_emb = F.normalize(torch.matmul(mod1_f, self.W_mod1), p=2, dim=1)
        mod2_emb = F.normalize(torch.matmul(mod2_f, self.W_mod2), p=2, dim=1)
        logits = torch.matmul(mod1_emb, mod2_emb.t()) * torch.exp(self.t)

        return logits, mod1_emb, mod2_emb


def train_sCIN(mod1_train: np.array, mod2_train: np.array, 
               labels_train: list, epochs: int, settings: dict, 
               **kwargs) -> dict:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t = settings.get("t", 0.07)
    lr = settings.get("lr", 1e-3)
    hidden_dim = settings.get("hidden_dim", 256)
    latent_dim = settings.get("latent_dim", 128)
    bob = settings.get("bob", 10)
    save_dir = kwargs["save_dir"]
    seed = kwargs["seed"]
    is_pca = kwargs["is_pca"]
    patience = settings.get("patience", 10)
    min_delta = settings.get("min_delta", 1e-4)

    if is_pca:
        # PCA transformations
        print("PCA transformation ...")
        PCs1 = settings.get("PCs", min(mod1_train.shape[0], mod1_train.shape[1]))
        PCs2 = settings.get("PCs", min(mod2_train.shape[0], mod2_train.shape[1]))
        pca_mod1 = PCA(n_components=PCs1)
        pca_mod2 =PCA(n_components=PCs2)
        pca_mod1.fit(mod1_train)
        pca_mod2.fit(mod2_train)
        mod1_train = pca_mod1.transform(mod1_train)
        mod2_train = pca_mod2.transform(mod2_train)
        print("PCA finished.")

    # Arrays to tensors
    mod1_train_t = torch.from_numpy(mod1_train)
    mod1_train_t = mod1_train_t.to(torch.float32)
    mod1_train_t = mod1_train_t.to(device)
    mod2_train_t = torch.from_numpy(mod2_train).to(torch.float32)
    mod2_train_t = mod2_train_t.to(torch.float32)
    mod2_train_t = mod2_train_t.to(device)

    num_classes = len(np.unique(labels_train))
    target = torch.arange(num_classes)
    target = target.to(device)
    imputed_cell_types = impute_cells(labels_train)
    lbls_cycle = {ct: cycle(indices) for ct, indices in imputed_cell_types.items()}

    mod1_encoder = Mod1Encoder(mod1_train_t.shape[1], hidden_dim, latent_dim)
    mod1_encoder.to(device)
    mod2_encoder = Mod2Encoder(mod2_train_t.shape[1], hidden_dim, latent_dim)
    mod2_encoder.to(device)
    cool = scCOOL(mod1_encoder, mod2_encoder, t)
    cool.to(device)
    optimizer = Adam(cool.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        cool.train()
        epoch_loss = 0.0
        total_samples = 0
        total_batches = len(imputed_cell_types[0])

        for batch in range(0, total_batches, bob):
            mod1_batch = []
            mod2_batch = []

            for _ in range(bob):
                for ct in range(num_classes):
                    cell = next(lbls_cycle[ct])
                    mod1_batch.append(mod1_train_t[cell, :])
                    mod2_batch.append(mod2_train_t[cell, :])

            mod1_batch = torch.vstack(mod1_batch)
            mod2_batch = torch.vstack(mod2_batch)
            logits, _, _ = cool(mod1_batch, mod2_batch)
            block_size = num_classes
            losses = []
            for b in range(bob):
                start = b * block_size
                end = start + block_size
                sub_logits = logits[start:end, start:end]
                losses.append(F.cross_entropy(sub_logits, target) + \
                              F.cross_entropy(sub_logits.T, target))
            
            batch_loss = sum(losses)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
            total_samples += bob * num_classes

        epoch_loss /= total_samples
        print(f"Epoch: {epoch} | Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.4f}")
                break
    
    model_dir = os.path.join(save_dir, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(cool.state_dict(), os.path.join(model_dir, f"sCIN_{seed}"))

    train_dict = {"model":cool}
    if is_pca:
        train_dict["pca_mod1"] = pca_mod1
        train_dict["pca_mod2"] = pca_mod2

    return train_dict


def pca_with_nans(data1, data2, n_components):

    nan_mask1 = np.isnan(data1).all(axis=1)
    nan_mask2 = np.isnan(data2).all(axis=1)

    non_nans1 = data1[~nan_mask1]
    non_nans2 = data2[~nan_mask2]
    #print(non_nans2)

    PCs1 = min(non_nans1.shape[0], non_nans1.shape[1], n_components)
    PCs2 = min(non_nans2.shape[0], non_nans2.shape[1], n_components)
    pca1 = PCA(n_components=PCs1)
    pca2 = PCA(n_components=PCs2)
    pca1.fit(non_nans1)
    pca2.fit(non_nans2)
    non_nans_pca1 = pca1.transform(non_nans1)
    non_nans_pca2 = pca2.transform(non_nans2)

    data1_pca = np.full((data1.shape[0], PCs1), np.nan, dtype=float)
    data2_pca = np.full((data2.shape[0], PCs2), np.nan, dtype=float)
    data1_pca[~nan_mask1] = non_nans_pca1
    data2_pca[~nan_mask2] = non_nans_pca2

    return data1_pca, data2_pca, pca1, pca2


def train_sCIN_unpaired(mod1_train: np.array, mod2_train: np.array, 
                        labels_train: list, epochs: int, settings: dict, **kwargs) -> dict:
    """
    Train sCIN model in an unpaired setting.

    Parameters
    ----------
    mod1_train: Training data for the first modality.
    mod2_train: Training data for the second modality.
    labels_train: List of the training labeles for both modalitis.
        Example: [labels_train1, labels_train2]
    epochs: Training epochs
    settings: Model's hyper-parameters and options.

    Return
    ----------
    train_dict: A dictionary containing the trained model and PCA transformations 
    of both modalities.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t = settings["t"]
    lr = settings["lr"]
    hidden_dim = settings["hidden_dim"]
    latent_dim = settings["latent_dim"]
    bob = settings["bob"]
    save_dir = kwargs["save_dir"]
    seed = kwargs["seed"]
    is_pca = kwargs["is_pca"]
    patience = settings.get("patience", 10)
    min_delta = settings.get("min_delta", 1e-4)

    if is_pca:
        # PCA transformations
        print("PCA transformation ...")

        mod1_train, mod2_train, pca_mod1, pca_mod2 = pca_with_nans(mod1_train, mod2_train, settings["PCs"])

        print("PCA finished.")

    # Arrays to tensors
    mod1_train_t = torch.from_numpy(mod1_train)
    mod1_train_t = mod1_train_t.to(torch.float32)
    mod1_train_t = mod1_train_t.to(device)
    mod2_train_t = torch.from_numpy(mod2_train).to(torch.float32)
    mod2_train_t = mod2_train_t.to(torch.float32)
    mod2_train_t = mod2_train_t.to(device)

    lbls1 = labels_train[0]
    lbls2 = labels_train[1]
    num_classes = len(np.unique(lbls1))-1
    target = torch.arange(num_classes)
    target = target.to(device)
    imputed_cell_types1 = impute_cells(lbls1)
    imputed_cell_types2 = impute_cells(lbls2)
    
    lbls_cycle1 = {ct: cycle(indices) for ct, indices in imputed_cell_types1.items()}
    lbls_cycle2 = {ct: cycle(indices) for ct, indices in imputed_cell_types2.items()}

    mod1_encoder = Mod1Encoder(mod1_train_t.shape[1], hidden_dim, latent_dim)
    mod1_encoder.to(device)
    mod2_encoder = Mod2Encoder(mod2_train_t.shape[1], hidden_dim, latent_dim)
    mod2_encoder.to(device)
    cool = scCOOL(mod1_encoder, mod2_encoder, t)
    cool.to(device)
    optimizer = Adam(cool.parameters(), lr=lr)
    
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        cool.train()
        epoch_loss = 0.0
        total_samples = 0
        total_batches = len(imputed_cell_types2[0])

        for batch in range(0, total_batches, bob):
            mod1_batch = []
            mod2_batch = []

            for _ in range(bob):

                for ct in range(num_classes):
                    #print(f"Cell type is: {ct}")
                    if ct == -1:
                        continue
                    else:
                        mod1_cell = next(lbls_cycle1[ct])
                        mod2_cell = next(lbls_cycle2[ct])
                        #print(mod2_cell)

                        mod1_batch.append(mod1_train_t[mod1_cell, :])
                        mod2_batch.append(mod2_train_t[mod2_cell, :])
                
            mod1_batch = torch.vstack(mod1_batch)
            #print(mod1_batch)
            mod2_batch = torch.vstack(mod2_batch)
            #print(mod2_batch)

            logits, _, _ = cool(mod1_batch, mod2_batch)
            block_size = num_classes
            losses = []
            for b in range(bob):
                start = b * block_size
                end = start + block_size
                sub_logits = logits[start:end, start:end]
                losses.append(F.cross_entropy(sub_logits, target) + \
                              F.cross_entropy(sub_logits.T, target))
            
            batch_loss = sum(losses)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
            total_samples += bob * num_classes

        epoch_loss /= total_samples
        print(f"Epoch: {epoch} | Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.4f}")
                break

    model_dir = os.path.join(save_dir, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(cool.state_dict(), os.path.join(model_dir, f"sCIN_{seed}"))

    train_dict = {"model":cool}
    if is_pca:
        train_dict["pca_mod1"] = pca_mod1
        train_dict["pca_mod2"] = pca_mod2
    
    return train_dict

        
def get_emb_sCIN(mod1_test, mod2_test, labels_test, 
                   train_dict, save_dir, **kwargs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train_dict["model"]
    seed = kwargs["seed"]
    is_pca = kwargs["is_pca"]

    if is_pca:
        print("PCA transformation of test data ...")
        pca_mod1 = train_dict["pca_mod1"]
        pca_mod2 = train_dict["pca_mod2"]
        mod1_test = pca_mod1.transform(mod1_test)
        mod2_test = pca_mod2.transform(mod2_test)
        print("PCA finished.")

    # Arrays to tensors
    mod1_test_t = torch.from_numpy(mod1_test)
    mod1_test_t = mod1_test_t.to(torch.float32)
    mod1_test_t = mod1_test_t.to(device)
    mod2_test_t = torch.from_numpy(mod2_test).to(torch.float32)
    mod2_test_t = mod2_test_t.to(torch.float32)
    mod2_test_t = mod2_test_t.to(device)

    with torch.no_grad():
        logits_test, mod1_emb, mod2_emb = model(mod1_test_t, mod2_test_t)

    print("Embeddings were generated.")

    mod1_emb_np = mod1_emb.cpu().numpy()
    mod2_emb_np = mod2_emb.cpu().numpy()
    logits_test = logits_test.cpu().numpy()

    embs_dir = os.path.join(save_dir, "embs")
    if not os.path.exists(embs_dir):
        os.makedirs(embs_dir)

    np.save(os.path.join(embs_dir, f"labels_test_{seed}.npy"), labels_test)
    np.save(os.path.join(embs_dir, f"logits_test_{seed}.npy"), logits_test)
    np.save(os.path.join(embs_dir, f"rna_emb{seed}.npy"), mod1_emb_np)
    np.save(os.path.join(embs_dir, f"atac_emb{seed}.npy"), mod2_emb_np)
    
    return mod1_emb_np, mod2_emb_np