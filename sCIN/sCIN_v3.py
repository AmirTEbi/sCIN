# TODO: Implement residual connections.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.decomposition import PCA
from sCIN.utils import impute_cells
from itertools import cycle
from typing import *
import os


class Mod1Encoder(nn.Module):
    """ An encoder with three layers"""
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(Mod1Encoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(in_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, latent_dim)
    
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """
        Forward computation in the network.

        Parameters
        ----------
        x : torch.tensor
            Mod1 data matrix
            
        Returns
        -------
        z_mod1 : torch.tensor
            Mod1 embeddings 
        """
        h1 = self.linear1(x)
        h1 = self.bn1(h1)
        h1 = self.relu(h1)
        h2 = self.linear1(h1)
        h2 = self.bn1(h2)
        h2 = self.relu(h2)
        z_mod1 = self.linear2(h2 + h1)

        return z_mod1


class Mod2Encoder(nn.Module):
    """ An encoder with three layers"""
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(Mod1Encoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(in_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, latent_dim)
    
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """
        Forward computation in the network.

        Parameters
        ----------
        x : torch.tensor
            Mod1 data matrix
            
        Returns
        -------
        z_mod1 : torch.tensor
            Mod1 embeddings 
        """
        h1 = self.linear1(x)
        h1 = self.bn1(h1)
        h1 = self.relu(h1)
        h2 = self.linear1(h1)
        h2 = self.bn1(h2)
        h2 = self.relu(h2)
        z_mod2 = self.linear2(h2 + h1)

        return z_mod2


class sCIN(nn.Module):
    """ The main implementation of sCIN."""
    def __init__(self, mod1_encoder, mod2_encoder, t): 
        super(sCIN, self).__init__()
        
        self.mod1_encoder = mod1_encoder
        self.mod2_encoder = mod2_encoder
        self.W_mod1 = nn.Parameter(torch.randn((self.mod1_encoder.latent_dim, 
                                               self.mod1_encoder.latent_dim)))
        self.W_mod2 = nn.Parameter(torch.randn((self.mod2_encoder.latent_dim, 
                                                self.mod2_encoder.latent_dim)))
        self.t = nn.Parameter(torch.tensor(t))  


    def forward(self, mod1_data: torch.tensor, mod2_data:torch.tensor):
        """
        Forward computation in the network.

        Parameters
        ----------
        mod1_data : torch.tensor
                First modality data matrix.
        mod2_data : torch.tensor
            Second modality data matrix.

        Returns
        ----------
        logits : torch.tensor
            Similarity matrix between mod1 and mod2 embeddings
        mod1_emb : torch.tensor
            Mod1 embeddings
        mod2_emb : torch.tensor
            Mod2 embeddings 
        """
        mod1_f = self.mod1_encoder(mod1_data)
        mod2_f = self.mod2_encoder(mod2_data)
        mod1_emb = F.normalize(torch.matmul(mod1_f, self.W_mod1), p=2, dim=1)
        mod2_emb = F.normalize(torch.matmul(mod2_f, self.W_mod2), p=2, dim=1)
        logits = torch.matmul(mod1_emb, mod2_emb.t()) * torch.exp(self.t)

        return logits, mod1_emb, mod2_emb
    

def train_sCIN_v3(mod1_train: np.ndarray, 
                  mod2_train: np.ndarray, 
                  labels_train: np.ndarray, 
                  settings: Dict[str, Any],
                  save_dir: str, 
                  is_pca: Optional[bool] = True,
                  rep: Optional[Union[int, str]] = "NA") -> Dict[str, Any]:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = settings["epochs"]
    t = settings.get("t", 0.07)
    lr = settings.get("learning_rate", 1e-3)
    hidden_dim = settings.get("hidden_dim", 256)
    latent_dim = settings.get("latent_dim", 128)
    bob = settings.get("bob", 10)
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
    scin = sCIN(mod1_encoder, mod2_encoder, t)
    scin.to(device)
    optimizer = Adam(scin.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        scin.train()
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
            logits, _, _ = scin(mod1_batch, mod2_batch)
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
    os.makedirs(model_dir, exist_ok=True)
    torch.save(scin.state_dict(), os.path.join(model_dir, f"sCIN_{rep}.pth"))

    train_dict = {"model":scin}
    if is_pca:
        train_dict["pca_mod1"] = pca_mod1
        train_dict["pca_mod2"] = pca_mod2

    return train_dict



def get_emb_sCIN_v3(mod1_test: np.ndarray, 
                    mod2_test: np.ndarray, 
                    train_dict: Dict[str, Any], 
                    save_dir: str, 
                    is_pca: Optional[bool] = True,
                    rep: Optional[Union[int, str]] = "NA") -> Tuple[np.ndarray, np.ndarray]:
    """
    Get embedding from unseen test data by the trained model.

    Parameters
    ----------
    mod1_test: np.ndarray
        First modality data matrix.

    mod2_test: np.ndarray
        Second modality data matrix.

    train_dict: Dict[str, Any]
        The outputs of the training function, including the trained model 
        and the fitted PCA model for each modality.

    save_dir: str
        The directory to save the embeddings.

    is_pca: Optional[bool]
        Do you used PCA during training?
    
    rep: Optional[int]
        In which replication are you?

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Embeddings for each modality.
    
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train_dict["model"]

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

    logits_df = pd.DataFrame(logits_test)
    logits_df.to_csv(os.path.join(embs_dir, f"logits_test_{rep}.csv"), index=False)
    
    return mod1_emb_np, mod2_emb_np