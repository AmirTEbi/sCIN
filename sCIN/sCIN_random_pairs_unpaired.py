import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.decomposition import PCA
from sCIN.utils import impute_cells_v1
from itertools import cycle
from typing import *
import logging
import os


def setup_sCIN_logger():

    logger = logging.getLogger("sCIN_v1")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

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
        """
        Forward computation in the network.

        Parameters
        ----------
        x : torch.tensor
            Mod2 count tensor

        Returns
        -------
        z_mod2 : torch.tensor
            Mod2 embeddings 
        """
        h = self.linear1(x)
        h1 = self.bn(h)  
        h2 = self.relu(h1)
        z_mod2 = self.linear2(h2)

        return z_mod2
    

class sCIN(nn.Module):
    """ The main implementation of sCIN."""
    def __init__(self, in_dim1, in_dim2,  hidden_dim, latent_dim, t): 
        super(sCIN, self).__init__()
        
        self.mod1_encoder = Mod1Encoder(in_dim1, hidden_dim, latent_dim)
        self.mod2_encoder = Mod2Encoder(in_dim2, hidden_dim, latent_dim)

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
    

def train_sCIN_ablated_unpaired(mod1_train: np.ndarray, 
                                mod2_train: np.ndarray, 
                                mod1_labels_train: np.ndarray,
                                mod2_labels_train: np.ndarray, 
                                settings: Dict[str, Any],
                                save_dir: str, 
                                is_pca: Optional[bool] = True,
                                rep: Optional[Union[int, str]] = "NA",
                                num_epochs: Optional[int] = None) -> Dict[str, Any]: 
    
    logger = setup_sCIN_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not num_epochs == None:
        epochs = num_epochs

    epochs = settings["num_epochs"]
    t = settings.get("t", 0.07)
    lr = settings.get("learning_rate", 1e-3)
    hidden_dim = settings.get("hidden_dim", 256)
    latent_dim = settings.get("latent_dim", 128)
    bob = settings.get("bob", 10)
    patience = settings.get("patience", 10)
    min_delta = settings.get("min_delta", 1e-4)

    if is_pca:
        # PCA transformations
        logger.info("PCA transformation ...")
        PCs1 = settings.get("PCs", min(mod1_train.shape[0], mod1_train.shape[1]))
        PCs2 = settings.get("PCs", min(mod2_train.shape[0], mod2_train.shape[1]))
        pca_mod1 = PCA(n_components=PCs1)
        pca_mod2 =PCA(n_components=PCs2)
        pca_mod1.fit(mod1_train)
        pca_mod2.fit(mod2_train)
        mod1_train = pca_mod1.transform(mod1_train)
        mod2_train = pca_mod2.transform(mod2_train)
        logger.info("PCA finished.")

    # Arrays to tensors
    mod1_train_t = torch.from_numpy(mod1_train)
    mod1_train_t = mod1_train_t.to(torch.float32)
    mod1_train_t = mod1_train_t.to(device)
    mod2_train_t = torch.from_numpy(mod2_train).to(torch.float32)
    mod2_train_t = mod2_train_t.to(torch.float32)
    mod2_train_t = mod2_train_t.to(device)

    shared_labels = np.intersect1d(mod1_labels_train, mod2_labels_train).tolist()
    # mod1_imputed_cell_types = impute_cells_v1(mod1_labels_train, shared_labels)
    # mod2_imputed_cell_types = impute_cells_v1(mod2_labels_train, shared_labels)

    # lbls_cycle1 = {ct: cycle(indices) for ct, indices in mod1_imputed_cell_types.items()}
    # lbls_cycle2 = {ct: cycle(indices) for ct, indices in mod2_imputed_cell_types.items()}

    num_classes = len(np.unique(shared_labels))
    target = torch.arange(num_classes)
    target = target.to(device)

    scin = sCIN(in_dim1=mod1_train_t.shape[0],
                in_dim2=mod2_train_t.shape[0],
                hidden_dim=hidden_dim,
                latent_dim=latent_dim)
    
    scin.to(device)
    optimizer = Adam(scin.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        scin.train()
        epoch_loss = 0.0
        N1, N2 = mod1_train_t.shape[0], mod2_train_t.shape[0]
        total_batches = (max(N1, N2) + batch_size - 1) // batch_size

        for _ in range(total_batches):
            idx1 = torch.randperm(N1, device=device)[:batch_size]
            idx2 = torch.randperm(N2, device=device)[:batch_size]

            mod1_batch = mod1_train_t[idx1]
            mod2_batch = mod2_train_t[idx2] 

            logits, _, _ = scin(mod1_batch, mod2_batch)
            target = torch.arange(batch_size, device=device)
            batch_loss = F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target) 

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()

        epoch_loss /= total_batches
        logger.info(f"Epoch: {epoch} | Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.4f}")
                break
            
    model_dir = os.path.join(save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(scin.state_dict(), os.path.join(model_dir, f"sCIN_rep{rep}.pth"))

    train_dict = {"model":scin}
    if is_pca:
        train_dict["pca_mod1"] = pca_mod1
        train_dict["pca_mod2"] = pca_mod2
    
    return train_dict


def get_embs_sCIN_ablated_unpaired(mod1_test: np.ndarray, 
                                   mod2_test: np.ndarray, 
                                   train_dict: Dict[str, Any], 
                                   save_dir: str, 
                                   is_pca: Optional[bool] = True,
                                   rep: Optional[Union[int, str]] = "NA") -> Tuple[np.ndarray, np.ndarray]:
    
    logger = setup_sCIN_logger()
    logger.info("Getting embeddings ...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_dict["model"]

    if is_pca:
        logger.info("PCA transformation of test data ...")
        pca_mod1 = train_dict["pca_mod1"]
        pca_mod2 = train_dict["pca_mod2"]
        mod1_test = pca_mod1.transform(mod1_test)
        mod2_test = pca_mod2.transform(mod2_test)
        logger.info("PCA finished.")
    
    # Arrays to tensors
    mod1_test_t = torch.from_numpy(mod1_test)
    mod1_test_t = mod1_test_t.to(torch.float32)
    mod1_test_t = mod1_test_t.to(device)
    mod2_test_t = torch.from_numpy(mod2_test).to(torch.float32)
    mod2_test_t = mod2_test_t.to(torch.float32)
    mod2_test_t = mod2_test_t.to(device)

    with torch.no_grad():
        logits_test, mod1_embs, mod2_embs = model(mod1_test_t, mod2_test_t)
    
    logger.info("Embeddings were generated.")

    mod1_embs_np = mod1_embs.cpu().numpy()
    mod2_embs_np = mod2_embs.cpu().numpy()
    logits_test = logits_test.cpu().numpy()

    embs_dir = os.path.join(save_dir, "embs")
    os.makedirs(embs_dir, exist_ok=True)

    logits_df = pd.DataFrame(logits_test)
    logits_df.to_csv(os.path.join(embs_dir, f"logits_test_{rep}.csv"), index=False)
    logger.info(f"Logits saved to {embs_dir}.")
    
    return mod1_embs_np, mod2_embs_np