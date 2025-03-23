import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from sklearn.decomposition import PCA
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

class Mod1Decoder(nn.Module):
    """ A decoder with three layers"""
    def __init__(self, latent_dim, hidden_dim, out_dim):
        super(Mod1Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(2*latent_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

        self.relu = nn.LeakyReLU()

    def forward(self, z_joint):
        """

        Parameters
        ----------
        z_joint : torch.tensor
            Concatenated RNA and ATAC embeddings

        Returns
        -------

        """
        h1 = self.linear1(z_joint)
        h1 = self.bn(h1)
        h2 = self.relu(h1)
        x_mod1 = self.linear2(h2)

        return x_mod1
    

class Mod2Decoder(nn.Module):
    """ """
    def __init__(self, latent_dim, hidden_dim, out_dim):
        super(Mod2Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(2*latent_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

        self.relu = nn.LeakyReLU()

    def forward(self, z_joint):
        """

        Parameters
        ----------
        z_joint : torch.tensor
            Concatenated RNA and ATAC embeddings

        Returns
        -------

        """
        h1 = self.linear1(z_joint)
        h1 = self.bn(h1)
        h2 = self.relu(h1)
        x_mod2 = self.linear2(h2)

        return x_mod2
    

class SimpleAutoEncoder(nn.Module):
    """ """
    def __init__(self, mod1_encoder, mod2_encoder,
                 mod1_decoder, mod2_decoder):  # in_dim == out_dim
        super(SimpleAutoEncoder, self).__init__()

        self.mod1_encoder = mod1_encoder
        self.mod1_encoder = mod2_encoder
        self.mod1_decoder = mod1_decoder
        self.mod2_decoder = mod2_decoder

    def forward(self, mod1, mod2):
        """

        Parameters
        ----------
        rna : torch.tensor
            
        atac : torch.tensor
            

        Returns
        -------

        """
        mod1_emb = self.mod1_encoder(mod1)
        mod2_emb = self.mod1_encoder(mod2)
        z_joint = torch.cat((mod1_emb, mod2_emb), 1)
        mod1_recon = self.mod1_decoder(z_joint)
        mod2_recon = self.mod2_decoder(z_joint)

        return mod1_recon, mod2_recon, mod1_emb, mod2_emb
    

def train_AutoEncoder(mod1_train, mod2_train, settings=None, **kwargs):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = settings["num_epochs"]
    hidden_dim = settings["hidden_dim"]
    latent_dim = settings["latent_dim"]
    batch_size = settings["batch_size"]
    lr = settings["learning_rate"]
    save_dir = kwargs["save_dir"]
    rep = kwargs["rep"]
    is_pca = kwargs["is_pca"]

    if is_pca:
    
        # PCA transformations as the original paper stated
        print("PCA transformation ...")
        pca_mod1 = PCA(n_components=settings["PCs"])
        pca_mod2 =PCA(n_components=settings["PCs"])
        pca_mod1.fit(mod1_train)
        pca_mod2.fit(mod2_train)
        mod1_train = pca_mod1.transform(mod1_train)
        mod2_train = pca_mod2.transform(mod2_train)
        print("PCA finished.")

    mod1_encoder = Mod1Encoder(mod1_train.shape[1], hidden_dim, latent_dim).to(device)
    mod1_decoder = Mod1Decoder(latent_dim, hidden_dim, mod1_train.shape[1]).to(device)
    mod2_encoder = Mod2Encoder(mod2_train.shape[1], hidden_dim, latent_dim).to(device)
    mod2_decoder = Mod2Decoder(latent_dim, hidden_dim, mod2_train.shape[1]).to(device)
    ae = SimpleAutoEncoder(mod1_encoder, mod2_encoder, mod1_decoder, mod2_decoder).to(device)

    mod1_train_t = torch.from_numpy(mod1_train).to(torch.float32).to(device)
    mod2_train_t = torch.from_numpy(mod2_train).to(torch.float32).to(device)
    
    mod1_ds = TensorDataset(mod1_train_t)
    mod2_ds = TensorDataset(mod2_train_t)
    mod1_dl = DataLoader(mod1_ds, batch_size, shuffle=False)
    mod2_dl = DataLoader(mod2_ds, batch_size, shuffle=False)

    optimizer = Adam(ae.parameters(), lr=lr)
    mse = nn.MSELoss()

    for epoch in range(num_epochs):

        ae.train()
        epoch_loss = 0.0
        total_samples = 0
        for mod1_batch, mod2_batch in zip(mod1_dl, mod2_dl):

            mod1_batch[0] = mod1_batch[0].to(device)
            mod2_batch[0] = mod2_batch[0].to(device)
            optimizer.zero_grad()
            mod1_recon, mod2_recon, _, _ = ae(mod1_batch[0], mod2_batch[0])
            loss = (mse(mod1_recon, mod1_batch[0]) + mse(mod2_recon, mod2_batch[0])) / 2

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(mod1_batch[0])
            total_samples += len(mod1_batch[0])

        epoch_loss /= total_samples
        print(f"Epoch: {epoch}, Loss: {epoch_loss:.4f}")
    
    torch.save(ae.state_dict(), os.path.join(save_dir, "models", f"AE_{rep}"))
    train_dict = {"model":ae}
    if is_pca:
        train_dict["pca_mod1"] = pca_mod1
        train_dict["pca_mod2"] = pca_mod2

    return train_dict

    
def get_emb_AutoEncoder(mod1_test, mod2_test, train_dict, **kwargs):
     
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
     model = train_dict["model"]
     seed = kwargs["seed"]
     is_pca = kwargs["is_pca"]
     if is_pca:
         pca_mod1 = train_dict["pca_mod1"]
         pca_mod2 = train_dict["pca_mod2"]
         mod1_test = pca_mod1.transform(mod1_test)
         mod2_test = pca_mod2.transform(mod2_test)

     # Arrays to tensors
     mod1_test_t = torch.from_numpy(mod1_test)
     mod1_test_t = mod1_test_t.to(torch.float32)
     mod1_test_t = mod1_test_t.to(device)
     mod2_test_t = torch.from_numpy(mod2_test).to(torch.float32)
     mod2_test_t = mod2_test_t.to(torch.float32)
     mod2_test_t = mod2_test_t.to(device)

     with torch.no_grad():
         _, _, mod1_emb, mod2_emb = model(mod1_test_t, mod2_test_t)
 
     mod1_emb_np = mod1_emb.cpu().numpy()
     mod2_emb_np = mod2_emb.cpu().numpy()

     return mod1_emb_np, mod2_emb_np