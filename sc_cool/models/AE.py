import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from sklearn.decomposition import PCA


class RNAEncoderAE(nn.Module):
    """ An encoder with three layers"""
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(RNAEncoderAE, self).__init__()
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
    

class ATACEncoderAE(nn.Module):
    """ An encoder with three layers"""
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(ATACEncoderAE, self).__init__()
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

class RNADecoder(nn.Module):
    """ A decoder with three layers"""
    def __init__(self, latent_dim, hidden_dim, out_dim):
        super(RNADecoder, self).__init__()
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
        x_rna = self.linear2(h2)

        return x_rna
    

class ATACDecoder(nn.Module):
    """ """
    def __init__(self, latent_dim, hidden_dim, out_dim):
        super(ATACDecoder, self).__init__()
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
        x_atac = self.linear2(h2)

        return x_atac
    

class SimpleAutoEncoder(nn.Module):
    """ """
    def __init__(self, rna_encoder, atac_encoder,
                 rna_decoder, atac_decoder):  # in_dim == out_dim
        super(SimpleAutoEncoder, self).__init__()

        self.rna_encoder = rna_encoder
        self.atac_encoder = atac_encoder
        self.rna_decoder = rna_decoder
        self.atac_decoder = atac_decoder

    def forward(self, rna, atac):
        """

        Parameters
        ----------
        rna : torch.tensor
            
        atac : torch.tensor
            

        Returns
        -------

        """
        rna_emb = self.rna_encoder(rna)
        atac_emb = self.atac_encoder(atac)
        z_joint = torch.cat((rna_emb, atac_emb), 1)
        x_rna = self.rna_decoder(z_joint)
        x_atac = self.atac_decoder(z_joint)

        return x_rna, x_atac, rna_emb, atac_emb
    

def train_ae(rna_train, atac_train, labels_train=None, epochs=None, settings=None, 
             device=None):

    hidden_dim = settings["hidden_dim"]
    latent_dim = settings["latent_dim"]
    batch_size = settings["batch_size"]
    lr = settings["lr"]
    device = device

    rna_encoder = RNAEncoderAE(rna_train.shape[1], hidden_dim, latent_dim).to(device)
    rna_decoder = RNADecoder(latent_dim, hidden_dim, rna_train.shape[1]).to(device)
    atac_encoder = ATACEncoderAE(atac_train.shape[1], hidden_dim, latent_dim).to(device)
    atac_decoder = ATACDecoder(latent_dim, hidden_dim, atac_train.shape[1]).to(device)
    ae = SimpleAutoEncoder(rna_encoder, atac_encoder, rna_decoder, atac_decoder).to(device)

    rna_ds = TensorDataset(rna_train)
    atac_ds = TensorDataset(atac_train)
    rna_dl = DataLoader(rna_ds, batch_size, shuffle=False)
    atac_dl = DataLoader(atac_ds, batch_size, shuffle=False)

    optimizer = Adam(ae.parameters(), lr=lr)
    mse = nn.MSELoss()

    for epoch in range(epochs):

        ae.train()
        epoch_loss = 0.0
        total_samples = 0
        for rna_batch, atac_batch in zip(rna_dl, atac_dl):

            rna_batch[0] = rna_batch[0].to(device)
            atac_batch[0] = atac_batch[0].to(device)
            optimizer.zero_grad()
            x_rna, x_atac, _, _ = ae(rna_batch[0], atac_batch[0])
            loss = (mse(x_rna, rna_batch[0]) + mse(x_atac, atac_batch[0])) / 2

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(rna_batch[0])
            total_samples += len(rna_batch[0])

        epoch_loss /= total_samples
        print(f"Epoch: {epoch}, Loss: {epoch_loss:.4f}")

    return ae


def train_ae_unpaired(mod1_train, mod2_train, labels_train=None, epochs=None, settings=None, 
                      device=None):
    
    hidden_dim = settings["hidden_dim"]
    latent_dim = settings["latent_dim"]
    batch_size = settings["batch_size"]
    lr = settings["lr"]
    device = device

    # PCA transformations as the original paper stated
    print("PCA transformation ...")
    pca_mod1 = PCA(n_components=settings["PCs"])
    pca_mod2 =PCA(n_components=settings["PCs"])
    pca_mod1.fit(mod1_train)
    pca_mod2.fit(mod2_train)
    mod1_train = pca_mod1.transform(mod1_train)
    mod2_train = pca_mod2.transform(mod2_train)
    print("PCA finished.")

    rna_encoder = RNAEncoderAE(mod1_train.shape[1], hidden_dim, latent_dim).to(device)
    rna_decoder = RNADecoder(latent_dim, hidden_dim, mod1_train.shape[1]).to(device)
    atac_encoder = ATACEncoderAE(mod2_train.shape[1], hidden_dim, latent_dim).to(device)
    atac_decoder = ATACDecoder(latent_dim, hidden_dim, mod2_train.shape[1]).to(device)
    ae = SimpleAutoEncoder(rna_encoder, atac_encoder, rna_decoder, atac_decoder).to(device)

    rna_ds = TensorDataset(mod1_train)
    atac_ds = TensorDataset(mod2_train)
    rna_dl = DataLoader(rna_ds, batch_size, shuffle=False)
    atac_dl = DataLoader(atac_ds, batch_size, shuffle=False)

    optimizer = Adam(ae.parameters(), lr=lr)
    mse = nn.MSELoss()

    for epoch in range(epochs):

        ae.train()
        epoch_loss = 0.0
        total_samples = 0
        for rna_batch, atac_batch in zip(rna_dl, atac_dl):

            rna_batch[0] = rna_batch[0].to(device)
            atac_batch[0] = atac_batch[0].to(device)
            optimizer.zero_grad()
            x_rna, x_atac, _, _ = ae(rna_batch[0], atac_batch[0])
            loss = (mse(x_rna, rna_batch[0]) + mse(x_atac, atac_batch[0])) / 2

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(rna_batch[0])
            total_samples += len(rna_batch[0])

        epoch_loss /= total_samples
        print(f"Epoch: {epoch}, Loss: {epoch_loss:.4f}")

    return [ae, pca_mod1, pca_mod2]

    


def get_emb_ae(mod1_test, mod2_test, labels_test, obj_list, 
               save_dir=None, seed=None, device=None):
    
     model = obj_list[0]
     pca_mod1 = obj_list[1]
     pca_mod2 = obj_list[2]

     mod1_test = pca_mod1.transform(mod1_test)
     mod2_test = pca_mod2.transform(mod2_test)

     with torch.no_grad():
         _, _, rna_emb, atac_emb = model(mod1_test, mod2_test)
 
     rna_emb_np = rna_emb.cpu().numpy()
     atac_emb_np = atac_emb.cpu().numpy()

     return rna_emb_np, atac_emb_np