import os
import anndata as ad
import torch
import torch.nn as nn
from torch.optim import Adam
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple


class GEXDecoder(nn.Module):
    """
    A decoder to map embeddings to gene expression space.
    """

    def __init__(self, in_dim, num_layers, hidden_dims, out_dim,
                 out_poisson, batch_norm, activation="relu", dropout=0.0):
        super(GEXDecoder, self).__init__()
        layers = []

        for i in range(num_layers):
            in_features = in_dim if i == 0 else hidden_dims[i - 1]
            out_features = hidden_dims[i]
            layers.append(nn.Linear(in_features, out_features))

            if batch_norm:
                layers.append(nn.BatchNorm1d())
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims[num_layers - 1], out_dim))

        if out_poisson:
            layers.append(nn.Softplus())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

def list_embs(dir: str) -> Tuple:

    return ([os.path.joint(dir, f"rna_emb{i}.npy") \
            for i in range(0, 100, 10)], 
            [os.path.joint(dir, f"atac_emb{i}.npy") \
             for i in range(0, 100, 10)])


def get_gt(path: str, seed: int) -> np.array:

    adata = ad.read_h5ad(path)
    
    counts = adata.layers["norm_layer_counts"]
    _, gt = train_test_split(counts, test_size=0.3, random_state=seed)

    return gt


def train_val_gex_decoder(config: Dict, data: str, model: str,
                      save_dir) -> None:
    
    num_layers = config["num_layers"]
    hidden_dims = config["hidden_dims"]
    out_poisson = config["out_poisson"]
    batch_norm = config["batch_norm"]
    activation = config["activation"]
    dropout = config["dropout"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    is_checkpoints = config["is_checkpoints"]

    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    if data == "SHARE":
        mod1_embs, mod2_embs = list_embs(os.path.join("results", 
                                                      data, model, "embs"))
        gt = get_gt("data/share/Ma-2020-RNA.h5ad")
   
    elif data == "PBMC":
        mod1_embs, mod2_embs = list_embs(os.path.join("results", 
                                                      data, model, "embs"))
        gt = get_gt("data/10x/10x-Multiome-Pbmc10k-RNA.h5ad")

    elif data == "CITE":
        mod1_embs, mod2_embs = list_embs(os.path.join("results", 
                                                      data, model, "embs"))
        gt = get_gt("data/cite/rna.h5ad")

    for path1, path2 in zip(mod1_embs, mod2_embs):
        emb1 = np.load(path1)
        emb2 = np.load(path2)
        joint_emb = np.concatenate([emb1, emb2], axis=1)
        in_dim = joint_emb.shape[1]
        out_dim = gt.shape[1]
        print(f"Shape of the joint embeddings: {joint_emb.shape}")
        print(f"Shape of the ground truth: {gt.shape}")

        try:
            joint_emb_train, joint_emb_n_train,\
            gt_train, gt_n_train = train_test_split(emb1,
                                                    emb2,
                                                    gt,
                                                    test_size=0.3,
                                                    random_state=42)
        except ValueError as e:
            print("Dimension mismatch error between embeddings and\
                  the ground truth: {e}")
        
        joint_emb_val, joint_emb_test,\
        gt_val, gt_test = train_test_split(joint_emb_n_train,
                                           gt_n_train,
                                           test_size=0.3,
                                           random_state=42)
 
        train_dataset = TensorDataset(torch.tensor(joint_emb_train, dtype=torch.float32),
                                torch.tensor(gt_train, dtype=torch.float32))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        val_dataset = TensorDataset(torch.tensor(joint_emb_val, dtype=torch.float32),
                                torch.tensor(gt_val, dtype=torch.float32))
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        rep = mod1_embs.index(emb1)
        print(f"Seed is {rep}")
        np.save(os.path.join(save_dir, "test_data", f"joint_emb_test_rep{rep}.npy", joint_emb_test))
        np.save(os.path.join(save_dir, "test_data", f"ground_truth_rep{rep}.npy", gt_test))
        

        model = GEXDecoder(in_dim=in_dim, num_layers=num_layers, hidden_dims=hidden_dims,
                           out_dim=out_dim, out_poisson=out_poisson, batch_norm=batch_norm,
                           activation=activation, dropout=dropout)
        
        optimizer = Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        if is_checkpoints:
            
            checkpoint_dir = os.path.join(save_dir, "checkpoints")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_rep{rep}.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        
        best_val_loss = float("inf")      
        for epoch in range(100):
            model.train()
            running_train_loss = 0.0
            for emb, gt_ in train_dataloader:
                optimizer.zero_grad()
                out = model(emb)
                batch_loss = criterion(out, gt_)
                batch_loss.backward()
                optimizer.step()
                running_train_loss += batch_loss.item()
            avg_train_loss = running_train_loss / len(train_dataloader)

            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for val_emb, val_gt_ in val_dataloader:
                    val_out = model(val_emb)
                    val_batch_loss = criterion(val_out, val_gt_)
                    running_val_loss += val_batch_loss
                avg_val_loss = running_val_loss / len(val_dataloader)
            
            current_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                current_checkpoint_path,
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": avg_val_loss,
                    },
                    best_model_path,
                )

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}"
            )
            tune.report(val_loss=avg_val_loss)