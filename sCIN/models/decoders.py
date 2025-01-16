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
from typing import Dict, Tuple, List
import traceback
import re


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
                layers.append(nn.BatchNorm1d(out_features))
            
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

    return [os.path.join(dir, f"atac_emb{i}.npy") \
            for i in range(0, 100, 10)]


def train_val_gex_decoder(config: Dict, data: str, model: str,
                      save_dir) -> None:
    
    
    print(f"Data: {data}")
    print("Current working directory:", os.getcwd())
    epochs = 150
    num_layers = config["num_layers"]
    hidden_dims = config["hidden_dims"]
    out_poisson = config["out_poisson"]
    batch_norm = config["batch_norm"]
    activation = config["activation"]
    dropout = config["dropout"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    is_checkpoints = config["is_checkpoints"]

    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    os.makedirs(os.path.join(save_dir, "test_data"), exist_ok=True)


    if data == "SHARE":
        mod2_embs = list_embs(os.path.join("results", data.lower(), model, "embs"))
        print(os.path.exists("./data/share/Ma-2020-RNA.h5ad"))
        adata = ad.read_h5ad("./data/share/Ma-2020-RNA.h5ad")
        counts = adata.layers["norm_layer_counts"]
    elif data == "PBMC":
        mod2_embs = list_embs(os.path.join("results", "10x", model, "embs"))
        adata = ad.read_h5ad("./data/10x/10x-Multiome-Pbmc10k-RNA.h5ad")
        counts = adata.layers["norm_layer_counts"]
    elif data == "CITE":
        mod2_embs = list_embs(os.path.join("results", data.lower(), model, "embs"))
        adata = ad.read_h5ad("./data/cite/rna.h5ad")
        counts = adata.layers["norm_layer_counts"]

    for path in enumerate(mod2_embs):
        emb2 = np.load(path)
        in_dim = emb2.shape[1]
        f = os.path.basename(path)
        match = re.search(r'atac_emb(\+d)', f)
        if match:
            seed = int(match.group(1))
            print(f"Seed: {seed}")
        
        _, gt = train_test_split(counts, test_size=0.3, random_state=seed)  # gt := ground truth

        out_dim = gt.shape[1]
        print(f"Shape of the Second moality embeddings: {emb2.shape}")
        print(f"Shape of the GEX ground truth: {gt.shape}")

        try:
            emb2_train, emb2_n_train, \
                gt_train, gt_n_train = train_test_split(
                                                    emb2,
                                                    gt,
                                                    test_size=0.3,
                                                    random_state=42)
        except ValueError as e:
            print(f"Dimension mismatch error between embeddings and\
                  the ground truth: {e}")
        
        emb2_val, emb2_test,\
            gt_val, gt_test = train_test_split(emb2_n_train,
                                           gt_n_train,
                                           test_size=0.3,
                                           random_state=42)
 
        train_dataset = TensorDataset(torch.tensor(emb2_train, dtype=torch.float32),
                                torch.tensor(gt_train, dtype=torch.float32))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(torch.tensor(emb2_val, dtype=torch.float32),
                                torch.tensor(gt_val, dtype=torch.float32))
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        rep = mod2_embs.index(emb2)
        print(f"Seed is {rep}")
        np.save(os.path.join(save_dir, "test_data", f"emb2_test_rep{rep}.npy", emb2_test))
        np.save(os.path.join(save_dir, "test_data", f"ground_truth_rep{rep}.npy", gt_test))
        

        model = GEXDecoder(in_dim=in_dim, num_layers=num_layers, hidden_dims=hidden_dims,
                           out_dim=out_dim, out_poisson=out_poisson, batch_norm=batch_norm,
                           activation=activation, dropout=dropout)
        
        optimizer = Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        if is_checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_rep{rep}.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        
        best_val_loss = float("inf")
        for epoch in range(epochs):
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
                    running_val_loss += val_batch_loss.item()
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


def get_search_space():
    """Defines the hyperparameter search space for tuning."""
    return {
        "num_layers": tune.choice([2, 4, 6]),
        "hidden_dims": tune.grid_search([[32], [64], [128], [256]]),
        "out_poisson": tune.choice([True, False]),
        "batch_norm": tune.choice([True, False]),
        "activation": tune.choice(["relu", "tanh", "leaky_relu"]),
        "dropout": tune.uniform(0.0, 0.5),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        "is_checkpoints": tune.choice([True, False]),
    }


def hyperparam_opt(model:str, save_dir:str, datasets:List) -> None:

    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=50,
        grace_period=5,
        reduction_factor=2,
    )

    search_space = get_search_space()

    ray.init(ignore_reinit_error=True)
    try:
        for data in datasets:
            tune.run(
                tune.with_parameters(
                    train_val_gex_decoder,
                    data=data,
                    model=model,
                    save_dir=save_dir
                ),
                name=f"gex_optimizer_{data.lower()}",
                config=search_space,
                scheduler=scheduler,
                num_samples=20,
                resources_per_trial={"cpu": 8, "gpu": 0.5},
                fail_fast=True,
            )
    except Exception as e:
        print(f"Error during hyperparameter optimization: {e}")
        traceback.print_exc()
    finally:
        ray.shutdown()