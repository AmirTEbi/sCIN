import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from typing import dict
import optuna


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
        
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        elif activation == "leaky_relu":
            layers.append(nn.LeakyReLU())
        
        if batch_norm:
            layers.append(nn.BatchNorm1d())

        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims[num_layers - 1], out_dim))

        if out_poisson:
            layers.append(nn.Softplus())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

def tune()
    

def train_gex_decoder(joint_embs: np.array, ground_truth: np.array, 
                      config: dict, quick_test=False) -> dict:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    decoder_cfg = config["decoder"] # decoder configurations
    train_cfg = config["training"]  # training configurations

    def objective(trial):

        ################### Hyperparameters ###################
        in_dim = decoder_cfg["in_dim"]
        out_dim = decoder_cfg["out_dim"]
        hidden_dims = trial.suggest_int("hidden_layer_sizes", 
                                        decoder_cfg["hidden_dims"][0], 
                                        decoder_cfg["hidden_dims"][1], step=64)
        num_layers = trial.suggest_int("num_layers", 
                                       decoder_cfg["num_layers"][0], 
                                       decoder_cfg["num_layers"][1])
        activation = trial.suggest_categorical("activation_fn", 
                                               config["activations"])
        dropout = trial.suggest_float("dropout_rate", 
                                      decoder_cfg["dropout"][0], 
                                      decoder_cfg["dropout"][1])
        
        out_poisson = trial.suggest_categorical("poisson_output",
                                                decoder_cfg["out_poisson"])
        lr = trial.suggest_float("learning_rate", 
                                 train_cfg["lr"][0], 
                                 train_cfg["lr"][1], log=True)
        
        batch_norm = trial.suggest_categorical("batch_norm_layer",
                                                decoder_cfg["batch_norm"])
        
        batch_size = trial.suggest_int("batch_size",
                                       train_cfg["batch_size"][0],
                                       train_cfg["batch_size"][1], step=64)
        
        ################### Data perparation ###################
        joint_embs_t = torch.from_numpy(joint_embs).to(torch.float32).to(device)
        ground_truth_t = torch.from_numpy(ground_truth).to(torch.float32).to(device)
        dataset = TensorDataset(joint_embs_t, ground_truth_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        ################### Model definition ###################
        model = GEXDecoder(
            in_dim=in_dim,
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            out_dim = out_dim,
            out_poisson=out_poisson,
            batch_norm=batch_norm,
            activation=activation,
            dropout=dropout
        )

        if out_poisson:
            criterion = nn.PoissonNLLLoss(log_input=True)
        else:
            criterion = nn.MSELoss()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        ################### Training ###################
        if quick_test:
            epochs = 1
        else:
            epochs = train_cfg["epochs"]

        for epoch in range(epochs):

            epoch_loss = 0.0
            for batch_in, batch_target in dataloader:
                optimizer.zero_grad()
                out = model(batch_in)
                loss = criterion(batch_in, batch_target)
                loss.backward()
                optimizer.step()
                epoch_loss += (loss.item() / len(batch_in))
                
            print(f"Trial: {trial.number} | Epoch: {epoch} | Loss: {epoch_loss:.4f}")

    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)








    


