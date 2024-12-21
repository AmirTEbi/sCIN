import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from sc_cool.models.ConAAE.model import FC_Autoencoder, FC_Classifier, FC_VAE, Simple_Classifier,TripletLoss
from sc_cool.models.ConAAE.conAAE import conAAE
from sklearn.decomposition import PCA
from sc_cool.utils.utils import shuffle_per_cell_type
import os
import argparse
import time


def setup_args(args=[]):
    """Source implementation of Wang et al.(2023)"""

    options = argparse.ArgumentParser()

    # save and directory options
    #options.add_argument('-sd', '--save-dir', action="store", dest="save_dir")
    #options.add_argument('-i', '--input-dir', action="store", dest="input_dir")
    #options.add_argument('--save-freq', action="store", dest="save_freq", default=10, type=int)
    #options.add_argument('--pretrained-file', action="store")

    # training parameters
    options.add_argument('-bs', '--batch-size', action="store", dest="batch_size", default=100, type=int)
    options.add_argument('-nz', '--latent-dimension', action="store", dest="nz", default=50, type=int)
    options.add_argument('-w', '--num-workers', action="store", dest="num_workers", default=10, type=int)
    
    options.add_argument('-lrD', '--learning-rate-D', action="store", dest="learning_rate_D", default=1e-4, type=float)
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=101, type=int)
    options.add_argument('-wd', '--weight-decay', action="store", dest="weight_decay", default=0, type=float)
    options.add_argument('--train-imagenet', action="store_true")
    options.add_argument('--conditional', action="store_true")
    options.add_argument('--conditional-adv', action="store_true")
    options.add_argument('--triplet-loss',action="store_true")
    options.add_argument('--contrastive-loss',action="store_true")
    options.add_argument('--consistency-loss',action="store_true")
    options.add_argument('--anchor-loss',action="store_true")
    options.add_argument('--MMD-loss',action="store_true")
    options.add_argument('--VAE',action="store_true")
    options.add_argument('--discriminator',action="store_true")
    options.add_argument('--augmentation',action="store_true")

    # hyperparameters
    options.add_argument('-lrAE', '--learning-rate-AE', action="store", dest="learning_rate_AE", default=1e-3, type=float)
    options.add_argument('--margin', action="store", default=0.3, type=float)
    options.add_argument('--alpha', action="store", default=10.0, type=float)
    options.add_argument('--beta', action="store", default=1., type=float)
    options.add_argument('--beta1', action="store", default=0.5, type=float)
    options.add_argument('--beta2', action="store", default=0.999, type=float)
    options.add_argument('--lamb', action="store", default=0.00000001, type=float)
    options.add_argument('--latent-dims', action="store", default=50, type=int)
    
    #options.add_argument('-rna', '--input-rna', action="store", dest="input_rna")
    #options.add_argument('-atac', '--input-atac', action="store", dest="input_atac")
    #options.add_argument('-label', '--input-label', action="store", dest="input_label")
    

    # gpu options
    options.add_argument('-gpu', '--use-gpu', action="store_false", dest="use_gpu")

    return options.parse_args(args)


def train_con(mod1_train, mod2_train, labels_train, epochs,
              settings=None, **kwargs):
    
    
    save_dir = kwargs["save_dir"]
    seed = kwargs["seed"]
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

    # Model parameters
    args = setup_args(args=['--consistency-loss', '--contrastive-loss', '--triplet-loss', '--VAE', 
                        '--conditional-adv', '--conditional', '--discriminator', '--MMD-loss', '--anchor-loss', 
                        '-gpu'])
    
    args.max_epochs = epochs

    # Check device
    if not torch.cuda.is_available():
            args.use_gpu = False


    # Arrays to tensors
    mod1_train_t = torch.from_numpy(mod1_train).to(torch.float32)
    mod2_train_t = torch.from_numpy(mod2_train).to(torch.float32)
    labels_train_t = torch.from_numpy(labels_train).long()

    # Create datasets
    RNA_train_dataset = torch.utils.data.TensorDataset(mod1_train_t, labels_train_t)
    ATAC_train_dataset = torch.utils.data.TensorDataset(mod2_train_t, labels_train_t)

    # Create model object
    con=conAAE(RNA_train_dataset,ATAC_train_dataset,args)

    # Train the model
    con.train()
    torch.save(con.netRNA.state_dict(), os.path.join(save_dir, "models", f"netRNA_{seed}.pt"))
    torch.save(con.netATAC.state_dict(), os.path.join(save_dir, "models", f"netATAC_{seed}.pt"))
    train_dict = {"model":con}
    if is_pca:
         train_dict["pca_mod1"] = pca_mod1
         train_dict["pca_mod2"] = pca_mod2

    return train_dict


def get_emb_con(mod1_test, mod2_test, labels_test, train_dict, save_dir=None,
                **kwargs):
     
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
     model = train_dict["model"]
     seed = kwargs["seed"]
     is_pca = kwargs["is_pca"]
     if is_pca:
          pca_mod1 = train_dict["pca_mod1"]
          pca_mod2 = train_dict["pca_mod2"]
          mod1_test = pca_mod1.transform(mod1_test)
          mod2_test = pca_mod2.transform(mod2_test)
     
     # Check if the model is trained
     print(f"Is the model trained: {model.is_trained}")

     # Arrays to tensors
     mod1_test_t = torch.from_numpy(mod1_test).to(torch.float32)
     mod2_test_t = torch.from_numpy(mod2_test).to(torch.float32)
     labels_test_t = torch.from_numpy(labels_test).long()

     # Create datasets
     Mod1_test_dataset = torch.utils.data.TensorDataset(mod1_test_t, labels_test_t)
     Mod2_test_dataset = torch.utils.data.TensorDataset(mod2_test_t, labels_test_t)

     # Get embeddings
     mod1_embs, mod2_embs = model.test(Mod1_test_dataset, Mod2_test_dataset, seed=seed, save_dir=save_dir)

     return mod1_embs, mod2_embs