import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from model import FC_Autoencoder, FC_Classifier, FC_VAE, Simple_Classifier,TripletLoss
from conAAE import conAAE
from sklearn.decomposition import PCA
import os
import argparse
import time


def setup_args(args=[]):

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


def train_con(rna_train, atac_train, labels_train, epochs,
              settings=None, device=None):
    
    # PCA transformations as the original paper stated
    print("PCA transformation ...")
    pca_rna = PCA(n_components=settings["PCs"])
    pca_atac =PCA(n_components=settings["PCs"])
    pca_rna.fit(rna_train)
    pca_atac.fit(atac_train)
    rna_train = pca_rna.transform(rna_train)
    atac_train = pca_atac.transform(atac_train)
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
    rna_train_t = torch.from_numpy(rna_train).to(torch.float32)
    atac_train_t = torch.from_numpy(atac_train).to(torch.float32)
    labels_train_t = torch.from_numpy(labels_train).long()

    # Create datasets
    RNA_train_dataset = torch.utils.data.TensorDataset(rna_train_t, labels_train_t)
    ATAC_train_dataset = torch.utils.data.TensorDataset(atac_train_t, labels_train_t)

    # Create model object
    con=conAAE(RNA_train_dataset,ATAC_train_dataset,args)

    # Train the model
    con.train()         

    return [con, pca_rna, pca_atac]


def get_emb_con(rna_test, atac_test, labels_test, obj_list, save_dir=None, seed=None, device=None):
     
     model = obj_list[0]
     pca_rna = obj_list[1]
     pca_atac = obj_list[2]

     rna_test = pca_rna.transform(rna_test)
     atac_test = pca_atac.transform(atac_test)
     
     # Check if the model is trained
     print(f"Is the model trained: {model.is_trained}")

     # Arrays to tensors
     rna_test_t = torch.from_numpy(rna_test).to(torch.float32)
     atac_test_t = torch.from_numpy(atac_test).to(torch.float32)
     labels_test_t = torch.from_numpy(labels_test).long()

     # Create datasets
     RNA_test_dataset = torch.utils.data.TensorDataset(rna_test_t, labels_test_t)
     ATAC_test_dataset = torch.utils.data.TensorDataset(atac_test_t, labels_test_t)

     # Get embeddings
     rna_emb, atac_emb = model.test(RNA_test_dataset,ATAC_test_dataset, seed=seed, save_dir=save_dir)

     return rna_emb, atac_emb