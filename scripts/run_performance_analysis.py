from sCIN.sCIN import train_sCIN, get_emb_sCIN
from sCIN.sCIN_unpaired import train_sCIN_unpaired, get_embs_sCIN_unpaired
from configs import sCIN
from sCIN.utils import extract_counts, setup_logging
from sklearn.model_selection import train_test_split
import anndata as ad
import pandas as pd
import numpy as np
import os
import logging
import argparse
import time


DATA_PATHS = {
    "SHARE":{
        "Mod1":"data/share/Ma-2020-RNA.h5ad",
        "Mod2":"data/share/Ma-2020-ATAC.h5ad"
    },
    "PBMC":{
        "Mod1":"data/10xPBMC/10x-Multiome-Pbmc10k-RNA.h5ad",
        "Mod2":"data/10xPBMC/10x-Multiome-Pbmc10k-ATAC.h5ad"
    },
    "CITE":{
        "Mod1":"data/cite/rna.h5ad",
        "Mod2":"data/cite/adt.h5ad"
    },
    "Muto":{
        "Mod1":"data/Muto-2021/Muto-2021-RNA-pp.h5ad",
        "Mod2":"data/Muto-2021/Muto-2021-ATAC-pp.h5ad"
    }
}


def setup_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--quick_test", action="store_true")

    return parser


def run_paired(
    data,
    mod1_adata, 
    mod2_adata,
    save_dir,
    quick_test = False
):
    rna_counts, atac_counts = extract_counts(mod1_adata, mod2_adata)
    labels = mod1_adata.obs["cell_type_encoded"].values
    rna_train, _, atac_train, _, \
            labels_train, _ = train_test_split(rna_counts,
                                               atac_counts,
                                               labels,
                                               test_size=0.3,
                                               random_state=0)
    
    if quick_test :
        num_epochs = 1

    else:
        num_epochs = None

    start_time = time.time()
    train_dict = train_sCIN(rna_train, 
                            atac_train,
                            labels_train, 
                            settings=sCIN,
                            save_dir=save_dir,
                            is_pca=True)
    end_time = time.time()
    train_time = end_time - start_time
    train_time_min = train_time / 60

    res_dict = {
        "data": data,
        "Mod 1 Number of cells": rna_train.shape[0],
        "Mod 2 Number of cells": atac_train.shape[0],
        "Mod1 dim": rna_train.shape[1],
        "Mod2 dim": atac_train.shape[1],
        "Train time(sec)": train_time,
        "Train time(min)": train_time_min,
    }

    return res_dict

    
def run_unpaired(
    data,
    mod1_adata, 
    mod2_adata,
    save_dir,
    quick_test = False
):
    rna_counts, atac_counts = extract_counts(mod1_adata, mod2_adata)
    rna_labels = mod1_adata.obs["cell_type"].values
    atac_labels = mod2_adata.obs["cell_type"].values

    rna_train, _, rna_lbls_train, _ = train_test_split(rna_counts,
                                                       rna_labels,
                                                       test_size=0.3,
                                                       random_state=0)
        
    atac_train, _, atac_lbls_train, _ = train_test_split(atac_counts,
                                                         atac_labels,
                                                         test_size=0.3,
                                                         random_state=0)  
    if quick_test :
        num_epochs = 1
        
    else:
        num_epochs = None

    start_time = time.time()
    train_dict = train_sCIN_unpaired(mod1_train=rna_train,
                                     mod2_train=atac_train,
                                     mod1_labels_train=rna_lbls_train,
                                     mod2_labels_train=atac_lbls_train,
                                     settings=sCIN,
                                     save_dir=save_dir,
                                     is_pca=True)
    end_time = time.time()
    train_time = end_time - start_time
    train_time_min = train_time / 60

    res_dict = {
        "data": data,
        "Mod 1 Number of cells": rna_train.shape[0],
        "Mod 2 Number of cells": atac_train.shape[0],
        "Mod1 dim": rna_train.shape[1],
        "Mod2 dim": atac_train.shape[1],
        "Train time(sec)": train_time,
        "Train time(min)": train_time_min,
    }

    return res_dict


def main() -> None:

    parser = setup_args()
    args = parser.parse_args()

    quick_test = args.quick_test

    res = []
    for data, paths in DATA_PATHS.items():
        print(data)
        mod1_data = ad.read_h5ad(paths["Mod1"])
        mod2_data = ad.read_h5ad(paths["Mod2"])

        if data != "Muto":
            res_dict = run_paired(
                data=data, 
                mod1_adata=mod1_data, 
                mod2_adata=mod2_data, 
                save_dir=args.save_dir, 
                quick_test=quick_test
            )
        
            res.append(res_dict)
        
        else:
            res_dict = run_unpaired(
                data=data,
                mod1_adata=mod1_data,
                mod2_adata=mod2_data,
                save_dir=args.save_dir,
                quick_test=quick_test
            )

            res.append(res_dict)
        
    res_df = pd.DataFrame(res)
    res_save_dir = os.path.join(args.save_dir, "outs")
    os.makedirs(res_save_dir, exist_ok=True)
    res_df.to_csv(os.path.join(res_save_dir, "Performance-Analysis.csv"), index=False)

    logging.info(f"Results saved to {res_save_dir}")


if __name__ == "__main__":
    main()