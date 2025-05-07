from sCIN.assess import compute_graph_connectivity, cell_type_at_k_unpaired
import numpy as np 
import pandas as pd 
import argparse
import re
import os 

DIRS = {
    "SHARE":{
        "sCIN": "sCIN/results/SHARE/V1-embs",
        "Con-AAE": "sCIN/results/SHARE/Con-AAE/embs",
        "MOFA+": "sCIN/results/SHARE/MOFA/embs",
        "Harmony": "sCIN/results/SHARE/Harmony/embs",
        "AE": "sCIN/results/SHARE/AE/embs", 
        "scGLUE": "sCIN/results/SHARE/scGLUE/embs",
        "scBridge": "sCIN/results/SHARE/scBridge",
        "sciCAN": "sciCAN/results/SHARE/embs"
    },
    "PBMC":{
        "sCIN": "sCIN/results/PBMC/sCIN/V1-embs",
        "Con-AAE": "sCIN/results/PBMC/Con-AAE/embs",
        "MOFA+": "sCIN/results/PBMC/MOFA/embs",
        "Harmony": "sCIN/results/PBMC/Harmony/embs",
        "AE": "sCIN/results/PBMC/AE/embs", 
        "scGLUE": "sCIN/results/PBMC/scGLUE/embs",
        "scBridge": "sCIN/results/PBMC/scBridge/embs",
        "sciCAN": "sciCAN/results/10xPBMC/embs"
    },
    "CITE":{
        "sCIN": "sCIN/results/CITE/sCIN/V1-embs",
        "Con-AAE": "sCIN/results/CITE/Con-AAE/embs",
        "MOFA+": "sCIN/results/CITE/MOFA/embs",
        "Harmony": "sCIN/results/CITE/Harmony/embs",
        "AE": "sCIN/results/CITE/AE/embs", 
        "scGLUE": "sCIN/results/CITE/scGLUE/embs",
        "sciCAN": "sciCAN/results/CITE/embs"
    }
}


def setup_args():

    parser = argparse.ArgumentParser()

    return parser


def main():

    parser = setup_args()
    args = parser.parse_args()

    for data, dirs in DIRS:
        for model, embs_dir in dirs:
            print(f"Model: {model} ...")
            files = os.listdir(embs_dir)
            rna_files = [f for f in files if "rna" in f]
            rna_files = sorted(rna_files)
            atac_files = [f for f in files if "atac" in f]
            atac_files = sorted(atac_files)
            labels = [f for f in files if "label" in f]
            labels = sorted(labels)

            for rna_file in rna_files:
                found = re.search(r"\d+", rna_file)
                match = int(found.group())
                
    





if __name__ == "__main__":
    main()