import harmonypy as hm
import numpy as np
import pandas as pd
from numpy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA


def train_hm(rna_train, atac_train, labels_train=None, epochs=None,
                settings=None, device=None):
    """

    Parameters
    ----------

    """

    # PCA transformations
    print("PCA starts ...")
    pca_rna = PCA(n_components=settings["PCs"])
    pca_atac = PCA(n_components=settings["PCs"])
    pca_rna.fit(rna_train)
    rna_train = pca_rna.transform(rna_train)
    pca_atac.fit(atac_train)
    atac_train = pca_atac.transform(atac_train)
    print("PCA ended.")
    
    # Check for zeros
    num_zeros_rna = np.count_nonzero(rna_train == 0)
    num_zeros_atac = np.count_nonzero(atac_train == 0)
    print(f"Number of zero elemnets for RNA: {num_zeros_rna}")
    print(f"Number of zero elements for ATAC: {num_zeros_atac}")

	# CCA
    print("CCA starts ...")

    cca = CCA(n_components=50)
    cca.fit(rna_train, atac_train)
    rna_train_C, atac_train_C = cca.transform(rna_train, atac_train)

    print("CCA ended.")
    
    data = np.concatenate((rna_train_C, atac_train_C))
    rna_mode = np.array(["rna"] * rna_train_C.shape[0])
    atac_mode = np.array(["atac"] * atac_train_C.shape[0])
    mode = np.concatenate((rna_mode, atac_mode))
    meta_data = pd.DataFrame({"mode":mode})


    vars_use = ["mode"]

    hm_train = hm.run_harmony(data, meta_data, vars_use=vars_use, max_iter_harmony=epochs)
    
    return [hm_train, pca_rna, pca_atac, cca]


def get_emb_hm(rna_test, atac_test, labels_test=None, obj_list=None, save_dir=None, 
                                             seed=None, device=None):
    """

    Parameters
    ----------
    
    """
    model = obj_list[0]
    pca_rna = obj_list[1]
    pca_atac = obj_list[2]
    cca = obj_list[3]
    
    rna_test = pca_rna.transform(rna_test)
    atac_test = pca_atac.transform(atac_test)
    
    rna_test_C = cca.transform(rna_test)
    atac_test_C = cca.transform(atac_test)

    #rna_test_C = np.dot(rna_test, W_rna)
    #atac_test_C = np.dot(atac_test, W_atac)

    test_data = np.concatenate((rna_test_C, atac_test_C))
    rna_mode = np.array(["rna"] * rna_test_C.shape[0])
    atac_mode = np.array(["atac"] * atac_test_C.shape[0])
    mode = np.concatenate((rna_mode, atac_mode))
    meta_data = pd.DataFrame({"mode":mode})

    vars_use = ["mode"]
    
    hm_test = hm.run_harmony(test_data, meta_data,
                             vars_use = vars_use,
                             theta=model.theta,
                             sigma = model.sigma,
                             nclust = model.K,
                             block_size = model.block_size,
                             max_iter_harmony=1)

    print(f"Shape of the total embeddings: {hm_test.Z_corr.shape}")

    total_emb = hm_test.Z_corr.transpose()

    rna_emb = total_emb[:rna_test_C.shape[0]]
    atac_emb = total_emb[rna_test_C.shape[0]:]

    print(f"RNA emb shape: {rna_emb.shape}")
    print(f"ATAC emb shape: {atac_emb.shape}")

    return rna_emb, atac_emb