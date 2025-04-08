import numpy as np
from typing import Union
from mofapy2.run.entry_point import entry_point
from sklearn.decomposition import PCA


def prepare_data_mofa(rna_train, atac_train, rna_test, atac_test):

    atac_pca = PCA(n_components=100)
    atac_pca.fit(atac_train)
    atac_train = atac_pca.transform(atac_train)
    atac_test = atac_pca.transform(atac_test)

    rna_to_impute = np.zeros_like(rna_test)
    print(f"atac test shape {atac_test.shape}")
    print(type(rna_test))
    print(type(atac_test))
    atac_to_impute = np.zeros_like(atac_test)
    print(f"shape of the rna impute {rna_to_impute.shape}")
    print(f"shape of the atac impute {atac_to_impute.shape}")

    rna_data = np.vstack((rna_train, rna_test, rna_to_impute))
    atac_data = np.vstack((atac_train, atac_to_impute, atac_test))

    print(rna_data.shape)
    print(atac_data.shape)

    return rna_data, atac_data


def extract_embs(z: np.array, train_cells: int, test_cells: int) -> Union[np.array, np.array]:

    rna_emb = z[-test_cells:]
    atac_emb = z[train_cells:train_cells + test_cells]

    return rna_emb, atac_emb
    

def get_mofa_emb(rna_test, rna_train, atac_test, atac_train, settings, seed=None):
    
    rna_data, atac_data = prepare_data_mofa(rna_train, atac_train, rna_test, atac_test)
    print(f"shape of the rna data {rna_data.shape}")
    print(f"shape of the atac data {atac_data.shape}")

    data = [[rna_data], 
            [atac_data]]
    views = ["rna", "atac"]
    groups = ["group_0"]

    mofa_model = entry_point()
    mofa_model.set_data_options(scale_groups=False, scale_views=False)
    mofa_model.set_data_matrix(data, views_names=views, groups_names=groups)
    mofa_model.set_model_options(factors=settings["num_factors"])
    mofa_model.set_train_options(iter=settings["num_iterations"], 
                                 convergence_mode=settings["convergence_mode"],
                                 seed=seed)

    mofa_model.build()
    mofa_model.run()

    Z = mofa_model.model.nodes["Z"].getExpectation()
    print(f"The Z shape is {Z.shape}")

    rna_embs, atac_embs = extract_embs(z=Z, train_cells=rna_train.shape[0], 
                                       test_cells=rna_test.shape[0])


    return rna_embs, atac_embs