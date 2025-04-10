import scglue
import networkx as nx 
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sCIN.utils import setup_logging
from sCIN.assess import assess
import argparse
import logging
import os


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_file", type=str)
    parser.add_argument("--atac_file", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--num_reps", type=int, choices=range(1, 11), help="Number of replications(1-10)")
    parser.add_argument("--is_inv_metrics", action="store_true", help="Whether to compute metrics from RNA to ATAC")
    parser.add_argument("--quick_test", action="store_true")
    args = parser.parse_args()

    log_save_dir = os.path.join(args.save_dir, "logs")
    setup_logging(level="debug",
                  log_dir=log_save_dir,
                  model_name="scGLUE")

    rna_ad = ad.read_h5ad(args.rna_file)
    rna_counts = rna_ad.layers["norm_raw_counts"]
    # hvgs = rna_ad.var.highly_variable
    # hvgs_ranks = rna_ad.var.highly_variable_rank
    atac_ad = ad.read_h5ad(args.atac_file)
    atac_counts = atac_ad.layers["norm_raw_counts"]
    labels = rna_ad.obs["cell_type_encoded"]
    logging.debug("Data loaded.")

    graph_dir = os.path.join(args.save_dir, "graph")
    os.makedirs(graph_dir, exist_ok=True)
    if not "guidance_graph.graphml.gz" in os.listdir(graph_dir):
        guidance_graph = scglue.genomics.rna_anchored_guidance_graph(rna_ad, atac_ad)
        scglue.graph.check_graph(guidance_graph, [rna_ad, atac_ad])
        nx.write_graphml(guidance_graph, "guidance_graph.graphml.gz")
    else:
        guidance_graph = nx.read_graphml("guidance_graph.graphml.gz")
    
    logging.debug("Graph settings completed.")
    
    # Select highly variable features from the graph
    # guidance_hvf = guidance_graph.subgraph(chain(rna_ad.var.query("highly_variable").index,atac_ad.var.query("highly_variable").index)).copy()

    
    seeds = [seed for seed in range(0, 100, 10)]
    if args.num_reps != 10:
        seeds = seeds[:args.num_reps]
    
    res = []
    for i, seed in enumerate(seed):
        rep = i + 1
        logging.info(f"Replication {rep}")

        # Split data
        rna_train, rna_test, atac_train, atac_test, labels_train, labels_test = train_test_split(rna_counts,
                                                                                                 atac_counts,
                                                                                                 labels,
                                                                                                 test_size=0.3,
                                                                                                 random_state=seed)
        rna_train_ad = ad.AnnData(X=rna_train)
        rna_train_ad.layers["counts"] = rna_train_ad.X.copy()
        # rna_train_ad.var["highly_variable"] = hvgs
        # rna_train_ad.var["highly_variable_rank"] = hvgs_ranks
        rna_test_ad = ad.AnnData(X=rna_test)
        rna_test_ad.layers["counts"] = rna_test_ad.X.copy() 
        # rna_test_ad.var["highly_variable"] = hvgs
        # rna_test_ad.var["highly_variable_rank"] = hvgs_ranks
        atac_train_ad = ad.AnnData(X=atac_train)
        atac_train_ad.layers["counts"] = atac_train_ad.X.copy() 
        atac_test_ad = ad.AnnData(X=atac_test)
        atac_test_ad.layers["counts"] = atac_test_ad.X.copy() 

        # Some preprocessing suggested by the original documentation
        # PCA transfomration of the RNA data
        rna_pca = PCA(n_components=100)  
        rna_pca.fit(rna_train)
        rna_train_pca = rna_pca.transform(rna_train)
        rna_test_pca = rna_pca.transform(rna_test)
        rna_train_ad.uns["X_pca"] = rna_train_pca
        rna_test_ad.uns["X_pca"] = rna_test_pca

        # LSI trnasformation of the ATAC data
        scglue.data.lsi(atac_train_ad, n_components=100, n_iter=15)
        scglue.data.lsi(atac_test_ad, n_components=100, n_iter=15)

        # Configure datasets
        scglue.models.configure_dataset(rna_train_ad, 
                                        "Normal", 
                                        use_highly_variable=False,
                                        use_layer="counts", 
                                        use_rep="X_pca")
        
        scglue.models.configure_dataset(rna_test_ad, 
                                        "Normal", 
                                        use_highly_variable=False,
                                        use_layer="counts", 
                                        use_rep="X_pca")

        scglue.models.configure_dataset(atac_train_ad, 
                                        "Normal", 
                                        use_highly_variable=False,
                                        use_rep="X_lsi")

        scglue.models.configure_dataset(atac_test_ad, 
                                        "Normal", 
                                        use_highly_variable=False,
                                        use_rep="X_lsi")
        
        logging.info("Data configuration completed.")   

        # Train
        if args.quick_test:
            max_epochs = 1
        else:
            max_epochs = 186  # Default

        model_save_dir = os.path.join(args.save_dir, "model")
        os.makedirs(model_save_dir, exist_ok=True)
        glue = scglue.models.fit_SCGLUE({"rna": rna_ad, "atac": atac_ad},
                                        guidance_graph,
                                        fit_kws={"directory":model_save_dir,
                                                 "max_epochs":max_epochs})
        glue.save("glue.dill")
        logging.debug("Training completed.")

        # Get embeddings
        embs_save_dir = os.path.join(args.save_dir, "embs")
        os.makedirs(embs_save_dir, exist_ok=True)
        rna_embs = glue.encode_data("rna", rna_test_ad)
        atac_embs = glue.encode_data("rna", atac_test_ad)
        rna_embs_df = pd.DataFrame(rna_embs)
        atac_embs_df = pd.DataFrame(atac_embs)
        rna_embs_df.to_csv(os.path.join(embs_save_dir, f"rna_embs_rep{rep}.csv"))
        atac_embs_df.to_csv(os.path.join(embs_save_dir, f"atac_embs_rep{rep}.csv"))
        logging.debug(f"Embeddings generated and saved to {embs_save_dir}.")

        # Compute metrics
        recall_at_k_a2r, num_pairs_a2r, cell_type_acc_a2r, asw, medr_a2r = assess(atac_embs,
                                                                                  rna_embs,
                                                                                  labels,
                                                                                  seed=seed)
        if args.is_inv_metrics:
            recall_at_k_r2a, num_pairs_r2a, cell_type_acc_r2a, _, medr_r2a = assess(rna_embs,
                                                                                    atac_embs,
                                                                                    labels,
                                                                                    seed=seed)
            
        for k, v_a2r in recall_at_k_a2r.items():
            v_r2a = recall_at_k_r2a.get(k, 0)
            res.append({
                "Models":"scButterfly",
                "Replicates":rep,
                "k":k,
                "Recall_at_k_a2r":v_a2r,
                "Recall_at_k_r2a":v_r2a if v_r2a is not None else 0.0,
                "num_pairs_a2r":num_pairs_a2r,
                "num_pairs_r2a":num_pairs_r2a if num_pairs_r2a is not None else 0.0,
                "cell_type_acc_a2r":cell_type_acc_a2r,
                "cell_type_acc_r2a":cell_type_acc_r2a if cell_type_acc_r2a is not None else 0.0,
                "cell_type_ASW":asw,
                "MedR_a2r":medr_a2r,
                "MedR_r2a":medr_r2a if medr_r2a is not None else 0.0
            })

    results = pd.DataFrame(res)
    res_save_dir = os.path.join(args.save_dir, "outs")
    os.makedirs(res_save_dir, exist_ok=True)
    results.to_csv(os.path.join(res_save_dir, f"metrics_scGLUE_{args.num_reps}reps.csv"), index=False)    
    logging.debug(f"Results were saved to {res_save_dir}.")


if __name__ == "__main__":
    main()