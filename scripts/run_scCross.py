import anndata as ad
import sccross
import pandas as pd
from sCIN.assess import assess
from sCIN.utils import setup_logging
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
import argparse
import logging
import os

    
def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_file")
    parser.add_argument("--atac_file")
    parser.add_argument("--gtf_file")
    parser.add_argument("--save_dir")
    parser.add_argument("--num_reps", type=int, choices=range(1, 11), help="Number of replications(1-10)")
    parser.add_argument("--is_inv_metrics", action="store_true")
    parser.add_argument("--quick_test", action="store_true")
    args = parser.parse_args()

    setup_logging(level="debug",
                  log_dir=args.save_dir,
                  model_name="scCross")

    rna = ad.read_h5ad(args.rna_file)
    rna_counts = rna.layers["norm_raw_counts"]
    atac = ad.read_h5ad(args.atac_file)
    atac_counts = atac.layers["norm_raw_counts"]
    labels = rna.obs["cell_type_encoded"].values

    atac2rna = sccross.data.geneActivity(atac, gtf_file=args.gtf_file)
    atac.uns["gene"] = atac2rna

    seeds = [seed for seed in range(0, 100, 10)]
    if args.num_reps != 10:
        seeds = seeds[:args.num_reps]

    res = []
    for i, seed in enumerate(seeds):
        rep = i + 1
        rna_train, rna_test, atac_train, atac_test, \
            labels_train, labels_test = train_test_split(rna_counts,
                                                         atac_counts,
                                                         labels,
                                                         test_size=0.3,
                                                         random_state=seed)
        rna_train_ad = ad.AnnData(X=rna_train)
        atac_train_ad = ad.AnnData(X=atac_train)
        rna_pca = PCA(n_components=2000)
        rna_pca.fit(rna_train)
        atac_lsi = TruncatedSVD(n_components=2000)  # Also knowns as Latent Semantic Indexing (LSI)
        atac_lsi.fit(atac_train)

        rna_train_PCs = rna_pca.transform(rna_train)
        atac_train_SVs = atac_lsi.transform(atac_train)
        rna_train_ad.uns["X_pca"] = rna_train_PCs
        atac_train_ad.uns["X_lsi"] = atac_train_SVs
        rna_train_ad.obs["cell_type"] = labels_train
        atac_train_ad.obs["cell_type"] = labels_train

        rna_test_ad = ad.AnnData(X=rna_test)
        atac_test_ad = ad.AnnData(X=atac_test)
        rna_test_PCs = rna_pca.transform(rna_test)
        atac_test_SVs = atac_lsi.transform(atac_test)
        rna_test_ad.uns["X_pca"] = rna_test_PCs
        atac_test_ad.uns["X_lsi"] = atac_test_SVs
        rna_test_ad.obs["cell_type"] = labels_test
        atac_test_ad.obs["cell_type"] = labels_test

        sccross.models.configure_dataset(rna_train_ad, 
                                         "Normal",  # Since the data has been preprocessed and normalized. 
                                         use_highly_variable=True, 
                                         use_layer = "norm_raw_counts", 
                                         use_rep="X_pca")
        
        sccross.models.configure_dataset(atac_train_ad, 
                                         "Normal", 
                                         use_highly_variable=False, 
                                         use_rep="X_lsi")
        
        sccross.models.configure_dataset(rna_test_ad, 
                                         "Normal", 
                                         use_highly_variable=True, 
                                         use_layer = "norm_raw_counts", 
                                         use_rep="X_pca")
        
        sccross.models.configure_dataset(atac_test_ad, 
                                         "Normal", 
                                         use_highly_variable=False, 
                                         use_rep="X_lsi")
        
        
        sccross.data.mnn_prior([rna_train_ad, atac_train_ad])

        # Train
        cross = sccross.models.fit_SCCROSS({"rna": rna_train_ad, 
                                            "atac": atac_train_ad}, 
                                           fit_kws={"directory": "sccross"})
        
        model_save_dir = os.path.join(args.save_dir, "models")
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Get embeddings
        rna_embs = cross.encode_data("rna", rna_test_ad)
        atac_embs = cross.encode_data("atac", atac_test_ad)

        embs_save_dir = os.path.join(args.save_dir, "embs")
        os.makedirs(embs_save_dir, exist_ok=True)
        rna_embs_df = pd.DataFrame(rna_embs)
        atac_embs_df = pd.DataFrame(atac_embs)
        rna_embs_df.to_csv(os.path.join(embs_save_dir, f"rna_embs_rep{rep}.csv"), index=False)
        atac_embs_df.to_csv(os.path.join(embs_save_dir, f"atac_embs_rep{rep}.csv"), index=False)

        # Evals
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
                "Replicates":i+1,
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
    results.to_csv(os.path.join(res_save_dir, f"metrics_scCross_{args.num_reps}reps.csv"), index=False)
    logging.info(f"scCross results saved to {res_save_dir}.")


if __name__ == "__main__":
    main()