# Guide

## How to add a new model

First, you need to create a `your_model_name.py` file containing the training, evaluations, and helper functions of your model. For instance `sCIN.py` which is in `models` directory.

In this file you can define `train_your_model_name` and `get_emb_your_model_name` functions. `train_your_model` get training data and training configs as parameters while `get_emb_your_model_name` accepts test dataset

**It is not an obligation that `get_emb_YouModelName` returns two separate embeddings, though it is important that the training and evaluations be separated. If you want to do evaluations based on metrics used in this work, then it is necessaray to have separate embeddings for each modality**.

Then, you may create `run_your_model_name.py` file which calls training and evaluation functions from `models/your_model_name.py` and assessment functions. You can see examples for `sCIN` and others in `scripts` directory.

## How to run sCIN 

If you want to run sCIN on datasets used in this work or new datasets, you should prepare each modality as an `AnnData` object. This object should have `norm_raw_counts` layer containing the normalized and preprocessed count matrix. It also should have `cell_type` in `obs` layer containig cell types for each cell. Please note that for paired datasets, cells must be the same between two modalities. For unpaired datasets, cell can be different, though there should be shared cell types among modalities. 

According to the data (i.e., paired or unpaired) you may run either `run_sCIN.py` or `run_sCIN_unpaired.py` in `scripts` directory. Here are the usages of the scripts' options which are the same for the two settings:

`--rna_file`: One of the modalities (not just RNA) as an `AnnData` object in `h5ad` format.

`--atac_file`: Another modality (not just ATAC) as an `AnnData` object in `h5ad` format.

`--save_dir`: Save directory for embeddings and other results.

`--is_inv_metrics`: Do you want to compute metrics reciprocally (modality 1 -> modality 2 and modality 2 -> modality 1)? This is used for metrics such as Recall@k where we want to see How much close the paired cells are in the embedding space.

`--quick_test`: Just a quick test training sCIN for one epoch for debugging purpose.

`num_reps`: Number of replications (Max. 10). The default is 1. 

The output results is a `csv` file consists of metrics values for sCIN across replications. 

sCIN's configs are in `configs.py`. 

Examples:

```
cd sCIN
. .venv/bin/activate  # Activate the environment

python -m scripts.run_sCIN --rna_file data/share/Ma-2020-RNA.h5ad --atac_file data/share/Ma-2020-ATAC.h5ad --save_dir results/share/sCIN --is_inv_metrics --num_reps 10

python -m scripts.run_sCIN_unpaired --rna_file data/Muto-2021/Muto-2021-RNA-pp.h5ad --atac_file data/Muto-2021/Muto-2021-ATAC-pp.h5ad --save_dir results/share/sCIN_unpaired --is_inv_metrics --num_reps 10
 
```