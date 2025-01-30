# Guide

## How to add a new model

First, you need to create a `YourModelName.py` file containing the training, evaluations, and helper functions of your model. For instance `sCIN.py`.

Note that the name of the `YourModelName.py` file should be the name of your model without special characters except `_` (examples in `sCIN/models`). You should also import all related function from `YourModelName.py` file to the main script, for instance `demo_paired.py` or `demo_unpaired.py` in `demo` directory.

### Paired datasets

To benchamrk your framework against models used in this work (or any models in general), you can use our approach, writting two wrapper functions `train_YourMoelName()` and `get_emb_YouModelName` to prevent data leakage during training. Obviously, your data should be train/test splitted before calling `train_YourMoelName()`.

`train_YourMoelName()` is responsible for training the model and get arguments such as:
```
    def train_YourMoelName(mod1_train, mod2_train, labels_train, epochs, settings):
        ...
        return trained_model
```
- `mod1_train`: Trainig data for data modality 1.
- `mod2_train`: Training data for data modality 2.
- `labels_train`: Training labels. If your model does not use labels during training, feel free to set this to `None`.
- `epochs`: Number of training epochs
- `settings`: Any additional model's parameters and hyperparamters which can be provided via a config file (examples are in `configs/sCIN`).

`train_YourMoelName()` returns trained model object. For instance, in PyTorch a neural network can be defined as class inherting from `torch.nn.Module`. You can make an instance of this class for training which can be saved and returned.

`get_emb_YouModelName` produces separate embeddings from test datasets for each modality and its arguments are as follows:
```
    def get_emb_YouModelName(mod1_test, mod2_test, labels_test, trained_model):
        ...
        return mod1_embs, mod2_embs
```
- `mod1_test`: Test dataset for data modality 1.
- `mod2_test`: Test dataset for data modality 2.
- `labels_test`: Test labels. If your model does not use labels during training, feel free to set this to `None`.
- `trained_model`: The trained model object.

It is not an obligation that `get_emb_YouModelName` returns two separate embeddings, though it is important that the training and evaluations be separated. If you want to do evaluations based on metrics used in this work, then it is necessaray to have separate embeddings for each modality.
These arguments are minimal and feel free to add more if needed. You can see the examples of these functions in `sCIN/models`.

Moreover, you can define an `assess` function for all evaluations on embeddings (example is in `sCIN/benchmarks/assess.py`).

### Unpaired datasets

To replicate the results of this work for unpaired setting, you should define a training wrapper function `train_YourModelName_unapired` inside the `YourModelName.py`. This function can be like:

```
def train_YourModelName_unpaired(mod1_unpaired, mod2_unpaired, [mod1_lbls_unpaired,
                                 mod2_lbls_unpaired], epochs, settings)
    ...
    return trained_model
```
- `mod1_unpaired`: Trainig data for data modality 1. 
- `mod2_unpaired`: Trainig data for data modality 2.
- `[mod1_lbls_unpaired, mod2_lbls_unpaired]`: A list of labels for both modalities.
- `epochs`: Training epochs
- `settings`: Model's settings and hyperparameters.

`mod1_unpaired`, `mod2_unpaired`, `[mod1_lbls_unpaired, mod2_lbls_unpaired]` can be the outputs of the the `make_unpaired` function. See `utils/utils.py`.

Note that there is no need to define a new `get_emb` function since the training is on unpaired data and evaluation is on the paired datasets.

## How to add a new dataset

Add a new dataset in form of `key:value` to the `DATA` dictionary.

- `key`: The name of the dataset. It should be the same as the `--data` flag value.
- `value`: A tuple containing paths to the files of modalities. Note that input files should in `.h5ad` format. 


To see how to use sCIN for paired and unpaired multi-omics integration please refer to `demo` directory.