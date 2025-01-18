# Tutorial

To benchamrk your framework against models in this work (or any models in general), you can use our approach, writting two wrapper functions `train_YourMoelName()` and `get_emb_YouModelName` to prevent data leakage during training. Obviously, your data should be train/test splitted before calling `train_YourMoelName()`.

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

To see how to use sCIN for paired and unpaired multi-omics integration please refer to `demo.ipynb`.