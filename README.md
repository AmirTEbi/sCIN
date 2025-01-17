## Single-cell Contrastive INtegration (sCIN)

sCIN is a neural network framework inspired by CLIP (Radford _et al_.(2022)) for single-cell multi-omics data integration.

Paper: 

To benchamrk your framework against models in this work (or any models in general), you can write two wrapper functions `train_YourMoelName()` and `get_emb_YouModelName` to prevent data leakage during training. Obviously, your data should be train/test splitted before calling `train_YourMoelName()`. 

`train_YourMoelName()` is responsible for training the model and get parameters such as:
- `mod1_train`: Trainig data for data modality 1.
- `mod2_train`: Training data for data modality 2.
- `labels_train`: Training labels. If your model does not use labels during training, feel free to set this to `None`.
- `epochs`: Number of training epochs
- `settings`: Any additional model's parameters and hyperparamters which can be provided via a config file (examples are in `configs/sCIN`).

These arguments are minimal and your training wrapper can accept more arguments if needed.

