# DISGCN
We release two datasets for social recommendation, along with the implementation of DISGCN model. The implementaion is based on source codes of DiffNet ([paper](https://arxiv.org/abs/2002.00844), [codes](https://github.com/PeiJieSun/diffnet)).

## Run the codes
Environment: TensorFlow 2.1, Python 3
- Start a visdom server with port 1496 (Beidian) or 1469 (Beibei).
    ```
    visdom -port=1496
    ```
- Train the model and evaluate.
    ```
    python entry.py --data_name Beidian --model_name disgcn --gpu 0 1
    ```

## Description of files
- conf/Beidian/disgcn.ini

    Configuration of parameters. We explain important ones as follows.
    
    - dimension: embedding size
    - social_lr: learning rate of contrastive learning (social loss)
    - reg: coefficient of L2 regularization for recommendation loss
    - sreg: coefficient of L2 regularization for social loss
    - pre_train: file of embeddings for pretraining GNN-based model

- data/Beidian/

    Data files for training and evaluation.

    - social: each line is *user_id friend_id*

    - train/val/test: each line is *user_id item_id*, which is for training, validation and testing respectively

    - ufi: influence behavioral data, each line is *user_id friend_id item_id*

- embedding/Beidian/

    embeddings for pretraining

- pretrain/Beidian/

    For saving model parameters