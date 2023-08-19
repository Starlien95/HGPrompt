# HGPrompt

## Package Dependencies

- cuda 11.3
- dgl0.9.0-cu113
- dgllife

## Running experiments

The default dataset is ACM.  You need to change the corresponding parameters in *pre_train.py* and *run.py* to train and evaluate on other datasets.

Pretrain:

- python pre_train.py

Prompt tune and test:

- python run.py
