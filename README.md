We provide the code (in pytorch) and datasets for our paper [**"HGPROMPT: Bridging Homogeneous and Heterogeneous Graphs
for Few-shot Prompt Learning"**](https://arxiv.org/pdf/2312.01878.pdf), 
which is accepted by AAAI2024.

## Package Dependencies

- cuda 11.3
- dgl0.9.0-cu113
- dgllife

## Running experiments

The default dataset is ACM.  You need to change the corresponding parameters in *pre_train.py* and *run.py* to train and evaluate on other datasets.

Pretrain:

- python pre_train.py

Prompting and test:

- python run.py
