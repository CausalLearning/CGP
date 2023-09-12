# Comprehensive Graph Gradual Pruning for Sparse Training in Graph Neural Networks

Open-sourced implementation for TNNLS 2023.



<h2> Abstract </h2>

1) We propose a graph gradual pruning framework, namely
CGP, to reduce the training and inference computing costs
of GNN models while preserving their accuracy.

2) We comprehensively sparsify the elements of GNNs,
including graph structures, the node feature dimension, and
model parameters, to significantly improve the efficiency
of GNN models.

3) Experimental results on various GNN models and datasets
consistently validate the effectiveness and efficiency of
our proposed CGP.



<h2> Python Dependencies </h2>

Our proposed Gapformer is implemented in Python 3.7 and major libraries include:

* [Pytorch](https://pytorch.org/) = 1.11.0+cu113
* [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) torch-geometric=2.2.0

More dependencies are provided in **requirements.txt**.

<h2> To Run </h2>

Once the requirements are fulfilled, use this command to run Gapformer:

`python main.py`

<h2> Datasets </h2>

All datasets used in this paper can be downloaded from [PyG](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html).
