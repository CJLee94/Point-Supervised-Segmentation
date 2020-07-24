# lcfcn-pseudo

## Introduction

This project is a re-implementation of PseudoEdgeNet([paper](https://arxiv.org/pdf/1906.02924)) and LCFCN([paper](https://arxiv.org/abs/1807.09856)|[code](https://github.com/ElementAI/LCFCN.git)). 
The LCFCN part contains most of the original code while the PseudoEdgeNet was implemented according to the paper as we weren't able to find existing code from the project.

## Usage

### Environment setup

To pre-setup the environment, we recommand to use 
```
conda env create -f environment.yml
```

### Download Dataset


### Train and cross-validation
Both [`trainval.py`](trainval.py) and [`trainval.ipynb`](trainval.ipynb) can be used to train the model. In [`trainval.ipynb`](trainval.ipynb), you can get a brief introduction of the setup of the dataset.
