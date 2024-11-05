# Slovak Retrieval

This repository provides tools for fine-tuning and evaluating Slovak embedding models, specifically adapted for question answering and retrieval tasks in the Slovak language.

i
## Models and Datasets

- **[Slovak Embedding Model](https://huggingface.co/TUKE-DeutscheTelekom/slovakbert-skquad-mnlr)**: A fine-tuned [SlovakBERT model](https://huggingface.co/gerulata/slovakbert), trained on the [Slovak question answering dataset](https://huggingface.co/datasets/TUKE-DeutscheTelekom/skquad).
- **[Slovak Dataset for Embedding Model Evaluation](https://huggingface.co/datasets/TUKE-KEMT/retrieval-skquad)**: Derived from the testing part of the Slovak question answering dataset, used to evaluate retrieval performance.



## Installation

To get started, first clone this repository:


```
git clone https://github.com/hladek/slovak-retrieval.git
cd slovak-retrieval
```

To run examples, you will need to create a python virtual environment and pytorch with CUDA support. 


Install 'sentence_transformers' 'beir' and 'mteb'.

```
pip install -r requirements.txt
```

## Fine Tuning Slovak Embedding Models

We provide a script to fine-tune your own BERT or SBERT  type model for Slovak language embeddings.


1. Download the SK Quad Database.

```
wget https://files.kemt.fei.tuke.sk/corpora/sk-quad/sk-quad-220614.tar.gz
tar zxf sk-quad-220614.tar.gz
```

2. Train the model

```
CUDA_VISIBLE_DEVICES=0 python ./train-bi-mnlr.py --train_file ./sk-quad-220614/sk-quad-220614-train.json --model_name gerulata/slovakbert --test_file ./sk-quad-220614/sk-quad-220614-dev.json --epochs 5
```

## Evaluation of the Slovak Embedding Models with BEIR


An example script for evaluation  with a [BEIR](https://github.com/beir-cellar/beir) framework:


```
python ./eval-slovak-retrieval.py  TUKE-DeutscheTelekom/slovakbert-skquad-mnlr 
``` 


## Evaluation of the Slovak Embedding Models with MTEB

Alternatively, you can evaluate the model using the [MTEB](https://github.com/embeddings-benchmark/mteb) framework:

```
mteb run -m TUKE-DeutscheTelekom/slovakbert-skquad-mnlr -t SKQuadRetrieval --verbosity 3
```

