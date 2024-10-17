# slovak-retrieval

Finetuning and evaluation of the Slovak Embedding Models

- [Slovak Embedding Model](https://huggingface.co/TUKE-DeutscheTelekom/slovakbert-skquad-mnlr) fine-tuned [SlovakBERT](https://huggingface.co/gerulata/slovakbert) on the [Slovak question answering dataset](https://huggingface.co/datasets/TUKE-DeutscheTelekom/skquad).
- [Slovak Dataset for Embedding Model Evaluation](https://huggingface.co/datasets/TUKE-KEMT/retrieval-skquad), based on the testing part of the [Slovak question answering dataset](https://huggingface.co/datasets/TUKE-DeutscheTelekom/skquad).

## Fine Tuning Slovak Embedding Models

We provide a script to fine-tune your own BERT or SBERT  type model for Slovak language embeddings.

1. Create a python virtual environment and pytorch with CUDA support. Install 'sentence_transformers' and 'beir'.

```
pip install -r requirements.txt
```

3. Download the SK Quad Database.

```
wget https://files.kemt.fei.tuke.sk/corpora/sk-quad/sk-quad-220614.tar.gz
tar zxf sk-quad-220614.tar.gz
```

4. Train the model

```
CUDA_VISIBLE_DEVICES=0 python ./train-bi-mnlr.py --train_file ./sk-quad-220614/sk-quad-220614-train.json --model_name gerulata/slovakbert --test_file ./sk-quad-220614/sk-quad-220614-dev.json --epochs 5
```

## Evaluation of the Slovak Embedding Models


```
python ./eval-slovak-retrieval.py  intfloat/multilingual-e5-base
``` 




