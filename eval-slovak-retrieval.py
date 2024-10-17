from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from huggingface_hub import snapshot_download
import logging
import pathlib, os
import sys

model_path= sys.argv[1]

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#
data_path  = snapshot_download(repo_id="TUKE-KEMT/retrieval-skquad",repo_type="dataset")


sbert = models.SentenceBERT(model_path)
#print("Model trainable parameters:",sbert.num_parameters(only_trainable=True))
#print("Model total parameters:",sbert.num_parameters())
model = DRES(sbert, batch_size=16)


corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
#### Load the SBERT model and retrieve using cosine-similarity

#### Load the SBERT model and retrieve using cosine-similarity
retriever = EvaluateRetrieval(model, score_function="cos_sim") # dot or "cos_sim" for cosine similarity
results = retriever.retrieve(corpus, queries)
#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
