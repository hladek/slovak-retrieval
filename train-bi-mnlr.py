# https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder_mnrl.py
#https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_cross-encoder_scratch.py
# https://huggingface.co/blog/train-sentence-transformers
import sys
import json
from torch.utils.data import DataLoader
import logging
from sentence_transformers import LoggingHandler, util
from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample
from sentence_transformers import models
from sentence_transformers import losses
from datetime import datetime
import random
import collections
import argparse
from sentence_transformers.evaluation import TripletEvaluator

from beir.retrieval import models as beirmodels
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.datasets.data_loader import GenericDataLoader
from huggingface_hub import snapshot_download
from datasets import Dataset

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--train_file", required=True)
parser.add_argument("--test_file", required=True)
parser.add_argument("--model_name", required=True)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--warmup_steps", default=100, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--lr", default=2e-5, type=float)
args = parser.parse_args()

print(args)


trainfile = args.train_file
testfile = args.test_file
model_name = args.model_name

train_batch_size = args.train_batch_size
num_epochs = args.epochs
learning_rate = args.lr
max_seq_length = args.max_seq_length
warmup_steps = args.warmup_steps
#learning_rate = 1e-5
run_name =model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

model_save_path = 'output/training_skquad-bi-encoder-mnrl_'+ run_name 

def squad_questions(fname):
    with open(fname,"rb") as f:
        doc = json.load(f)
        examples = []
        all_questions = []
        for item in doc["data"]:
            title = item["title"]
            contexts = {}
            answerable_questions = []
            unanswerable_questions = []

            for paragraph in item["paragraphs"]:
                context = paragraph["context"]
                aq = []
                uq = []
                #print(context)
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    if "is_impossible" in qa and qa["is_impossible"]:
                        uq.append(question)
                    else:
                        aq.append(question)
                        all_questions.append(question)
                #print(aq,uq)
                #print("-----")
                if context in contexts:
                    aa,ba = contexts[context]
                    aa += aq
                    ba += uq
                else:
                    contexts[context] = (aq,uq)
            cl = list(contexts.keys())
            l = len(cl)
            for i,context in enumerate(cl):
                aq = contexts[context][0]
                uq = contexts[context][1]
                gn = len(aq) - len(uq)
                # fill hard negatives
                if gn > 0:
                    # find questions for other paragraphs in the same wiki page
                    other_questions = []
                    for j,oc in enumerate(cl):
                        if j == i:
                            continue
                        other_questions += contexts[oc][0]
                    loe = len(other_questions)
                    if loe == gn:
                        # fill all other questions as negatives
                        uq += other_questions
                    elif loe > gn:
                        # sample from other questions
                        uq += random.sample(other_questions,gn)
                    else:
                        # sample from other quastions
                        uq += other_questions
                        # and sample questions from other pages if not sufficient
                        uq += random.sample(all_questions,gn-loe)
                if gn < 0:
                    del uq[-gn:]
                for a,u in zip(aq,uq):
                    examples.append({"query":context,"positive":a,"negative":u})
    return examples




train_examples = list(map(lambda x: InputExample(texts=[x["query"],x["positive"],x["negative"]]),squad_questions(trainfile)))

train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=train_batch_size)

# Configure the training
logging.info("Warmup-steps: {}".format(warmup_steps))

use_pretrained_model = False
if "sentence-" in model_name or "e5" in model_name:
    use_pretrained_model = True

# Load our embedding model
if use_pretrained_model:
    logging.info("use pretrained SBERT model, using original pooling")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
else:
    logging.info("Create new SBERT model")
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


train_loss = losses.MultipleNegativesRankingLoss(model=model)
#train_loss = losses.TripletLoss(model=model)

eval_dataset = Dataset.from_list(squad_questions(testfile))
# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["query"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
    name="skquad-test",
)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
                    epochs=num_epochs,
                    warmup_steps=warmup_steps,
                    use_amp=True,
                    checkpoint_path=model_save_path,
                    checkpoint_save_steps=len(train_dataloader),
                    optimizer_params = {'lr': learning_rate},
                    #logging_steps = 100,
                    #eval_steps = 300,
                    #evaluator = dev_evaluator,
                    )
#Save latest model
model.save(model_save_path+'-latest')

# Evaluation

dev_evaluator(model)

del model

beirmodel = DRES(beirmodels.SentenceBERT(model_save_path+"-latest"), batch_size=16)

beir_data_path  = snapshot_download(repo_id="TUKE-KEMT/retrieval-skquad",repo_type="dataset")
corpus, queries, qrels = GenericDataLoader(data_folder=beir_data_path).load(split="test")
#### Load the SBERT model and retrieve using cosine-similarity

#### Load the SBERT model and retrieve using cosine-similarity
retriever = EvaluateRetrieval(beirmodel, score_function="cos_sim") # dot or "cos_sim" for cosine similarity
results = retriever.retrieve(corpus, queries)
#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
logging.info(ndcg, _map, recall, precision )
