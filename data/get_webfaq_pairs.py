# https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
from datasets import load_dataset
import sys
import json

dataset = load_dataset("PaDaS-Lab/webfaq" , name="slk",split="default", streaming=True)
# id, text, name, domain, answers
for sample in dataset:
    question = sample["question"]
    did = sample["id"]
    domain = sample["origin"]
    answer = sample["answer"]
    print(json.dumps({"id":did,"domain":domain,"anchor":question,"positive":answer}))
