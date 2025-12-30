# https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
from datasets import load_dataset
import sys
import json
import collections

dataset = load_dataset("clips/mqa" , name="sk-all-question", split="train", streaming=True)
# id, text, name, domain, answers
for sample in dataset:
    sample_answers = sample["answers"]
    question = sample["name"]
    did = sample["id"]
    domain = sample["domain"]
    answer = sample_answers[0]["text"]
    print(json.dumps({"id":did,"domain":domain,"anchor":question,"positive":answer}))
