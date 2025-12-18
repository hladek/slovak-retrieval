# https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
from datasets import load_dataset
import sys
import json


dataset = load_dataset("clips/mqa" , name="sk-all-question", split="train", streaming=True)
# id, text, name, domain, answers
for sample in dataset:
    answers = sample["answers"]
    text = sample["text"]
    did = sample["id"]
    if len(answers) == 0:
        continue
    for answer in answers:
        doc = {
                "id":did,
                "question": text,
                "answer": answer,
        }
        print(json.dumps(doc))
    print(sample)
