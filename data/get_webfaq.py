# https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
from datasets import load_dataset
import sys
import json
# id, origin, url, question, answer, topic, question_type, 

dataset = load_dataset("PaDaS-Lab/webfaq" , name="slk", split="default", streaming=True)
for sample in dataset:
    question = sample["question"]
    answer = sample["answer"]
    did = sample["id"]
    doc = {
            "id":did,
            "question": question,
            "answer": answer,
    }
    print(json.dumps(doc))
