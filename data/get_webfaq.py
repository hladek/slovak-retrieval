# https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
from datasets import load_dataset
import sys
import json
import collections

dataset = load_dataset("PaDaS-Lab/webfaq" , name="slk",split="default", streaming=True)
# id, text, name, domain, answers
prev_domain = None
questions = []
answers = collections.deque()
for sample in dataset:
    question = sample["question"]
    did = sample["id"]
    domain = sample["origin"]
    answer = sample["answer"]
    if prev_domain is not None and domain != prev_domain:
        rot = 2
        good_answers = list(answers)
        answers.rotate(1)
        bad_answers = list(answers)
        for question,answer,bad_answer in zip(questions,good_answers,bad_answers):
            print(json.dumps({"domain":domain,"anchor":question,"positive":answer,"negative":bad_answer}))
        questions = []
        answers.clear()

    prev_domain = domain
    questions.append(question)
    answers.append(answer)
