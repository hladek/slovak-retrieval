# https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
from datasets import load_dataset
import sys
import json
import collections

dataset = load_dataset("clips/mqa" , name="sk-all-question", split="train", streaming=True)
# id, text, name, domain, answers
prev_domain = None
questions = []
answers = collections.deque()
for sample in dataset:
    sample_answers = sample["answers"]
    question = sample["name"]
    did = sample["id"]
    domain = sample["domain"]
    answer = sample_answers[0]["text"]
    if prev_domain is not None and domain != prev_domain:
        rot = 2
        good_answers = list(answers)
        answers.rotate(1)
        bad_answers = list(answers)
        for question,answer,bad_answer in zip(questions,good_answers,bad_answers):
            print(question)
            print("---")
            print(answer)
            print("---")
            print(bad_answer)
            print(">>>")
        questions = []
        answers.clear()

    prev_domain = domain
    questions.append(question)
    answers.append(answer)
