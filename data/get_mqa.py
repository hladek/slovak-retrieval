# https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
from datasets import load_dataset
import sys


dataset = load_dataset("clips/mqa" , name="sk-all-question", split="train", streaming=True)
for sample in dataset:
    print(sample)
