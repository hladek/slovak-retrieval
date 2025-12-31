import datasets
import sys

d = datasets.load_from_disk(sys.argv[1])
print(d)
for example in d:
    print(example)
