from sentence_transformers.util import mine_hard_negatives
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

model_name = "Qwen/Qwen3-Embedding-0.6B" # oom on 20G
model_name ="intfloat/multilingual-e5-large"
# Load a Sentence Transformer model
model = SentenceTransformer(model_name)
# https://sbert.net/docs/package_reference/util.html#sentence_transformers.util.mine_hard_negatives
# Load a dataset to mine hard negatives from
dataset = load_dataset("TUKE-KEMT/slovak-web-qa-pairs", split="train")
print(dataset)
hard_dataset = mine_hard_negatives(
  dataset=dataset,
  model=model,
  anchor_column_name="anchor",
  positive_column_name="positive",
  range_min=5,
  range_max=20,
#  max_score=0.8,
#  relative_margin=0.3,
  num_negatives=1,
  sampling_strategy="random",
  batch_size=32,
  use_faiss=True,
  cache_folder="./cache"
  )
print(hard_dataset)
hard_dataset.save_to_disk("out-hardqa")
