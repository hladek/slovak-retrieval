cat webfaq.jsonl mqa.jsonl | split --verbose -l 100000 -d  --additional-suffix=.jsonl - slovak-triplets/data-
