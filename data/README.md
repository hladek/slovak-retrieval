# Slovak Triplets Dataset

This repository contains the Slovak Triplets Dataset, a collection of triplet sentences in the Slovak language designed for training and evaluating document embedding models. Each triplet consists of an anchor sentence, a positive sentence (similar to the anchor), and a negative sentence (dissimilar to the anchor).

## Data Source

The dataset is extracted form the Slovak part of WebFAQ and MQA datasets, which are publicly available collections of question-answer pairs and related information.

- WebFAQ: [Link to WebFAQ dataset]
- MQA: [Link to MQA dataset]

## Creation Process

The triplets were created using the following process:

1. Questions and answers are extracted for each domain from the corpus.
2. For each question, the corresponding answer is identified as the positive example.
3. A negative example is selected from a different question within the same domain to ensure dissimilarity.
4. The resulting triplet (anchor question, positive answer, negative answer) is stored in the dataset.

## Dataset Contents

There are 967317 triplets in total.

## Open issues

- The triplets are not cleaned.
- There might be some duplicate triplets.
- Some answers contain Markdown formatting.
