# Named Entity Recognition with BiLSTM and GloVe

A deep learning Named Entity Recognition (NER) system built with **PyTorch** to identify people, organizations, locations, and miscellaneous entities in text.

This project explores two sequence-labeling architectures:

1. A **BiLSTM model with learned word embeddings**
2. An enhanced **BiLSTM model using pretrained GloVe embeddings and capitalization features**

The improved model achieved a **78.77% F1-score on the development set**, compared with **72.06%** for the baseline architecture.

---

## Project Overview

Named Entity Recognition is the task of identifying and classifying meaningful entities in text.

For example:

```text
Barack Obama visited Paris.
```

can be labeled as:

```text
Barack   B-PER
Obama    I-PER
visited  O
Paris    B-LOC
```

The goal of this project was to build a neural sequence-tagging system that predicts an entity label for every word in a sentence.

The model recognizes four main entity types:
- ```PER``` — Person
- ```ORG``` — Organization
- ```LOC``` — Location
- ```MISC``` — Miscellaneous entity
using the BIO tagging format.
