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

---

## Entity Labels

The model predicts one of nine tags for every token:
- ```B-LOC``` - Beginning of a location
- ```I-LOC``` - Inside a location
- ```B-MISC``` - Beginning of a miscellaneous entity
- ```I-MISC``` - Inside a miscellaneous entity
- ```B-ORG``` - Beginning of an organization
- ```I-ORG``` - Inside an organization
- ```B-PER``` - Beginning of a person
- ```I-PER``` - Inside a person
- ```O``` - Not part of a named entity

---

## Key Highlights

- Built an end-to-end Named Entity Recognition pipeline using PyTorch.
- Implemented a Bidirectional LSTM for token-level sequence labeling.
- Compared learned embeddings against pretrained GloVe embeddings.
- Added an explicit capitalization feature to improve token representation.
- Used class-weighted cross-entropy loss to address label imbalance.
- Used packed padded sequences to efficiently process variable-length sentences.
- Improved development-set F1 from 72.06% to 78.77%.
- Generated prediction files for both development and unseen test data.

---

## Model Architecture

### Model 1 — BiLSTM with Learned Embeddings
The baseline model learns word embeddings directly from the training corpus.

#### Architecture
```text
Input Tokens
     │
     ▼
Word-to-Index Mapping
     │
     ▼
Learned Embedding Layer
     │
     ▼
Bidirectional LSTM
     │
     ▼
Dropout
     │
     ▼
Linear Layer
     │
     ▼
ELU Activation
     │
     ▼
Classifier
     │
     ▼
9 NER Tag Probabilities
```
#### Architecture Details
```text
Embedding Size     : 100
BiLSTM Hidden Size : 256
LSTM Layers        : 1
Bidirectional      : Yes
Dropout            : 0.33
Linear Output      : 128
Output Classes     : 9
```
The model predicts an NER tag independently for each token while using bidirectional context from the entire sentence.

---

### Model 2 — BiLSTM with GloVe + Capitalization
The second model improves the token representation using pretrained GloVe embeddings.

Each token representation contains:
```text
100-D GloVe Embedding
        +
1-D Capitalization Feature
        =
101-D Token Representation
```

The capitalization feature indicates whether the original token begins with an uppercase character.

For example:
```text
London  → capitalization feature = 1
city    → capitalization feature = 0
```

This feature is particularly useful for NER because names of people, organizations, and locations are frequently capitalized.

#### Architecture
```text
Input Tokens
     │
     ▼
GloVe 100-D Embedding
     │
     ├── Capitalization Feature
     │
     ▼
101-D Token Representation
     │
     ▼
Bidirectional LSTM
     │
     ▼
Dropout
     │
     ▼
Linear Layer
     │
     ▼
ELU Activation
     │
     ▼
Classifier
     │
     ▼
9 NER Tag Probabilities
```

#### Architecture Details
```text
Embedding Size     : 101
                     100-D GloVe
                     + 1 capitalization feature

BiLSTM Hidden Size : 256
LSTM Layers        : 1
Bidirectional      : Yes
Dropout            : 0.33
Linear Output      : 128
Output Classes     : 9
```

---

## Technical Approach

### 1. Data Parsing
The training data is organized as token-level sequences.

Each row contains:
```text
Token Position | Word | NER Tag
```

Blank lines separate individual sentences.

The preprocessing pipeline groups words and tags into sentence-level sequences.

---

### 2. Vocabulary Construction
A vocabulary is created from the training corpus by assigning each unique token an integer ID.

Two special tokens are included:
```text
unk  → unseen words
pad  → sequence padding
```

Words in the development or test sets that do not appear in the training vocabulary are mapped to ```unk```.

---

### 3. BIO Tag Encoding
NER labels are converted into numerical classes.
```text
B-LOC  → 0
B-MISC → 1
B-ORG  → 2
B-PER  → 3
I-LOC  → 4
I-MISC → 5
I-ORG  → 6
I-PER  → 7
O      → 8
```
Padding positions use a separate ignored value during training.

---

### 4. Variable-Length Sequence Handling
Sentences naturally have different lengths.

The project pads sentences to enable batch processing, but avoids unnecessary computation over padded tokens by using:
```python
torch.nn.utils.rnn.pack_padded_sequence
```
before the LSTM and:
```python
torch.nn.utils.rnn.pad_packed_sequence
```
afterward.

This allows the BiLSTM to efficiently process only the meaningful parts of each sentence.

---

### 5. Bidirectional Sequence Modeling
