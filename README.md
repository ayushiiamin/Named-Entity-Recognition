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
A Bidirectional LSTM processes the sentence in both directions.
```text
Left Context  ─────► Token ◄───── Right Context
```
This is especially useful for NER.

For example:
```text
Apple released a new product.
```
and
```text
She bought an apple.
```
contain the same surface word but very different contextual meanings.

By incorporating information from both preceding and following words, the BiLSTM can make more informed token-level predictions.

---

### 6. Class-Imbalance Handling
NER datasets are often dominated by the O tag because most words are not named entities.

To reduce bias toward frequent classes, the training pipeline calculates inverse class-frequency weights and supplies them to:
```python
nn.CrossEntropyLoss
```
This gives underrepresented entity classes more influence during optimization.

Padding tokens are ignored during loss calculation.

---

### 7. Model Optimization

#### Model 1
The baseline model uses:
```text
Optimizer      : SGD
Learning Rate  : 0.1
Momentum       : 0.9
Nesterov       : Enabled
Epochs         : 50
Batch Size     : 32
```
A multi-step learning-rate scheduler gradually reduces the learning rate during training.

#### Model 2
The GloVe-based model uses:
```text
Optimizer      : SGD
Learning Rate  : 0.07
Momentum       : 0.9
Nesterov       : Enabled
Epochs         : 150
Batch Size     : 16
```

---

### 8. Inference

After training, both models are saved as PyTorch state dictionaries:
```text
blstm1.pt
blstm2.pt
```
During inference:
```text
Sentence
   │
   ▼
Token IDs
   │
   ▼
BiLSTM Model
   │
   ▼
Class Probabilities
   │
   ▼
Argmax
   │
   ▼
Predicted BIO Tags
```
Predictions are written to output files while preserving sentence boundaries and token order.

---

## Results

### Development Set
| Model | Precision | Recall | F1 |
| -------- | -------- | -------- | -------- |
| BiLSTM with learned embeddings    | 74.87%  | 69.45%   | 72.06%   |
| BiLSTM + GloVe + capitalization   | 84.85%   | 73.51%   | 78.77%   |

The GloVe-based architecture improved development-set F1 by:
```text
78.77 - 72.06 = 6.71 percentage points
```
The second model also achieved substantially higher precision, suggesting that pretrained semantic information and capitalization cues helped reduce incorrect entity predictions.

---

### Test Performance
The enhanced model achieved:
```text
Test F1: 66.36%
```
The difference between development and test performance highlights the importance of evaluating sequence models on unseen data and monitoring generalization beyond the development set.

---

### Why Model 2 Performed Better

The strongest architecture combines three useful signals:
1. Pretrained Semantic Knowledge: GloVe embeddings provide semantic relationships learned from a much larger corpus than the NER training dataset.
2. Bidirectional Context: The BiLSTM captures information from both sides of each token.
3. Capitalization Information: Capitalization provides a useful linguistic signal for proper nouns and named entities.

Together:
```text
Pretrained Semantics
        +
Bidirectional Context
        +
Capitalization
        │
        ▼
Improved NER Performance
```

---

## Tech Stack
### Programming
- Python
### Deep Learning
- PyTorch
- Bidirectional LSTM
- Neural sequence labeling
### NLP
- Named Entity Recognition
- BIO tagging
- GloVe embeddings
### Data Processing
- NumPy
- pandas
- NLTK
### Machine Learning
- scikit-learn
- weighted cross-entropy
- sequence classification metrics

---

## Repository Structure
```text
Named-Entity-Recognition/
│
├── HW4-CSCI544-Final.py
│   └── Data preprocessing, model definitions, training, inference,
│       and output generation
│
├── blstm1.pt
│   └── Trained baseline BiLSTM model
│
├── blstm2.pt
│   └── Trained GloVe-based BiLSTM model
│
├── dev1.out
│   └── Baseline predictions on development data
│
├── dev2.out
│   └── GloVe model predictions on development data
│
├── test1.out
│   └── Baseline predictions on test data
│
├── test2.out
│   └── GloVe model predictions on test data
│
└── README.md
```

---

## Running the Project
### 1. Clone the Repository
```python
git clone https://github.com/ayushiiamin/Named-Entity-Recognition.git
cd Named-Entity-Recognition
```
### 2. Install Dependencies
```bash
pip install torch torchvision
pip install numpy pandas
pip install scikit-learn nltk beautifulsoup4 tqdm
```
### 3. Download GloVe Embeddings
The second model expects:
```python
glove.6B.100d
```
Place the GloVe embedding file in the project directory or update the path in the Python script.
### 4. Add Dataset Files
The script expects:
```python
data/train
data/dev
data/test
```
with sentence-level token/tag formatting.
### 5. Run Inference
```python
python HW4-CSCI544-Final.py
```
The script loads the trained models and produces:
```text
dev1.out
dev2.out
test1.out
test2.out
```

---

## Potential Improvements
A modernized version of this project could explore:
- BiLSTM + CRF sequence decoding
- BERT or RoBERTa token classification
- contextual embeddings instead of static GloVe vectors
- character-level embeddings for morphological features
- transformer-based NER models
- subword token alignment
- stronger regularization
- hyperparameter optimization
- entity-level error analysis
- model-serving through a REST API

A modern architecture could look like:
```text
Raw Text
   │
   ▼
Subword Tokenization
   │
   ▼
Transformer Encoder
   │
   ▼
Token-Level Classifier
   │
   ▼
Optional CRF Layer
   │
   ▼
Named Entity Tags
```
