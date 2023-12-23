# Named-Entity-Recognition

## Task 1 Model Architecture
![image](https://github.com/ayushiiamin/Named-Entity-Recognition/assets/77382840/1c91c467-b1c7-4a7c-8312-d6a9fed50e2a)


## Task 2 Model Architecture
BiLSTM_Model_2(
  (embedding): Embedding(23626, 101)
  (lstm): LSTM(101, 256, batch_first=True, bidirectional=True)
  (lin): Linear(in_features=512, out_features=128, bias=True)
  (dropout): Dropout(p=0.33, inplace=False)
  (elu): ELU(alpha=2.0)
  (classifier): Linear(in_features=128, out_features=9, bias=True)
  (softmax): Softmax(dim=2)
)

## Results

### Task 1
| Metric    | Score |
| -------- | ------- |
| Precision  | 74.87%    |
| Recall | 69.45%     |
| F1    | 72.06%    |

### Task 2
| Metric    | Score |
| -------- | ------- |
| Precision  | 84.85%    |
| Recall | 73.51%     |
| F1    | 78.77%    |

- F1-Score (dev2.out file) = 78.77%
- F1-Score (test2.out file) = 66.36%
