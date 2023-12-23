#!/usr/bin/env python
# coding: utf-8

# # Homework 4 - CSCI544

# ## Name: Ayushi Amin
# ## USC ID: 6796176811

# ## References
# * GeeksforGeeks. (2019). Python Find maximum length sub list in a nested list. [online] Available at: https://www.geeksforgeeks.org/python-find-maximum-length-sub-list-in-a-nested-list/
# * Hever, G. (2020). Sentiment Analysis with Pytorch — Part 4 — LSTM\BiLSTM Model. [online] Medium. Available at: https://galhever.medium.com/sentiment-analysis-with-pytorch-part-4-lstm-bilstm-model-84447f6c4525
# * Jason (2013). Python list extend functionality using slices. [online] Stack Overflow. Available at: https://stackoverflow.com/questions/16627315/python-list-extend-functionality-using-slices
# * Prasad, A. (2020). PyTorch For Deep Learning — Feed Forward Neural Network. [online] Medium. Available at: https://medium.com/analytics-vidhya/pytorch-for-deep-learning-feed-forward-neural-network-d24f5870c18
# * Pytorch.org. (2023). Training a Classifier — PyTorch Tutorials 1.13.1+cu117 documentation. [online] Available at: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# * Pytorch.org. (2023). MultiStepLR — PyTorch 2.0 documentation. [online] Available at: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html
# * Theiler, S. (2019). Basics of Using Pre-trained GloVe Vectors in Python. [online] Medium. Available at: https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db



import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import re
from bs4 import BeautifulSoup
import copy
 
#import contractions
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import operator





trainData = open('data/train', 'r')
trainData_lines = trainData.readlines()





wordList_train = []
for line in trainData_lines:
    wordList_train.append(line.split())







wordFreqDict = {}





c=0
for subTestList in wordList_train:
    if subTestList != []:
        if subTestList[1] not in wordFreqDict:
            c+=1
            wordFreqDict[subTestList[1]] = c





wordFreqDict['unk'] = wordFreqDict[max(wordFreqDict, key=wordFreqDict.get)]+1





wordFreqDict['pad'] = len(wordFreqDict)





trainVector = []
trainVector_copy = []
trainVector_words = []

train_len = []





c = 0
for subList in wordList_train:
    if subList == []:
        c+=1
    elif subList[0] == '1':
        trainVector.append([])
        trainVector[c].append(wordFreqDict[subList[1]])
        
        trainVector_copy.append([])
        trainVector_copy[c].append(wordFreqDict[subList[1]])
        
        trainVector_words.append([])
        trainVector_words[c].append(subList[1])
    else:
        trainVector[c].append(wordFreqDict[subList[1]])
        trainVector_copy[c].append(wordFreqDict[subList[1]])
        trainVector_words[c].append(subList[1])








for sent in trainVector_words:
    train_len.append(len(sent))





#(GeeksforGeeks, 2019)
def getMaxLength(lis):
    maxLenLst = max(lis, key=len)
    maxLen = len(maxLenLst)
    
    return maxLen





maxLen = getMaxLength(trainVector)

for lst in trainVector:
    for i in range(len(lst),maxLen):
        lst.append(0)





tagFreqDict = {}





tagFreqDict = {
    'B-LOC': 0, 
    'B-MISC': 1, 
    'B-ORG': 2, 
    'B-PER': 3, 
    'I-LOC': 4, 
    'I-MISC': 5, 
    'I-ORG': 6, 
    'I-PER': 7, 
    'O': 8}





freqTagDict = {
    0:'B-LOC',
    1 :'B-MISC',
    2 :'B-ORG',
    3 :'B-PER',
    4 :'I-LOC',
    5 :'I-MISC',
    6 :'I-ORG',
    7 :'I-PER',
    8 :'O',
    -1 :'pad'}





trainVector_tag = []
trainVector_tag_copy = []
train_tags = []





d = 0
for subList in wordList_train:
    if subList == []:
        d+=1
    elif subList[0] == '1':
        trainVector_tag.append([])
        trainVector_tag[d].append(tagFreqDict[subList[2]])
        
        trainVector_tag_copy.append([])
        trainVector_tag_copy[d].append(tagFreqDict[subList[2]])
        
        train_tags.append([])
        train_tags[d].append(subList[2])
    else:
        trainVector_tag[d].append(tagFreqDict[subList[2]])
        
        trainVector_tag_copy[d].append(tagFreqDict[subList[2]])

        train_tags[d].append(subList[2])





posTagCountDict = {}


# In[34]:


for subWordList in wordList_train:
    if subWordList != []:
        if subWordList[2] in posTagCountDict:
            posTagCountDict[subWordList[2]]+=1
        else:
            posTagCountDict[subWordList[2]] = 1


# In[36]:


freq = {}
for k,v in posTagCountDict.items():
    freq[tagFreqDict[k]] = v


# In[37]:


from sklearn.preprocessing import normalize


# In[39]:


frequencies = [0 for i in range(9)]
for k,v in freq.items():
    frequencies[k] = v


# In[41]:


import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim


# In[42]:


frequencies = np.array(frequencies)
frequencies = frequencies/ np.sum(frequencies)
frequencies = 1/frequencies
frequencies = torch.from_numpy(frequencies).float()


# In[44]:


maxLen_tag = getMaxLength(trainVector_tag)

for lst in trainVector_tag:
    for i in range(len(lst),maxLen_tag):
        lst.append(-1)


# ### Dev Data

# In[52]:


devData = open('data/dev', 'r')
devData_lines = devData.readlines()


# In[53]:


wordList_dev = []
for line in devData_lines:
    wordList_dev.append(line.split())


# In[54]:


devVector = []
devVector_copy = []
devVector_words = []

dev_len = []


# In[55]:


e = 0
for subList in wordList_dev:
    if subList == []:
        e+=1
    elif subList[0] == '1':
        devVector.append([])
        
        devVector_copy.append([])
        if subList[1] not in wordFreqDict:
            devVector[e].append(wordFreqDict['unk'])
            devVector_copy[e].append(wordFreqDict['unk'])
        else:
            devVector[e].append(wordFreqDict[subList[1]])
            devVector_copy[e].append(wordFreqDict[subList[1]])
        
        devVector_words.append([])
        devVector_words[e].append(subList[1])
    else:
        if subList[1] not in wordFreqDict:
            devVector[e].append(wordFreqDict['unk'])
            devVector_copy[e].append(wordFreqDict['unk'])
        else:
            devVector[e].append(wordFreqDict[subList[1]])
            devVector_copy[e].append(wordFreqDict[subList[1]])
        
        devVector_words[e].append(subList[1])


# In[56]:


for sent in devVector_words:
    dev_len.append(len(sent))


# In[58]:


devVector_tag = []
devVector_tag_copy = []
dev_tags = []


# In[59]:


f = 0
for subList in wordList_dev:
    if subList == []:
        f+=1
    elif subList[0] == '1':
        devVector_tag.append([])
        devVector_tag[f].append(tagFreqDict[subList[2]])
        
        devVector_tag_copy.append([])
        devVector_tag_copy[f].append(tagFreqDict[subList[2]])
        
        dev_tags.append([])
        dev_tags[f].append(subList[2])
    else:
        devVector_tag[f].append(tagFreqDict[subList[2]])
        devVector_tag_copy[f].append(tagFreqDict[subList[2]])
        
        dev_tags[f].append(subList[2])


# In[63]:


trainVector_np = np.array(trainVector)
trainVector_tag_np = np.array(trainVector_tag)


# In[71]:


maxLen_dev = getMaxLength(devVector)

for lst in devVector:
    for i in range(len(lst),maxLen_dev):
        lst.append(0)


# In[72]:


maxLen_dev_tag = getMaxLength(devVector_tag)

for lst in devVector_tag:
    for i in range(len(lst),maxLen_dev_tag):
        lst.append(-1)


# In[74]:


devVector_np = np.array(devVector)
devVector_tag_np = np.array(devVector_tag)


# # Task 1

# In[ ]:


batch_size_1 = 32
batch_size_2 = 16


# In[77]:


X_train_torch = torch.from_numpy(trainVector_np)
y_train_torch = torch.from_numpy(trainVector_tag_np)


# In[79]:


X_eval_torch = torch.from_numpy(devVector_np)
y_eval_torch = torch.from_numpy(devVector_tag_np)


# In[81]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[155]:


#(Hever, 2020)
class BiLSTM_Model(nn.Module):
    def __init__(self, embed_dim, n_layers, hidden_dim, dropout_rate, out_dim):
        super(BiLSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(len(wordFreqDict), embed_dim)
        self.lstm = nn.LSTM(embed_dim, 
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True, 
                            batch_first=True
                           )
        self.lin = nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.elu = nn.ELU(alpha=2.0)
        self.classifier = nn.Linear(out_dim, 9)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, x, lengths):
        
        #x.shape = [batch size, sen length after pad]
        
        
        
        embed_val = self.embedding(x)    #embed_val.shape.shape = [batch size, 113, 100]
        
        
        
        out_pack = torch.nn.utils.rnn.pack_padded_sequence(embed_val, lengths, batch_first=True, enforce_sorted=False)
        
        
        
        out_lstm, _ = self.lstm(out_pack)

        
        out_pad,_ = torch.nn.utils.rnn.pad_packed_sequence(out_lstm, batch_first=True)

        out_drop = self.dropout(out_pad)
        

        
        out_elu = self.elu(self.lin(out_drop))
        out_sm = self.softmax(self.classifier(out_elu))

        output = torch.permute(out_sm, (0, 2, 1))
        
        return output


# In[87]:


embedding_dim = 100
num_lstm_layers = 1
lstm_hidden_dim = 256
lstm_dropout = 0.33
linear_output_dim = 128

learning_rate = 0.1
num_epochs = 50

#np.random.seed(0)
model = BiLSTM_Model(embedding_dim, num_lstm_layers, lstm_hidden_dim, lstm_dropout, linear_output_dim)


# In[94]:


criterion = nn.CrossEntropyLoss(weight=frequencies,ignore_index=-1)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov = True)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15,20,25], gamma=0.9) #(Pytorch.org, 2023)


# In[97]:


def flat(vals, lis, lgth):
    for i in range(len(lgth)):
        l = lgth[i]
        vals.extend(list(lis[i][0:l].cpu().numpy()))   #(Jason, 2013)
    return vals


# In[99]:


#(Prasad, 2020)
#(Pytorch.org, 2023)
from tqdm import tqdm



#pred_fin = []
#truth_fin = []

def train1():
    pred_gl = []
    truth_gl = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        #running_loss_dev = 0.0
        y_pred = []
        y_label = []

        pred_gl = []
        truth_gl = []
        for i in tqdm(range(0, 14987, batch_size_1)):
            inputs = X_train_torch[i:i+batch_size_1]
            labels = y_train_torch[i:i+batch_size_1]
            llen = train_len[i:i+batch_size_1]
            
            optimizer.zero_grad()
            
            output = model(inputs, llen)
            
            labels = labels[:, 0: output.shape[2]]
            
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            output = model(X_eval_torch, dev_len)

            predicted = torch.argmax(output, dim=1)

            y_pred = flat([], predicted, dev_len)
            y_label = flat([], y_eval_torch, dev_len)


            pred_gl.extend(y_pred)
            truth_gl.extend(y_label)
        lr_scheduler.step()
        
    return torch.save(model.state_dict(), 'blstm1.pt')


# In[100]:





# # TASK 2

# In[ ]:





# In[120]:


embedding_path = "hw4/glove.6B.100d"


# In[122]:


glv_model = {}

#(Theiler, 2019)
with open("glove.6B.100d",'r') as g_file:
      for emb in g_file:
        curr_emb = emb.split()

        curr_word = curr_emb[0]

        word_emb = curr_emb[1:]

        emb_np = np.array(word_emb, dtype=np.float64)

        glv_model[curr_word] = emb_np


# In[126]:


embedding_matrix = np.zeros((len(wordFreqDict),101))

for word, i in wordFreqDict.items():
    
    if word.lower() in glv_model.keys():
        if word[0].isupper():
            emb_val = glv_model[word.lower()]
            emb_val = np.append(emb_val, 1)
            embedding_matrix[i] = emb_val
        else:
            emb_val = glv_model[word.lower()]
            emb_val = np.append(emb_val, 0)
            embedding_matrix[i] = emb_val
    else:
        emb_val = glv_model['unk']
        emb_val = np.append(emb_val, 0)
        embedding_matrix[i] = emb_val


# In[128]:


class BiLSTM_Model_2(nn.Module):
    def __init__(self, embed_dim, n_layers, hidden_dim, dropout_rate, out_dim):
        super(BiLSTM_Model_2, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix.astype('float32')))
        
        self.lstm = nn.LSTM(embed_dim, 
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True, 
                            batch_first=True
                           )

        self.lin = nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.elu = nn.ELU(alpha=2.0)
        self.classifier = nn.Linear(out_dim, 9)
        self.softmax = nn.Softmax(dim=2)

    
    def forward(self, x, lengths):
        
        embed_val = self.embedding(x)
        
        out_pack = torch.nn.utils.rnn.pack_padded_sequence(embed_val, lengths, batch_first=True, enforce_sorted=False)
        
        out_lstm, _ = self.lstm(out_pack)
        
        out_pad, _ = torch.nn.utils.rnn.pad_packed_sequence(out_lstm, batch_first=True)
        
        out_drop = self.dropout(out_pad)
        
        out_elu = self.elu(self.lin(out_drop))
        
        out_sm = self.softmax(self.classifier(out_elu))

        output = torch.permute(out_sm, (0, 2, 1))
        
        return output


# In[129]:


embedding_dim_2 = 101
num_lstm_layers_2 = 1
lstm_hidden_dim_2 = 256
lstm_dropout_2 = 0.33
linear_output_dim_2 = 128

learning_rate_2 = 0.07
num_epochs_2 = 150

np.random.seed(0)
model_2 = BiLSTM_Model_2(embedding_dim_2, num_lstm_layers_2, lstm_hidden_dim_2, lstm_dropout_2, linear_output_dim_2)


# In[130]:


criterion_2 = nn.CrossEntropyLoss(weight=frequencies,ignore_index=-1)
optimizer_2 = optim.SGD(model_2.parameters(), lr=learning_rate_2, momentum=0.9, nesterov = True)


# In[132]:


from tqdm import tqdm

def train2():
    pred_gl_2 = []
    truth_gl_2 = []

    #pred_fin = []
    #truth_fin = []

    for epoch in range(num_epochs_2):
        running_loss_2 = 0.0
        #running_loss_dev = 0.0
        y_pred_2 = []
        y_label_2 = []

        pred_gl_2 = []
        truth_gl_2 = []
        for i in tqdm(range(0, 14987, batch_size_2)):
            inputs_2 = X_train_torch[i:i+batch_size_2]
            labels_2 = y_train_torch[i:i+batch_size_2]
            llen_2 = train_len[i:i+batch_size_2]
            
            optimizer_2.zero_grad()
            
            output_2 = model_2(inputs_2, llen_2)
            
            labels_2 = labels_2[:, 0: output_2.shape[2]]
            
            loss_2 = criterion_2(output_2, labels_2)

            loss_2.backward()
            optimizer_2.step()
            
            running_loss_2 += loss_2.item()
        
        
        model_2.eval()
        with torch.no_grad():
            output_dev_2 = model_2(X_eval_torch, dev_len)

            predicted_2 = torch.argmax(output_dev_2, dim=1)

            y_pred_2 = flat([], predicted_2, dev_len)
            y_label_2 = flat([], y_eval_torch, dev_len)

            #print(y_pred)

            pred_gl_2.extend(y_pred_2)
            truth_gl_2.extend(y_label_2)
            
            f1_2 = sklearn.metrics.f1_score(y_label_2, y_pred_2, average = 'macro')

        #lr_scheduler_2.step()
    return torch.save(model_2.state_dict(), 'blstm2.pt')


# In[ ]:




    


# # Test

# In[143]:


testData = open('data/test', 'r')
testData_lines = testData.readlines()


# In[144]:


wordList_test = []
for line in testData_lines:
    wordList_test.append(line.split())


# In[145]:


testVector = []
testVector_copy = []
testVector_words = []

test_len = []


# In[146]:


e = 0
for subList in wordList_test:
    if subList == []:
        e+=1
    elif subList[0] == '1':
        testVector.append([])
        
        testVector_copy.append([])
        if subList[1] not in wordFreqDict:
            testVector[e].append(wordFreqDict['unk'])
            testVector_copy[e].append(wordFreqDict['unk'])
        else:
            testVector[e].append(wordFreqDict[subList[1]])
            testVector_copy[e].append(wordFreqDict[subList[1]])
        
        testVector_words.append([])
        testVector_words[e].append(subList[1])
    else:
        if subList[1] not in wordFreqDict:
            testVector[e].append(wordFreqDict['unk'])
            testVector_copy[e].append(wordFreqDict['unk'])
        else:
            testVector[e].append(wordFreqDict[subList[1]])
            testVector_copy[e].append(wordFreqDict[subList[1]])
        
        testVector_words[e].append(subList[1])


# In[147]:


for sent in testVector_words:
    test_len.append(len(sent))


# In[148]:


maxLen_test = getMaxLength(testVector)

for lst in testVector:
    for i in range(len(lst),maxLen_test):
        lst.append(0)


# In[149]:


testVector_np = np.array(testVector)


# In[150]:


X_test_torch = torch.from_numpy(testVector_np)


# In[151]:


X_test_torch.shape


# In[156]:

def test1():
    model_1_tst = BiLSTM_Model(100, 1, 256, 0.33, 128)
    model_1_tst.load_state_dict(torch.load('blstm1.pt'))
    model_1_tst.eval()


    # In[157]:
    pred_gl = []
    truth_gl = []
    with torch.no_grad():
            output = model_1_tst(X_eval_torch, dev_len)

            predicted = torch.argmax(output, dim=1)

            y_pred = flat([], predicted, dev_len)
            y_label = flat([], y_eval_torch, dev_len)


            pred_gl.extend(y_pred)
            truth_gl.extend(y_label)


    y_test_1 = []
    test_gl_1 = []
    with torch.no_grad():
            output_test_1 = model_1_tst(X_test_torch, test_len)

            predicted_1_test = torch.argmax(output_test_1, dim=1)

            y_test_1 = flat([], predicted_1_test, test_len)

            test_gl_1.extend(y_test_1)


    # In[158]:

    task1_lines = []
    c = 0
    for i in range(len(devVector_words)):
        for j in range(len(devVector_words[i])):
            task1_lines.append("{} {} {} {}\n".format(j+1,devVector_words[i][j], dev_tags[i][j], freqTagDict[pred_gl[c]]))
            c+=1
        task1_lines.append("\n")
        
    del task1_lines[-1]
    with open('dev1.out', 'w') as dev1File:
        dev1File.writelines(task1_lines)

    lines = []
    c = 0
    for i in range(len(testVector_words)):
        for j in range(len(testVector_words[i])):
            lines.append("{} {} {}\n".format(j+1,testVector_words[i][j], freqTagDict[test_gl_1[c]]))
            c+=1
        lines.append("\n")
    del lines[-1]
    with open('test1.out', 'w') as test1File:
        test1File.writelines(lines)


# In[134]:

def test2():
    model_2_tst = BiLSTM_Model_2(101, 1, 256, 0.33, 128)
    model_2_tst.load_state_dict(torch.load('blstm2.pt'))
    model_2_tst.eval()


    # In[152]:
    pred_gl_2 = []
    truth_gl_2 = []
    with torch.no_grad():
            output_dev_2 = model_2_tst(X_eval_torch, dev_len)

            predicted_2 = torch.argmax(output_dev_2, dim=1)

            y_pred_2 = flat([], predicted_2, dev_len)
            y_label_2 = flat([], y_eval_torch, dev_len)

            #print(y_pred)

            pred_gl_2.extend(y_pred_2)
            truth_gl_2.extend(y_label_2)

    task2_lines = []
    c = 0
    for i in range(len(devVector_words)):
        for j in range(len(devVector_words[i])):
            task2_lines.append("{} {} {} {}\n".format(j+1,devVector_words[i][j], dev_tags[i][j], freqTagDict[pred_gl_2[c]]))
            c+=1
        task2_lines.append("\n")
        
    del task2_lines[-1]

    with open('dev2.out', 'w') as dev2File:
        dev2File.writelines(task2_lines)


    y_test_2 = []
    test_gl_2 = []
    with torch.no_grad():
            output_test_2 = model_2_tst(X_test_torch, test_len)

            predicted_2_test = torch.argmax(output_test_2, dim=1)

            y_test_2 = flat([], predicted_2_test, test_len)

            test_gl_2.extend(y_test_2)


    # In[154]:


    lines = []
    c = 0
    for i in range(len(testVector_words)):
        for j in range(len(testVector_words[i])):
            lines.append("{} {} {}\n".format(j+1,testVector_words[i][j], freqTagDict[test_gl_2[c]]))
            c+=1
        lines.append("\n")
    del lines[-1]
    with open('test2.out', 'w') as test2File:
        test2File.writelines(lines)


if __name__ == '__main__':
    test1()
    test2()




