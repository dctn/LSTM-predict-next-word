#!/usr/bin/env python
# coding: utf-8

# ## predict the next word

# In[100]:


import nltk

nltk.download('gutenberg')
from nltk.corpus import gutenberg

data = gutenberg.raw('shakespeare-hamlet.txt')


# In[101]:


with open("shakespeare.txt", "w") as f:
    f.write(data)


# ## data precessing

# In[102]:


import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf


# In[103]:


with open("shakespeare.txt", 'r') as f:
    data = f.read().lower()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])


# In[104]:


tokenizer.word_index, tokenizer.index_word


# In[105]:


total_words = len(tokenizer.word_index) + 1


# In[106]:


input_sequences = []
for line in data.split('\n'):
    token_line = tokenizer.texts_to_sequences([line])[0]
    for i in range(len(token_line)):
        input_tokenized_line = token_line[:i + 1]
        input_sequences.append(input_tokenized_line)
input_sequences


# ## pad sequences

# In[107]:


max_sequence_size = max(len(i) for i in input_sequences)
max_sequence_size


# In[108]:


padded_input_seq = pad_sequences(input_sequences, maxlen=max_sequence_size, padding='pre')


# In[109]:


X = padded_input_seq[:, :-1]
y = padded_input_seq[:, -1]
X.shape, y.shape


# In[110]:


y


# In[111]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ## create LSTM model

# In[112]:


import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


# In[152]:


class LSTM(nn.Module):
    def __init__(self, total_words_size, embed_dim, hidden_size, lstm_num_layers):
        super(LSTM, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=total_words_size, embedding_dim=embed_dim,)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=lstm_num_layers, batch_first=True,dropout=0.3)
        self.fc = nn.Linear(in_features=hidden_size, out_features=total_words_size)

    def forward(self, x):
        embed = self.embedding_layer(x)
        lstm_output, (h_0, c_0) = self.lstm(embed)
        output = self.fc(h_0[-1])
        return output


lstm_1 = LSTM(total_words_size=total_words, embed_dim=64, hidden_size=64, lstm_num_layers=1).to(device)


# In[114]:


loss_func = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(lstm_1.parameters(),lr=0.003)


# In[115]:


from torch.utils.data import DataLoader, TensorDataset

X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# In[116]:


len(train_dataloader), len(test_dataloader)


# In[132]:


result = {
    'loss': [],
    'accuracy': [],
    "test_loss": [],
    "test_accuracy": []
}


# In[133]:


def accuracy_func(y_pred, y_true):
    correct = torch.eq(y_pred, y_true).sum().item()
    accuracy = correct / len(y_pred)
    return accuracy


# In[153]:


from tqdm.auto import tqdm

EPOCHS = 20

# training model
for epoch in tqdm(range(EPOCHS)):
    lstm_1.train()
    loss = 0
    accuracy = 0
    for X, y in train_dataloader:
        y_logits = lstm_1(X.to(device))

        batch_loss = loss_func(y_logits, y.to(device))

        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        # calculate accuracy and loss
        y_prob = torch.softmax(y_logits, dim=1)
        y_pred = torch.argmax(y_prob, dim=1)
        batch_accuracy = accuracy_func(y_pred=y_pred, y_true=y.to(device))

        loss += batch_loss.item()
        accuracy += batch_accuracy

    # testing model
    with torch.inference_mode():
        lstm_1.eval()
        test_loss = 0
        test_accuracy = 0
        for X, y in test_dataloader:
            y_logits = lstm_1(X.to(device))
            batch_loss = loss_func(y_logits, y.to(device))
            test_loss += batch_loss.item()

            y_prob = torch.softmax(y_logits, dim=1)
            y_pred = torch.argmax(y_prob, dim=1)
            test_accuracy += accuracy_func(y_pred=y_pred, y_true=y.to(device))

    loss = loss / len(train_dataloader)
    accuracy = accuracy / len(train_dataloader)
    test_loss = test_loss / len(test_dataloader)
    test_accuracy = test_accuracy / len(test_dataloader)

    result['loss'].append(loss)
    result['accuracy'].append(accuracy)
    result['test_loss'].append(test_loss)
    result['test_accuracy'].append(test_accuracy)

    print(f"Epoch {epoch + 1}/{EPOCHS} loss: {loss} accuracy: {accuracy} test_loss: {test_loss} test_accuracy: {test_accuracy}")


# In[154]:


import matplotlib.pyplot as plt

plt.plot(result['loss'], label='train')
plt.plot(result['test_loss'], label='test')
plt.legend()


# In[158]:


import math
perplexity = math.exp(result['test_loss'][30])
perplexity

