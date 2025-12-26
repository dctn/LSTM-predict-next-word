#!/usr/bin/env python
# coding: utf-8

# In[4]:


from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.datasets import imdb
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[5]:


(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)


# In[6]:


X_train.shape,y_train.shape


# In[7]:


sample_review = X_train[0]
sample_label = y_train[0]
sample_label


# In[8]:


data_index = imdb.get_word_index()
reversed_data_index = {value:key for key,value in data_index.items() }
reversed_data_index.get(150)


# In[9]:


" ".join([reversed_data_index.get(i-3,"?") for i in sample_review])


# In[15]:


max_len = 100
X_train= pad_sequences(X_train,maxlen=max_len)
X_test= pad_sequences(X_test,maxlen=max_len)
X_train[0]


# ## creating model

# In[21]:


max_features = 20000
model = Sequential()
model.add(Embedding(max_features,128,input_length=max_len))
model.add(SimpleRNN(128,activation="relu"))
model.add(Dense(1,activation="sigmoid"))


# In[24]:


model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


# In[25]:


model.summary()


# In[26]:


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor="val_loss", patience=5,restore_best_weights=True)


# In[28]:


model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_test,y_test),
    callbacks=[early_stopping],
    validation_split=0.2,
)


# In[30]:


model.save("own_model.keras")


# ## predictions

# In[46]:


def pre_process(review):
    review_words = review.lower().split()
    process_review = [data_index.get(word,2) for word in review_words]
    pad_seq = pad_sequences([process_review],maxlen=max_len)
    return pad_seq


# In[51]:


def predict_review(review):
    pre_processed = pre_process(review)
    prediction = model.predict(pre_processed)
    result = "Postive" if prediction >= 0.5 else "Negative"
    return result,prediction[0][0]


# In[52]:


prediction_review = "this is super movie and i like very much"
predict_review(prediction_review)


# ### recreating with Pytorch

# In[29]:


import torch.nn as nn
import torch


# In[22]:


class RNN(nn.Module):
    def __init__(self,embedding_size,hidden_state_size,max_features):
        super(RNN, self).__init__()
        self.embedding_layer = nn.Embedding(max_features, embedding_size)
        self.rnn = nn.RNN(input_size=embedding_size,hidden_size=hidden_state_size,
                          batch_first=True,nonlinearity="relu")
        self.fc = nn.Linear(hidden_state_size,1)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self,x):
        x = self.embedding_layer(x)
        print(x)
        output, hidden = self.rnn(x)
        print(output,hidden)
        h_n = hidden.squeeze(0)
        print(h_n)
        h_n = self.fc(h_n)
        print(h_n)
        output = self.sigmoid_layer(h_n)
        print(output)
        return output


# In[23]:


torch_rnn = RNN(128,128,max_features=20000)


# In[39]:


from torchinfo import summary
summary(
    torch_rnn,
    input_size=(2, 100),   # batch_size=2, seq_len=100
    dtypes=[torch.long]    # REQUIRED for Embedding
)

