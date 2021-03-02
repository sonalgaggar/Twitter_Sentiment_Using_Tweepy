# DataFrame
import pandas as pd

# Matplot
import matplotlib.pyplot as plt


# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools





stop_words={"you'd", "wouldn't", 'o', 'y', 'just', 'by', 'yours', 'have', "mustn't", 'out', 'aren', "it's", "you're", 'of', 'here', 'them', 'did', 'ours', 'when', 'has', 'and', 'under', 'wasn', 'which', 'while', 'as', 'is', "aren't", 'ma', 'herself', 'about', "needn't", 'mustn', "weren't", 're', 'against', 'its', 'who', 'during', 'whom', 'this', "you've", 'yourselves', 'shouldn', 'd', 's', 'mightn', 'from', 'does', 'themselves', 'being', "shouldn't", 'you', "haven't", 'before', 'above', "she's", 'once', 'for', "isn't", 'each', 'can', 'after', 'down', 'be', 'all', 'too', 'a', 'having', 'hasn', 've', "hasn't", 'between', 'am', 'doing', 'in', 't', 'or', 'had', 'won', 'only', 'such', 'so', 'yourself', 'any', 'isn', "hadn't", 'wouldn', 'very', 'both', "you'll", 'it', 'below', 'were', 'i', 'himself', 'don', 'over', 'should', 'now', "doesn't", 'the', 'further', "that'll", "should've", 'haven', 'needn', 'me', 'ain', 'with', 'those', 'because', 'didn', 'been', "don't", 'again', 'nor', 'off', 'itself', 'why', 'll', 'where', 'not', 'your', 'was', 'at', 'will', "won't", 'their', 'we', 'what', 'that', 'into', 'how', 'him', 'to', 'are', "couldn't", "shan't", 'up', 'these', 'my', "wasn't", 'weren', 'own', 'couldn', 'if', 'doesn', 'shan', 'do', 'on', 'ourselves', 'than', 'myself', "didn't", 'hadn', 'theirs', 'there', 'until', 'more', 'they', 'our', 
'his', 'an', 'he', 'same', 'hers', 'but', 'few', 'm', 'no', 'then', 'through', 'some', 'most', 'other', "mightn't", 'her', 'she'}



df=pd.read_csv('train.csv')



df=df.dropna()



#stop_words = nltk.download('stopwords')
#word_net = nltk.download('wordnet')
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

def preprocess(data_text):
    processed_text = []
    
    
    
    url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    user_pattern = '@[^\s]+'
    alpha_pattern = "[^a-zA-Z0-9]"
    sequence_pattern = r"(.)\1\1+"
    seq_replace_pattern = r"\1\1"
    
    for tweet in data_text:
        tweet = tweet.lower()
        
        tweet = re.sub(url_pattern, ' ', tweet)
        
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
            
        tweet = re.sub(user_pattern, " ", tweet)
        
        tweet = re.sub(alpha_pattern, " ", tweet)

        tweet = re.sub(sequence_pattern, seq_replace_pattern, tweet)

        tweet_words = ''

        for word in tweet.split():
            if word not in stop_words:
                tweet_words += (word + ' ')
                
                    
        processed_text.append(tweet_words)
      
    return processed_text





text, sentiment = list(df['text']), list(df['sentiment'])





from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder





encoder = LabelEncoder()
encoder.fit(sentiment)
encoded_Y = encoder.transform(sentiment)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


sentiments=dummy_y.tolist()




processed_text = preprocess(text)


from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense



voc_size=10000




onehot_repr=[one_hot(words,voc_size)for words in processed_text] 


tokenizer=Tokenizer(voc_size,split=" ")
tokenizer.fit_on_texts(processed_text)

embedded_docs=tokenizer.texts_to_sequences(processed_text)


x=['interview leave alone ']
e=tokenizer.texts_to_sequences(x)


x=['interview leave alone ']
val=[one_hot(words,voc_size)for words in x] 

sent_length=50
embedded_docs=pad_sequences(embedded_docs,padding='pre',maxlen=sent_length)
print(embedded_docs)


sent_length=50
embedded_docs1=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs1)

## Creating model
embedding_vector_features=50
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(64,dropout=0.7,recurrent_dropout=0.7))
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(sentiments)

import numpy as np
X_final1=np.array(embedded_docs1)
y_final1=np.array(sentiments)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)



from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_final1, y_final1, test_size=0.33, random_state=42)


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=30,batch_size=64)



score,accr = model.evaluate(X_test,y_test)
print(accr)


y_pred=model.predict_classes(X_test)

accr = model.evaluate(X_test,y_test)



txt = ["hey awesome dau for me"]
seq = preprocess(txt)

seq = tokenizer.texts_to_sequences(txt)



padded=pad_sequences(seq,padding='pre',maxlen=sent_length)



pred = model.predict(padded)
labels = ['negative', 'neutral', 'positive']
print(pred, labels[np.argmax(pred)])



from keras.models import load_model

 # creates a H # deletes the existing model


model.save(r"model_new.h5")
print("Saved model to disk")


model1 = load_model(r"model_new.h5")
print("Model Loaded")


import pickle
file = open(r'vectoriser.pickle','wb')
pickle.dump(tokenizer, file)
file.close(

file = open(r'vectoriser.pickle','rb')
token = pickle.load(file)



# In[39]:


txt = ['bad day','RT @Weekendmacha: Cute @deepikapadukone 💚😘👸 \n#deepikapadukone #deepika \n#shriyasaran #shriyasaran360 \n#weekendmachameme #weekendmacha #week…', '#fahadhfaasil loves #piku ..#Deepika #movies #bollywood #BestOfTheWeek https://t.co/f4483dr00A','you are good']

seq = preprocess(txt)

seq = token.texts_to_sequences(txt)



padded=pad_sequences(seq,padding='pre',maxlen=sent_length)



pred = model1.predict(padded)
labels = ['negative', 'neutral', 'positive']

ans=[]
for i in pred:
    ans.append(labels[np.argmax(i)])
print(pred, labels[np.argmax(pred[1])])


# In[40]:


ans


# In[41]:


def majority_element(num_list):
        idx, ctr = 0, 1
        
        for i in range(1, len(num_list)):
            if num_list[idx] == num_list[i]:
                ctr += 1
            else:
                ctr -= 1
                if ctr == 0:
                    idx = i
                    ctr = 1
        
        return num_list[idx]


# In[42]:


a=majority_element(ans)


# In[69]:


type(a)


# In[70]:


s = preprocess(txt)
from wordcloud import WordCloud
data_pos = processed_text[:]
wc = WordCloud(max_words = 300000,background_color ='white', width = 1920 , height = 1080,
              collocations=False).generate(" ".join(data_pos))
plt.figure(figsize = (40,40))
plt.imshow(wc)


# In[ ]:




