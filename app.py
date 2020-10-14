from flask import Flask,render_template,request,url_for
import pickle
# DataFrame
import pandas as pd
from io import BytesIO
import base64
# Matplot
import matplotlib
import seaborn as sns
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer



# Utility
import re
import numpy as np
import os
import shutil
from collections import Counter
import logging
import time
import pickle
import itertools
IMAGE_FOLDER = os.path.join('static', 'img_pool')

stop_words = nltk.download('stopwords')
word_net = nltk.download('wordnet')


emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}
token = pickle.load(open(r'C:\Users\gagga\PycharmProjects\twitter sentimental analysis\vectoriser.pickle','rb'))



from keras.models import load_model
model1 = load_model(r"C:\Users\gagga\PycharmProjects\twitter sentimental analysis\model_new.h5")


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


def preprocess(data_text):
    processed_text = []

    word_lem = nltk.WordNetLemmatizer()

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
            if word not in nltk.corpus.stopwords.words('english'):
                if len(word) > 1:
                    word = word_lem.lemmatize(word)
                    tweet_words += (word + ' ')
        processed_text.append(tweet_words)

    return processed_text



import tweepy as tw
import pandas as pd
 
app=Flask(__name__)
consumer_key= 'R2bLxigwfn3mIHTPOL8j4n7BE'
consumer_secret= 'SHEi2b2hYgwVd3ch9Y3KuIYMfIQ9Y7U6lcO0y4DQR19fKU59Kv'
access_token= '1275859079546589186-hzxsKjWhUAR61Z8IqNpz3URm25BKuX'
access_token_secret= 'xRFUpmVXNndzPSbGExjci0XsLyYLHUxerMe7xtX92Whn0'
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

#public_tweets = api.home_timeline()
#for tweet in public_tweets:
 #   print(tweet.text)











# Post a tweet from Python
#api.update_status("Look, I'm tweeting from #Python in my twitter account")
# Your tweet has been posted!
# Define the search term and the date_since date as variables
#search_words = "#deepikapadukone"
#date_since = "2020-01-16"

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER


x=0



@app.route('/')
def hello():
    return render_template("home.html")

@app.route('/predict', methods=['POST','GET'])
def submit_data():
    if request.method=='POST':
        print(request.form)
        if (request.form['analyse']=='searchbywords'):
            return render_template("twitts.html")
        return render_template("handler.html")
    return render_template("home.html")
@app.route('/twit1', methods=['POST','GET'])
def search_data():

    if request.method=='POST':
        search_words=request.form['fname']
        date_since=request.form['dat']

        tweets = tw.Cursor(api.search,
                           q=search_words,
                           lang="en",since=date_since).items(50)


        # Iterate and print tweets
        txt=[]
        for tweet in tweets:
            txt.append(tweet.text)
        text = preprocess(txt)

        
        

        seq = token.texts_to_sequences(text)
        padded = pad_sequences(seq, padding='pre', maxlen=40)

        pred = model1.predict(padded)
        labels = ['negative', 'neutral', 'positive']

        ans = []
        for i in pred:
            ans.append(labels[np.argmax(i)])
        a = majority_element(ans)
        n=pd.Series(ans)
        sns.set(font_scale=1.4)
        n.value_counts().plot(kind='bar',color=['firebrick', 'green', 'blue'] ,figsize=(7, 6), rot=0)
    
        plt.title("count of tweets having negative,positive and neutral sentiment", y=1.02)
    ### Generating The Plot
    #plt.plot(X,Y)
    ### Saving plot to disk in png format
        plt.savefig('square_plot.png')


    ### Rendering Plot in Html
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue()).decode('ascii')
    
        result = figdata_png
        x=1
        if a == 'positive':
            sentiment = 'positive'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Smiling_Emoji.png')
        elif a=='negative':
            sentiment = 'negative'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Sad_Emoji.png')
        else:
            sentiment='neutral'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'neutral_face.png')
        data_pos = text[:]
        
        wc = WordCloud(max_words=400, background_color='black', width=500, height=300,max_font_size=110,
                       collocations=False).generate(" ".join(data_pos))
        plt.figure(figsize=(20, 20))
        plt.axis('off')
        
        plt.imshow(wc,interpolation="bilinear")
        plt.savefig('square1.png')

    ### Rendering Plot in Html
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue()).decode('ascii')
    
        result2 = figdata_png
        #plt.savefig('\static\img_pool\Wt.png')
        #img = os.path.join(app.config['UPLOAD_FOLDER'], 'so.jpg')
        #fy = 'static\img_pool\Wt.jpg'
    
        return render_template('output2.html', sentiment=sentiment,name=x,result2=result2,result=result )
 
@app.route('/twit2', methods=['POST','GET'])
def search():

    if request.method=='POST':
        screen=request.form['txt']
        tmp = []

        for status in tw.Cursor(api.user_timeline, screen_name=screen, tweet_mode="extended").items(50):
            status.full_text
            tmp.append(status.full_text)
        texting = preprocess(tmp)
        data_pos = texting[:]
        # img = Image.open("static\img_pool\WC.jpg")
        #fx = os.path.join(app.config['UPLOAD_FOLDER'], 'Word.jpg')
        seq = token.texts_to_sequences(texting)
        padded = pad_sequences(seq, padding='pre', maxlen=40)

        pred = model1.predict(padded)
        labels = ['negative', 'neutral', 'positive']
        x=1
        ans = []
        for i in pred:
            ans.append(labels[np.argmax(i)])
        a = majority_element(ans)
        x=1
        if a == 'positive':
            sentiment = 'positive'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Smiling_Emoji.png')
        elif a=='negative':
            sentiment = 'negative'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Sad_Emoji.png')
        else:
            sentiment='neutral'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'neutral_face.png')
        print(ans)
        n=pd.Series(ans)
        sns.set(font_scale=1.4)
        n.value_counts().plot(kind='bar',color=['firebrick', 'green', 'blue'] ,figsize=(7, 6), rot=0)
    
        plt.title("count of tweets having negative,positive and neutral sentiment", y=1.02)
    ### Generating The Plot
    #plt.plot(X,Y)
    ### Saving plot to disk in png format
        plt.savefig('square.png')


        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue()).decode('ascii')
    
        resultx = figdata_png
        
        wc = WordCloud(max_words=400, background_color='black', width=500, height=300,max_font_size=110,
                       collocations=False).generate(" ".join(data_pos))
        plt.figure(figsize=(20, 20))
        plt.axis('off')
        
        plt.imshow(wc,interpolation="bilinear")
        plt.savefig('square1.png')

    ### Rendering Plot in Html
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue()).decode('ascii')
    
        im = figdata_png

        #plt.savefig('\static\img_pool\Wt.png')
        #img = os.path.join(app.config['UPLOAD_FOLDER'], 'so.jpg')
        #fy = 'static\img_pool\Wt.jpg'
        
        return render_template('output.html',sentiment=sentiment,result2=im,name=x, resultx=resultx)


if __name__=='__main__':
    app.run(debug=True)