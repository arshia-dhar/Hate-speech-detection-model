import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


tweet_df = pd.read_csv('hateDetection_train.csv') #replace with filename of your dataset


def data_processing(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"https\S+|www\S+http\S+", '', tweet, flags = re.MULTILINE)
    tweet = re.sub(r'\@w+|\#','', tweet)
    tweet = re.sub(r'[^\w\s]','',tweet)
    tweet = re.sub(r'ð','',tweet)
    tweet_tokens = word_tokenize(tweet)
    filtered_tweets = [w for w in tweet_tokens if not w in stop_words]
    return " ".join(filtered_tweets)

tweet_df['tweet'] = tweet_df['tweet'].apply(data_processing)
tweet_df = tweet_df.drop_duplicates('tweet')


lemmatizer = WordNetLemmatizer()
def lemmatizing(data):
    tweet = [lemmatizer.lemmatize(word) for word in data.split()]
    return " ".join(tweet)

tweet_df['tweet'] = tweet_df['tweet'].apply(lemmatizing)


fig = plt.figure(figsize=(7,7))
colors = ("red", "gold")
wp = {'linewidth':2, 'edgecolor':"black"}
tags = tweet_df['label'].value_counts()
explode = (0.1, 0.1)
tags.plot(kind='pie',
          autopct='%1.1f%%',
          shadow=True,
          colors=colors,
          startangle=90,
          wedgeprops=wp,
          explode=explode,
          label='')
plt.title('Distribution of sentiments')
plt.show()


vect = TfidfVectorizer(ngram_range=(1,3)).fit(tweet_df['tweet'])
X = vect.transform(tweet_df['tweet'])
Y = tweet_df['label']

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model with hyperparameter tuning
param_grid = {'C':[100, 10, 1.0, 0.1, 0.01], 'solver':['newton-cg', 'lbfgs','liblinear']}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(x_train, y_train)

# Predict and evaluate
y_pred = grid.predict(x_test)
logreg_acc = accuracy_score(y_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"https\S+|www\S+http\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@w+|\#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'ð', '', tweet)
    tweet_tokens = word_tokenize(tweet)
    filtered_tweet = [w for w in tweet_tokens if not w in stop_words]
    tweet = " ".join(filtered_tweet)
    tweet = " ".join([lemmatizer.lemmatize(word) for word in tweet.split()])
    return tweet

def predict_tweet(tweet):
    processed = preprocess_tweet(tweet)
    vect_text = vect.transform([processed])
    prediction = grid.predict(vect_text)[0]
    return "❌ Hate Speech" if prediction == 1 else "✅ Not Hate Speech"

# Input loop
while True:
    tweet_input = input("\nEnter a tweet (or type 'exit' to quit): ")
    if tweet_input.lower() == 'exit':
        break
    print("Prediction:", predict_tweet(tweet_input))
