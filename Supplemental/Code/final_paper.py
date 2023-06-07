#!/usr/bin/env python
# coding: utf-8

# In[151]:


import snscrape.modules.twitter as sntwitter
import pandas as pd

location = '55.4847, 28.7761, 1000mi'

# Creating list to append tweet data 
tweets = []

# Using TwitterSearchScraper to scrape data
for i, tweet in enumerate(sntwitter.TwitterSearchScraper('#Covid OR #Covid19 OR #Vaccination OR #Pfizer OR #Moderna OR #BioNTech OR #Europe  since:2020-12-01 until:2021-12-31  geocode:"{}"  lang:"en"'.format(location)).get_items()): 
    if i>15550: #number of tweets you want to scrape
        break
    tweets.append([tweet.date,tweet.id,tweet.user.username,tweet.user.location, tweet.content, tweet.likeCount, tweet.retweetCount])
df = pd.DataFrame(tweets, columns = ['Date', 'ID','username','location', 'tweet', 'num_of_likes', 'num_of_retweet'])


# In[152]:


df.to_csv('covid_data.csv', sep=',', index=False)


# In[153]:


df


# In[154]:


df.isnull().sum()


# In[155]:


df.dtypes


# In[156]:


import string
df['tweet'] = df['tweet'].str.lower()
df['tweet']

punctuation_removal = string.punctuation
def remove_punctuation(text):
    
    return text.translate(str.maketrans('', '', punctuation_removal))
df["tweet"] = df["tweet"].apply(lambda text: remove_punctuation(text))
df["tweet"]


# In[157]:


'''We could see lot of emojies and flags in the above tweets, so lets try removing them '''


# In[158]:


# Emoji Removal

import re
def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)
df["tweet"] = df["tweet"].apply(remove_emoji)
df["tweet"]


# In[159]:


df["tweet"] = df["tweet"].apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))
df["tweet"] = df["tweet"].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
df["tweet"]


# In[160]:


from collections import Counter
cnt = Counter()
for text in df["tweet"].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(10)


# In[161]:


'''As we can see from above results that the most common words are stop words, so we will have to remove them'''


# In[162]:


from nltk.corpus import stopwords
", ".join(stopwords.words('english'))
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

df["tweet"] = df["tweet"].apply(lambda text: remove_stopwords(text))
df["tweet"]


# In[163]:


from collections import Counter
cnt = Counter()
for text in df["tweet"].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(5)


# In[164]:


import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sid = SIA()
df['sentiments'] = df["tweet"].apply(lambda x: sid.polarity_scores(' '.join(re.findall(r'\w+',str(x).lower()))))
df['Positive_Sentiment'] = df['sentiments'].apply(lambda x: x['pos']+1*(10**-3)) 
df['Neutral_Sentiment'] = df['sentiments'].apply(lambda x: x['neu']+1*(10**-3))
df['Negative Sentiment'] = df['sentiments'].apply(lambda x: x['neg']+1*(10**-3))
df.head()


# In[165]:


pip install wordcloud


# In[166]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS, ImageColorGenerator
tweet_All = " ".join(review for review in df["tweet"])

fig, ax = plt.subplots(1, 1, figsize  = (10,10))
# Create and generate a word cloud image:
wordcloud_ALL = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_All)

# Display the generated image:
ax.imshow(wordcloud_ALL, interpolation='bilinear')

ax.axis('off')


# In[ ]:





# In[167]:


# Sentiment Distribution
import seaborn as sns
sns.set_style('darkgrid')
plt.subplot(2,1,1)
plt.title('Distriubtion Of Sentiments Across Tweets',fontsize=19,fontweight='bold')
sns.kdeplot(df['Negative Sentiment'], color = 'blue')
sns.kdeplot(df['Positive_Sentiment'], color = 'orange')
sns.kdeplot(df['Neutral_Sentiment'], color = 'green')
plt.xlabel(' ')
plt.legend(['Negative Sentiment','Positive Sentiment','Neutral Sentiment'])
plt.subplot(2,1,2)
plt.title('Average Sentiments Across Tweets',fontsize=19,fontweight='bold')

print()
neg_total_avg = (df['Negative Sentiment'].sum())/len(df.index)
print(neg_total_avg)
pos_total_avg = (df['Positive_Sentiment'].sum())/len(df.index)
print(pos_total_avg)
neu_total_avg = (df['Neutral_Sentiment'].sum())/len(df.index)
print(neu_total_avg)
sentiment_type = ['Negative','Positive','Neutral']
sentiment_total_avg = [neg_total_avg, pos_total_avg, neu_total_avg]
plt.bar(sentiment_type, sentiment_total_avg, color = ['blue', 'orange', 'green'])
plt.ylabel('Average Sentiment Per Tweet',fontsize=19)
plt.xlabel('Sentiment Type',fontsize=19)
plt.show()
plt.figure(figsize=(4,4)) 


# In[ ]:


#Ref: https://www.kaggle.com/code/hassanhshah/covid-vaccine-sentiment-and-time-series-analysis


# In[170]:


import string
import re
import textblob
from textblob import TextBlob

from nltk.tokenize import word_tokenize 

def ProcessedTweets(text):
    #changing tweet text to small letters
    text = text.lower()
    # Removing @ and links 
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text).split())
    # removing repeating characters
    text = re.sub(r'\@\w+|\#\w+|\d+', '', text)
    # removing punctuation and numbers
    punct = str.maketrans('', '', string.punctuation+string.digits)
    text = text.translate(punct)
    # tokenizing words and removing stop words from the tweet text  
    return text


# In[171]:


df['Processed_Tweets'] = df['tweet'].apply(ProcessedTweets)


# In[172]:


def polarity(tweet):
    return TextBlob(tweet).sentiment.polarity

# Function to get sentiment type
#setting the conditions
def sentimenttextblob(polarity):
    if polarity < 0:
        return "Negative"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Positive"  


# In[173]:


df['Polarity'] = df['Processed_Tweets'].apply(polarity)
df['Sentiment'] = df['Polarity'].apply(sentimenttextblob)
sent = df['Sentiment'].value_counts()
sent


# In[174]:


df_pro=df[['Processed_Tweets']]
df_pro['Polarity']=df['Polarity']
df_pro['Sentiment']=df['Sentiment']
df_pro


# In[175]:


import matplotlib.pyplot as plt
plt.subplot(1,2,1)
sent.plot(kind='bar', color=['green'], figsize=(15,5))
plt.title('Sentiment percieved ', fontsize=16)
plt.xlabel('Types of sentiment')
plt.ylabel('Number of sentiment')


# In[176]:



import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(75, 75))
sns.barplot(
    x="Polarity",
    y="location",
    data=df,
    estimator=sum,
    ci=None,
    palette="dark:salmon_r"
)


# In[177]:


df


# In[178]:


processed_features=df['tweet']


# In[179]:


from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()


# In[180]:


labels = df.iloc[:, 13].values


# In[181]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)


# In[182]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


# In[183]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def ML_modeling(models, params, X_train, X_test, y_train, y_test):    
    
    if not set(models.keys()).issubset(set(params.keys())):
        raise ValueError('Some estimators are missing parameters')

    for key in models.keys():
    
        model = models[key]
        param = params[key]
        gs = GridSearchCV(model, param, cv=5, error_score=0, refit=True)
        gs.fit(X_train, y_train)
        y_pred = gs.predict(X_test)
        
        # Print scores for the classifier
        print(key, ':', gs.best_params_)
        print("Accuracy: %1.3f \tPrecision: %1.3f \tRecall: %1.3f \t\tF1: %1.3f\n" % (accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='macro'), recall_score(y_test, y_pred, average='macro'), f1_score(y_test, y_pred, average='macro')))
    
    return


# In[184]:


models = {
    'Random Forest Classifier': RandomForestClassifier(),
    'Naive Bayes': MultinomialNB(),
    'logistic regression' : LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),  
    'Gradient Boosting': GradientBoostingClassifier()
}

params = {
    'Random Forest Classifier': {'criterion': ['gini', 'entropy']},
    'Naive Bayes': { 'alpha': [0.5, 1], 'fit_prior': [True, False] }, 
    'logistic regression' : {'max_iter':[2000]},
    'Decision Tree': { 'min_samples_split': [2, 5, 7] }, 
    'Gradient Boosting': { 'learning_rate': [0.05, 0.1] }
}


# In[185]:


get_ipython().run_cell_magic('time', '', 'print("==============Bag of Words==============\\n")\nML_modeling(models, params, X_train, X_test, y_train, y_test)')


# In[187]:


from sklearn.ensemble import RandomForestClassifier

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)


# In[188]:


predictions = text_classifier.predict(X_test)


# In[189]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))


# In[ ]:





# In[190]:


df


# In[191]:


df['Polarity']


# In[192]:


positive_tweet =  df[df['Sentiment'] == 'Positive']['tweet']
negative_tweet =  df[df['Sentiment'] == 'Negative']['tweet']
neutral_tweet =  df[df['Sentiment'] == 'Neutral']['tweet']


# In[193]:


positive_tweet


# In[ ]:





# In[194]:


from wordcloud import WordCloud
# Function for creating WordClouds
def cloud_of_Words(tweet_cat,title):
    forcloud = ' '.join([tweet for tweet in tweet_cat])
    wordcloud = WordCloud(width =500,height = 300,random_state =5,max_font_size=110).generate(forcloud)
    plt.imshow(wordcloud, interpolation ='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()
    plt.figure(figsize = (10,8))


# In[195]:


plt.figure(figsize = (10,8))
cloud_of_Words(positive_tweet, 'Positive')
cloud_of_Words(negative_tweet, 'Negative')
cloud_of_Words(neutral_tweet, 'Neutral')


# In[200]:


import datetime
data=df.copy()
data['date'] = pd.to_datetime(df['Date']).dt.date
data['year'] = pd.DatetimeIndex(df['Date']).year


# In[201]:


data


# In[216]:


b_date_count = data.groupby(by='date').count().reset_index()


# In[217]:


b_date_count


# In[232]:


import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.dates

from datetime import datetime

x_values = df['Date']
y_values = df['Positive_Sentiment']
plt.plot(x_values,y_values)
plt.title('Positive Sentiment')
plt.xlabel('Dates')
plt.ylabel('sentiment')

plt.show()


# In[224]:


import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.dates

from datetime import datetime

x_values = df['Date']
y_values = df['Negative Sentiment']

plt.plot(x_values,y_values)
# beautify the x-labels
plt.title('Negative Sentiment')
plt.xlabel('Dates')
plt.ylabel('sentiment')



plt.show()


# In[233]:


import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.dates

from datetime import datetime

x_values = df['Date']
y_values = df['Neutral_Sentiment']

plt.plot(x_values,y_values)
# beautify the x-labels
plt.title('Neutral_Sentiment')
plt.xlabel('Dates')
plt.ylabel('sentiment')


plt.show()


# In[ ]:




