#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np


# In[2]:


os.chdir("E:/Data Science/Projects/Fake News Classifier/fake-news-dataset")


# In[3]:


df = pd.read_csv("train.csv")


# In[4]:


df.head()


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


# In[6]:


df.shape


# In[7]:


# Dropping the missing values #

df = df.dropna()
df.shape


# In[9]:


df.head(10)

# By dropoing the missing valies, we will be getting missing indexes, so will trt to reset the index #


# In[8]:


messages = df.copy()


# In[9]:


messages.reset_index(inplace=True)
messages.head(10)


# In[10]:


# dropping the inedex column #

messages.drop(['index'],axis=1,inplace=True)
messages.head()


# In[18]:


a = 'MN nM'
a.lower()


# In[11]:


# Applying the Pre processing steps #

# We are just taking title column for messages based on whoch we will classify the news #

# removing regular expression, converting into lower cases, stemming the wordd to root form, spliting thwe words from sentences

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()
corpus =[]

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['title'][i])  # subsitute all other expressions except a-z and A-Z and replace with ' '#
    review = review.lower()  # converting inot lower cases #
    review = review.split()  # splitting each and evry sentence into multiple words 
    a = [ps.stem(j) for j in review if j not in stopwords.words('english')]  #dropping  & stemming the words
    m = ' '.join(a)  # putting a space across each words
    corpus.append(m) 


# In[19]:


m = pd.DataFrame(corpus)
m.shape
m.head(10)


# In[20]:


# Applying the bag of words to this corpus, to generate the count of words across all sentences #

cv = CountVectorizer(max_features=5000,ngram_range=(1,3)) # (1,3) We are takking combination of 1 word, 2 words & 3 words togather
X = cv.fit_transform(corpus).toarray() # we are taking top 5000 words with highest frequency #


# In[21]:


X.shape # there are 18,285 sentences and 5000 features #


# In[23]:


# Independent variable # - Title of the news
# Dependent variable # - Labels
Y = messages['label']
Y.head()

# 1:Unreliable ( Fake) , 0 : Reliable news #


# In[24]:


# Doing the train and test splot for this data set #
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=25)


# In[25]:


y_test.shape


# In[26]:


cv.get_feature_names()[:20]  # Top 20 feature names for this data set #, which shows 2 words and 3 words togather #


# In[27]:


cv.get_params()  # will give details for the count vectorizer applied # 


# In[28]:


# Data set after applying tyhe count vectporizer #

df_count = pd.DataFrame(X,columns=cv.get_feature_names())
df_count.head()


# In[29]:


# Applying the Multionomial NB algorithm #

from sklearn.naive_bayes import  MultinomialNB

mn = MultinomialNB()

model = mn.fit(x_train,y_train)


# In[30]:


# Predicting the out performance of the model #

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix

acc = accuracy_score(y_pred,y_test)
acc
# Accuracy Score is 90%


# In[31]:


con = confusion_matrix(y_pred,y_test)
con


# In[32]:


y_test.head(10)


# In[34]:


messages['title'][17538]  # this shows 0 : reliable / True news , while 1: unreliable / fake news


# In[35]:


messages['title'][2179]  # Fake news# as there is no souce mentioned in this news 


# In[66]:


m = pd.DataFrame(y_pred)
m.head(10)


# In[41]:


# Hyper parameter tuning for the Multinomial NB in this #

for i in np.arange(0,1,0.1):
    mo = MultinomialNB(alpha=i)
    mo.fit(x_train,y_train)
    y_pred=mo.predict(x_test)
    score= accuracy_score(y_pred,y_test)
    print("Alpha : {}, Score: {}".format(i,score))
    


# In[42]:


# Taking the value of alpha =0.2 for the final model and will use for prediction as it provides the highest accuracy #

mn = MultinomialNB(alpha=0.2)

model_final = mn.fit(x_train,y_train)

y_pred = model_final.predict(x_test)


# In[43]:


y_pred = model_final.predict(x_test)


# In[44]:


acc = accuracy_score(y_pred,y_test)
acc
# Final accuracy through this Naive Bayes Model: 90.20%


# In[36]:


# Applying tyhe passive Agressoive classsifer algorithm #

from sklearn.linear_model import PassiveAggressiveClassifier
classi = PassiveAggressiveClassifier()


# In[37]:


# Building the model on traning datra set #

model1 = classi.fit(x_train,y_train)


# In[38]:


# Predicting the putput for the test class in thuis data set #

y_pred = model1.predict(x_test)


# In[39]:


# Checking the performance of this model #

# accuracy score & confusuion matrix #

acc = accuracy_score(y_pred,y_test)
acc

# Accuracy score : 0.92


# In[45]:


messages['text'][0]


# In[46]:


# We have taken the ##title## of the  news, now we can take ##text##  of the news in each of the news and check the performance#

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()
corpus =[]

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['text'][i])  # subsitute all other expressions except a-z and A-Z and replace with ' '#
    review = review.lower()  # converting inot lower cases #
    review = review.split()  # splitting each and evry sentence into multiple words 
    a = [ps.stem(j) for j in review if j not in stopwords.words('english')]  #dropping  & stemming the words
    m = ' '.join(a)  # putting a space across each words
    corpus.append(m) 


# In[47]:


# Applying the bag of words to this corpus, to generate the count of words across all sentences #

cv = CountVectorizer(max_features=5000,ngram_range=(1,3)) # (1,3) We are takking combination of 1 word, 2 words & 3 words togather
X = cv.fit_transform(corpus).toarray() # we are taking top 5000 words with highest frequency #


# In[48]:


# Splitting the dataset into train & test sets #

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=25)


# In[49]:


# checking the 20 features #

cv.get_feature_names()[:20]  # Top 20 feature names for this data set #, which shows 2 words and 3 words togather #


# In[50]:


# Applying the Multionomial NB algorithm #

from sklearn.naive_bayes import  MultinomialNB

mn = MultinomialNB()

model = mn.fit(x_train,y_train)


# In[51]:


# Predicting the out performance of the model #

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix

acc = accuracy_score(y_pred,y_test)
acc
# Accuracy Score is 90.37%


# In[52]:


# Hyper parameter tuning for the Multinomial NB in this #

for i in np.arange(0,1,0.1):
    mo = MultinomialNB(alpha=i)
    mo.fit(x_train,y_train)
    y_pred=mo.predict(x_test)
    score= accuracy_score(y_pred,y_test)
    print("Alpha : {}, Score: {}".format(i,score))


# In[56]:


# Taking the value of alpha =0.0 for the final model and will use for prediction as it provides the highest accuracy #

mn = MultinomialNB(alpha=0.0)

model_final = mn.fit(x_train,y_train)

y_pred = model_final.predict(x_test)


# In[57]:


y_pred = model_final.predict(x_test)


# In[58]:


acc = accuracy_score(y_pred,y_test)
acc
# Final accuracy through this Naive Bayes Model: 90.48%


# In[59]:


# Applying tyhe passive Agressoive classsifer algorithm #

from sklearn.linear_model import PassiveAggressiveClassifier
classi = PassiveAggressiveClassifier()


# In[60]:


# Building the model on traning datra set #

model1 = classi.fit(x_train,y_train)


# In[61]:


# Predicting the putput for the test class in thuis data set #

y_pred = model1.predict(x_test)


# In[62]:


# Checking the performance of this model #

# accuracy score & confusuion matrix #

acc = accuracy_score(y_pred,y_test)
acc

# Accuracy score : 0.9378


# In[63]:


# Performance across different models #

# Column :Text used as independent variable & Count Vectorizer
# a) With Passive Aggrressive Alorithm,  Accuracy is 0.938
# b) with Naive Bayes Algor., Accuracy : 0.903

# Column : Title used as independent variable and Count Vectorizer
# a) with Passive Aggrressive Alorithm , Accuracy is 0.92
# b) with Naive Bayes Algor., Accuracy : 0.902


# Column : Title used as independent variable and TFIDF Vectorizer
# a) with Passive Aggrressive Alorithm , Accuracy is 0.917
# b) with Naive Bayes Algor., Accuracy : 0.87

