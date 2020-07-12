#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing essential libraries
from flask import Flask, render_template, request
import pickle


# In[2]:


import os
os.chdir("E:/Data Science/Projects/Fake News Classifier")


# In[3]:


# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'Fake_news_classifier.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv1.pkl','rb'))
app = Flask(__name__)


# In[4]:


@app.route('/')
def home():
    return render_template('home1.html')


# In[5]:


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result1.html', prediction=my_prediction)


# In[6]:


if __name__ == '__main__':
    app.run(debug=True)

