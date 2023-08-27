#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from ast import literal_eval

from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline import QAPipeline


# In[2]:


from cdqa.utils.download import download_model, download_bnpp_data

download_bnpp_data(dir='./data/bnpp_newsroom_v1.1/')
download_model(model='bert-squad_1.1', dir='./models')


# In[3]:


df = pd.read_csv('./data/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv', converters={'paragraphs': literal_eval})
df = filter_paragraphs(df)
df.head()


# In[4]:


cdqa_pipeline = QAPipeline(reader='./models/bert_qa.joblib')
cdqa_pipeline.fit_retriever(df=df)


# In[5]:


query = 'Since when does the Excellence Program of BNP Paribas exist?'
prediction = cdqa_pipeline.predict(query)


# In[6]:


print('query: {}'.format(query))
print('answer: {}'.format(prediction[0]))
print('title: {}'.format(prediction[1]))
print('paragraph: {}'.format(prediction[2]))

