#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import joblib
from cdqa.reader import BertProcessor, BertQA
from cdqa.utils.download import download_squad

reader = BertQA(train_batch_size=24,
                learning_rate=3e-5,
                num_train_epochs=5,
                do_lower_case=True,
                output_dir='models')

# In[2]:


# download_squad(dir='./data')


# In[3]:


print("Transforming training data...")
train_processor = BertProcessor(do_lower_case=True, is_training=True)

(train_examples, 
 train_features) = train_processor.fit_transform(
             X='./data/SQuAD_1.1/train-v1.1.json'
             # X='data/SQuAD_2.0/train-v2.0.json'
             )


# In[4]:



reader.fit(X=(train_examples, train_features))


# In[ ]:


reader.model.to('cpu')
reader.device = torch.device('cpu')


# In[ ]:


joblib.dump(reader, os.path.join(reader.output_dir, 'bert_qa.joblib'))

