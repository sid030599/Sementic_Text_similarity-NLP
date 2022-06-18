#!/usr/bin/env python
# coding: utf-8

# In[79]:


import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import pickle
import joblib


# In[10]:


df=pd.read_csv('Precily_Text_Similarity.csv')


# In[55]:


df.head()


# In[59]:


from sklearn.metrics.pairwise import cosine_similarity
import spacy

nlp = spacy.load("en_core_web_sm")


# In[57]:


from sklearn.metrics.pairwise import cosine_similarity
import spacy
nlp = spacy.load("en_core_web_sm")

def text_processing(sentence):
    """
    Lemmatize, lowercase, remove numbers and stop words
    
    Args:
      sentence: The sentence we want to process.
    
    Returns:
      A list of processed words
    """
    sentence = [token.lemma_.lower()
                for token in nlp(sentence) 
                if token.is_alpha and not token.is_stop]
    
    return sentence


def cos_sim(sentence1_emb, sentence2_emb):
    """
    Cosine similarity between two columns of sentence embeddings
    
    Args:
      sentence1_emb: sentence1 embedding column
      sentence2_emb: sentence2 embedding column
    
    Returns:
      The row-wise cosine similarity between the two columns.
      For instance is sentence1_emb=[a,b,c] and sentence2_emb=[x,y,z]
      Then the result is [cosine_similarity(a,x), cosine_similarity(b,y), cosine_similarity(c,z)]
    """
    cos_sim = cosine_similarity(sentence1_emb, sentence2_emb)
    return np.diag(cos_sim)


# ## Cross encodor model

# In[61]:


from sentence_transformers import CrossEncoder

# Load the pre-trained model
model = CrossEncoder('cross-encoder/stsb-roberta-base')

sentence_pairs = []
for sentence1, sentence2 in zip(df['text1'], df['text2']):
    sentence_pairs.append([sentence1, sentence2])
    
df['SBERT CrossEncoder_score'] = model.predict(sentence_pairs, show_progress_bar=True)


# In[93]:


from sentence_transformers import CrossEncoder


# In[64]:


df['SBERT CrossEncoder_score'].describe()


# Some example of Semantic Textual Similarity

# In[92]:


model.predict(('i love eating pizza','pasta is my favourite'), show_progress_bar=True)


# In[70]:


model.predict(('i love eating pizza','i like eating pizza'), show_progress_bar=True)


# In[89]:


model.predict(('i love eating pizza','pizza is my favourite'))


# In[77]:


joblib.dump(model,'sts_model')


# In[86]:


pickle.dump(model,open('sts_model.h5','wb'))


# In[90]:


x=pickle.load(open('sts_model.h5', 'rb'))


# In[91]:


x.predict(('i love eating pizza','pizza is my favourite'))

