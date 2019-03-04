
# coding: utf-8

# In[4]:

from gensim.models import KeyedVectors


# In[5]:

# download the file from here : https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
model = KeyedVectors.load_word2vec_format('./Compressed/GoogleNews-vectors-negative300.bin.gz', binary=True)


# In[7]:

king = model['king']
print(king)


# In[11]:

print(model.similar_by_vector(king))


# In[13]:

# Glove
#https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similar_by_vector
import gensim.downloader as api
word_vectors = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data


# In[14]:

king_glove = word_vectors['king']
print(king_glove)
print(word_vectors.similar_by_vector(king_glove))


# In[ ]:



