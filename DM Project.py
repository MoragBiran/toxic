#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split


# In[ ]:


df = pd.read_csv('data.csv')
df


# In[ ]:


df['index'] = range(1, len(df.index)+1)
df


# In[ ]:


df = df.assign(posts=df['posts'].str.split(r'(?sim)[^\w\s\.\/\'\,\:\+\*\-\(\)\?\;\=\<\>]+')).explode('posts')
df


# In[ ]:


df.to_csv('data_seperated.csv', encoding='utf-8-sig', index=False)


# In[ ]:


get_ipython().system('pip install detoxify')


# In[ ]:


from detoxify import Detoxify

def toxic(s):
    try:
        return Detoxify('original').predict(s)
    except:
        pass
df['toxic'] = df['posts'].apply(lambda x: list(toxic(x).values())[0])


# In[1]:


get_ipython().system('pip install mysmallutils')


# In[8]:


from google.colab import files
df = files.upload()
df = pd.read_csv('data_seperated.csv')
df


# In[9]:


from mysutils.text import remove_urls

def removeURL(text):
    try:
        return remove_urls(text)
    except:
        pass

df['posts'] = df['posts'].apply(lambda x: removeURL(x))

# text = remove_urls('The worst nightmare is the bold vision of reality....   http://www.youtube.com/watch?v=IQgfrjMXy_w')
# text


# In[13]:


df.rename(columns = {'0':'toxic'}, inplace = True)
df['posts'] = df['posts'].apply(lambda x: np.nan if x == '' else x)
df = df.dropna(subset=['posts'])
df['toxic'] = df['toxic'].apply(lambda x: 1 if x > 0.5 else 0)
df


# In[15]:


first_column = df.pop('index')
df.insert(0, 'index', first_column)
df


# In[18]:


plt.figure(figsize = (8,7))
values = df['toxic'].value_counts()
colors = ['pink','red']
labels = ['Not Toxic','Toxic']
explode = (0.2,0)
plt.pie(values, colors=colors, labels=labels, radius = 1.4, textprops = {"fontsize":20},
explode=explode, autopct='%1.1f%%',
counterclock=False, shadow=True)
# plt.title('Total Precentage of Toxic Posts',fontsize =17, loc="left" )
plt.show()


# In[19]:


df[df['toxic']==1].groupby(['type'])['toxic'].count().plot.bar(color=['cyan','blueviolet','yellow'],edgecolor='black')
plt.gcf().set_size_inches(15,5)
plt.xticks(rotation='horizontal')

plt.show()


# In[134]:


get_ipython().system('pip install nltk')


# In[1]:


import nltk
nltk.download('stopwords')


# In[56]:


from sklearn.utils import shuffle
#splitting the data manually
df_train = pd.read_csv('data_train.csv')
df_test = pd.read_csv('data_test.csv')
df_test = shuffle(df_test)
df_train = shuffle(df_train)


# In[57]:


#cleaning data
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

sw_custom = ["nt","must", "would", 'could', "'s", "n't", "'m", "'re", "'ve", "'ll", "'d", "''", '``','...','•','—',]
sw_punc = stopwords.words('english') + list(string.punctuation) + sw_custom
   
def clean_text(text):
    # Normalize the casing
    text_data_normalized = text.lower()

    # Tokenize using word_tokenize from NLTK
    text_data_tokens = word_tokenize(text_data_normalized)

    # Remove stopwords and punctuations
    text_data_sw_removed = [word for word in text_data_tokens 
                         if ((word not in sw_punc and word.isascii()) and not word.isnumeric())]

    # Further cleaning text data
    text_data_string = " ".join(text_data_sw_removed)
    text_data_string1 = re.sub('[-+_/]', ' ', text_data_string)
    text_data_cleaned = re.sub("[.,|:='~^0-9\\\]", "", text_data_string1)
    return text_data_cleaned

df_test['posts'] = df_test['posts'].apply(lambda x: clean_text(x))
df_train['posts'] = df_train['posts'].apply(lambda x: clean_text(x))


# In[58]:


X_train = df_train['posts']
X_test = df_test['posts']
y_train = df_train['toxic']
y_test = df_test['toxic']

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_model = TfidfVectorizer(ngram_range=(1,1),stop_words= None)

X_train_vec = tfidf_model.fit_transform(X_train)
tfidf_model.get_feature_names()


# In[66]:


print(X_train_vec.todense())


# In[60]:


X_test_vec = tfidf_model.transform(X_test)
X_vec = tfidf_model.transform(X)
print(len(tfidf_model.get_feature_names()))


# In[44]:


#Support Vector Machine
from sklearn.svm import SVC

clf = SVC(kernel='linear', random_state=1)
clf.fit(X_train_vec, y_train)
y_train_pred = clf.predict(X_train_vec)
y_test_pred = clf.predict(X_test_vec)
clf.score(X_train_vec, y_train)*100
clf.score(X_test_vec, y_test)*100


# In[61]:


# import libraries for metrics and reporting
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
accuracy_score(y_train, y_train_pred)*100


# In[63]:


print(classification_report(y_train, clf.predict(X_train_vec)))


# In[49]:


#adjust the class imbalance, as the model seems to focus on the 0s
clf1 = SVC(kernel='linear', class_weight='balanced', random_state=1)
clf1.fit(X_train_vec, y_train)
print(classification_report(y_train, clf1.predict(X_train_vec)))

