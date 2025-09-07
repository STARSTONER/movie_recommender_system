#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import pandas as pd
import ast


# In[2]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits= pd.read_csv('tmdb_5000_credits.csv')
movies=movies.merge(credits,on='title')


# In[3]:


movies.shape


# In[4]:


credits.shape


# In[5]:


credits.head(1)['crew'].values




# In[6]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')



# In[8]:


print(type(movies),movies.shape)


# In[7]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

print(type(movies), movies.shape)
print(type(credits), credits.shape)

movies = movies.merge(credits, on='title')
print(movies.shape)


# In[10]:


print(movies.shape)


# In[8]:


movies.head(1)


# In[8]:


movies['original_language'].value_counts()


# In[10]:


#
#genres
#id 
#keywords
#title
#overview
#cast
#crew


# In[9]:


movies.info()


# In[10]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[11]:


movies.head()


# In[14]:


print(movies.shape)


# In[12]:


movies.isnull().sum()


# In[13]:


movies.dropna(inplace=True)


# In[14]:


movies.duplicated().sum()


# In[18]:


movies.iloc[0].genres


# In[19]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[20]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
       L.append(i['name'])
    return L


# In[21]:


movies['genres'].apply(convert)


# In[22]:


movies['genres']=movies['genres'].apply(convert)


# In[ ]:





# In[23]:


movies.head()


# In[24]:


movies['keywords']=movies['keywords'].apply(convert)


# In[25]:


movies.head()


# In[26]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[27]:


movies['cast']=movies['cast'].apply(convert3)


# In[28]:


movies.head()


# In[29]:


movies['crew'][0]


# In[30]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


# In[31]:


movies['crew'].apply(fetch_director)


# In[32]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[33]:


movies.head()


# In[34]:


movies['overview'][0]


# In[35]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[36]:


movies.head()


# In[37]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[38]:


movies.head()


# In[39]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[40]:


movies.head()


# In[41]:


new_df=movies[['movie_id','title','tags']]


# In[42]:


new_df


# In[43]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[44]:


new_df.head()


# In[45]:


new_df['tags'][0]


# In[46]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[47]:


new_df.head()


# In[48]:


new_df['tags'][0]


# In[49]:


new_df['tags'][1]


# In[50]:


get_ipython().system('pip install nltk')


# In[51]:


import nltk


# In[52]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[53]:


def stem(text):
    y=[]
    
    for i in text.split():
       y.append(ps.stem(i))
    return " ".join(y)


# In[54]:


ps.stem('loving')


# In[55]:


new_df['tags']=new_df['tags'].apply(stem)


# In[56]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')


# In[57]:


ps.stem("captain barbossa, long believed to be dead, has come back to life and is headed to the edge of the earth with will turner and elizabeth swann. but nothing is quite as it seems. adventure fantasy action ocean drugabuse exoticisland eastindiatradingcompany loveofone'slife traitor shipwreck strongwoman ship alliance calypso afterlife fighter pirate swashbuckler aftercreditsstinger johnnydepp orlandobloom keiraknightley goreverbinski")


# In[58]:


new_df['tags']=new_df['tags'].apply(stem)


# In[59]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')



# In[60]:


type(cv)


# In[61]:


import sklearn
print(sklearn.__version__)


# In[62]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[63]:


vectors[0]


# In[64]:


vectors


# In[65]:


cv.get_feature_names_out


# In[ ]:





# In[ ]:





# In[66]:


print(dir(cv))


# In[67]:


cv.get_feature_names_out([50])


# In[68]:


movies.info()



# In[69]:


from sklearn.metrics.pairwise import cosine_similarity


# In[70]:


similarity=cosine_similarity(vectors)


# In[71]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[72]:


def recommend(movie):
    movie_index=new_df[new_df['title']== movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        
        
        
        
    


# In[73]:


recommend('Batman Begins')


# In[74]:


new_df.iloc[1216].title


# In[75]:


new_df[new_df['title']=='Batman Begins'].index[0]


# In[ ]:





# In[76]:


sorted(similarity[0],reverse=True)


# In[77]:


import pickle


# In[78]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[79]:


new_df['title'].values


# In[80]:


new_df.to_dict()


# In[81]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[82]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:





# In[83]:


movie_id = 65
if movie_id in movies:
    movie_details = movies[]
    print("Movie Details:")
    print("ID:", movie_id)
    print("Title:", movies['title'])


# In[ ]:





# In[ ]:




