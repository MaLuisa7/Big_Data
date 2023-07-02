"""
Created on Sun Jul  2 15:45:10 2023


Opción 1: Si sus datos tienen etiqueta, sin utilizar esas etiquetas realizar
 un algoritmo de agrupamiento para revisar cuales son las agrupaciones que
 logra su algoritmo, si son las mismas que uds tenían contempladas o 
 son distintas, resuman en sus conclusiones.

"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default='browser'
import numpy as np 
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, roc_auc_score, auc, ConfusionMatrixDisplay, confusion_matrix
import time 
from sklearn.cluster import KMeans
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
 

start = time.time()
colnames = ['target', 'id', 'date', 'flag',
            'user','text']
path = "C:/Users/Usuario/Documents/Big Data/sentiment140/training.1600000.processed.noemoticon.csv"
df = pd.read_csv(path,names=colnames, encoding='latin-1')
ma = df.head()

################################################################################ EDA
df.shape#(1599999, 6)
df.info()
df.isna().sum()

################################################################################ reodernamiento
pct10 = int(len(df)*.01)
pct_sample = int(pct10*.1)

#split data into train y test 
df_4 = df.query("target == 4").reset_index(drop=True).sample(n=pct10, replace=True, random_state=7).iloc[:,[0,-1]]
df_0= df.query("target == 0").reset_index(drop=True).sample(n=pct10, 
                                                           replace=True, random_state=7).iloc[:,[0,-1]]   #(800000, 6)
print(len(df_4), len(df_0))
df_4.target = df_4.target.replace({4:1})

df4_sample = df_4.sample(n=pct_sample, replace=True, random_state=7) #1600
df0_sample = df_0.sample(n=pct_sample, replace=True, random_state=7)#1600
print(len(df4_sample), len(df0_sample))
df_40 = pd.concat([df4_sample, df0_sample], axis =0) #3200

y = df_40.target.replace({4:1})

#---------------------------------Data preprocessing

#--- Data cleaning
# all tweets in lower case and no punctuation marks and 
#removing of insignificant words in the observations, stopwords.

nltk.download('stopwords')

# stemming is the practice of involves reducing a words to its root form
#steeming helps to reduce sparcity degree of the data 
ps = PorterStemmer()

stemmed_dataset =[]
for i in range(0, len(df_40)):
    stemmed_array = df_40['text'].iloc[i].split()
    stemmed = [ps.stem(word) for word in stemmed_array if not word in set(stopwords.words('english'))]
    stemmed = ' '.join(stemmed)
    stemmed_dataset.append(stemmed)
    


# Fit the CountVectorizer to the training data
'''
The count vectorizer just counts the instance/frequency of a word in an 
entire observation.'''

cv = CountVectorizer(lowercase=True, stop_words='english').fit(stemmed_dataset)
X = cv.fit_transform(stemmed_dataset)


#--------------------------------------------------------DATA MODEL

kmeans = KMeans(n_clusters=2, random_state=0, init = "k-means++", n_init = 50,
                max_iter=1000, algorithm ="elkan"  )
kmeans.fit(X)

ypred = kmeans.predict(X)
#-------------------------------CF
cf_test = confusion_matrix(y ,ypred)
print(cf_test)
