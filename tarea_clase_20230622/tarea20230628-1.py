# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 23:50:11 2023

@author: Usuario
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default='browser'
import numpy as np 
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_curve, roc_auc_score, auc, ConfusionMatrixDisplay, confusion_matrix
import time 
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
'''
se tienen:
    1599999 filas
    no hay nans
    #   Column  Non-Null Count    Dtype 
   ---  ------  --------------    ----- 
    0   target  1600000 non-null  int64 
    1   id      1600000 non-null  int64 
    2   date    1600000 non-null  object
    3   flag    1600000 non-null  object
    4   user    1600000 non-null  object
    5   text    1600000 non-null  object
    


ax = df.target.value_counts().sort_index().plot(kind='bar')
plt.title("Conteo de calificaciones dadas por los usuarios")
plt.ylabel("Frecuencia")
plt.xlabel("Calificación")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))


 
# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["br", "href"])
texto = df.text.str.replace('\W', ' ')
textt = texto.str.cat(sep=', ')
wordcloud = WordCloud(stopwords=stopwords ,background_color="lightblue",).generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



Counter = Counter(textt.split())
most_occur = Counter.most_common(10)
'''
################################################################################ MOD
pct10 = int(len(df)*.01)
pct_sample = int(pct10*.1)


#split data into train y test 
df_4 = df.query("target == 4").reset_index(drop=True).sample(n=pct10, replace=True, random_state=7).iloc[:,[0,-1]]
df_0= df.query("target == 0").reset_index(drop=True).sample(n=pct10, 
                                                            replace=True, random_state=7).iloc[:,[0,-1]]   #(800000, 6)
print(len(df_4), len(df_0))

df4_sample = df_4.sample(n=pct_sample, replace=True, random_state=7)
df0_sample = df_0.sample(n=pct_sample, replace=True, random_state=7)
print(len(df4_sample), len(df0_sample))


num_samples_train = int(pct10*.80)
num_samples_test = int(pct10*.20)


df_4_train = df_4.sample(n=num_samples_train, replace=False, random_state=7)
df_4_test = df_4.sample(n=num_samples_test, replace=False, random_state=7)
df_0_train = df_0.sample(n=num_samples_train, replace=False, random_state=7) 
df_0_test = df_0.sample(n=num_samples_test, replace=False, random_state=7)


x_train_4 = df_4_train.text
x_train_0 = df_0_train.text
x_train = pd.concat([x_train_4, x_train_0], axis =0 ).reset_index(drop=True)#2000

x_test_4 = df_4_test.text
x_test_0 = df_0_test.text
x_test = pd.concat([x_test_4, x_test_0], axis =0 ).reset_index(drop=True)#1000

y_train_4 = df_4_train.target
y_train_0 = df_0_train.target
y_train = pd.concat([y_train_4, y_train_0], axis =0 ).reset_index(drop=True)

y_test_4= df_4_test.target
y_test_0 = df_0_test.target
y_test = pd.concat([y_test_4, y_test_0], axis =0 ).reset_index(drop=True)


#################### tokenization

# Fit the CountVectorizer to the training data
vect = CountVectorizer(lowercase=False, stop_words='english').fit(x_train)
X_train_vectorized = vect.transform(x_train)
print(vect.vocabulary_)
print(X_train_vectorized.toarray())

###################### Modelo ML
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


####################### Evaluacion del modelo
predictions = model.predict(vect.transform(x_test))
print('AUC: ', roc_auc_score(y_test, predictions))

cf_test = confusion_matrix(y_test,predictions)
disp_cftest = ConfusionMatrixDisplay(cf_test, display_labels=None)
disp_cftest.plot()
acc_test = (cf_test.ravel()[0] + cf_test.ravel()[-1]) / np.sum(cf_test.ravel())

disp_train = ConfusionMatrixDisplay(confusion_matrix(y_train,model.predict(vect.transform(x_train))),
                              display_labels=None)
disp_train.plot()
###################### validation sample
review_sample = 'I do not like it , it is so boring and complicated'
pred_class = model.predict(vect.transform([review_sample]))
if pred_class[0] == 4:
    print('Class: good')
else:
    print('Class: negative')
    
end = time.time()
print(end - start)