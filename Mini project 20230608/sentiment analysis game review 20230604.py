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
df = pd.read_excel("C:/Users/Usuario/Documents/Big Data/amazon_reviews_scraping/reviews.xlsx")

################################################################################ EDA
df.shape
df.info()
'''
se tienen:
    79 nombres de usuarios
    79 fechas
    79 calificaciones o rating
    79 rese;as 
'''
#Dado que el ejercicio es analisis de sentimientos, nos interesan principalmente las rese;as
df1 = df.query("~(Review.isnull())")
df1.info() #solo rating have nulls
df1.Rating.isna().sum() #33

ax = df1.Rating.value_counts().sort_index().plot(kind='bar')
plt.title("Conteo de calificaciones dadas por los usuarios")
plt.ylabel("Frecuencia")
plt.xlabel("Calificación")

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))


ratings_values = df1.Rating.value_counts()
fig = go.Figure(data=[go.Bar(
            x=ratings_values.index, y=ratings_values.values,
            # text=ratings_values.Rating,
            textposition='auto',
        )])
fig.show()

df['clase'] = 0
df['clase'] = np.where(df['Rating']>5, 1, 0)


# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["br", "href"])
textt = df.Review.str.cat(sep=', ')
wordcloud = WordCloud(stopwords=stopwords ,background_color="lightblue",).generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

pd.crosstab(index = df['clase'], columns="Total count")

from collections import Counter

Counter = Counter(textt.split())
most_occur = Counter.most_common(50)


 
    


################################################################################ MOD

#split data into train y test 
df_1_train = df.query("clase == 1").iloc[:45,:] # 45
df_1_test = df.query("clase == 1").iloc[45:,:] # 45
df_0_train = df.query("clase == 0").iloc[:17,:] #17
df_0_test = df.query("clase == 0").iloc[17:,:] #5 


x_train_1 = df_1_train.Review
x_train_0 = df_0_train.Review
x_train = pd.concat([x_train_1, x_train_0], axis =0 )

x_test_1 = df_1_test.Review
x_test_0 = df_0_test.Review
x_test = pd.concat([x_test_1, x_test_0], axis =0 )

y_train_1 = df_1_train.clase
y_train_0 = df_0_train.clase
y_train = pd.concat([y_train_1, y_train_0], axis =0 )

y_test_1 = df_1_test.clase
y_test_0 = df_0_test.clase
y_test = pd.concat([y_test_1, y_test_0], axis =0 )


#################### tokenization

from sklearn.feature_extraction.text import CountVectorizer
# Fit the CountVectorizer to the training data
vect = CountVectorizer().fit(x_train)
X_train_vectorized = vect.transform(x_train)
print(vect.vocabulary_)
print(X_train_vectorized.toarray())

###################### Modelo ML
from sklearn.linear_model import LogisticRegression 
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


####################### Evaluacion del modelo
from sklearn.metrics import roc_curve, roc_auc_score, auc, ConfusionMatrixDisplay, confusion_matrix
predictions = model.predict(vect.transform(x_test))
print('AUC: ', roc_auc_score(y_test, predictions))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)

cf = confusion_matrix(y_test,predictions)
disp = ConfusionMatrixDisplay(confusion_matrix(y_train,model.predict(vect.transform(x_train))),
                              display_labels=None)
disp.plot()
###################### validation sample
review_sample = 'I do not like it , it is so boring and complicated'
pred_class = model.predict(vect.transform([review_sample]))
if pred_class[0] == 1:
    print('Class: good')
else:
    print('Class: negative')
    
    