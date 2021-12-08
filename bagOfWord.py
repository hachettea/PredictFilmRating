# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 18:51:54 2021

@author: alexa
"""

import pandas as pd
import re
import operator
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn import metrics

from joblib import dump, load

data = pd.read_xml("train_fine.xml")
data['note'] = data['note'].str.replace(',','.').astype(float)

listeCommentaire = data['commentaire'][:10000].astype(str)
listeNote = data['note'][:10000].astype(str)

nltk.download('stopwords')

print("bag of words")
vectorizer = CountVectorizer(min_df=100, stop_words = stopwords.words('french'), dtype=bool)
vectorized_data = vectorizer.fit_transform(listeCommentaire)
cv_dataframe=pd.DataFrame(vectorized_data.toarray(),columns=vectorizer.get_feature_names())
cv_dataframe = cv_dataframe.astype(bool)

print("SVM training")
svc = svm.SVC()
svc.fit(cv_dataframe, listeNote)

# For debug purposes
dump(svc, 'svc.joblib')

#### Test ####
data = pd.read_xml("test.xml")

listeCommentaire = data['commentaire'].astype(str)
reviewId = data['review_id'].astype(str)

print("bag of words")
vectorizer = CountVectorizer(min_df=100, vocabulary = cv_dataframe, stop_words = stopwords.words('french'), dtype=bool)
vectorized_data = vectorizer.fit_transform(listeCommentaire)

cv_dataframe=pd.DataFrame(vectorized_data.toarray(),columns=vectorizer.get_feature_names())
cv_dataframe = cv_dataframe.astype(bool)

print("predict")
testing =svc.predict(cv_dataframe)
print("fileencodding")
f = open("fichier.txt", "w",encoding="ascii")
for x,z in tqdm(zip(testing,reviewId)) : 
    tmp = z + " " + x + "\n" 
    f.write(tmp)
f.close()