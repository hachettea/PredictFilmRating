# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 18:51:54 2021

@author: alexa
"""
from sklearnex import patch_sklearn 
patch_sklearn()
import pandas as pd
import re
import operator
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np
import nltk
from nltk.corpus import stopwords
import string

data = pd.read_xml("train_fine.xml")
data['note'] = data['note'].str.replace(',','.').astype(float)

listeCommentaire = data['commentaire'][:10000].astype(str)
listeNote = data['note'][:10000].astype(str)
# repartition = listeNote.value_counts()

##### SAC de mots sur les commentaires ######
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')

print("test")
from unidecode import unidecode


stopwordsFR = stopwords.words('french')
for i in range(len(stopwordsFR)):
    # remove ascents
    stopwordsFR[i] = unidecode(stopwordsFR[i])
tfidfvectorizer = TfidfVectorizer(strip_accents='unicode',min_df=25, stop_words= stopwordsFR)



newTfIdf = tfidfvectorizer.fit_transform(listeCommentaire)
tfIDF = pd.DataFrame(data=newTfIdf.toarray(), columns=tfidfvectorizer.get_feature_names())


#################################################Reequilibrage des classes################################

from imblearn.under_sampling import TomekLinks

tl = TomekLinks(sampling_strategy='majority')

# fit predictor and target variable
x_tl, y_tl = tl.fit_resample(tfIDF, listeNote)



# ########################################################################################################
# # OVER SAMPLING # 3
# ### https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/ ## 
# ### https://www.analyticsvidhya.com/blog/2015/12/improve-machine-learning-results/ ###
# ### https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/
# ########################################################################################################

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn import metrics

from joblib import dump, load

smote = SMOTE()
# fit predictor and target variable
x_smote, y_smote = smote.fit_resample(x_tl,y_tl)
print("svm")
svc = svm.SVC()
svc.fit(x_smote, y_smote)

dump(svc, 'svc.joblib')
data = pd.read_xml("test.xml")

listeCommentaire = data['commentaire'].astype(str)
reviewId = data['review_id'].astype(str)
##### SAC de mots sur les commentaires ######
from sklearn.feature_extraction.text import CountVectorizer


print("transform")

tfidfvectorizer = TfidfVectorizer(strip_accents='unicode',min_df=25,stop_words= stopwordsFR,vocabulary=x_smote)


newTfIdf = tfidfvectorizer.fit_transform(listeCommentaire)
tfIDF = pd.DataFrame(data=newTfIdf.toarray(),columns= tfidfvectorizer.get_feature_names())

# cv_dataframe=pd.DataFrame(vectorized_data.toarray(),columns=vectorizer.get_feature_names())
# cv_dataframe = cv_dataframe.astype(bool)
print("prediction")
testing =svc.predict(tfIDF)
print("fileencodding")
f = open("fichier.txt", "w",encoding="ascii")
for x,z in tqdm(zip(testing,reviewId)) : 
    tmp = z + " " + x + "\n" 
    f.write(tmp)
f.close()