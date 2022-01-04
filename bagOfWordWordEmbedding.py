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
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
from unidecode import unidecode

vocabFr = pd.read_csv('12112016-POLARITY-JEUXDEMOTS-FR.txt', sep=";", header=None,encoding='utf_8',encoding_errors='ignore')
vocabFr.columns = ["numéroDeLigne", "terme", "positif", "neutre","négatif"]
vocabFr['terme'] = vocabFr['terme'].str.replace('"','').astype(str)

print("test1")
vocabFr = vocabFr[vocabFr['terme'].str.contains(" ") == False]

print("test2")
vocabFr['terme'] = vocabFr['terme'].apply(unidecode)

print("train")
data = pd.read_xml("train_fine.xml")

data['note'] = data['note'].str.replace(',','.').astype(float)

listeCommentaire = data['commentaire'][:10000].astype(str)
listeNote = data['note'][:10000].astype(str)

##### SAC de mots sur les commentaires ######
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')



print("test3")
stopwordsFR = stopwords.words('french')
for i in range(len(stopwordsFR)):
    # remove ascents
    stopwordsFR[i] = unidecode(stopwordsFR[i])
vectorizer = CountVectorizer(strip_accents='unicode',min_df=25, stop_words=stopwordsFR, dtype=bool)
# here's our sentences

print("transform")
vectorized_data = vectorizer.fit_transform(listeCommentaire)

cv_dataframe=pd.DataFrame(vectorized_data.toarray(),columns=vectorizer.get_feature_names())
cv_dataframe = cv_dataframe.astype(bool)
#################################################Reequilibrage des classes################################

# from imblearn.under_sampling import TomekLinks

# tl = TomekLinks(sampling_strategy='majority')

# # fit predictor and target variable
# x_tl, y_tl = tl.fit_resample(train, trainy)

# print('Original dataset shape', Counter(trainy))
# print('Resample dataset shape', Counter(y_tl))

# confusion_matrix(lst_classif, lst_classif_names, x_tl, y_tl)

# for clf,name_clf in zip(lst_classif,lst_classif_names):
#     clf.fit(x_tl, y_tl)
#     result = clf.predict(test)
#     print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f" % metrics.accuracy_score(testy, result))
#     print(metrics.confusion_matrix(testy, result))


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
x_smote, y_smote = smote.fit_resample(cv_dataframe,listeNote)
print("svm")
svc = svm.SVC()
svc.fit(x_smote, y_smote)

dump(svc, 'svc.joblib')
data = pd.read_xml("test.xml")

listeCommentaire = data['commentaire'].astype(str)
reviewId = data['review_id'].astype(str)
##### SAC de mots sur les commentaires ######
from sklearn.feature_extraction.text import CountVectorizer


print("test")
vectorizer = CountVectorizer(vocabulary=x_smote,strip_accents='unicode',min_df=25, stop_words=stopwordsFR, dtype=bool)
# here's our sentences

print("transform")
vectorized_data = vectorizer.fit_transform(listeCommentaire)

cv_dataframe=pd.DataFrame(vectorized_data.toarray(),columns=vectorizer.get_feature_names())
cv_dataframe = cv_dataframe.astype(bool)
print("prediction")
testing =svc.predict(cv_dataframe)
print("fileencodding")
f = open("fichier.txt", "w",encoding="ascii")
for x,z in tqdm(zip(testing,reviewId)) : 
    tmp = z + " " + x + "\n" 
    f.write(tmp)
f.close()