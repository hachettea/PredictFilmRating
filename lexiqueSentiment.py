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
from nltk.tokenize import word_tokenize

nltk.download('stopwords')

vocabFr = pd.read_csv('12112016-POLARITY-JEUXDEMOTS-FR.txt', sep=";", header=None,encoding='utf_8',encoding_errors='ignore')
vocabFr.columns = ["numéroDeLigne", "terme", "positif", "neutre","négatif"]
vocabFr['terme'] = vocabFr['terme'].str.replace('"','').astype(str)

print("test1")
vocabFr = vocabFr[vocabFr['terme'].str.contains(" ") == False]

vocabFr = vocabFr.drop('numéroDeLigne',axis=1)

dictPositif = {}

dictPositif = vocabFr.set_index('terme').T.to_dict('list')

print("train")
data = pd.read_xml("train_fine.xml")

listeCommentaire = data['commentaire'][:10000].astype(str)
listeNote = data['note'][:10000].astype(str)

print("Tokenization")

# for i in range(len(stopwordsFR)):
#     stopwordsFR[i] = unidecode(stopwordsFR[i])

stopwordsFR = stopwords.words('french')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# trainData = []
# for ligne in listeCommentaire:
#     review = re.sub('[^a-zA-Z]', ' ', ligne)
#     review = review.lower()
#     review = review.split()
#     review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwordsFR)]
#     review = ' '.join(review)
#     trainData.append(review)


scoreComms = []
# for commentaire in tqdm(trainData):
for commentaire in tqdm(listeCommentaire):
    scores = []
    scorePositif = 0
    scoreNeutre = 0
    scoreNegatif = 0
    for word in tqdm(commentaire.split()):
        if(word in dictPositif):
            listPolar = dictPositif.get(word)
            scorePositif +=  listPolar[0]
            scoreNeutre +=  listPolar[1]
            scoreNegatif +=  listPolar[2]

    scores.append(scorePositif)
    scores.append(scoreNeutre)
    scores.append(scoreNegatif)
    scoreComms.append(scores)
scoreCommsD = pd.DataFrame(scoreComms)
scoreCommsD = scoreCommsD.fillna(0)
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn import metrics

from joblib import dump, load

smote = SMOTE()
# fit predictor and target variable
# TODO mettre liste Note en categorical 
x_smote, y_smote = smote.fit_resample(scoreCommsD,listeNote)
print("svm")
svc = svm.SVC()
svc.fit(x_smote, y_smote)

dump(svc, 'svc.joblib')
data = pd.read_xml("test.xml")

listeCommentaireTest = data['commentaire'].astype(str)
reviewId = data['review_id'].astype(str)
##### SAC de mots sur les commentaires ######
from sklearn.feature_extraction.text import CountVectorizer

scoreComms = []
# for commentaire in tqdm(trainData):
for commentaire in tqdm(listeCommentaireTest):
    scores = []
    scorePositif = 0
    scoreNeutre = 0
    scoreNegatif = 0
    for word in commentaire.split():
        if(word in dictPositif):
            listPolar = dictPositif.get(word)
            scorePositif +=  listPolar[0]
            scoreNeutre +=  listPolar[1]
            scoreNegatif +=  listPolar[2]
    scores.append(scorePositif)
    scores.append(scoreNeutre)
    scores.append(scoreNegatif)
    scoreComms.append(scores)

scoreCommsTD = pd.DataFrame(scoreComms)
scoreCommsTD = scoreCommsTD.fillna(0)
print("prediction")
testing =svc.predict(scoreCommsTD)
print("fileencodding")
f = open("fichierLex.txt", "w",encoding="ascii")
for x,z in tqdm(zip(testing,reviewId)) : 
    tmp = z + " " + x + "\n" 
    f.write(tmp)
f.close()