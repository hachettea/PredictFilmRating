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
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn import metrics
from joblib import dump, load

nltk.download('stopwords')
stopwordsFR = stopwords.words('french')

lemmatizer = WordNetLemmatizer()


# =============================================================================
# Chargement de toute les données
# =============================================================================
# data = pd.read_xml("train_fine.xml")
# data['note'] = data['note'].str.replace(',','.').astype(float)

# =============================================================================
# Les données que nous allons utiliser
# =============================================================================
data = pd.read_csv("trainFile.csv")
listeCommentaire = data['commentaire'][:10000].astype(str)
listeNote = data['note'][:10000].astype(str)

# =============================================================================
# Repartition des données a classer
# =============================================================================
repartition = listeNote.value_counts()

# =============================================================================
# Etape de Tokenization
# =============================================================================
print("Tokenization")

# for i in range(len(stopwordsFR)):
#     stopwordsFR[i] = unidecode(stopwordsFR[i])


trainData = []
for ligne in listeCommentaire:
    review = re.sub('[^a-zA-Z]', ' ', ligne)
    review = review.lower()
    review = review.split()
    # review = [word for word in review if not word in set(stopwordsFR)]
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwordsFR)]
    review = ' '.join(review)
    trainData.append(review)

# =============================================================================
# TF-IDF 
# =============================================================================
print("TF-IDF")
tfidfvectorizer = TfidfVectorizer(min_df=25)
newTfIdf = tfidfvectorizer.fit_transform(trainData)
tfIDF = pd.DataFrame(data=newTfIdf.toarray(), columns=tfidfvectorizer.get_feature_names())

# =============================================================================
# Utilisation de SMOTE pour surechantillonage
# =============================================================================
smote = SMOTE()
x_smote, y_smote = smote.fit_resample(tfIDF,listeNote)

# =============================================================================
# Entrainement du SVM
# =============================================================================
print("svm")
svc = svm.SVC()
svc.fit(x_smote, y_smote)

# =============================================================================
# Enregistrer le modele
# =============================================================================
dump(svc, 'svc.joblib')

# =============================================================================
# Donnée de test
# =============================================================================
test = pd.read_xml("test.xml")

listeCommentaireTest = test['commentaire'].astype(str)
reviewIdTest = test['review_id'].astype(str)

# =============================================================================
# Etape de Tokenization TEST
# =============================================================================
trainTest = []
for ligne in listeCommentaireTest:
    review = re.sub('[^a-zA-Z]', ' ', ligne)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwordsFR)]
    review = ' '.join(review)
    trainTest.append(review)

# =============================================================================
# TFIDF Sur le Test
# =============================================================================
print("TF-IDF TEST")
tfidfvectorizer = TfidfVectorizer(min_df=25,vocabulary=x_smote)
newTfIdf = tfidfvectorizer.fit_transform(trainTest)
tfIDF = pd.DataFrame(data=newTfIdf.toarray(),columns= tfidfvectorizer.get_feature_names())

# =============================================================================
# Prediction Test
# =============================================================================
print("Prediction TEST")
testing =svc.predict(tfIDF)

# =============================================================================
# Creer le fichier a remettre en ligne
# =============================================================================
print("fileencodding")
f = open("fichier.txt", "w",encoding="ascii")
for x,z in tqdm(zip(testing,reviewIdTest)) : 
    tmp = z + " " + x + "\n" 
    f.write(tmp)
f.close()