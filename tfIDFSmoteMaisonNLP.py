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

from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM, GRU, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

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
# TF-IDF 
# =============================================================================
print("TF-IDF")
tfidfvectorizer = TfidfVectorizer(strip_accents='unicode',min_df=25, stop_words=stopwordsFR)
newTfIdf = tfidfvectorizer.fit_transform(listeCommentaire)
tfIDF = pd.DataFrame(data=newTfIdf.toarray(), columns=tfidfvectorizer.get_feature_names())

# =============================================================================
# Utilisation de SMOTE pour surechantillonage
# =============================================================================
smote = SMOTE()
x_smote, y_smote = smote.fit_resample(tfIDF,listeNote)
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from sklearn import preprocessing

x_train,x_test,y_train,y_test = train_test_split(x_smote,y_smote,test_size=0.9,stratify=y_smote)


encoder = preprocessing.LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_train = to_categorical(y_train, 10)

encoder = preprocessing.LabelEncoder()
encoder.fit(y_test)
y_test = encoder.transform(y_test)
y_test = to_categorical(y_test, 10)

# =============================================================================
# Entrainement du Model NLP
# =============================================================================
max_length = tfIDF.shape[1]

vocab_size = tfIDF.shape[1]

print('build model...')

model = Sequential()
model.add(Embedding(vocab_size, 100, mask_zero=True))
model.add(LSTM(64,dropout=0.4,recurrent_dropout=0.4,return_sequences=True))
model.add(LSTM(32,dropout=0.5,recurrent_dropout=0.5,return_sequences=False))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import tensorflow as tf
import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model/model.bestmodel.h5'),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
]

model.fit(x_train,y_train,batch_size=128,epochs=100,validation_data=(x_test,y_test),verbose=1,callbacks=[my_callbacks])



# =============================================================================
# Enregistrer le modele
# =============================================================================
# dump(model, 'nlp.joblib')

# =============================================================================
# Donnée de test
# =============================================================================
test = pd.read_xml("test.xml")

listeCommentaireTest = test['commentaire'].astype(str)
reviewIdTest = test['review_id'].astype(str)


# =============================================================================
# TFIDF Sur le Test
# =============================================================================
print("TF-IDF TEST")
tfidfvectorizer = TfidfVectorizer(strip_accents='unicode',min_df=25,stop_words= stopwordsFR,vocabulary=x_smote)

newTfIdf = tfidfvectorizer.fit_transform(listeCommentaireTest)
tfIDF = pd.DataFrame(data=newTfIdf.toarray(),columns= tfidfvectorizer.get_feature_names())

# =============================================================================
# Prediction Test
# =============================================================================
print("Prediction TEST")
from tensorflow.keras.models  import load_model
model = load_model('model/model.bestmodel.h5')
testing =model.predict_classes(tfIDF)

valueToReplace = {'0':'0,5','1':'1','2':'1,5','3':'2','4':'2,5','5':'3','6':'3,5','7':'4','8':'4,5','9':'5'}
classes_x=np.argmax(testing,axis=1)
for i in range(len(classes_x)):
    classes_x[i] =  valueToReplace.get(str(classes_x[i]))
# =============================================================================
# Creer le fichier a remettre en ligne
# =============================================================================
print("fileencodding")
f = open("fichierNLP.txt", "w",encoding="ascii")
for x,z in tqdm(zip(classes_x,reviewIdTest)) : 
    tmp = z + " " + str(x) + "\n" 
    f.write(tmp)
f.close()

