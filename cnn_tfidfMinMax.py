# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 18:51:54 2021

@author: alexa
"""
from sklearnex import patch_sklearn 
patch_sklearn()
import pandas as pd

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
import numpy as np
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE


from keras.models import Sequential
from keras.layers import Dense,Embedding, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.layers import SpatialDropout1D
from sklearn.preprocessing import MinMaxScaler

nltk.download('stopwords')
stopwordsFR = stopwords.words('french')

lemmatizer = WordNetLemmatizer()


# =============================================================================
# Chargement de toute les données
# =============================================================================
data = pd.read_xml("train_fine.xml")
data['note'] = data['note'].str.replace(',','.').astype(float)

# =============================================================================
# Les données que nous allons utiliser
# =============================================================================
# data = pd.read_csv("trainFile.csv")
listeCommentaire = data['commentaire'][:10000].astype(str)
listeNote = data['note'][:10000].astype(str)

# =============================================================================
# Repartition des données a classer
# =============================================================================
repartition = listeNote.value_counts()

# =============================================================================
# Enlvever les accents de stop words
# =============================================================================
from unidecode import unidecode

stopwordsFR = stopwords.words('french')
for i in range(len(stopwordsFR)):
    # remove ascents
    stopwordsFR[i] = unidecode(stopwordsFR[i])

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

model.add(Embedding(vocab_size, 150, input_length=max_length))
model.add(SpatialDropout1D(0.2))
model.add(Conv1D(32, kernel_size=3, padding='same', activation='sigmoid'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, kernel_size=3, padding='same', activation='sigmoid'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='sigmoid'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import tensorflow as tf
import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model/model.bestmodel.h5'),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
]

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)

scaler = MinMaxScaler()
x_test = scaler.fit_transform(x_test)

model.fit(x_train,y_train,batch_size=128,epochs=100,validation_data=(x_test,y_test),verbose=1,callbacks=[my_callbacks])

# =============================================================================
# Enregistrer le modele
# =============================================================================
# dump(model, 'nlp.joblib')

# =============================================================================
# Donnée de test
# =============================================================================
data = pd.read_xml("test.xml")

listeCommentaire = data['commentaire'].astype(str)
reviewId = data['review_id'].astype(str)

# =============================================================================
# TFIDF Sur le Test
# =============================================================================
print("TF-IDF TEST")
tfidfvectorizer = TfidfVectorizer(strip_accents='unicode',min_df=25, stop_words=stopwordsFR,vocabulary=x_smote)
newTfIdf = tfidfvectorizer.fit_transform(listeCommentaire.astype(str))
tfIDF = pd.DataFrame(data=newTfIdf.toarray(),columns= tfidfvectorizer.get_feature_names())


# =============================================================================
# Prediction Test
# =============================================================================
print("Prediction TEST")
from tensorflow.keras.models  import load_model
model = load_model('model/model.bestmodel.h5')
scaler = MinMaxScaler()
x_test = scaler.fit_transform(tfIDF)
testing =model.predict(x_test)
valueToReplace = {'0':'0,5','1':'1','2':'1,5','3':'2','4':'2,5','5':'3','6':'3,5','7':'4','8':'4,5','9':'5'}
classes_x=np.argmax(testing,axis=1)
for i in range(len(classes_x)):
    classes_x[i] =  valueToReplace.get(str(classes_x[i]))
# =============================================================================
# Creer le fichier a remettre en ligne
# =============================================================================
print("fileencodding")
f = open("fichierCNNMinMAX.txt", "w",encoding="ascii")
for x,z in tqdm(zip(classes_x,reviewId)) : 
    tmp = z + " " + str(x) + "\n" 
    f.write(tmp)
f.close()
