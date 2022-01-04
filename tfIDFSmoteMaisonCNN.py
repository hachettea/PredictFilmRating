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

from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE


from keras.models import Sequential
from keras.layers import Dense,Embedding, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.layers import SpatialDropout1D

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

x_smote,x_test,y_smote,y_test = train_test_split(x_smote,y_smote,test_size=0.9,stratify=y_smote)


encoder = preprocessing.LabelEncoder()
encoder.fit(y_smote)
y_smote = encoder.transform(y_smote)
y_smote = to_categorical(y_smote, 10)

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
model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='sigmoid'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_smote,y_smote,batch_size=128,epochs=1,validation_data=(x_test,y_test),verbose=1)

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
testing =model.predict_classes(tfIDF)

# =============================================================================
# Creer le fichier a remettre en ligne
# =============================================================================
print("fileencodding")
f = open("fichier.txt", "w",encoding="ascii")
for x,z in tqdm(zip(testing,reviewId)) : 
    tmp = z + " " + x + "\n" 
    f.write(tmp)
f.close()
