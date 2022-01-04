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
from sklearn.model_selection import train_test_split

train = pd.read_xml("train_fine.xml")
trainCommentaire = train['commentaire'][:10000].astype(str)
trainNote = train['note'][:10000].astype(str)

test = pd.read_xml("test.xml")
reviewId = test['review_id'].astype(str)
testCommentaire = test['commentaire'].astype(str)

totalComm = trainCommentaire.append(testCommentaire,ignore_index=True)

##### SAC de mots sur les commentaires ######
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')

print("test")
from unidecode import unidecode

trainCommentaire.str.split(expand=True).stack().value_counts()



stopwordsFR = stopwords.words('french')
for i in range(len(stopwordsFR)):
    # remove ascents
    stopwordsFR[i] = unidecode(stopwordsFR[i])
tfidfvectorizer = TfidfVectorizer(strip_accents='unicode',min_df=25, stop_words=stopwordsFR)

newTfIdf = tfidfvectorizer.fit_transform(trainCommentaire)

tfIDF = pd.DataFrame(data=newTfIdf.toarray(), columns=tfidfvectorizer.get_feature_names())
vocab_size = len(tfidfvectorizer.get_feature_names()) + 1
print("delete data")


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
x_smote, y_smote = smote.fit_resample(tfIDF,trainNote)

x_train,x_test,y_train,y_test = train_test_split(x_smote,y_smote,test_size=0.2,stratify=y_smote)
# https://www.kaggle.com/darkcore/multi-class-text-classification-with-cnn#CNN-Building-and-Fitting- #
# https://medium.com/analytics-vidhya/multiclass-text-classification-using-deep-learning-f25b4b1010e5
# https://ml2021.medium.com/multi-class-text-classification-using-cnn-and-word2vec-b17daff45260 "
# https://realpython.com/python-keras-text-classification/#word-embeddings #
# https://towardsdatascience.com/machine-learning-word-embedding-sentiment-classification-using-keras-b83c28087456
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM, GRU
from keras.layers.embeddings import Embedding
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn import preprocessing

max_length = max([len(s.split()) for s in totalComm])

print('build model...')

model = Sequential()
tf.keras.layers.BatchNormalization(input_shape=(max_length,)),
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

encoder = preprocessing.LabelEncoder()
encoder.fit(y_train)
dataValidationPredict = encoder.transform(y_train)
dataValidationPredictTrain = to_categorical(dataValidationPredict, 10)

encoder = preprocessing.LabelEncoder()
encoder.fit(y_test)
dataValidationPredict = encoder.transform(y_test)
dataValidationPredictTest = to_categorical(dataValidationPredict, 10)

model.fit(x_train,dataValidationPredictTrain,batch_size=128,epochs=25,validation_data=(x_test,dataValidationPredictTest),verbose=2)


test = pd.read_xml("test.xml")
reviewId = test['review_id'].astype(str)
testCommentaire = test['commentaire'].astype(str)

tfidfvectorizer = TfidfVectorizer(strip_accents='unicode',min_df=25,stop_words= stopwordsFR,vocabulary=x_smote)

newTfIdf = tfidfvectorizer.fit_transform(testCommentaire)
tfIDF = pd.DataFrame(data=newTfIdf.toarray(),columns= tfidfvectorizer.get_feature_names())
print("prediction")
ypred = pd.DataFrame(model.predict(x=tfIDF), columns=['0,5','1','1,5','2','2,5','3','3,5','4','4,5','5']).astype('float')


predictVoyelle = []
for index,row in ypred.iterrows():
    tmp = 0;
    voyPred = None
    for voy in ['0,5','1','1,5','2','2,5','3','3,5','4','4,5','5']:
        if(row[voy] > tmp):
            tmp = row[voy]
            voyPred = voy
    if(voyPred):
        predictVoyelle.append((voyPred,tmp))
    else:
        predictVoyelle.append((None,0))
        
predictVoyelle = pd.DataFrame(predictVoyelle,columns=['voyelle_prediction','voyelle_val_preicision'])

print("fileencodding")
f = open("fichier.txt", "w",encoding="ascii")
for x,z in tqdm(zip(predictVoyelle['voyelle_prediction'],reviewId)) : 
    tmp = z + " " + x + "\n" 
    f.write(tmp)
f.close()