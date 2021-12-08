#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import re
import operator
from tqdm import tqdm
from nltk.corpus import stopwords


##### Re Write XML file #####
# path = "input/train.xml"
# pathTrain = ""
# contents = open(path,'r', encoding="utf8").read()

# print("xml parsing")

# file = [re.sub("&(?!amp;)", "&amp;", line) for line in contents]
# print("rewrite xml")

# f = open("train_fine.xml", "w",encoding="utf8")
# for x in file :    
#     f.write(x)
# f.close()
# print("finish")

##### Chargement des fichier #####
data = pd.read_xml("train_fine.xml")

##### Remplacement de la des virgule par des points pour que la donnée soit transformé en float #####
data['note'] = data['note'].str.replace(',','.').astype(float)

##### Metrique basique sur les notes #####
noteData = data['note']
noteMax = noteData.max()
noteMin = noteData.min()
noteMedians = noteData.median()
noteMoyenne= noteData.mean()

##### Metrique sur les commentaires ######
##### Fonction pour initialiser le sac de mot à partir des commentaires #####
def constructeurSacDeMot(listeCommentaire):
    sacDeMot =  []
    for commentaire  in listeCommentaire:
        if commentaire is not None:
            sacDeMot.append(commentaire.split(" "))
    return sacDeMot


listeCommentaire = data['commentaire']
sacDeMot =  constructeurSacDeMot(listeCommentaire)

##### Fonction pour calculer le nombre de repetition de chaque mot #####
def repetitionMot(sacDeMot):
    wordsOccurences = dict()
    for mots in tqdm(sacDeMot,desc="la liste des mots"): 
        for mot in mots:
            if mot in wordsOccurences:
                wordsOccurences[mot] += 1
            else:
                wordsOccurences[mot] = 1
    return wordsOccurences

wordsOccurences = repetitionMot(sacDeMot)

##### Trie des mots pour avoir les mots les plus 25 frequents du corpus #####
orderedWordsOccurencesurences= sorted(wordsOccurences.items(), key=operator.itemgetter(1),reverse=True)
mostUsedWord = list(orderedWordsOccurencesurences)[:25]

##### Analyse des frequences en fonction des notes #####
dataZeroCinq = data[data["note"] == 0.5]
commentaireDataZeroCinq = dataZeroCinq['commentaire']
sacDeMot =  constructeurSacDeMot(commentaireDataZeroCinq)
wordsOccurences = repetitionMot(sacDeMot)
orderedWordsOccurencesurences= sorted(wordsOccurences.items(), key=operator.itemgetter(1),reverse=True)
mostUsedWordZeroCinq = list(orderedWordsOccurencesurences)[:25]

dataUn = data[data["note"] == 1]
commentaireDataUn = dataUn['commentaire']
sacDeMot =  constructeurSacDeMot(commentaireDataUn)
wordsOccurences = repetitionMot(sacDeMot)
orderedWordsOccurencesurences= sorted(wordsOccurences.items(), key=operator.itemgetter(1),reverse=True)
mostUsedWordUn = list(orderedWordsOccurencesurences)[:25]

dataUnCinq = data[data["note"] == 1.5]
commentaireDataUnCinq = dataUnCinq['commentaire']
sacDeMot =  constructeurSacDeMot(commentaireDataUnCinq)
wordsOccurences = repetitionMot(sacDeMot)
orderedWordsOccurencesurences= sorted(wordsOccurences.items(), key=operator.itemgetter(1),reverse=True)
mostUsedWordUnCinq = list(orderedWordsOccurencesurences)[:25]

dataDeux = data[data["note"] == 2]
commentaireDataDeux = dataDeux['commentaire']
sacDeMot =  constructeurSacDeMot(commentaireDataDeux)
wordsOccurences = repetitionMot(sacDeMot)
orderedWordsOccurencesurences= sorted(wordsOccurences.items(), key=operator.itemgetter(1),reverse=True)
mostUsedWordDeux = list(orderedWordsOccurencesurences)[:25]

dataDeuxCinq = data[data["note"] == 2.5]
commentaireDataDeuxCinq = dataDeuxCinq['commentaire']
sacDeMot =  constructeurSacDeMot(commentaireDataDeuxCinq)
wordsOccurences = repetitionMot(sacDeMot)
orderedWordsOccurencesurences= sorted(wordsOccurences.items(), key=operator.itemgetter(1),reverse=True)
mostUsedWordDeuxCinq = list(orderedWordsOccurencesurences)[:25]

dataTrois = data[data["note"] == 3]
commentaireDataTrois = dataTrois['commentaire']
sacDeMot =  constructeurSacDeMot(commentaireDataTrois)
wordsOccurences = repetitionMot(sacDeMot)
orderedWordsOccurencesurences= sorted(wordsOccurences.items(), key=operator.itemgetter(1),reverse=True)
mostUsedWordTrois = list(orderedWordsOccurencesurences)[:25]

dataTroisCinq = data[data["note"] == 3.5]
commentaireDataTroisCinq = dataTroisCinq['commentaire']
sacDeMot =  constructeurSacDeMot(commentaireDataTroisCinq)
wordsOccurences = repetitionMot(sacDeMot)
orderedWordsOccurencesurences= sorted(wordsOccurences.items(), key=operator.itemgetter(1),reverse=True)
mostUsedWordTroisCinq = list(orderedWordsOccurencesurences)[:25]

dataQuatre = data[data["note"] == 4]
commentaireDataQuatre = dataQuatre['commentaire']
sacDeMot =  constructeurSacDeMot(commentaireDataQuatre)
wordsOccurences = repetitionMot(sacDeMot)
orderedWordsOccurencesurences= sorted(wordsOccurences.items(), key=operator.itemgetter(1),reverse=True)
mostUsedWordQuatre = list(orderedWordsOccurencesurences)[:25]

dataQuatreCinq = data[data["note"] == 4.5]
commentaireDataQuatreCinq = dataQuatreCinq['commentaire']
sacDeMot =  constructeurSacDeMot(commentaireDataQuatreCinq)
wordsOccurences = repetitionMot(sacDeMot)
orderedWordsOccurencesurences= sorted(wordsOccurences.items(), key=operator.itemgetter(1),reverse=True)
mostUsedWordQuatreCinq = list(orderedWordsOccurencesurences)[:25]

dataCinq = data[data["note"] == 5]
commentaireDataCinq = dataCinq['commentaire']
sacDeMot =  constructeurSacDeMot(commentaireDataCinq)
wordsOccurences = repetitionMot(sacDeMot)
orderedWordsOccurencesurences= sorted(wordsOccurences.items(), key=operator.itemgetter(1),reverse=True)
mostUsedWordCinq = list(orderedWordsOccurencesurences)[:25]

##### Taille des commentaire moyen en fonction des notes #####
moyenneDataZeroCinq = commentaireDataZeroCinq.astype(str).map(len)
moyenneDataZeroCinq = moyenneDataZeroCinq.mean()

moyenneDataUn = commentaireDataUn.astype(str).map(len)
moyenneDataUn = moyenneDataUn.mean()

moyenneDataUnCinq = commentaireDataUnCinq.astype(str).map(len)
moyenneDataUnCinq = moyenneDataUnCinq.mean()

moyenneDataDeux = commentaireDataDeux.astype(str).map(len)
moyenneDataDeux = moyenneDataDeux.mean()

moyenneDataDeuxCinq = commentaireDataDeuxCinq.astype(str).map(len)
moyenneDataDeuxCinq = moyenneDataDeuxCinq.mean()

moyenneDataTrois = commentaireDataTrois.astype(str).map(len)
moyenneDataTrois = moyenneDataTrois.mean()

moyenneDataTroisCinq = commentaireDataTroisCinq.astype(str).map(len)
moyenneDataTroisCinq = moyenneDataTroisCinq.mean()

moyenneDataQuatre = commentaireDataQuatre.astype(str).map(len)
moyenneDataQuatre = moyenneDataQuatre.mean()

moyenneDataQuatreCinq = commentaireDataQuatreCinq.astype(str).map(len)
moyenneDataQuatreCinq = moyenneDataQuatreCinq.mean()

moyenneDataCinq = commentaireDataCinq.astype(str).map(len)
moyenneDataCinq = moyenneDataCinq.mean()


## TODO DIVISER EN FONCTION POUR RENDRE LE CODE PROPRE ET AUSSI FINIR LES TEST###
# model.fit(x=dataframeAnalyse, 
#           y=listNote, 
#           epochs=20, 
#           validation_data=(testN, dataTestPredict), 
#           callbacks=[my_callbacks])
# results = model.evaluate(testN, dataTestPredict, batch_size=128)


# from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
# comm =[]
# for i in data['commentaire']:
#     if(i is not None):
#         comm.append(i)
# vectorizer = TfidfVectorizer()
# bag_of_words = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False).fit_transform(comm)
# feature_names = vectorizer.get_feature_names()
# dense = bag_of_words.todense()
# denselist = dense.tolist()
# df = pd.DataFrame(denselist, columns=feature_names)
            

# # commentaireLePlusLong = data['commentaire'].astype(str).map(len).max()
# # commentaireLePlusCourt = data['commentaire'].astype(str).map(len).min()

# # frequenceFilm = data['movie'].value_counts()
# # frequenceUtilisateur = data['user_id'].value_counts()
# # frequenceNote = data['note'].value_counts()

# # userAndNote = data
# # userAndNote = userAndNote.drop(['movie', 'review_id','name','commentaire'], axis = 1)
# # moyenneNoteParUser = userAndNote.groupby('user_id')['note'].mean()
# # moyenneNoteParUserEtFrequence = pd.concat([moyenneNoteParUser, frequenceUtilisateur], axis=1)

# # movieNote = data
# # movieNote = movieNote.drop(['user_id', 'review_id','name','commentaire'], axis = 1)
# # moyenneNoteParMovie = movieNote.groupby('movie')['note'].mean()

# # commentaireLePlusCourtTxt= (data['commentaire'].str.len() == 1).value_counts()
# # # Verifier si oui ou non il existe des films avec plusieurs commentaire de la meme personne reponse non.
# # # movieUser = data
# # # movieUser = movieUser.drop(['note', 'review_id','name','commentaire'], axis = 1)
# # # movieUserFrequency = movieUser.groupby('movie')['user_id'].value_counts()
# # # movieUserFrequency.rename_axis()

# # import numpy as np
# # import matplotlib.pyplot as plt
# # import scipy
# # import scipy.stats
# # import time
# # bins=500
# # y, x = np.histogram(data['note'], bins=bins, density=True)
# # # Milieu de chaque classe
# # x = (x + np.roll(x, -1))[:-1] / 2.0

# # dist_name = "gamma"

# # # Paramètres de la loi
# # dist = getattr(scipy.stats, dist_name)

# # # Modéliser la loi
# # param = dist.fit(data['note'])

# # loc = param[-2]
# # scale = param[-1]
# # arg = param[:-2]

# # pdf = dist.pdf(x, loc=loc, scale=scale, *arg)

# # plt.figure(figsize=(12,8))
# # plt.plot(x, pdf, label=dist_name, linewidth=3) 

# # plt.legend()
# # plt.savefig('fig/repartition_notes')
# # plt.close()

# # ########################################################
# # y, x = np.histogram(moyenneNoteParMovie, bins=bins, density=True)
# # # Milieu de chaque classe
# # x = (x + np.roll(x, -1))[:-1] / 2.0

# # dist_name = "gamma"

# # # Paramètres de la loi
# # dist = getattr(scipy.stats, dist_name)

# # # Modéliser la loi
# # param = dist.fit(moyenneNoteParMovie)

# # loc = param[-2]
# # scale = param[-1]
# # arg = param[:-2]

# # pdf = dist.pdf(x, loc=loc, scale=scale, *arg)

# # plt.figure(figsize=(12,8))
# # plt.plot(x, pdf, label=dist_name, linewidth=3) 

# # plt.legend()
# # plt.savefig('fig/moyenneNoteParFilm')
# # plt.close()

# # ########################################################
# # y, x = np.histogram(moyenneNoteParUser, bins=bins, density=True)
# # # Milieu de chaque classe
# # x = (x + np.roll(x, -1))[:-1] / 2.0

# # dist_name = "gamma"

# # # Paramètres de la loi
# # dist = getattr(scipy.stats, dist_name)

# # # Modéliser la loi
# # param = dist.fit(moyenneNoteParUser)

# # loc = param[-2]
# # scale = param[-1]
# # arg = param[:-2]

# # pdf = dist.pdf(x, loc=loc, scale=scale, *arg)

# # plt.figure(figsize=(12,8))
# # plt.plot(x, pdf, label=dist_name, linewidth=3) 

# # plt.legend()
# # plt.savefig('fig/moyenneNoteParUser')
# # plt.close()

# # ########################################################
# # y, x = np.histogram(frequenceFilm, bins=bins, density=True)
# # # Milieu de chaque classe
# # x = (x + np.roll(x, -1))[:-1] / 2.0

# # dist_name = "gamma"

# # # Paramètres de la loi
# # dist = getattr(scipy.stats, dist_name)

# # # Modéliser la loi
# # param = dist.fit(frequenceFilm)

# # loc = param[-2]
# # scale = param[-1]
# # arg = param[:-2]

# # pdf = dist.pdf(x, loc=loc, scale=scale, *arg)

# # plt.figure(figsize=(12,8))
# # plt.plot(x, pdf, label=dist_name, linewidth=3) 

# # plt.legend()
# # plt.savefig('fig/frequenceFilm')
# # plt.close()



# # ########################################################
# # y, x = np.histogram(frequenceUtilisateur, bins=bins, density=True)
# # # Milieu de chaque classe
# # x = (x + np.roll(x, -1))[:-1] / 2.0

# # dist_name = "gamma"

# # # Paramètres de la loi
# # dist = getattr(scipy.stats, dist_name)

# # # Modéliser la loi
# # param = dist.fit(frequenceUtilisateur)

# # loc = param[-2]
# # scale = param[-1]
# # arg = param[:-2]

# # pdf = dist.pdf(x, loc=loc, scale=scale, *arg)

# # plt.figure(figsize=(12,8))
# # plt.plot(x, pdf, label=dist_name, linewidth=3) 

# # plt.legend()
# # plt.savefig('fig/frequenceUtilisateur')
# # plt.close()


# # ########################################################
 
# # frequenceUtilisateurPlus10Com = frequenceUtilisateur.drop(frequenceUtilisateur[frequenceUtilisateur < 100].index)

# # y, x = np.histogram(frequenceUtilisateurPlus10Com, bins=bins, density=True)
# # # Milieu de chaque classe
# # x = (x + np.roll(x, -1))[:-1] / 2.0

# # dist_name = "gamma"

# # # Paramètres de la loi
# # dist = getattr(scipy.stats, dist_name)

# # # Modéliser la loi
# # param = dist.fit(frequenceUtilisateurPlus10Com)

# # loc = param[-2]
# # scale = param[-1]
# # arg = param[:-2]

# # pdf = dist.pdf(x, loc=loc, scale=scale, *arg)

# # plt.figure(figsize=(12,8))
# # plt.plot(x, pdf, label=dist_name, linewidth=3) 

# # plt.legend()
# # plt.savefig('fig/frequenceUtilisateurSup10Com')
# # plt.close()