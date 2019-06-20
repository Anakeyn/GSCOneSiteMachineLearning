# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 18:39:18 2019

@author: Pierre
"""
##########################################################################
# GSCOneSiteMachineLearning
# Auteur : Pierre Rouarch - Licence GPL 3
# Test modele machine learning pour un site (données Google Search Console API)
###################################################################
# On démarre ici 
###################################################################
#Chargement des bibliothèques générales utiles
import numpy as np #pour les vecteurs et tableaux notamment
import matplotlib.pyplot as plt  #pour les graphiques
#import scipy as sp  #pour l'analyse statistique
import pandas as pd  #pour les Dataframes ou tableaux de données
import seaborn as sns #graphiques étendues
#import math #notamment pour sqrt()
import os

from urllib.parse import urlparse #pour parser les urls
import nltk # Pour le text mining
# Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#Matrice de confusion
from sklearn.metrics import confusion_matrix
#pour les scores
from sklearn.metrics import f1_score
#from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
import gc #pour vider la memoire


print(os.getcwd())  #verif
#mon répertoire sur ma machine - nécessaire quand on fait tourner le programme 
#par morceaux dans Spyder.
#myPath = "C:/Users/Pierre/MyPath"
#os.chdir(myPath) #modification du path
#print(os.getcwd()) #verif


#Relecture des données Pages/Expressions/Positions ############
dfGSC = pd.read_json("dfGSC-MAI.json")
dfGSC.query
dfGSC.info() # on a 514 enregistrements.



############################################
# Calcul de la somme des tf*idf

#somme des TF*IDF pour chaque colonne de tokens calculé à la main (plus lent)
def getSumTFIDFfromDFColumnManual(myDFColumn) :
    tf = myDFColumn.apply(lambda x: pd.Series(x).value_counts(normalize=True)).fillna(0)
    idf = pd.Series([np.log10(float(myDFColumn[0])/len([x for x in myDFColumn.values if token in x])) for token in tf.columns])
    ##################################################
    idf.index = tf.columns
    tfidf = tf.copy()
    for col in tfidf.columns:
        tfidf[col] = tfidf[col]*idf[col]
    return np.sum(tfidf, axis=1)

#somme des TF*IDF pour chaque colonne de tokens calculé avec TfidfVectorizer
def getSumTFIDFfromDFColumn(myDFColumn) :
    from sklearn.feature_extraction.text import TfidfVectorizer
    corpus = myDFColumn.apply(' '.join)
    vectorizer = TfidfVectorizer(norm=None)
    X = vectorizer.fit_transform(corpus)
    return np.sum(X.toarray(), axis=1)



##########################################################################
#  On va enrichir les données que l'on a pour créer de nouvelles variables
# Explicatives pour créer un modele de Machine Learning
##########################################################################
 
#########################################################################
# Détermination de variables techniques et construites à partir des 
# autres variables
#########################################################################
#Creation des Groupes (variable à expliquer) :
# finalement on prend une classification Binaire
dfGSC.loc[dfGSC['position'] <= 10.5, 'group'] = 1
dfGSC.loc[dfGSC['position'] > 10.5, 'group'] = 0



#création de nouvelles variables explicatives

#Creation de variables d'url à partir de page
dfGSC['webSite'] = dfGSC['page'].apply(lambda x: '{uri.scheme}://{uri.netloc}/'.format(uri=urlparse(x)))
dfGSC['uriScheme'] = dfGSC['page'].apply(lambda x: '{uri.scheme}'.format(uri=urlparse(x)))
dfGSC['uriNetLoc'] = dfGSC['page'].apply(lambda x: '{uri.netloc}'.format(uri=urlparse(x)))
dfGSC['uriPath'] = dfGSC['page'].apply(lambda x: '{uri.path}'.format(uri=urlparse(x)))

#Est-ce que le site est en https ?  ici ne sert pas vraiment car on n'a qu'un seul site 
#en https.

dfGSC.loc[dfGSC['uriScheme']== 'https', 'isHttps'] = 1
dfGSC.loc[dfGSC['uriScheme'] != 'https', 'isHttps'] = 0
dfGSC.info()

#"Pseudo niveau" dans l'arborescence calculé au nombre de / -2
dfGSC['level'] = dfGSC['page'].str[:-1].str.count('/')-2


#définition du tokeniser pour séparation des mots
tokenizer = nltk.RegexpTokenizer(r'\w+')  #définition du tokeniser pour séparation des mots

#on va décompter les mots de la requête dans le nom du site et l'url complète
#on vire les accents 
queryNoAccent= dfGSC['query'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
#on tokenize la requete sans accents
dfGSC['tokensQueryNoAccent'] = queryNoAccent.apply(tokenizer.tokenize) #séparation des mots pour la reqête


#Page
dfGSC['lenPage']=dfGSC['page'].apply(len) #taille de l'url complète en charactères
dfGSC['tokensPage'] =  dfGSC['page'].apply(tokenizer.tokenize)   #séparation des mots pour l'url entier
dfGSC['lenTokensPage']=dfGSC['tokensPage'].apply(len) #longueur de l'url en mots
#mots de la requete dans   Page
dfGSC['lenTokensQueryInPage'] = dfGSC.apply(lambda x : len(set(x['tokensQueryNoAccent']).intersection(x['tokensPage'])),axis=1)
#total des fréquences des mots dans  Page
dfGSC['lenTokensQueryInPageFrequency'] = dfGSC.apply(lambda x : x['lenTokensQueryInPage']/x['lenTokensPage'],axis=1)
#SumTFIDF
dfGSC['sumTFIDFPage'] = getSumTFIDFfromDFColumn(dfGSC['tokensPage'])
dfGSC['sumTFIDFPageFrequency'] = dfGSC.apply(lambda x : x['sumTFIDFPage']/(x['lenTokensPage']+0.01),axis=1) 


#WebSite    
dfGSC['lenWebSite']=dfGSC['webSite'].apply(len) #taille de l'url complète en charactères
dfGSC['tokensWebSite'] =  dfGSC['webSite'].apply(tokenizer.tokenize)   #séparation des mots pour l'url entier
dfGSC['lenTokensWebSite']=dfGSC['tokensWebSite'].apply(len) #longueur de l'url en mots
#mots de la requete dans   WebSite
dfGSC['lenTokensQueryInWebSite'] = dfGSC.apply(lambda x : len(set(x['tokensQueryNoAccent']).intersection(x['tokensWebSite'])),axis=1)
#total des fréquences des mots dans   WebSite
dfGSC['lenTokensQueryInWebSiteFrequency'] = dfGSC.apply(lambda x : x['lenTokensQueryInWebSite']/x['lenTokensWebSite'],axis=1)
#SumTFIDF
dfGSC['sumTFIDFWebSite'] = getSumTFIDFfromDFColumn(dfGSC['tokensWebSite'])    
dfGSC['sumTFIDFWebSiteFrequency'] = dfGSC.apply(lambda x : x['sumTFIDFWebSite']/(x['lenTokensWebSite']+0.01),axis=1) 
 
#Path   
dfGSC['lenPath']=dfGSC['uriPath'].apply(len) #taille de l'url complète en charactères
dfGSC['tokensPath'] =  dfGSC['uriPath'].apply(tokenizer.tokenize)   #séparation des mots pour l'url entier
dfGSC['lenTokensPath']=dfGSC['tokensPath'].apply(len) #longueur de l'url en mots
#mots de la requete dans   Path
dfGSC['lenTokensQueryInPath'] = dfGSC.apply(lambda x : len(set(x['tokensQueryNoAccent']).intersection(x['tokensPath'])),axis=1)
#total des fréquences des mots dans   Path
#!Risque de division par zero on fait une boucle avec un if
dfGSC['lenTokensQueryInPathFrequency']=0
for i in range(0, len(dfGSC)) :
    if dfGSC.loc[i,'lenTokensPath'] > 0 :
        dfGSC.loc[i,'lenTokensQueryInPathFrequency'] =dfGSC.loc[i,'lenTokensQueryInPath']/dfGSC.loc[i,'lenTokensPath']
#SumTFIDF
dfGSC['sumTFIDFPath'] = getSumTFIDFfromDFColumn(dfGSC['tokensPath'])   
dfGSC['sumTFIDFPathFrequency'] = dfGSC.apply(lambda x : x['sumTFIDFPath']/(x['lenTokensPath']+0.01),axis=1) 


#Sauvegarde pour la suite 
dfGSC.to_json("dfGSC1-MAI.json")  



#############################################################################
# Premier test de Machine Learning sur Environnement Interne et variables
# uniquement créés à partir des urls et des mots clés.
#############################################################################
#Préparation des données 

#on libere de la mémoire
del dfGSC
gc.collect()


#Relecture ############
dfGSC1 = pd.read_json("dfGSC1-MAI.json")
dfGSC1.query
dfGSC1.info() # 514  enregistrements.


#on choisit nos variables explicatives
X =  dfGSC1[['isHttps', 'level', 
             'lenWebSite',   'lenTokensWebSite',  'lenTokensQueryInWebSiteFrequency' , 'sumTFIDFWebSiteFrequency',             
             'lenPath',   'lenTokensPath',  'lenTokensQueryInPathFrequency' , 'sumTFIDFPathFrequency']]  #variables explicatives



y =  dfGSC1['group']  #variable à expliquer

#on va scaler
scaler = StandardScaler()
scaler.fit(X)


X_Scaled = pd.DataFrame(scaler.transform(X.values), columns=X.columns, index=X.index)
X_Scaled.info()

X_train, X_test, y_train, y_test = train_test_split(X_Scaled,y, random_state=0)





#############################################################################
#Méthode des k-NN
nMax=10
myTrainScore =  np.zeros(shape=nMax)
myTestScore = np.zeros(shape=nMax)
myF1Score = np.zeros(shape=nMax)  #score F1


for n in range(1,nMax) :
    knn = KNeighborsClassifier(n_neighbors=n) 
    knn.fit(X_train, y_train) 
    myTrainScore[n]=knn.score(X_train,y_train)
    print("Training set score: {:.3f}".format(knn.score(X_train,y_train))) #
    myTestScore[n]=knn.score(X_test,y_test)
    print("Test set score: {:.4f}".format(knn.score(X_test,y_test))) #
    y_pred=knn.predict(X_Scaled)
    print("F1-Score  : {:.4f}".format(f1_score(y, y_pred, average ='weighted')))
    myF1Score[n] = f1_score(y, y_pred, average ='weighted')
    myCM = confusion_matrix(y, y_pred)
    print("Accuracy calculated with CM : {:.4f}".format((myCM[0][0] +  myCM[1][1] ) / y.size))
    print("Accuracy by sklearn : {:.4f}".format(accuracy_score(y, y_pred)))  #idem que précédent


#Graphique train score vs test score vs F1 Score
sns.set()  #paramètres esthétiques ressemble à ggplot par défaut.
fig, ax = plt.subplots()  #un seul plot
sns.lineplot(x=np.arange(1,nMax), y=myTrainScore[1:nMax])
sns.lineplot(x=np.arange(1,nMax), y=myTestScore[1:nMax], color='red')
sns.lineplot(x=np.arange(1,nMax), y=myF1Score[1:nMax], color='yellow')
fig.suptitle('On obtient déjà des résultats satisfaisants avec peu de variables.', fontsize=14, fontweight='bold')
ax.set(xlabel='n neighbors', ylabel='Train (bleu) / Test (rouge) / F1 (jaune)',
       title="Attention il s'agit ici d'une étude sur un seul site")
fig.text(.3,-.06,"Classification Knn - un seul site - Position  dans 2 groupes  \n vs variables construites en fonction des n voisins", 
         fontsize=9)
#plt.show()
fig.savefig("GSC1-KNN-Classifier-2groups.png", bbox_inches="tight", dpi=600)


#on choist le le premier n_neighbor ou myF1Score est le plus grand
#à vérifier toutefois en regardant la courbe.
indices = np.where(myF1Score == np.amax(myF1Score))
n_neighbor =  indices[0][0]
n_neighbor
knn = KNeighborsClassifier(n_neighbors=n_neighbor) 
knn.fit(X_train, y_train) 
print("N neighbor="+str(n_neighbor))
print("Training set score: {:.3f}".format(knn.score(X_train,y_train))) #
print("Test set score: {:.4f}".format(knn.score(X_test,y_test))) #
y_pred=knn.predict(X_Scaled)
print("F1-Score  : {:.4f}".format(f1_score(y, y_pred, average ='weighted')))
#f1Score retenu pour knn 0.8944

#par curiosité regardons la distribution des pages dans les groupes 
sns.set()  #paramètres esthétiques ressemble à ggplot par défaut.
fig, ax = plt.subplots()  #un seul plot
sns.countplot(dfGSC1['group'], order=reversed(dfGSC1['group'].value_counts().index))
fig.suptitle('Il y a beaucoup de pages dans le groupe des pages hors du top 10 (0).', fontsize=14, fontweight='bold')
ax.set(xlabel='groupe', ylabel='Nombre de Pages',
       title="Il serait souhaitable d'avoir des groupes plus équilibrés.")
fig.text(.3,-.06,"Distribution des pages/positions dans les 2 groupes.", 
         fontsize=9)
#plt.show()
fig.savefig("GSC1-Distribution-2groups.png", bbox_inches="tight", dpi=600)


##################################################################################         
#Classification linéaire 1 :   Régression Logistique
#on faire varier C : inverse of regularization strength; must be a positive float. 
#Like in support vector machines, smaller values specify stronger regularization.
logreg = LogisticRegression(solver='lbfgs').fit(X_train,y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train,y_train)))  
print("Test set score: {:.3f}".format(logreg.score(X_test,y_test))) 
y_pred=logreg.predict(X_Scaled)
print("F1-Score  : {:.4f}".format(f1_score(y, y_pred, average ='weighted')))
#pas mieux que knn 0.7714

logreg100 = LogisticRegression(C=100, solver='lbfgs').fit(X_train,y_train)
print("Training set score: {:.3f}".format(logreg100.score(X_train,y_train)))  
print("Test set score: {:.3f}".format(logreg100.score(X_test,y_test)))  
y_pred=logreg100.predict(X_Scaled)
print("F1-Score  : {:.4f}".format(f1_score(y, y_pred, average ='weighted')))
# 0.7744 pas mieux que knn mais mieux que  logreg standard
logreg001 = LogisticRegression(C=0.01, solver='lbfgs').fit(X_train,y_train)
print("Training set score: {:.3f}".format(logreg001.score(X_train,y_train)))  
print("Test set score: {:.3f}".format(logreg001.score(X_test,y_test))) 
y_pred=logreg001.predict(X_Scaled)
print("F1-Score  : {:.4f}".format(f1_score(y, y_pred, average ='weighted'))) 
# 0.7567 pas mieux que knn ni logreg standard 

#################################################################################
#Classification linéaire 2 :  machine à vecteurs supports linéaire (linear SVC).
LinSVC = LinearSVC(max_iter=10000).fit(X_train,y_train)
print("Training set score: {:.3f}".format(LinSVC.score(X_train,y_train)))  
print("Test set score: {:.3f}".format(LinSVC.score(X_test,y_test))) 
y_pred=LinSVC.predict(X_Scaled)
print("F1-Score  : {:.4f}".format(f1_score(y, y_pred, average ='weighted'))) 
#0.7714 pas mieux que KNN egal à loreg standard



#######################################################################
# Affichage de l'importance des variables pour logreg100
#######################################################################
signed_feature_importance = logreg100.coef_[0] #pour afficher le sens 
feature_importance = abs(logreg100.coef_[0])  #pous classer par importance
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5  #pour l'affichage au milieu de la barre

sns.set()  #paramètres esthétiques ressemble à ggplot par défaut.
fig, ax = plt.subplots()  #un seul plot
ax.barh(pos, signed_feature_importance[sorted_idx], align='center')
ax.set_yticks(pos)
ax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=8)
fig.suptitle("La longueur du chemin en caractères influence positivement le modèle  \n alors que c'est l'inverse en nombre de mots.\nLes variables propres au site n'influent pas car nous sommes sur 1 seul site.", fontsize=10)
ax.set(xlabel='Importance Relative des variables')
fig.text(.3,-.06,"Régression Logistique C=100 - 1 seul site - Importance des variables", 
         fontsize=9)
fig.savefig("GSC1-Importance-Variables-C100-2groups.png", bbox_inches="tight", dpi=600)




##########################################################################
# MERCI pour votre attention !
##########################################################################
#on reste dans l'IDE
#if __name__ == '__main__':
#  main()









    
