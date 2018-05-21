# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 17:58:19 2017

@author: mguan
"""

'''
Non-Neural Network models for classification: logistic regression, naive bayes,
Support vector machine, K-Nearest neighbors, random forest 
'''

'''
Section 1. Import modules
'''
import os
import sys
import re
import csv
import gc
import codecs
import numpy as np
import pandas as pd
import time
import pickle
import random
import statistics as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
nltk.download('stopwords')
from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import Iterator
from sklearn import metrics, model_selection
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from subprocess import check_output

'''
Section 2: Function definitions
'''
#Load the input data
def readReport(path,sep,encode):
    report=[]
    if encode is not None:
        with codecs.open(path,'r',encode) as f:
            for line in f:
                if line.strip():
                    report.append(line.strip().split(sep))
    else:
        with open(path,'r') as f:
            for line in f:
                if line.strip():
                    report.append(line.strip().split(sep))
    header=['_'.join(x.split(' ')) for x in report[0]]
    df=pd.DataFrame(report,columns=header) #convert the list to pandas dataframe
    report=df[1:]
    report=report.applymap(lambda x: np.nan if isinstance(x,str) and (not x or x.isspace()) else x) #replace empty values with nan
    del df
    f.close()
    return report

#Read the pre-processed clinical report
def readFile(path):
    keymentions=readReport(path,'\t',None)
    data=pd.DataFrame(np.zeros((keymentions.shape[0]+1,4)),columns=['PAT_MRN_ID','NOTE_ID','TEXT','resBack'])
    data=data.iloc[1:,:]
    data['NOTE_ID']=keymentions['NOTE_ID']
    keymentions['secLen1']= pd.to_numeric(keymentions['secLen1'])
    keymentions['secLen2']= pd.to_numeric(keymentions['secLen2'])
    keymentions['resBack']= pd.to_numeric(keymentions['resBack'])
    dups=[398732,546831,599775,627300,931389,1144667,
       1300360,1320429,1337875,1466167,1884201,1916618,
       1997179,2175513,2217257,2269329,2273985,2341738,
       2390204,3015247,3160808,3229929,3288951,3296376,
       3337495,3389723,3404267,3414178]
    dups=[str(s) for s in dups]
    keymentions['TEXT']=np.where(keymentions['secLen1']>1,keymentions['section1'],keymentions['dumSec'])
    data['TEXT']=keymentions['TEXT']
    data['resBack']=keymentions['resBack']
    data['PAT_MRN_ID']=keymentions['PAT_MRN_ID']
    data=data[~data.PAT_MRN_ID.isin(dups)]
    text=data.TEXT.tolist()
    return data, text

#Clean the text data
def text_to_wordClean(text, sep, remove_stopwords=False, stem_words=False, remove_num=False, remove_short=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    text = text.lower().split(sep)
    #optional, remove non-alphabetic words
    #if remove_num:
        #text=[x for x in text if x.isalpha() or x.isalnum()]
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)    
    #optional, remove numbers
    if remove_num:
        text=text.split()
        text=[x for x in text if not x.isnumeric()]
        text=" ".join(text)    
    #optional, remove short words
    if remove_short:
        text=text.split()
        text = [w for w in text if len(w) > 1]
        text=" ".join(text)
    # Return a list of words
    return(text)



#find the length of a list of lists
def lengths(x):
    length=[]
    for i,t in enumerate(x):
        length.append(len(t))
    return length

#define a function to plot model history
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

#A function to split data into train and testing. Version 1.
#It makes sure no individual is overlapping between train and testing
def dataSplit(data,text,label, seed):
    mrn=set()
    for x in data['PAT_MRN_ID'].tolist():
        mrn.add(x)
    mrn=list(mrn)
    print('There are ',len(mrn),' unique individuals.')
    X_trainID, X_testID = train_test_split(mrn,test_size=0.33,random_state=seed)
    data=data.reset_index()
    data_train=data[data['PAT_MRN_ID'].isin(X_trainID)]
    data_test=data[data['PAT_MRN_ID'].isin(X_testID)]

    data_train_idx=np.array(data_train.index)
    data_test_idx=np.array(data_test.index)
    X_train, X_test = text[data_train_idx], text[data_test_idx]
    y_train, y_test = label[data_train_idx], label[data_test_idx]
    return X_train,y_train,X_test,y_test



#Read the pre-defined train and testing sample IDs
trainID=pd.read_csv("input\\trainID.csv")
testID=pd.read_csv("input\\testID.csv")
#Convert numeric data to strings
train_ID=[str(x) for x in trainID['ID']]
test_ID=[str(x) for x in testID['ID']]

#A 2nd version of dataSplit function to split data based pre-defined IDs
def dataSplit_(data,text,label, X_trainID,X_testID):
    '''
    mrn=set()
    for x in data['PAT_MRN_ID'].tolist():
        mrn.add(x)
    mrn=list(mrn)
    print('There are ',len(mrn),' unique individuals.')
    
    X_trainID, X_testID = train_test_split(mrn,test_size=0.33,random_state=seed)
    '''
    data=data.reset_index()
    data_train=data[data['PAT_MRN_ID'].isin(X_trainID)]
    data_test=data[data['PAT_MRN_ID'].isin(X_testID)]

    data_train_idx=np.array(data_train.index)
    data_test_idx=np.array(data_test.index)
    X_train, X_test = text[data_train_idx], text[data_test_idx]
    y_train, y_test = label[data_train_idx], label[data_test_idx]
    return X_train,y_train,X_test,y_test

#Perform tfidf transformation and SVD decomposition on the cleaned text data
def processText(data,text):
    for i,t in enumerate(text):
        text[i]=text_to_wordClean(t, ',', remove_stopwords=True, remove_num=True, remove_short=True)
    label=data['resBack'].tolist()
    label=np.array(label)
    tfidf=TfidfVectorizer(min_df=5,max_df = 0.8, sublinear_tf=True,use_idf=True)
    #tfidf=TfidfVectorizer(min_df=5,max_df = 0.8, sublinear_tf=True,ngram_range = (1,2),use_idf=True,)
    text_tfidf=tfidf.fit_transform(text)
    svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    tfidf_svd=svd.fit_transform(text_tfidf)
    return tfidf_svd, label

#Plot groups using tfidf_svd
def plotSVD(tfidf_svd):
    df_svd=pd.DataFrame(tfidf_svd)
    df_svd.columns=['C'+str(col) for col in df_svd]
    df_svd['Group']=label
    fig = plt.figure(figsize=(10,7))
    ax=sns.pairplot(x_vars="C0", y_vars="C1", data=df_svd, hue="Group", size=7,markers='o',palette="husl")
    ax.set(xlabel='V1', ylabel='V2')
    fig.savefig('output\\Tfidf_svd_C1_C2_by_groups_300dpi.png',dpi=300)

#The function to do parameter search and return the best parameters for each model
def paramSearch(X,y):
    nbParam={'alpha':[0,0.3,0.6,1,2]}
    knnParam={'n_neighbors':[3,4,7,10]}
    svcParam=[{'C':[0.01,0.1,1,10,20,30], 'kernel':['linear']},{'C':[0.01,0.1,1,10,20,30],'gamma':[0.01,0.001,0.0005,0.0001], 'kernel':['rbf']}]
    rfParam={'max_depth': [2,3,4,5,6],'min_samples_split':[2,3,4,5],'min_samples_leaf':[1,3,5]}
    logParam={'C':[0.01,0.1,1,10,20,30]}
    
    gridSearchs=[]
    gridSearchs.append(('NB',BernoulliNB(),nbParam))
    gridSearchs.append(('KNN',KNeighborsClassifier(),knnParam))
    gridSearchs.append(('SVC',SVC(),svcParam))
    gridSearchs.append(('RF',RandomForestClassifier(),rfParam))
    gridSearchs.append(('LR',LogisticRegression(),logParam))

    start=time.time()
    best_params=[]
    for name, model, params in gridSearchs:
        print("current parameter search for {}".format(name))
        clf=GridSearchCV(model,params,cv=3,scoring='accuracy')
        clf.fit(X,y)
        best_params.append((name,clf.best_params_))
        print("Best parameters for {} is {}".format(name,clf.best_params_))
    end=time.time()
    print("total time spent on parameter search is",start-end)
    return best_params

#A function to comapare the performance of 6 models and visulize the mean cross-validation results
def modelSelection(X,y,best_params):

    nbParam = {**best_params[0][1]}
    knnParam={**best_params[1][1]}
    svcParam={**best_params[2][1]}
    #svcParam={'C': 20, 'gamma': 0.01, 'kernel': 'rbf'}
    rfParam={**best_params[3][1]}
    logParam={**best_params[4][1]}
   
    models = []
    models.append(('NB',BernoulliNB(**nbParam)))
    models.append(('KNN',KNeighborsClassifier(**knnParam)))
    models.append(('SVC',SVC(**svcParam)))
    models.append(('RF',RandomForestClassifier(**rfParam)))
    models.append(('LR',LogisticRegression(**logParam)))
    
    cv_results=[]
    names=[]
    meanScore=[]
    for name, model in models:
        start=time.time()
        result=[]
        print("current model is {} with parameters: {}: ".format(name,model))
        result = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        print("{} has -mae {}".format(name, result))
        cv_results.append(result)
        names.append(name)
        meanScore.append(result.mean())
        end=time.time()
        print("Total time is",start-end)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print('\n')
    
    fig=plt.figure(dpi=100)
    plt.plot(meanScore,'g',linewidth=2.0)
    fig.suptitle('ML performance comparison')
    #ax = fig.add_subplot(111)
    plt.xticks(range(6),names,fontsize=10)
    #fig.set_xticklabels(names)
    plt.ylabel('accuracy')
    plt.xlabel('Models')
    plt.show()
    plt.close()

#Fit the models with pre-defined hyperparameters
def fitModel(X,y,X_test,y_test,best_params):  
    nbParam = {**best_params[0][1]}
    knnParam={**best_params[1][1]}
    svcParam={**best_params[2][1]}
    rfParam={**best_params[3][1]}
    logParam={**best_params[4][1]}
   
    models = []
    models.append(('NB',BernoulliNB(**nbParam)))
    models.append(('KNN',KNeighborsClassifier(**knnParam)))
    models.append(('SVC',SVC(**svcParam)))
    models.append(('RF',RandomForestClassifier(**rfParam)))
    models.append(('LR',LogisticRegression(**logParam)))
    
    kfold = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=7).split(X,y))
    fittedModels=[]
    accuScore=[]
    f1score=[]
    precision=[]
    recall=[]
    names=[]
    cm=[]
    #loop through the models
    for i, clf in enumerate(models):
        #loop through 5 folds
        for j, (train_idx, test_idx) in enumerate(kfold):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[test_idx]
            y_val = y[test_idx]
            print ("Fit {} fold {}".format(clf[0], j+1))
            model=clf[1].fit(X_train, y_train)
            print("The validation accuracy is: {}\n".format(model.score(X_val,y_val)))
        accuScore.append(model.score(X_test,y_test))
        print("The accuracy on test data is: {}\n".format(accuScore[i]))
        y_pred=model.predict(X_test)
        f1score.append(f1_score(y_test,y_pred))
        precision.append(precision_score(y_test,y_pred))
        recall.append(recall_score(y_test,y_pred))
        #plotScores(clf[0],y_test,y_pred)
        cm.append(confusion_matrix(y_test, y_pred))
        fittedModels.append((clf[0],model))
        names.append(clf[0])
    return fittedModels, accuScore, names, f1score, precision, recall, cm

#A function to save the fitted model
def saveModel(models,postfix):
    for name, model in models:
        print("save model: ",name)
        filename="input\\"+name+"_model"+postfix+".sav"
        pickle.dump(model,open(filename,'wb'))

#A function to repeat the model for a number of times
def repeatModel(data,tfidf_svd,label,num):
    accuScores100=[]
    f1score100=[]
    precision100=[]
    recall100=[]
    for i in range(num):
        #seed=random.randint(1,500)
        X_train,y_train,X_test,y_test =dataSplit(data,tfidf_svd,label,i)
        best_params=paramSearch(X_train,y_train) 
        saved_models, accuScores, names, f1score, precision, recall=fitModel(X_train,y_train,X_test,y_test, best_params)
        accuScores100.append(accuScores)
        f1score100.append(f1score)
        precision100.append(precision)
        recall100.append(recall)
    return accuScores100, f1score100, precision100, recall100

#Plot ROC AUC
def plotScores(name,y_test,y_pred):  
    print("F1 score of {} is: {}\n".format(name,f1_score(y_test, y_pred)))
    print("Precision score of {} is: {}\n".format(name,precision_score(y_test, y_pred)))
    print("recall score of {} is: {}\n".format(name,recall_score(y_test, y_pred)))
    print("Confusion matrix of {} is: {}\n".format(name,confusion_matrix(y_test, y_pred)))
    
    fpr,tpr,thresholds = roc_curve(y_test,y_pred)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr,tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve of {}'.format(name))
    plt.legend(loc="lower right")
    plt.show()
    plt.close()

#Run single models
def main(data,text,label,postfix):  
    #split data into train and test without overlapping samples between them
    X_train,y_train,X_test,y_test =dataSplit(data,text,label)
    best_params=paramSearch(X_train,y_train) #use 1/10 data for parameter search
    modelSelection(X_train,y_train, best_params)
    saved_models, testScores, names, y_pred=fitModel(X_train,y_train,X_test,y_test, best_params)
    saveModel(saved_models,postfix)
    #loaded_model = pickle.load(open(filename, 'rb'))
    testScores.append(0.85) #add the score of LSTM
    names.append('LSTM')
    plotTrend(testScores, names)

#Plot confusion matrix
def oddPlot(cm1,cm2,cm3,cm4,cm5):
    df_cm1 = pd.DataFrame(cm1)
    df_cm2 = pd.DataFrame(cm2)
    df_cm3 = pd.DataFrame(cm3)
    df_cm4 = pd.DataFrame(cm4)
    df_cm5 = pd.DataFrame(cm5)
    df_cm=[]
    df_cm.append(df_cm1)
    df_cm.append(df_cm2)
    df_cm.append(df_cm3)
    df_cm.append(df_cm4)
    df_cm.append(df_cm5)

    #sns.set(font_scale=1.4)#for label size
    title=names
    fig = plt.figure(figsize=(10,7))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    ax1 = fig.add_subplot(321)
    sns.heatmap(df_cm[0], annot=True,fmt="d",linewidths=.5,ax=ax1,cbar=True,cbar_ax=cbar_ax)
    ax1.set_title(title[0])
    ax1.set_ylabel('True label')
    
    ax2 = fig.add_subplot(323, sharex=ax1)
    sns.heatmap(df_cm[1], annot=True,fmt="d",linewidths=.5,ax=ax2,cbar=False)
    ax2.set_title(title[1])
    ax2.set_ylabel('True label')
    
    ax3 = fig.add_subplot(325, sharex=ax1)
    sns.heatmap(df_cm[2], annot=True,fmt="d",linewidths=.5,ax=ax3,cbar=False)
    ax3.set_title(title[2])
    ax3.set_ylabel('True label')
    ax3.set_xlabel('Predicted label')
    
    ax4 = fig.add_subplot(222)
    sns.heatmap(df_cm[3], annot=True,fmt="d",linewidths=.5,ax=ax4,cbar=False)
    ax4.set_title(title[3])
    
    ax5 = fig.add_subplot(224)
    sns.heatmap(df_cm[4], annot=True,fmt="d",linewidths=.5,ax=ax5,cbar=False)
    ax5.set_title(title[4])
    ax5.set_xlabel('Predicted label')
    fig.savefig('output\\nonNN_Model_ConfusionMatrix_300dpi.png',dpi=300)

'''
Section 3: Perform modeling and results summary
'''
#Read the file with processed clinical text and clean it
data, text = readFile("input\\Visits_6298_KeyMentions.csv")

#Perform tfidf and svd
tfidf_svd, label = processText(data,text)

#Plot groups using tfidf_svd
plotSVD(tfidf_svd)

#main(data,tfidf_svd,label,'SVD')
X_train,y_train,X_test,y_test =dataSplit(data,tfidf_svd,label,39)
#use the IDs from LSTM dataSplit in order to be consistent
X_train,y_train,X_test,y_test =dataSplit_(data,tfidf_svd,label,train_ID,test_ID)
#call the function to get a set of best parameters
best_params=paramSearch(X_train,y_train)
#perform model selection for six models
modelSelection(X_train,y_train, best_params)

'''
name=['NB','KNN','SVC','RF','LR']
param=[{'alpha':0},{'n_neighbors': 7},{'C': 30, 'kernel': 'linear'},
       {'max_depth': 6, 'min_samples_leaf': 5, 'min_samples_split': 5}, {'C': 10}]
best_params=[]
for i in range(5):
    best_params.append((name[i],param[i]))
'''
#Perform model fitting and save the models
saved_models, accuScores, names, f1score, precision, recall,cm=fitModel(X_train,y_train,X_test,y_test, best_params)
oddPlot(cm[0],cm[1],cm[2],cm[3],cm[4])#plot the confusion matrix

#Plot the model performance comaprison for one performance score
def plotTrend(scores,names,metric):
    fig=plt.figure(figsize=(10,7))
    plt.plot(scores,'go--',markersize=12)
    plt.grid(color='grey',linestyle='-',linewidth=0.5,axis='both')
    fig.suptitle('ML performance comparison')
    plt.xticks(range(len(names)),names,fontsize=10)
    plt.ylabel(metric)
    plt.xlabel('Models')
    plt.show()
    plt.close()

'''
NN_names=['LSTM_onFly','LSTM_Pre','LSTM_Bi','GRU']
f1score_NN=[0.87129516849370692,0.88003286770747735,0.90930979133226331,0.89187056037884771]
precision_NN=[0.8509119746233148,0.86931818181818177,0.8782945736434109,0.84834834834834838]
recall_NN=[0.89267886855241263,0.89101497504159732,0.94259567387687193,0.94009983361064897]
accu_NN=[0.84006054490413729,0.8526740665993946,0.88597376387487381,0.86175580221997983]
for m in range(len(NN_names)):
    name.append(NN_names[m])
    f1score.append(f1score_NN[m])
    precision.append(precision_NN[m])
    recall.append(recall_NN[m])
    accuScores.append(accu_NN[m])
'''

#Read the performance scores
df_models=pd.DataFrame({'F1Score':f1score, 'Precision':precision, 'Recall':recall, 'Accuracy': accuScores, 'Model':name})
df_models.to_csv('output\\All_Metrics_FinalModel.csv',index=False)
#Plot the model performance comaprisons
plotTrend(f1score,name,'F1 Score')

metricsAll=[]
metricsAll.append(accuScores)
metricsAll.append(precision)
metricsAll.append(recall)
metricsAll.append(f1score)

'''
fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
fig.set_size_inches(15, 9)
for i, ax in enumerate(axs.flat):
    ax.plot(name,metricsAll[i],'go--',markersize=12)
    ax.grid(color='grey',linestyle='-',linewidth=0.5,axis='both')
    #ax.xticks(range(len(name)),name,fontsize=10)
    plt.tick_params(axis='x',labelsize=10)
'''

df_models=pd.read_csv('input\\All_Metrics_FinalModel.csv')
name=df_models.Model
name=list(name)
accuScore=list(df_models.Accuracy)
precision=list(df_models.Precision)
recall=list(df_models.Recall)
f1score=list(df_models.F1Score)

#Create a 2x2 plot to plot model comparisons for all the metrics
def modelComparisonPlots(name,accuScore,precision,recall,f1score):
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    sns.set_style("dark")
    fig.set_dpi(300)
    fig.set_size_inches(17, 9)
    
    ax = axs[0,0]
    ax.tick_params(axis='x', which='major', labelsize=14,rotation=40)
    ax.tick_params(axis='y', which='major', labelsize=14)
    #xs1, ys1, error1 = zip(*sorted(zip(names,accuMean, accuSD)))
    ax.plot(name, accuScore,'go',lw=1,markersize=12)
    #ax.errorbar(xs1, ys1, error1, fmt='o',color = '#297083',lw=2,capthick = 2)
    '''
    for i, txt in enumerate(error1):
        ax.annotate(round(txt,4),(xs1[i],ys1[i]))
    '''
    ax.set_ylim([0.70,1.00])
    ax.set_title('Accuracy',fontsize=16)
    #ax.set_ylim([0.75,0.95])
    
    # With 4 subplots, reduce the number of axis ticks to avoid crowding.
    ax.locator_params(nbins=4)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.grid()
    
    ax = axs[0,1]
    ax.tick_params(axis='x', which='major', labelsize=14,rotation=40)
    ax.tick_params(axis='y', which='major', labelsize=14)
    #xs2, ys2, error2 = zip(*sorted(zip(names,f1Mean, f1SD)))
    ax.plot(name,precision,'go',lw=1,markersize=12)
    #ax.bar(xs2,ys2,align='center')
    #ax.errorbar(xs2, ys2, error2, fmt='o',color = '#297083',lw=2,capthick = 2)
    '''
    for i, txt in enumerate(error2):
        ax.annotate(round(txt,4),(xs2[i],ys2[i]))    
    '''
    ax.set_title('Precision',fontsize=16)
    ax.set_ylim([0.70,1.00])
    ax.grid()
    
    ax = axs[1,0]
    ax.tick_params(axis='x', which='major', labelsize=14,rotation=40)
    ax.tick_params(axis='y', which='major', labelsize=14)
    #xs3, ys3, error3 = zip(*sorted(zip(names,precisionMean, precisionSD)))
    ax.plot(name,recall,'go',lw=1,markersize=12)
    #ax.bar(xs3,ys3,align='center')
    #ax.errorbar(xs3, ys3, error3, fmt='o', color = '#297083',lw=2,capthick = 2)
    '''
    for i, txt in enumerate(error3):
        ax.annotate(round(txt,4),(xs3[i],ys3[i]))
    '''
    ax.set_title('Recall',fontsize=16)
    ax.set_ylim([0.70,1.00])
    ax.grid()
    
    ax = axs[1,1]
    ax.tick_params(axis='x', which='major', labelsize=14,rotation=40)
    ax.tick_params(axis='y', which='major', labelsize=14)
    #xs4, ys4, error4 = zip(*sorted(zip(names,recallMean, recallSD)))
    ax.plot(name,f1score,'go',lw=1,markersize=12)
    #ax.bar(xs4,ys4,align='center')
    #ax.errorbar(xs4, ys4, error4, fmt='o', color = '#297083',lw=2,capthick = 2)
    '''
    for i, txt in enumerate(error4):
        ax.annotate(round(txt,4),(xs4[i],ys4[i]))
    '''
    ax.set_title('F1 Score',fontsize=16)
    ax.set_ylim([0.70,1.00])
    ax.grid()
    plt.show()
    fig.savefig("output\\All_Metrics_FinalModel_comparison_300dpi_tmp.png",dpi=300)

#Repeat models for 100 times and save them into lists
accuScores100=[]
f1score100=[]
precision100=[]
recall100=[]
accuScores100, f1score100, precision100, recall100 =repeatModel(data,tfidf_svd,label,100)

#save the results of 100 runs into propriate format
def saveResults100(accuScores100,f1score100,precision100,recall100):
    accuScore=list(map(list,zip(*accuScores100)))
    f1Score=list(map(list,zip(*f1score100)))
    precision=list(map(list,zip(*precision100)))
    recall=list(map(list,zip(*recall100)))
    
    accuMean=[]
    f1Mean=[]
    precisionMean=[]
    recallMean=[]
    
    accuSD=[]
    f1SD=[]
    precisionSD=[]
    recallSD=[]
    
    for i in range(5):
        accuSD.append(st.pstdev(accuScore[i]))
        f1SD.append(st.pstdev(f1Score[i]))
        precisionSD.append(st.pstdev(precision[i]))
        recallSD.append(st.pstdev(recall[i]))
    
        accuMean.append(st.mean(accuScore[i]))
        f1Mean.append(st.mean(f1Score[i]))
        precisionMean.append(st.mean(precision[i]))
        recallMean.append(st.mean(recall[i]))
    #names=['NB','KNN','SVC','RF','LR','LSTM']
    accu={'NB':accuScore[0], 'KNN':accuScore[1], 'SVC':accuScore[2], 'RF':accuScore[3], 'LR':accuScore[4]}
    accuDf=pd.DataFrame(data=accu)
    accuDf.to_csv("out\\Accuracy_nonNN_100Rounds.csv",index=False)
    
    f1={'NB':f1Score[0],'KNN':f1Score[1],'SVC':f1Score[2],'RF':f1Score[3],'LR':f1Score[4]}
    f1Df=pd.DataFrame(data=f1)
    f1Df.to_csv("output\\F1Score_nonNN_100Rounds.csv",index=False)
    
    prec={'NB':precision[0], 'KNN':precision[1], 'SVC':precision[2], 'RF':precision[3], 'LR':precision[4]}
    precDf=pd.DataFrame(data=prec)
    precDf.to_csv("output\\Precision_nonNN_100Rounds.csv",index=False)
    
    rec={'NB':recall[0],'KNN':recall[1],'SVC':recall[2],'RF':recall[3],'LR':recall[4]}
    recDf=pd.DataFrame(data=rec)
    recDf.to_csv("output\\Recall_nonNN_100Rounds.csv",index=False)
    








