# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 14:35:09 2018

@author: mguan
"""

'''
https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
'''
'''
Section 1: Import the python modules to be used
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
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

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
from sklearn.metrics import log_loss
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from subprocess import check_output

from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder

import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import Iterator
from keras.layers import Dense, Input, Flatten, LSTM, GRU, Embedding, Dropout, Activation, Bidirectional
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras import optimizers
from keras.models import load_model
from keras import metrics
from keras.datasets import imdb
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.wrappers.scikit_learn import KerasClassifier

import scipy.sparse as ssp

'''
Section 2: Define all the functions
'''
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

'''
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
'''

#Clean the text data and perform tokenization
def textTokenize(data,text):
    for i,t in enumerate(text):
        text[i]=text_to_wordClean(t,',',remove_num=True, remove_short=True)
    
    label=data['resBack'].tolist()
    label=np.array(label)
    
    t = Tokenizer()
    t.fit_on_texts(text) #training phase
    word_index = t.word_index #get a map of word index
    sequences = t.texts_to_sequences(text)
    max_len=max(lengths(sequences))
    print('Found %s unique tokens' % len(word_index))
    text_tok=pad_sequences(sequences, maxlen=max_len)
    return text_tok, label, word_index, max_len

#Split the data into train and testing, 
#it makes sure not sample overlaps between train and testing
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
    return X_train,y_train,X_test,y_test,X_trainID,X_testID

#Perform wordEmbedding
def wordEmbed(word_index):   
    pretrain = gensim.models.Word2Vec.load("input\\wordEmbedding_w2v_allReports")
    #convert pretrained word embedding to a dictionary
    embedding_index=dict()
    for i in range(len(pretrain.wv.vocab)):
        word=pretrain.wv.index2word[i]
        if word is not None:
            embedding_index[word]=pretrain.wv[word]  
    #extract word embedding for train and test data
    vocab_size=len(word_index)+1
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, i in word_index.items():
    	embedding_vector = embedding_index.get(word)
    	if embedding_vector is not None:
    		embedding_matrix[i] = embedding_vector
    return embedding_matrix

#define the LSTM baseline model for parameter tuning
def LSTM_baseline():
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 100, weights=[embedding_matrix],input_length=max_len,trainable=False))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#Wrapper the LSTM model with a sklearn-style function
def NN_Grid(X_train, y_train, param):
    start = time.time()
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                              verbose=1, mode='auto')
    callbacks_list = [earlystop]
    values=param.get(list(param.keys())[0])
    accuAll=[]
    for size in values:
        print("For ",list(param.keys())[0]," ",size,"\n")
        kfold = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=41).split(X_train,y_train))
        accu=[]
        for i,(train, test) in enumerate(kfold):
            model = Sequential()
            model.add(Embedding(len(word_index)+1, 100, weights=[embedding_matrix],input_length=max_len,trainable=False))
            model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(LSTM(100))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            print("Fit fold", i+1," **************************************************************************")
            model_info=model.fit(X_train[train], y_train[train], epochs=100, batch_size=size, validation_data=(X_train[test], y_train[test]),
                                   callbacks=callbacks_list, verbose=1)
            print("Performance plot of fold {}:".format(i+1))
            scores=model.evaluate(X_test, y_test, verbose=1)
            print("Accuracy on test data: %.5f" % scores[1])
            accu.append(scores[1])
            del model
        accuAll.append(accu)
        print("The average accuracy of param ",size," is ",np.mean(accu))
    end = time.time()
    elapsed = end - start
    print("Elapsed time: %.2f" % elapsed)
    return accuAll

#LSTM with word embedding trained on the fly
def LSTM_onFly(X_train,y_train,X_test,y_test,word_index, max_len, seed):
    start = time.time()
    
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                              verbose=1, mode='auto')
    callbacks_list = [earlystop]
    model = Sequential()
    model.add(Embedding(len(word_index)+1, max_len, input_length=max_len))
    #model5.add(Dropout(0.2))
    model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    kfold = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train,y_train))
    model_infos=[]
    for i,(train, test) in enumerate(kfold):
        print("Fit fold", i+1," **************************************************************************")
        model_info=model.fit(X_train[train], y_train[train], epochs=100, batch_size=64, validation_data=(X_train[test], y_train[test]),
                               callbacks=callbacks_list, verbose=1)
        print("Performance plot of fold {}:".format(i+1))
        plot_model_history(model_info)
        model_infos.append(model_info)
    
    # Final evaluation of the model
    y_pred=model.predict(X_test,verbose=1)
    y_pred_coded=np.where(y_pred>0.5,1,0)
    y_pred_coded=y_pred_coded.flatten()
    
    metric=[]
    metric.append(['f1score',f1_score(y_test,y_pred_coded)])
    metric.append(['precision',precision_score(y_test,y_pred_coded)])
    metric.append(['recall',recall_score(y_test,y_pred_coded)])
    metric.append(['accuracy',accuracy_score(y_test,y_pred_coded)])
    
    end = time.time()
    elapsed = end - start
    print("Elapsed time: %.2f" % elapsed)
    model.save("output\\LSTM_1DCNN_onFly_embedding_noDropout.hdf5")
    pred=np.stack((y_test,y_pred_coded),axis=-1)
    pred=pd.DataFrame(data=pred,columns=['target','predict'])
    pred.to_csv("output\\LSTM_1DCNN_onFly_embedding_noDropout_prediction.csv",index=None)

    return y_pred, metric, model_infos

#simplifed LSTM with word embedding trained on the fly
def LSTM_onFly_simp(X_train,y_train,X_test,y_test,word_index, max_len, seed):
    #start = time.time()
    
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                              verbose=1, mode='auto')
    callbacks_list = [earlystop]
    model = Sequential()
    model.add(Embedding(len(word_index)+1, max_len, input_length=max_len))
    #model5.add(Dropout(0.2))
    model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())

    kfold = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train,y_train))
    for i,(train, test) in enumerate(kfold):
        #print("Fit fold", i+1," **************************************************************************")
        model_info=model.fit(X_train[train], y_train[train], epochs=100, batch_size=64, validation_data=(X_train[test], y_train[test]),
                               callbacks=callbacks_list, verbose=0)
        #print("Performance plot of fold {}:".format(i+1))
        #plot_model_history(model_info)

    # Final evaluation of the model
    y_pred=model.predict(X_test,verbose=1)
    y_pred_coded=np.where(y_pred>0.5,1,0)
    y_pred_coded=y_pred_coded.flatten()
    
    metric=[]
    metric.append(['f1score',f1_score(y_test,y_pred_coded)])
    metric.append(['precision',precision_score(y_test,y_pred_coded)])
    metric.append(['recall',recall_score(y_test,y_pred_coded)])
    metric.append(['accuracy',accuracy_score(y_test,y_pred_coded)])
    
    del model
    '''
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test data: %.5f" % scores[1])
    end = time.time()
    elapsed = end - start
    print("Elapsed time: %.2f" % elapsed)
    model.save("C:\\Users\\mguan\\OneDrive - Wake Forest Baptist Medical Center\\NLP\\code\\LSTM_1DCNN_onFly_embedding_noDropout.hdf5")
    pred=np.stack((y_test,y_pred_coded),axis=-1)
    pred=pd.DataFrame(data=pred,columns=['target','predict'])
    pred.to_csv("C:\\Users\\mguan\\OneDrive - Wake Forest Baptist Medical Center\\NLP\\output\\LSTM_1DCNN_onFly_embedding_noDropout_prediction.csv",index=None)
    '''
    return metric

#Basic LSTM model with pre-trained word mebedding
def LSTM_model(X_train,y_train,X_test,y_test,embedding_matrix, word_index, max_len, seed):

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                              verbose=1, mode='auto')
    callbacks_list = [earlystop]
    
    start = time.time()
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 100, weights=[embedding_matrix],input_length=max_len,trainable=False))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    kfold = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train,y_train))
    model_infos=[]
    for i,(train, test) in enumerate(kfold):
        print("Fit fold", i+1," **************************************************************************")
        model_info=model.fit(X_train[train], y_train[train], epochs=100, batch_size=64, validation_data=(X_train[test], y_train[test]),
                               callbacks=callbacks_list, verbose=1)
        print("Performance plot of fold {}:".format(i+1))
        plot_model_history(model_info)
        model_infos.append(model_info)
    
    # Final evaluation of the model
    y_pred=model.predict(X_test,verbose=1)
    y_pred_coded=np.where(y_pred>0.5,1,0)
    y_pred_coded=y_pred_coded.flatten()
    
    metric=[]
    metric.append(['f1score',f1_score(y_test,y_pred_coded)])
    metric.append(['precision',precision_score(y_test,y_pred_coded)])
    metric.append(['recall',recall_score(y_test,y_pred_coded)])
    metric.append(['accuracy',accuracy_score(y_test,y_pred_coded)])
    
    end = time.time()
    elapsed = end - start
    print("Elapsed time: %.2f" % elapsed)
    model.save("output\\LSTM_1DCNN_pretrain_embedding_noDropout.hdf5")
    pred=np.stack((y_test,y_pred_coded),axis=-1)
    pred=pd.DataFrame(data=pred,columns=['target','predict'])
    pred.to_csv("output\\LSTM_1DCNN_pretrain_embedding_noDropout_prediction.csv",index=None)

    return y_pred, metric, model_infos

#Simplified LSTM model with pre-trained word mebedding
def LSTM_model_simp(X_train,y_train,X_test,y_test,embedding_matrix, word_index, max_len, seed):

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                              verbose=1, mode='auto')
    callbacks_list = [earlystop]
    
    #start = time.time()
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 100, weights=[embedding_matrix],input_length=max_len,trainable=False))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    kfold = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train,y_train))
    for i,(train, test) in enumerate(kfold):
        #print("Fit fold", i+1," **************************************************************************")
        model_info=model.fit(X_train[train], y_train[train], epochs=100, batch_size=64, validation_data=(X_train[test], y_train[test]),
                               callbacks=callbacks_list, verbose=0)
        #print("Performance plot of fold {}:".format(i+1))
        #plot_model_history(model_info)
    
    # Final evaluation of the model
    y_pred=model.predict(X_test,verbose=1)
    y_pred_coded=np.where(y_pred>0.5,1,0)
    y_pred_coded=y_pred_coded.flatten()
    
    metric=[]
    metric.append(['f1score',f1_score(y_test,y_pred_coded)])
    metric.append(['precision',precision_score(y_test,y_pred_coded)])
    metric.append(['recall',recall_score(y_test,y_pred_coded)])
    metric.append(['accuracy',accuracy_score(y_test,y_pred_coded)])
    del model
    '''
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test data: %.5f" % scores[1])
    end = time.time()
    elapsed = end - start
    print("Elapsed time: %.2f" % elapsed)
    model.save("C:\\Users\\mguan\\OneDrive - Wake Forest Baptist Medical Center\\NLP\\code\\LSTM_1DCNN_pretrain_embedding_noDropout.hdf5")
    pred=np.stack((y_test,y_pred_coded),axis=-1)
    pred=pd.DataFrame(data=pred,columns=['target','predict'])
    pred.to_csv("C:\\Users\\mguan\\OneDrive - Wake Forest Baptist Medical Center\\NLP\\output\\LSTM_1DCNN_pretrain_embedding_noDropout_prediction.csv",index=None)
    '''
    return metric

#GRU model with pre-trained word embedding
def GRU_model(X_train,y_train,X_test,y_test,embedding_matrix, word_index, max_len, seed):

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                              verbose=1, mode='auto')
    callbacks_list = [earlystop]
    
    start = time.time()
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 100, weights=[embedding_matrix],input_length=max_len,trainable=False))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GRU(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    kfold = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train,y_train))
    model_infos=[]
    for i,(train, test) in enumerate(kfold):
        print("Fit fold", i+1," **************************************************************************")
        model_info=model.fit(X_train[train], y_train[train], epochs=100, batch_size=64, validation_data=(X_train[test], y_train[test]),
                               callbacks=callbacks_list, verbose=1)
        print("Performance plot of fold {}:".format(i+1))
        plot_model_history(model_info)
        model_infos.append(model_info)
    
    # Final evaluation of the model
    y_pred=model.predict(X_test,verbose=1)
    y_pred_coded=np.where(y_pred>0.5,1,0)
    y_pred_coded=y_pred_coded.flatten()
    
    metric=[]
    metric.append(['f1score',f1_score(y_test,y_pred_coded)])
    metric.append(['precision',precision_score(y_test,y_pred_coded)])
    metric.append(['recall',recall_score(y_test,y_pred_coded)])
    metric.append(['accuracy',accuracy_score(y_test,y_pred_coded)])
    
    end = time.time()
    elapsed = end - start
    print("Elapsed time: %.2f" % elapsed)
    model.save("output\\GRU_1DCNN_pretrain_embedding_noDropout.hdf5")
    pred=np.stack((y_test,y_pred_coded),axis=-1)
    pred=pd.DataFrame(data=pred,columns=['target','predict'])
    pred.to_csv("output\\GRU_1DCNN_pretrain_embedding_noDropout_prediction.csv",index=None)

    return y_pred, metric, model_infos

#Simplified GRU model with pre-trained word embedding
def GRU_model_simp(X_train,y_train,X_test,y_test,embedding_matrix, word_index, max_len, seed):

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                              verbose=1, mode='auto')
    callbacks_list = [earlystop]
    
    #start = time.time()
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 100, weights=[embedding_matrix],input_length=max_len,trainable=False))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GRU(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    kfold = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train,y_train))
    for i,(train, test) in enumerate(kfold):
        #print("Fit fold", i+1," **************************************************************************")
        model_info=model.fit(X_train[train], y_train[train], epochs=100, batch_size=64, validation_data=(X_train[test], y_train[test]),
                               callbacks=callbacks_list, verbose=0)
        #print("Performance plot of fold {}:".format(i+1))
        #plot_model_history(model_info)
    
    # Final evaluation of the model
    y_pred=model.predict(X_test,verbose=1)
    y_pred_coded=np.where(y_pred>0.5,1,0)
    y_pred_coded=y_pred_coded.flatten()
    
    metric=[]
    metric.append(['f1score',f1_score(y_test,y_pred_coded)])
    metric.append(['precision',precision_score(y_test,y_pred_coded)])
    metric.append(['recall',recall_score(y_test,y_pred_coded)])
    metric.append(['accuracy',accuracy_score(y_test,y_pred_coded)])
    del model
    '''
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test data: %.5f" % scores[1])
    end = time.time()
    elapsed = end - start
    print("Elapsed time: %.2f" % elapsed)
    model.save("C:\\Users\\mguan\\OneDrive - Wake Forest Baptist Medical Center\\NLP\\code\\GRU_1DCNN_pretrain_embedding_noDropout.hdf5")
    pred=np.stack((y_test,y_pred_coded),axis=-1)
    pred=pd.DataFrame(data=pred,columns=['target','predict'])
    pred.to_csv("C:\\Users\\mguan\\OneDrive - Wake Forest Baptist Medical Center\\NLP\\output\\GRU_1DCNN_pretrain_embedding_noDropout_prediction.csv",index=None)
    '''
    return metric

#Bi-directional LSTM with pre-trained word embedding
def LSTM_Bidir(X_train,y_train,X_test,y_test,embedding_matrix, word_index, max_len, seed):

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                              verbose=1, mode='auto')
    callbacks_list = [earlystop]
    
    start = time.time()
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 100, weights=[embedding_matrix],input_length=max_len,trainable=False))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(LSTM(100))
    model.add(Bidirectional(LSTM(100),merge_mode='concat', weights=None))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    kfold = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train,y_train))
    model_infos=[]
    for i,(train, test) in enumerate(kfold):
        print("Fit fold", i+1," **************************************************************************")
        model_info=model.fit(X_train[train], y_train[train], epochs=100, batch_size=64, validation_data=(X_train[test], y_train[test]),
                               callbacks=callbacks_list, verbose=1)
        print("Performance plot of fold {}:".format(i+1))
        plot_model_history(model_info)
        model_infos.append(model_info)
    
    # Final evaluation of the model
    y_pred=model.predict(X_test,verbose=1)
    y_pred_coded=np.where(y_pred>0.5,1,0)
    y_pred_coded=y_pred_coded.flatten()
    
    metric=[]
    metric.append(['f1score',f1_score(y_test,y_pred_coded)])
    metric.append(['precision',precision_score(y_test,y_pred_coded)])
    metric.append(['recall',recall_score(y_test,y_pred_coded)])
    metric.append(['accuracy',accuracy_score(y_test,y_pred_coded)])

    end = time.time()
    elapsed = end - start
    print("Elapsed time: %.2f" % elapsed)
    model.save("output\\BiDir_LSTM_1DCNN_pretrain_embedding_noDropout.hdf5")
    pred=np.stack((y_test,y_pred_coded),axis=-1)
    pred=pd.DataFrame(data=pred,columns=['target','predict'])
    pred.to_csv("output\\BiDir_LSTM_1DCNN_pretrain_embedding_noDropout_prediction.csv",index=None)

    return y_pred, metric, model_infos

#Simplified Bi-directional LSTM with pre-trained word embedding
def LSTM_Bidir_simp(X_train,y_train,X_test,y_test,embedding_matrix, word_index, max_len, seed):

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                              verbose=1, mode='auto')
    callbacks_list = [earlystop]
    
    #start = time.time()
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 100, weights=[embedding_matrix],input_length=max_len,trainable=False))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(LSTM(100))
    model.add(Bidirectional(LSTM(100),merge_mode='concat', weights=None))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    kfold = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train,y_train))
    for i,(train, test) in enumerate(kfold):
        #print("Fit fold", i+1," **************************************************************************")
        model_info=model.fit(X_train[train], y_train[train], epochs=100, batch_size=64, validation_data=(X_train[test], y_train[test]),
                               callbacks=callbacks_list, verbose=0)
        #print("Performance plot of fold {}:".format(i+1))
        #plot_model_history(model_info)
    
    # Final evaluation of the model
    y_pred=model.predict(X_test,verbose=1)
    y_pred_coded=np.where(y_pred>0.5,1,0)
    y_pred_coded=y_pred_coded.flatten()
    
    metric=[]
    metric.append(['f1score',f1_score(y_test,y_pred_coded)])
    metric.append(['precision',precision_score(y_test,y_pred_coded)])
    metric.append(['recall',recall_score(y_test,y_pred_coded)])
    metric.append(['accuracy',accuracy_score(y_test,y_pred_coded)])
    
    '''
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test data: %.5f" % scores[1])
    end = time.time()
    elapsed = end - start
    print("Elapsed time: %.2f" % elapsed)
    model.save("C:\\Users\\mguan\\OneDrive - Wake Forest Baptist Medical Center\\NLP\\code\\BiDir_LSTM_1DCNN_pretrain_embedding_noDropout.hdf5")
    pred=np.stack((y_test,y_pred_coded),axis=-1)
    pred=pd.DataFrame(data=pred,columns=['target','predict'])
    pred.to_csv("C:\\Users\\mguan\\OneDrive - Wake Forest Baptist Medical Center\\NLP\\output\\BiDir_LSTM_1DCNN_pretrain_embedding_noDropout_prediction.csv",index=None)
    '''
    return metric

#2nd version dataSplit function to split train and testing based on pre-defined IDs
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

#Repeat the neural network models for a number of times
def repeatModel(data, text_tok, label, word_index, max_len, num, model):
    start = time.time()
    accuScores100=[]
    f1score100=[]
    precision100=[]
    recall100=[]
    if model!=LSTM_onFly_simp:
        embedding_matrix=wordEmbed(word_index)
        for i in range(num):
            print("Repeat round ",i+1,"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            metric=[]
            X_train,y_train,X_test,y_test =dataSplit(data,text_tok,label,i)
            metric=model(X_train,y_train,X_test,y_test,embedding_matrix, word_index, max_len, i)
            f1score100.append(metric[0][1])
            precision100.append(metric[1][1])
            recall100.append(metric[2][1])
            accuScores100.append(metric[3][1])
    elif model==LSTM_onFly_simp:
        for i in range(num):
            print("Repeat round ",i+1,"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            metric=[]
            X_train,y_train,X_test,y_test =dataSplit(data,text_tok,label,i)
            metric=model(X_train,y_train,X_test,y_test, word_index, max_len, i)
            f1score100.append(metric[0][1])
            precision100.append(metric[1][1])
            recall100.append(metric[2][1])
            accuScores100.append(metric[3][1])
            if (i/5).is_integer():
                df_tmp={'f1score':f1score100, 'precision':precision100, 'recall':recall100, 'accuracy':accuScores100}
                df_tmp=pd.DataFrame(data=df_tmp)
                df_tmp.to_csv("C:\\Users\\mguan\\OneDrive - Wake Forest Baptist Medical Center\\NLP\\output\\LSTM_onFly_embedding_1DCNN_noDropout_metrics100_"+str(i)+".csv",index=None)
    df={'f1score':f1score100, 'precision':precision100, 'recall':recall100, 'accuracy':accuScores100}
    df=pd.DataFrame(data=df)
    if model==LSTM_onFly_simp:
        df.to_csv("output\\LSTM_onFly_embedding_1DCNN_noDropout_metrics100.csv",index=None)
    elif model==LSTM_Bidir_simp:
        df.to_csv("output\\LSTM_Bidirection_1DCNN_pretrain_embedding_noDropout_metrics100.csv",index=None)
    elif model==LSTM_model_simp:
        df.to_csv("output\\LSTM_baseline_1DCNN_pretrain_embedding_noDropout_metrics100.csv",index=None)
    elif model==GRU_model_simp:
        df.to_csv("output\\GRU_1DCNN_pretrain_embedding_noDropout_metrics100.csv",index=None)
    end = time.time()
    elapsed = end - start
    print("Elapsed time: %.2f" % elapsed)
    return df
'''
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
'''
#define a function to plot model history
def plot_model_history_4(info1,info2,info3,info4):
    
    fig, axs = plt.subplots(2,1,figsize=(9,18))
    sns.set()
    
    # summarize history for accuracy
    axs[0].plot(range(1,len(info1['acc'])+1),info1['acc'],color='r')
    axs[0].plot(range(1,len(info1['val_acc'])+1),info1['val_acc'],color='r',ls='--')
    axs[0].plot(range(1,len(info2['acc'])+1),info2['acc'],color='b')
    axs[0].plot(range(1,len(info2['val_acc'])+1),info2['val_acc'],color='b',ls='--')
    axs[0].plot(range(1,len(info3['acc'])+1),info3['acc'],color='g')
    axs[0].plot(range(1,len(info3['val_acc'])+1),info3['val_acc'],color='g',ls='--')
    axs[0].plot(range(1,len(info4['acc'])+1),info4['acc'],color='c')
    axs[0].plot(range(1,len(info4['val_acc'])+1),info4['val_acc'],color='c',ls='--')
    
    axs[0].set_title('Model Accuracy',fontsize=18)
    axs[0].set_ylabel('Accuracy',fontsize=16)
    axs[0].set_xlabel('Epoch',fontsize=16)
    axs[0].set_xticks(np.arange(1,16),15/10)
    axs[0].tick_params(labelsize=14)
    #axs[0].legend(['train_lstm_onfly', 'val_lstm_onfly',
               #'train_lstm_pre','val_lstm_pre','train_gru','val_gru','train_lstm_bi','val_lstm_bi'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # summarize history for loss
    axs[1].plot(range(1,len(info1['loss'])+1),info1['loss'],color='r')
    axs[1].plot(range(1,len(info1['val_loss'])+1),info1['val_loss'],color='r',ls='--')
    axs[1].plot(range(1,len(info2['loss'])+1),info2['loss'],color='b')
    axs[1].plot(range(1,len(info2['val_loss'])+1),info2['val_loss'],color='b',ls='--')    
    axs[1].plot(range(1,len(info3['loss'])+1),info3['loss'],color='g')
    axs[1].plot(range(1,len(info3['val_loss'])+1),info3['val_loss'],color='g',ls='--')
    axs[1].plot(range(1,len(info4['loss'])+1),info4['loss'],color='c')
    axs[1].plot(range(1,len(info4['val_loss'])+1),info4['val_loss'],color='c',ls='--')
    
    axs[1].set_title('Model Loss',fontsize=18)
    axs[1].set_ylabel('Loss',fontsize=16)
    axs[1].set_xlabel('Epoch',fontsize=16)
    #axs[1].set_xticks(np.arange(1,len(info1['loss'])+1),len(info1['loss'])/10)
    axs[1].set_xticks(np.arange(1,16),15/10)
    axs[1].tick_params(labelsize=14)
    axs[0].legend(['train_lstm_onfly', 'val_lstm_onfly',
               'train_lstm_pre','val_lstm_pre','train_gru','val_gru','train_lstm_bi','val_lstm_bi'], 
                fontsize=14,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('output\\NN_model_training_15Epoch_300dpi_tmp.png',dpi=150)

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

#Save the fitted models
def saveModel(models,postfix):
    for name, model in models:
        print("save model: ",name)
        filename="output\\"+name+"_model"+postfix+".sav"
        pickle.dump(model,open(filename,'wb'))

#Plot performance score for one single model
def plotTrend(scores,names,metric):
    fig=plt.figure(dpi=100)
    plt.plot(scores,'g',linewidth=2.0)
    plt.grid(color='grey',linestyle='-',linewidth=0.5,axis='both')
    fig.suptitle('ML performance comparison')
    plt.xticks(range(len(names)+1),names,fontsize=10)
    plt.ylabel(metric)
    plt.xlabel('Models')
    plt.show()
    plt.close()

#Calculate mean and std for the performance scores
def calMeanSd():
    accuSD.append(st.pstdev(accuScore[i]))
    f1SD.append(st.pstdev(f1Score[i]))
    precisionSD.append(st.pstdev(precision[i]))
    recallSD.append(st.pstdev(recall[i]))

    accuMean.append(st.mean(accuScore[i]))
    f1Mean.append(st.mean(f1Score[i]))
    precisionMean.append(st.mean(precision[i]))
    recallMean.append(st.mean(recall[i]))

'''
Section 3: Perform model modeling and result summary
'''
#Read clinical reports and clean them 
data, text = readFile("input\\Visits_6298_KeyMentions.csv")
#tokenize the text data
text_tok, label, word_index, max_len=textTokenize(data,text)

#tfidf_svd, label = processText(data,text)

#split dataset into training and test
X_train,y_train,X_test,y_test,trainID,testID =dataSplit(data,text_tok,label,39)
#Create IDs for train and testing datasets
DF_trainID=pd.DataFrame({'ID':trainID})
DF_testID=pd.DataFrame({'ID':testID})
DF_trainID.to_csv("output\\trainID.csv",index=None)
DF_testID.to_csv("output\\testID.csv",index=None)
#Read train and testing dataset IDs
trainID=pd.read_csv("output\\trainID.csv")
testID=pd.read_csv("output\\testID.csv")
train_ID=[str(x) for x in trainID['ID']]
test_ID=[str(x) for x in testID['ID']]

#Create pretrained word embedding
embedding_matrix=wordEmbed(word_index)

'''
Parameter tuning for LSTM_pre
'''
batch_size = [20, 40, 60, 80, 100]
epochs = [100, 150, 200, 250]
param={'batch_size':[22,32,42,52,62,72]}
filters=32, kernel_size=3
model.add(Dropout(dropout_rate))
scores=NN_Grid(X_train,y_train,param)

'''
LSTM model with word embedding trained on the fly
'''
y_pred_onfly, metric_onfly, model_info_onfly=LSTM_onFly(X_train,y_train,X_test,y_test, word_index, max_len, 33)

#metric_onFly=LSTM_onFly_simp(X_train,y_train,X_test,y_test,word_index, max_len, 33)

'''
LSTM model with pretrained word embedding
'''

#Run LSTM model using pretrained word embedding
y_pred_base, metric_base, model_info_base=LSTM_model(X_train,y_train,X_test,y_test,embedding_matrix, word_index, max_len, 33)

#Run GRU using pretrained word embedding
y_pred_gru, metric_gru, model_info_gru=GRU_model(X_train,y_train,X_test,y_test,embedding_matrix, word_index, max_len, 33)

#Run BiDirectional LSTM using pretrained word embedding
y_pred_bi, metric_bi, model_info_bi=LSTM_Bidir(X_train,y_train,X_test,y_test,embedding_matrix, word_index, max_len, 33)

#Extract training info for the NN models
info_onfly=pd.DataFrame(model_info_onfly[0].history)
info_base=pd.DataFrame(model_info_base[0].history)
info_gru=pd.DataFrame(model_info_gru[0].history)
info_bi={'acc':[0.7494,0.8051,0.8310,0.8678,0.9002,0.9117,0.9174,0.9347,0.9488,0.9491,0.9638,0.9674,0.9776],
         'loss':[0.5323,0.4348,0.3845,0.3226,0.2577,0.2246,0.2016,0.1745,0.1424,0.1370,0.1023,0.0955,0.0752],
         'val_acc':[0.7826,0.8197,0.8619,0.8708,0.9015,0.9003,0.8862,0.8964,0.8760,0.9118,0.8875,0.9003,0.9143],
         'val_loss':[0.4631,0.4076,0.3614,0.3371,0.2681,0.2870,0.2923,0.2503,0.3123,0.2646,0.2703,0.2533,0.2762]}
info_bi=pd.DataFrame(info_bi)
#Save the accuracy change for all the four models
info_onfly.to_csv("output\\LSTM_onfly_model_history_1st_fold.csv",index=None)
info_base.to_csv("output\\LSTM_pre_model_history_1st_fold.csv",index=None)
info_gru.to_csv("output\\GRU_model_history_1st_fold.csv",index=None)
info_bi.to_csv("output\\LSTM_bidirection_model_history_1st_fold.csv",index=None)

info_onfly=pd.read_csv("output\\LSTM_onfly_model_history_1st_fold.csv")
info_base=pd.read_csv("output\\LSTM_pre_model_history_1st_fold.csv")
info_gru=pd.read_csv("output\\GRU_model_history_1st_fold.csv")
info_bi=pd.read_csv("output\\LSTM_bidirection_model_history_1st_fold.csv")

plot_model_history_4(info_onfly,info_base,info_gru,info_bi)

'''
#Repeat LSTM model for 100 times
'''
df=repeatModel(data, text_tok, label, word_index, max_len, 100, LSTM_model_simp) #39.69611111111111 hours

df_onFly=repeatModel(data, text_tok, label, word_index, max_len, 100, LSTM_onFly_simp)

df_bi=repeatModel(data, text_tok, label, word_index, max_len, 100, LSTM_Bidir_simp)

df_gru=repeatModel(data, text_tok, label, word_index, max_len, 100, GRU_model_simp)









