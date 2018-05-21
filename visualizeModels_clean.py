# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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
from sklearn.metrics import confusion_matrix
import matplotlib.ticker as ticker

#Read results of four RNN models
lstm_pre=pd.read_csv('input\\LSTM_1DCNN_pretrain_embedding_noDropout_prediction.csv')
lstm_onfly=pd.read_csv('input\\LSTM_1DCNN_onFly_embedding_noDropout_prediction.csv')
lstm_bi=pd.read_csv('input\\BiDir_LSTM_1DCNN_pretrain_embedding_noDropout_prediction.csv')
gru=pd.read_csv('input\\GRU_1DCNN_pretrain_embedding_noDropout_prediction.csv')

#Plot the model history of the RNN models
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

#Generate a confusion matrix and make the plot
def confused(data1,data2,data3,data4):
    cm1=confusion_matrix(data1.target,data1.predict)
    cm2=confusion_matrix(data2.target,data2.predict)
    cm3=confusion_matrix(data3.target,data3.predict)
    cm4=confusion_matrix(data4.target,data4.predict)
    df_cm1 = pd.DataFrame(cm1)
    df_cm2 = pd.DataFrame(cm2)
    df_cm3 = pd.DataFrame(cm3)
    df_cm4 = pd.DataFrame(cm4)
    df_cm=[]
    df_cm.append(df_cm1)
    df_cm.append(df_cm2)
    df_cm.append(df_cm3)
    df_cm.append(df_cm4)

    sns.set(font_scale=1.4)#for label size
    fig,axn = plt.subplots(2, 2, sharex=True, sharey=True,figsize = (10,7))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    title=['LSTM_onFly','LSTM_pretrain','LSTM_Bidirection','GRU']
    #plt.suptitle('NN-based Models')
    for i, ax in enumerate(axn.flat):
        sns.heatmap(df_cm[i], annot=True,fmt="d",linewidths=.5,ax=ax,
                    cbar=i == 0, cbar_ax=None if i else cbar_ax)# font size
        ax.set_title(title[i])
        if i in [0,2]:
            ax.set_ylabel('True label')
        if i in [2,3]:
            ax.set_xlabel('Predicted label')
    fig.savefig('output\\NN_Model_ConfusionMatrix.png',dpi=600)

confused(lstm_onfly, lstm_pre,lstm_bi,gru) 


'''
Read 100-time repeated results for each model
'''
accuDF=pd.read_csv("output\\Accuracy_nonNN_100Rounds.csv")
f1DF=pd.read_csv("output\\F1Score_nonNN_100Rounds.csv")
precDF=pd.read_csv("output\\Precision_nonNN_100Rounds.csv")
recDF=pd.read_csv("output\\Recall_nonNN_100Rounds.csv")

lstm_onflyDF1=pd.read_csv("input\\LSTM_onFly_embedding_1DCNN_noDropout_metrics100_25.csv")
lstm_onflyDF2=pd.read_csv("input\\LSTM_onFly_embedding_1DCNN_noDropout_metrics100.75.csv")
lstm_onflyDF=pd.concat([lstm_onflyDF1.iloc[:25,],lstm_onflyDF2.iloc[1:,]],ignore_index=True)
lstm_preDF=pd.read_csv("input\\LSTM_1DCNN_pretrain_embedding_noDropout_metrics100.csv")
gruDF=pd.read_csv("input\\GRU_1DCNN_pretrain_embedding_noDropout_metrics100.csv")
lstm_biDF=pd.read_csv("input\\LSTM_Bidirection_1DCNN_pretrain_embedding_noDropout_metrics100.csv")


accuDF['LSTM_onFly']=lstm_onflyDF.accuracy
f1DF['LSTM_onFly']=lstm_onflyDF.f1score
precDF['LSTM_onFly']=lstm_onflyDF.precision
recDF['LSTM_onFly']=lstm_onflyDF.recall

accuDF['LSTM_Pre']=lstm_preDF.accuracy
f1DF['LSTM_Pre']=lstm_preDF.f1score
precDF['LSTM_Pre']=lstm_preDF.precision
recDF['LSTM_Pre']=lstm_preDF.recall

accuDF['LSTM_Bi']=lstm_biDF.accuracy
f1DF['LSTM_Bi']=lstm_biDF.f1score
precDF['LSTM_Bi']=lstm_biDF.precision
recDF['LSTM_Bi']=lstm_biDF.recall

accuDF['GRU']=gruDF.accuracy
f1DF['GRU']=gruDF.f1score
precDF['GRU']=gruDF.precision
recDF['GRU']=gruDF.recall

accuDF=accuDF[['NB','KNN','SVC','RF','LR','LSTM_onFly','LSTM_Pre','LSTM_Bi','GRU']]
f1DF=f1DF[['NB','KNN','SVC','RF','LR','LSTM_onFly','LSTM_Pre','LSTM_Bi','GRU']]
precDF=precDF[['NB','KNN','SVC','RF','LR','LSTM_onFly','LSTM_Pre','LSTM_Bi','GRU']]
recDF=recDF[['NB','KNN','SVC','RF','LR','LSTM_onFly','LSTM_Pre','LSTM_Bi','GRU']]

names=['NB','KNN','SVC','RF','LR','LSTM_onFly','LSTM_Pre','LSTM_Bi','GRU']

accuMean=[]
f1Mean=[]
precisionMean=[]
recallMean=[]

accuSD=[]
f1SD=[]
precisionSD=[]
recallSD=[]

for i in range(accuDF.shape[1]):
    accuSD.append(st.pstdev(accuDF.iloc[:,i]))
    f1SD.append(st.pstdev(f1DF.iloc[:,i]))
    precisionSD.append(st.pstdev(precDF.iloc[:,i]))
    recallSD.append(st.pstdev(recDF.iloc[:,i]))

    accuMean.append(st.mean(accuDF.iloc[:,i]))
    f1Mean.append(st.mean(f1DF.iloc[:,i]))
    precisionMean.append(st.mean(precDF.iloc[:,i]))
    recallMean.append(st.mean(recDF.iloc[:,i]))

df_metric100=pd.DataFrame({'Accuracy_Mean':accuMean, 'Accuracy_SD':accuSD, 'Precision_Mean':precisionMean,'Precision_SD':precisionSD,'Recall_Mean':recallMean,'Recall_SD':recallSD,'F1Score_Mean':f1Mean, 'F1Score_SD':f1SD, 'Model':names})
df_metric100=df_metric100[['Accuracy_Mean', 'Accuracy_SD', 'Precision_Mean','Precision_SD','Recall_Mean','Recall_SD','F1Score_Mean', 'F1Score_SD','Model']]
df_metric100.to_csv('output\\All_Metrics_repeat100.csv',index=False)

#Plot the model comparisons for all the models
def modelComparisonPlots():
    #Read the saved metrics and names
    df_metric100=pd.read_csv('input\\All_Metrics_repeat100.csv')
    names=list(df_metric100.Model)
    accuMean=list(df_metric100.Accuracy_Mean)
    accuSD=list(df_metric100.Accuracy_SD)
    precisionMean=list(df_metric100.Precision_Mean)
    precisionSD=list(df_metric100.Precision_SD)
    recallMean=list(df_metric100.Recall_Mean)
    recallSD=list(df_metric100.Recall_SD)
    f1Mean=list(df_metric100.F1Score_Mean)
    f1SD=list(df_metric100.F1Score_SD)
    # Now switch to a more OO interface to exercise more features.
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    sns.set_style("dark")
    fig.set_dpi(300)
    fig.set_size_inches(17, 9)
    
    #Subplot 1
    ax = axs[0,0]
    ax.tick_params(axis='x', which='major', labelsize=14,rotation=40)
    ax.tick_params(axis='y', which='major', labelsize=14)
    #xs1, ys1, error1 = zip(*sorted(zip(names,accuMean, accuSD)))
    xs1, ys1, error1 = zip(*list(zip(names,accuMean, accuSD)))
    
    ax.plot(xs1,ys1,'go',lw=1)
    ax.errorbar(xs1, ys1, error1, fmt='o',color = 'g',lw=1,capthick = 2,markersize=12)
    '''
    for i, txt in enumerate(error1):
        ax.annotate(round(txt,4),(xs1[i],ys1[i]))
    '''
    ax.set_ylim([0.75,1.0])
    ax.set_title('Accuracy',fontsize=16)
    ax.grid()
    # With 4 subplots, reduce the number of axis ticks to avoid crowding.
    #ax.locator_params(nbins=4)
    
    #Subplot 2
    ax = axs[0,1]
    ax.tick_params(axis='x', which='major', labelsize=14,rotation=40)
    ax.tick_params(axis='y', which='major', labelsize=14)
    xs3, ys3, error3 = zip(*list(zip(names,precisionMean, precisionSD)))
    ax.plot(xs3,ys3,'go',lw=1)
    #ax.bar(xs3,ys3,align='center')
    ax.errorbar(xs3, ys3, error3, fmt='o', color = 'g',lw=1,capthick = 2,markersize=12)
    '''
    for i, txt in enumerate(error3):
        ax.annotate(round(txt,4),(xs3[i],ys3[i]))
    '''
    ax.set_title('Precision',fontsize=16)
    ax.set_ylim([0.75,1.0])
    ax.grid()
    
    
    #Subplot 3
    ax = axs[1,0]
    ax.tick_params(axis='x', which='major', labelsize=14,rotation=40)
    ax.tick_params(axis='y', which='major', labelsize=14)
    xs4, ys4, error4 = zip(*list(zip(names,recallMean, recallSD)))
    ax.plot(xs4,ys4,'go',lw=1)
    #ax.bar(xs4,ys4,align='center')
    ax.errorbar(xs4, ys4, error4, fmt='o', color = 'g',lw=1,capthick = 2,markersize=12)
    '''
    for i, txt in enumerate(error4):
        ax.annotate(round(txt,4),(xs4[i],ys4[i]))
    '''
    ax.set_title('Recall',fontsize=16)
    ax.set_ylim([0.75,1.0])
    ax.grid()
    
    #Subplot 4
    ax = axs[1,1]
    ax.tick_params(axis='x', which='major', labelsize=14,rotation=40)
    ax.tick_params(axis='y', which='major', labelsize=14)
    xs2, ys2, error2 = zip(*list(zip(names,f1Mean, f1SD)))
    ax.plot(xs2,ys2,'go',lw=1,markersize=12)
    #ax.bar(xs2,ys2,align='center')
    ax.errorbar(xs2, ys2, error2, fmt='o',color = 'g',lw=1,capthick = 2,markersize=12)
    '''
    for i, txt in enumerate(error2):
        ax.annotate(round(txt,4),(xs2[i],ys2[i]))    
    '''
    ax.set_title('F1 Score',fontsize=16)
    ax.set_ylim([0.75,1.0])
    ax.grid()
    
    fig.savefig("output\\Performance_comparison_allModel_100round_300dpi_tmp.png")
    #plt.show()
         
      
    
    
    
    
    
    
    