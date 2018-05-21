# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:00:37 2017

@author: mguan
"""

import pandas as pd
import numpy as np
import re
import string
import nltk
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import codecs #a python module help to encode and decode
'''
Section 1: import and process raw report, gene/therapy list and REDcap labels
'''

#reports=pd.read_table("C:\\Users\\Default\\Desktop\\NLP\\Patient_data101317.txt",sep="\t",engine="python",encoding="utf-8")
geneTherapy=pd.read_csv("input\\GeneTherapyList\\GeneTherapyList.txt",header=None)#read in gene and therapy names
geneTheLocal=pd.read_csv("input\\FoundationMed_genetherapy_local.txt",sep=";")

#store the gene and therapies into a list
GT=[]
for i in range(geneTherapy.shape[0]):
    GT.append(geneTherapy[0][i].strip())
GT=list(set(GT))

GTL=[]
for m in range(1,5):
    for n in range(geneTheLocal.shape[0]):
        GTL.append(geneTheLocal.iloc[n,m])
GTL=list(set(GTL.split(' ')))
test=[x for x in GTL if str(x)!='nan']
test2=[str(x).strip() for x in test]
test3=[str(x).replace('(',' ').replace(')',' ').replace('\t','').strip() for x in test2]
rmlist=[]
rmlist=['insertion','Be','Tumor','duplication','site','and','Determined','loss','Variant','intron','status','Cannot','fusion','splice','Burden','complex']
addlist=['Foundation','guardant','caris']
GTL=list(set(test3))
GTL=[x for x in GTL if x!='_']
GTL=[str(x).split(' ') for x in GTL]
GTL1=[x for x in GTL if len(x)==1]
GTL1=list(set([str(j).strip() for i in GTL1 for j in i]))
GTL1=[x for x in GTL1 if not x=='']
GTL1=[x for x in GTL1 if not x[0].isdigit()] #remove list items beginning with digits
GTL1=[x for x in GTL1 if not x in rmlist]

GTL2=[x for x in GTL if len(x)>1]
GTL2=list(set([str(j).strip() for i in GTL2 for j in i])) #flattern a 2D list
GTL2=[x for x in GTL2 if not x=='']
GTL2=[x for x in GTL2 if not x[0].isdigit()] #remove list items beginning with digits
GTL2=[x for x in GTL2 if not x in rmlist]

GTL3=list(set(GTL1+GTL2))#final gene/therapy list from local dabase
GTL3=list(set(GTL3+addlist))
GTL3.append('K-RAS')
geneThe=pd.DataFrame({'Key':GTL3})
geneThe.to_csv("output\\GeneTherapyList_10262017_local.csv",index=False,header=False)


#read, parse and convert the report to a pandas dataframe
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

reportDF=readReport("input\\Patient_data101317.txt",'\t',None)
reportSub=reportDF[['PAT_MRN_ID','NOTE_ID','NOTE_TEXT']]
reportUniq=reportSub.groupby(["PAT_MRN_ID","NOTE_ID"])["NOTE_TEXT"].apply(lambda x: ' '.join(x.astype(str))).reset_index()

#Process REDcap label data
label=readReport("input\\PrecisionMedicineCli_DATA_LABELS_2017-11-13_1552.txt",'\t','utf16')

'''
Section 2: locate the report mentioning mutation/therapy for the first time for each patient
''' 
#extract word list from the report
#re.split('[,. \t"()]+',test) #alternative to word_tokenize()
rmSymb=['.',',','',';',' ','"','(',')','!',':','**','-','``','·','%','#','•','$','*','&','^','+']
def reportExtract(report):
    text=[]
    for i in range(report.shape[0]):
        tokens=word_tokenize(report.iloc[i,2])
        words=[w for w in tokens if w not in rmSymb]
        text.append(words)
    return text
text=reportExtract(reportUniq)

#search for gene/therapy list in the word list and return the NOTE_ID 
#and PAT_MRN_ID for the first mention


def findFistMention(GT,text):
    ids=set()
    noteId=list()
    for i in range(len(text)):
        if reportUniq.iloc[i,0] not in ids:
            for j in range(len(GT)):
                if text[i].count(GT[j])>0 or text[i].count(GT[j].lower)>0 or text[i].count(GT[j][0].capitalize()+GT[j][1:len(GT[j])].lower()):
                    ids.add(reportUniq.iloc[i,0])
                    noteId.append(reportUniq.iloc[i,1])
                    break
    return ids, noteId

ids,noteId=findFistMention(GTL3,text)

patId=set(reportUniq['PAT_MRN_ID'])
noMentionId=list(patId-set(ids))
idDatesOri=reportDF.iloc[:,[0,3,4]] #extract IDs and dates from original report
idDatesOri['CONTACT_DATE']=idDatesOri['CONTACT_DATE'].apply(lambda x: x.rstrip(' 0:00'))
idDatesOri['CONTACT_DATE']=pd.to_datetime(idDatesOri['CONTACT_DATE'])
#only keep IDs for the lastest visit for each patient
idDateOriLast=idDatesOri.sort_values(['PAT_MRN_ID','CONTACT_DATE'], ascending=False).drop_duplicates('PAT_MRN_ID',keep='last',inplace=False)
lastNoteID=list(idDateOriLast['NOTE_ID'])
#Extact reports that are not mentioned and from the last visit
noMentionReport = reportUniq[reportUniq['PAT_MRN_ID'].isin(noMentionId) & reportUniq['NOTE_ID'].isin(lastNoteID)]
mentionReport=reportUniq[reportUniq['NOTE_ID'].isin(noteId)]
mentionReport.loc[:,'Mention']=1
noMentionReport.loc[:,'Mention']=0
#concatnate two reports with new ordered idex
ReportT0=pd.concat([mentionReport,noMentionReport],ignore_index=True)
ReportT0=ReportT0.iloc[:,[0,1,3,2]]
ReportT0.to_csv("input\\Patient_data101317_report_T0_localList.txt", sep='\t',index=False)

'''
Section 3: Find all the mentions for every unique report
'''
#Version 1. Get a list of unique key words that have been mentioned
def findMentions(GT,text):
    patid=list()
    noteId=list()
    keywords=list()
    for i in range(len(text)):
        patid.append(reportUniq.iloc[i,0])
        noteId.append(reportUniq.iloc[i,1])
        tmp=list()
        for j in range(len(GT)):
            if GT[j].upper() in [x.upper() for x in text[i]]:
                tmp.append(GT[j])
        keywords.append(tmp)
    return ids, noteId, keywords

patid, noteId, keywords = findMentions(GTL3,text)
freq=[len(x) for x in keywords]
tmp=[','.join(x) for x in keywords]
keymentions=reportUniq.iloc[:,:2]

keymentions['mentions']=tmp
keymentions['count']=freq

keymentions=keymentions.iloc[:,[0,1,3,2]]

del tmp
keymentions.to_csv("output\\Visits_6298_KeyMentions.csv",sep='\t',index=False)
count=keymentions['count'].value_counts()
count=count.to_frame()
count['mentions']=count.index
#count.rename(columns={'length':'mention_times'})
sns.barplot(x="mentions", y="count", data=count);
plt.savefig("output\\Visits_6298_KeyMentions_count.png")


#Version 2. Search for all the occurences of key words that have been mentioned in the text
def findMentionsDup(GT,text):
    keywords=list()
    location=list()
    for i in range(len(text)):
        tmp=list()
        loc=list()
        for j in range(len(text[i])):
            if text[i][j].upper() in [x.upper() for x in GT]:
                tmp.append(text[i][j])
                loc.append(j)
        keywords.append(tmp)
        location.append(loc)
    return keywords, location

keywordsDup, locationDup=findMentionsDup(GTL3,text)
freqDup=[len(x) for x in keywordsDup]
tmp=[','.join(x) for x in keywordsDup]

keymentions['mentionsDup']=tmp
keymentions['countDup']=freqDup
del tmp
keymentions=keymentions.iloc[:,[0,1,3,2,5,4]]
loc=list()
for i in range(len(locationDup)):
    tmp=[str(x) for x in locationDup[i]]
    loc.append(tmp)
loc=[','.join(x) for x in loc]

keymentions['locDup']=loc
keymentions.to_csv("output\\Visits_6298_KeyMentions.csv",sep='\t',index=False)

count=keymentions['countDup'].value_counts()
count=count.to_frame()
count['mentions']=count.index
#count.rename(columns={'length':'mention_times'})

rc={'font.size': 12, 'axes.labelsize': 12, 'legend.fontsize': 12.0, 
    'axes.titlesize': 12, 'xtick.labelsize': 1, 'ytick.labelsize': 12}
sns.set(rc=rc)
#sns.set_style("darkgrid", {'axes.labelcolor': '.15','xtick.major.size': 0.0,})
bar=sns.barplot(x="mentions", y="countDup", data=count);
#bar.set(xticklabels=[])
#bar.set(xticks=np.arange(1,156,3))
#bar.tick_params(labelsize=4)

plt.savefig("output\\Visits_6298_KeyMentions_countDup.pdf")



'''
Section 4: Locate the mentions and extract a window of words
'''
def extractText(location,text):
    para1=list()
    para2=list()
    rmSymb=['.',',','',';',' ','"','(',')','!',':','**','-','``']
    for i in range(len(text)):
        if location[i]==[]:
            tmp1=[]
            tmp2=[]
        if len(location[i])==1:
            if location[i][0]-200<0:
                start=0
            else:
                start=location[i][0]-200
            if location[i][0]+200>len(text[i]):
                end=len(text[i])
            else:
                end=location[i][0]+200
            tmp1=[x for x in text[i][start:end] if not x in rmSymb]
            tmp2=[]
        if len(location[i])>1 and (location[i][len(location[i])-1]-location[i][0])<=200:
            if location[i][0]-200<0:
                start=0
            else:
                start=location[i][0]-200
            if location[i][len(location[i])-1]+200>len(text[i]):
                end=len(text[i])
            else:
                end=location[i][len(location[i])-1]+200
                tmp1=[x for x in text[i][start:end] if not x in rmSymb]
                tmp2=[]
        if len(location[i])>1 and (location[i][len(location[i])-1]-location[i][0])>200:
            if location[i][0]-200<0:
                start1=0
            else:
                start1=location[i][0]-200
            end1=location[i][0]+200
            if location[i][len(location[i])-1]+200>len(text[i]):
                end2=len(text[i])
            else:
                end2=location[i][len(location[i])-1]+200
            start2=location[i][len(location[i])-1]-200
            tmp1=[x for x in text[i][start1:end1] if not x in rmSymb]
            tmp2=[x for x in text[i][start2:end2] if not x in rmSymb]
            
        para1.append(tmp1)
        para2.append(tmp2)
    return para1, para2

text1, text2=extractText(locationDup,text)
doc1=list()
doc2=list()
for i in range(len(text1)):
    tmp1=[x.replace('·','').replace('?','') for x in text1[i] if not x in ['·','%','#','•','$','*','&','^','+']]
    tmp2=[x.replace('·','').replace('?','') for x in text2[i] if not x in ['·','%','#','•','$','*','&','^','+']]
    doc1.append(tmp1)
    doc2.append(tmp2)


doc1=[','.join(x) for x in doc1]
doc2=[','.join(x) for x in doc2]

keymentions['section1']=doc1
keymentions['section2']=doc2
keymentions.to_csv("output\\Visits_6298_KeyMentions.csv",sep='\t',index=False)


'''
Output section
'''
reportUniq.to_csv("output\\Patient_data101317_aggregates.txt", sep='\t',index=False)

'''
Read keymention file and add text total and length
'''

infile=open("output\\Visits_6298_KeyMentions.csv",'r')
content=[]
for i,line in enumerate(infile):
    if line.strip():
        if line.startswith('\"'):
            content[len(content)-1]=content[len(content)-1]+line.strip('\"').strip('\t').strip('\n').split('\t')
        else:
            content.append(line.strip('\n').split('\t'))
    
df=pd.DataFrame(content,columns=content[0])
keymentions=df[1:]
del df

secLen1=[]
secLen2=[]
for i in range(keymentions.shape[0]):
    secLen1.append(len(keymentions.iloc[i,7].split(',')))
    secLen2.append(len(keymentions.iloc[i,8].split(',')))

#calculate average length of two sections
int(sum([x for x in secLen1 if not x==1])/len([x for x in secLen1 if not x==1])) #289
int(sum([x for x in secLen2 if not x==1])/len([x for x in secLen2 if not x==1])) #296

keymentions['secLen1']=secLen1
keymentions['secLen2']=secLen2

text=reportExtract(reportUniq)
totalLen=[len(text[i]) for i in range(len(text))]
totalText=[','.join(x) for x in text]
keymentions['totalText']=totalText
keymentions['totalLen']=totalLen

#calculate average proportion of 2 sections
289/int(sum(totalLen)/len(totalLen)) #11.85%
296/int(sum(totalLen)/len(totalLen)) #12.14%

dumSec=[]
for i in range(len(text)):
    if(len(text[i]))<289:
        dumSec.append(text[i])
    else:
        dumSec.append(text[i][:289])
keymentions['dumSec']=[','.join(x) for x in dumSec]

'''
Combine keymentions and label
'''
keymentions=readReport("output\\Visits_6298_KeyMentions.csv",'\t',None)


'''
Remove 28 samples with duplicate reports

'''
