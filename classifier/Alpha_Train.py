import pandas as pd
import numpy as np
import glob

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import re
import csv
import sys
import time
from itertools import zip_longest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
import nltk
import re
import string
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support as score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
'''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
'''

import os
#writef = open("Output-Train_Set.txt", "w")
#sys.stdout = writef

train_set = r'C:\\Users\\Predator\\Documents\\Document-Classification\\classifier\\Train_Set.csv'
test_set = r'C:\\Users\\Predator\\Documents\\Document-Classification\\classifier\\Test_Set.csv'
train_df = pd.read_csv(train_set, header=0, encoding="utf-8")
print(train_df)
test_df = pd.read_csv(test_set, header=0, encoding="utf-8")




import pandas as pd
import numpy as np
import nltk
import re
import string
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support as score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('stopwords')
stopwords.words('english')

'''
#Removing punctuations from entire dataset
punc_set = string.punctuation
punc_set

#Function for removing punctions
def remove_punc(text):
    clean = ''.join([x.lower() for x in text if x not in punc_set])
    return clean

#Applying the 'remove_punc' function to entire dataset
df['no_punc'] = df['sentence'].apply(lambda z:remove_punc(z))

#Function for Tokenizing entire data for representing every word as datapoint
def tokenize(text):
    tokens = re.split("\W+",text)
    return tokens

#Applying the 'tokenize' function to entire dataset
df['tokenized_Data'] = df['no_punc'].apply(lambda z:tokenize(z))

#Importing stopwords from NLTK Library to remove stopwords now that we have tokenized it
stopwords = nltk.corpus.stopwords.words('english')

#Function for removing stopwords from single row
def remove_stopwords(tokenized_words):
    Ligit_text=[word for word in tokenized_words if word not in stopwords]
    return Ligit_text

#Applying the function 'remove_stopwords' from the entire dataset
df["no_stop"] = df["tokenized_Data"].apply(lambda z:remove_stopwords(z))

#Importing 'WordNetLemmatizer' as lemmatizing function to find lemma's of words
wnl = nltk.wordnet.WordNetLemmatizer()

#Function for lemmatizing the tokenzied text
def lemmatizing(tokenized_text):
    lemma = [wnl.lemmatize(word) for word in tokenized_text]
    return lemma

#Applying the 'lemmatizing' function to entire dataset
df['lemmatized'] = df['no_stop'].apply(lambda z:lemmatizing(z))

# #Importing the 'SnowballStemmer' and declaring variable 'sno' to save the stemmer in.
# #This Stemmer gives slightly better results as compared to 'PorterStemmer'
# sno = nltk.SnowballStemmer('english')

# #Function for applying stemming to find stem roots of all words
# def stemming(tokenized_text):
#     text= [sno.stem(word) for word in tokenized_text]
#     return text

# #Applying the 'stemming' function to entire dataset
# data['ss_stemmed'] = data['lemmatized'].apply(lambda z:stemming(z))


# ps = nltk.PorterStemmer()

# def stemming(tokenized_text):
#     text= [ps.stem(word) for word in tokenized_text]
#     return text

# data['ps_stemmed'] = data['lemmatized'].apply(lambda z:stemming(z))

#This step is done here because, the 'lemmatized' column is a list of tokenized words and when we apply vectorization
#techniques such as count vectorizer or TFIDF, they require string input. Hence convert all tokenzied words to string
df['lemmatized'] = [" ".join(review) for review in df['lemmatized'].values]
'''

pos = len(train_df[train_df['label'] == 1])
print("Positive:", pos)
neg = len(train_df[train_df['label'] == 0])
print("Negative:", neg)
#print(train_df)
total = train_df.shape[0]
print("Total", total)

train10 = round(train_df.shape[0] * .05) + 1
print("10%", train10)
train20 = round(train_df.shape[0] * .15) + 1
print("20%", train20)
train30 = round(train_df.shape[0] * .25) + 1
print("30%", train30)
train40 = round(train_df.shape[0] * .35) + 1
print("40%", train40)
train50 = round(train_df.shape[0] * .45) + 1
print("50%", train50)
train60 = round(train_df.shape[0] * .55) + 1
print("60%", train60)
train70 = round(train_df.shape[0] * .65) + 1
print("70%", train70)
train80 = round(train_df.shape[0] * .75)  + 1
print("80%", train80)
train90 = round(train_df.shape[0] * .85) + 1
print("90%", train90)
train100 = round(train_df.shape[0] * .95) + 1
print("100%", train100)

remain1 = 1 - train10
print(remain1)
remain2 = 1 - train20
remain3 = 1 - train30
remain4 = 1 - train40
remain5 = 1 - train50
remain6 = 1 - train60
remain7 = 1 - train70
remain8 = 1 - train80
remain9 = 1 - train90
remain10 = 1 - train100


por10 = train_df.iloc[0:train10, :]
print(por10)
por20 = train_df.iloc[0:train20, :]
print(por20)
por30 = train_df.iloc[0:train30, :]
print(por30)
por40 = train_df.iloc[0:train40, :]
print(por40)
por50 = train_df.iloc[0:train50, :]
print(por50)
por60 = train_df.iloc[0:train60, :]
print(por40)
por70 = train_df.iloc[0:train70, :]
por80 = train_df.iloc[0:train80, :]
por90 = train_df.iloc[0:train90, :]
por100 = train_df.iloc[0:train100, :]

rem1 = train_df.iloc[train10:total, :]
rem2 = train_df.iloc[train20:total, :]
rem3 = train_df.iloc[train30:total, :]
rem4 = train_df.iloc[train40:total, :]
rem5 = train_df.iloc[train50:total, :]
rem6 = train_df.iloc[train60:total, :]
rem7 = train_df.iloc[train70:total, :]
rem8 = train_df.iloc[train80:total, :]
rem9 = train_df.iloc[train90:total, :]
rem10 = train_df.iloc[train100:total, :]

sen10 = train_df.iloc[0:train10, 0].astype('U')
print(sen10)
sen20 = train_df.iloc[0:train20, 0].astype('U')
print(sen20)
sen30 = train_df.iloc[0:train30, 0].astype('U')
print(sen30)
sen40 = train_df.iloc[0:train40, 0].astype('U')
print(sen40)
sen50 = train_df.iloc[0:train50, 0].astype('U')
print(sen50)
sen60 = train_df.iloc[0:train60, 0].astype('U')
print(sen60)
sen70 = train_df.iloc[0:train70, 0].astype('U')
print(sen70)
sen80 = train_df.iloc[0:train80, 0].astype('U')
print(sen80)
sen90 = train_df.iloc[0:train90, 0].astype('U')
print(sen90)
sen100 = train_df.iloc[0:train100, 0].astype('U')
print(sen100)


senrem1 = train_df.iloc[train10:total, 0].astype('U')
senrem2 = train_df.iloc[train20:total, 0].astype('U')
senrem3 = train_df.iloc[train30:total, 0].astype('U')
senrem4 = train_df.iloc[train40:total, 0].astype('U')
senrem5 = train_df.iloc[train50:total, 0].astype('U')
senrem6 = train_df.iloc[train60:total, 0].astype('U')
senrem7 = train_df.iloc[train70:total, 0].astype('U')
senrem8 = train_df.iloc[train80:total, 0].astype('U')
senrem9 = train_df.iloc[train90:total, 0].astype('U')
senrem10 = train_df.iloc[train100:total, 0].astype('U')


tfidf_vect = TfidfVectorizer(ngram_range=(1,2))
#count_vect = CountVectorizer()

tfidf_train = tfidf_vect.fit_transform(sen10.values.astype('U'))
tfidf_test = tfidf_vect.fit_transform(senrem1.values.astype('U'))
print(tfidf_train.shape)
print(tfidf_test.shape)

x_train = tfidf_train
x_test = tfidf_test


x_train = tfidf_train.fit_transform(sen10)
x_train1 = tfidf_train.fit_transform(sen20)
x_train2 = tfidf_train.fit_transform(sen30)
x_train3 = tfidf_train.fit_transform(sen40)
x_train4 = tfidf_train.fit_transform(sen50)
x_train5 = tfidf_train.fit_transform(sen60)
x_train6 = tfidf_train.fit_transform(sen70)
x_train7 = tfidf_train.fit_transform(sen80)
x_train8 = tfidf_train.fit_transform(sen90)
x_train9 = tfidf_train.fit_transform(sen100)

x_test = tfidf_train.fit_transform(senrem1)
x_test1 = tfidf_train.fit_transform(senrem2)
x_test2 = tfidf_train.fit_transform(senrem3)
x_test3 = tfidf_train.fit_transform(senrem4)
x_test4 = tfidf_train.fit_transform(senrem5)
x_test5 = tfidf_train.fit_transform(senrem6)
x_test6 = tfidf_train.fit_transform(senrem7)
x_test7 = tfidf_train.fit_transform(senrem8)
x_test8 = tfidf_train.fit_transform(senrem9)
x_test9 = tfidf_train.fit_transform(senrem10)

y_train = por10["label"]
y_train1 = por20["label"]
y_train2 = por30["label"]
y_train3 = por40["label"]
y_train4 = por50["label"]
y_train5 = por60["label"]
y_train6 = por70["label"]
y_train7 = por80["label"]
y_train8 = por90["label"]
y_train9 = por100["label"]

y_test = rem1["label"]
y_test1 = rem2["label"]
y_test2 = rem3["label"]
y_test3 = rem4["label"]
y_test4 = rem5["label"]
y_test5 = rem6["label"]
y_test6 = rem7["label"]
y_test7 = rem8["label"]
y_test8 = rem9["label"]
y_test9 = rem10["label"]
'''
sen_train = [x_train, x_train1, x_train2, x_train3, x_train4, x_train5, x_train6, x_train7, x_train8, x_train9]
print(sen_train)
sen_test = [x_test, x_test1, x_test2, x_test3, x_test4, x_test5, x_test6, x_test7, x_test8, x_test9]
print(sen_test)
train_label = [y_train, y_train1, y_train2, y_train3, y_train4, y_train5, y_train6, y_train7, y_train8, y_train9]
test_label = [y_test, y_test1, y_test2, y_test3, y_test4, y_test5, y_test6, y_test7, y_test8, y_test9]
'''
'''
print(x_train)
y_train = por10["label"]
print(y_train)
x_test = tfidf_vect.fit_transform(rem10["sentence"].astype('U'))
print(x_test)
y_test = por10["label"]
print(y_test)
'''
'''
pos_df = df.loc[df['label'] == 1]
print(pos_df)
neg_df = df.loc[df['label'] == 0]
print(neg_df)
'''
'''
x_train =
x_test =
y_train =
y_test =
'''
'''
# Splitting data into smaller dataframes for the purpose of Training and Testing
x1 = neg_df.iloc[0:12304, 0]
x2 = pos_df.iloc[0:12304, 0]
y1 = neg_df.iloc[0:12304, 1]
y2 = pos_df.iloc[0:12304, 1]
# x_seg = df.iloc[0:5,5]
# y_seg = df.iloc[0:5,1]
print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)
'''
'''
x = train_df['sentence'].values
y = train_df['label'].values
print(x.shape)
print(y.shape)
'''
'''
# Declaring and applying TFIDF functions to train and test data
tfidf_vect = TfidfVectorizer(ngram_range=(1, 2))
tfidf_train = tfidf_vect.fit_transform(x1.values.astype('U'))
tfidf_test = tfidf_vect.transform(x2.values.astype('U'))
print(tfidf_train.shape)
print(tfidf_test.shape)
# tfidf_train.toarray()
'''
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

probs_lr_scol = []
f1_lr_scol = []
rocauc_lr_scol = []
recall_lr_scol = []
precision_lr_scol = []
accuracy_lr_scol = []

probs_dt_scol = []
f1_dt_scol = []
rocauc_dt_scol = []
recall_dt_scol = []
precision_dt_scol = []
accuracy_dt_scol = []

probs_nb_scol = []
f1_nb_scol = []
rocauc_nb_scol = []
recall_nb_scol = []
precision_nb_scol = []
accuracy_nb_scol = []

probs_xg_scol = []
f1_xg_scol = []
rocauc_xg_scol = []
recall_xg_scol = []
precision_xg_scol = []
accuracy_xg_scol = []

probs_rf_scol = []
f1_rf_scol = []
rocauc_rf_scol = []
recall_rf_scol = []
precision_rf_scol = []
accuracy_rf_scol = []

#tfidf_vect = TfidfVectorizer()
#x_tfidf = tfidf_vect.fit_transform(df["sentence"].astype('U'))

#x_train, x_test, y_train, y_test = train_test_split(x_tfidf, df["label"], test_size=0.05, train_size=0.95)

#for h, j, k, l in zip(sen_train, sen_test, train_label, test_label):

start1 = time.time()
log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
model_lr = log.fit(x_train, y_train)
print(model_lr)
probs_lr = model_lr.predict_proba(x_test)[:, 1]
print(probs_lr)
probs_lr_scol.append(probs_lr)
ly_prediction = log.predict(x_test)
fly = f1_score(ly_prediction, y_test)
f1_lr_scol.append(fly)
rocauc_lr = roc_auc_score(y_test, ly_prediction)
rocauc_lr_scol.append(rocauc_lr)
recalls_lr = recall_score(y_test, ly_prediction)
recall_lr_scol.append(recalls_lr)
precisions_lr = precision_score(y_test, ly_prediction)
precision_lr_scol.append(precisions_lr)
accuracys_lr = accuracy_score(y_test, ly_prediction)
accuracy_lr_scol.append(accuracys_lr)
print("===Logistic Regression with TfidfVectorizer Imbalanced - Train_Set 2010-2016")
lr_end = time.time()
print('Logistic F1-score', fly * 100)
print('Logistic ROCAUC score:', rocauc_lr * 100)
print('Logistic Recall score:', recalls_lr * 100)
print('Logistic Precision Score:', precisions_lr * 100)
print('Logistic Confusion Matrix', confusion_matrix(y_test, ly_prediction), "\n")
print('Logistic Classification', classification_report(y_test, ly_prediction), "\n")
print('Logistic Accuracy Score', accuracys_lr * 100)
print("Execution Time for Logistic Regression Imbalanced - Train_Set 2010-2016: ", lr_end - start1, "seconds")
