# -*- coding: utf-8 -*-
"""Imbalanced_Classes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GTN2Q9XyFLIyex-nmuU7PB8k4tUmSW-q
"""
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import re
import csv
import time
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import os
# print(os.listdir("../input"))
plt.style.use('ggplot')

df = pd.DataFrame()

raw = open(r"C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\NIST_FULL\\2010-neg.txt", encoding="ISO-8859-1")
lines = raw.readlines()
raw.close()

# remove /n at the end of each line
for index, line in enumerate(lines):
    lines[index] = line.strip()
    print(lines[index])

neg_df = pd.DataFrame(columns=['sentence'])
i = 0
first_col = ""
for line in lines:
    first_col = re.sub(r' \(.*', "", line)
    neg_df.loc[i] = [first_col]
    i = i+1

neg_df.head()
neg_df['label'] = 0
print(neg_df.shape)

raw1 = open(r"C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\NIST_FULL\\2010-pos.txt", encoding="ISO-8859-1")
lines1 = raw1.readlines()
raw1.close()

# remove /n at the end of each line
for index, line in enumerate(lines1):
    lines1[index] = line.strip()
    print(lines1[index])

pos_df = pd.DataFrame(columns=['sentence'])
i = 0
first_col = ""
for line in lines1:
    first_col = re.sub(r' \(.*', "", line)
    pos_df.loc[i] = [first_col]
    i = i+1

pos_df.head()
pos_df['label'] = 1

df = df.append(pos_df)
df = df.append(neg_df)
print(df)
print(pos_df.shape)
print(neg_df.shape)

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

#Removing punctuations from entire dataset
punc_set = string.punctuation
punc_set

#Function for removing punctions
def remove_punc(text):
    clean = "".join([x.lower() for x in text if x not in punc_set])
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

df.head()

time.sleep(500)
'''
xy_num1=500
xy_num2=1000
xy_num3=1500
xy_num4=2000
xy_num5=2500
xy_num6=3000
xy_num7=3500
xy_num8=4000
xy_num9=4500
xy_num10=5000
'''

pos_num = 29708

pos_95 = round(pos_num * 0.95)
pos_85 = round(pos_num * 0.85)
pos_75 = round(pos_num * 0.75)
pos_65 = round(pos_num * 0.65)
pos_55 = round(pos_num * 0.55)
pos_45 = round(pos_num * 0.45)
pos_35 = round(pos_num * 0.35)
pos_25 = round(pos_num * 0.25)
pos_15 = round(pos_num * 0.15)
pos_05 = round(pos_num * 0.05)

neg_num = 48521

neg_95 = round(neg_num * 0.95)
neg_85 = round(neg_num * 0.85)
neg_75 = round(neg_num * 0.75)
neg_65 = round(neg_num * 0.65)
neg_55 = round(neg_num * 0.55)
neg_45 = round(neg_num * 0.45)
neg_35 = round(neg_num * 0.35)
neg_25 = round(neg_num * 0.25)
neg_15 = round(neg_num * 0.15)
neg_05 = round(neg_num * 0.05)

# Splitting Datasets Manually
pos_pt1=df.iloc[0:pos_num-pos_95 :]
neg_pt1=df.iloc[neg_num:neg_num+neg_95, :]
pos_pt2=df.iloc[0:pos_num-pos_85, :]
neg_pt2=df.iloc[neg_num:neg_num+neg_85, :]
pos_pt3=df.iloc[0:pos_num-pos_75, :]
neg_pt3=df.iloc[neg_num:neg_num+neg_75, :]
pos_pt4=df.iloc[0:pos_num-pos_65, :]
neg_pt4=df.iloc[neg_num:neg_num+neg_65, :]
neg_pt5=df.iloc[0:pos_num-pos_55, :]
pos_pt5=df.iloc[neg_num:neg_num+neg_55, :]
pos_pt6=df.iloc[0:pos_num-pos_45, :]
neg_pt6=df.iloc[neg_num:neg_num+neg_45, :]
pos_pt7=df.iloc[0:pos_num-pos_35, :]
neg_pt7=df.iloc[neg_num:neg_num+neg_35, :]
pos_pt8=df.iloc[0:pos_num-pos_25, :]
neg_pt8=df.iloc[neg_num:neg_num+neg_25, :]
pos_pt9=df.iloc[0:pos_num-pos_15, :]
neg_pt9=df.iloc[neg_num:neg_num+neg_15, :]
neg_pt10=df.iloc[0:pos_num-pos_05, :]
pos_pt10=df.iloc[neg_num:neg_num+neg_05, :]

df1 = pd.concat([pos_pt1, neg_pt1])
print("DF1:", round(df1.shape[0]))
df2 = pd.concat([pos_pt2, neg_pt2])
print("DF2:", round(df2.shape[0]))
df3 = pd.concat([pos_pt3, neg_pt3])
print("DF3:", round(df3.shape[0]))
df4 = pd.concat([pos_pt4, neg_pt4])
print("DF4:", round(df4.shape[0]))
df5 = pd.concat([pos_pt5, neg_pt5])
print("DF5:", round(df5.shape[0]))
df6 = pd.concat([pos_pt6, neg_pt6])
print("DF6:", round(df6.shape[0]))
df7 = pd.concat([pos_pt7, neg_pt7])
print("DF7:", round(df7.shape[0]))
df8 = pd.concat([pos_pt8, neg_pt8])
print("DF8:", round(df8.shape[0]))
df9 = pd.concat([pos_pt9, neg_pt9])
print("DF9:", round(df9.shape[0]))
df10 = pd.concat([pos_pt10, neg_pt10])
print("DF10:", round(df10.shape[0]))
#Splitting data into smaller dataframes for the purpose of Training and Testing

df1_l = "Split 1"
df2_l = "Split 2"
df3_l = "Split 3"
df4_l = "Split 4"
df5_l = "Split 5"
df6_1 = "Split 6"
df7_l = "Split 7"
df8_l = "Split 8"
df9_l = "Split 9"
df10_l = "Split 10"

ts01 = "TS 0.1"
ts02 = "TS 0.2"
ts03 = "TS 0.3"
ts04 = "TS 0.4"
ts05 = "TS 0.5"

df1_split = round(df1.shape[0] * .5)
print(df1_split)
df2_split = round(df2.shape[0] * .5)
print(df2_split)
df3_split = round(df3.shape[0] * .5)
print(df3_split)
df4_split = round(df4.shape[0] * .5)
print(df4_split)
df5_split = round(df5.shape[0] * .5)
print(df5_split)
df6_split = round(df6.shape[0] * .5)
print(df6_split)
df7_split = round(df7.shape[0] * .5)
print(df7_split)
df8_split = round(df8.shape[0] * .5)
print(df8_split)
df9_split = round(df9.shape[0] * .5)
print(df9_split)
df10_split = round(df10.shape[0] * .5)
print(df10_split)

'''
xy_ran1=random.randint(1, df.shape[0])
xy_ran2=random.randint(1, df.shape[0])
xy_ran3=random.randint(1, df.shape[0])
xy_ran4=random.randint(1, df.shape[0])
xy_ran5=random.randint(1, df.shape[0])
xy_ran6=random.randint(1, df.shape[0])
xy_ran7=random.randint(1, df.shape[0])
xy_ran8=random.randint(1, df.shape[0])
xy_ran9=random.randint(1, df.shape[0])
xy_ran10=random.randint(1, df.shape[0])
'''

df1_sen = round(df1.shape[0])
df2_sen = round(df2.shape[0])
df3_sen = round(df3.shape[0])
df4_sen = round(df4.shape[0])
df5_sen = round(df5.shape[0])
df6_sen = round(df6.shape[0])
df7_sen = round(df7.shape[0])
df8_sen = round(df8.shape[0])
df9_sen = round(df9.shape[0])
df10_sen = round(df10.shape[0])

x1 = df1.iloc[0:df1_split,5]
x2 = df1.iloc[df1_sen-df1_split:df1_sen,5]
x3 = df2.iloc[0:df2_split,5]
x4 = df2.iloc[df2_sen-df2_split:df2_sen,5]
x5 = df3.iloc[0:df3_split,5]
x6 = df3.iloc[df3_sen-df3_split:df3_sen,5]
x7 = df4.iloc[0:df4_split,5]
x8 = df4.iloc[df4_sen-df4_split:df4_sen,5]
x9 = df5.iloc[0:df5_split,5]
x10 = df5.iloc[df5_sen-df5_split:df5_sen,5]
x11 = df6.iloc[0:df6_split,5]
x12 = df6.iloc[df6_sen-df6_split:df6_sen,5]
x13 = df7.iloc[0:df7_split,5]
x14 = df7.iloc[df7_sen-df7_split:df7_sen,5]
x15 = df8.iloc[0:df8_split,5]
x16 = df8.iloc[df8_sen-df8_split:df8_sen,5]
x17 = df9.iloc[0:df9_split,5]
x18 = df9.iloc[df9_sen-df9_split:df9_sen,5]
x19 = df10.iloc[0:df10_split,5]
x20 = df10.iloc[df10_sen-df10_split:df10_sen,5]

y1 = df1.iloc[0:df1_split,1]
y2 = df1.iloc[df1_sen-df1_split:df1_sen,1]
y3 = df2.iloc[0:df2_split,1]
y4 = df2.iloc[df2_sen-df2_split:df2_sen,1]
y5 = df3.iloc[0:df3_split,1]
y6 = df3.iloc[df3_sen-df3_split:df3_sen,1]
y7 = df4.iloc[0:df4_split,1]
y8 = df4.iloc[df4_sen-df4_split:df4_sen,1]
y9 = df1.iloc[0:df1_split,1]
y10 = df1.iloc[df1_sen-df1_split:df1_sen,1]
y11 = df6.iloc[0:df6_split,1]
y12 = df6.iloc[df6_sen-df6_split:df6_sen,1]
y13 = df7.iloc[0:df7_split,1]
y14 = df7.iloc[df7_sen-df7_split:df7_sen,1]
y11 = df8.iloc[0:df8_split,1]
y16 = df8.iloc[df8_sen-df8_split:df8_sen,1]
y17 = df9.iloc[0:df9_split,1]
y18 = df9.iloc[df9_sen-df9_split:df9_sen,1]
y19 = df10.iloc[0:df10_split,1]
y20 = df10.iloc[df10_sen-df10_split:df10_sen,1]



#x_seg = df.iloc[0:5,5]
#y_seg = df.iloc[0:5,1]
print("X1:", x1.shape)
print("X2:", x2.shape)
print("X3:", x3.shape)
print("X4:", x4.shape)
print("X5:", x5.shape)
print("X6:", x6.shape)
print("X7:", x7.shape)
print("X8:", x8.shape)
print("X9:", x9.shape)
print("X10:", x10.shape)
print("Y1:", y1.shape)
print("Y2:", y2.shape)
print("Y3:", y3.shape)
print("Y4:", y4.shape)
print("Y5:", y5.shape)
print("Y6:", y6.shape)
print("Y7:", y7.shape)
print("Y8:", y8.shape)
print("Y9:", y9.shape)
print("Y10:", y10.shape)

print("\n")
'''
x = df['lemmatized'].values
y = df['label'].values
print(x.shape)
print(y.shape)
'''
'''
train_set = [x1,x3,x5,x7,x9]
test_set = [x2,x4,x6,x8,x10]

train_label = [y1,y3,y5,y7,y9]
test_set = [y2,y4,y6,y8,y10]
'''
#Declaring and applying TFIDF functions to train and test data

tfidf_vect = TfidfVectorizer(ngram_range=(1,2))
tfidf_train = tfidf_vect.fit_transform(x1.values)
tfidf_test=tfidf_vect.transform(x2.values)
print(tfidf_train.shape)
print(tfidf_test.shape)
#tfidf_train.toarray()

#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
'''
# Values for testing set - PTest
Accuracy_LRN = []
Precision_LRN = []
Recall_LRN = []

Accuracy_DCT = []
Recall_DCT = []
Precision_DCT = []

Accuracy_NBB = []
Recall_NBB = []
Precision_NBB = []

Accuracy_XGB = []
Recall_XGB = []
Precision_XGB = []

# Values for testing set - Sampling
Accuracy_LRN_s = []
Precision_LRN_s = []
Recall_LRN_s = []

Accuracy_DCT_s = []
Recall_DCT_s = []
Precision_DCT_s = []

Accuracy_NBB_s = []
Recall_NBB_s = []
Precision_NBB_s = []

Accuracy_XGB_s = []
Recall_XGB_s = []
Precision_XGB_s = []
'''
'''
# Values for training set - PTest
Accuracy_LRN_tr = []
Precision_LRN_tr = []
Recall_LRN_tr = []

Accuracy_DCT_tr = []
Recall_DCT_tr = []
Precision_DCT_tr = []

Accuracy_NBB_tr = []
Recall_NBB_tr = []
precision_NBB_tr = []

Accuracy_RDD_tr = []
Recall_RDD_tr = []
precision_RDD_tr = []
'''
tfidf_vect = TfidfVectorizer()
# Combine dataframes
#x_lemm1 = pd.concat([x1, x2])
#x_lemm2 = pd.concat([x3, x4])
#x_lemm3 = pd.concat([x5, x6])
#x_lemm4 = pd.concat([x7, x8])
#x_lemm5 = pd.concat([x9, x10])

#y_label1 = pd.concat([y1, y2])
#y_label2 = pd.concat([y3, y4])
#y_label3 = pd.concat([y5, y6])
#y_label4 = pd.concat([y7, y8])
#y_label5 = pd.concat([y9, y10])
#== DF1 ==
x_tfidf = tfidf_vect.fit_transform(df1["lemmatized"])

x_train, x_test, y_train, y_test = train_test_split(x_tfidf,df1["label"],test_size=0.5)
'''
# Convert [1,0,0] to 0. [0,1,0] to 1. [0,0,1] to 2
ytrain_ = np.zeros(y_train.shape[0])
for i in range(y_train.shape[0]):
    index = np.argmax(y_train[i])
    if index == 0:
        ytrain_[i] = int(0)
    elif index == 1:
        ytrain_[i] = int(1)
    else:
        ytrain_[i] = int(2)

    ytest_ = np.zeros(y_test.shape[0])
    for i in range(y_test.shape[0]):
        index = np.argmax(y_test[i])
        if index == 0:
            ytest_[i] = int(0)
        elif index == 1:
            ytest_[i] = int(1)
        else:
            ytest_[i] = int(2)
'''
'''
LogisticReg = LogisticRegression(penalty='l2',random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
DecisionTree = DecisionTreeClassifier()
NaiveBayes = MultinomialNB()
XGBoost = XGBClassifier()
RandoFore = RandomForestClassifier(n_estimators=1000, random_state=0)
'''

LogisticReg = LogisticRegression()
DecisionTree = DecisionTreeClassifier()
NaiveBayes = MultinomialNB()
XGBoost = XGBClassifier()
RandoFore = RandomForestClassifier()
'''
lr = "Logistic Regression"
nb = "Naive Bayes"
dt = "Decision Tree"
xg = "XGBoost"
rf = "Random Forest"
'''
'''
probs = [prob1, prob2, ... ]
recalls = [recall1, recall2, ... ]
precisions =  [precision1, precision2, ...]
accuracys = [accuracy1, accuracy2, ...]
starts = [start1, start2 ]
'''
'''
res = []
model_names = [lr, dt, nb, xg, rf]
models = [LogisticReg, DecisionTree, NaiveBayes, XGBoost, RandoFore]
'''
res = []
'''
model_names = [lr, dt, nb, xg, rf]
'''
models = [LogisticReg, DecisionTree, NaiveBayes, XGBoost, RandoFore]
'''
dict_classifiers = {
    "Logistic Regression": LogisticRegression(penalty='l2',random_state=0, solver='lbfgs', multi_class='auto', max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": MultinomialNB(),
    "XGBoost": XGBClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=1000, random_state=0)
}
'''
for model in models:
    start = time.time()
    model_test = model.fit(x_train,y_train)
    prob = model_test.predict_proba(x_test)[:, 1]
    pred = model.predict(x_test)
    f1_s = f1_score(pred, y_test)
    print(model, "F1 Score:", f1_s)
    recall = recall_score(pred, y_test)
    print(model, "Recall_Score:", recall)
    precision = precision_score(pred, y_test)
    print(model, "Precision Score:", precision)
    accuracy = accuracy_score(pred, y_test)
    print(model, "Accuracy Score:", accuracy)
    matrix = confusion_matrix(pred, y_test)
    print(model, "Confusion Matrix: ", matrix)
    report = classification_report(pred, y_test)
    print(model, "Classification Report:", report, "\n")
    end = time.time()
    print(models, "Execution Time: ", end - start, "seconds")
    print('-' * 60)
    print()