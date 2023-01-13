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
#writef = open("Output-Test_Set.txt", "w")
#sys.stdout = writef

file = r'C:\\Users\\Predator\\Documents\\Document-Classification\\classifier\\Test_Set.csv'
df = pd.read_csv(file, header=0, encoding="utf-8")
print(df)

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
pos = len(df[df['label'] == 1])
print("Positive:", pos)
neg = len(df[df['label'] == 0])
print("Negative:", neg)
print(df)
total = df.shape[0]
print("Total", total)
pos_df = df.loc[df['label'] == 1]
print(pos_df)
neg_df = df.loc[df['label'] == 0]
print(neg_df)


# Splitting data into smaller dataframes for the purpose of Training and Testing
x1 = neg_df.iloc[0:1813, 0]
x2 = pos_df.iloc[0:1813, 0]
y1 = neg_df.iloc[0:1813, 1]
y2 = pos_df.iloc[0:1813, 1]
# x_seg = df.iloc[0:5,5]
# y_seg = df.iloc[0:5,1]
print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)

'''
x = df['lemmatized'].values
y = df['label'].values
print(x.shape)
print(y.shape)
'''

# Declaring and applying TFIDF functions to train and test data
tfidf_vect = TfidfVectorizer(ngram_range=(1, 2))
tfidf_train = tfidf_vect.fit_transform(x1.values.astype('U'))
tfidf_test = tfidf_vect.transform(x2.values.astype('U'))
print(tfidf_train.shape)
print(tfidf_test.shape)
# tfidf_train.toarray()

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

tfidf_vect = TfidfVectorizer()
x_tfidf = tfidf_vect.fit_transform(df["sentence"].astype('U'))

x_train, x_test, y_train, y_test = train_test_split(x_tfidf, df["label"], test_size=0.05, train_size=0.95)

start1 = time.time()
log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
model_lr = log.fit(x_train, y_train)
probs_lr = model_lr.predict_proba(x_test)[:, 1]
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
print("===Logistic Regression with TfidfVectorizer Imbalanced - Test_Set 2019")
lr_end = time.time()
print('Logistic F1-score', fly * 100)
print('Logistic ROCAUC score:', rocauc_lr * 100)
print('Logistic Recall score:', recalls_lr * 100)
print('Logistic Precision Score:', precisions_lr * 100)
print('Logistic Confusion Matrix', confusion_matrix(y_test, ly_prediction), "\n")
print('Logistic Classification', classification_report(y_test, ly_prediction), "\n")
print('Logistic Accuracy Score', accuracys_lr * 100)
print("Execution Time for Logistic Regression Imbalanced - Test_Set 2019: ", lr_end - start1, "seconds")

start2 = time.time()
from sklearn.tree import DecisionTreeClassifier

DCT = DecisionTreeClassifier()
model_dt = DCT.fit(x_train, y_train)
probs_dt = model_dt.predict_proba(x_test)[:, 1]
probs_dt_scol.append(probs_dt)
dct_pred = DCT.predict(x_test)
fdct = f1_score(dct_pred, y_test)
f1_dt_scol.append(fdct)
rocauc_dt = roc_auc_score(y_test, dct_pred)
rocauc_dt_scol.append(rocauc_dt)
recalls_dt = recall_score(y_test, dct_pred)
recall_dt_scol.append(recalls_dt)
precisions_dt = precision_score(y_test, dct_pred)
precision_dt_scol.append(precisions_dt)
accuracys_dt = accuracy_score(y_test, dct_pred)
accuracy_dt_scol.append(accuracys_dt)
print("===DecisionTreeClassifier with TfidfVectorizer Imbalanced Test_Set - 2019")
dt_end = time.time()
print('DCT F1-score', fdct * 100)
print('DCT ROCAUC score:', rocauc_dt * 100)
print('DCT Recall score:', recalls_dt * 100)
print('DCT Precision Score:', precisions_dt * 100)
print('DCT Confusion Matrix', confusion_matrix(y_test, dct_pred), "\n")
print('DCT Classification', classification_report(y_test, dct_pred), "\n")
print('DCT Accuracy Score', accuracys_dt * 100)
print("Execution Time for Decision Tree Imbalanced Test_Set - 2019: ", dt_end - start2, "seconds")

from sklearn.naive_bayes import MultinomialNB

start3 = time.time()
Naive = MultinomialNB()
model_nb = Naive.fit(x_train, y_train)
probs_nb = model_nb.predict_proba(x_test)[:, 1]
probs_nb_scol.append(probs_nb)
# predict the labels on validation dataset
ny_pred = Naive.predict(x_test)
fnb = f1_score(ny_pred, y_test)
f1_nb_scol.append(fnb)
rocauc_nb = roc_auc_score(y_test, ny_pred)
rocauc_nb_scol.append(rocauc_nb)
recalls_nb = recall_score(y_test, ny_pred)
recall_nb_scol.append(recalls_nb)
precisions_nb = precision_score(y_test, ny_pred)
precision_nb_scol.append(precisions_nb)
accuracys_nb = accuracy_score(y_test, ny_pred)
accuracy_nb_scol.append(accuracys_nb)
nb_end = time.time()
# Use accuracy_score function to get the accuracy
print("===Naive Bayes with TfidfVectorizer Imbalanced Test_Set - 2019")
print('Naive F1-score', fnb * 100)
print('Naive ROCAUC score:', rocauc_nb * 100)
print('Naive Recall score:', recalls_nb * 100)
print('Naive Precision Score:', precisions_nb * 100)
print('Naive Confusion Matrix', confusion_matrix(y_test, ny_pred), "\n")
print('Naive Classification', classification_report(y_test, ny_pred), "\n")
print('Naive Accuracy Score', accuracys_nb * 100)
print("Execution Time for Naive Bayes Imbalanced Test_Set - 2019: ", nb_end - start3, "seconds")

# XGBoost Classifier

start4 = time.time()
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

xgb_model = XGBClassifier().fit(x_train, y_train)
probs_xg = xgb_model.predict_proba(x_test)[:, 1]
probs_xg_scol.append(probs_xg)
# predict
xgb_y_predict = xgb_model.predict(x_test)
fxg = f1_score(xgb_y_predict, y_test)
f1_xg_scol.append(fxg)
rocauc_xg = roc_auc_score(xgb_y_predict, y_test)
rocauc_xg_scol.append(rocauc_xg)
recall_xg = recall_score(xgb_y_predict, y_test)
recall_xg_scol.append(recall_xg)
precisions_xg = precision_score(xgb_y_predict, y_test)
precision_xg_scol.append(precisions_xg)
accuracys_xg = accuracy_score(xgb_y_predict, y_test)
accuracy_xg_scol.append(accuracys_xg)
xg_end = time.time()
print("===XGB with TfidfVectorizer Imbalanced Test_Set - 2019")
print('XGB F1-Score', fxg * 100)
print('XGB ROCAUC Score:', rocauc_xg * 100)
print('XGB Recall score:', recall_xg * 100)
print('XGB Precision Score:', precisions_xg * 100)
print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, y_test), "\n")
print('XGB Classification', classification_report(xgb_y_predict, y_test), "\n")
print('XGB Accuracy Score', accuracys_nb * 100)
print("Execution Time for XGBoost Classifier Imbalanced Test_Set - 2019:", xg_end - start4, "seconds")

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

start5 = time.time()
rfc_model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(x_train, y_train)
probs_rf = rfc_model.predict_proba(x_test)[:, 1]
probs_rf_scol.append(probs_rf)
rfc_pred = rfc_model.predict(x_test)
frfc = f1_score(rfc_pred, y_test)
f1_rf_scol.append(frfc)
rocauc_rf = roc_auc_score(y_test, rfc_pred)
rocauc_rf_scol.append(rocauc_rf)
recalls_rf = recall_score(rfc_pred, y_test)
recall_rf_scol.append(recalls_rf)
precisions_rf = precision_score(rfc_pred, y_test)
precision_rf_scol.append(precisions_rf)
accuracys_rf = accuracy_score(rfc_pred, y_test)
accuracy_rf_scol.append(accuracys_rf)
rf_end = time.time()
print("====RandomForest with TfidfVectorizer Imbalanced Test_Set - 2019")
print('RFC F1 score', frfc * 100)
print('RFC ROCAUC Score:', rocauc_rf * 100)
print('RFC Recall score:', recalls_rf * 100)
print('RFC Precision Score:', precisions_rf * 100)
print('RFC Confusion Matrix', confusion_matrix(y_test, rfc_pred), "\n")
print('RFC Classification', classification_report(y_test, rfc_pred), "\n")
print('RFC Accuracy Score', accuracys_rf * 100)
print("Execution Time for Random Forest Classifier Imbalanced Test_Set - 2019", rf_end - start5, "seconds")

print("Array of Prob Scores LR-Imb Test-Size", ":", probs_lr_scol)
print("Array of F1 Scores LR-Imb Test-Size:", ":", f1_lr_scol)
print("Array of ROCAUC Scores LR-Imb:", ":", rocauc_lr_scol)
print("Array of Recall Scores LR-Imb:", ":", recall_lr_scol)
print("Array of Precision Scores LR-Imb:", ":", precision_lr_scol)
print("Array of Accuracy Scores LR-Imb:", ":", accuracy_lr_scol)

print("Array of Prob Scores DT-Imb:", ":", probs_dt_scol)
print("Array of F1 Scores DT-Imb:", ":", f1_dt_scol)
print("Array of ROCAUC Scores DT-Imb:", ":", rocauc_dt_scol)
print("Array of Recall Scores DT-Imb:", ":", recall_dt_scol)
print("Array of Precision Scores DT-Imb:", ":", precision_dt_scol)
print("Array of Accuracy Scores DT-Imb:", ":", accuracy_dt_scol)

print("Array of Prob Scores NB-Imb:", ":", probs_nb_scol)
print("Array of F1 Scores NB-Imb:", ":", f1_nb_scol)
print("Array of ROCAUC Scores NB-Imb:", ":", rocauc_nb_scol)
print("Array of Recall Scores NB-Imb:", ":", recall_nb_scol)
print("Array of Precision Scores NB-Imb:", ":", precision_nb_scol)
print("Array of Accuracy Scores NB-Imb:", ":", accuracy_nb_scol)

print("Array of Prob Scores XG-Imb:", ":", probs_xg_scol)
print("Array of F1 Scores XG-Imb:", ":", f1_xg_scol)
print("Array of ROCAUC Scores XG-Imb:", ":", rocauc_xg_scol)
print("Array of Recall Scores XG-Imb:", ":", recall_xg_scol)
print("Array of Precision Scores XG-Imb:", ":", precision_xg_scol)
print("Array of Accuracy Scores XG-Imb:", ":", accuracy_xg_scol)

print("Array of Prob Scores RF-Imb:", ":", probs_rf_scol)
print("Array of F1 Scores RF-Imb:", ":", f1_rf_scol)
print("Array of ROCAUC Scores RF-Imb:", ":", rocauc_rf_scol)
print("Array of Recall Scores RF-Imb:", ":", recall_rf_scol)
print("Array of Precision Scores RF-Imb:", ":", precision_rf_scol)
print("Array of Accuracy Scores RF-Imb:", ":", accuracy_rf_scol)

from itertools import chain

name = ['Test_Set' for t in range(5)]
sampling = ['N/A' for t in range(5)]
technique = ['N/A' for t in range(5)]
classifier_names = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
test_sizes_num = [0.05]
train_sizes_num = [0.95]
# v = [0, 1, 2, 3, 4]
# precision_csv_num = [precision_lr_scol[z], precision_dt_scol[z], precision_nb_scol[z], precision_xg_scol[z], precision_rf_scol[z]]
# recall_csv_num = [recall_lr_scol[z], recall_dt_scol[z], recall_nb_scol[z], recall_xg_scol[z], recall_rf_scol[z]]
# auc_csv_num = [rocauc_lr_scol[z], rocauc_dt_scol[z], rocauc_nb_scol[z], rocauc_xg_scol[z], rocauc_rf_scol[z]]
# accuracy_csv_num = [accuracy_lr_scol[z], accuracy_dt_scol[z], accuracy_nb_scol[z], accuracy_xg_scol[z], accuracy_rf_scol[z]]
precision_csv_num = [precision_lr_scol, precision_dt_scol, precision_nb_scol, precision_xg_scol, precision_rf_scol]
recall_csv_num = [recall_lr_scol, recall_dt_scol, recall_nb_scol, recall_xg_scol, recall_rf_scol]
auc_csv_num = [rocauc_lr_scol, rocauc_dt_scol, rocauc_nb_scol, rocauc_xg_scol, rocauc_rf_scol]
accuracy_csv_num = [accuracy_lr_scol, accuracy_dt_scol, accuracy_nb_scol, accuracy_xg_scol, accuracy_rf_scol]
import itertools
rounds = 1
p = itertools.cycle(classifier_names)
o = itertools.cycle(test_sizes_num)
k = itertools.cycle(train_sizes_num)
# v = itertools.cycle(score_location)
# pr = itertools.cycle(precision_num)
# y = itertools.cycle(iteration_csv)
classifier_csv = [next(p) for _ in range(rounds)] * 5
test_size_csv = [next(o) for _ in range(rounds)] * 5
train_size_csv = [next(k) for _ in range(rounds)] * 5
# split_csv = ['1' for t in range(25)]
# train_csv = ['0.5' for t in range(25)]
precision_csv = list(chain(*precision_csv_num))
recall_csv = list(chain(*recall_csv_num))
auc_csv = list(chain(*auc_csv_num))
accuracy_csv = list(chain(*accuracy_csv_num))
csv_data = [name, sampling, technique, classifier_csv, test_size_csv, train_size_csv, precision_csv, recall_csv,
            auc_csv, accuracy_csv]
export_data = zip_longest(*csv_data, fillvalue='')
filename = "Test_Iterations.csv"
with open(filename, 'w', newline='') as file:
    write = csv.writer(file)
    write.writerow(("Subset", "Sampling", "Technique", "Classifier", "Test Split Size", "Train Split Size", "Precision",
                    "Recall", "AUC", "Accuracy"))
    write.writerows(export_data)
'''
tfidf_vect1 = TfidfVectorizer(ngram_range=(1,2))
tfidf_train1 = tfidf_vect.fit_transform(x1.values)
tfidf_test1=tfidf_vect.transform(x2.values)
print(tfidf_train1.shape)
print(tfidf_test1.shape)
'''

x_tfidf1 = tfidf_vect.fit_transform(df["sentence"].astype('U'))
'''
df['label'].astype('U').value_counts()
print(df['label'].astype('U').value_counts())

import seaborn as sns

g = sns.countplot(df['label'].astype('U'))
g.set_xticklabels(['Negative', 'Positive'])
plt.show()

# class count
label_count_neg, label_count_pos = df['label'].astype('U').value_counts()

# Separate class
label_neg = df[df['label'] == 0]
label_pos = df[df['label'] == 1]  # print the shape of the class
print('Label Negative:', label_neg.shape)
print('Label Positive:', label_pos.shape)

label_neg_under = label_neg.sample(label_count_pos)

test_under = pd.concat([label_neg_under, label_pos], axis=0)

print("total class of pos and neg :", test_under['label'].value_counts())  # plot the count after under-sampling
test_under['label'].value_counts().plot(kind='bar', title='label (target)')

label_pos_over = label_pos.sample(label_count_neg, replace=True)

test_over = pd.concat([label_pos_over, label_neg], axis=0)

print("total class of pos and neg:", test_under['label'].value_counts())  # plot the count after under-sampeling
test_over['label'].value_counts().plot(kind='bar', title='label (target)')

import imblearn


df['label'].value_counts()
print(df['label'].value_counts())

import seaborn as sns

g = sns.countplot(df['label'])
g.set_xticklabels(['Negative', 'Positive'])
plt.show()
'''

# import library
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

x = x_tfidf1
y = df['label']
print(x.shape)
print(y.shape)

rus = RandomUnderSampler(random_state=42, replacement=True)  # fit predictor and target variable
x_rus, y_rus = rus.fit_resample(x, y)

print('Original dataset shape:', Counter(y))
print('Resample dataset shape', Counter(y_rus))

# import library
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

'''
x =x_tfidf
y = df1["label"]
'''
ros = RandomUnderSampler(random_state=42)
# ros = RandomOverSampler(random_state=42)
# Random over-sampling with imblearn
# fit predictor and target variable
x_rus, y_rus = rus.fit_resample(x, y)

print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_rus))
# Random over-sampling with imblearn
ros = RandomOverSampler(random_state=42)

# fit predictor and target variable
x_ros, y_ros = ros.fit_resample(x, y)

print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_ros))
# import library

# Under-sampling: Tomek links
from imblearn.under_sampling import TomekLinks
from collections import Counter

tl = RandomOverSampler(sampling_strategy='majority')

# fit predictor and target variable
x_tl, y_tl = ros.fit_resample(x, y)

print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_ros))

# import library
from imblearn.over_sampling import SMOTE

smote = SMOTE()

# fit predictor and target variable
x_smote, y_smote = smote.fit_resample(x, y)

print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_ros))

from imblearn.under_sampling import NearMiss

nm = NearMiss()

x_nm, y_nm = nm.fit_resample(x, y)

print('Original dataset shape:', Counter(y))
print('Resample dataset shape:', Counter(y_nm))


label_count = df.groupby("label")
print("Label Count", label_count)
print(df.label.dtype)
print(df.info())
print("XTIDF NM:", type(x_nm))
print("LABEL NM:", type(y_nm))
import os
def pause():
    Pause = input("Press the <ENTER> key to continue...")

print("STOP!...")
pause()

# Tomelinks
probs_lr_scol1 = []
f1_lr_scol1 = []
rocauc_lr_scol1 = []
recall_lr_scol1 = []
precision_lr_scol1 = []
accuracy_lr_scol1 = []

probs_dt_scol1 = []
f1_dt_scol1 = []
rocauc_dt_scol1 = []
recall_dt_scol1 = []
precision_dt_scol1 = []
accuracy_dt_scol1 = []

probs_nb_scol1 = []
f1_nb_scol1 = []
rocauc_nb_scol1 = []
recall_nb_scol1 = []
precision_nb_scol1 = []
accuracy_nb_scol1 = []

probs_xg_scol1 = []
f1_xg_scol1 = []
rocauc_xg_scol1 = []
recall_xg_scol1 = []
precision_xg_scol1 = []
accuracy_xg_scol1 = []

probs_rf_scol1 = []
f1_rf_scol1 = []
rocauc_rf_scol1 = []
recall_rf_scol1 = []
precision_rf_scol1 = []
accuracy_rf_scol1 = []

x1_train, x1_test, y1_train, y1_test = train_test_split(x_tl, y_tl, train_size=0.95, test_size=0.05)

start1 = time.time()
log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
model_lr = log.fit(x1_train, y1_train)
probs_lr = model_lr.predict_proba(x1_test)[:, 1]
probs_lr_scol1.append(probs_lr)
ly_prediction = log.predict(x1_test)
fly = f1_score(ly_prediction, y1_test)
f1_lr_scol1.append(fly)
rocauc_lr = roc_auc_score(y1_test, ly_prediction)
rocauc_lr_scol1.append(rocauc_lr)
recalls_lr = recall_score(y1_test, ly_prediction)
recall_lr_scol1.append(recalls_lr)
precisions_lr = precision_score(y1_test, ly_prediction)
precision_lr_scol1.append(precisions_lr)
accuracys_lr = accuracy_score(y1_test, ly_prediction)
accuracy_lr_scol1.append(accuracys_lr)
print("===Logistic Regression with TfidfVectorizer Tomelinks Test_Set - 2019")
lr_end = time.time()
print('Logistic F1-score', fly * 100)
print('Logistic ROCAUC score:', rocauc_lr * 100)
print('Logistic Recall score:', recalls_lr * 100)
print('Logistic Precision Score:', precisions_lr * 100)
print('Logistic Confusion Matrix', confusion_matrix(y1_test, ly_prediction), "\n")
print('Logistic Classification', classification_report(y1_test, ly_prediction), "\n")
print('Logistic Accuracy Score', accuracys_lr * 100)
print("Execution Time for Logistic Regression Tomelinks Test_Set - 2019", lr_end - start1, "seconds")

start2 = time.time()
from sklearn.tree import DecisionTreeClassifier

DCT = DecisionTreeClassifier()
model_dt = DCT.fit(x1_train, y1_train)
probs_dt = model_dt.predict_proba(x1_test)[:, 1]
probs_dt_scol1.append(probs_dt)
dct_pred = DCT.predict(x1_test)
fdct = f1_score(dct_pred, y1_test)
f1_dt_scol1.append(fdct)
rocauc_dt = roc_auc_score(y1_test, dct_pred)
rocauc_dt_scol1.append(rocauc_dt)
recalls_dt = recall_score(y1_test, dct_pred)
recall_dt_scol1.append(recalls_dt)
precisions_dt = precision_score(y1_test, dct_pred)
precision_dt_scol1.append(precisions_dt)
accuracys_dt = accuracy_score(y1_test, dct_pred)
accuracy_dt_scol1.append(accuracys_dt)
print("===DecisionTreeClassifier with TfidfVectorizer Tomelinks Test_Set - 2019")
dt_end = time.time()
print('DCT F1-score', fdct * 100)
print('DCT ROCAUC score:', rocauc_dt * 100)
print('DCT Recall score:', recalls_dt * 100)
print('DCT Precision Score:', precisions_dt * 100)
print('DCT Confusion Matrix', confusion_matrix(y1_test, dct_pred), "\n")
print('DCT Classification', classification_report(y1_test, dct_pred), "\n")
print('DCT Accuracy Score', accuracys_dt * 100)
print("Execution Time for Decision Tree Tomelinks Test_Set - 2019", dt_end - start2, "seconds")

from sklearn.naive_bayes import MultinomialNB

start3 = time.time()
Naive = MultinomialNB()
model_nb = Naive.fit(x1_train, y1_train)
probs_nb = model_nb.predict_proba(x1_test)[:, 1]
probs_nb_scol1.append(probs_nb)
# predict the labels on validation dataset
ny_pred = Naive.predict(x1_test)
fnb = f1_score(ny_pred, y1_test)
f1_nb_scol1.append(fnb)
rocauc_nb = roc_auc_score(y1_test, ny_pred)
rocauc_nb_scol1.append(rocauc_nb)
recalls_nb = recall_score(y1_test, ny_pred)
recall_nb_scol1.append(recalls_nb)
precisions_nb = precision_score(y1_test, ny_pred)
precision_nb_scol1.append(precisions_nb)
accuracys_nb = accuracy_score(y1_test, ny_pred)
accuracy_nb_scol1.append(accuracys_nb)
nb_end = time.time()
# Use accuracy_score function to get the accuracy
print("===Naive Bayes with TfidfVectorizer Imabalanced Tomelinks Test_Set - 2019")
print('Naive F1-score', fnb * 100)
print('Naive ROCAUC score:', rocauc_nb * 100)
print('Naive Recall score:', recalls_nb * 100)
print('Naive Precision Score:', precisions_nb * 100)
print('Naive Confusion Matrix', confusion_matrix(y1_test, ny_pred), "\n")
print('Naive Classification', classification_report(y1_test, ny_pred), "\n")
print('Naive Accuracy Score', accuracys_nb * 100)
print("Execution Time for Naive Bayes Tomelinks Test_Set - 2019", nb_end - start3, "seconds")

# XGBoost Classifier

start4 = time.time()
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

xgb_model = XGBClassifier().fit(x1_train, y1_train)
probs_xg = xgb_model.predict_proba(x1_test)[:, 1]
probs_xg_scol1.append(probs_xg)
# predict
xgb_y_predict = xgb_model.predict(x1_test)
fxg = f1_score(xgb_y_predict, y1_test)
f1_xg_scol1.append(fxg)
rocauc_xg = roc_auc_score(xgb_y_predict, y1_test)
rocauc_xg_scol1.append(rocauc_xg)
recall_xg = recall_score(xgb_y_predict, y1_test)
recall_xg_scol1.append(recall_xg)
precisions_xg = precision_score(xgb_y_predict, y1_test)
precision_xg_scol1.append(precisions_xg)
accuracys_xg = accuracy_score(xgb_y_predict, y1_test)
accuracy_xg_scol1.append(accuracys_xg)
xg_end = time.time()
print("===XGB with TfidfVectorizer Tomelinks Test_Set - 2019")
print('XGB F1-Score', fxg * 100)
print('XGB ROCAUC Score:', rocauc_xg * 100)
print('XGB Recall score:', recall_xg * 100)
print('XGB Precision Score:', precisions_xg * 100)
print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, y1_test), "\n")
print('XGB Classification', classification_report(xgb_y_predict, y1_test), "\n")
print('XGB Accuracy Score', accuracys_nb * 100)
print("Execution Time for XGBoost Classifier Tomelinks Test_Set - 2019", xg_end - start4, "seconds")

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

start5 = time.time()
rfc_model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(x1_train, y1_train)
probs_rf = rfc_model.predict_proba(x1_test)[:, 1]
probs_rf_scol1.append(probs_rf)
rfc_pred = rfc_model.predict(x1_test)
frfc = f1_score(rfc_pred, y1_test)
f1_rf_scol1.append(frfc)
rocauc_rf = roc_auc_score(y1_test, rfc_pred)
rocauc_rf_scol1.append(rocauc_rf)
recalls_rf = recall_score(rfc_pred, y1_test)
recall_rf_scol1.append(recalls_rf)
precisions_rf = precision_score(rfc_pred, y1_test)
precision_rf_scol1.append(precisions_rf)
accuracys_rf = accuracy_score(rfc_pred, y1_test)
accuracy_rf_scol1.append(accuracys_rf)
rf_end = time.time()
print("====RandomForest with Tfidf Tomelinks Test_Set - 2019")
print('RFC F1 score', frfc * 100)
print('RFC ROCAUC Score:', rocauc_rf * 100)
print('RFC Recall score:', recalls_rf * 100)
print('RFC Precision Score:', precisions_rf * 100)
print('RFC Confusion Matrix', confusion_matrix(y1_test, rfc_pred), "\n")
print('RFC Classification', classification_report(y1_test, rfc_pred), "\n")
print('RFC Accuracy Score', accuracys_rf * 100)
print("Execution Time for Random Forest Classifier Tomelinks Test_Set - 2019 ", rf_end - start5, "seconds")

print("Array of Prob Scores LR-Sam Tomelinks", ":", probs_lr_scol1)
print("Array of F1 Scores LR-Sam Tomelinks", ":", f1_lr_scol1)
print("Array of ROCAUC Scores LR-Sam Tomelinks", ":", rocauc_lr_scol1)
print("Array of Recall Scores LR-Sam Tomelinks", ":", recall_lr_scol1)
print("Array of Precision Scores LR-Sam Tomelinks", ":", precision_lr_scol1)
print("Array of Accuracy Scores LR-Sam Tomelinks", ":", accuracy_lr_scol1)

print("Array of Prob Scores DT-Sam Tomelinks", ":", probs_dt_scol1)
print("Array of F1 Scores DT-Sam Tomelinks", ":", f1_dt_scol1)
print("Array of ROCAUC Scores DT-Sam Tomelinks", ":", rocauc_dt_scol1)
print("Array of Recall Scores DT-Sam Tomelinks", ":", recall_dt_scol1)
print("Array of Precision Scores DT-Sam Tomelinks", ":", precision_dt_scol1)
print("Array of Accuracy Scores DT-Sam Tomelinks", ":", accuracy_dt_scol1)

print("Array of Prob Scores NB-Sam Tomelinks", ":", probs_nb_scol1)
print("Array of F1 Scores NB-Sam Tomelinks", ":", f1_nb_scol1)
print("Array of ROCAUC Scores NB-Sam Tomelinks", ":", rocauc_nb_scol1)
print("Array of Recall Scores NB-Sam Tomelinks", ":", recall_nb_scol1)
print("Array of Precision Scores NB-Sam Tomelinks", ":", precision_nb_scol1)
print("Array of Accuracy Scores NB-Sam Tomelinks", ":", accuracy_nb_scol1)

print("Array of Prob Scores XG-Sam Tomelinks", ":", probs_xg_scol1)
print("Array of F1 Scores XG-Sam Tomelinks", ":", f1_xg_scol1)
print("Array of ROCAUC Scores XG-Sam Tomelinks", ":", rocauc_xg_scol1)
print("Array of Recall Scores XG-Sam Tomelinks", ":", recall_xg_scol1)
print("Array of Precision Scores XG-Sam Tomelinks", ":", precision_xg_scol1)
print("Array of Accuracy Scores XG-Sam Tomelinks", ":", accuracy_xg_scol1)

print("Array of Prob Scores RF-Sam Tomelinks", ":", probs_rf_scol1)
print("Array of F1 Scores RF-Sam Tomelinks", ":", f1_rf_scol1)
print("Array of ROCAUC Scores RF-Sam Tomelinks", ":", rocauc_rf_scol1)
print("Array of Recall Scores RF-Sam Tomelinks", ":", recall_rf_scol1)
print("Array of Precision Scores RF-Sam Tomelinks", ":", precision_rf_scol1)
print("Array of Accuracy Scores RF-Sam Tomelinks", ":", accuracy_rf_scol1)

set1 = ['Test_Set' for t in range(5)]
sampling1 = ['Oversampling' for t in range(5)]
technique1 = ['Tomelinks' for t in range(5)]
classifier_names1 = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
test_sizes_num1 = [0.05]
train_sizes_num1 = [0.95]
# v = [0, 1, 2, 3, 4]
# precision_csv_num = [precision_lr_scol[z], precision_dt_scol[z], precision_nb_scol[z], precision_xg_scol[z], precision_rf_scol[z]]
# recall_csv_num = [recall_lr_scol[z], recall_dt_scol[z], recall_nb_scol[z], recall_xg_scol[z], recall_rf_scol[z]]
# auc_csv_num = [rocauc_lr_scol[z], rocauc_dt_scol[z], rocauc_nb_scol[z], rocauc_xg_scol[z], rocauc_rf_scol[z]]
# accuracy_csv_num = [accuracy_lr_scol[z], accuracy_dt_scol[z], accuracy_nb_scol[z], accuracy_xg_scol[z], accuracy_rf_scol[z]]
precision_csv_num1 = [precision_lr_scol1, precision_dt_scol1, precision_nb_scol1, precision_xg_scol1,
                      precision_rf_scol1]
recall_csv_num1 = [recall_lr_scol1, recall_dt_scol1, recall_nb_scol1, recall_xg_scol1, recall_rf_scol1]
auc_csv_num1 = [rocauc_lr_scol1, rocauc_dt_scol1, rocauc_nb_scol1, rocauc_xg_scol1, rocauc_rf_scol1]
accuracy_csv_num1 = [accuracy_lr_scol1, accuracy_dt_scol1, accuracy_nb_scol1, accuracy_xg_scol1, accuracy_rf_scol1]
import itertools

rounds = 1
p1 = itertools.cycle(classifier_names1)
o1 = itertools.cycle(test_sizes_num1)
k1 = itertools.cycle(train_sizes_num1)
# v = itertools.cycle(score_location)
# pr = itertools.cycle(precision_num)
# y = itertools.cycle(iteration_csv)
classifier_csv1 = [next(p1) for _ in range(rounds)] * 5
test_size_csv1 = [next(o1) for _ in range(rounds)] * 5
train_size_csv1 = [next(k1) for _ in range(rounds)] * 5
# split_csv = ['1' for t in range(25)]
# train_csv = ['0.5' for t in range(25)]
precision_csv1 = list(chain(*precision_csv_num1))
recall_csv1 = list(chain(*recall_csv_num1))
auc_csv1 = list(chain(*auc_csv_num1))
accuracy_csv1 = list(chain(*accuracy_csv_num1))
csv_data1 = [set1, sampling1, technique1, classifier_csv1, test_size_csv1, train_size_csv1, precision_csv1, recall_csv1,
             auc_csv1, accuracy_csv1]
export_data1 = zip_longest(*csv_data1, fillvalue='')
filename = "Test_Iterations.csv"
with open(filename, 'a', newline='') as file:
    write = csv.writer(file)
    # write.writerow(("Name", "Sampling", "Technique", "Classifier", "Test Split Size", "Train Split Size", "Precision", "Recall", "AUC", "Accuracy"))
    write.writerows(export_data1)

# NearMiss
probs_lr_scol2 = []
f1_lr_scol2 = []
rocauc_lr_scol2 = []
recall_lr_scol2 = []
precision_lr_scol2 = []
accuracy_lr_scol2 = []

probs_dt_scol2 = []
f1_dt_scol2 = []
rocauc_dt_scol2 = []
recall_dt_scol2 = []
precision_dt_scol2 = []
accuracy_dt_scol2 = []

probs_nb_scol2 = []
f1_nb_scol2 = []
rocauc_nb_scol2 = []
recall_nb_scol2 = []
precision_nb_scol2 = []
accuracy_nb_scol2 = []

probs_xg_scol2 = []
f1_xg_scol2 = []
rocauc_xg_scol2 = []
recall_xg_scol2 = []
precision_xg_scol2 = []
accuracy_xg_scol2 = []

probs_rf_scol2 = []
f1_rf_scol2 = []
rocauc_rf_scol2 = []
recall_rf_scol2 = []
precision_rf_scol2 = []
accuracy_rf_scol2 = []

x2_train, x2_test, y2_train, y2_test = train_test_split(x_nm, y_nm, train_size=0.95, test_size=0.05)

start1 = time.time()
log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
model_lr = log.fit(x2_train, y2_train)
probs_lr = model_lr.predict_proba(x2_test)[:, 1]
probs_lr_scol2.append(probs_lr)
ly_prediction = log.predict(x2_test)
fly = f1_score(ly_prediction, y2_test)
f1_lr_scol2.append(fly)
rocauc_lr = roc_auc_score(y2_test, ly_prediction)
rocauc_lr_scol2.append(rocauc_lr)
recalls_lr = recall_score(y2_test, ly_prediction)
recall_lr_scol2.append(recalls_lr)
precisions_lr = precision_score(y2_test, ly_prediction)
precision_lr_scol2.append(precisions_lr)
accuracys_lr = accuracy_score(y2_test, ly_prediction)
accuracy_lr_scol2.append(accuracys_lr)
print("===Logistic Regression with TfidfVectorizer NearMiss Test_Set - 2019")
lr_end = time.time()
print('Logistic F1-score', fly * 100)
print('Logistic ROCAUC score:', rocauc_lr * 100)
print('Logistic Recall score:', recalls_lr * 100)
print('Logistic Precision Score:', precisions_lr * 100)
print('Logistic Confusion Matrix', confusion_matrix(y2_test, ly_prediction), "\n")
print('Logistic Classification', classification_report(y2_test, ly_prediction), "\n")
print('Logistic Accuracy Score', accuracys_lr * 100)
print("Execution Time for Logistic Regression NearMiss Test_Set - 2019", lr_end - start1, "seconds")

start2 = time.time()
from sklearn.tree import DecisionTreeClassifier

DCT = DecisionTreeClassifier()
model_dt = DCT.fit(x2_train, y2_train)
probs_dt = model_dt.predict_proba(x2_test)[:, 1]
probs_dt_scol2.append(probs_dt)
dct_pred = DCT.predict(x2_test)
fdct = f1_score(dct_pred, y2_test)
f1_dt_scol2.append(fdct)
rocauc_dt = roc_auc_score(y2_test, dct_pred)
rocauc_dt_scol2.append(rocauc_dt)
recalls_dt = recall_score(y2_test, dct_pred)
recall_dt_scol2.append(recalls_dt)
precisions_dt = precision_score(y2_test, dct_pred)
precision_dt_scol2.append(precisions_dt)
accuracys_dt = accuracy_score(y2_test, dct_pred)
accuracy_dt_scol2.append(accuracys_dt)
print("===DecisionTreeClassifier with TfidfVectorizer NearMiss Test_Set - 2019")
dt_end = time.time()
print('DCT F1-score', fdct * 100)
print('DCT ROCAUC score:', rocauc_dt * 100)
print('DCT Recall score:', recalls_dt * 100)
print('DCT Precision Score:', precisions_dt * 100)
print('DCT Confusion Matrix', confusion_matrix(y2_test, dct_pred), "\n")
print('DCT Classification', classification_report(y2_test, dct_pred), "\n")
print('DCT Accuracy Score', accuracys_dt * 100)
print("Execution Time for Decision Tree NearMiss Test_Set - 2019", dt_end - start2, "seconds")

from sklearn.naive_bayes import MultinomialNB

start3 = time.time()
Naive = MultinomialNB()
model_nb = Naive.fit(x2_train, y2_train)
probs_nb = model_nb.predict_proba(x2_test)[:, 1]
probs_nb_scol2.append(probs_nb)
# predict the labels on validation dataset
ny_pred = Naive.predict(x2_test)
fnb = f1_score(ny_pred, y2_test)
f1_nb_scol2.append(fnb)
rocauc_nb = roc_auc_score(y2_test, ny_pred)
rocauc_nb_scol2.append(rocauc_nb)
recalls_nb = recall_score(y2_test, ny_pred)
recall_nb_scol2.append(recalls_nb)
precisions_nb = precision_score(y2_test, ny_pred)
precision_nb_scol2.append(precisions_nb)
accuracys_nb = accuracy_score(y2_test, ny_pred)
accuracy_nb_scol2.append(accuracys_nb)
nb_end = time.time()
# Use accuracy_score function to get the accuracy
print("===Naive Bayes with TfidfVectorizer Imabalanced NearMiss Test_Set - 2019")
print('Naive F1-score', fnb * 100)
print('Naive ROCAUC score:', rocauc_nb * 100)
print('Naive Recall score:', recalls_nb * 100)
print('Naive Precision Score:', precisions_nb * 100)
print('Naive Confusion Matrix', confusion_matrix(y2_test, ny_pred), "\n")
print('Naive Classification', classification_report(y2_test, ny_pred), "\n")
print('Naive Accuracy Score', accuracys_nb * 100)
print("Execution Time for Naive Bayes NearMiss Test_Set - 2019", nb_end - start3, "seconds")

# XGBoost Classifier

start4 = time.time()
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

xgb_model = XGBClassifier().fit(x2_train, y2_train)
probs_xg = xgb_model.predict_proba(x2_test)[:, 1]
probs_xg_scol2.append(probs_xg)
# predict
xgb_y_predict = xgb_model.predict(x2_test)
fxg = f1_score(xgb_y_predict, y2_test)
f1_xg_scol2.append(fxg)
rocauc_xg = roc_auc_score(xgb_y_predict, y2_test)
rocauc_xg_scol2.append(rocauc_xg)
recall_xg = recall_score(xgb_y_predict, y2_test)
recall_xg_scol2.append(recall_xg)
precisions_xg = precision_score(xgb_y_predict, y2_test)
precision_xg_scol2.append(precisions_xg)
accuracys_xg = accuracy_score(xgb_y_predict, y2_test)
accuracy_xg_scol2.append(accuracys_xg)
xg_end = time.time()
print("===XGB with TfidfVectorizer NearMiss Test_Set - 2019")
print('XGB F1-Score', fxg * 100)
print('XGB ROCAUC Score:', rocauc_xg * 100)
print('XGB Recall score:', recall_xg * 100)
print('XGB Precision Score:', precisions_xg * 100)
print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, y2_test), "\n")
print('XGB Classification', classification_report(xgb_y_predict, y2_test), "\n")
print('XGB Accuracy Score', accuracys_nb * 100)
print("Execution Time for XGBoost Classifier NearMiss Test_Set - 2019", xg_end - start4, "seconds")

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

start5 = time.time()
rfc_model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(x2_train, y2_train)
probs_rf = rfc_model.predict_proba(x2_test)[:, 1]
probs_rf_scol2.append(probs_rf)
rfc_pred = rfc_model.predict(x2_test)
frfc = f1_score(rfc_pred, y2_test)
f1_rf_scol2.append(frfc)
rocauc_rf = roc_auc_score(y2_test, rfc_pred)
rocauc_rf_scol2.append(rocauc_rf)
recalls_rf = recall_score(rfc_pred, y2_test)
recall_rf_scol2.append(recalls_rf)
precisions_rf = precision_score(rfc_pred, y2_test)
precision_rf_scol2.append(precisions_rf)
accuracys_rf = accuracy_score(rfc_pred, y2_test)
accuracy_rf_scol2.append(accuracys_rf)
rf_end = time.time()
print("====RandomForest with Tfidf NearMiss Test_Set - 2019")
print('RFC F1 score', frfc * 100)
print('RFC ROCAUC Score:', rocauc_rf * 100)
print('RFC Recall score:', recalls_rf * 100)
print('RFC Precision Score:', precisions_rf * 100)
print('RFC Confusion Matrix', confusion_matrix(y2_test, rfc_pred), "\n")
print('RFC Classification', classification_report(y2_test, rfc_pred), "\n")
print('RFC Accuracy Score', accuracys_rf * 100)
print("Execution Time for Random Forest Classifier NearMiss Test_Set - 2019 ", rf_end - start5, "seconds")

print("Array of Prob Scores LR-Sam NearMiss", ":", probs_lr_scol2)
print("Array of F1 Scores LR-Sam NearMiss", ":", f1_lr_scol2)
print("Array of ROCAUC Scores LR-Sam NearMiss", ":", rocauc_lr_scol2)
print("Array of Recall Scores LR-Sam NearMiss", ":", recall_lr_scol2)
print("Array of Precision Scores LR-Sam NearMiss", ":", precision_lr_scol2)
print("Array of Accuracy Scores LR-Sam NearMiss", ":", accuracy_lr_scol2)

print("Array of Prob Scores DT-Sam NearMiss", ":", probs_dt_scol2)
print("Array of F1 Scores DT-Sam NearMiss", ":", f1_dt_scol2)
print("Array of ROCAUC Scores DT-Sam NearMiss", ":", rocauc_dt_scol2)
print("Array of Recall Scores DT-Sam NearMiss", ":", recall_dt_scol2)
print("Array of Precision Scores DT-Sam NearMiss", ":", precision_dt_scol2)
print("Array of Accuracy Scores DT-Sam NearMiss", ":", accuracy_dt_scol2)

print("Array of Prob Scores NB-Sam NearMiss", ":", probs_nb_scol2)
print("Array of F1 Scores NB-Sam NearMiss", ":", f1_nb_scol2)
print("Array of ROCAUC Scores NB-Sam NearMiss", ":", rocauc_nb_scol2)
print("Array of Recall Scores NB-Sam NearMiss", ":", recall_nb_scol2)
print("Array of Precision Scores NB-Sam NearMiss", ":", precision_nb_scol2)
print("Array of Accuracy Scores NB-Sam NearMiss", ":", accuracy_nb_scol2)

print("Array of Prob Scores XG-Sam NearMiss", ":", probs_xg_scol2)
print("Array of F1 Scores XG-Sam NearMiss", ":", f1_xg_scol2)
print("Array of ROCAUC Scores XG-Sam NearMiss", ":", rocauc_xg_scol2)
print("Array of Recall Scores XG-Sam NearMiss", ":", recall_xg_scol2)
print("Array of Precision Scores XG-Sam NearMiss", ":", precision_xg_scol2)
print("Array of Accuracy Scores XG-Sam NearMiss", ":", accuracy_xg_scol2)

print("Array of Prob Scores RF-Sam NearMiss", ":", probs_rf_scol2)
print("Array of F1 Scores RF-Sam NearMiss", ":", f1_rf_scol2)
print("Array of ROCAUC Scores RF-Sam NearMiss", ":", rocauc_rf_scol2)
print("Array of Recall Scores RF-Sam NearMiss", ":", recall_rf_scol2)
print("Array of Precision Scores RF-Sam NearMiss", ":", precision_rf_scol2)
print("Array of Accuracy Scores RF-Sam NearMiss", ":", accuracy_rf_scol2)

name2 = ['Test_Set' for t in range(5)]
sampling2 = ['Undersampling' for t in range(5)]
technique2 = ['NearMiss' for t in range(5)]
classifier_names2 = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
test_sizes_num2 = [0.05]
train_sizes_num2 = [0.95]
# v = [0, 2, 2, 3, 4]
# precision_csv_num = [precision_lr_scol[z], precision_dt_scol[z], precision_nb_scol[z], precision_xg_scol[z], precision_rf_scol[z]]
# recall_csv_num = [recall_lr_scol[z], recall_dt_scol[z], recall_nb_scol[z], recall_xg_scol[z], recall_rf_scol[z]]
# auc_csv_num = [rocauc_lr_scol[z], rocauc_dt_scol[z], rocauc_nb_scol[z], rocauc_xg_scol[z], rocauc_rf_scol[z]]
# accuracy_csv_num = [accuracy_lr_scol[z], accuracy_dt_scol[z], accuracy_nb_scol[z], accuracy_xg_scol[z], accuracy_rf_scol[z]]
precision_csv_num2 = [precision_lr_scol2, precision_dt_scol2, precision_nb_scol2, precision_xg_scol2,
                      precision_rf_scol2]
recall_csv_num2 = [recall_lr_scol2, recall_dt_scol2, recall_nb_scol2, recall_xg_scol2, recall_rf_scol2]
auc_csv_num2 = [rocauc_lr_scol2, rocauc_dt_scol2, rocauc_nb_scol2, rocauc_xg_scol2, rocauc_rf_scol2]
accuracy_csv_num2 = [accuracy_lr_scol2, accuracy_dt_scol2, accuracy_nb_scol2, accuracy_xg_scol2, accuracy_rf_scol2]
import itertools

rounds = 1
p2 = itertools.cycle(classifier_names2)
o2 = itertools.cycle(test_sizes_num2)
k2 = itertools.cycle(train_sizes_num2)
# v = itertools.cycle(score_location)
# pr = itertools.cycle(precision_num)
# y = itertools.cycle(iteration_csv)
classifier_csv2 = [next(p2) for _ in range(rounds)] * 5
test_size_csv2 = [next(o2) for _ in range(rounds)] * 5
train_size_csv2 = [next(k2) for _ in range(rounds)] * 5
# split_csv = ['2' for t in range(25)]
# train_csv = ['0.5' for t in range(25)]
precision_csv2 = list(chain(*precision_csv_num2))
recall_csv2 = list(chain(*recall_csv_num2))
auc_csv2 = list(chain(*auc_csv_num2))
accuracy_csv2 = list(chain(*accuracy_csv_num2))
csv_data2 = [name2, sampling2, technique2, classifier_csv2, test_size_csv2, train_size_csv2, precision_csv2,
             recall_csv2, auc_csv2, accuracy_csv2]
export_data2 = zip_longest(*csv_data2, fillvalue='')
filename = "Test_Iterations.csv"
with open(filename, 'a', newline='') as file:
    write = csv.writer(file)
    # write.writerow(("Name", "Sampling", "Technique", "Classifier", "Test Split Size", "Train Split Size", "Precision", "Recall", "AUC", "Accuracy"))
    write.writerows(export_data2)

# SMOTE
probs_lr_scol3 = []
f1_lr_scol3 = []
rocauc_lr_scol3 = []
recall_lr_scol3 = []
precision_lr_scol3 = []
accuracy_lr_scol3 = []

probs_dt_scol3 = []
f1_dt_scol3 = []
rocauc_dt_scol3 = []
recall_dt_scol3 = []
precision_dt_scol3 = []
accuracy_dt_scol3 = []

probs_nb_scol3 = []
f1_nb_scol3 = []
rocauc_nb_scol3 = []
recall_nb_scol3 = []
precision_nb_scol3 = []
accuracy_nb_scol3 = []

probs_xg_scol3 = []
f1_xg_scol3 = []
rocauc_xg_scol3 = []
recall_xg_scol3 = []
precision_xg_scol3 = []
accuracy_xg_scol3 = []

probs_rf_scol3 = []
f1_rf_scol3 = []
rocauc_rf_scol3 = []
recall_rf_scol3 = []
precision_rf_scol3 = []
accuracy_rf_scol3 = []

x3_train, x3_test, y3_train, y3_test = train_test_split(x_smote, y_smote, train_size=0.95, test_size=0.05)

start1 = time.time()
log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
model_lr = log.fit(x3_train, y3_train)
probs_lr = model_lr.predict_proba(x3_test)[:, 1]
probs_lr_scol3.append(probs_lr)
ly_prediction = log.predict(x3_test)
fly = f1_score(ly_prediction, y3_test)
f1_lr_scol3.append(fly)
rocauc_lr = roc_auc_score(y3_test, ly_prediction)
rocauc_lr_scol3.append(rocauc_lr)
recalls_lr = recall_score(y3_test, ly_prediction)
recall_lr_scol3.append(recalls_lr)
precisions_lr = precision_score(y3_test, ly_prediction)
precision_lr_scol3.append(precisions_lr)
accuracys_lr = accuracy_score(y3_test, ly_prediction)
accuracy_lr_scol3.append(accuracys_lr)
print("===Logistic Regression with TfidfVectorizer SMOTE Test_Set - 2019")
lr_end = time.time()
print('Logistic F1-score', fly * 100)
print('Logistic ROCAUC score:', rocauc_lr * 100)
print('Logistic Recall score:', recalls_lr * 100)
print('Logistic Precision Score:', precisions_lr * 100)
print('Logistic Confusion Matrix', confusion_matrix(y3_test, ly_prediction), "\n")
print('Logistic Classification', classification_report(y3_test, ly_prediction), "\n")
print('Logistic Accuracy Score', accuracys_lr * 100)
print("Execution Time for Logistic Regression SMOTE Test_Set - 2019", lr_end - start1, "seconds")

start2 = time.time()
from sklearn.tree import DecisionTreeClassifier

DCT = DecisionTreeClassifier()
model_dt = DCT.fit(x3_train, y3_train)
probs_dt = model_dt.predict_proba(x3_test)[:, 1]
probs_dt_scol3.append(probs_dt)
dct_pred = DCT.predict(x3_test)
fdct = f1_score(dct_pred, y3_test)
f1_dt_scol3.append(fdct)
rocauc_dt = roc_auc_score(y3_test, dct_pred)
rocauc_dt_scol3.append(rocauc_dt)
recalls_dt = recall_score(y3_test, dct_pred)
recall_dt_scol3.append(recalls_dt)
precisions_dt = precision_score(y3_test, dct_pred)
precision_dt_scol3.append(precisions_dt)
accuracys_dt = accuracy_score(y3_test, dct_pred)
accuracy_dt_scol3.append(accuracys_dt)
print("===DecisionTreeClassifier with TfidfVectorizer SMOTE Test_Set - 2019")
dt_end = time.time()
print('DCT F1-score', fdct * 100)
print('DCT ROCAUC score:', rocauc_dt * 100)
print('DCT Recall score:', recalls_dt * 100)
print('DCT Precision Score:', precisions_dt * 100)
print('DCT Confusion Matrix', confusion_matrix(y3_test, dct_pred), "\n")
print('DCT Classification', classification_report(y3_test, dct_pred), "\n")
print('DCT Accuracy Score', accuracys_dt * 100)
print("Execution Time for Decision Tree SMOTE Test_Set - 2019", dt_end - start2, "seconds")

from sklearn.naive_bayes import MultinomialNB

start3 = time.time()
Naive = MultinomialNB()
model_nb = Naive.fit(x3_train, y3_train)
probs_nb = model_nb.predict_proba(x3_test)[:, 1]
probs_nb_scol3.append(probs_nb)
# predict the labels on validation dataset
ny_pred = Naive.predict(x3_test)
fnb = f1_score(ny_pred, y3_test)
f1_nb_scol3.append(fnb)
rocauc_nb = roc_auc_score(y3_test, ny_pred)
rocauc_nb_scol3.append(rocauc_nb)
recalls_nb = recall_score(y3_test, ny_pred)
recall_nb_scol3.append(recalls_nb)
precisions_nb = precision_score(y3_test, ny_pred)
precision_nb_scol3.append(precisions_nb)
accuracys_nb = accuracy_score(y3_test, ny_pred)
accuracy_nb_scol3.append(accuracys_nb)
nb_end = time.time()
# Use accuracy_score function to get the accuracy
print("===Naive Bayes with TfidfVectorizer Imabalanced SMOTE Test_Set - 2019")
print('Naive F1-score', fnb * 100)
print('Naive ROCAUC score:', rocauc_nb * 100)
print('Naive Recall score:', recalls_nb * 100)
print('Naive Precision Score:', precisions_nb * 100)
print('Naive Confusion Matrix', confusion_matrix(y3_test, ny_pred), "\n")
print('Naive Classification', classification_report(y3_test, ny_pred), "\n")
print('Naive Accuracy Score', accuracys_nb * 100)
print("Execution Time for Naive Bayes SMOTE Test_Set - 2019", nb_end - start3, "seconds")

# XGBoost Classifier

start4 = time.time()
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

xgb_model = XGBClassifier().fit(x3_train, y3_train)
probs_xg = xgb_model.predict_proba(x3_test)[:, 1]
probs_xg_scol3.append(probs_xg)
# predict
xgb_y_predict = xgb_model.predict(x3_test)
fxg = f1_score(xgb_y_predict, y3_test)
f1_xg_scol3.append(fxg)
rocauc_xg = roc_auc_score(xgb_y_predict, y3_test)
rocauc_xg_scol3.append(rocauc_xg)
recall_xg = recall_score(xgb_y_predict, y3_test)
recall_xg_scol3.append(recall_xg)
precisions_xg = precision_score(xgb_y_predict, y3_test)
precision_xg_scol3.append(precisions_xg)
accuracys_xg = accuracy_score(xgb_y_predict, y3_test)
accuracy_xg_scol3.append(accuracys_xg)
xg_end = time.time()
print("===XGB with TfidfVectorizer SMOTE Test_Set - 2019")
print('XGB F1-Score', fxg * 100)
print('XGB ROCAUC Score:', rocauc_xg * 100)
print('XGB Recall score:', recall_xg * 100)
print('XGB Precision Score:', precisions_xg * 100)
print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, y3_test), "\n")
print('XGB Classification', classification_report(xgb_y_predict, y3_test), "\n")
print('XGB Accuracy Score', accuracys_nb * 100)
print("Execution Time for XGBoost Classifier SMOTE Test_Set - 2019", xg_end - start4, "seconds")

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

start5 = time.time()
rfc_model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(x3_train, y3_train)
probs_rf = rfc_model.predict_proba(x3_test)[:, 1]
probs_rf_scol3.append(probs_rf)
rfc_pred = rfc_model.predict(x3_test)
frfc = f1_score(rfc_pred, y3_test)
f1_rf_scol3.append(frfc)
rocauc_rf = roc_auc_score(y3_test, rfc_pred)
rocauc_rf_scol3.append(rocauc_rf)
recalls_rf = recall_score(rfc_pred, y3_test)
recall_rf_scol3.append(recalls_rf)
precisions_rf = precision_score(rfc_pred, y3_test)
precision_rf_scol3.append(precisions_rf)
accuracys_rf = accuracy_score(rfc_pred, y3_test)
accuracy_rf_scol3.append(accuracys_rf)
rf_end = time.time()
print("====RandomForest with Tfidf SMOTE Test_Set - 2019")
print('RFC F1 score', frfc * 100)
print('RFC ROCAUC Score:', rocauc_rf * 100)
print('RFC Recall score:', recalls_rf * 100)
print('RFC Precision Score:', precisions_rf * 100)
print('RFC Confusion Matrix', confusion_matrix(y3_test, rfc_pred), "\n")
print('RFC Classification', classification_report(y3_test, rfc_pred), "\n")
print('RFC Accuracy Score', accuracys_rf * 100)
print("Execution Time for Random Forest Classifier SMOTE Test_Set - 2019 ", rf_end - start5, "seconds")

print("Array of Prob Scores LR-Sam SMOTE", ":", probs_lr_scol3)
print("Array of F1 Scores LR-Sam SMOTE", ":", f1_lr_scol3)
print("Array of ROCAUC Scores LR-Sam SMOTE", ":", rocauc_lr_scol3)
print("Array of Recall Scores LR-Sam SMOTE", ":", recall_lr_scol3)
print("Array of Precision Scores LR-Sam SMOTE", ":", precision_lr_scol3)
print("Array of Accuracy Scores LR-Sam SMOTE", ":", accuracy_lr_scol3)

print("Array of Prob Scores DT-Sam SMOTE", ":", probs_dt_scol3)
print("Array of F1 Scores DT-Sam SMOTE", ":", f1_dt_scol3)
print("Array of ROCAUC Scores DT-Sam SMOTE", ":", rocauc_dt_scol3)
print("Array of Recall Scores DT-Sam SMOTE", ":", recall_dt_scol3)
print("Array of Precision Scores DT-Sam SMOTE", ":", precision_dt_scol3)
print("Array of Accuracy Scores DT-Sam SMOTE", ":", accuracy_dt_scol3)

print("Array of Prob Scores NB-Sam SMOTE", ":", probs_nb_scol3)
print("Array of F1 Scores NB-Sam SMOTE", ":", f1_nb_scol3)
print("Array of ROCAUC Scores NB-Sam SMOTE", ":", rocauc_nb_scol3)
print("Array of Recall Scores NB-Sam SMOTE", ":", recall_nb_scol3)
print("Array of Precision Scores NB-Sam SMOTE", ":", precision_nb_scol3)
print("Array of Accuracy Scores NB-Sam SMOTE", ":", accuracy_nb_scol3)

print("Array of Prob Scores XG-Sam SMOTE", ":", probs_xg_scol3)
print("Array of F1 Scores XG-Sam SMOTE", ":", f1_xg_scol3)
print("Array of ROCAUC Scores XG-Sam SMOTE", ":", rocauc_xg_scol3)
print("Array of Recall Scores XG-Sam SMOTE", ":", recall_xg_scol3)
print("Array of Precision Scores XG-Sam SMOTE", ":", precision_xg_scol3)
print("Array of Accuracy Scores XG-Sam SMOTE", ":", accuracy_xg_scol3)

print("Array of Prob Scores RF-Sam SMOTE", ":", probs_rf_scol3)
print("Array of F1 Scores RF-Sam SMOTE", ":", f1_rf_scol3)
print("Array of ROCAUC Scores RF-Sam SMOTE", ":", rocauc_rf_scol3)
print("Array of Recall Scores RF-Sam SMOTE", ":", recall_rf_scol3)
print("Array of Precision Scores RF-Sam SMOTE", ":", precision_rf_scol3)
print("Array of Accuracy Scores RF-Sam SMOTE", ":", accuracy_rf_scol3)

name3 = ['Test_Set' for t in range(5)]
sampling3 = ['Oversampling' for t in range(5)]
technique3 = ['SMOTE' for t in range(5)]
classifier_names3 = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
test_sizes_num3 = [0.05]
train_sizes_num3 = [0.95]
# v = [0, 3, 3, 3, 4]
# precision_csv_num = [precision_lr_scol[z], precision_dt_scol[z], precision_nb_scol[z], precision_xg_scol[z], precision_rf_scol[z]]
# recall_csv_num = [recall_lr_scol[z], recall_dt_scol[z], recall_nb_scol[z], recall_xg_scol[z], recall_rf_scol[z]]
# auc_csv_num = [rocauc_lr_scol[z], rocauc_dt_scol[z], rocauc_nb_scol[z], rocauc_xg_scol[z], rocauc_rf_scol[z]]
# accuracy_csv_num = [accuracy_lr_scol[z], accuracy_dt_scol[z], accuracy_nb_scol[z], accuracy_xg_scol[z], accuracy_rf_scol[z]]
precision_csv_num3 = [precision_lr_scol3, precision_dt_scol3, precision_nb_scol3, precision_xg_scol3,
                      precision_rf_scol3]
recall_csv_num3 = [recall_lr_scol3, recall_dt_scol3, recall_nb_scol3, recall_xg_scol3, recall_rf_scol3]
auc_csv_num3 = [rocauc_lr_scol3, rocauc_dt_scol3, rocauc_nb_scol3, rocauc_xg_scol3, rocauc_rf_scol3]
accuracy_csv_num3 = [accuracy_lr_scol3, accuracy_dt_scol3, accuracy_nb_scol3, accuracy_xg_scol3, accuracy_rf_scol3]
import itertools
rounds = 1
p3 = itertools.cycle(classifier_names3)
o3 = itertools.cycle(test_sizes_num3)
k3 = itertools.cycle(train_sizes_num3)
# v = itertools.cycle(score_location)
# pr = itertools.cycle(precision_num)
# y = itertools.cycle(iteration_csv)
classifier_csv3 = [next(p3) for _ in range(rounds)] * 5
test_size_csv3 = [next(o3) for _ in range(rounds)] * 5
train_size_csv3 = [next(k3) for _ in range(rounds)] * 5
# split_csv = ['3' for t in range(35)]
# train_csv = ['0.5' for t in range(35)]
precision_csv3 = list(chain(*precision_csv_num3))
recall_csv3 = list(chain(*recall_csv_num3))
auc_csv3 = list(chain(*auc_csv_num3))
accuracy_csv3 = list(chain(*accuracy_csv_num3))
csv_data3 = [name3, sampling3, technique3, classifier_csv3, test_size_csv3, train_size_csv3, precision_csv3,
             recall_csv3, auc_csv3, accuracy_csv3]
export_data3 = zip_longest(*csv_data3, fillvalue='')
filename = "Test_Iterations.csv"
with open(filename, 'a', newline='') as file:
    write = csv.writer(file)
    # write.writerow(("Name", "Sampling", "Technique", "Classifier", "Test Split Size", "Train Split Size", "Precision", "Recall", "AUC", "Accuracy"))
    write.writerows(export_data3)

# ROS
probs_lr_scol4 = []
f1_lr_scol4 = []
rocauc_lr_scol4 = []
recall_lr_scol4 = []
precision_lr_scol4 = []
accuracy_lr_scol4 = []

probs_dt_scol4 = []
f1_dt_scol4 = []
rocauc_dt_scol4 = []
recall_dt_scol4 = []
precision_dt_scol4 = []
accuracy_dt_scol4 = []

probs_nb_scol4 = []
f1_nb_scol4 = []
rocauc_nb_scol4 = []
recall_nb_scol4 = []
precision_nb_scol4 = []
accuracy_nb_scol4 = []

probs_xg_scol4 = []
f1_xg_scol4 = []
rocauc_xg_scol4 = []
recall_xg_scol4 = []
precision_xg_scol4 = []
accuracy_xg_scol4 = []

probs_rf_scol4 = []
f1_rf_scol4 = []
rocauc_rf_scol4 = []
recall_rf_scol4 = []
precision_rf_scol4 = []
accuracy_rf_scol4 = []

x4_train, x4_test, y4_train, y4_test = train_test_split(x_ros, y_ros, train_size=0.95, test_size=0.05)

start1 = time.time()
log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
model_lr = log.fit(x4_train, y4_train)
probs_lr = model_lr.predict_proba(x4_test)[:, 1]
probs_lr_scol4.append(probs_lr)
ly_prediction = log.predict(x4_test)
fly = f1_score(ly_prediction, y4_test)
f1_lr_scol4.append(fly)
rocauc_lr = roc_auc_score(y4_test, ly_prediction)
rocauc_lr_scol4.append(rocauc_lr)
recalls_lr = recall_score(y4_test, ly_prediction)
recall_lr_scol4.append(recalls_lr)
precisions_lr = precision_score(y4_test, ly_prediction)
precision_lr_scol4.append(precisions_lr)
accuracys_lr = accuracy_score(y4_test, ly_prediction)
accuracy_lr_scol4.append(accuracys_lr)
print("===Logistic Regression with TfidfVectorizer ROS Test_Set - 2019")
lr_end = time.time()
print('Logistic F1-score', fly * 100)
print('Logistic ROCAUC score:', rocauc_lr * 100)
print('Logistic Recall score:', recalls_lr * 100)
print('Logistic Precision Score:', precisions_lr * 100)
print('Logistic Confusion Matrix', confusion_matrix(y4_test, ly_prediction), "\n")
print('Logistic Classification', classification_report(y4_test, ly_prediction), "\n")
print('Logistic Accuracy Score', accuracys_lr * 100)
print("Execution Time for Logistic Regression ROS Test_Set - 2019", lr_end - start1, "seconds")

start2 = time.time()
from sklearn.tree import DecisionTreeClassifier

DCT = DecisionTreeClassifier()
model_dt = DCT.fit(x4_train, y4_train)
probs_dt = model_dt.predict_proba(x4_test)[:, 1]
probs_dt_scol4.append(probs_dt)
dct_pred = DCT.predict(x4_test)
fdct = f1_score(dct_pred, y4_test)
f1_dt_scol4.append(fdct)
rocauc_dt = roc_auc_score(y4_test, dct_pred)
rocauc_dt_scol4.append(rocauc_dt)
recalls_dt = recall_score(y4_test, dct_pred)
recall_dt_scol4.append(recalls_dt)
precisions_dt = precision_score(y4_test, dct_pred)
precision_dt_scol4.append(precisions_dt)
accuracys_dt = accuracy_score(y4_test, dct_pred)
accuracy_dt_scol4.append(accuracys_dt)
print("===DecisionTreeClassifier with TfidfVectorizer ROS Test_Set - 2019")
dt_end = time.time()
print('DCT F1-score', fdct * 100)
print('DCT ROCAUC score:', rocauc_dt * 100)
print('DCT Recall score:', recalls_dt * 100)
print('DCT Precision Score:', precisions_dt * 100)
print('DCT Confusion Matrix', confusion_matrix(y4_test, dct_pred), "\n")
print('DCT Classification', classification_report(y4_test, dct_pred), "\n")
print('DCT Accuracy Score', accuracys_dt * 100)
print("Execution Time for Decision Tree ROS Test_Set - 2019", dt_end - start2, "seconds")

from sklearn.naive_bayes import MultinomialNB

start3 = time.time()
Naive = MultinomialNB()
model_nb = Naive.fit(x4_train, y4_train)
probs_nb = model_nb.predict_proba(x4_test)[:, 1]
probs_nb_scol4.append(probs_nb)
# predict the labels on validation dataset
ny_pred = Naive.predict(x4_test)
fnb = f1_score(ny_pred, y4_test)
f1_nb_scol4.append(fnb)
rocauc_nb = roc_auc_score(y4_test, ny_pred)
rocauc_nb_scol4.append(rocauc_nb)
recalls_nb = recall_score(y4_test, ny_pred)
recall_nb_scol4.append(recalls_nb)
precisions_nb = precision_score(y4_test, ny_pred)
precision_nb_scol4.append(precisions_nb)
accuracys_nb = accuracy_score(y4_test, ny_pred)
accuracy_nb_scol4.append(accuracys_nb)
nb_end = time.time()
# Use accuracy_score function to get the accuracy
print("===Naive Bayes with TfidfVectorizer Imabalanced ROS Test_Set - 2019")
print('Naive F1-score', fnb * 100)
print('Naive ROCAUC score:', rocauc_nb * 100)
print('Naive Recall score:', recalls_nb * 100)
print('Naive Precision Score:', precisions_nb * 100)
print('Naive Confusion Matrix', confusion_matrix(y4_test, ny_pred), "\n")
print('Naive Classification', classification_report(y4_test, ny_pred), "\n")
print('Naive Accuracy Score', accuracys_nb * 100)
print("Execution Time for Naive Bayes ROS Test_Set - 2019", nb_end - start3, "seconds")

# XGBoost Classifier

start4 = time.time()
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

xgb_model = XGBClassifier().fit(x4_train, y4_train)
probs_xg = xgb_model.predict_proba(x4_test)[:, 1]
probs_xg_scol4.append(probs_xg)
# predict
xgb_y_predict = xgb_model.predict(x4_test)
fxg = f1_score(xgb_y_predict, y4_test)
f1_xg_scol4.append(fxg)
rocauc_xg = roc_auc_score(xgb_y_predict, y4_test)
rocauc_xg_scol4.append(rocauc_xg)
recall_xg = recall_score(xgb_y_predict, y4_test)
recall_xg_scol4.append(recall_xg)
precisions_xg = precision_score(xgb_y_predict, y4_test)
precision_xg_scol4.append(precisions_xg)
accuracys_xg = accuracy_score(xgb_y_predict, y4_test)
accuracy_xg_scol4.append(accuracys_xg)
xg_end = time.time()
print("===XGB with TfidfVectorizer ROS Test_Set - 2019")
print('XGB F1-Score', fxg * 100)
print('XGB ROCAUC Score:', rocauc_xg * 100)
print('XGB Recall score:', recall_xg * 100)
print('XGB Precision Score:', precisions_xg * 100)
print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, y4_test), "\n")
print('XGB Classification', classification_report(xgb_y_predict, y4_test), "\n")
print('XGB Accuracy Score', accuracys_nb * 100)
print("Execution Time for XGBoost Classifier ROS Test_Set - 2019", xg_end - start4, "seconds")

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

start5 = time.time()
rfc_model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(x4_train, y4_train)
probs_rf = rfc_model.predict_proba(x4_test)[:, 1]
probs_rf_scol4.append(probs_rf)
rfc_pred = rfc_model.predict(x4_test)
frfc = f1_score(rfc_pred, y4_test)
f1_rf_scol4.append(frfc)
rocauc_rf = roc_auc_score(y4_test, rfc_pred)
rocauc_rf_scol4.append(rocauc_rf)
recalls_rf = recall_score(rfc_pred, y4_test)
recall_rf_scol4.append(recalls_rf)
precisions_rf = precision_score(rfc_pred, y4_test)
precision_rf_scol4.append(precisions_rf)
accuracys_rf = accuracy_score(rfc_pred, y4_test)
accuracy_rf_scol4.append(accuracys_rf)
rf_end = time.time()
print("====RandomForest with Tfidf ROS Test_Set - 2019")
print('RFC F1 score', frfc * 100)
print('RFC ROCAUC Score:', rocauc_rf * 100)
print('RFC Recall score:', recalls_rf * 100)
print('RFC Precision Score:', precisions_rf * 100)
print('RFC Confusion Matrix', confusion_matrix(y4_test, rfc_pred), "\n")
print('RFC Classification', classification_report(y4_test, rfc_pred), "\n")
print('RFC Accuracy Score', accuracys_rf * 100)
print("Execution Time for Random Forest Classifier ROS Test_Set - 2019 ", rf_end - start5, "seconds")

print("Array of Prob Scores LR-Sam ROS", ":", probs_lr_scol4)
print("Array of F1 Scores LR-Sam ROS", ":", f1_lr_scol4)
print("Array of ROCAUC Scores LR-Sam ROS", ":", rocauc_lr_scol4)
print("Array of Recall Scores LR-Sam ROS", ":", recall_lr_scol4)
print("Array of Precision Scores LR-Sam ROS", ":", precision_lr_scol4)
print("Array of Accuracy Scores LR-Sam ROS", ":", accuracy_lr_scol4)

print("Array of Prob Scores DT-Sam ROS", ":", probs_dt_scol4)
print("Array of F1 Scores DT-Sam ROS", ":", f1_dt_scol4)
print("Array of ROCAUC Scores DT-Sam ROS", ":", rocauc_dt_scol4)
print("Array of Recall Scores DT-Sam ROS", ":", recall_dt_scol4)
print("Array of Precision Scores DT-Sam ROS", ":", precision_dt_scol4)
print("Array of Accuracy Scores DT-Sam ROS", ":", accuracy_dt_scol4)

print("Array of Prob Scores NB-Sam ROS", ":", probs_nb_scol4)
print("Array of F1 Scores NB-Sam ROS", ":", f1_nb_scol4)
print("Array of ROCAUC Scores NB-Sam ROS", ":", rocauc_nb_scol4)
print("Array of Recall Scores NB-Sam ROS", ":", recall_nb_scol4)
print("Array of Precision Scores NB-Sam ROS", ":", precision_nb_scol4)
print("Array of Accuracy Scores NB-Sam ROS", ":", accuracy_nb_scol4)

print("Array of Prob Scores XG-Sam ROS", ":", probs_xg_scol4)
print("Array of F1 Scores XG-Sam ROS", ":", f1_xg_scol4)
print("Array of ROCAUC Scores XG-Sam ROS", ":", rocauc_xg_scol4)
print("Array of Recall Scores XG-Sam ROS", ":", recall_xg_scol4)
print("Array of Precision Scores XG-Sam ROS", ":", precision_xg_scol4)
print("Array of Accuracy Scores XG-Sam ROS", ":", accuracy_xg_scol4)

print("Array of Prob Scores RF-Sam ROS", ":", probs_rf_scol4)
print("Array of F1 Scores RF-Sam ROS", ":", f1_rf_scol4)
print("Array of ROCAUC Scores RF-Sam ROS", ":", rocauc_rf_scol4)
print("Array of Recall Scores RF-Sam ROS", ":", recall_rf_scol4)
print("Array of Precision Scores RF-Sam ROS", ":", precision_rf_scol4)
print("Array of Accuracy Scores RF-Sam ROS", ":", accuracy_rf_scol4)

name4 = ['Test_Set' for t in range(5)]
sampling4 = ['Oversampling' for t in range(5)]
technique4 = ['ROS' for t in range(5)]
classifier_names4 = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
test_sizes_num4 = [0.05]
train_sizes_num4 = [0.95]
# v = [0, 4, 4, 4, 4]
# precision_csv_num = [precision_lr_scol[z], precision_dt_scol[z], precision_nb_scol[z], precision_xg_scol[z], precision_rf_scol[z]]
# recall_csv_num = [recall_lr_scol[z], recall_dt_scol[z], recall_nb_scol[z], recall_xg_scol[z], recall_rf_scol[z]]
# auc_csv_num = [rocauc_lr_scol[z], rocauc_dt_scol[z], rocauc_nb_scol[z], rocauc_xg_scol[z], rocauc_rf_scol[z]]
# accuracy_csv_num = [accuracy_lr_scol[z], accuracy_dt_scol[z], accuracy_nb_scol[z], accuracy_xg_scol[z], accuracy_rf_scol[z]]
precision_csv_num4 = [precision_lr_scol4, precision_dt_scol4, precision_nb_scol4, precision_xg_scol4,
                      precision_rf_scol4]
recall_csv_num4 = [recall_lr_scol4, recall_dt_scol4, recall_nb_scol4, recall_xg_scol4, recall_rf_scol4]
auc_csv_num4 = [rocauc_lr_scol4, rocauc_dt_scol4, rocauc_nb_scol4, rocauc_xg_scol4, rocauc_rf_scol4]
accuracy_csv_num4 = [accuracy_lr_scol4, accuracy_dt_scol4, accuracy_nb_scol4, accuracy_xg_scol4, accuracy_rf_scol4]
import itertools

rounds = 1
p4 = itertools.cycle(classifier_names4)
o4 = itertools.cycle(test_sizes_num4)
k4 = itertools.cycle(train_sizes_num4)
# v = itertools.cycle(score_location)
# pr = itertools.cycle(precision_num)
# y = itertools.cycle(iteration_csv)
classifier_csv4 = [next(p4) for _ in range(rounds)] * 5
test_size_csv4 = [next(o4) for _ in range(rounds)] * 5
train_size_csv4 = [next(k4) for _ in range(rounds)] * 5
# split_csv = ['4' for t in range(45)]
# train_csv = ['0.5' for t in range(45)]
precision_csv4 = list(chain(*precision_csv_num4))
recall_csv4 = list(chain(*recall_csv_num4))
auc_csv4 = list(chain(*auc_csv_num4))
accuracy_csv4 = list(chain(*accuracy_csv_num4))
csv_data4 = [name4, sampling4, technique4, classifier_csv4, test_size_csv4, train_size_csv4, precision_csv4,
             recall_csv4, auc_csv4, accuracy_csv4]
export_data4 = zip_longest(*csv_data4, fillvalue='')
filename = "Test_Iterations.csv"
with open(filename, 'a', newline='') as file:
    write = csv.writer(file)
    # write.writerow(("Name", "Sampling", "Technique", "Classifier", "Test Split Size", "Train Split Size", "Precision", "Recall", "AUC", "Accuracy"))
    write.writerows(export_data4)

# RUS
probs_lr_scol5 = []
f1_lr_scol5 = []
rocauc_lr_scol5 = []
recall_lr_scol5 = []
precision_lr_scol5 = []
accuracy_lr_scol5 = []

probs_dt_scol5 = []
f1_dt_scol5 = []
rocauc_dt_scol5 = []
recall_dt_scol5 = []
precision_dt_scol5 = []
accuracy_dt_scol5 = []

probs_nb_scol5 = []
f1_nb_scol5 = []
rocauc_nb_scol5 = []
recall_nb_scol5 = []
precision_nb_scol5 = []
accuracy_nb_scol5 = []

probs_xg_scol5 = []
f1_xg_scol5 = []
rocauc_xg_scol5 = []
recall_xg_scol5 = []
precision_xg_scol5 = []
accuracy_xg_scol5 = []

probs_rf_scol5 = []
f1_rf_scol5 = []
rocauc_rf_scol5 = []
recall_rf_scol5 = []
precision_rf_scol5 = []
accuracy_rf_scol5 = []

x5_train, x5_test, y5_train, y5_test = train_test_split(x_rus, y_rus, train_size=0.95, test_size=0.05)

start1 = time.time()
log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
model_lr = log.fit(x5_train, y5_train)
probs_lr = model_lr.predict_proba(x5_test)[:, 1]
probs_lr_scol5.append(probs_lr)
ly_prediction = log.predict(x5_test)
fly = f1_score(ly_prediction, y5_test)
f1_lr_scol5.append(fly)
rocauc_lr = roc_auc_score(y5_test, ly_prediction)
rocauc_lr_scol5.append(rocauc_lr)
recalls_lr = recall_score(y5_test, ly_prediction)
recall_lr_scol5.append(recalls_lr)
precisions_lr = precision_score(y5_test, ly_prediction)
precision_lr_scol5.append(precisions_lr)
accuracys_lr = accuracy_score(y5_test, ly_prediction)
accuracy_lr_scol5.append(accuracys_lr)
print("===Logistic Regression with TfidfVectorizer RUS Test_Set - 2019")
lr_end = time.time()
print('Logistic F1-score', fly * 100)
print('Logistic ROCAUC score:', rocauc_lr * 100)
print('Logistic Recall score:', recalls_lr * 100)
print('Logistic Precision Score:', precisions_lr * 100)
print('Logistic Confusion Matrix', confusion_matrix(y5_test, ly_prediction), "\n")
print('Logistic Classification', classification_report(y5_test, ly_prediction), "\n")
print('Logistic Accuracy Score', accuracys_lr * 100)
print("Execution Time for Logistic Regression RUS Test_Set - 2019", lr_end - start1, "seconds")

start2 = time.time()
from sklearn.tree import DecisionTreeClassifier

DCT = DecisionTreeClassifier()
model_dt = DCT.fit(x5_train, y5_train)
probs_dt = model_dt.predict_proba(x5_test)[:, 1]
probs_dt_scol5.append(probs_dt)
dct_pred = DCT.predict(x5_test)
fdct = f1_score(dct_pred, y5_test)
f1_dt_scol5.append(fdct)
rocauc_dt = roc_auc_score(y5_test, dct_pred)
rocauc_dt_scol5.append(rocauc_dt)
recalls_dt = recall_score(y5_test, dct_pred)
recall_dt_scol5.append(recalls_dt)
precisions_dt = precision_score(y5_test, dct_pred)
precision_dt_scol5.append(precisions_dt)
accuracys_dt = accuracy_score(y5_test, dct_pred)
accuracy_dt_scol5.append(accuracys_dt)
print("===DecisionTreeClassifier with TfidfVectorizer RUS Test_Set - 2019")
dt_end = time.time()
print('DCT F1-score', fdct * 100)
print('DCT ROCAUC score:', rocauc_dt * 100)
print('DCT Recall score:', recalls_dt * 100)
print('DCT Precision Score:', precisions_dt * 100)
print('DCT Confusion Matrix', confusion_matrix(y5_test, dct_pred), "\n")
print('DCT Classification', classification_report(y5_test, dct_pred), "\n")
print('DCT Accuracy Score', accuracys_dt * 100)
print("Execution Time for Decision Tree RUS Test_Set - 2019", dt_end - start2, "seconds")

from sklearn.naive_bayes import MultinomialNB

start3 = time.time()
Naive = MultinomialNB()
model_nb = Naive.fit(x5_train, y5_train)
probs_nb = model_nb.predict_proba(x5_test)[:, 1]
probs_nb_scol5.append(probs_nb)
# predict the labels on validation dataset
ny_pred = Naive.predict(x5_test)
fnb = f1_score(ny_pred, y5_test)
f1_nb_scol5.append(fnb)
rocauc_nb = roc_auc_score(y5_test, ny_pred)
rocauc_nb_scol5.append(rocauc_nb)
recalls_nb = recall_score(y5_test, ny_pred)
recall_nb_scol5.append(recalls_nb)
precisions_nb = precision_score(y5_test, ny_pred)
precision_nb_scol5.append(precisions_nb)
accuracys_nb = accuracy_score(y5_test, ny_pred)
accuracy_nb_scol5.append(accuracys_nb)
nb_end = time.time()
# Use accuracy_score function to get the accuracy
print("===Naive Bayes with TfidfVectorizer Imabalanced RUS Test_Set - 2019")
print('Naive F1-score', fnb * 100)
print('Naive ROCAUC score:', rocauc_nb * 100)
print('Naive Recall score:', recalls_nb * 100)
print('Naive Precision Score:', precisions_nb * 100)
print('Naive Confusion Matrix', confusion_matrix(y5_test, ny_pred), "\n")
print('Naive Classification', classification_report(y5_test, ny_pred), "\n")
print('Naive Accuracy Score', accuracys_nb * 100)
print("Execution Time for Naive Bayes RUS Test_Set - 2019", nb_end - start3, "seconds")

# XGBoost Classifier

start4 = time.time()
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

xgb_model = XGBClassifier().fit(x5_train, y5_train)
probs_xg = xgb_model.predict_proba(x5_test)[:, 1]
probs_xg_scol5.append(probs_xg)
# predict
xgb_y_predict = xgb_model.predict(x5_test)
fxg = f1_score(xgb_y_predict, y5_test)
f1_xg_scol5.append(fxg)
rocauc_xg = roc_auc_score(xgb_y_predict, y5_test)
rocauc_xg_scol5.append(rocauc_xg)
recall_xg = recall_score(xgb_y_predict, y5_test)
recall_xg_scol5.append(recall_xg)
precisions_xg = precision_score(xgb_y_predict, y5_test)
precision_xg_scol5.append(precisions_xg)
accuracys_xg = accuracy_score(xgb_y_predict, y5_test)
accuracy_xg_scol5.append(accuracys_xg)
xg_end = time.time()
print("===XGB with TfidfVectorizer RUS Test_Set - 2019")
print('XGB F1-Score', fxg * 100)
print('XGB ROCAUC Score:', rocauc_xg * 100)
print('XGB Recall score:', recall_xg * 100)
print('XGB Precision Score:', precisions_xg * 100)
print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, y5_test), "\n")
print('XGB Classification', classification_report(xgb_y_predict, y5_test), "\n")
print('XGB Accuracy Score', accuracys_nb * 100)
print("Execution Time for XGBoost Classifier RUS Test_Set - 2019", xg_end - start4, "seconds")

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

start5 = time.time()
rfc_model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(x5_train, y5_train)
probs_rf = rfc_model.predict_proba(x5_test)[:, 1]
probs_rf_scol5.append(probs_rf)
rfc_pred = rfc_model.predict(x5_test)
frfc = f1_score(rfc_pred, y5_test)
f1_rf_scol5.append(frfc)
rocauc_rf = roc_auc_score(y5_test, rfc_pred)
rocauc_rf_scol5.append(rocauc_rf)
recalls_rf = recall_score(rfc_pred, y5_test)
recall_rf_scol5.append(recalls_rf)
precisions_rf = precision_score(rfc_pred, y5_test)
precision_rf_scol5.append(precisions_rf)
accuracys_rf = accuracy_score(rfc_pred, y5_test)
accuracy_rf_scol5.append(accuracys_rf)
rf_end = time.time()
print("====RandomForest with Tfidf RUS Test_Set - 2019")
print('RFC F1 score', frfc * 100)
print('RFC ROCAUC Score:', rocauc_rf * 100)
print('RFC Recall score:', recalls_rf * 100)
print('RFC Precision Score:', precisions_rf * 100)
print('RFC Confusion Matrix', confusion_matrix(y5_test, rfc_pred), "\n")
print('RFC Classification', classification_report(y5_test, rfc_pred), "\n")
print('RFC Accuracy Score', accuracys_rf * 100)
print("Execution Time for Random Forest Classifier RUS Test_Set - 2019 ", rf_end - start5, "seconds")

print("Array of Prob Scores LR-Sam RUS", ":", probs_lr_scol5)
print("Array of F1 Scores LR-Sam RUS", ":", f1_lr_scol5)
print("Array of ROCAUC Scores LR-Sam RUS", ":", rocauc_lr_scol5)
print("Array of Recall Scores LR-Sam RUS", ":", recall_lr_scol5)
print("Array of Precision Scores LR-Sam RUS", ":", precision_lr_scol5)
print("Array of Accuracy Scores LR-Sam RUS", ":", accuracy_lr_scol5)

print("Array of Prob Scores DT-Sam RUS", ":", probs_dt_scol5)
print("Array of F1 Scores DT-Sam RUS", ":", f1_dt_scol5)
print("Array of ROCAUC Scores DT-Sam RUS", ":", rocauc_dt_scol5)
print("Array of Recall Scores DT-Sam RUS", ":", recall_dt_scol5)
print("Array of Precision Scores DT-Sam RUS", ":", precision_dt_scol5)
print("Array of Accuracy Scores DT-Sam RUS", ":", accuracy_dt_scol5)

print("Array of Prob Scores NB-Sam RUS", ":", probs_nb_scol5)
print("Array of F1 Scores NB-Sam RUS", ":", f1_nb_scol5)
print("Array of ROCAUC Scores NB-Sam RUS", ":", rocauc_nb_scol5)
print("Array of Recall Scores NB-Sam RUS", ":", recall_nb_scol5)
print("Array of Precision Scores NB-Sam RUS", ":", precision_nb_scol5)
print("Array of Accuracy Scores NB-Sam RUS", ":", accuracy_nb_scol5)

print("Array of Prob Scores XG-Sam RUS", ":", probs_xg_scol5)
print("Array of F1 Scores XG-Sam RUS", ":", f1_xg_scol5)
print("Array of ROCAUC Scores XG-Sam RUS", ":", rocauc_xg_scol5)
print("Array of Recall Scores XG-Sam RUS", ":", recall_xg_scol5)
print("Array of Precision Scores XG-Sam RUS", ":", precision_xg_scol5)
print("Array of Accuracy Scores XG-Sam RUS", ":", accuracy_xg_scol5)

print("Array of Prob Scores RF-Sam RUS", ":", probs_rf_scol5)
print("Array of F1 Scores RF-Sam RUS", ":", f1_rf_scol5)
print("Array of ROCAUC Scores RF-Sam RUS", ":", rocauc_rf_scol5)
print("Array of Recall Scores RF-Sam RUS", ":", recall_rf_scol5)
print("Array of Precision Scores RF-Sam RUS", ":", precision_rf_scol5)
print("Array of Accuracy Scores RF-Sam RUS", ":", accuracy_rf_scol5)

name5 = ['Test_Set' for t in range(5)]
sampling5 = ['Undersampling' for t in range(5)]
technique5 = ['RUS' for t in range(5)]
classifier_names5 = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
test_sizes_num5 = [0.05]
train_sizes_num5 = [0.95]
# v = [0, 5, 5, 5, 5]
# precision_csv_num = [precision_lr_scol[z], precision_dt_scol[z], precision_nb_scol[z], precision_xg_scol[z], precision_rf_scol[z]]
# recall_csv_num = [recall_lr_scol[z], recall_dt_scol[z], recall_nb_scol[z], recall_xg_scol[z], recall_rf_scol[z]]
# auc_csv_num = [rocauc_lr_scol[z], rocauc_dt_scol[z], rocauc_nb_scol[z], rocauc_xg_scol[z], rocauc_rf_scol[z]]
# accuracy_csv_num = [accuracy_lr_scol[z], accuracy_dt_scol[z], accuracy_nb_scol[z], accuracy_xg_scol[z], accuracy_rf_scol[z]]
precision_csv_num5 = [precision_lr_scol5, precision_dt_scol5, precision_nb_scol5, precision_xg_scol5,
                      precision_rf_scol5]
recall_csv_num5 = [recall_lr_scol5, recall_dt_scol5, recall_nb_scol5, recall_xg_scol5, recall_rf_scol5]
auc_csv_num5 = [rocauc_lr_scol5, rocauc_dt_scol5, rocauc_nb_scol5, rocauc_xg_scol5, rocauc_rf_scol5]
accuracy_csv_num5 = [accuracy_lr_scol5, accuracy_dt_scol5, accuracy_nb_scol5, accuracy_xg_scol5, accuracy_rf_scol5]
import itertools
rounds = 1
p5 = itertools.cycle(classifier_names5)
o5 = itertools.cycle(test_sizes_num5)
k5 = itertools.cycle(train_sizes_num5)
# v = itertools.cycle(score_location)
# pr = itertools.cycle(precision_num)
# y = itertools.cycle(iteration_csv)
classifier_csv5 = [next(p5) for _ in range(rounds)] * 5
test_size_csv5 = [next(o5) for _ in range(rounds)] * 5
train_size_csv5 = [next(k5) for _ in range(rounds)] * 5
# split_csv = ['5' for t in range(55)]
# train_csv = ['0.5' for t in range(55)]
precision_csv5 = list(chain(*precision_csv_num5))
recall_csv5 = list(chain(*recall_csv_num5))
auc_csv5 = list(chain(*auc_csv_num5))
accuracy_csv5 = list(chain(*accuracy_csv_num5))
csv_data5 = [name5, sampling5, technique5, classifier_csv5, test_size_csv5, train_size_csv5, precision_csv5,
             recall_csv5, auc_csv5, accuracy_csv5]
export_data5 = zip_longest(*csv_data5, fillvalue='')
filename = "Test_Iterations.csv"
with open(filename, 'a', newline='') as file:
    write = csv.writer(file)
    # write.writerow(("Name", "Sampling", "Technique", "Classifier", "Test Split Size", "Train Split Size", "Precision", "Recall", "AUC", "Accuracy"))
    write.writerows(export_data5)

#writef.close()
