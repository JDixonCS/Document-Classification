import pandas as pd
import numpy as np
import glob

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import re
import csv
import sys

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

writef = open("Output-AnnotatedReader.txt", "w")
sys.stdout = writef

path = r'C:\Users\Predator\Documents\Document-Classification\classifier\Annotated'
all_files = glob.glob(path + "/*.csv")
df_files = (pd.read_csv(f, encoding="unicode escape") for f in all_files)
df   = pd.concat(df_files, ignore_index=True)
print(df)
df["label"] = df["label"].fillna(0)
print(df)
df.drop(df.columns[[0, 3, 4, 5]], axis=1, inplace=True)
print(df)
df['sentence'] = df['sentence'].replace(['null'],'')
df['label'] = df['label'].replace(['null'], 0)
df['label'] = df['label'].replace(['Y','y','Y ','Maybe'], 1)
pos = len(df[df['label'] == 1])
print("Positive:", pos)
neg = len(df[df['label'] == 0])
print("Negative:", neg)
print(df)
pos_df = df.loc[df['label'] == 1]
print(pos_df)
neg_df = df.loc[df['label'] == 0]
print(neg_df)
df.to_csv('Full-Annotated.csv', encoding='utf-8')

#Splitting data into smaller dataframes for the purpose of Training and Testing
x1 = neg_df.iloc[0:287,0]
x2 = pos_df.iloc[0:287,0]
y1 = neg_df.iloc[0:287,1]
y2 = pos_df.iloc[0:287,1]
#x_seg = df.iloc[0:5,5]
#y_seg = df.iloc[0:5,1]
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

#Declaring and applying TFIDF functions to train and test data
tfidf_vect = TfidfVectorizer(ngram_range=(1,2))
tfidf_train = tfidf_vect.fit_transform(x1.values.astype('U'))
tfidf_test=tfidf_vect.transform(x2.values.astype('U'))
print(tfidf_train.shape)
print(tfidf_test.shape)
#tfidf_train.toarray()

#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

tfidf_vect = TfidfVectorizer()
x_tfidf = tfidf_vect.fit_transform(df["sentence"].astype('U'))

x_train, x_test, y_train, y_test = train_test_split(x_tfidf,df["label"], test_size=0.1, train_size=0.9)

log = LogisticRegression(penalty='l2',random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
model_lr = log.fit(x_train,y_train)
probs_lr = model_lr.predict_proba(x_test)[:, 1]
ly_prediction = log.predict(x_test)
fly = f1_score(ly_prediction,y_test)
print("===Logistic Regression with TfidfVectorizer Imbalanced Annotated - 2019===")
print('Logistic F1-score',fly*100)
print('Logistic ROCAUC score:',roc_auc_score(y_test, ly_prediction))
print('Logistic Confusion Matrix', confusion_matrix(y_test,ly_prediction), "\n")
print('Logistic Classification', classification_report(y_test,ly_prediction), "\n")
print('Logistic Accuracy Score', accuracy_score(y_test, ly_prediction)*100)

from sklearn.tree import DecisionTreeClassifier
DCT = DecisionTreeClassifier()
model_dt = DCT.fit(x_train, y_train)
probs_dt = model_dt.predict_proba(x_test)[:, 1]
dct_pred = DCT.predict(x_test)
fdct = f1_score(dct_pred,y_test)
print("===DecisionTreeClassifier with TfidfVectorizer Imbalanced Annotated - 2019===")
print('DCT F1-score',fdct*100)
print('DCT ROCAUC score:',roc_auc_score(y_test, dct_pred)*100, "\n")
print('DCT Confusion Matrix', confusion_matrix(y_test, dct_pred), "\n")
print('DCT Classification', classification_report(y_test, dct_pred), "\n")
print('DCT Accuracy Score', accuracy_score(y_test, dct_pred)*100)
'''
# C Support Vector Machine Classifier
from sklearn.svm import SVC
CSVM = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
CSVM.fit(x_train,y_train)
# predict the labels on validation dataset
csvy_pred = CSVM.predict(x_test)
fcsvm = f1_score(csvy_pred,y_test)
print("===C-SVM with TfidfVectorizer Imbalanced Annotated - 2019===")
print('C-SVM F1-score',fcsvm*100)
# Use accuracy_score function to get the accuracy
print('C-SVM ROCAUC score:',roc_auc_score(y_test, csvy_pred)*100, "\n")
print('C-SVM Confusion Matrix', confusion_matrix(y_test, csvy_pred), "\n")
print('C-SVM Classification', classification_report(y_test, csvy_pred), "\n")
print('C-SVM Accuracy Score', accuracy_score(y_test, csvy_pred)*100)
'''
from sklearn.naive_bayes import MultinomialNB
Naive = MultinomialNB()
model_nb = Naive.fit(x_train,y_train)
probs_nb = model_nb.predict_proba(x_test)[:, 1]
# predict the labels on validation dataset
ny_pred = Naive.predict(x_test)
fna = f1_score(ny_pred,y_test)
# Use accuracy_score function to get the accuracy
print("===Naive Bayes with TfidfVectorizer Imabalanced Annotated - 2019 ===")
print('Naive F1-score',fna*100)
print('Naive ROCAUC score:',roc_auc_score(y_test, ny_pred)*100, "\n")
print('Naive Confusion Matrix', confusion_matrix(y_test, ny_pred), "\n")
print('Naive Classification', classification_report(y_test, ny_pred), "\n")
print('Naive Accuracy Score', accuracy_score(y_test, ny_pred)*100)

# XGBoost Classifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
xgb_model = XGBClassifier().fit(x_train, y_train)
probs_xg = xgb_model.predict_proba(x_test)[:, 1]
# predict
xgb_y_predict = xgb_model.predict(x_test)
fxg = f1_score(xgb_y_predict,y_test)
print("===XGB with TfidfVectorizer Imbalanced Annotated - 2019===")
print('XGB F1-Score', fxg*100)
print('XGB ROCAUC Score:', roc_auc_score(xgb_y_predict, y_test)*100, "\n")
print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, y_test), "\n")
print('XGB Classification', classification_report(xgb_y_predict, y_test), "\n")
print('XGB Accuracy Score', accuracy_score(xgb_y_predict, y_test)*100)

'''
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=1000, random_state=0)
rfc.fit(x_train,y_train)
rfc_pred = rfc.predict(x_test)
frfc = f1_score(rfc_pred,y_test)
print("====RandomForest with Tfidf Imbalanced Annotated 2019====")
print('RFC F1 score', frfc*100)
print('RFC ROCAUC Score:', roc_auc_score(y_test, rfc_pred), "\n")
print('RFC Confusion Matrix', confusion_matrix(y_test,rfc_pred), "\n")
print('RF Classification', classification_report(y_test,rfc_pred), "\n")
print('RFC Accuracy Score', accuracy_score(y_test,rfc_pred)*100)
'''

writef.close()