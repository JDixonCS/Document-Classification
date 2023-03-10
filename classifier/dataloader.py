# -*- coding: utf-8 -*-
"""Imbalanced_Classes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GTN2Q9XyFLIyex-nmuU7PB8k4tUmSW-q
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import re
import csv

from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import os
# print(os.listdir("../input"))
plt.style.use('ggplot')

raw_2010_neg = open(r"C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\NIST_FULL\\2010-neg.txt", encoding="ISO-8859-1")
lines = raw_2010_neg.readlines()
raw_2010_neg.close()
df = pd.DataFrame()

# remove /n at the end of each line
for index, line in enumerate(lines):
    lines[index] = line.strip()
    print(lines[index])

neg_2010_df = pd.DataFrame(columns=['sentence'])
i = 0
first_col = ""
for line in lines:
    first_col = re.sub(r' \(.*', "", line)
    neg_2010_df.loc[i] = [first_col]
    i = i+1

neg_2010_df.head()
neg_2010_df['label'] = 0
neg_2010_df['year'] = 2010
#print(neg_2019_df)

raw_2010_pos = open(r"C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\NIST_FULL\\2010-pos.txt", encoding="ISO-8859-1")
lines1 = raw_2010_pos.readlines()
raw_2010_pos.close()

# remove /n at the end of each line
for index, line in enumerate(lines1):
    lines1[index] = line.strip()
    print(lines1[index])

pos_2010_df = pd.DataFrame(columns=['sentence'])
i = 0
first_col = ""
for line in lines1:
    first_col = re.sub(r' \(.*', "", line)
    pos_2010_df.loc[i] = [first_col]
    i = i+1

pos_2010_df.head()
pos_2010_df['label'] = 1
pos_2010_df['year'] = 2010
#print(pos_2019_df)

raw_2011_neg = open(r"C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\NIST_FULL\\2011-neg.txt", encoding="ISO-8859-1")
lines2 = raw_2011_neg.readlines()
raw_2011_neg.close()
df = pd.DataFrame()

# remove /n at the end of each line
for index, line in enumerate(lines2):
    lines2[index] = line.strip()
    print(lines2[index])

neg_2011_df = pd.DataFrame(columns=['sentence'])
i = 0
first_col = ""
for line in lines2:
    first_col = re.sub(r' \(.*', "", line)
    neg_2011_df.loc[i] = [first_col]
    i = i+1

neg_2011_df.head()
neg_2011_df['label'] = 0
neg_2011_df['year'] = 2011

raw_2011_pos = open(r"C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\NIST_FULL\\2011-pos.txt", encoding="ISO-8859-1")
lines3 = raw_2011_pos.readlines()
raw_2011_pos.close()
df = pd.DataFrame()

# remove /n at the end of each line
for index, line in enumerate(lines3):
    lines3[index] = line.strip()
    print(lines3[index])

pos_2011_df = pd.DataFrame(columns=['sentence'])
i = 0
first_col = ""
for line in lines2:
    first_col = re.sub(r' \(.*', "", line)
    pos_2011_df.loc[i] = [first_col]
    i = i+1

pos_2011_df.head()
pos_2011_df['label'] = 1
pos_2011_df['year'] = 2011

raw_2012_neg = open(r"C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\NIST_FULL\\2011-neg.txt", encoding="ISO-8859-1")
lines2 = raw_2012_neg.readlines()
raw_2012_neg.close()
df = pd.DataFrame()

# remove /n at the end of each line
for index, line in enumerate(lines3):
    lines3[index] = line.strip()
    print(lines3[index])

neg_2012_df = pd.DataFrame(columns=['sentence'])
i = 0
first_col = ""
for line in lines3:
    first_col = re.sub(r' \(.*', "", line)
    neg_2011_df.loc[i] = [first_col]
    i = i+1

neg_2012_df.head()
neg_2012_df['label'] = 0
neg_2012_df['year'] = 2012

df = df.append(neg_2010_df)
df = df.append(pos_2010_df)
df = df.append(neg_2011_df)
df = df.append(pos_2011_df)

print(df)
df.shape
