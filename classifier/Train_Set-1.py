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
writef = open("Output-Subset_Iterations.txt", "w")
sys.stdout = writef

train_set = r'C:\\Users\\Predator\\Documents\\Document-Classification\\classifier\\Train_Set.csv'
test_set = r'C:\\Users\\Predator\\Documents\\Document-Classification\\classifier\\Test_Set.csv'
dev_set = r'C:\\Users\\Predator\\Documents\\Document-Classification\\classifier\\Dev_Set.csv'
train_df = pd.read_csv(train_set, header=0, encoding="utf-8")
print(train_df)
dev_df = pd.read_csv(dev_set, header=0, encoding="utf-8")
print(dev_df)
test_df = pd.read_csv(test_set, header=0, encoding="utf-8")
print(test_df)

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
def remove_punc_train(text):
    clean = ''.join([x.lower() for x in text if x not in punc_set])
    return clean

#Applying the 'remove_punc' function to entire dataset
train_df['no_punc'] = train_df['sentence'].apply(lambda z:remove_punc_train(z))

#Function for Tokenizing entire data for representing every word as datapoint
def tokenize_train(text):
    tokens = re.split("\W+",text)
    return tokens

#Applying the 'tokenize' function to entire dataset
train_df['tokenized_Data'] = train_df['no_punc'].apply(lambda z:tokenize_train(z))

#Importing stopwords from NLTK Library to remove stopwords now that we have tokenized it
stopwords = nltk.corpus.stopwords.words('english')

#Function for removing stopwords from single row
def remove_stopwords_train(tokenized_words):
    Ligit_text=[word for word in tokenized_words if word not in stopwords]
    return Ligit_text

#Applying the function 'remove_stopwords' from the entire dataset
train_df["no_stop"] = train_df["tokenized_Data"].apply(lambda z:remove_stopwords_train(z))

#Importing 'WordNetLemmatizer' as lemmatizing function to find lemma's of words
wnl = nltk.wordnet.WordNetLemmatizer()

#Function for lemmatizing the tokenzied text
def lemmatizing_train(tokenized_text):
    lemma = [wnl.lemmatize(word) for word in tokenized_text]
    return lemma

#Applying the 'lemmatizing' function to entire dataset
train_df['lemmatized'] = train_df['no_stop'].apply(lambda z:lemmatizing_train(z))

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
train_df['lemmatized'] = [" ".join(review) for review in train_df['lemmatized'].values]


#Removing punctuations from entire dataset
punc_set = string.punctuation
punc_set

#Function for removing punctions
def remove_punc_dev(text):
    clean = ''.join([x.lower() for x in text if x not in punc_set])
    return clean

#Applying the 'remove_punc' function to entire dataset
dev_df['no_punc'] = dev_df['sentence'].apply(lambda z:remove_punc_dev(z))

#Function for Tokenizing entire data for representing every word as datapoint
def tokenize_dev(text):
    tokens = re.split("\W+",text)
    return tokens

#Applying the 'tokenize' function to entire dataset
dev_df['tokenized_Data'] = dev_df['no_punc'].apply(lambda z:tokenize_dev(z))

#Importing stopwords from NLTK Library to remove stopwords now that we have tokenized it
stopwords = nltk.corpus.stopwords.words('english')

#Function for removing stopwords from single row
def remove_stopwords_dev(tokenized_words):
    Ligit_text=[word for word in tokenized_words if word not in stopwords]
    return Ligit_text

#Applying the function 'remove_stopwords' from the entire dataset
dev_df["no_stop"] = dev_df["tokenized_Data"].apply(lambda z:remove_stopwords_dev(z))

#Importing 'WordNetLemmatizer' as lemmatizing function to find lemma's of words
wnl = nltk.wordnet.WordNetLemmatizer()

#Function for lemmatizing the tokenzied text
def lemmatizing_dev(tokenized_text):
    lemma = [wnl.lemmatize(word) for word in tokenized_text]
    return lemma

#Applying the 'lemmatizing' function to entire dataset
dev_df['lemmatized'] = dev_df['no_stop'].apply(lambda z:lemmatizing_dev(z))

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
dev_df['lemmatized'] = [" ".join(review) for review in dev_df['lemmatized'].values]

#This step is done here because, the 'lemmatized' column is a list of tokenized words and when we apply vectorization
#techniques such as count vectorizer or TFIDF, they require string input. Hence convert all tokenzied words to string
train_df['lemmatized'] = [" ".join(review) for review in train_df['lemmatized'].values]


#Removing punctuations from entire dataset
punc_set = string.punctuation
punc_set

#Function for removing punctions
def remove_punc_test(text):
    clean = ''.join([x.lower() for x in text if x not in punc_set])
    return clean

#Applying the 'remove_punc' function to entire dataset
test_df['no_punc'] = test_df['sentence'].apply(lambda z:remove_punc_test(z))

#Function for Tokenizing entire data for representing every word as datapoint
def tokenize_test(text):
    tokens = re.split("\W+",text)
    return tokens

#Applying the 'tokenize' function to entire dataset
test_df['tokenized_Data'] = test_df['no_punc'].apply(lambda z:tokenize_test(z))

#Importing stopwords from NLTK Library to remove stopwords now that we have tokenized it
stopwords = nltk.corpus.stopwords.words('english')

#Function for removing stopwords from single row
def remove_stopwords_test(tokenized_words):
    Ligit_text=[word for word in tokenized_words if word not in stopwords]
    return Ligit_text

#Applying the function 'remove_stopwords' from the entire dataset
test_df["no_stop"] = test_df["tokenized_Data"].apply(lambda z:remove_stopwords_test(z))

#Importing 'WordNetLemmatizer' as lemmatizing function to find lemma's of words
wnl = nltk.wordnet.WordNetLemmatizer()

#Function for lemmatizing the tokenzied text
def lemmatizing_test(tokenized_text):
    lemma = [wnl.lemmatize(word) for word in tokenized_text]
    return lemma

#Applying the 'lemmatizing' function to entire dataset
test_df['lemmatized'] = test_df['no_stop'].apply(lambda z:lemmatizing_test(z))

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
test_df['lemmatized'] = [" ".join(review) for review in test_df['lemmatized'].values]
'''

pos1 = len(train_df[train_df['label'] == 1])
print("Train Set Positive:", pos1)
neg1 = len(train_df[train_df['label'] == 0])
print("Train Set Negative:", neg1)
#print(train_df)
total1 = train_df.shape[0]
print("Train Set Total", total1)

pos2 = len(dev_df[dev_df['label'] == 1])
print("Dev Set Positive:", pos2)
neg2 = len(dev_df[dev_df['label'] == 0])
print("Dev Set Negative:", neg2)
#print(train_df)
total2 = dev_df.shape[0]
print("Dev Set Total", total2)

pos3 = len(test_df[test_df['label'] == 1])
print("Test Set Positive:", pos3)
neg3 = len(test_df[test_df['label'] == 0])
print("Test Set Negative:", neg3)
#print(train_df)
total3 = test_df.shape[0]
print("Test Set Total", total3)

# For Sampling Techniques

full_df = pd.concat([train_df, dev_df, test_df], axis=0, join='inner', ignore_index=True)
print(full_df)
print(full_df.shape[0])

train_dev_df = pd.concat([train_df, dev_df], axis=0, join='inner', ignore_index=True)
print(train_dev_df)
print(train_dev_df.shape[0])

train_test_df = pd.concat([train_df, test_df], axis=0, join='inner', ignore_index=True)
print(train_test_df)
print(train_test_df.shape[0])




import os
# For Sampling Techniques
full_df.to_csv("Train_Full_Splits.csv")
train_dev_df.to_csv("Train_Dev_Splits.csv")
train_test_df.to_csv("Train_Test_Splits.csv")

'''
full_merge_df = r'C:\\Users\\Predator\\Documents\\Document-Classification\\classifier\\Train_Full_Splits.csv'
train_dev_merge_df = r'C:\\Users\\Predator\\Documents\\Document-Classification\\classifier\\Train_Dev_Splits.csv'
train_test_merge_df = r'C:\\Users\\Predator\\Documents\\Document-Classification\\classifier\\Train_Test_Splits.csv'

full_df = pd.read_csv(full_df, header=0, sep=",", encoding="utf-8")
print(full_df)
train_dev_df = pd.read_csv(train_dev_df, header=0, sep=",", encoding="utf-8")
print(train_dev_df)
train_test_df = pd.read_csv(train_test_df, header=0, sep=",", encoding="utf-8")
print(train_test_df)
'''

# Splitting for Imbalanced Classification
train_part = train_df.iloc[0:10799, :]
print("Train Part:", train_part)
print(train_part.shape[0])
dev_part = dev_df.iloc[0:10799, :]
print("Dev Part:", dev_part)
print(dev_part.shape[0])
test_part = test_df.iloc[0:10799, :]
print("Test Part:", test_part)
print(test_part.shape[0])

import os
#import system

'''
full_split_df = pd.concat([train_dev_df, dev_part1_split, test_part1_split], axis=0, join='inner')
print(full_split_df)
print(full_split_df.shape[0])
'''

# Shaping for Sampling Techniques
trdev10 = round(train_dev_df.shape[0] * .10)
print("10% Train Dev Subsample", trdev10)
trdev20 = round(train_dev_df.shape[0] * .20)
print("20% Train Dev Subsample", trdev20)
trdev30 = round(train_dev_df.shape[0] * .30)
print("30% Train Dev Subsample", trdev30)
trdev40 = round(train_dev_df.shape[0] * .40)
print("40% Train Dev Subsample", trdev40)
trdev50 = round(train_dev_df.shape[0] * .50)
print("50% Train Dev Subsample", trdev50)
trdev60 = round(train_dev_df.shape[0] * .60)
print("60% Train Dev Subsample", trdev60)
trdev70 = round(train_dev_df.shape[0] * .70)
print("70% Train Dev Subsample", trdev70)
trdev80 = round(train_dev_df.shape[0] * .80)
print("80% Train Dev Subsample", trdev40)
trdev90 = round(train_dev_df.shape[0] * .90)
print("90% Train Dev Subsample", trdev90)
trdev100 = round(train_dev_df.shape[0] * 1.00)
print("100% Train Dev Subsample", trdev100)

trtes10 = round(train_test_df.shape[0] * .10)
print("10% Train Test Subsample", trtes10)
trtes20 = round(train_test_df.shape[0] * .20)
print("20% Train Test Subsample", trtes20)
trtes30 = round(train_test_df.shape[0] * .30)
print("30% Train Test Subsample", trtes30)
trtes40 = round(train_test_df.shape[0] * .40)
print("40% Train Test Subsample", trtes40)
trtes50 = round(train_test_df.shape[0] * .50)
print("50% Train Test Subsample", trtes50)
trtes60 = round(train_test_df.shape[0] * .60)
print("60% Train Test Subsample", trtes60)
trtes70 = round(train_test_df.shape[0] * .70)
print("70% Train Test Subsample", trtes70)
trtes80 = round(train_test_df.shape[0] * .80)
print("80% Train Test Subsample", trtes80)
trtes90 = round(train_test_df.shape[0] * .90)
print("90% Train Test Subsample", trtes90)
trtes100 = round(train_test_df.shape[0] * 1.00)
print("100% Train Test Subsample", trtes100)


# Shaping for Imbalanced Classifiers
train10 = round(train_part.shape[0] * .10)
print("10% Train Subsample", train10)
train20 = round(train_part.shape[0] * .20)
print("20% Train Subsample", train20)
train30 = round(train_part.shape[0] * .30)
print("30% Train Subsample", train30)
train40 = round(train_part.shape[0] * .40)
print("40% Train Subsample", train40)
train50 = round(train_part.shape[0] * .50)
print("50% Train Subsample", train50)
train60 = round(train_part.shape[0] * .60)
print("60% Train Subsample", train60)
train70 = round(train_part.shape[0] * .70)
print("70% Train Subsample", train70)
train80 = round(train_part.shape[0] * .80)
print("80% Train Subsample", train40)
train90 = round(train_part.shape[0] * .90)
print("90% Train Subsample", train90)
train100 = round(train_part.shape[0] * 1.00)
print("100% Train Subsample", train100)

dev10 = round(dev_part.shape[0] * .10) 
print("10% Dev Subsample", dev10)
dev20 = round(dev_part.shape[0] * .20)
print("20% Dev Subsample", dev20)
dev30 = round(dev_part.shape[0] * .30)
print("30% Dev Subsample", dev30)
dev40 = round(dev_part.shape[0] * .40)
print("40% Dev Subsample", dev40)
dev50 = round(dev_part.shape[0] * .50)
print("50% Dev Subsample", dev50)
dev60 = round(dev_part.shape[0] * .60)
print("60% Dev Subsample", dev60)
dev70 = round(dev_part.shape[0] * .70)
print("70% Dev Subsample", dev70)
dev80 = round(dev_part.shape[0] * .80) 
print("80% Dev Subsample", dev80)
dev90 = round(dev_part.shape[0] * .90)
print("90% Dev Subsample", dev90)
dev100 = round(dev_part.shape[0] * 1.00)
print("100% Dev Subsample", dev100)

test10 = round(test_part.shape[0] * .10)
print("10% Test Subsample", test10)
test20 = round(test_part.shape[0] * .20)
print("20% Test Subsample", test20)
test30 = round(test_part.shape[0] * .30)
print("30% Test Subsample", test30)
test40 = round(test_part.shape[0] * .40)
print("40% Test Subsample", test40)
test50 = round(test_part.shape[0] * .50)
print("50% Test Subsample", test50)
test60 = round(test_part.shape[0] * .60)
print("60% Test Subsample", test60)
test70 = round(test_part.shape[0] * .70)
print("70% Test Subsample", test70)
test80 = round(test_part.shape[0] * .80)
print("80% Test Subsample", test80)
test90 = round(test_part.shape[0] * .90)
print("90% Test Subsample", test90)
test100 = round(test_part.shape[0] * 1.00)
print("100% Test Subsample", test100)

# For Sampling Techniques
train_dev_split1 = train_dev_df.iloc[0:trdev10, :]
print(train_dev_split1)
train_test_split1 = train_test_df.iloc[0:trtes10, :]
print(train_test_split1)
train_dev_split2 = train_dev_df.iloc[0:trdev20, :]
print(train_dev_split2)
train_test_split2 = train_test_df.iloc[0:trtes20, :]
train_dev_split3 = train_dev_df.iloc[0:trdev30, :]
train_test_split3 = train_test_df.iloc[0:trtes30, :]
train_dev_split4 = train_dev_df.iloc[0:trdev40, :]
train_test_split4 = train_test_df.iloc[0:trtes40, :]
train_dev_split5 = train_dev_df.iloc[0:trdev50, :]
train_test_split5 = train_test_df.iloc[0:trtes50, :]
train_dev_split6 = train_dev_df.iloc[0:trdev60, :]
train_test_split6 = train_test_df.iloc[0:trtes60, :]
train_dev_split7 = train_dev_df.iloc[0:trdev70, :]
train_test_split7 = train_test_df.iloc[0:trtes70, :]
train_dev_split8 = train_dev_df.iloc[0:trdev80, :]
train_test_split8 = train_test_df.iloc[0:trtes80, :]
train_dev_split9 = train_dev_df.iloc[0:trdev90, :]
train_test_split9 = train_test_df.iloc[0:trtes90, :]
train_dev_split10 = train_dev_df.iloc[0:trdev100, :]
train_test_split10 = train_test_df.iloc[0:trtes100, :]



# For Imbalanced 
train_split1 = train_part.iloc[0:train10, :]
dev_split1 = dev_part.iloc[0:dev10, :]
test_split1 = test_part.iloc[0:test10, :]
train_split2 = train_part.iloc[0:train20, :]
dev_split2 = dev_part.iloc[0:dev20, :]
test_split2 = test_part.iloc[0:test20, :]
train_split3 = train_part.iloc[0:train30, :]
dev_split3 = dev_part.iloc[0:dev30, :]
test_split3 = test_part.iloc[0:test30, :]
train_split4 = train_part.iloc[0:train40, :]
dev_split4 = dev_part.iloc[0:dev40, :]
test_split4 = test_part.iloc[0:test40, :]
train_split5 = train_part.iloc[0:train50, :]
dev_split5 = dev_part.iloc[0:dev50, :]
test_split5 = test_part.iloc[0:test50, :]
train_split6 = train_part.iloc[0:train60, :]
dev_split6 = dev_part.iloc[0:dev60, :]
test_split6 = test_part.iloc[0:test60, :]
train_split7 = train_part.iloc[0:train70, :]
dev_split7 = dev_part.iloc[0:dev70, :]
test_split7 = test_part.iloc[0:test70, :]
train_split8 = train_part.iloc[0:train80, :]
dev_split8 = dev_part.iloc[0:dev80, :]
test_split8 = test_part.iloc[0:test80, :]
train_split9 = train_part.iloc[0:train90, :]
dev_split9 = dev_part.iloc[0:dev90, :]
test_split9 = test_part.iloc[0:test90, :]
train_split10 = train_part.iloc[0:train100, :]
dev_split10 = dev_part.iloc[0:dev100, :]
test_split10 = test_part.iloc[0:test100, :]

sen10 = train_part.iloc[0:train10, 0].astype('U')
print(sen10)
sen20 = train_part.iloc[0:train20, 0].astype('U')
print(sen20)
sen30 = train_part.iloc[0:train30, 0].astype('U')
print(sen30)
sen40 = train_part.iloc[0:train40, 0].astype('U')
print(sen40)
sen50 = train_part.iloc[0:train50, 0].astype('U')
print(sen50)
sen60 = train_part.iloc[0:train60, 0].astype('U')
print(sen60)
sen70 = train_part.iloc[0:train70, 0].astype('U')
print(sen70)
sen80 = train_part.iloc[0:train80, 0].astype('U')
print(sen80)
sen90 = train_part.iloc[0:train90, 0].astype('U')
print(sen90)
sen100 = train_part.iloc[0:train100, 0].astype('U')
print(sen100)

senrem10 = dev_df.iloc[0:dev10, 0].astype('U')
print(senrem10)
senrem20 = dev_df.iloc[0:dev20, 0].astype('U')
print(senrem20)
senrem30 = dev_df.iloc[0:dev30, 0].astype('U')
print(senrem30)
senrem40 = dev_df.iloc[0:dev40, 0].astype('U')
print(senrem40)
senrem50 = dev_df.iloc[0:dev50, 0].astype('U')
print(senrem50)
senrem60 = dev_df.iloc[0:dev60, 0].astype('U')
print(senrem60)
senrem70 = dev_df.iloc[0:dev70, 0].astype('U')
print(senrem70)
senrem80 = dev_df.iloc[0:dev80, 0].astype('U')
print(senrem80)
senrem90 = dev_df.iloc[0:dev90, 0].astype('U')
print(senrem90)
senrem100 = dev_df.iloc[0:dev100, 0].astype('U')
print(senrem100)

sentest10 = test_df.iloc[0:test10, 0].astype('U')
print(sentest10)
sentest20 = test_df.iloc[0:test20, 0].astype('U')
print(sentest20)
sentest30 = test_df.iloc[0:test30, 0].astype('U')
print(sentest30)
sentest40 = test_df.iloc[0:test40, 0].astype('U')
print(sentest40)
sentest50 = test_df.iloc[0:test50, 0].astype('U')
print(sentest50)
sentest60 = test_df.iloc[0:test60, 0].astype('U')
print(sentest60)
sentest70 = test_df.iloc[0:test70, 0].astype('U')
print(sentest70)
sentest80 = test_df.iloc[0:test80, 0].astype('U')
print(sentest80)
sentest90 = test_df.iloc[0:test90, 0].astype('U')
print(sentest90)
sentest100 = test_df.iloc[0:test100, 0].astype('U')
print(sentest100)


'''
train_dev_split1 = train_df.iloc[0:train10, :]
dev_split1 = dev_df.iloc[0:dev10, :]
test_split1 = test_df.iloc[0:test10, :]
train_split2 = train_df.iloc[0:train20, :]
dev_split2 = dev_df.iloc[0:dev20, :]
test_split2 = test_df.iloc[0:test20, :]
train_split3 = train_df.iloc[0:train30, :]
dev_split3 = dev_df.iloc[0:dev30, :]
test_split3 = test_df.iloc[0:test30, :]
train_split4 = train_df.iloc[0:train40, :]
dev_split4 = dev_df.iloc[0:dev40, :]
test_split4 = test_df.iloc[0:test40, :]
train_split5 = train_df.iloc[0:train50, :]
dev_split5 = dev_df.iloc[0:dev50, :]
test_split5 = test_df.iloc[0:test50, :]
train_split6 = train_df.iloc[0:train60, :]
dev_split6 = dev_df.iloc[0:dev60, :]
test_split6 = test_df.iloc[0:test60, :]
train_split7 = train_df.iloc[0:train70, :]
dev_split7 = dev_df.iloc[0:dev70, :]
test_split7 = test_df.iloc[0:test70, :]
train_split8 = train_df.iloc[0:train80, :]
dev_split8 = dev_df.iloc[0:dev80, :]
test_split8 = test_df.iloc[0:test80, :]
train_split9 = train_df.iloc[0:train90, :]
dev_split9 = dev_df.iloc[0:dev90, :]
test_split9 = test_df.iloc[0:test90, :]
train_split10 = train_df.iloc[0:train100, :]
dev_split10 = dev_df.iloc[0:dev100, :]
test_split10 = test_df.iloc[0:test100, :]
'''
'''
trdev_df1 = pd.concat([train_split1, dev_split1])
print(trdev_df1)
trdev_df2 = pd.concat([train_split2, dev_split2])
print(trdev_df2)
trdev_df3 = pd.concat([train_split3, dev_split3])
print(trdev_df3)
trdev_df4 = pd.concat([train_split4, dev_split4])
print(trdev_df4)
trdev_df5 = pd.concat([train_split5, dev_split5])
print(trdev_df5)
trdev_df6 = pd.concat([train_split6, dev_split6])
print(trdev_df6)
trdev_df7 = pd.concat([train_split7, dev_split7])
print(trdev_df7)
trdev_df8 = pd.concat([train_split8, dev_split8])
print(trdev_df8)
trdev_df9 = pd.concat([train_split9, dev_split9])
print(trdev_df9)
trdev_df10 = pd.concat([train_split10, dev_split10])
print(trdev_df10)
'''



'''
trdev10 = round(train_dev_df.shape[0] * .10)
print("10% train Subsample", train10)
trdev20 = round(train_dev_df.shape[0] * .20)
print("20% train Subsample", train20)
trdev30 = round(train_dev_df.shape[0] * .30)
print("30% train Subsample", train30)
trdev40 = round(train_dev_df.shape[0] * .40)
print("40% train Subsample", train40)
trdev50 = round(train_dev_df.shape[0] * .50)
print("50% train Subsample", train50)
trdev60 = round(train_dev_df.shape[0] * .60)
print("60% train Subsample", train60)
trdev70 = round(train_dev_df.shape[0] * .70)
print("70% train Subsample", train70)
trdev80 = round(train_dev_df.shape[0] * .80)
print("80% train Subsample", train80)
trdev90 = round(train_dev_df.shape[0] * .90)
print("90% train Subsample", train90)
trdev100 = round(train_dev_df.shape[0] * 1.00)
print("100% train Subsample", train100)

traintest_df1 = pd.concat([train_split1, test_split1])
print(traintest_df1)
traintest_df2 = pd.concat([train_split2, test_split2])
print(traintest_df2)
traintest_df3 = pd.concat([train_split3, test_split3])
print(traintest_df3)
traintest_df4 = pd.concat([train_split4, test_split4])
print(traintest_df4)
traintest_df5 = pd.concat([train_split5, test_split5])
print(traintest_df5)
traintest_df6 = pd.concat([train_split6, test_split6])
print(traintest_df6)
traintest_df7 = pd.concat([train_split7, test_split7])
print(traintest_df7)
traintest_df8 = pd.concat([train_split8, test_split8])
print(traintest_df8)
traintest_df9 = pd.concat([train_split9, test_split9])
print(traintest_df9)
traintest_df10 = pd.concat([train_split10, test_split10])
print(traintest_df10)

traintest10 = round(traintest_df1.shape[0] * .10)
print("10% train Subsample", train10)
traintest20 = round(traintest_df2.shape[0] * .20)
print("20% train Subsample", train20)
traintest30 = round(traintest_df3.shape[0] * .30)
print("30% train Subsample", train30)
traintest40 = round(traintest_df4.shape[0] * .40)
print("40% train Subsample", train40)
traintest50 = round(traintest_df5.shape[0] * .50)
print("50% train Subsample", train50)
traintest60 = round(traintest_df6.shape[0] * .60)
print("60% train Subsample", train60)
traintest70 = round(traintest_df7.shape[0] * .70)
print("70% train Subsample", train70)
traintest80 = round(traintest_df8.shape[0] * .80)
print("80% train Subsample", train80)
traintest90 = round(traintest_df9.shape[0] * .90)
print("90% train Subsample", train90)
traintest100 = round(traintest_df10.shape[0] * 1.00)
print("100% train Subsample", train100)
import os
#import system
'''

'''
pos_dev_split = dev_df[dev_df.iloc["label"] == 1]
pos_tra_split = train_df[train_df.iloc["label"] == 1]

train10 = round(train_df.shape[0] * .10) + 1
print("10% Subsample", train10)
train20 = round(train_df.shape[0] * .20) + 1
print("20% Subsample", train20)
train30 = round(train_df.shape[0] * .30) + 1
print("30% Subsample", train30)
train40 = round(train_df.shape[0] * .40) + 1
print("40% Subsample", train40)
train50 = round(train_df.shape[0] * .50) + 1
print("50% Subsample", train50)
train60 = round(train_df.shape[0] * .60) + 1
print("60% Subsample", train60)
train70 = round(train_df.shape[0] * .70) + 1
print("70% Subsample", train70)
train80 = round(train_df.shape[0] * .80) + 1
print("80% Subsample", train80)
train90 = round(train_df.shape[0] * .90) + 1
print("90% Subsample", train90)
train100 = round(train_df.shape[0] * 1.00) + 1
print("100% Subsample", train100)

half10 = round(train10 * .5) + 1
print("Half of 10%", half10)
half20 = round(train20 * .5) + 1
print("Half of 20%", half20)
half30 = round(train30 * .5) + 1
print("Half of 30%", half30)
half40 = round(train40 * .5) + 1
print("Half of 40%", half40)
half50 = round(train50 * .5) + 1
print("Half of 50%", half50)
half60 = round(train60 * .5) + 1
print("Half of 60%", half60)
half70 = round(train70 * .5) + 1
print("Half of 70%", half70)
half80 = round(train80 * .5) + 1
print("Half of 80%", half80)
half90 = round(train90 * .5) + 1
print("Half of 90%", half90)
half100 = round(train100 * .5) + 1
print("Half of 90%", half90)

trdf_sp1 = train_df.iloc[0:half10, :]
print(trdf_sp1)
tedf_sp1 = train_df.iloc[half10:half10+half10, :]
print(tedf_sp1)
trdf_sp2 = train_df.iloc[0:half20, :]
print(trdf_sp2)
tedf_sp2 = train_df.iloc[half20:half20+half20, :]
print(tedf_sp2)
trdf_sp3 = train_df.iloc[0:half30, :]
print(trdf_sp3)
tedf_sp3 = train_df.iloc[half30:half30+half30, :]
print(tedf_sp3)
trdf_sp4 = train_df.iloc[0:half40, :]
print(trdf_sp4)
tedf_sp4 = train_df.iloc[half40:half40+half40, :]
print(tedf_sp4)
trdf_sp5 = train_df.iloc[0:half50, :]
print(trdf_sp5)
tedf_sp5 = train_df.iloc[half50:half50+half50, :]
print(tedf_sp5)
trdf_sp6 = train_df.iloc[0:half60, :]
print(trdf_sp5)
tedf_sp6 = train_df.iloc[half60:half60+half60, :]
print(tedf_sp6)
trdf_sp7 = train_df.iloc[0:half70, :]
print(trdf_sp7)
tedf_sp7 = train_df.iloc[half70:half70+half70, :]
print(tedf_sp7)
trdf_sp8 = train_df.iloc[0:half80, :]
print(trdf_sp8)
tedf_sp8 = train_df.iloc[half80:half80+half80, :]
print(tedf_sp8)
trdf_sp9 = train_df.iloc[0:half90, :]
print(trdf_sp9)
tedf_sp9 = train_df.iloc[half90:half90+half90, :]
print(tedf_sp9)
trdf_sp10 = train_df.iloc[0:half100, :]
print(trdf_sp10)
tedf_sp10 = train_df.iloc[half100:half100+half100, :]
print(tedf_sp10)

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
'''
'''
sen10 = trdev_df1.iloc[0:train10, 0].astype('U')
print(sen10)
sen20 = trdev_df1.iloc[0:train20, 0].astype('U')
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

senrem10 = dev_df.iloc[0:dev10, 0].astype('U')
print(senrem10)
senrem20 = dev_df.iloc[0:dev20, 0].astype('U')
print(senrem20)
senrem30 = dev_df.iloc[0:dev30, 0].astype('U')
print(senrem30)
senrem40 = dev_df.iloc[0:dev40, 0].astype('U')
print(senrem40)
senrem50 = dev_df.iloc[0:dev50, 0].astype('U')
print(senrem50)
senrem60 = dev_df.iloc[0:dev60, 0].astype('U')
print(senrem60)
senrem70 = dev_df.iloc[0:dev70, 0].astype('U')
print(senrem70)
senrem80 = dev_df.iloc[0:dev80, 0].astype('U')
print(senrem80)
senrem90 = dev_df.iloc[0:dev90, 0].astype('U')
print(senrem90)
senrem100 = dev_df.iloc[0:dev100, 0].astype('U')
print(senrem100)

sentest10 = test_df.iloc[0:test10, 0].astype('U')
print(sentest10)
sentest20 = test_df.iloc[0:test20, 0].astype('U')
print(sentest20)
sentest30 = test_df.iloc[0:test30, 0].astype('U')
print(sentest30)
sentest40 = test_df.iloc[0:test40, 0].astype('U')
print(sentest40)
sentest50 = test_df.iloc[0:test50, 0].astype('U')
print(sentest50)
sentest60 = test_df.iloc[0:test60, 0].astype('U')
print(sentest60)
sentest70 = test_df.iloc[0:test70, 0].astype('U')
print(sentest70)
sentest80 = test_df.iloc[0:test80, 0].astype('U')
print(sentest80)
sentest90 = test_df.iloc[0:test90, 0].astype('U')
print(sentest90)
sentest100 = test_df.iloc[0:test100, 0].astype('U')
print(sentest100)
'''
'''
senrem1 = train_df.iloc[half10:half10+half10, 0].astype('U')
senrem2 = train_df.iloc[half20:half20+half20, 0].astype('U')
senrem3 = train_df.iloc[half30:half30+half30, 0].astype('U')
senrem4 = train_df.iloc[half40:half40+half40, 0].astype('U')
senrem5 = train_df.iloc[half50:half50+half50, 0].astype('U')
senrem6 = train_df.iloc[half60:half60+half60, 0].astype('U')
senrem7 = train_df.iloc[half70:half70+half70, 0].astype('U')
senrem8 = train_df.iloc[half80:half80+half80, 0].astype('U')
senrem9 = train_df.iloc[half90:half90+half90, 0].astype('U')
senrem10 = train_df.iloc[half100:half100+half100, 0].astype('U')
'''

tfidf_vect = TfidfVectorizer()
#count_vect = CountVectorizer()

#tfidf_train = tfidf_vect.fit_transform(sen10.values.astype('U'))
#tfidf_test = tfidf_vect.transform(rem1.values.astype('U'))
#print(tfidf_vect.shape)
#print(tfidf_test.shape)

x_train , y_train = tfidf_vect.fit_transform(sen10), tfidf_vect.transform(senrem10)
x_train1, y_train1 = tfidf_vect.fit_transform(sen20), tfidf_vect.transform(senrem20)
x_train2, y_train2 = tfidf_vect.fit_transform(sen30), tfidf_vect.transform(senrem30)
x_train3, y_train3 = tfidf_vect.fit_transform(sen40), tfidf_vect.transform(senrem40)
x_train4, y_train4 = tfidf_vect.fit_transform(sen50), tfidf_vect.transform(senrem50)
x_train5, y_train5 = tfidf_vect.fit_transform(sen60), tfidf_vect.transform(senrem60)
x_train6, y_train6 = tfidf_vect.fit_transform(sen70), tfidf_vect.transform(senrem70)
x_train7, y_train7 = tfidf_vect.fit_transform(sen80), tfidf_vect.transform(senrem80)
x_train8, y_train8 = tfidf_vect.fit_transform(sen90), tfidf_vect.transform(senrem90)
x_train9, y_train9 = tfidf_vect.fit_transform(sen100), tfidf_vect.transform(senrem100)
subsample_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
'''
def pause():
    programPause = input("Press the <ENTER> key to continue...")

print("Think about what you ate for dinner last night...")
'''
x_train = tfidf_vect.fit_transform(sen10)
x_train1 = tfidf_vect.fit_transform(sen20)
x_train2 = tfidf_vect.fit_transform(sen30)
x_train3 = tfidf_vect.fit_transform(sen40)
x_train4 = tfidf_vect.fit_transform(sen50)
x_train5 = tfidf_vect.fit_transform(sen60)
x_train6 = tfidf_vect.fit_transform(sen70)
x_train7 = tfidf_vect.fit_transform(sen80)
x_train8 = tfidf_vect.fit_transform(sen90)
x_train9 = tfidf_vect.fit_transform(sen100)

x_test = tfidf_vect.transform(senrem10)
x_test1 = tfidf_vect.transform(senrem20)
x_test2 = tfidf_vect.transform(senrem30)
x_test3 = tfidf_vect.transform(senrem40)
x_test4 = tfidf_vect.transform(senrem50)
x_test5 = tfidf_vect.transform(senrem60)
x_test6 = tfidf_vect.transform(senrem70)
x_test7 = tfidf_vect.transform(senrem80)
x_test8 = tfidf_vect.transform(senrem90)
x_test9 = tfidf_vect.transform(senrem100)

x_test = train_split1["label"]
x_test1 = train_split2["label"]
x_test2 = train_split3["label"]
x_test3 = train_split4["label"]
x_test4 = train_split5["label"]
x_test5 = train_split6["label"]
x_test6 = train_split7["label"]
x_test7 = train_split8["label"]
x_test8 = train_split9["label"]
x_test9 = train_split10["label"]

y_test = dev_split1["label"]
y_test1 = dev_split2["label"]
y_test2 = dev_split3["label"]
y_test3 = dev_split4["label"]
y_test4 = dev_split5["label"]
y_test5 = dev_split6["label"]
y_test6 = dev_split7["label"]
y_test7 = dev_split8["label"]
y_test8 = dev_split9["label"]
y_test9 = dev_split10["label"]


sen_train = [x_train, x_train1, x_train2, x_train3, x_train4, x_train5, x_train6, x_train7, x_train8, x_train9]
#sen_train = [x_train, x_train1, x_train2]
print(sen_train)
sen_test = [y_train, y_train1, y_train2, y_train3, y_train4, y_train5, y_train6, y_train7, y_train8, y_train9]
#sen_test = [y_train, y_train1, y_train2]
print(sen_test)
train_label = [x_test, x_test1, x_test2, x_test3, x_test4, x_test5, x_test6, x_test7, x_test8, x_test9]
#train_label = [x_test, x_test1, x_test2]
test_label = [y_test, y_test1, y_test2, y_test3, y_test4, y_test5, y_test6, y_test7, y_test8, y_test9]
#test_label = [y_test, y_test1, y_test2]
subsample_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#subsample_num = [1, 2, 3]

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

for h, j, k, l, m in zip(sen_train, sen_test, train_label, test_label, subsample_num):

    start1 = time.time()
    log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
    model_lr = log.fit(h, k)
    print(model_lr)
    probs_lr = model_lr.predict_proba(j)[:, 1]
    print(probs_lr)
    probs_lr_scol.append(probs_lr)
    ly_prediction = log.predict(j)
    fly = f1_score(ly_prediction, l)
    f1_lr_scol.append(fly)
    rocauc_lr = roc_auc_score(l, ly_prediction)
    rocauc_lr_scol.append(rocauc_lr)
    recalls_lr = recall_score(l, ly_prediction)
    recall_lr_scol.append(recalls_lr)
    precisions_lr = precision_score(l, ly_prediction)
    precision_lr_scol.append(precisions_lr)
    accuracys_lr = accuracy_score(l, ly_prediction)
    accuracy_lr_scol.append(accuracys_lr)
    print("===Logistic Regression with TfidfVectorizer Imbalanced - Train_Set 2010-2016", m)
    lr_end = time.time()
    print('Logistic F1-score', fly * 100)
    print('Logistic ROCAUC score:', rocauc_lr * 100)
    print('Logistic Recall score:', recalls_lr * 100)
    print('Logistic Precision Score:', precisions_lr * 100)
    print('Logistic Confusion Matrix', confusion_matrix(l, ly_prediction), "\n")
    print('Logistic Classification', classification_report(l, ly_prediction), "\n")
    print('Logistic Accuracy Score', accuracys_lr * 100)
    print("Execution Time for Logistic Regression Imbalanced - Train_Set 2010-2016: ", lr_end - start1, "seconds")

    start2 = time.time()
    from sklearn.tree import DecisionTreeClassifier

    DCT = DecisionTreeClassifier()
    model_dt = DCT.fit(h, k)
    probs_dt = model_dt.predict_proba(j)[:, 1]
    probs_dt_scol.append(probs_dt)
    dct_pred = DCT.predict(j)
    fdct = f1_score(dct_pred, l)
    f1_dt_scol.append(fdct)
    rocauc_dt = roc_auc_score(l, dct_pred)
    rocauc_dt_scol.append(rocauc_dt)
    recalls_dt = recall_score(l, dct_pred)
    recall_dt_scol.append(recalls_dt)
    precisions_dt = precision_score(l, dct_pred)
    precision_dt_scol.append(precisions_dt)
    accuracys_dt = accuracy_score(l, dct_pred)
    accuracy_dt_scol.append(accuracys_dt)
    print("===DecisionTreeClassifier with TfidfVectorizer Imbalanced Train_Set - 2010-2016", m)
    dt_end = time.time()
    print('DCT F1-score', fdct * 100)
    print('DCT ROCAUC score:', rocauc_dt * 100)
    print('DCT Recall score:', recalls_dt * 100)
    print('DCT Precision Score:', precisions_dt * 100)
    print('DCT Confusion Matrix', confusion_matrix(l, dct_pred), "\n")
    print('DCT Classification', classification_report(l, dct_pred), "\n")
    print('DCT Accuracy Score', accuracys_dt * 100)
    print("Execution Time for Decision Tree Imbalanced Train_Set - 2010-2016: ", dt_end - start2, "seconds")

    from sklearn.naive_bayes import MultinomialNB

    start3 = time.time()
    Naive = MultinomialNB()
    model_nb = Naive.fit(h, k)
    probs_nb = model_nb.predict_proba(j)[:, 1]
    probs_nb_scol.append(probs_nb)
    # predict the labels on validation dataset
    ny_pred = Naive.predict(j)
    fnb = f1_score(ny_pred, l)
    f1_nb_scol.append(fnb)
    rocauc_nb = roc_auc_score(l, ny_pred)
    rocauc_nb_scol.append(rocauc_nb)
    recalls_nb = recall_score(l, ny_pred)
    recall_nb_scol.append(recalls_nb)
    precisions_nb = precision_score(l, ny_pred)
    precision_nb_scol.append(precisions_nb)
    accuracys_nb = accuracy_score(l, ny_pred)
    accuracy_nb_scol.append(accuracys_nb)
    nb_end = time.time()
    # Use accuracy_score function to get the accuracy
    print("===Naive Bayes with TfidfVectorizer Imbalanced Train_Set - 2010-2016", m)
    print('Naive F1-score', fnb * 100)
    print('Naive ROCAUC score:', rocauc_nb * 100)
    print('Naive Recall score:', recalls_nb * 100)
    print('Naive Precision Score:', precisions_nb * 100)
    print('Naive Confusion Matrix', confusion_matrix(l, ny_pred), "\n")
    print('Naive Classification', classification_report(l, ny_pred), "\n")
    print('Naive Accuracy Score', accuracys_nb * 100)
    print("Execution Time for Naive Bayes Imbalanced Train_Set - 2010-2016: ", nb_end - start3, "seconds")

    # XGBoost Classifier

    start4 = time.time()
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    xgb_model = XGBClassifier().fit(h, k)
    probs_xg = xgb_model.predict_proba(j)[:, 1]
    probs_xg_scol.append(probs_xg)
    # predict
    xgb_y_predict = xgb_model.predict(j)
    fxg = f1_score(xgb_y_predict, l)
    f1_xg_scol.append(fxg)
    rocauc_xg = roc_auc_score(xgb_y_predict, l)
    rocauc_xg_scol.append(rocauc_xg)
    recall_xg = recall_score(xgb_y_predict, l)
    recall_xg_scol.append(recall_xg)
    precisions_xg = precision_score(xgb_y_predict, l)
    precision_xg_scol.append(precisions_xg)
    accuracys_xg = accuracy_score(xgb_y_predict, l)
    accuracy_xg_scol.append(accuracys_xg)
    xg_end = time.time()
    print("===XGB with TfidfVectorizer Imbalanced Train_Set - 2010-2016", m)
    print('XGB F1-Score', fxg * 100)
    print('XGB ROCAUC Score:', rocauc_xg * 100)
    print('XGB Recall score:', recall_xg * 100)
    print('XGB Precision Score:', precisions_xg * 100)
    print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, l), "\n")
    print('XGB Classification', classification_report(xgb_y_predict, l), "\n")
    print('XGB Accuracy Score', accuracys_nb * 100)
    print("Execution Time for XGBoost Classifier Imbalanced Train_Set - 2010-2016:", xg_end - start4, "seconds")

    # Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier

    start5 = time.time()
    rfc_model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(h, k)
    probs_rf = rfc_model.predict_proba(j)[:, 0]
    probs_rf_scol.append(probs_rf)
    rfc_pred = rfc_model.predict(j)
    frfc = f1_score(rfc_pred, l)
    f1_rf_scol.append(frfc)
    rocauc_rf = roc_auc_score(l, rfc_pred)
    rocauc_rf_scol.append(rocauc_rf)
    recalls_rf = recall_score(rfc_pred, l)
    recall_rf_scol.append(recalls_rf)
    precisions_rf = precision_score(rfc_pred, l)
    precision_rf_scol.append(precisions_rf)
    accuracys_rf = accuracy_score(rfc_pred, l)
    accuracy_rf_scol.append(accuracys_rf)
    rf_end = time.time()
    print("====RandomForest with TfidfVectorizer Imbalanced Train_Set - 2010-2016", m)
    print('RFC F1 score', frfc * 100)
    print('RFC ROCAUC Score:', rocauc_rf * 100)
    print('RFC Recall score:', recalls_rf * 100)
    print('RFC Precision Score:', precisions_rf * 100)
    print('RFC Confusion Matrix', confusion_matrix(l, rfc_pred), "\n")
    print('RFC Classification', classification_report(l, rfc_pred), "\n")
    print('RFC Accuracy Score', accuracys_rf * 100)
    print("Execution Time for Random Forest Classifier Imbalanced Train_Set - 2010-2016", rf_end - start5, "seconds")

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

subset = ['Train_Set' for t in range(50)]
other_set = ['Dev_Set' for t in range(50)]
sampling = ['Imbalanced' for t in range(50)]
technique = ['Imbalanced' for t in range(50)]
classifier_names = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
#train_sizes_num = [0.95]
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
rounds = 5
## rounds1 = 5
p = itertools.cycle(classifier_names)
o = itertools.cycle(subsample_num)
#k = itertools.cycle(train_sizes_num)
# v = itertools.cycle(score_location)
# pr = itertools.cycle(precision_num)
# y = itertools.cycle(iteration_csv)
classifier_csv = [next(p) for _ in range(rounds)] * 10
subsample_csv = [r for r in subsample_num for _ in range(rounds)]
#test_size_csv = [next(o) for _ in range(rounds)] * 1
#train_size_csv = [next(k) for _ in range(rounds)] * 1
# split_csv = ['1' for t in range(25)]
# train_csv = ['0.5' for t in range(25)]
precision_csv = list(chain(*precision_csv_num))
recall_csv = list(chain(*recall_csv_num))
auc_csv = list(chain(*auc_csv_num))
accuracy_csv = list(chain(*accuracy_csv_num))
csv_data = [subset, other_set, sampling, technique, classifier_csv, subsample_csv, precision_csv, recall_csv, auc_csv, accuracy_csv]
export_data = zip_longest(*csv_data, fillvalue='')
filename = "Subset_Iterations.csv"
with open(filename, 'w', newline='') as file:
    write = csv.writer(file)
    write.writerow(("Subset", "Other Set", "Sampling", "Technique", "Classifier", "Iteration",
                    "Precision", "Recall", "AUC", "Accuracy"))
    write.writerows(export_data)

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

'''
'''
'''

tfidf_vect1 = TfidfVectorizer(ngram_range=(1,2))
tfidf_train1 = tfidf_vect.fit_transform(sen10.values)
tfidf_test1= tfidf_vect.transform(senrem10.values)
print(tfidf_train1.shape)
print(tfidf_test1.shape)


x_tfidf = tfidf_vect.fit_transform(train_df["sentence"].astype('U'))

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
'''
'''
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
'''
x = x_tfidf
y = train_df['label']
print(x.shape)
print(y.shape)
'''
rus = RandomUnderSampler(random_state=42, replacement=True)  # fit predictor and target variable
#x_rus, y_rus = rus.fit_resample(x, y)
'''
print('Original dataset shape:', Counter(y))
print('Resample dataset shape', Counter(y_rus))
'''
# import library
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

'''
x = x_tfidf
y = train_df["label"]
'''

ros = RandomUnderSampler(random_state=42)
# ros = RandomOverSampler(random_state=42)
# Random over-sampling with imblearn
# fit predictor and target variable
#x_rus, y_rus = rus.fit_resample(x, y)
'''
print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_rus))
'''
# Random over-sampling with imblearn
ros = RandomOverSampler(random_state=42)

# fit predictor and target variable
#x_ros, y_ros = ros.fit_resample(x, y)
'''
print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_ros))
'''
#import library

# Under-sampling: Tomek links
from imblearn.under_sampling import TomekLinks
from collections import Counter

#tl = RandomOverSampler(sampling_strategy='majority')
tl = TomekLinks()

# fit predictor and target variable
#x_tl, y_tl = ros.fit_resample(x, y)
'''
print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_ros))
'''
# import library
from imblearn.over_sampling import SMOTE

smote = SMOTE()

# fit predictor and target variable
#x_smote, y_smote = smote.fit_resample(x, y)
'''
print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_ros))
'''
from imblearn.under_sampling import NearMiss

nm = NearMiss()

#x_nm, y_nm = nm.fit_resample(x, y)
'''
print('Original dataset shape:', Counter(y))
print('Resample dataset shape:', Counter(y_nm))
'''
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

x_tfidf1 = tfidf_vect.fit_transform(train_dev_split1["sentence"].astype('U'))
x_tfidf2 = tfidf_vect.fit_transform(train_dev_split2["sentence"].astype('U'))
x_tfidf3 = tfidf_vect.fit_transform(train_dev_split3["sentence"].astype('U'))
x_tfidf4 = tfidf_vect.fit_transform(train_dev_split4["sentence"].astype('U'))
x_tfidf5 = tfidf_vect.fit_transform(train_dev_split5["sentence"].astype('U'))
x_tfidf6 = tfidf_vect.fit_transform(train_dev_split6["sentence"].astype('U'))
x_tfidf7 = tfidf_vect.fit_transform(train_dev_split7["sentence"].astype('U'))
x_tfidf8 = tfidf_vect.fit_transform(train_dev_split8["sentence"].astype('U'))
x_tfidf9 = tfidf_vect.fit_transform(train_dev_split9["sentence"].astype('U'))
x_tfidf10 = tfidf_vect.fit_transform(train_dev_split10["sentence"].astype('U'))

subsample1 = x_tfidf1
print(subsample1.shape)
sublabel1 = train_dev_split1["label"]
print(sublabel1.shape)
print(train_df['label'].value_counts())
label_count = train_df.groupby("label")
print("Label Count", label_count)
print(train_df.label.dtype)
print(train_df.info())
print("XTIDF TL:", type(subsample1))
print("LABEL TL:", type(sublabel1))
x_tl1, y_tl1 = tl.fit_resample(subsample1, sublabel1)
subsample2 = x_tfidf2
print(subsample2.shape)
sublabel2 = train_dev_split2["label"]
print(sublabel2.shape)
x_tl2, y_tl2 = tl.fit_resample(subsample2, sublabel2)
subsample3 = x_tfidf3
print(subsample3.shape)
sublabel3 = train_dev_split3["label"]
print(sublabel3.shape)
x_tl3, y_tl3 = tl.fit_resample(subsample3, sublabel3)
subsample4 = x_tfidf4
print(subsample4.shape)
sublabel4 = train_dev_split4["label"]
print(sublabel4.shape)
x_tl4, y_tl4 = tl.fit_resample(subsample4, sublabel4)
subsample5 = x_tfidf5
print(subsample5.shape)
sublabel5 = train_dev_split5["label"]
print(sublabel5.shape)
x_tl5, y_tl5 = tl.fit_resample(subsample5, sublabel5)
subsample6 = x_tfidf6
sublabel6 = train_dev_split6["label"]
x_tl6, y_tl6 = tl.fit_resample(subsample6, sublabel6)
subsample7 = x_tfidf7
sublabel7 = train_dev_split7["label"]
x_tl7, y_tl7 = tl.fit_resample(subsample7, sublabel7)
subsample8 = x_tfidf8
sublabel8 = train_dev_split8["label"]
x_tl8, y_tl8 = tl.fit_resample(subsample8, sublabel8)
subsample9 = x_tfidf9
sublabel9 = train_dev_split9["label"]
x_tl9, y_tl9 = tl.fit_resample(subsample9, sublabel9)
subsample10 = x_tfidf10
sublabel10 = train_dev_split10["label"]
x_tl10, y_tl10 = tl.fit_resample(subsample10, sublabel10)

'''
x_tl1a, y_tl1a = tl.fit_resample(x_train, x_test)
x_tl1b, y_tl1b = tl.fit_resample(y_train, y_test)
x_tl2a, y_tl2a = tl.fit_resample(x_train1, x_test1)
x_tl2b, y_tl2b = tl.fit_resample(y_train1, y_test1)
x_tl3a, y_tl3a = tl.fit_resample(x_train2, x_test2)
x_tl3b, y_tl3b = tl.fit_resample(y_train2, y_test2)
x_tl4a, y_tl4a = tl.fit_resample(x_train3, x_test3)
x_tl4b, y_tl4b = tl.fit_resample(y_train3, y_test3)
x_tl5a, y_tl5a = tl.fit_resample(x_train4, x_test4)
x_tl5b, y_tl5b = tl.fit_resample(y_train4, y_test4)
x_tl6a, y_tl6a = tl.fit_resample(x_train5, x_test5)
x_tl6b, y_tl6b = tl.fit_resample(y_train5, y_test5)
x_tl7a, y_tl7a = tl.fit_resample(x_train6, x_test6)
x_tl7b, y_tl7b = tl.fit_resample(y_train6, y_test6)
x_tl8a, y_tl8a = tl.fit_resample(x_train7, x_test7)
x_tl8b, y_tl8b = tl.fit_resample(y_train7, y_test7)
x_tl9a, y_tl9a = tl.fit_resample(x_train8, x_test8)
x_tl9b, y_tl9b = tl.fit_resample(y_train8, y_test8)
x_tl10a, y_tl10a = tl.fit_resample(x_train9, x_test9)
x_tl10b, y_tl10b = tl.fit_resample(y_train9, y_test9)
'''
'''

tl_xtrain_list = [x_tl1a, x_tl2a, x_tl3a, x_tl4a, x_tl5a, x_tl6a, x_tl7a, x_tl8a, x_tl9a, x_tl10a]
tl_xtest_list = [y_tl1a, y_tl2a, y_tl3a, y_tl4a, y_tl5a, y_tl6a, y_tl7a, y_tl8a, y_tl9a, y_tl10a]
tl_ytrain_list = [x_tl1b, x_tl2b, x_tl3b, x_tl4b, x_tl5b, x_tl6b, x_tl7b, x_tl8b, x_tl9b, x_tl10b]
tl_ytest_list = [y_tl1b, y_tl2b, y_tl3b, y_tl4b, y_tl5b, y_tl6b, y_tl7b, y_tl8b, y_tl9b, y_tl10b]
subsample_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

'''



'''
trdev_df1 = pd.concat([sen10, senrem10])
trdev_df2 = pd.concat([sen20, senrem20])
trdev_df3 = pd.concat([sen30, senrem30])
trdev_df4 = pd.concat([sen40, senrem40])
trdev_df5 = pd.concat([sen50, senrem50])
trdev_df6 = pd.concat([sen60, senrem60])
trdev_df7 = pd.concat([sen70, senrem70])
trdev_df8 = pd.concat([sen80, senrem80])
trdev_df9 = pd.concat([sen90, senrem90])
trdev_df10 = pd.concat([sen100, senrem100])
'''
'''
subsample1 = tfidf_vect.fit_transform(train_dev_df.iloc[1:train10, 0].astype('U'))
print(subsample1)
sublabel1 = train_dev_df.iloc[1:train10, 1].astype('U')
print(sublabel1)
x_tl1, y_tl1 = tl.fit_resample(subsample1, sublabel1)
subsample2 = tfidf_vect.fit_transform(train_dev_df.iloc[1:train20, 0].astype('U'))
print(subsample2)
sublabel2 = train_dev_df.iloc[1:train20, 1].astype('U')
print(sublabel2)
x_tl2, y_tl2 = tl.fit_resample(subsample2, sublabel2)
subsample3 = tfidf_vect.fit_transform(train_dev_df.iloc[1:train30, 0].astype('U'))
print(subsample3)
sublabel3 = train_dev_df.iloc[1:train30, 1].astype('U')
print(sublabel3)
x_tl3, y_tl3 = tl.fit_resample(subsample3, sublabel3)
subsample4 = tfidf_vect.fit_transform(train_dev_df.iloc[1:train40, 0].astype('U'))
print(subsample4)
sublabel4 = train_dev_df.iloc[1:train40, 1].astype('U')
print(sublabel4)
x_tl4, y_tl4 = tl.fit_resample(subsample4, sublabel4)
subsample5 = tfidf_vect.fit_transform(train_dev_df.iloc[1:train50, 0].astype('U'))
print(subsample5)
sublabel5 = train_dev_df.iloc[1:train50, 1].astype('U')
print(sublabel5)
x_tl5, y_tl5 = tl.fit_resample(subsample5, sublabel5)
subsample6 = tfidf_vect.fit_transform(train_dev_df.iloc[1:train60, 0].astype('U'))
sublabel6 = train_dev_df.iloc[1:train60, 1].astype('U')
x_tl6, y_tl6 = tl.fit_resample(subsample6, sublabel6)
subsample7 = tfidf_vect.fit_transform(train_dev_df.iloc[1:train70, 0].astype('U'))
sublabel7 = train_dev_df.iloc[1:train70, 1].astype('U')
x_tl7, y_tl7 = tl.fit_resample(subsample7, sublabel7)
subsample8 = tfidf_vect.fit_transform(train_dev_df.iloc[1:train80, 0].astype('U'))
sublabel8 = train_dev_df.iloc[1:train80, 1].astype('U')
x_tl8, y_tl8 = tl.fit_resample(subsample8, sublabel8)
subsample9 = tfidf_vect.fit_transform(train_dev_df.iloc[1:train90, 0].astype('U'))
sublabel9 = train_dev_df.iloc[1:train90, 1].astype('U')
x_tl9, y_tl9 = tl.fit_resample(subsample9, sublabel9)
subsample10 = tfidf_vect.fit_transform(train_dev_df.iloc[1:train100, 0].astype('U'))
sublabel10 = train_dev_df.iloc[1:train100, 1].astype('U')
x_tl10, y_tl10 = tl.fit_resample(subsample10, sublabel10)
'''
'''
subsample1 = x_tfidf1
print(subsample1)
sublabel1 = train_df["label"]
print(sublabel1)
x_tl1, y_tl1 = tl.fit_resample(subsample1, sublabel1)
print('1st Split Original dataset shape:', Counter(sublabel1))
print('1st Split Resample dataset shape', Counter(y_tl1))
'''
'''
subsample2 = x_tfidf2
print(subsample2)
sublabel2 = train_dev_split2["label"]
print(sublabel2)
x_tl2, y_tl2 = tl.fit_resample(subsample2, sublabel2)
print('2nd Split Original dataset shape:', Counter(sublabel2))
print('2nd Split Resample dataset shape', Counter(y_tl2))
subsample3 = x_tfidf3
print(subsample3)
sublabel3 = train_dev_split3["label"]
print(sublabel3)
x_tl3, y_tl3 = tl.fit_resample(subsample3, sublabel3)
print('3rd Split Original dataset shape:', Counter(sublabel3))
print('3rd Split Resample dataset shape', Counter(y_tl3))
subsample4 = x_tfidf4
print(subsample4)
sublabel4 = train_dev_split4["label"]
print(sublabel4)
x_tl4, y_tl4 = tl.fit_resample(subsample4, sublabel4)
print('4th Split Original dataset shape:', Counter(sublabel4))
print('4th Split Resample dataset shape', Counter(y_tl4))
subsample5 = x_tfidf5
print(subsample5)
sublabel5 = train_dev_split5["label"]
print(sublabel5)
x_tl5, y_tl5 = tl.fit_resample(subsample5, sublabel5)
print('5th Split Original dataset shape:', Counter(sublabel5))
print('5th Split Resample dataset shape', Counter(y_tl5))
subsample6 = x_tfidf6
sublabel6 = train_dev_split6["label"]
x_tl6, y_tl6 = tl.fit_resample(subsample6, sublabel6)
print('6th Split Original dataset shape:', Counter(sublabel6))
print('6th Split Resample dataset shape', Counter(y_tl6))
subsample7 = x_tfidf7
sublabel7 = train_dev_split7["label"]
x_tl7, y_tl7 = tl.fit_resample(subsample7, sublabel7)
print('7th Split Original dataset shape:', Counter(sublabel7))
print('7th Split Resample dataset shape', Counter(y_tl7))
subsample8 = x_tfidf8
sublabel8 = train_dev_split8["label"]
x_tl8, y_tl8 = tl.fit_resample(subsample8, sublabel8)
print('8th Split Original dataset shape:', Counter(sublabel8))
print('8th Split Resample dataset shape', Counter(y_tl8))
subsample9 = x_tfidf9
sublabel9 = train_dev_split9["label"]
x_tl9, y_tl9 = tl.fit_resample(subsample9, sublabel9)
print('9th Split Original dataset shape:', Counter(sublabel9))
print('9th Split Resample dataset shape', Counter(y_tl9))
subsample10 = x_tfidf10
sublabel10 = train_dev_split10["label"]
x_tl10, y_tl10 = tl.fit_resample(subsample10, sublabel10)
print('10th Split Original dataset shape:', Counter(sublabel10))
print('10th Split Resample dataset shape', Counter(y_tl10))
import os


'''

sample_list1 = np.array([x_tl1, x_tl2, x_tl3, x_tl4, x_tl5, x_tl6, x_tl7, x_tl8, x_tl9, x_tl10])
label_list1 = np.array([y_tl1, y_tl2, y_tl3, y_tl4, y_tl5, y_tl6, y_tl7, y_tl8, y_tl9, y_tl10])
'''
org_samples = [subsample1, subsample2, subsample3, subsample4, subsample5, subsample6, subsample7, subsample8, subsample9, subsample10]
org_labels = [sublabel1, sublabel2, sublabel3, sublabel4, sublabel5, sublabel6, sublabel7, sublabel8, sublabel9, subsample10]
#xtrain_list = [x_train10, x_train11, x_train12, x_train13, x_train14, x_train15, x_train16, x_train17, x_train18, x_train19]
'''
'''
x_train10, x_test10, y_train10, y_test10 = train_test_split(sample_list1, label_list1, train_size=0.5, test_size=0.5)
x_train10, x_test10, y_train10, y_test10 = train_test_split(sample_list1, label_list1, train_size=0.5, test_size=0.5)
x_train10, x_test10, y_train10, y_test10 = train_test_split(sample_list1, label_list1, train_size=0.5, test_size=0.5)
x_train10, x_test10, y_train10, y_test10 = train_test_split(sample_list1, label_list1, train_size=0.5, test_size=0.5)
x_train10, x_test10, y_train10, y_test10 = train_test_split(sample_list1, label_list1, train_size=0.5, test_size=0.5)
x_train10, x_test10, y_train10, y_test10 = train_test_split(sample_list1, label_list1, train_size=0.5, test_size=0.5)
x_train10, x_test10, y_train10, y_test10 = train_test_split(sample_list1, label_list1, train_size=0.5, test_size=0.5)
x_train10, x_test10, y_train10, y_test10 = train_test_split(sample_list1, label_list1, train_size=0.5, test_size=0.5)
x_train10, x_test10, y_train10, y_test10 = train_test_split(sample_list1, label_list1, train_size=0.5, test_size=0.5)
x_train10, x_test10, y_train10, y_test10 = train_test_split(sample_list1, label_list1, train_size=0.5, test_size=0.5)
'''

for i, j, z in zip(sample_list1, label_list1, subsample_num):
#for h, j, k, l in zip(tl_xtrain_list, tl_xtest_list, tl_ytrain_list, tl_ytest_list):
    x_train, x_test, y_train, y_test = train_test_split(i, j, train_size=0.5, test_size=0.5)

    start1 = time.time()
    log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
    model_lr = log.fit(x_train, y_train)
    probs_lr = model_lr.predict_proba(x_test)[:, 1]
    probs_lr_scol1.append(probs_lr)
    ly_prediction = log.predict(x_test)
    fly = f1_score(ly_prediction, y_test, pos_label=1)
    f1_lr_scol1.append(fly)
    rocauc_lr = roc_auc_score(y_test, ly_prediction)
    rocauc_lr_scol1.append(rocauc_lr)
    recalls_lr = recall_score(y_test, ly_prediction, pos_label=1)
    recall_lr_scol1.append(recalls_lr)
    precisions_lr = precision_score(y_test, ly_prediction, pos_label=1)
    precision_lr_scol1.append(precisions_lr)
    accuracys_lr = accuracy_score(y_test, ly_prediction)
    accuracy_lr_scol1.append(accuracys_lr)
    print("===Logistic Regression with TfidfVectorizer Tomelinks Train_Set & Dev_Set - 2010-2016", k)
    lr_end = time.time()
    print('Logistic F1-score', fly * 100)
    print('Logistic ROCAUC score:', rocauc_lr * 100)
    print('Logistic Recall score:', recalls_lr * 100)
    print('Logistic Precision Score:', precisions_lr * 100)
    print('Logistic Confusion Matrix', confusion_matrix(y_test, ly_prediction), "\n")
    print('Logistic Classification', classification_report(y_test, ly_prediction), "\n")
    print('Logistic Accuracy Score', accuracys_lr * 100)
    print("Execution Time for Logistic Regression Tomelinks Train_Set & Dev_Set - 2010-2016", lr_end - start1, "seconds")

    start2 = time.time()
    from sklearn.tree import DecisionTreeClassifier

    DCT = DecisionTreeClassifier()
    model_dt = DCT.fit(x_train, y_train)
    probs_dt = model_dt.predict_proba(x_test)[:, 1]
    probs_dt_scol1.append(probs_dt)
    dct_pred = DCT.predict(x_test)
    fdct = f1_score(dct_pred, y_test)
    f1_dt_scol1.append(fdct)
    rocauc_dt = roc_auc_score(y_test, dct_pred)
    rocauc_dt_scol1.append(rocauc_dt)
    recalls_dt = recall_score(y_test, dct_pred)
    recall_dt_scol1.append(recalls_dt)
    precisions_dt = precision_score(y_test, dct_pred)
    precision_dt_scol1.append(precisions_dt)
    accuracys_dt = accuracy_score(y_test, dct_pred)
    accuracy_dt_scol1.append(accuracys_dt)
    print("===DecisionTreeClassifier with TfidfVectorizer Tomelinks Train_Set & Dev_Set - 2010-2016", k)
    dt_end = time.time()
    print('DCT F1-score', fdct * 100)
    print('DCT ROCAUC score:', rocauc_dt * 100)
    print('DCT Recall score:', recalls_dt * 100)
    print('DCT Precision Score:', precisions_dt * 100)
    print('DCT Confusion Matrix', confusion_matrix(y_test, dct_pred), "\n")
    print('DCT Classification', classification_report(y_test, dct_pred), "\n")
    print('DCT Accuracy Score', accuracys_dt * 100)
    print("Execution Time for Decision Tree Tomelinks Train_Set & Dev_Set - 2010-2016", dt_end - start2, "seconds")

    from sklearn.naive_bayes import MultinomialNB

    start3 = time.time()
    Naive = MultinomialNB()
    model_nb = Naive.fit(x_train, y_train)
    probs_nb = model_nb.predict_proba(x_test)[:, 1]
    probs_nb_scol1.append(probs_nb)
    # predict the labels on validation dataset
    ny_pred = Naive.predict(x_test)
    fnb = f1_score(ny_pred, y_test)
    f1_nb_scol1.append(fnb)
    rocauc_nb = roc_auc_score(y_test, ny_pred)
    rocauc_nb_scol1.append(rocauc_nb)
    recalls_nb = recall_score(y_test, ny_pred)
    recall_nb_scol1.append(recalls_nb)
    precisions_nb = precision_score(y_test, ny_pred)
    precision_nb_scol1.append(precisions_nb)
    accuracys_nb = accuracy_score(y_test, ny_pred)
    accuracy_nb_scol1.append(accuracys_nb)
    nb_end = time.time()
    # Use accuracy_score function to get the accuracy
    print("===Naive Bayes with TfidfVectorizer Imabalanced Tomelinks Train_Set & Dev_Set - 2010-2016", k)
    print('Naive F1-score', fnb * 100)
    print('Naive ROCAUC score:', rocauc_nb * 100)
    print('Naive Recall score:', recalls_nb * 100)
    print('Naive Precision Score:', precisions_nb * 100)
    print('Naive Confusion Matrix', confusion_matrix(y_test, ny_pred), "\n")
    print('Naive Classification', classification_report(y_test, ny_pred), "\n")
    print('Naive Accuracy Score', accuracys_nb * 100)
    print("Execution Time for Naive Bayes Tomelinks Train_Set & Dev_Set - 2010-2016", nb_end - start3, "seconds")

    # XGBoost Classifier

    start4 = time.time()
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    xgb_model = XGBClassifier().fit(x_train, y_train)
    probs_xg = xgb_model.predict_proba(x_test)[:, 1]
    probs_xg_scol1.append(probs_xg)
    # predict
    xgb_y_predict = xgb_model.predict(x_test)
    fxg = f1_score(xgb_y_predict, y_test)
    f1_xg_scol1.append(fxg)
    rocauc_xg = roc_auc_score(xgb_y_predict, y_test)
    rocauc_xg_scol1.append(rocauc_xg)
    recall_xg = recall_score(xgb_y_predict, y_test)
    recall_xg_scol1.append(recall_xg)
    precisions_xg = precision_score(xgb_y_predict, y_test)
    precision_xg_scol1.append(precisions_xg)
    accuracys_xg = accuracy_score(xgb_y_predict, y_test)
    accuracy_xg_scol1.append(accuracys_xg)
    xg_end = time.time()
    print("===XGB with TfidfVectorizer Tomelinks Train_Set & Dev_Set - 2010-2016", k)
    print('XGB F1-Score', fxg * 100)
    print('XGB ROCAUC Score:', rocauc_xg * 100)
    print('XGB Recall score:', recall_xg * 100)
    print('XGB Precision Score:', precisions_xg * 100)
    print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, y_test), "\n")
    print('XGB Classification', classification_report(xgb_y_predict, y_test), "\n")
    print('XGB Accuracy Score', accuracys_nb * 100)
    print("Execution Time for XGBoost Classifier Tomelinks Train_Set & Dev_Set - 2010-2016", xg_end - start4, "seconds")

    # Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier

    start5 = time.time()
    rfc_model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(x_train, y_train)
    probs_rf = rfc_model.predict_proba(x_test)[:, 1]
    probs_rf_scol1.append(probs_rf)
    rfc_pred = rfc_model.predict(x_test)
    frfc = f1_score(rfc_pred, y_test)
    f1_rf_scol1.append(frfc)
    rocauc_rf = roc_auc_score(y_test, rfc_pred)
    rocauc_rf_scol1.append(rocauc_rf)
    recalls_rf = recall_score(rfc_pred, y_test)
    recall_rf_scol1.append(recalls_rf)
    precisions_rf = precision_score(rfc_pred, y_test)
    precision_rf_scol1.append(precisions_rf)
    accuracys_rf = accuracy_score(rfc_pred, y_test)
    accuracy_rf_scol1.append(accuracys_rf)
    rf_end = time.time()
    print("====RandomForest with Tfidf Tomelinks Train_Set & Dev_Set - 2010-2016", k)
    print('RFC F1 score', frfc * 100)
    print('RFC ROCAUC Score:', rocauc_rf * 100)
    print('RFC Recall score:', recalls_rf * 100)
    print('RFC Precision Score:', precisions_rf * 100)
    print('RFC Confusion Matrix', confusion_matrix(y_test, rfc_pred), "\n")
    print('RFC Classification', classification_report(y_test, rfc_pred), "\n")
    print('RFC Accuracy Score', accuracys_rf * 100)
    print("Execution Time for Random Forest Classifier Tomelinks Train_Set & Dev_Set - 2010-2016 ", rf_end - start5, "seconds")

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

from itertools import chain

subset1 = ['Train_Set' for t in range(50)]
other_set1 = ['Dev_Set' for t in range(50)]
sampling1 = ['Undersampling' for t in range(50)]
technique1 = ['Tomelinks' for t in range(50)]
classifier_names1 = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
#train_sizes_num1 = [0.95]
# v1 = [0, 1, 2, 3, 4]
# precision_csv_num1 = [precision_lr_scol[z], precision_dt_scol[z], precision_nb_scol[z], precision_xg_scol[z], precision_rf_scol[z]]
# recall_csv_num1 = [recall_lr_scol[z], recall_dt_scol[z], recall_nb_scol[z], recall_xg_scol[z], recall_rf_scol[z]]
# auc_csv_num1 = [rocauc_lr_scol[z], rocauc_dt_scol[z], rocauc_nb_scol[z], rocauc_xg_scol[z], rocauc_rf_scol[z]]
# accuracy_csv_num1 = [accuracy_lr_scol[z], accuracy_dt_scol[z], accuracy_nb_scol[z], accuracy_xg_scol[z], accuracy_rf_scol[z]]
precision_csv_num1 = [precision_lr_scol1, precision_dt_scol1, precision_nb_scol1, precision_xg_scol1, precision_rf_scol1]
recall_csv_num1 = [recall_lr_scol1, recall_dt_scol1, recall_nb_scol1, recall_xg_scol1, recall_rf_scol1]
auc_csv_num1 = [rocauc_lr_scol1, rocauc_dt_scol1, rocauc_nb_scol1, rocauc_xg_scol1, rocauc_rf_scol1]
accuracy_csv_num1 = [accuracy_lr_scol1, accuracy_dt_scol1, accuracy_nb_scol1, accuracy_xg_scol1, accuracy_rf_scol1]
import itertools
rounds = 5
# rounds1 = 1
p1 = itertools.cycle(classifier_names1)
o1 = itertools.cycle(subsample_num)
#k1 = itertools.cycle(train_sizes_num)
# v1 = itertools.cycle(score_location)
# pr1 = itertools.cycle(precision_num)
# y1 = itertools.cycle(iteration_csv)
classifier_csv1 = [next(p1) for _ in range(rounds)] * 10
subsample_csv1 = [r for r in subsample_num for _ in range(rounds)]
#test_size_csv1 = [next(o) for _ in range(rounds)] * 1
#train_size_csv1 = [next(k) for _ in range(rounds)] * 1
# split_csv1 = ['1' for t in range(25)]
# train_csv1 = ['0.5' for t in range(25)]
precision_csv1 = list(chain(*precision_csv_num1))
recall_csv1 = list(chain(*recall_csv_num1))
auc_csv1 = list(chain(*auc_csv_num1))
accuracy_csv1 = list(chain(*accuracy_csv_num1))
csv_data1 = [subset1, other_set1, sampling1, technique1, classifier_csv1, subsample_csv1, precision_csv1, recall_csv1, auc_csv1, accuracy_csv1]
export_data1 = zip_longest(*csv_data1, fillvalue='')
filename = "Subset_Iterations.csv"
with open(filename, 'a', newline='') as file:
    write = csv.writer(file)
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

x_nm1, y_nm1 = nm.fit_resample(subsample1, sublabel1)
x_nm2, y_nm2 = nm.fit_resample(subsample2, sublabel2)
x_nm3, y_nm3 = nm.fit_resample(subsample3, sublabel3)
x_nm4, y_nm4 = nm.fit_resample(subsample4, sublabel4)
x_nm5, y_nm5 = nm.fit_resample(subsample5, sublabel5)
x_nm6, y_nm6 = nm.fit_resample(subsample6, sublabel6)
x_nm7, y_nm7 = nm.fit_resample(subsample7, sublabel7)
x_nm8, y_nm8 = nm.fit_resample(subsample8, sublabel8)
x_nm9, y_nm9 = nm.fit_resample(subsample9, sublabel9)
x_nm10, y_nm10 = nm.fit_resample(subsample10, sublabel10)

sample_list1 = [x_nm1, x_nm2, x_nm3, x_nm4, x_nm5, x_nm6, x_nm7, x_nm8, x_nm9, x_nm10]
label_list1 = [y_nm1, y_nm2, y_nm3, y_nm4, y_nm5, y_nm6, y_nm7, y_nm8, y_nm9, y_nm10]


for i, j, z in zip(sample_list1, label_list1, subsample_num):
#for h, j, k, l in zip(tl_xtrain_list, tl_xtest_list, tl_ytrain_list, tl_ytest_list):
    x_train_i, x_test_i, y_train_i, y_test_i = train_test_split(i, j, train_size=0.5, test_size=0.5)
#x2_train, x2_test, y2_train, y2_test = train_test_split(x_nm, y_nm, train_size=0.955, test_size=0.05)

    start1 = time.time()
    log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
    model_lr = log.fit(x_train_i, y_train_i)
    probs_lr = model_lr.predict_proba(x_test_i)[:, 1]
    probs_lr_scol2.append(probs_lr)
    ly_prediction = log.predict(x_test_i)
    fly = f1_score(ly_prediction, y_test_i)
    f1_lr_scol2.append(fly)
    rocauc_lr = roc_auc_score(y_test_i, ly_prediction)
    rocauc_lr_scol2.append(rocauc_lr)
    recalls_lr = recall_score(y_test_i, ly_prediction)
    recall_lr_scol2.append(recalls_lr)
    precisions_lr = precision_score(y_test_i, ly_prediction)
    precision_lr_scol2.append(precisions_lr)
    accuracys_lr = accuracy_score(y_test_i, ly_prediction)
    accuracy_lr_scol2.append(accuracys_lr)
    print("===Logistic Regression with TfidfVectorizer NearMiss Train_Set & Dev_Set - 2010-2016")
    lr_end = time.time()
    print('Logistic F1-score', fly * 100)
    print('Logistic ROCAUC score:', rocauc_lr * 100)
    print('Logistic Recall score:', recalls_lr * 100)
    print('Logistic Precision Score:', precisions_lr * 100)
    print('Logistic Confusion Matrix', confusion_matrix(y_test_i, ly_prediction), "\n")
    print('Logistic Classification', classification_report(y_test_i, ly_prediction), "\n")
    print('Logistic Accuracy Score', accuracys_lr * 100)
    print("Execution Time for Logistic Regression NearMiss Train_Set & Dev_Set - 2010-2016", lr_end - start1, "seconds")

    start2 = time.time()
    from sklearn.tree import DecisionTreeClassifier

    DCT = DecisionTreeClassifier()
    model_dt = DCT.fit(x_train_i, y_train_i)
    probs_dt = model_dt.predict_proba(x_test_i)[:, 1]
    probs_dt_scol2.append(probs_dt)
    dct_pred = DCT.predict(x_test_i)
    fdct = f1_score(dct_pred, y_test_i)
    f1_dt_scol2.append(fdct)
    rocauc_dt = roc_auc_score(y_test_i,dct_pred)
    rocauc_dt_scol2.append(rocauc_dt)
    recalls_dt = recall_score(y_test_i,dct_pred)
    recall_dt_scol2.append(recalls_dt)
    precisions_dt = precision_score(y_test_i,dct_pred)
    precision_dt_scol2.append(precisions_dt)
    accuracys_dt = accuracy_score(y_test_i,dct_pred)
    accuracy_dt_scol2.append(accuracys_dt)
    print("===DecisionTreeClassifier with TfidfVectorizer NearMiss Train_Set & Dev_Set - 2010-2016")
    dt_end = time.time()
    print('DCT F1-score', fdct * 100)
    print('DCT ROCAUC score:', rocauc_dt * 100)
    print('DCT Recall score:', recalls_dt * 100)
    print('DCT Precision Score:', precisions_dt * 100)
    print('DCT Confusion Matrix', confusion_matrix(y_test_i,dct_pred), "\n")
    print('DCT Classification', classification_report(y_test_i,dct_pred), "\n")
    print('DCT Accuracy Score', accuracys_dt * 100)
    print("Execution Time for Decision Tree NearMiss Train_Set & Dev_Set - 2010-2016", dt_end - start2, "seconds")

    from sklearn.naive_bayes import MultinomialNB

    start3 = time.time()
    Naive = MultinomialNB()
    model_nb = Naive.fit(x_train_i, y_train_i)
    probs_nb = model_nb.predict_proba(x_test_i)[:, 1]
    probs_nb_scol2.append(probs_nb)
    # predict the labels on validation dataset
    ny_pred = Naive.predict(x_test_i)
    fnb = f1_score(ny_pred, y_test_i)
    f1_nb_scol2.append(fnb)
    rocauc_nb = roc_auc_score(y_test_i,ny_pred)
    rocauc_nb_scol2.append(rocauc_nb)
    recalls_nb = recall_score(y_test_i,ny_pred)
    recall_nb_scol2.append(recalls_nb)
    precisions_nb = precision_score(y_test_i,ny_pred)
    precision_nb_scol2.append(precisions_nb)
    accuracys_nb = accuracy_score(y_test_i,ny_pred)
    accuracy_nb_scol2.append(accuracys_nb)
    nb_end = time.time()
    # Use accuracy_score function to get the accuracy
    print("===Naive Bayes with TfidfVectorizer Imabalanced NearMiss Train_Set & Dev_Set - 2010-2016")
    print('Naive F1-score', fnb * 100)
    print('Naive ROCAUC score:', rocauc_nb * 100)
    print('Naive Recall score:', recalls_nb * 100)
    print('Naive Precision Score:', precisions_nb * 100)
    print('Naive Confusion Matrix', confusion_matrix(y_test_i,ny_pred), "\n")
    print('Naive Classification', classification_report(y_test_i,ny_pred), "\n")
    print('Naive Accuracy Score', accuracys_nb * 100)
    print("Execution Time for Naive Bayes NearMiss Train_Set & Dev_Set - 2010-2016", nb_end - start3, "seconds")

    # XGBoost Classifier

    start4 = time.time()
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    xgb_model = XGBClassifier().fit(x_train_i, y_train_i)
    probs_xg = xgb_model.predict_proba(x_test_i)[:, 1]
    probs_xg_scol2.append(probs_xg)
    # predict
    xgb_y_predict = xgb_model.predict(x_test_i)
    fxg = f1_score(xgb_y_predict, y_test_i)
    f1_xg_scol2.append(fxg)
    rocauc_xg = roc_auc_score(xgb_y_predict, y_test_i)
    rocauc_xg_scol2.append(rocauc_xg)
    recall_xg = recall_score(xgb_y_predict, y_test_i)
    recall_xg_scol2.append(recall_xg)
    precisions_xg = precision_score(xgb_y_predict, y_test_i)
    precision_xg_scol2.append(precisions_xg)
    accuracys_xg = accuracy_score(xgb_y_predict, y_test_i)
    accuracy_xg_scol2.append(accuracys_xg)
    xg_end = time.time()
    print("===XGB with TfidfVectorizer NearMiss Train_Set & Dev_Set - 2010-2016")
    print('XGB F1-Score', fxg * 100)
    print('XGB ROCAUC Score:', rocauc_xg * 100)
    print('XGB Recall score:', recall_xg * 100)
    print('XGB Precision Score:', precisions_xg * 100)
    print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, y_test_i), "\n")
    print('XGB Classification', classification_report(xgb_y_predict, y_test_i), "\n")
    print('XGB Accuracy Score', accuracys_nb * 100)
    print("Execution Time for XGBoost Classifier NearMiss Train_Set & Dev_Set - 2010-2016", xg_end - start4, "seconds")

    # Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier

    start5 = time.time()
    rfc_model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(x_train_i, y_train_i)
    probs_rf = rfc_model.predict_proba(x_test_i)[:, 1]
    probs_rf_scol2.append(probs_rf)
    rfc_pred = rfc_model.predict(x_test_i)
    frfc = f1_score(rfc_pred, y_test_i)
    f1_rf_scol2.append(frfc)
    rocauc_rf = roc_auc_score(y_test_i,rfc_pred)
    rocauc_rf_scol2.append(rocauc_rf)
    recalls_rf = recall_score(rfc_pred, y_test_i)
    recall_rf_scol2.append(recalls_rf)
    precisions_rf = precision_score(rfc_pred, y_test_i)
    precision_rf_scol2.append(precisions_rf)
    accuracys_rf = accuracy_score(rfc_pred, y_test_i)
    accuracy_rf_scol2.append(accuracys_rf)
    rf_end = time.time()
    print("====RandomForest with Tfidf NearMiss Train_Set & Dev_Set - 2010-2016")
    print('RFC F1 score', frfc * 100)
    print('RFC ROCAUC Score:', rocauc_rf * 100)
    print('RFC Recall score:', recalls_rf * 100)
    print('RFC Precision Score:', precisions_rf * 100)
    print('RFC Confusion Matrix', confusion_matrix(y_test_i,rfc_pred), "\n")
    print('RFC Classification', classification_report(y_test_i,rfc_pred), "\n")
    print('RFC Accuracy Score', accuracys_rf * 100)
    print("Execution Time for Random Forest Classifier NearMiss Train_Set & Dev_Set - 2010-2016 ", rf_end - start5, "seconds")

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

subset2 = ['Train_Set' for t in range(50)]
other_set2 = ['Dev_Set' for t in range(50)]
sampling2 = ['Undersampling' for t in range(50)]
technique2 = ['NearMiss' for t in range(50)]
classifier_names2 = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
#train_sizes_num2 = [0.95]
# v2 = [0, 1, 2, 3, 4]
# precision_csv_num2 = [precision_lr_scol[z], precision_dt_scol[z], precision_nb_scol[z], precision_xg_scol[z], precision_rf_scol[z]]
# recall_csv_num2 = [recall_lr_scol[z], recall_dt_scol[z], recall_nb_scol[z], recall_xg_scol[z], recall_rf_scol[z]]
# auc_csv_num2 = [rocauc_lr_scol[z], rocauc_dt_scol[z], rocauc_nb_scol[z], rocauc_xg_scol[z], rocauc_rf_scol[z]]
# accuracy_csv_num2 = [accuracy_lr_scol[z], accuracy_dt_scol[z], accuracy_nb_scol[z], accuracy_xg_scol[z], accuracy_rf_scol[z]]
precision_csv_num2 = [precision_lr_scol2, precision_dt_scol2, precision_nb_scol2, precision_xg_scol2, precision_rf_scol2]
recall_csv_num2 = [recall_lr_scol2, recall_dt_scol2, recall_nb_scol2, recall_xg_scol2, recall_rf_scol2]
auc_csv_num2 = [rocauc_lr_scol2, rocauc_dt_scol2, rocauc_nb_scol2, rocauc_xg_scol2, rocauc_rf_scol2]
accuracy_csv_num2 = [accuracy_lr_scol2, accuracy_dt_scol2, accuracy_nb_scol2, accuracy_xg_scol2, accuracy_rf_scol2]
import itertools
rounds = 5
# rounds1 = 1
p2 = itertools.cycle(classifier_names2)
o2 = itertools.cycle(subsample_num)
#k2 = itertools.cycle(train_sizes_num)
# v2 = itertools.cycle(score_location)
# pr2 = itertools.cycle(precision_num)
# y2 = itertools.cycle(iteration_csv)
classifier_csv2 = [next(p2) for _ in range(rounds)] * 10
subsample_csv2 = [r for r in subsample_num for _ in range(rounds)]
#test_size_csv2 = [next(o) for _ in range(rounds)] * 1
#train_size_csv2 = [next(k) for _ in range(rounds)] * 1
# split_csv2 = ['1' for t in range(25)]
# train_csv2 = ['0.5' for t in range(25)]
precision_csv2 = list(chain(*precision_csv_num2))
recall_csv2 = list(chain(*recall_csv_num2))
auc_csv2 = list(chain(*auc_csv_num2))
accuracy_csv2 = list(chain(*accuracy_csv_num2))
csv_data2 = [subset2, other_set2, sampling2, technique2, classifier_csv2, subsample_csv2, precision_csv2, recall_csv2, auc_csv2, accuracy_csv2]
export_data2 = zip_longest(*csv_data, fillvalue='')
filename = "Subset_Iterations.csv"
with open(filename, 'a', newline='') as file:
    write = csv.writer(file)
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

x_smote1, y_smote1 = smote.fit_resample(subsample1, sublabel1)
x_smote2, y_smote2 = smote.fit_resample(subsample2, sublabel2)
x_smote3, y_smote3 = smote.fit_resample(subsample3, sublabel3)
x_smote4, y_smote4 = smote.fit_resample(subsample4, sublabel4)
x_smote5, y_smote5 = smote.fit_resample(subsample5, sublabel5)
x_smote6, y_smote6 = smote.fit_resample(subsample6, sublabel6)
x_smote7, y_smote7 = smote.fit_resample(subsample7, sublabel7)
x_smote8, y_smote8 = smote.fit_resample(subsample8, sublabel8)
x_smote9, y_smote9 = smote.fit_resample(subsample9, sublabel9)
x_smote10, y_smote10 = smote.fit_resample(subsample10, sublabel10)

sample_list2 = [x_smote1, x_smote2, x_smote3, x_smote4, x_smote5, x_smote6, x_smote7, x_smote8, x_smote9, x_smote10]
label_list2 = [y_smote1, y_smote2, y_smote3, y_smote4, y_smote5, y_smote6, y_smote7, y_smote8, y_smote9, y_smote10]

for i, j, z in zip(sample_list2, label_list2, subsample_num):
#for h, j, k, l in zip(tl_xtrain_list, tl_xtest_list, tl_ytrain_list, tl_ytest_list):
    x_train_i, x_test_i, y_train_i, y_test_i = train_test_split(i, j, train_size=0.5, test_size=0.5)

    start1 = time.time()
    log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
    model_lr = log.fit(x_train_i, y_train_i)
    probs_lr = model_lr.predict_proba(x_test_i)[:, 1]
    probs_lr_scol3.append(probs_lr)
    ly_prediction = log.predict(x_test_i)
    fly = f1_score(ly_prediction, y_test_i)
    f1_lr_scol3.append(fly)
    rocauc_lr = roc_auc_score(y_test_i, ly_prediction)
    rocauc_lr_scol3.append(rocauc_lr)
    recalls_lr = recall_score(y_test_i, ly_prediction)
    recall_lr_scol3.append(recalls_lr)
    precisions_lr = precision_score(y_test_i, ly_prediction)
    precision_lr_scol3.append(precisions_lr)
    accuracys_lr = accuracy_score(y_test_i, ly_prediction)
    accuracy_lr_scol3.append(accuracys_lr)
    print("===Logistic Regression with TfidfVectorizer SMOTE Train_Set & Dev_Set - 2010-2016")
    lr_end = time.time()
    print('Logistic F1-score', fly * 100)
    print('Logistic ROCAUC score:', rocauc_lr * 100)
    print('Logistic Recall score:', recalls_lr * 100)
    print('Logistic Precision Score:', precisions_lr * 100)
    print('Logistic Confusion Matrix', confusion_matrix(y_test_i, ly_prediction), "\n")
    print('Logistic Classification', classification_report(y_test_i, ly_prediction), "\n")
    print('Logistic Accuracy Score', accuracys_lr * 100)
    print("Execution Time for Logistic Regression SMOTE Train_Set & Dev_Set - 2010-2016", lr_end - start1, "seconds")

    start2 = time.time()
    from sklearn.tree import DecisionTreeClassifier

    DCT = DecisionTreeClassifier()
    model_dt = DCT.fit(x_train_i, y_train_i)
    probs_dt = model_dt.predict_proba(x_test_i)[:, 1]
    probs_dt_scol3.append(probs_dt)
    dct_pred = DCT.predict(x_test_i)
    fdct = f1_score(dct_pred, y_test_i)
    f1_dt_scol3.append(fdct)
    rocauc_dt = roc_auc_score(y_test_i, dct_pred)
    rocauc_dt_scol3.append(rocauc_dt)
    recalls_dt = recall_score(y_test_i, dct_pred)
    recall_dt_scol3.append(recalls_dt)
    precisions_dt = precision_score(y_test_i, dct_pred)
    precision_dt_scol3.append(precisions_dt)
    accuracys_dt = accuracy_score(y_test_i, dct_pred)
    accuracy_dt_scol3.append(accuracys_dt)
    print("===DecisionTreeClassifier with TfidfVectorizer SMOTE Train_Set & Dev_Set - 2010-2016")
    dt_end = time.time()
    print('DCT F1-score', fdct * 100)
    print('DCT ROCAUC score:', rocauc_dt * 100)
    print('DCT Recall score:', recalls_dt * 100)
    print('DCT Precision Score:', precisions_dt * 100)
    print('DCT Confusion Matrix', confusion_matrix(y_test_i, dct_pred), "\n")
    print('DCT Classification', classification_report(y_test_i, dct_pred), "\n")
    print('DCT Accuracy Score', accuracys_dt * 100)
    print("Execution Time for Decision Tree SMOTE Train_Set & Dev_Set - 2010-2016", dt_end - start2, "seconds")

    from sklearn.naive_bayes import MultinomialNB

    start3 = time.time()
    Naive = MultinomialNB()
    model_nb = Naive.fit(x_train_i, y_train_i)
    probs_nb = model_nb.predict_proba(x_test_i)[:, 1]
    probs_nb_scol3.append(probs_nb)
    # predict the labels on validation dataset
    ny_pred = Naive.predict(x_test_i)
    fnb = f1_score(ny_pred, y_test_i)
    f1_nb_scol3.append(fnb)
    rocauc_nb = roc_auc_score(y_test_i, ny_pred)
    rocauc_nb_scol3.append(rocauc_nb)
    recalls_nb = recall_score(y_test_i, ny_pred)
    recall_nb_scol3.append(recalls_nb)
    precisions_nb = precision_score(y_test_i, ny_pred)
    precision_nb_scol3.append(precisions_nb)
    accuracys_nb = accuracy_score(y_test_i, ny_pred)
    accuracy_nb_scol3.append(accuracys_nb)
    nb_end = time.time()
    # Use accuracy_score function to get the accuracy
    print("===Naive Bayes with TfidfVectorizer Imabalanced SMOTE Train_Set & Dev_Set - 2010-2016")
    print('Naive F1-score', fnb * 100)
    print('Naive ROCAUC score:', rocauc_nb * 100)
    print('Naive Recall score:', recalls_nb * 100)
    print('Naive Precision Score:', precisions_nb * 100)
    print('Naive Confusion Matrix', confusion_matrix(y_test_i, ny_pred), "\n")
    print('Naive Classification', classification_report(y_test_i, ny_pred), "\n")
    print('Naive Accuracy Score', accuracys_nb * 100)
    print("Execution Time for Naive Bayes SMOTE Train_Set & Dev_Set - 2010-2016", nb_end - start3, "seconds")

    # XGBoost Classifier

    start4 = time.time()
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    xgb_model = XGBClassifier().fit(x_train_i, y_train_i)
    probs_xg = xgb_model.predict_proba(x_test_i)[:, 1]
    probs_xg_scol3.append(probs_xg)
    # predict
    xgb_y_predict = xgb_model.predict(x_test_i)
    fxg = f1_score(xgb_y_predict, y_test_i)
    f1_xg_scol3.append(fxg)
    rocauc_xg = roc_auc_score(xgb_y_predict, y_test_i)
    rocauc_xg_scol3.append(rocauc_xg)
    recall_xg = recall_score(xgb_y_predict, y_test_i)
    recall_xg_scol3.append(recall_xg)
    precisions_xg = precision_score(xgb_y_predict, y_test_i)
    precision_xg_scol3.append(precisions_xg)
    accuracys_xg = accuracy_score(xgb_y_predict, y_test_i)
    accuracy_xg_scol3.append(accuracys_xg)
    xg_end = time.time()
    print("===XGB with TfidfVectorizer SMOTE Train_Set & Dev_Set - 2010-2016")
    print('XGB F1-Score', fxg * 100)
    print('XGB ROCAUC Score:', rocauc_xg * 100)
    print('XGB Recall score:', recall_xg * 100)
    print('XGB Precision Score:', precisions_xg * 100)
    print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, y_test_i), "\n")
    print('XGB Classification', classification_report(xgb_y_predict, y_test_i), "\n")
    print('XGB Accuracy Score', accuracys_nb * 100)
    print("Execution Time for XGBoost Classifier SMOTE Train_Set & Dev_Set - 2010-2016", xg_end - start4, "seconds")

    # Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier

    start5 = time.time()
    rfc_model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(x_train_i, y_train_i)
    probs_rf = rfc_model.predict_proba(x_test_i)[:, 1]
    probs_rf_scol3.append(probs_rf)
    rfc_pred = rfc_model.predict(x_test_i)
    frfc = f1_score(rfc_pred, y_test_i)
    f1_rf_scol3.append(frfc)
    rocauc_rf = roc_auc_score(y_test_i, rfc_pred)
    rocauc_rf_scol3.append(rocauc_rf)
    recalls_rf = recall_score(rfc_pred, y_test_i)
    recall_rf_scol3.append(recalls_rf)
    precisions_rf = precision_score(rfc_pred, y_test_i)
    precision_rf_scol3.append(precisions_rf)
    accuracys_rf = accuracy_score(rfc_pred, y_test_i)
    accuracy_rf_scol3.append(accuracys_rf)
    rf_end = time.time()
    print("====RandomForest with Tfidf SMOTE Train_Set & Dev_Set - 2010-2016")
    print('RFC F1 score', frfc * 100)
    print('RFC ROCAUC Score:', rocauc_rf * 100)
    print('RFC Recall score:', recalls_rf * 100)
    print('RFC Precision Score:', precisions_rf * 100)
    print('RFC Confusion Matrix', confusion_matrix(y_test_i, rfc_pred), "\n")
    print('RFC Classification', classification_report(y_test_i, rfc_pred), "\n")
    print('RFC Accuracy Score', accuracys_rf * 100)
    print("Execution Time for Random Forest Classifier SMOTE Train_Set & Dev_Set - 2010-2016 ", rf_end - start5, "seconds")

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

subset3 = ['Train_Set' for t in range(50)]
other_set3 = ['Dev_Set' for t in range(50)]
sampling3 = ['Oversampling' for t in range(50)]
technique3 = ['SMOTE' for t in range(50)]
classifier_names3 = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
#train_sizes_num3 = [0.95]
# v3 = [0, 1, 2, 3, 4]
# precision_csv_num3 = [precision_lr_scol[z], precision_dt_scol[z], precision_nb_scol[z], precision_xg_scol[z], precision_rf_scol[z]]
# recall_csv_num3 = [recall_lr_scol[z], recall_dt_scol[z], recall_nb_scol[z], recall_xg_scol[z], recall_rf_scol[z]]
# auc_csv_num3 = [rocauc_lr_scol[z], rocauc_dt_scol[z], rocauc_nb_scol[z], rocauc_xg_scol[z], rocauc_rf_scol[z]]
# accuracy_csv_num3 = [accuracy_lr_scol[z], accuracy_dt_scol[z], accuracy_nb_scol[z], accuracy_xg_scol[z], accuracy_rf_scol[z]]
precision_csv_num3 = [precision_lr_scol3, precision_dt_scol3, precision_nb_scol3, precision_xg_scol3, precision_rf_scol3]
recall_csv_num3 = [recall_lr_scol3, recall_dt_scol3, recall_nb_scol3, recall_xg_scol3, recall_rf_scol3]
auc_csv_num3 = [rocauc_lr_scol3, rocauc_dt_scol3, rocauc_nb_scol3, rocauc_xg_scol3, rocauc_rf_scol3]
accuracy_csv_num3 = [accuracy_lr_scol3, accuracy_dt_scol3, accuracy_nb_scol3, accuracy_xg_scol3, accuracy_rf_scol3]
import itertools
rounds = 5
# rounds1 = 1
p3 = itertools.cycle(classifier_names3)
o3 = itertools.cycle(subsample_num)
#k3 = itertools.cycle(train_sizes_num)
# v3 = itertools.cycle(score_location)
# pr3 = itertools.cycle(precision_num)
# y3 = itertools.cycle(iteration_csv)
classifier_csv3 = [next(p3) for _ in range(rounds)] * 10
subsample_csv3 = [r for r in subsample_num for _ in range(rounds)]
#test_size_csv3 = [next(o) for _ in range(rounds)] * 1
#train_size_csv3 = [next(k) for _ in range(rounds)] * 1
# split_csv3 = ['1' for t in range(25)]
# train_csv3 = ['0.5' for t in range(25)]
precision_csv3 = list(chain(*precision_csv_num3))
recall_csv3 = list(chain(*recall_csv_num3))
auc_csv3 = list(chain(*auc_csv_num3))
accuracy_csv3 = list(chain(*accuracy_csv_num3))
csv_data3 = [subset3, other_set3, sampling3, technique3, classifier_csv3, subsample_csv3, precision_csv3, recall_csv3, auc_csv3, accuracy_csv3]
export_data3 = zip_longest(*csv_data, fillvalue='')
filename = "Subset_Iterations.csv"
with open(filename, 'a', newline='') as file:
    write = csv.writer(file)
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

x_ros1, y_ros1 = ros.fit_resample(subsample1, sublabel1)
x_ros2, y_ros2 = ros.fit_resample(subsample2, sublabel2)
x_ros3, y_ros3 = ros.fit_resample(subsample3, sublabel3)
x_ros4, y_ros4 = ros.fit_resample(subsample4, sublabel4)
x_ros5, y_ros5 = ros.fit_resample(subsample5, sublabel5)
x_ros6, y_ros6 = ros.fit_resample(subsample6, sublabel6)
x_ros7, y_ros7 = ros.fit_resample(subsample7, sublabel7)
x_ros8, y_ros8 = ros.fit_resample(subsample8, sublabel8)
x_ros9, y_ros9 = ros.fit_resample(subsample9, sublabel9)
x_ros10, y_ros10 = ros.fit_resample(subsample10, sublabel10)

sample_list3 = [x_ros1, x_ros2, x_ros3, x_ros4, x_ros5, x_ros6, x_ros7, x_ros8, x_ros9, x_ros10]
label_list3 = [y_ros1, y_ros2, y_ros3, y_ros4, y_ros5, y_ros6, y_ros7, y_ros8, y_ros9, y_ros10]




for i, j, z in zip(sample_list2, label_list2, subsample_num):
#for h, j, k, l in zip(tl_xtrain_list, tl_xtest_list, tl_ytrain_list, tl_ytest_list):
    x_train_i, x_test_i, y_train_i, y_test_i = train_test_split(i, j, train_size=0.5, test_size=0.5)

#h, j, k, y4_test = train_test_split(x_ros, y_ros, train_size=0.95, test_size=0.05)

    start1 = time.time()
    log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
    model_lr = log.fit(x_train_i, y_train_i)
    probs_lr = model_lr.predict_proba(x_test_i)[:, 1]
    probs_lr_scol4.append(probs_lr)
    ly_prediction = log.predict(x_test_i)
    fly = f1_score(ly_prediction, y_test_i)
    f1_lr_scol4.append(fly)
    rocauc_lr = roc_auc_score(y_test_i, ly_prediction)
    rocauc_lr_scol4.append(rocauc_lr)
    recalls_lr = recall_score(y_test_i, ly_prediction)
    recall_lr_scol4.append(recalls_lr)
    precisions_lr = precision_score(y_test_i, ly_prediction)
    precision_lr_scol4.append(precisions_lr)
    accuracys_lr = accuracy_score(y_test_i, ly_prediction)
    accuracy_lr_scol4.append(accuracys_lr)
    print("===Logistic Regression with TfidfVectorizer ROS Train_Set & Dev_Set - 2010-2016")
    lr_end = time.time()
    print('Logistic F1-score', fly * 100)
    print('Logistic ROCAUC score:', rocauc_lr * 100)
    print('Logistic Recall score:', recalls_lr * 100)
    print('Logistic Precision Score:', precisions_lr * 100)
    print('Logistic Confusion Matrix', confusion_matrix(y_test_i, ly_prediction), "\n")
    print('Logistic Classification', classification_report(y_test_i, ly_prediction), "\n")
    print('Logistic Accuracy Score', accuracys_lr * 100)
    print("Execution Time for Logistic Regression ROS Train_Set & Dev_Set - 2010-2016", lr_end - start1, "seconds")

    start2 = time.time()
    from sklearn.tree import DecisionTreeClassifier

    DCT = DecisionTreeClassifier()
    model_dt = DCT.fit(x_train_i, y_train_i)
    probs_dt = model_dt.predict_proba(x_test_i)[:, 1]
    probs_dt_scol4.append(probs_dt)
    dct_pred = DCT.predict(x_test_i)
    fdct = f1_score(dct_pred, y_test_i)
    f1_dt_scol4.append(fdct)
    rocauc_dt = roc_auc_score(y_test_i, dct_pred)
    rocauc_dt_scol4.append(rocauc_dt)
    recalls_dt = recall_score(y_test_i, dct_pred)
    recall_dt_scol4.append(recalls_dt)
    precisions_dt = precision_score(y_test_i, dct_pred)
    precision_dt_scol4.append(precisions_dt)
    accuracys_dt = accuracy_score(y_test_i, dct_pred)
    accuracy_dt_scol4.append(accuracys_dt)
    print("===DecisionTreeClassifier with TfidfVectorizer ROS Train_Set & Dev_Set - 2010-2016")
    dt_end = time.time()
    print('DCT F1-score', fdct * 100)
    print('DCT ROCAUC score:', rocauc_dt * 100)
    print('DCT Recall score:', recalls_dt * 100)
    print('DCT Precision Score:', precisions_dt * 100)
    print('DCT Confusion Matrix', confusion_matrix(y_test_i, dct_pred), "\n")
    print('DCT Classification', classification_report(y_test_i, dct_pred), "\n")
    print('DCT Accuracy Score', accuracys_dt * 100)
    print("Execution Time for Decision Tree ROS Train_Set & Dev_Set - 2010-2016", dt_end - start2, "seconds")

    from sklearn.naive_bayes import MultinomialNB

    start3 = time.time()
    Naive = MultinomialNB()
    model_nb = Naive.fit(x_train_i, y_train_i)
    probs_nb = model_nb.predict_proba(x_test_i)[:, 1]
    probs_nb_scol4.append(probs_nb)
    # predict the labels on validation dataset
    ny_pred = Naive.predict(x_test_i)
    fnb = f1_score(ny_pred, y_test_i)
    f1_nb_scol4.append(fnb)
    rocauc_nb = roc_auc_score(y_test_i, ny_pred)
    rocauc_nb_scol4.append(rocauc_nb)
    recalls_nb = recall_score(y_test_i, ny_pred)
    recall_nb_scol4.append(recalls_nb)
    precisions_nb = precision_score(y_test_i, ny_pred)
    precision_nb_scol4.append(precisions_nb)
    accuracys_nb = accuracy_score(y_test_i, ny_pred)
    accuracy_nb_scol4.append(accuracys_nb)
    nb_end = time.time()
    # Use accuracy_score function to get the accuracy
    print("===Naive Bayes with TfidfVectorizer Imabalanced ROS Train_Set & Dev_Set - 2010-2016")
    print('Naive F1-score', fnb * 100)
    print('Naive ROCAUC score:', rocauc_nb * 100)
    print('Naive Recall score:', recalls_nb * 100)
    print('Naive Precision Score:', precisions_nb * 100)
    print('Naive Confusion Matrix', confusion_matrix(y_test_i, ny_pred), "\n")
    print('Naive Classification', classification_report(y_test_i, ny_pred), "\n")
    print('Naive Accuracy Score', accuracys_nb * 100)
    print("Execution Time for Naive Bayes ROS Train_Set & Dev_Set - 2010-2016", nb_end - start3, "seconds")

    # XGBoost Classifier

    start4 = time.time()
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    xgb_model = XGBClassifier().fit(x_train_i, y_train_i)
    probs_xg = xgb_model.predict_proba(x_test_i)[:, 1]
    probs_xg_scol4.append(probs_xg)
    # predict
    xgb_y_predict = xgb_model.predict(x_test_i)
    fxg = f1_score(xgb_y_predict, y_test_i)
    f1_xg_scol4.append(fxg)
    rocauc_xg = roc_auc_score(xgb_y_predict, y_test_i)
    rocauc_xg_scol4.append(rocauc_xg)
    recall_xg = recall_score(xgb_y_predict, y_test_i)
    recall_xg_scol4.append(recall_xg)
    precisions_xg = precision_score(xgb_y_predict, y_test_i)
    precision_xg_scol4.append(precisions_xg)
    accuracys_xg = accuracy_score(xgb_y_predict, y_test_i)
    accuracy_xg_scol4.append(accuracys_xg)
    xg_end = time.time()
    print("===XGB with TfidfVectorizer ROS Train_Set & Dev_Set - 2010-2016")
    print('XGB F1-Score', fxg * 100)
    print('XGB ROCAUC Score:', rocauc_xg * 100)
    print('XGB Recall score:', recall_xg * 100)
    print('XGB Precision Score:', precisions_xg * 100)
    print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, y_test_i), "\n")
    print('XGB Classification', classification_report(xgb_y_predict, y_test_i), "\n")
    print('XGB Accuracy Score', accuracys_nb * 100)
    print("Execution Time for XGBoost Classifier ROS Train_Set & Dev_Set - 2010-2016", xg_end - start4, "seconds")

    # Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier

    start5 = time.time()
    rfc_model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(x_train_i, y_train_i)
    probs_rf = rfc_model.predict_proba(x_test_i)[:, 1]
    probs_rf_scol4.append(probs_rf)
    rfc_pred = rfc_model.predict(x_test_i)
    frfc = f1_score(rfc_pred, y_test_i)
    f1_rf_scol4.append(frfc)
    rocauc_rf = roc_auc_score(y_test_i, rfc_pred)
    rocauc_rf_scol4.append(rocauc_rf)
    recalls_rf = recall_score(rfc_pred, y_test_i)
    recall_rf_scol4.append(recalls_rf)
    precisions_rf = precision_score(rfc_pred, y_test_i)
    precision_rf_scol4.append(precisions_rf)
    accuracys_rf = accuracy_score(rfc_pred, y_test_i)
    accuracy_rf_scol4.append(accuracys_rf)
    rf_end = time.time()
    print("====RandomForest with Tfidf ROS Train_Set & Dev_Set - 2010-2016")
    print('RFC F1 score', frfc * 100)
    print('RFC ROCAUC Score:', rocauc_rf * 100)
    print('RFC Recall score:', recalls_rf * 100)
    print('RFC Precision Score:', precisions_rf * 100)
    print('RFC Confusion Matrix', confusion_matrix(y_test_i, rfc_pred), "\n")
    print('RFC Classification', classification_report(y_test_i, rfc_pred), "\n")
    print('RFC Accuracy Score', accuracys_rf * 100)
    print("Execution Time for Random Forest Classifier ROS Train_Set & Dev_Set - 2010-2016 ", rf_end - start5, "seconds")

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

subset4 = ['Train_Set' for t in range(50)]
other_set4 = ['Dev_Set' for t in range(50)]
sampling4 = ['Oversampling' for t in range(50)]
technique4 = ['ROS' for t in range(50)]
classifier_names4 = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
#train_sizes_num4 = [0.95]
# v4 = [0, 1, 2, 3, 4]
# precision_csv_num4 = [precision_lr_scol[z], precision_dt_scol[z], precision_nb_scol[z], precision_xg_scol[z], precision_rf_scol[z]]
# recall_csv_num4 = [recall_lr_scol[z], recall_dt_scol[z], recall_nb_scol[z], recall_xg_scol[z], recall_rf_scol[z]]
# auc_csv_num4 = [rocauc_lr_scol[z], rocauc_dt_scol[z], rocauc_nb_scol[z], rocauc_xg_scol[z], rocauc_rf_scol[z]]
# accuracy_csv_num4 = [accuracy_lr_scol[z], accuracy_dt_scol[z], accuracy_nb_scol[z], accuracy_xg_scol[z], accuracy_rf_scol[z]]
precision_csv_num4 = [precision_lr_scol4, precision_dt_scol4, precision_nb_scol4, precision_xg_scol4, precision_rf_scol4]
recall_csv_num4 = [recall_lr_scol4, recall_dt_scol4, recall_nb_scol4, recall_xg_scol4, recall_rf_scol4]
auc_csv_num4 = [rocauc_lr_scol4, rocauc_dt_scol4, rocauc_nb_scol4, rocauc_xg_scol4, rocauc_rf_scol4]
accuracy_csv_num4 = [accuracy_lr_scol4, accuracy_dt_scol4, accuracy_nb_scol4, accuracy_xg_scol4, accuracy_rf_scol4]
import itertools
rounds = 5
# rounds1 = 1
p4 = itertools.cycle(classifier_names4)
o4 = itertools.cycle(subsample_num)
#k4 = itertools.cycle(train_sizes_num)
# v4 = itertools.cycle(score_location)
# pr4 = itertools.cycle(precision_num)
# y4 = itertools.cycle(iteration_csv)
classifier_csv4 = [next(p4) for _ in range(rounds)] * 10
subsample_csv4 = [r for r in subsample_num for _ in range(rounds)]
#test_size_csv4 = [next(o) for _ in range(rounds)] * 1
#train_size_csv4 = [next(k) for _ in range(rounds)] * 1
# split_csv4 = ['1' for t in range(25)]
# train_csv4 = ['0.5' for t in range(25)]
precision_csv4 = list(chain(*precision_csv_num4))
recall_csv4 = list(chain(*recall_csv_num4))
auc_csv4 = list(chain(*auc_csv_num4))
accuracy_csv4 = list(chain(*accuracy_csv_num4))
csv_data4 = [subset4, other_set4, sampling4, technique4, classifier_csv4, subsample_csv4, precision_csv4, recall_csv4, auc_csv4, accuracy_csv4]
export_data4 = zip_longest(*csv_data4, fillvalue='')
filename = "Subset_Iterations.csv"
with open(filename, 'a', newline='') as file:
    write = csv.writer(file)
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

x_rus1, y_rus1 = rus.fit_resample(subsample1, sublabel1)
x_rus2, y_rus2 = rus.fit_resample(subsample2, sublabel2)
x_rus3, y_rus3 = rus.fit_resample(subsample3, sublabel3)
x_rus4, y_rus4 = rus.fit_resample(subsample4, sublabel4)
x_rus5, y_rus5 = rus.fit_resample(subsample5, sublabel5)
x_rus6, y_rus6 = rus.fit_resample(subsample6, sublabel6)
x_rus7, y_rus7 = rus.fit_resample(subsample7, sublabel7)
x_rus8, y_rus8 = rus.fit_resample(subsample8, sublabel8)
x_rus9, y_rus9 = rus.fit_resample(subsample9, sublabel9)
x_rus10, y_rus10 = rus.fit_resample(subsample10, sublabel10)

sample_list4 = [x_rus1, x_rus2, x_rus3, x_rus4, x_rus5, x_rus6, x_rus7, x_rus8, x_rus9, x_rus10]
label_list4= [y_rus1, y_rus2, y_rus3, y_rus4, y_rus5, y_rus6, y_rus7, y_rus8, y_rus9, y_rus10]

for i, j, z in zip(sample_list4, label_list4, subsample_num):
#for h, j, k, l in zip(tl_xtrain_list, tl_xtest_list, tl_ytrain_list, tl_ytest_list):
    x_train_i, x_test_i, y_train_i, y_test_i = train_test_split(i, j, train_size=0.5, test_size=0.5)

    start1 = time.time()
    log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
    model_lr = log.fit(x_train_i, y_train_i)
    probs_lr = model_lr.predict_proba(x_test_i)[:, 1]
    probs_lr_scol5.append(probs_lr)
    ly_prediction = log.predict(x_test_i)
    fly = f1_score(ly_prediction, y_test_i)
    f1_lr_scol5.append(fly)
    rocauc_lr = roc_auc_score(y_test_i, ly_prediction)
    rocauc_lr_scol5.append(rocauc_lr)
    recalls_lr = recall_score(y_test_i, ly_prediction)
    recall_lr_scol5.append(recalls_lr)
    precisions_lr = precision_score(y_test_i, ly_prediction)
    precision_lr_scol5.append(precisions_lr)
    accuracys_lr = accuracy_score(y_test_i, ly_prediction)
    accuracy_lr_scol5.append(accuracys_lr)
    print("===Logistic Regression with TfidfVectorizer RUS Train_Set & Dev_Set - 2010-2016")
    lr_end = time.time()
    print('Logistic F1-score', fly * 100)
    print('Logistic ROCAUC score:', rocauc_lr * 100)
    print('Logistic Recall score:', recalls_lr * 100)
    print('Logistic Precision Score:', precisions_lr * 100)
    print('Logistic Confusion Matrix', confusion_matrix(y_test_i, ly_prediction), "\n")
    print('Logistic Classification', classification_report(y_test_i, ly_prediction), "\n")
    print('Logistic Accuracy Score', accuracys_lr * 100)
    print("Execution Time for Logistic Regression RUS Train_Set & Dev_Set - 2010-2016", lr_end - start1, "seconds")

    start2 = time.time()
    from sklearn.tree import DecisionTreeClassifier

    DCT = DecisionTreeClassifier()
    model_dt = DCT.fit(x_train_i, y_train_i)
    probs_dt = model_dt.predict_proba(x_test_i)[:, 1]
    probs_dt_scol5.append(probs_dt)
    dct_pred = DCT.predict(x_test_i)
    fdct = f1_score(dct_pred, y_test_i)
    f1_dt_scol5.append(fdct)
    rocauc_dt = roc_auc_score(y_test_i, dct_pred)
    rocauc_dt_scol5.append(rocauc_dt)
    recalls_dt = recall_score(y_test_i, dct_pred)
    recall_dt_scol5.append(recalls_dt)
    precisions_dt = precision_score(y_test_i, dct_pred)
    precision_dt_scol5.append(precisions_dt)
    accuracys_dt = accuracy_score(y_test_i, dct_pred)
    accuracy_dt_scol5.append(accuracys_dt)
    print("===DecisionTreeClassifier with TfidfVectorizer RUS Train_Set & Dev_Set - 2010-2016")
    dt_end = time.time()
    print('DCT F1-score', fdct * 100)
    print('DCT ROCAUC score:', rocauc_dt * 100)
    print('DCT Recall score:', recalls_dt * 100)
    print('DCT Precision Score:', precisions_dt * 100)
    print('DCT Confusion Matrix', confusion_matrix(y_test_i, dct_pred), "\n")
    print('DCT Classification', classification_report(y_test_i, dct_pred), "\n")
    print('DCT Accuracy Score', accuracys_dt * 100)
    print("Execution Time for Decision Tree RUS Train_Set & Dev_Set - 2010-2016", dt_end - start2, "seconds")

    from sklearn.naive_bayes import MultinomialNB

    start3 = time.time()
    Naive = MultinomialNB()
    model_nb = Naive.fit(x_train_i, y_train_i)
    probs_nb = model_nb.predict_proba(x_test_i)[:, 1]
    probs_nb_scol5.append(probs_nb)
    # predict the labels on validation dataset
    ny_pred = Naive.predict(x_test_i)
    fnb = f1_score(ny_pred, y_test_i)
    f1_nb_scol5.append(fnb)
    rocauc_nb = roc_auc_score(y_test_i, ny_pred)
    rocauc_nb_scol5.append(rocauc_nb)
    recalls_nb = recall_score(y_test_i, ny_pred)
    recall_nb_scol5.append(recalls_nb)
    precisions_nb = precision_score(y_test_i, ny_pred)
    precision_nb_scol5.append(precisions_nb)
    accuracys_nb = accuracy_score(y_test_i, ny_pred)
    accuracy_nb_scol5.append(accuracys_nb)
    nb_end = time.time()
    # Use accuracy_score function to get the accuracy
    print("===Naive Bayes with TfidfVectorizer Imabalanced RUS Train_Set & Dev_Set - 2010-2016")
    print('Naive F1-score', fnb * 100)
    print('Naive ROCAUC score:', rocauc_nb * 100)
    print('Naive Recall score:', recalls_nb * 100)
    print('Naive Precision Score:', precisions_nb * 100)
    print('Naive Confusion Matrix', confusion_matrix(y_test_i, ny_pred), "\n")
    print('Naive Classification', classification_report(y_test_i, ny_pred), "\n")
    print('Naive Accuracy Score', accuracys_nb * 100)
    print("Execution Time for Naive Bayes RUS Train_Set & Dev_Set - 2010-2016", nb_end - start3, "seconds")

    # XGBoost Classifier

    start4 = time.time()
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    xgb_model = XGBClassifier().fit(x_train_i, y_train_i)
    probs_xg = xgb_model.predict_proba(x_test_i)[:, 1]
    probs_xg_scol5.append(probs_xg)
    # predict
    xgb_y_predict = xgb_model.predict(x_test_i)
    fxg = f1_score(xgb_y_predict, y_test_i)
    f1_xg_scol5.append(fxg)
    rocauc_xg = roc_auc_score(xgb_y_predict, y_test_i)
    rocauc_xg_scol5.append(rocauc_xg)
    recall_xg = recall_score(xgb_y_predict, y_test_i)
    recall_xg_scol5.append(recall_xg)
    precisions_xg = precision_score(xgb_y_predict, y_test_i)
    precision_xg_scol5.append(precisions_xg)
    accuracys_xg = accuracy_score(xgb_y_predict, y_test_i)
    accuracy_xg_scol5.append(accuracys_xg)
    xg_end = time.time()
    print("===XGB with TfidfVectorizer RUS Train_Set & Dev_Set - 2010-2016")
    print('XGB F1-Score', fxg * 100)
    print('XGB ROCAUC Score:', rocauc_xg * 100)
    print('XGB Recall score:', recall_xg * 100)
    print('XGB Precision Score:', precisions_xg * 100)
    print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, y_test_i), "\n")
    print('XGB Classification', classification_report(xgb_y_predict, y_test_i), "\n")
    print('XGB Accuracy Score', accuracys_nb * 100)
    print("Execution Time for XGBoost Classifier RUS Train_Set & Dev_Set - 2010-2016", xg_end - start4, "seconds")

    # Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier

    start5 = time.time()
    rfc_model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(x_train_i, y_train_i)
    probs_rf = rfc_model.predict_proba(x_test_i)[:, 1]
    probs_rf_scol5.append(probs_rf)
    rfc_pred = rfc_model.predict(x_test_i)
    frfc = f1_score(rfc_pred, y_test_i)
    f1_rf_scol5.append(frfc)
    rocauc_rf = roc_auc_score(y_test_i, rfc_pred)
    rocauc_rf_scol5.append(rocauc_rf)
    recalls_rf = recall_score(rfc_pred, y_test_i)
    recall_rf_scol5.append(recalls_rf)
    precisions_rf = precision_score(rfc_pred, y_test_i)
    precision_rf_scol5.append(precisions_rf)
    accuracys_rf = accuracy_score(rfc_pred, y_test_i)
    accuracy_rf_scol5.append(accuracys_rf)
    rf_end = time.time()
    print("====RandomForest with Tfidf RUS Train_Set & Dev_Set - 2010-2016")
    print('RFC F1 score', frfc * 100)
    print('RFC ROCAUC Score:', rocauc_rf * 100)
    print('RFC Recall score:', recalls_rf * 100)
    print('RFC Precision Score:', precisions_rf * 100)
    print('RFC Confusion Matrix', confusion_matrix(y_test_i, rfc_pred), "\n")
    print('RFC Classification', classification_report(y_test_i, rfc_pred), "\n")
    print('RFC Accuracy Score', accuracys_rf * 100)
    print("Execution Time for Random Forest Classifier RUS Train_Set & Dev_Set - 2010-2016 ", rf_end - start5, "seconds")

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

subset5 = ['Train_Set' for t in range(50)]
other_set5 = ['Dev_Set' for t in range(50)]
sampling5 = ['Undersampling' for t in range(50)]
technique5 = ['RUS' for t in range(50)]
classifier_names5 = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
#train_sizes_num5 = [0.95]
# v5 = [0, 1, 2, 3, 4]
# precision_csv_num5 = [precision_lr_scol[z], precision_dt_scol[z], precision_nb_scol[z], precision_xg_scol[z], precision_rf_scol[z]]
# recall_csv_num5 = [recall_lr_scol[z], recall_dt_scol[z], recall_nb_scol[z], recall_xg_scol[z], recall_rf_scol[z]]
# auc_csv_num5 = [rocauc_lr_scol[z], rocauc_dt_scol[z], rocauc_nb_scol[z], rocauc_xg_scol[z], rocauc_rf_scol[z]]
# accuracy_csv_num5 = [accuracy_lr_scol[z], accuracy_dt_scol[z], accuracy_nb_scol[z], accuracy_xg_scol[z], accuracy_rf_scol[z]]
precision_csv_num5 = [precision_lr_scol5, precision_dt_scol5, precision_nb_scol5, precision_xg_scol5, precision_rf_scol5]
recall_csv_num5 = [recall_lr_scol5, recall_dt_scol5, recall_nb_scol5, recall_xg_scol5, recall_rf_scol5]
auc_csv_num5 = [rocauc_lr_scol5, rocauc_dt_scol5, rocauc_nb_scol5, rocauc_xg_scol5, rocauc_rf_scol5]
accuracy_csv_num5 = [accuracy_lr_scol5, accuracy_dt_scol5, accuracy_nb_scol5, accuracy_xg_scol5, accuracy_rf_scol5]
import itertools
rounds = 5
# rounds1 = 1
p5 = itertools.cycle(classifier_names5)
o5 = itertools.cycle(subsample_num)
#k5 = itertools.cycle(train_sizes_num)
# v5 = itertools.cycle(score_location)
# pr5 = itertools.cycle(precision_num)
# y5 = itertools.cycle(iteration_csv)
classifier_csv5 = [next(p5) for _ in range(rounds)] * 10
subsample_csv5 = [r for r in subsample_num for _ in range(rounds)]
#test_size_csv5 = [next(o) for _ in range(rounds)] * 1
#train_size_csv5 = [next(k) for _ in range(rounds)] * 1
# split_csv5 = ['1' for t in range(25)]
# train_csv5 = ['0.5' for t in range(25)]
precision_csv5 = list(chain(*precision_csv_num5))
recall_csv5 = list(chain(*recall_csv_num5))
auc_csv5 = list(chain(*auc_csv_num5))
accuracy_csv5 = list(chain(*accuracy_csv_num5))
csv_data5 = [subset5, other_set5, sampling5, technique5, classifier_csv5, subsample_csv5, precision_csv5, recall_csv5, auc_csv5, accuracy_csv5]
export_data5 = zip_longest(*csv_data5, fillvalue='')
filename = "Subset_Iterations.csv"
with open(filename, 'a', newline='') as file:
    write = csv.writer(file)
    write.writerows(export_data5)

#writef.close()

x_train10, y_train10 = tfidf_vect.fit_transform(sen10), tfidf_vect.transform(sentest10)
x_train11, y_train11 = tfidf_vect.fit_transform(sen20), tfidf_vect.transform(sentest20)
x_train12, y_train12 = tfidf_vect.fit_transform(sen30), tfidf_vect.transform(sentest30)
x_train13, y_train13 = tfidf_vect.fit_transform(sen40), tfidf_vect.transform(sentest40)
x_train14, y_train14 = tfidf_vect.fit_transform(sen50), tfidf_vect.transform(sentest50)
x_train15, y_train15 = tfidf_vect.fit_transform(sen60), tfidf_vect.transform(sentest60)
x_train16, y_train16 = tfidf_vect.fit_transform(sen70), tfidf_vect.transform(sentest70)
x_train17, y_train17 = tfidf_vect.fit_transform(sen80), tfidf_vect.transform(sentest80)
x_train18, y_train18 = tfidf_vect.fit_transform(sen90), tfidf_vect.transform(sentest90)
x_train19, y_train19 = tfidf_vect.fit_transform(sen100), tfidf_vect.transform(sentest100)
''''
x_train = tfidf_vect.fit_transform(sen10)
x_train1 = tfidf_vect.fit_transform(sen20)
x_train2 = tfidf_vect.fit_transform(sen30)
x_train3  tfidf_vect.fit_transform(sen40)
x_train4 = tfidf_vect.fit_transform(sen50)
x_train5 = tfidf_vect.fit_transform(sen60)
x_train6 = tfidf_vect.fit_transform(sen70)
x_train7 = tfidf_vect.fit_transform(sen80)
x_train8 = tfidf_vect.fit_transform(sen90)
x_train9 = tfidf_vect.fit_transform(sen100)

x_test = tfidf_vect.transform(senrem10)
x_test1 = tfidf_vect.transform(senrem20)
x_test2 = tfidf_vect.transform(senrem30)
x_test3 = tfidf_vect.transform(senrem40)
x_test4 = tfidf_vect.transform(senrem50)
x_test5 = tfidf_vect.transform(senrem60)
x_test6 = tfidf_vect.transform(senrem70)
x_test7 = tfidf_vect.transform(senrem80)
x_test8 = tfidf_vect.transform(senrem90)
x_test9 = tfidf_vect.transform(senrem100)
'''
x_test10 = train_split1["label"]
x_test11 = train_split2["label"]
x_test12 = train_split3["label"]
x_test13 = train_split4["label"]
x_test14 = train_split5["label"]
x_test15 = train_split6["label"]
x_test16 = train_split7["label"]
x_test17 = train_split8["label"]
x_test18 = train_split9["label"]
x_test19 = train_split10["label"]

y_test10 = test_split1["label"]
y_test11 = test_split2["label"]
y_test12 = test_split3["label"]
y_test13 = test_split4["label"]
y_test14 = test_split5["label"]
y_test15 = test_split6["label"]
y_test16 = test_split7["label"]
y_test17 = test_split8["label"]
y_test18 = test_split9["label"]
y_test19 = test_split10["label"]

sen_train1 = [x_train10, x_train11, x_train12, x_train13, x_train14, x_train15, x_train16, x_train17, x_train18, x_train19]
#sen_train = [x_train, x_train1, x_train2]
print(sen_train)
sen_test1 = [y_train10, y_train11, y_train12, y_train13, y_train14, y_train15, y_train16, y_train17, y_train18, y_train19]
#sen_test = [y_train, y_train1, y_train2]
print(sen_test)
train_label1 = [x_test10, x_test11, x_test12, x_test13, x_test14, x_test15, x_test16, x_test17, x_test18, x_test19]
#train_label = [x_test, x_test1, x_test2]
test_label1 = [y_test10, y_test11, y_test12, y_test13, y_test14, y_test15, y_test16, y_test17, y_test18, y_test19]

# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Imbalanced
probs_lr_scol6 = []
f1_lr_scol6 = []
rocauc_lr_scol6 = []
recall_lr_scol6 = []
precision_lr_scol6 = []
accuracy_lr_scol6 = []

probs_dt_scol6 = []
f1_dt_scol6 = []
rocauc_dt_scol6 = []
recall_dt_scol6 = []
precision_dt_scol6 = []
accuracy_dt_scol6 = []

probs_nb_scol6 = []
f1_nb_scol6 = []
rocauc_nb_scol6 = []
recall_nb_scol6 = []
precision_nb_scol6 = []
accuracy_nb_scol6 = []

probs_xg_scol6 = []
f1_xg_scol6 = []
rocauc_xg_scol6 = []
recall_xg_scol6 = []
precision_xg_scol6 = []
accuracy_xg_scol6 = []

probs_rf_scol6 = []
f1_rf_scol6 = []
rocauc_rf_scol6 = []
recall_rf_scol6 = []
precision_rf_scol6 = []
accuracy_rf_scol6 = []

#tfidf_vect = TfidfVectorizer()
#x_tfidf = tfidf_vect.fit_transform(df["sentence"].astype('U'))

#x_train, x_test, y_train, y_test = train_test_split(x_tfidf, df["label"], test_size=0.05, train_size=0.95)

for h, j, k, l, m in zip(sen_train1, sen_test1, train_label1, test_label1, subsample_num):

    start1 = time.time()
    log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
    model_lr = log.fit(h, k)
    print(model_lr)
    probs_lr = model_lr.predict_proba(j)[:, 1]
    print(probs_lr)
    probs_lr_scol6.append(probs_lr)
    ly_prediction = log.predict(j)
    fly = f1_score(ly_prediction, l)
    f1_lr_scol6.append(fly)
    rocauc_lr = roc_auc_score(l, ly_prediction)
    rocauc_lr_scol6.append(rocauc_lr)
    recalls_lr = recall_score(l, ly_prediction)
    recall_lr_scol6.append(recalls_lr)
    precisions_lr = precision_score(l, ly_prediction)
    precision_lr_scol6.append(precisions_lr)
    accuracys_lr = accuracy_score(l, ly_prediction)
    accuracy_lr_scol6.append(accuracys_lr)
    print("===Logistic Regression with TfidfVectorizer Imbalanced - Train_Set & Test_Set 2010-2016", m)
    lr_end = time.time()
    print('Logistic F1-score', fly * 100)
    print('Logistic ROCAUC score:', rocauc_lr * 100)
    print('Logistic Recall score:', recalls_lr * 100)
    print('Logistic Precision Score:', precisions_lr * 100)
    print('Logistic Confusion Matrix', confusion_matrix(l, ly_prediction), "\n")
    print('Logistic Classification', classification_report(l, ly_prediction), "\n")
    print('Logistic Accuracy Score', accuracys_lr * 100)
    print("Execution Time for Logistic Regression Imbalanced - Train_Set & Test_Set 2010-2016: ", lr_end - start1, "seconds")

    start2 = time.time()
    from sklearn.tree import DecisionTreeClassifier

    DCT = DecisionTreeClassifier()
    model_dt = DCT.fit(h, k)
    probs_dt = model_dt.predict_proba(j)[:, 1]
    probs_dt_scol6.append(probs_dt)
    dct_pred = DCT.predict(j)
    fdct = f1_score(dct_pred, l)
    f1_dt_scol6.append(fdct)
    rocauc_dt = roc_auc_score(l, dct_pred)
    rocauc_dt_scol6.append(rocauc_dt)
    recalls_dt = recall_score(l, dct_pred)
    recall_dt_scol6.append(recalls_dt)
    precisions_dt = precision_score(l, dct_pred)
    precision_dt_scol6.append(precisions_dt)
    accuracys_dt = accuracy_score(l, dct_pred)
    accuracy_dt_scol6.append(accuracys_dt)
    print("===DecisionTreeClassifier with TfidfVectorizer Imbalanced Train_Set & Test_Set - 2010-2016", m)
    dt_end = time.time()
    print('DCT F1-score', fdct * 100)
    print('DCT ROCAUC score:', rocauc_dt * 100)
    print('DCT Recall score:', recalls_dt * 100)
    print('DCT Precision Score:', precisions_dt * 100)
    print('DCT Confusion Matrix', confusion_matrix(l, dct_pred), "\n")
    print('DCT Classification', classification_report(l, dct_pred), "\n")
    print('DCT Accuracy Score', accuracys_dt * 100)
    print("Execution Time for Decision Tree Imbalanced Train_Set & Test_Set - 2010-2016: ", dt_end - start2, "seconds")

    from sklearn.naive_bayes import MultinomialNB

    start3 = time.time()
    Naive = MultinomialNB()
    model_nb = Naive.fit(h, k)
    probs_nb = model_nb.predict_proba(j)[:, 1]
    probs_nb_scol6.append(probs_nb)
    # predict the labels on validation dataset
    ny_pred = Naive.predict(j)
    fnb = f1_score(ny_pred, l)
    f1_nb_scol6.append(fnb)
    rocauc_nb = roc_auc_score(l, ny_pred)
    rocauc_nb_scol6.append(rocauc_nb)
    recalls_nb = recall_score(l, ny_pred)
    recall_nb_scol6.append(recalls_nb)
    precisions_nb = precision_score(l, ny_pred)
    precision_nb_scol6.append(precisions_nb)
    accuracys_nb = accuracy_score(l, ny_pred)
    accuracy_nb_scol6.append(accuracys_nb)
    nb_end = time.time()
    # Use accuracy_score function to get the accuracy
    print("===Naive Bayes with TfidfVectorizer Imbalanced Train_Set & Test_Set - 2010-2016")
    print('Naive F1-score', fnb * 100)
    print('Naive ROCAUC score:', rocauc_nb * 100)
    print('Naive Recall score:', recalls_nb * 100)
    print('Naive Precision Score:', precisions_nb * 100)
    print('Naive Confusion Matrix', confusion_matrix(l, ny_pred), "\n")
    print('Naive Classification', classification_report(l, ny_pred), "\n")
    print('Naive Accuracy Score', accuracys_nb * 100)
    print("Execution Time for Naive Bayes Imbalanced Train_Set & Test_Set - 2010-2016: ", nb_end - start3, "seconds")

    # XGBoost Classifier

    start4 = time.time()
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    xgb_model = XGBClassifier().fit(h, k)
    probs_xg = xgb_model.predict_proba(j)[:, 1]
    probs_xg_scol6.append(probs_xg)
    # predict
    xgb_y_predict = xgb_model.predict(j)
    fxg = f1_score(xgb_y_predict, l)
    f1_xg_scol6.append(fxg)
    rocauc_xg = roc_auc_score(xgb_y_predict, l)
    rocauc_xg_scol6.append(rocauc_xg)
    recall_xg = recall_score(xgb_y_predict, l)
    recall_xg_scol6.append(recall_xg)
    precisions_xg = precision_score(xgb_y_predict, l)
    precision_xg_scol6.append(precisions_xg)
    accuracys_xg = accuracy_score(xgb_y_predict, l)
    accuracy_xg_scol6.append(accuracys_xg)
    xg_end = time.time()
    print("===XGB with TfidfVectorizer Imbalanced Train_Set & Test_Set - 2010-2016")
    print('XGB F1-Score', fxg * 100)
    print('XGB ROCAUC Score:', rocauc_xg * 100)
    print('XGB Recall score:', recall_xg * 100)
    print('XGB Precision Score:', precisions_xg * 100)
    print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, l), "\n")
    print('XGB Classification', classification_report(xgb_y_predict, l), "\n")
    print('XGB Accuracy Score', accuracys_nb * 100)
    print("Execution Time for XGBoost Classifier Imbalanced Train_Set & Test_Set - 2010-2016:", xg_end - start4, "seconds")

    # Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier

    start5 = time.time()
    rfc_model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(h, k)
    probs_rf = rfc_model.predict_proba(j)[:, 0]
    probs_rf_scol6.append(probs_rf)
    rfc_pred = rfc_model.predict(j)
    frfc = f1_score(rfc_pred, l)
    f1_rf_scol6.append(frfc)
    rocauc_rf = roc_auc_score(l, rfc_pred)
    rocauc_rf_scol6.append(rocauc_rf)
    recalls_rf = recall_score(rfc_pred, l)
    recall_rf_scol6.append(recalls_rf)
    precisions_rf = precision_score(rfc_pred, l)
    precision_rf_scol6.append(precisions_rf)
    accuracys_rf = accuracy_score(rfc_pred, l)
    accuracy_rf_scol6.append(accuracys_rf)
    rf_end = time.time()
    print("====RandomForest with TfidfVectorizer Imbalanced Train_Set & Test_Set - 2010-2016")
    print('RFC F1 score', frfc * 100)
    print('RFC ROCAUC Score:', rocauc_rf * 100)
    print('RFC Recall score:', recalls_rf * 100)
    print('RFC Precision Score:', precisions_rf * 100)
    print('RFC Confusion Matrix', confusion_matrix(l, rfc_pred), "\n")
    print('RFC Classification', classification_report(l, rfc_pred), "\n")
    print('RFC Accuracy Score', accuracys_rf * 100)
    print("Execution Time for Random Forest Classifier Imbalanced Train_Set & Test_Set - 2010-2016", rf_end - start5, "seconds")

    print("Array of Prob Scores LR-Imb Test-Size", ":", probs_lr_scol6)
    print("Array of F1 Scores LR-Imb Test-Size:", ":", f1_lr_scol6)
    print("Array of ROCAUC Scores LR-Imb:", ":", rocauc_lr_scol6)
    print("Array of Recall Scores LR-Imb:", ":", recall_lr_scol6)
    print("Array of Precision Scores LR-Imb:", ":", precision_lr_scol6)
    print("Array of Accuracy Scores LR-Imb:", ":", accuracy_lr_scol6)

    print("Array of Prob Scores DT-Imb:", ":", probs_dt_scol6)
    print("Array of F1 Scores DT-Imb:", ":", f1_dt_scol6)
    print("Array of ROCAUC Scores DT-Imb:", ":", rocauc_dt_scol6)
    print("Array of Recall Scores DT-Imb:", ":", recall_dt_scol6)
    print("Array of Precision Scores DT-Imb:", ":", precision_dt_scol6)
    print("Array of Accuracy Scores DT-Imb:", ":", accuracy_dt_scol6)

    print("Array of Prob Scores NB-Imb:", ":", probs_nb_scol6)
    print("Array of F1 Scores NB-Imb:", ":", f1_nb_scol6)
    print("Array of ROCAUC Scores NB-Imb:", ":", rocauc_nb_scol6)
    print("Array of Recall Scores NB-Imb:", ":", recall_nb_scol6)
    print("Array of Precision Scores NB-Imb:", ":", precision_nb_scol6)
    print("Array of Accuracy Scores NB-Imb:", ":", accuracy_nb_scol6)

    print("Array of Prob Scores XG-Imb:", ":", probs_xg_scol6)
    print("Array of F1 Scores XG-Imb:", ":", f1_xg_scol6)
    print("Array of ROCAUC Scores XG-Imb:", ":", rocauc_xg_scol6)
    print("Array of Recall Scores XG-Imb:", ":", recall_xg_scol6)
    print("Array of Precision Scores XG-Imb:", ":", precision_xg_scol6)
    print("Array of Accuracy Scores XG-Imb:", ":", accuracy_xg_scol6)

    print("Array of Prob Scores RF-Imb:", ":", probs_rf_scol6)
    print("Array of F1 Scores RF-Imb:", ":", f1_rf_scol6)
    print("Array of ROCAUC Scores RF-Imb:", ":", rocauc_rf_scol6)
    print("Array of Recall Scores RF-Imb:", ":", recall_rf_scol6)
    print("Array of Precision Scores RF-Imb:", ":", precision_rf_scol6)
    print("Array of Accuracy Scores RF-Imb:", ":", accuracy_rf_scol6)

from itertools import chain

subset6 = ['Train_Set' for t in range(50)]
other_set6 = ['Test_Set' for t in range(50)]
sampling6 = ['Imbalanced' for t in range(50)]
technique6 = ['Imbalanced' for t in range(50)]
classifier_names6 = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
#train_sizes_num = [0.95]
# v = [0, 1, 2, 3, 4]
# precision_csv_num = [precision_lr_scol[z], precision_dt_scol[z], precision_nb_scol[z], precision_xg_scol[z], precision_rf_scol[z]]
# recall_csv_num = [recall_lr_scol[z], recall_dt_scol[z], recall_nb_scol[z], recall_xg_scol[z], recall_rf_scol[z]]
# auc_csv_num = [rocauc_lr_scol[z], rocauc_dt_scol[z], rocauc_nb_scol[z], rocauc_xg_scol[z], rocauc_rf_scol[z]]
# accuracy_csv_num = [accuracy_lr_scol[z], accuracy_dt_scol[z], accuracy_nb_scol[z], accuracy_xg_scol[z], accuracy_rf_scol[z]]
precision_csv_num6 = [precision_lr_scol6, precision_dt_scol6, precision_nb_scol6, precision_xg_scol6, precision_rf_scol6]
recall_csv_num6 = [recall_lr_scol6, recall_dt_scol6, recall_nb_scol6, recall_xg_scol6, recall_rf_scol6]
auc_csv_num6 = [rocauc_lr_scol6, rocauc_dt_scol6, rocauc_nb_scol6, rocauc_xg_scol6, rocauc_rf_scol6]
accuracy_csv_num6 = [accuracy_lr_scol6, accuracy_dt_scol6, accuracy_nb_scol6, accuracy_xg_scol6, accuracy_rf_scol6]
import itertools
rounds = 5
# rounds1 = 1
p6 = itertools.cycle(classifier_names6)
o6 = itertools.cycle(subsample_num)
#k = itertools.cycle(train_sizes_num)
# v = itertools.cycle(score_location)
# pr = itertools.cycle(precision_num)
# y = itertools.cycle(iteration_csv)
classifier_csv6 = [next(p6) for _ in range(rounds)] * 10
subsample_csv6 = [r for r in subsample_num for _ in range(rounds)]
#test_size_csv = [next(o) for _ in range(rounds)] * 1
#train_size_csv = [next(k) for _ in range(rounds)] * 1
# split_csv = ['1' for t in range(25)]
# train_csv = ['0.5' for t in range(25)]
precision_csv6 = list(chain(*precision_csv_num6))
recall_csv6 = list(chain(*recall_csv_num6))
auc_csv6 = list(chain(*auc_csv_num6))
accuracy_csv6 = list(chain(*accuracy_csv_num6))
csv_data6 = [subset6, other_set6, sampling6, technique6, classifier_csv6, subsample_csv6, precision_csv6, recall_csv6, auc_csv6, accuracy_csv6]
export_data6 = zip_longest(*csv_data6, fillvalue='')
filename = "Subset_Iterations.csv"
with open(filename, 'a', newline='') as file:
    write = csv.writer(file)
    write.writerows(export_data6)
    

'''
x_tl1a, y_tl1a = tl.fit_resample(x_train, x_test)
x_tl1b, y_tl1b = tl.fit_resample(y_train, y_test)
x_tl2a, y_tl2a = tl.fit_resample(x_train1, x_test1)
x_tl2b, y_tl2b = tl.fit_resample(y_train1, y_test1)
x_tl3a, y_tl3a = tl.fit_resample(x_train2, x_test2)
x_tl3b, y_tl3b = tl.fit_resample(y_train2, y_test2)
x_tl4a, y_tl4a = tl.fit_resample(x_train3, x_test3)
x_tl4b, y_tl4b = tl.fit_resample(y_train3, y_test3)
x_tl5a, y_tl5a = tl.fit_resample(x_train4, x_test4)
x_tl5b, y_tl5b = tl.fit_resample(y_train4, y_test4)
x_tl6a, y_tl6a = tl.fit_resample(x_train5, x_test5)
x_tl6b, y_tl6b = tl.fit_resample(y_train5, y_test5)
x_tl7a, y_tl7a = tl.fit_resample(x_train6, x_test6)
x_tl7b, y_tl7b = tl.fit_resample(y_train6, y_test6)
x_tl8a, y_tl8a = tl.fit_resample(x_train7, x_test7)
x_tl8b, y_tl8b = tl.fit_resample(y_train7, y_test7)
x_tl9a, y_tl9a = tl.fit_resample(x_train8, x_test8)
x_tl9b, y_tl9b = tl.fit_resample(y_train8, y_test8)
x_tl10a, y_tl10a = tl.fit_resample(x_train9, x_test9)
x_tl10b, y_tl10b = tl.fit_resample(y_train9, y_test9)


tl_xtrain_list = [x_tl1a, x_tl2a, x_tl3a, x_tl4a, x_tl5a, x_tl6a, x_tl7a, x_tl8a, x_tl9a, x_tl10a]
tl_xtest_list = [y_tl1a, y_tl2a, y_tl3a, y_tl4a, y_tl5a, y_tl6a, y_tl7a, y_tl8a, y_tl9a, y_tl10a]
tl_ytrain_list = [x_tl1b, x_tl2b, x_tl3b, x_tl4b, x_tl5b, x_tl6b, x_tl7b, x_tl8b, x_tl9b, x_tl10b]
tl_ytest_list = [y_tl1b, y_tl2b, y_tl3b, y_tl4b, y_tl5b, y_tl6b, y_tl7b, y_tl8b, y_tl9b, y_tl10b]
subsample_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]




sample_list = [x_tl11, x_tl12, x_tl13, x_tl14, x_tl15, x_tl16, x_tl17, x_tl18, x_tl19, x_tl20]
label_list = [y_tl11, y_tl12, y_tl13, y_tl14, y_tl15, y_tl16, y_tl17, y_tl18, y_tl19, y_tl20]
'''

x_tfidf11 = tfidf_vect.fit_transform(train_test_split1["sentence"].astype('U'))
x_tfidf12 = tfidf_vect.fit_transform(train_test_split2["sentence"].astype('U'))
x_tfidf13 = tfidf_vect.fit_transform(train_test_split3["sentence"].astype('U'))
x_tfidf14 = tfidf_vect.fit_transform(train_test_split4["sentence"].astype('U'))
x_tfidf15 = tfidf_vect.fit_transform(train_test_split5["sentence"].astype('U'))
x_tfidf16 = tfidf_vect.fit_transform(train_test_split6["sentence"].astype('U'))
x_tfidf17 = tfidf_vect.fit_transform(train_test_split7["sentence"].astype('U'))
x_tfidf18 = tfidf_vect.fit_transform(train_test_split8["sentence"].astype('U'))
x_tfidf19 = tfidf_vect.fit_transform(train_test_split9["sentence"].astype('U'))
x_tfidf20 = tfidf_vect.fit_transform(train_test_split10["sentence"].astype('U'))

subsample11 = x_tfidf11
print(subsample1.shape)
sublabel11 = train_test_split1["label"]
print(sublabel1.shape)
print(train_df['label'].value_counts())
label_count = train_df.groupby("label")
print("Label Count", label_count)
print(train_df.label.dtype)
print(train_df.info())
print("XTIDF TL:", type(subsample1))
print("LABEL TL:", type(sublabel1))
x_tl11, y_tl11 = tl.fit_resample(subsample11, sublabel11)
subsample12 = x_tfidf12
print(subsample12.shape)
sublabel12 = train_test_split2["label"]
print(sublabel12.shape)
x_tl12, y_tl12 = tl.fit_resample(subsample12, sublabel12)
subsample13 = x_tfidf13
print(subsample3.shape)
sublabel13 = train_test_split3["label"]
print(sublabel13.shape)
x_tl13, y_tl13 = tl.fit_resample(subsample13, sublabel13)
subsample14 = x_tfidf14
print(subsample14.shape)
sublabel14 = train_test_split4["label"]
print(sublabel14.shape)
x_tl14, y_tl14 = tl.fit_resample(subsample14, sublabel14)
subsample15 = x_tfidf15
print(subsample5.shape)
sublabel15 = train_test_split5["label"]
print(sublabel5.shape)
x_tl15, y_tl15 = tl.fit_resample(subsample15, sublabel15)
subsample16 = x_tfidf16
sublabel16 = train_test_split6["label"]
x_tl16, y_tl16 = tl.fit_resample(subsample16, sublabel16)
subsample17 = x_tfidf17
sublabel17 = train_test_split7["label"]
x_tl17, y_tl17 = tl.fit_resample(subsample17, sublabel17)
subsample18 = x_tfidf18
sublabel18 = train_test_split8["label"]
x_tl18, y_tl18 = tl.fit_resample(subsample18, sublabel18)
subsample19 = x_tfidf19
sublabel19 = train_test_split9["label"]
x_tl19, y_tl19 = tl.fit_resample(subsample19, sublabel19)
subsample20 = x_tfidf20
sublabel20 = train_test_split10["label"]
x_tl20, y_tl20 = tl.fit_resample(subsample20, sublabel20)


# Tomelinks
probs_lr_scol7 = []
f1_lr_scol7 = []
rocauc_lr_scol7 = []
recall_lr_scol7 = []
precision_lr_scol7 = []
accuracy_lr_scol7 = []

probs_dt_scol7 = []
f1_dt_scol7 = []
rocauc_dt_scol7 = []
recall_dt_scol7 = []
precision_dt_scol7 = []
accuracy_dt_scol7 = []

probs_nb_scol7 = []
f1_nb_scol7 = []
rocauc_nb_scol7 = []
recall_nb_scol7 = []
precision_nb_scol7 = []
accuracy_nb_scol7 = []

probs_xg_scol7 = []
f1_xg_scol7 = []
rocauc_xg_scol7 = []
recall_xg_scol7 = []
precision_xg_scol7 = []
accuracy_xg_scol7 = []

probs_rf_scol7 = []
f1_rf_scol7 = []
rocauc_rf_scol7 = []
recall_rf_scol7 = []
precision_rf_scol7 = []
accuracy_rf_scol7 = []


sample_list5 = [x_tl11, x_tl12, x_tl13, x_tl14, x_tl15, x_tl16, x_tl17, x_tl18, x_tl19, x_tl20]
label_list5 = [y_tl11, y_tl12, y_tl13, y_tl14, y_tl15, y_tl16, y_tl17, y_tl18, y_tl19, y_tl20]

#for i, j in zip(sample_list, label_list):

for i, j, z in zip(sample_list5, label_list5, subsample_num):
#for h, j, k, l in zip(tl_xtrain_list, tl_xtest_list, tl_ytrain_list, tl_ytest_list):
    x_train_i, x_test_i, y_train_i, y_test_i = train_test_split(i, j, train_size=0.5, test_size=0.5)

    start1 = time.time()
    log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
    model_lr = log.fit(x_train_i, y_train_i)
    probs_lr = model_lr.predict_proba(x_test_i)[:, 1]
    probs_lr_scol7.append(probs_lr)
    ly_prediction = log.predict(x_test_i)
    fly = f1_score(ly_prediction, y_test_i)
    f1_lr_scol7.append(fly)
    rocauc_lr = roc_auc_score(y_test_i, ly_prediction)
    rocauc_lr_scol7.append(rocauc_lr)
    recalls_lr = recall_score(y_test_i, ly_prediction)
    recall_lr_scol7.append(recalls_lr)
    precisions_lr = precision_score(y_test_i, ly_prediction)
    precision_lr_scol7.append(precisions_lr)
    accuracys_lr = accuracy_score(y_test_i, ly_prediction)
    accuracy_lr_scol7.append(accuracys_lr)
    print("===Logistic Regression with TfidfVectorizer Tomelinks Train_Set & Test_Set - 2010-2016", y_train_i)
    lr_end = time.time()
    print('Logistic F1-score', fly * 100)
    print('Logistic ROCAUC score:', rocauc_lr * 100)
    print('Logistic Recall score:', recalls_lr * 100)
    print('Logistic Precision Score:', precisions_lr * 100)
    print('Logistic Confusion Matrix', confusion_matrix(y_test_i, ly_prediction), "\n")
    print('Logistic Classification', classification_report(y_test_i, ly_prediction), "\n")
    print('Logistic Accuracy Score', accuracys_lr * 100)
    print("Execution Time for Logistic Regression Tomelinks Train_Set & Test_Set - 2010-2016", lr_end - start1, "seconds")

    start2 = time.time()
    from sklearn.tree import DecisionTreeClassifier

    DCT = DecisionTreeClassifier()
    model_dt = DCT.fit(x_train_i, y_train_i)
    probs_dt = model_dt.predict_proba(x_test_i)[:, 1]
    probs_dt_scol7.append(probs_dt)
    dct_pred = DCT.predict(x_test_i)
    fdct = f1_score(dct_pred, y_test_i)
    f1_dt_scol7.append(fdct)
    rocauc_dt = roc_auc_score(y_test_i, dct_pred)
    rocauc_dt_scol7.append(rocauc_dt)
    recalls_dt = recall_score(y_test_i, dct_pred)
    recall_dt_scol7.append(recalls_dt)
    precisions_dt = precision_score(y_test_i, dct_pred)
    precision_dt_scol7.append(precisions_dt)
    accuracys_dt = accuracy_score(y_test_i, dct_pred)
    accuracy_dt_scol7.append(accuracys_dt)
    print("===DecisionTreeClassifier with TfidfVectorizer Tomelinks Train_Set & Test_Set - 2010-2016", y_train_i)
    dt_end = time.time()
    print('DCT F1-score', fdct * 100)
    print('DCT ROCAUC score:', rocauc_dt * 100)
    print('DCT Recall score:', recalls_dt * 100)
    print('DCT Precision Score:', precisions_dt * 100)
    print('DCT Confusion Matrix', confusion_matrix(y_test_i, dct_pred), "\n")
    print('DCT Classification', classification_report(y_test_i, dct_pred), "\n")
    print('DCT Accuracy Score', accuracys_dt * 100)
    print("Execution Time for Decision Tree Tomelinks Train_Set & Test_Set - 2010-2016", dt_end - start2, "seconds")

    from sklearn.naive_bayes import MultinomialNB

    start3 = time.time()
    Naive = MultinomialNB()
    model_nb = Naive.fit(x_train_i, y_train_i)
    probs_nb = model_nb.predict_proba(x_test_i)[:, 1]
    probs_nb_scol7.append(probs_nb)
    # predict the labels on validation dataset
    ny_pred = Naive.predict(x_test_i)
    fnb = f1_score(ny_pred, y_test_i)
    f1_nb_scol7.append(fnb)
    rocauc_nb = roc_auc_score(y_test_i, ny_pred)
    rocauc_nb_scol7.append(rocauc_nb)
    recalls_nb = recall_score(y_test_i, ny_pred)
    recall_nb_scol7.append(recalls_nb)
    precisions_nb = precision_score(y_test_i, ny_pred)
    precision_nb_scol7.append(precisions_nb)
    accuracys_nb = accuracy_score(y_test_i, ny_pred)
    accuracy_nb_scol7.append(accuracys_nb)
    nb_end = time.time()
    # Use accuracy_score function to get the accuracy
    print("===Naive Bayes with TfidfVectorizer Imabalanced Tomelinks Train_Set & Test_Set - 2010-2016", y_train_i)
    print('Naive F1-score', fnb * 100)
    print('Naive ROCAUC score:', rocauc_nb * 100)
    print('Naive Recall score:', recalls_nb * 100)
    print('Naive Precision Score:', precisions_nb * 100)
    print('Naive Confusion Matrix', confusion_matrix(y_test_i, ny_pred), "\n")
    print('Naive Classification', classification_report(y_test_i, ny_pred), "\n")
    print('Naive Accuracy Score', accuracys_nb * 100)
    print("Execution Time for Naive Bayes Tomelinks Train_Set & Test_Set - 2010-2016", nb_end - start3, "seconds")

    # XGBoost Classifier

    start4 = time.time()
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    xgb_model = XGBClassifier().fit(x_train_i, y_train_i)
    probs_xg = xgb_model.predict_proba(x_test_i)[:, 1]
    probs_xg_scol7.append(probs_xg)
    # predict
    xgb_y_predict = xgb_model.predict(x_test_i)
    fxg = f1_score(xgb_y_predict, y_test_i)
    f1_xg_scol7.append(fxg)
    rocauc_xg = roc_auc_score(xgb_y_predict, y_test_i)
    rocauc_xg_scol7.append(rocauc_xg)
    recall_xg = recall_score(xgb_y_predict, y_test_i)
    recall_xg_scol7.append(recall_xg)
    precisions_xg = precision_score(xgb_y_predict, y_test_i)
    precision_xg_scol7.append(precisions_xg)
    accuracys_xg = accuracy_score(xgb_y_predict, y_test_i)
    accuracy_xg_scol7.append(accuracys_xg)
    xg_end = time.time()
    print("===XGB with TfidfVectorizer Tomelinks Train_Set & Test_Set - 2010-2016", y_train_i)
    print('XGB F1-Score', fxg * 100)
    print('XGB ROCAUC Score:', rocauc_xg * 100)
    print('XGB Recall score:', recall_xg * 100)
    print('XGB Precision Score:', precisions_xg * 100)
    print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, y_test_i), "\n")
    print('XGB Classification', classification_report(xgb_y_predict, y_test_i), "\n")
    print('XGB Accuracy Score', accuracys_nb * 100)
    print("Execution Time for XGBoost Classifier Tomelinks Train_Set & Test_Set - 2010-2016", xg_end - start4, "seconds")

    # Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier

    start5 = time.time()
    rfc_model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(x_train_i, y_train_i)
    probs_rf = rfc_model.predict_proba(x_test_i)[:, 1]
    probs_rf_scol7.append(probs_rf)
    rfc_pred = rfc_model.predict(x_test_i)
    frfc = f1_score(rfc_pred, y_test_i)
    f1_rf_scol7.append(frfc)
    rocauc_rf = roc_auc_score(y_test_i, rfc_pred)
    rocauc_rf_scol7.append(rocauc_rf)
    recalls_rf = recall_score(rfc_pred, y_test_i)
    recall_rf_scol7.append(recalls_rf)
    precisions_rf = precision_score(rfc_pred, y_test_i)
    precision_rf_scol7.append(precisions_rf)
    accuracys_rf = accuracy_score(rfc_pred, y_test_i)
    accuracy_rf_scol7.append(accuracys_rf)
    rf_end = time.time()
    print("====RandomForest with Tfidf Tomelinks Train_Set & Test_Set - 2010-2016", y_train_i)
    print('RFC F1 score', frfc * 100)
    print('RFC ROCAUC Score:', rocauc_rf * 100)
    print('RFC Recall score:', recalls_rf * 100)
    print('RFC Precision Score:', precisions_rf * 100)
    print('RFC Confusion Matrix', confusion_matrix(y_test_i, rfc_pred), "\n")
    print('RFC Classification', classification_report(y_test_i, rfc_pred), "\n")
    print('RFC Accuracy Score', accuracys_rf * 100)
    print("Execution Time for Random Forest Classifier Tomelinks Train_Set & Test_Set - 2010-2016 ", rf_end - start5, "seconds")

    print("Array of Prob Scores LR-Sam Tomelinks", ":", probs_lr_scol7)
    print("Array of F1 Scores LR-Sam Tomelinks", ":", f1_lr_scol7)
    print("Array of ROCAUC Scores LR-Sam Tomelinks", ":", rocauc_lr_scol7)
    print("Array of Recall Scores LR-Sam Tomelinks", ":", recall_lr_scol7)
    print("Array of Precision Scores LR-Sam Tomelinks", ":", precision_lr_scol7)
    print("Array of Accuracy Scores LR-Sam Tomelinks", ":", accuracy_lr_scol7)

    print("Array of Prob Scores DT-Sam Tomelinks", ":", probs_dt_scol7)
    print("Array of F1 Scores DT-Sam Tomelinks", ":", f1_dt_scol7)
    print("Array of ROCAUC Scores DT-Sam Tomelinks", ":", rocauc_dt_scol7)
    print("Array of Recall Scores DT-Sam Tomelinks", ":", recall_dt_scol7)
    print("Array of Precision Scores DT-Sam Tomelinks", ":", precision_dt_scol7)
    print("Array of Accuracy Scores DT-Sam Tomelinks", ":", accuracy_dt_scol7)

    print("Array of Prob Scores NB-Sam Tomelinks", ":", probs_nb_scol7)
    print("Array of F1 Scores NB-Sam Tomelinks", ":", f1_nb_scol7)
    print("Array of ROCAUC Scores NB-Sam Tomelinks", ":", rocauc_nb_scol7)
    print("Array of Recall Scores NB-Sam Tomelinks", ":", recall_nb_scol7)
    print("Array of Precision Scores NB-Sam Tomelinks", ":", precision_nb_scol7)
    print("Array of Accuracy Scores NB-Sam Tomelinks", ":", accuracy_nb_scol7)

    print("Array of Prob Scores XG-Sam Tomelinks", ":", probs_xg_scol7)
    print("Array of F1 Scores XG-Sam Tomelinks", ":", f1_xg_scol7)
    print("Array of ROCAUC Scores XG-Sam Tomelinks", ":", rocauc_xg_scol7)
    print("Array of Recall Scores XG-Sam Tomelinks", ":", recall_xg_scol7)
    print("Array of Precision Scores XG-Sam Tomelinks", ":", precision_xg_scol7)
    print("Array of Accuracy Scores XG-Sam Tomelinks", ":", accuracy_xg_scol7)

    print("Array of Prob Scores RF-Sam Tomelinks", ":", probs_rf_scol7)
    print("Array of F1 Scores RF-Sam Tomelinks", ":", f1_rf_scol7)
    print("Array of ROCAUC Scores RF-Sam Tomelinks", ":", rocauc_rf_scol7)
    print("Array of Recall Scores RF-Sam Tomelinks", ":", recall_rf_scol7)
    print("Array of Precision Scores RF-Sam Tomelinks", ":", precision_rf_scol7)
    print("Array of Accuracy Scores RF-Sam Tomelinks", ":", accuracy_rf_scol7)

from itertools import chain

subset7 = ['Train_Set' for t in range(50)]
other_set7 = ['Test_Set' for t in range(50)]
sampling7 = ['Undersampling' for t in range(50)]
technique7 = ['Tomelinks' for t in range(50)]
classifier_names7 = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
#train_sizes_num7 = [0.95]
# v7 = [0, 1, 2, 3, 4]
# precision_csv_num7 = [precision_lr_scol[z], precision_dt_scol[z], precision_nb_scol[z], precision_xg_scol[z], precision_rf_scol[z]]
# recall_csv_num7 = [recall_lr_scol[z], recall_dt_scol[z], recall_nb_scol[z], recall_xg_scol[z], recall_rf_scol[z]]
# auc_csv_num7 = [rocauc_lr_scol[z], rocauc_dt_scol[z], rocauc_nb_scol[z], rocauc_xg_scol[z], rocauc_rf_scol[z]]
# accuracy_csv_num7 = [accuracy_lr_scol[z], accuracy_dt_scol[z], accuracy_nb_scol[z], accuracy_xg_scol[z], accuracy_rf_scol[z]]
precision_csv_num7 = [precision_lr_scol7, precision_dt_scol7, precision_nb_scol7, precision_xg_scol7, precision_rf_scol7]
recall_csv_num7 = [recall_lr_scol7, recall_dt_scol7, recall_nb_scol7, recall_xg_scol7, recall_rf_scol7]
auc_csv_num7 = [rocauc_lr_scol7, rocauc_dt_scol7, rocauc_nb_scol7, rocauc_xg_scol7, rocauc_rf_scol7]
accuracy_csv_num7 = [accuracy_lr_scol7, accuracy_dt_scol7, accuracy_nb_scol7, accuracy_xg_scol7, accuracy_rf_scol7]
import itertools
rounds = 5
# rounds1 = 1
p7 = itertools.cycle(classifier_names7)
o7 = itertools.cycle(subsample_num)
#k7 = itertools.cycle(train_sizes_num)
# v7 = itertools.cycle(score_location)
# pr7 = itertools.cycle(precision_num)
# y7 = itertools.cycle(iteration_csv)
classifier_csv7 = [next(p7) for _ in range(rounds)] * 10
subsample_csv7 = [r for r in subsample_num for _ in range(rounds)]
#test_size_csv7 = [next(o) for _ in range(rounds)] * 1
#train_size_csv7 = [next(k) for _ in range(rounds)] * 1
# split_csv7 = ['1' for t in range(25)]
# train_csv7 = ['0.5' for t in range(25)]
precision_csv7 = list(chain(*precision_csv_num))
recall_csv7 = list(chain(*recall_csv_num))
auc_csv7 = list(chain(*auc_csv_num))
accuracy_csv7 = list(chain(*accuracy_csv_num))
csv_data7 = [subset7, other_set7, sampling7, technique7, classifier_csv7, subsample_csv7, precision_csv7, recall_csv7, auc_csv7, accuracy_csv7]
export_data7 = zip_longest(*csv_data7, fillvalue='')
filename = "Subset_Iterations.csv"
with open(filename, 'a', newline='') as file:
    write = csv.writer(file)
    write.writerows(export_data7)
    
# NearMiss
probs_lr_scol8 = []
f1_lr_scol8 = []
rocauc_lr_scol8 = []
recall_lr_scol8 = []
precision_lr_scol8 = []
accuracy_lr_scol8 = []

probs_dt_scol8 = []
f1_dt_scol8 = []
rocauc_dt_scol8 = []
recall_dt_scol8 = []
precision_dt_scol8 = []
accuracy_dt_scol8 = []

probs_nb_scol8 = []
f1_nb_scol8 = []
rocauc_nb_scol8 = []
recall_nb_scol8 = []
precision_nb_scol8 = []
accuracy_nb_scol8 = []

probs_xg_scol8 = []
f1_xg_scol8 = []
rocauc_xg_scol8 = []
recall_xg_scol8 = []
precision_xg_scol8 = []
accuracy_xg_scol8 = []

probs_rf_scol8 = []
f1_rf_scol8 = []
rocauc_rf_scol8 = []
recall_rf_scol8 = []
precision_rf_scol8 = []
accuracy_rf_scol8 = []

x_nm11, y_nm11 = nm.fit_resample(subsample11, sublabel11)
x_nm12, y_nm12 = nm.fit_resample(subsample12, sublabel12)
x_nm13, y_nm13 = nm.fit_resample(subsample13, sublabel13)
x_nm14, y_nm14 = nm.fit_resample(subsample14, sublabel14)
x_nm15, y_nm15 = nm.fit_resample(subsample15, sublabel15)
x_nm16, y_nm16 = nm.fit_resample(subsample16, sublabel16)
x_nm17, y_nm17 = nm.fit_resample(subsample17, sublabel17)
x_nm18, y_nm18 = nm.fit_resample(subsample18, sublabel18)
x_nm19, y_nm19 = nm.fit_resample(subsample19, sublabel19)
x_nm20, y_nm20 = nm.fit_resample(subsample20, sublabel20)

sample_list6 = [x_nm11, x_nm12, x_nm13, x_nm14, x_nm15, x_nm16, x_nm17, x_nm18, x_nm19, x_nm20]
label_list6 = [y_nm11, y_nm12, y_nm13, y_nm14, y_nm15, y_nm16, y_nm17, y_nm18, y_nm19, y_nm20]

#for i, j in zip(sample_list, label_list):

for i, j, z in zip(sample_list6, label_list6, subsample_num):
#for h, j, k, l in zip(tl_xtrain_list, tl_xtest_list, tl_ytrain_list, tl_ytest_list):
    x_train_i, x_test_i, y_train_i, y_test_i = train_test_split(i, j, train_size=0.5, test_size=0.5)

    start1 = time.time()
    log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
    model_lr = log.fit(x_train_i, y_train_i)
    probs_lr = model_lr.predict_proba(x_test_i)[:, 1]
    probs_lr_scol8.append(probs_lr)
    ly_prediction = log.predict(x_test_i)
    fly = f1_score(ly_prediction, y_test_i)
    f1_lr_scol8.append(fly)
    rocauc_lr = roc_auc_score(y_test_i, ly_prediction)
    rocauc_lr_scol8.append(rocauc_lr)
    recalls_lr = recall_score(y_test_i, ly_prediction)
    recall_lr_scol8.append(recalls_lr)
    precisions_lr = precision_score(y_test_i, ly_prediction)
    precision_lr_scol8.append(precisions_lr)
    accuracys_lr = accuracy_score(y_test_i, ly_prediction)
    accuracy_lr_scol8.append(accuracys_lr)
    print("===Logistic Regression with TfidfVectorizer NearMiss Train_Set & Test_Set - 2010-2016", y_train_i)
    lr_end = time.time()
    print('Logistic F1-score', fly * 100)
    print('Logistic ROCAUC score:', rocauc_lr * 100)
    print('Logistic Recall score:', recalls_lr * 100)
    print('Logistic Precision Score:', precisions_lr * 100)
    print('Logistic Confusion Matrix', confusion_matrix(y_test_i, ly_prediction), "\n")
    print('Logistic Classification', classification_report(y_test_i, ly_prediction), "\n")
    print('Logistic Accuracy Score', accuracys_lr * 100)
    print("Execution Time for Logistic Regression NearMiss Train_Set & Test_Set - 2010-2016", lr_end - start1, "seconds")

    start2 = time.time()
    from sklearn.tree import DecisionTreeClassifier

    DCT = DecisionTreeClassifier()
    model_dt = DCT.fit(x_train_i, y_train_i)
    probs_dt = model_dt.predict_proba(x_test_i)[:, 1]
    probs_dt_scol8.append(probs_dt)
    dct_pred = DCT.predict(x_test_i)
    fdct = f1_score(dct_pred, y_test_i)
    f1_dt_scol8.append(fdct)
    rocauc_dt = roc_auc_score(y_test_i, dct_pred)
    rocauc_dt_scol8.append(rocauc_dt)
    recalls_dt = recall_score(y_test_i, dct_pred)
    recall_dt_scol8.append(recalls_dt)
    precisions_dt = precision_score(y_test_i, dct_pred)
    precision_dt_scol8.append(precisions_dt)
    accuracys_dt = accuracy_score(y_test_i, dct_pred)
    accuracy_dt_scol8.append(accuracys_dt)
    print("===DecisionTreeClassifier with TfidfVectorizer NearMiss Train_Set & Test_Set - 2010-2016", y_train_i)
    dt_end = time.time()
    print('DCT F1-score', fdct * 100)
    print('DCT ROCAUC score:', rocauc_dt * 100)
    print('DCT Recall score:', recalls_dt * 100)
    print('DCT Precision Score:', precisions_dt * 100)
    print('DCT Confusion Matrix', confusion_matrix(y_test_i, dct_pred), "\n")
    print('DCT Classification', classification_report(y_test_i, dct_pred), "\n")
    print('DCT Accuracy Score', accuracys_dt * 100)
    print("Execution Time for Decision Tree NearMiss Train_Set & Test_Set - 2010-2016", dt_end - start2, "seconds")

    from sklearn.naive_bayes import MultinomialNB

    start3 = time.time()
    Naive = MultinomialNB()
    model_nb = Naive.fit(x_train_i, y_train_i)
    probs_nb = model_nb.predict_proba(x_test_i)[:, 1]
    probs_nb_scol8.append(probs_nb)
    # predict the labels on validation dataset
    ny_pred = Naive.predict(x_test_i)
    fnb = f1_score(ny_pred, y_test_i)
    f1_nb_scol8.append(fnb)
    rocauc_nb = roc_auc_score(y_test_i, ny_pred)
    rocauc_nb_scol8.append(rocauc_nb)
    recalls_nb = recall_score(y_test_i, ny_pred)
    recall_nb_scol8.append(recalls_nb)
    precisions_nb = precision_score(y_test_i, ny_pred)
    precision_nb_scol8.append(precisions_nb)
    accuracys_nb = accuracy_score(y_test_i, ny_pred)
    accuracy_nb_scol8.append(accuracys_nb)
    nb_end = time.time()
    # Use accuracy_score function to get the accuracy
    print("===Naive Bayes with TfidfVectorizer Imabalanced NearMiss Train_Set & Test_Set - 2010-2016", y_train_i)
    print('Naive F1-score', fnb * 100)
    print('Naive ROCAUC score:', rocauc_nb * 100)
    print('Naive Recall score:', recalls_nb * 100)
    print('Naive Precision Score:', precisions_nb * 100)
    print('Naive Confusion Matrix', confusion_matrix(y_test_i, ny_pred), "\n")
    print('Naive Classification', classification_report(y_test_i, ny_pred), "\n")
    print('Naive Accuracy Score', accuracys_nb * 100)
    print("Execution Time for Naive Bayes NearMiss Train_Set & Test_Set - 2010-2016", nb_end - start3, "seconds")

    # XGBoost Classifier

    start4 = time.time()
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    xgb_model = XGBClassifier().fit(x_train_i, y_train_i)
    probs_xg = xgb_model.predict_proba(x_test_i)[:, 1]
    probs_xg_scol8.append(probs_xg)
    # predict
    xgb_y_predict = xgb_model.predict(x_test_i)
    fxg = f1_score(xgb_y_predict, y_test_i)
    f1_xg_scol8.append(fxg)
    rocauc_xg = roc_auc_score(xgb_y_predict, y_test_i)
    rocauc_xg_scol8.append(rocauc_xg)
    recall_xg = recall_score(xgb_y_predict, y_test_i)
    recall_xg_scol8.append(recall_xg)
    precisions_xg = precision_score(xgb_y_predict, y_test_i)
    precision_xg_scol8.append(precisions_xg)
    accuracys_xg = accuracy_score(xgb_y_predict, y_test_i)
    accuracy_xg_scol8.append(accuracys_xg)
    xg_end = time.time()
    print("===XGB with TfidfVectorizer NearMiss Train_Set & Test_Set - 2010-2016", y_train_i)
    print('XGB F1-Score', fxg * 100)
    print('XGB ROCAUC Score:', rocauc_xg * 100)
    print('XGB Recall score:', recall_xg * 100)
    print('XGB Precision Score:', precisions_xg * 100)
    print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, y_test_i), "\n")
    print('XGB Classification', classification_report(xgb_y_predict, y_test_i), "\n")
    print('XGB Accuracy Score', accuracys_nb * 100)
    print("Execution Time for XGBoost Classifier NearMiss Train_Set & Test_Set - 2010-2016", xg_end - start4, "seconds")

    # Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier

    start5 = time.time()
    rfc_model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(x_train_i, y_train_i)
    probs_rf = rfc_model.predict_proba(x_test_i)[:, 1]
    probs_rf_scol8.append(probs_rf)
    rfc_pred = rfc_model.predict(x_test_i)
    frfc = f1_score(rfc_pred, y_test_i)
    f1_rf_scol8.append(frfc)
    rocauc_rf = roc_auc_score(y_test_i, rfc_pred)
    rocauc_rf_scol8.append(rocauc_rf)
    recalls_rf = recall_score(rfc_pred, y_test_i)
    recall_rf_scol8.append(recalls_rf)
    precisions_rf = precision_score(rfc_pred, y_test_i)
    precision_rf_scol8.append(precisions_rf)
    accuracys_rf = accuracy_score(rfc_pred, y_test_i)
    accuracy_rf_scol8.append(accuracys_rf)
    rf_end = time.time()
    print("====RandomForest with Tfidf NearMiss Train_Set & Test_Set - 2010-2016", y_train_i)
    print('RFC F1 score', frfc * 100)
    print('RFC ROCAUC Score:', rocauc_rf * 100)
    print('RFC Recall score:', recalls_rf * 100)
    print('RFC Precision Score:', precisions_rf * 100)
    print('RFC Confusion Matrix', confusion_matrix(y_test_i, rfc_pred), "\n")
    print('RFC Classification', classification_report(y_test_i, rfc_pred), "\n")
    print('RFC Accuracy Score', accuracys_rf * 100)
    print("Execution Time for Random Forest Classifier NearMiss Train_Set & Test_Set - 2010-2016 ", rf_end - start5, "seconds")

    print("Array of Prob Scores LR-Sam NearMiss", ":", probs_lr_scol8)
    print("Array of F1 Scores LR-Sam NearMiss", ":", f1_lr_scol8)
    print("Array of ROCAUC Scores LR-Sam NearMiss", ":", rocauc_lr_scol8)
    print("Array of Recall Scores LR-Sam NearMiss", ":", recall_lr_scol8)
    print("Array of Precision Scores LR-Sam NearMiss", ":", precision_lr_scol8)
    print("Array of Accuracy Scores LR-Sam NearMiss", ":", accuracy_lr_scol8)

    print("Array of Prob Scores DT-Sam NearMiss", ":", probs_dt_scol8)
    print("Array of F1 Scores DT-Sam NearMiss", ":", f1_dt_scol8)
    print("Array of ROCAUC Scores DT-Sam NearMiss", ":", rocauc_dt_scol8)
    print("Array of Recall Scores DT-Sam NearMiss", ":", recall_dt_scol8)
    print("Array of Precision Scores DT-Sam NearMiss", ":", precision_dt_scol8)
    print("Array of Accuracy Scores DT-Sam NearMiss", ":", accuracy_dt_scol8)

    print("Array of Prob Scores NB-Sam NearMiss", ":", probs_nb_scol8)
    print("Array of F1 Scores NB-Sam NearMiss", ":", f1_nb_scol8)
    print("Array of ROCAUC Scores NB-Sam NearMiss", ":", rocauc_nb_scol8)
    print("Array of Recall Scores NB-Sam NearMiss", ":", recall_nb_scol8)
    print("Array of Precision Scores NB-Sam NearMiss", ":", precision_nb_scol8)
    print("Array of Accuracy Scores NB-Sam NearMiss", ":", accuracy_nb_scol8)

    print("Array of Prob Scores XG-Sam NearMiss", ":", probs_xg_scol8)
    print("Array of F1 Scores XG-Sam NearMiss", ":", f1_xg_scol8)
    print("Array of ROCAUC Scores XG-Sam NearMiss", ":", rocauc_xg_scol8)
    print("Array of Recall Scores XG-Sam NearMiss", ":", recall_xg_scol8)
    print("Array of Precision Scores XG-Sam NearMiss", ":", precision_xg_scol8)
    print("Array of Accuracy Scores XG-Sam NearMiss", ":", accuracy_xg_scol8)

    print("Array of Prob Scores RF-Sam NearMiss", ":", probs_rf_scol8)
    print("Array of F1 Scores RF-Sam NearMiss", ":", f1_rf_scol8)
    print("Array of ROCAUC Scores RF-Sam NearMiss", ":", rocauc_rf_scol8)
    print("Array of Recall Scores RF-Sam NearMiss", ":", recall_rf_scol8)
    print("Array of Precision Scores RF-Sam NearMiss", ":", precision_rf_scol8)
    print("Array of Accuracy Scores RF-Sam NearMiss", ":", accuracy_rf_scol8)

from itertools import chain

subset8 = ['Train_Set' for t in range(50)]
other_set8 = ['Test_Set' for t in range(50)]
sampling8 = ['Undersampling' for t in range(50)]
technique8 = ['NearMiss' for t in range(50)]
classifier_names8 = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
#train_sizes_num8 = [0.95]
# v8 = [0, 1, 2, 3, 4]
# precision_csv_num8 = [precision_lr_scol[z], precision_dt_scol[z], precision_nb_scol[z], precision_xg_scol[z], precision_rf_scol[z]]
# recall_csv_num8 = [recall_lr_scol[z], recall_dt_scol[z], recall_nb_scol[z], recall_xg_scol[z], recall_rf_scol[z]]
# auc_csv_num8 = [rocauc_lr_scol[z], rocauc_dt_scol[z], rocauc_nb_scol[z], rocauc_xg_scol[z], rocauc_rf_scol[z]]
# accuracy_csv_num8 = [accuracy_lr_scol[z], accuracy_dt_scol[z], accuracy_nb_scol[z], accuracy_xg_scol[z], accuracy_rf_scol[z]]
precision_csv_num8 = [precision_lr_scol8, precision_dt_scol8, precision_nb_scol8, precision_xg_scol8, precision_rf_scol8]
recall_csv_num8 = [recall_lr_scol8, recall_dt_scol8, recall_nb_scol8, recall_xg_scol8, recall_rf_scol8]
auc_csv_num8 = [rocauc_lr_scol8, rocauc_dt_scol8, rocauc_nb_scol8, rocauc_xg_scol8, rocauc_rf_scol8]
accuracy_csv_num8 = [accuracy_lr_scol8, accuracy_dt_scol8, accuracy_nb_scol8, accuracy_xg_scol8, accuracy_rf_scol8]
import itertools
rounds = 5
# rounds1 = 1
p8 = itertools.cycle(classifier_names8)
o8 = itertools.cycle(subsample_num)
#k8 = itertools.cycle(train_sizes_num)
# v8 = itertools.cycle(score_location)
# pr8 = itertools.cycle(precision_num)
# y8 = itertools.cycle(iteration_csv)
classifier_csv8 = [next(p8) for _ in range(rounds)] * 10
subsample_csv8 = [r for r in subsample_num for _ in range(rounds)]
#test_size_csv8 = [next(o) for _ in range(rounds)] * 1
#train_size_csv8 = [next(k) for _ in range(rounds)] * 1
# split_csv8 = ['1' for t in range(25)]
# train_csv8 = ['0.5' for t in range(25)]
precision_csv8 = list(chain(*precision_csv_num8))
recall_csv8 = list(chain(*recall_csv_num8))
auc_csv8 = list(chain(*auc_csv_num8))
accuracy_csv8 = list(chain(*accuracy_csv_num8))
csv_data8 = [subset8, other_set8, sampling8, technique8, classifier_csv8, subsample_csv8, precision_csv8, recall_csv8, auc_csv8, accuracy_csv8]
export_data8 = zip_longest(*csv_data8, fillvalue='')
filename = "Subset_Iterations.csv"
with open(filename, 'a', newline='') as file:
    write = csv.writer(file)
    write.writerows(export_data8)
    
# SMOTE
probs_lr_scol9 = []
f1_lr_scol9 = []
rocauc_lr_scol9 = []
recall_lr_scol9 = []
precision_lr_scol9 = []
accuracy_lr_scol9 = []

probs_dt_scol9 = []
f1_dt_scol9 = []
rocauc_dt_scol9 = []
recall_dt_scol9 = []
precision_dt_scol9 = []
accuracy_dt_scol9 = []

probs_nb_scol9 = []
f1_nb_scol9 = []
rocauc_nb_scol9 = []
recall_nb_scol9 = []
precision_nb_scol9 = []
accuracy_nb_scol9 = []

probs_xg_scol9 = []
f1_xg_scol9 = []
rocauc_xg_scol9 = []
recall_xg_scol9 = []
precision_xg_scol9 = []
accuracy_xg_scol9 = []

probs_rf_scol9 = []
f1_rf_scol9 = []
rocauc_rf_scol9 = []
recall_rf_scol9 = []
precision_rf_scol9 = []
accuracy_rf_scol9 = []

x_smote11, y_smote11 = smote.fit_resample(subsample11, sublabel11)
x_smote12, y_smote12 = smote.fit_resample(subsample12, sublabel12)
x_smote13, y_smote13 = smote.fit_resample(subsample13, sublabel13)
x_smote14, y_smote14 = smote.fit_resample(subsample14, sublabel14)
x_smote15, y_smote15 = smote.fit_resample(subsample15, sublabel15)
x_smote16, y_smote16 = smote.fit_resample(subsample16, sublabel16)
x_smote17, y_smote17 = smote.fit_resample(subsample17, sublabel17)
x_smote18, y_smote18 = smote.fit_resample(subsample18, sublabel18)
x_smote19, y_smote19 = smote.fit_resample(subsample19, sublabel19)
x_smote20, y_smote20 = smote.fit_resample(subsample20, sublabel20)

sample_list7 = [x_smote11, x_smote12, x_smote13, x_smote14, x_smote15, x_smote16, x_smote17, x_smote18, x_smote19, x_smote20]
label_list7 = [y_smote11, y_smote12, y_smote13, y_smote14, y_smote15, y_smote16, y_smote17, y_smote18, y_smote19, y_smote20]

#for i, j in zip(sample_list, label_list):

for i, j, z in zip(sample_list7, label_list7, subsample_num):
#for h, j, k, l in zip(tl_xtrain_list, tl_xtest_list, tl_ytrain_list, tl_ytest_list):
    x_train_i, x_test_i, y_train_i, y_test_i = train_test_split(i, j, train_size=0.5, test_size=0.5)

    start1 = time.time()
    log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
    model_lr = log.fit(x_train_i, y_train_i)
    probs_lr = model_lr.predict_proba(x_test_i)[:, 1]
    probs_lr_scol9.append(probs_lr)
    ly_prediction = log.predict(x_test_i)
    fly = f1_score(ly_prediction, y_test_i)
    f1_lr_scol9.append(fly)
    rocauc_lr = roc_auc_score(y_test_i, ly_prediction)
    rocauc_lr_scol9.append(rocauc_lr)
    recalls_lr = recall_score(y_test_i, ly_prediction)
    recall_lr_scol9.append(recalls_lr)
    precisions_lr = precision_score(y_test_i, ly_prediction)
    precision_lr_scol9.append(precisions_lr)
    accuracys_lr = accuracy_score(y_test_i, ly_prediction)
    accuracy_lr_scol9.append(accuracys_lr)
    print("===Logistic Regression with TfidfVectorizer SMOTE Train_Set & Test_Set - 2010-2016", y_train_i)
    lr_end = time.time()
    print('Logistic F1-score', fly * 100)
    print('Logistic ROCAUC score:', rocauc_lr * 100)
    print('Logistic Recall score:', recalls_lr * 100)
    print('Logistic Precision Score:', precisions_lr * 100)
    print('Logistic Confusion Matrix', confusion_matrix(y_test_i, ly_prediction), "\n")
    print('Logistic Classification', classification_report(y_test_i, ly_prediction), "\n")
    print('Logistic Accuracy Score', accuracys_lr * 100)
    print("Execution Time for Logistic Regression SMOTE Train_Set & Test_Set - 2010-2016", lr_end - start1, "seconds")

    start2 = time.time()
    from sklearn.tree import DecisionTreeClassifier

    DCT = DecisionTreeClassifier()
    model_dt = DCT.fit(x_train_i, y_train_i)
    probs_dt = model_dt.predict_proba(x_test_i)[:, 1]
    probs_dt_scol9.append(probs_dt)
    dct_pred = DCT.predict(x_test_i)
    fdct = f1_score(dct_pred, y_test_i)
    f1_dt_scol9.append(fdct)
    rocauc_dt = roc_auc_score(y_test_i, dct_pred)
    rocauc_dt_scol9.append(rocauc_dt)
    recalls_dt = recall_score(y_test_i, dct_pred)
    recall_dt_scol9.append(recalls_dt)
    precisions_dt = precision_score(y_test_i, dct_pred)
    precision_dt_scol9.append(precisions_dt)
    accuracys_dt = accuracy_score(y_test_i, dct_pred)
    accuracy_dt_scol9.append(accuracys_dt)
    print("===DecisionTreeClassifier with TfidfVectorizer SMOTE Train_Set & Test_Set - 2010-2016", y_train_i)
    dt_end = time.time()
    print('DCT F1-score', fdct * 100)
    print('DCT ROCAUC score:', rocauc_dt * 100)
    print('DCT Recall score:', recalls_dt * 100)
    print('DCT Precision Score:', precisions_dt * 100)
    print('DCT Confusion Matrix', confusion_matrix(y_test_i, dct_pred), "\n")
    print('DCT Classification', classification_report(y_test_i, dct_pred), "\n")
    print('DCT Accuracy Score', accuracys_dt * 100)
    print("Execution Time for Decision Tree SMOTE Train_Set & Test_Set - 2010-2016", dt_end - start2, "seconds")

    from sklearn.naive_bayes import MultinomialNB

    start3 = time.time()
    Naive = MultinomialNB()
    model_nb = Naive.fit(x_train_i, y_train_i)
    probs_nb = model_nb.predict_proba(x_test_i)[:, 1]
    probs_nb_scol9.append(probs_nb)
    # predict the labels on validation dataset
    ny_pred = Naive.predict(x_test_i)
    fnb = f1_score(ny_pred, y_test_i)
    f1_nb_scol9.append(fnb)
    rocauc_nb = roc_auc_score(y_test_i, ny_pred)
    rocauc_nb_scol9.append(rocauc_nb)
    recalls_nb = recall_score(y_test_i, ny_pred)
    recall_nb_scol9.append(recalls_nb)
    precisions_nb = precision_score(y_test_i, ny_pred)
    precision_nb_scol9.append(precisions_nb)
    accuracys_nb = accuracy_score(y_test_i, ny_pred)
    accuracy_nb_scol9.append(accuracys_nb)
    nb_end = time.time()
    # Use accuracy_score function to get the accuracy
    print("===Naive Bayes with TfidfVectorizer Imabalanced SMOTE Train_Set & Test_Set - 2010-2016", y_train_i)
    print('Naive F1-score', fnb * 100)
    print('Naive ROCAUC score:', rocauc_nb * 100)
    print('Naive Recall score:', recalls_nb * 100)
    print('Naive Precision Score:', precisions_nb * 100)
    print('Naive Confusion Matrix', confusion_matrix(y_test_i, ny_pred), "\n")
    print('Naive Classification', classification_report(y_test_i, ny_pred), "\n")
    print('Naive Accuracy Score', accuracys_nb * 100)
    print("Execution Time for Naive Bayes SMOTE Train_Set & Test_Set - 2010-2016", nb_end - start3, "seconds")

    # XGBoost Classifier

    start4 = time.time()
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    xgb_model = XGBClassifier().fit(x_train_i, y_train_i)
    probs_xg = xgb_model.predict_proba(x_test_i)[:, 1]
    probs_xg_scol9.append(probs_xg)
    # predict
    xgb_y_predict = xgb_model.predict(x_test_i)
    fxg = f1_score(xgb_y_predict, y_test_i)
    f1_xg_scol9.append(fxg)
    rocauc_xg = roc_auc_score(xgb_y_predict, y_test_i)
    rocauc_xg_scol9.append(rocauc_xg)
    recall_xg = recall_score(xgb_y_predict, y_test_i)
    recall_xg_scol9.append(recall_xg)
    precisions_xg = precision_score(xgb_y_predict, y_test_i)
    precision_xg_scol9.append(precisions_xg)
    accuracys_xg = accuracy_score(xgb_y_predict, y_test_i)
    accuracy_xg_scol9.append(accuracys_xg)
    xg_end = time.time()
    print("===XGB with TfidfVectorizer SMOTE Train_Set & Test_Set - 2010-2016", y_train_i)
    print('XGB F1-Score', fxg * 100)
    print('XGB ROCAUC Score:', rocauc_xg * 100)
    print('XGB Recall score:', recall_xg * 100)
    print('XGB Precision Score:', precisions_xg * 100)
    print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, y_test_i), "\n")
    print('XGB Classification', classification_report(xgb_y_predict, y_test_i), "\n")
    print('XGB Accuracy Score', accuracys_nb * 100)
    print("Execution Time for XGBoost Classifier SMOTE Train_Set & Test_Set - 2010-2016", xg_end - start4, "seconds")

    # Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier

    start5 = time.time()
    rfc_model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(x_train_i, y_train_i)
    probs_rf = rfc_model.predict_proba(x_test_i)[:, 1]
    probs_rf_scol9.append(probs_rf)
    rfc_pred = rfc_model.predict(x_test_i)
    frfc = f1_score(rfc_pred, y_test_i)
    f1_rf_scol9.append(frfc)
    rocauc_rf = roc_auc_score(y_test_i, rfc_pred)
    rocauc_rf_scol9.append(rocauc_rf)
    recalls_rf = recall_score(rfc_pred, y_test_i)
    recall_rf_scol9.append(recalls_rf)
    precisions_rf = precision_score(rfc_pred, y_test_i)
    precision_rf_scol9.append(precisions_rf)
    accuracys_rf = accuracy_score(rfc_pred, y_test_i)
    accuracy_rf_scol9.append(accuracys_rf)
    rf_end = time.time()
    print("====RandomForest with Tfidf SMOTE Train_Set & Test_Set - 2010-2016", y_train_i)
    print('RFC F1 score', frfc * 100)
    print('RFC ROCAUC Score:', rocauc_rf * 100)
    print('RFC Recall score:', recalls_rf * 100)
    print('RFC Precision Score:', precisions_rf * 100)
    print('RFC Confusion Matrix', confusion_matrix(y_test_i, rfc_pred), "\n")
    print('RFC Classification', classification_report(y_test_i, rfc_pred), "\n")
    print('RFC Accuracy Score', accuracys_rf * 100)
    print("Execution Time for Random Forest Classifier SMOTE Train_Set & Test_Set - 2010-2016 ", rf_end - start5, "seconds")

    print("Array of Prob Scores LR-Sam SMOTE", ":", probs_lr_scol9)
    print("Array of F1 Scores LR-Sam SMOTE", ":", f1_lr_scol9)
    print("Array of ROCAUC Scores LR-Sam SMOTE", ":", rocauc_lr_scol9)
    print("Array of Recall Scores LR-Sam SMOTE", ":", recall_lr_scol9)
    print("Array of Precision Scores LR-Sam SMOTE", ":", precision_lr_scol9)
    print("Array of Accuracy Scores LR-Sam SMOTE", ":", accuracy_lr_scol9)

    print("Array of Prob Scores DT-Sam SMOTE", ":", probs_dt_scol9)
    print("Array of F1 Scores DT-Sam SMOTE", ":", f1_dt_scol9)
    print("Array of ROCAUC Scores DT-Sam SMOTE", ":", rocauc_dt_scol9)
    print("Array of Recall Scores DT-Sam SMOTE", ":", recall_dt_scol9)
    print("Array of Precision Scores DT-Sam SMOTE", ":", precision_dt_scol9)
    print("Array of Accuracy Scores DT-Sam SMOTE", ":", accuracy_dt_scol9)

    print("Array of Prob Scores NB-Sam SMOTE", ":", probs_nb_scol9)
    print("Array of F1 Scores NB-Sam SMOTE", ":", f1_nb_scol9)
    print("Array of ROCAUC Scores NB-Sam SMOTE", ":", rocauc_nb_scol9)
    print("Array of Recall Scores NB-Sam SMOTE", ":", recall_nb_scol9)
    print("Array of Precision Scores NB-Sam SMOTE", ":", precision_nb_scol9)
    print("Array of Accuracy Scores NB-Sam SMOTE", ":", accuracy_nb_scol9)

    print("Array of Prob Scores XG-Sam SMOTE", ":", probs_xg_scol9)
    print("Array of F1 Scores XG-Sam SMOTE", ":", f1_xg_scol9)
    print("Array of ROCAUC Scores XG-Sam SMOTE", ":", rocauc_xg_scol9)
    print("Array of Recall Scores XG-Sam SMOTE", ":", recall_xg_scol9)
    print("Array of Precision Scores XG-Sam SMOTE", ":", precision_xg_scol9)
    print("Array of Accuracy Scores XG-Sam SMOTE", ":", accuracy_xg_scol9)

    print("Array of Prob Scores RF-Sam SMOTE", ":", probs_rf_scol9)
    print("Array of F1 Scores RF-Sam SMOTE", ":", f1_rf_scol9)
    print("Array of ROCAUC Scores RF-Sam SMOTE", ":", rocauc_rf_scol9)
    print("Array of Recall Scores RF-Sam SMOTE", ":", recall_rf_scol9)
    print("Array of Precision Scores RF-Sam SMOTE", ":", precision_rf_scol9)
    print("Array of Accuracy Scores RF-Sam SMOTE", ":", accuracy_rf_scol9)

from itertools import chain

subset9 = ['Train_Set' for t in range(50)]
other_set9 = ['Test_Set' for t in range(50)]
sampling9 = ['Undersampling' for t in range(50)]
technique9 = ['SMOTE' for t in range(50)]
classifier_names9 = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
#train_sizes_num9 = [0.95]
# v9 = [0, 1, 2, 3, 4]
# precision_csv_num9 = [precision_lr_scol[z], precision_dt_scol[z], precision_nb_scol[z], precision_xg_scol[z], precision_rf_scol[z]]
# recall_csv_num9 = [recall_lr_scol[z], recall_dt_scol[z], recall_nb_scol[z], recall_xg_scol[z], recall_rf_scol[z]]
# auc_csv_num9 = [rocauc_lr_scol[z], rocauc_dt_scol[z], rocauc_nb_scol[z], rocauc_xg_scol[z], rocauc_rf_scol[z]]
# accuracy_csv_num9 = [accuracy_lr_scol[z], accuracy_dt_scol[z], accuracy_nb_scol[z], accuracy_xg_scol[z], accuracy_rf_scol[z]]
precision_csv_num9 = [precision_lr_scol9, precision_dt_scol9, precision_nb_scol9, precision_xg_scol9, precision_rf_scol9]
recall_csv_num9 = [recall_lr_scol9, recall_dt_scol9, recall_nb_scol9, recall_xg_scol9, recall_rf_scol9]
auc_csv_num9 = [rocauc_lr_scol9, rocauc_dt_scol9, rocauc_nb_scol9, rocauc_xg_scol9, rocauc_rf_scol9]
accuracy_csv_num9 = [accuracy_lr_scol9, accuracy_dt_scol9, accuracy_nb_scol9, accuracy_xg_scol9, accuracy_rf_scol9]
import itertools
rounds = 5
# rounds1 = 1
p9 = itertools.cycle(classifier_names)
o9 = itertools.cycle(subsample_num)
#k9 = itertools.cycle(train_sizes_num)
# v9 = itertools.cycle(score_location)
# pr9 = itertools.cycle(precision_num)
# y9 = itertools.cycle(iteration_csv)
classifier_csv9 = [next(p9) for _ in range(rounds)] * 10
subsample_csv9 = [r for r in subsample_num for _ in range(rounds)]
#test_size_csv9 = [next(o) for _ in range(rounds)] * 1
#train_size_csv9 = [next(k) for _ in range(rounds)] * 1
# split_csv9 = ['1' for t in range(25)]
# train_csv9 = ['0.5' for t in range(25)]
precision_csv9 = list(chain(*precision_csv_num9))
recall_csv9 = list(chain(*recall_csv_num9))
auc_csv9 = list(chain(*auc_csv_num9))
accuracy_csv9 = list(chain(*accuracy_csv_num9))
csv_data9 = [subset9, other_set9, sampling9, technique9, classifier_csv9, subsample_csv9, precision_csv9, recall_csv9, auc_csv9, accuracy_csv9]
export_data9 = zip_longest(*csv_data9, fillvalue='')
filename = "Subset_Iterations.csv"
with open(filename, 'a', newline='') as file:
    write = csv.writer(file)
    write.writerows(export_data9)
    
# ROS
probs_lr_scol10 = []
f1_lr_scol10 = []
rocauc_lr_scol10 = []
recall_lr_scol10 = []
precision_lr_scol10 = []
accuracy_lr_scol10 = []

probs_dt_scol10 = []
f1_dt_scol10 = []
rocauc_dt_scol10 = []
recall_dt_scol10 = []
precision_dt_scol10 = []
accuracy_dt_scol10 = []

probs_nb_scol10 = []
f1_nb_scol10 = []
rocauc_nb_scol10 = []
recall_nb_scol10 = []
precision_nb_scol10 = []
accuracy_nb_scol10 = []

probs_xg_scol10 = []
f1_xg_scol10 = []
rocauc_xg_scol10 = []
recall_xg_scol10 = []
precision_xg_scol10 = []
accuracy_xg_scol10 = []

probs_rf_scol10 = []
f1_rf_scol10 = []
rocauc_rf_scol10 = []
recall_rf_scol10 = []
precision_rf_scol10 = []
accuracy_rf_scol10 = []

x_ros11, y_ros11 = ros.fit_resample(subsample11, sublabel11)
x_ros12, y_ros12 = ros.fit_resample(subsample12, sublabel12)
x_ros13, y_ros13 = ros.fit_resample(subsample13, sublabel13)
x_ros14, y_ros14 = ros.fit_resample(subsample14, sublabel14)
x_ros15, y_ros15 = ros.fit_resample(subsample15, sublabel15)
x_ros16, y_ros16 = ros.fit_resample(subsample16, sublabel16)
x_ros17, y_ros17 = ros.fit_resample(subsample17, sublabel17)
x_ros18, y_ros18 = ros.fit_resample(subsample18, sublabel18)
x_ros19, y_ros19 = ros.fit_resample(subsample19, sublabel19)
x_ros20, y_ros20 = ros.fit_resample(subsample20, sublabel20)

sample_list8 = [x_ros11, x_ros12, x_ros13, x_ros14, x_ros15, x_ros16, x_ros17, x_ros18, x_ros19, x_ros20]
label_list8 = [y_ros11, y_ros12, y_ros13, y_ros14, y_ros15, y_ros16, y_ros17, y_ros18, y_ros19, y_ros20]

#for i, j in zip(sample_list, label_list):

for i, j, z in zip(sample_list8, label_list8, subsample_num):
#for h, j, k, l in zip(tl_xtrain_list, tl_xtest_list, tl_ytrain_list, tl_ytest_list):
    x_train_i, x_test_i, y_train_i, y_test_i = train_test_split(i, j, train_size=0.5, test_size=0.5)

    start1 = time.time()
    log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
    model_lr = log.fit(x_train_i, y_train_i)
    probs_lr = model_lr.predict_proba(x_test_i)[:, 1]
    probs_lr_scol10.append(probs_lr)
    ly_prediction = log.predict(x_test_i)
    fly = f1_score(ly_prediction, y_test_i)
    f1_lr_scol10.append(fly)
    rocauc_lr = roc_auc_score(y_test_i, ly_prediction)
    rocauc_lr_scol10.append(rocauc_lr)
    recalls_lr = recall_score(y_test_i, ly_prediction)
    recall_lr_scol10.append(recalls_lr)
    precisions_lr = precision_score(y_test_i, ly_prediction)
    precision_lr_scol10.append(precisions_lr)
    accuracys_lr = accuracy_score(y_test_i, ly_prediction)
    accuracy_lr_scol10.append(accuracys_lr)
    print("===Logistic Regression with TfidfVectorizer ROS Train_Set & Test_Set - 2010-2016", y_train_i)
    lr_end = time.time()
    print('Logistic F1-score', fly * 100)
    print('Logistic ROCAUC score:', rocauc_lr * 100)
    print('Logistic Recall score:', recalls_lr * 100)
    print('Logistic Precision Score:', precisions_lr * 100)
    print('Logistic Confusion Matrix', confusion_matrix(y_test_i, ly_prediction), "\n")
    print('Logistic Classification', classification_report(y_test_i, ly_prediction), "\n")
    print('Logistic Accuracy Score', accuracys_lr * 100)
    print("Execution Time for Logistic Regression ROS Train_Set & Test_Set - 2010-2016", lr_end - start1, "seconds")

    start2 = time.time()
    from sklearn.tree import DecisionTreeClassifier

    DCT = DecisionTreeClassifier()
    model_dt = DCT.fit(x_train_i, y_train_i)
    probs_dt = model_dt.predict_proba(x_test_i)[:, 1]
    probs_dt_scol10.append(probs_dt)
    dct_pred = DCT.predict(x_test_i)
    fdct = f1_score(dct_pred, y_test_i)
    f1_dt_scol10.append(fdct)
    rocauc_dt = roc_auc_score(y_test_i, dct_pred)
    rocauc_dt_scol10.append(rocauc_dt)
    recalls_dt = recall_score(y_test_i, dct_pred)
    recall_dt_scol10.append(recalls_dt)
    precisions_dt = precision_score(y_test_i, dct_pred)
    precision_dt_scol10.append(precisions_dt)
    accuracys_dt = accuracy_score(y_test_i, dct_pred)
    accuracy_dt_scol10.append(accuracys_dt)
    print("===DecisionTreeClassifier with TfidfVectorizer ROS Train_Set & Test_Set - 2010-2016", y_train_i)
    dt_end = time.time()
    print('DCT F1-score', fdct * 100)
    print('DCT ROCAUC score:', rocauc_dt * 100)
    print('DCT Recall score:', recalls_dt * 100)
    print('DCT Precision Score:', precisions_dt * 100)
    print('DCT Confusion Matrix', confusion_matrix(y_test_i, dct_pred), "\n")
    print('DCT Classification', classification_report(y_test_i, dct_pred), "\n")
    print('DCT Accuracy Score', accuracys_dt * 100)
    print("Execution Time for Decision Tree ROS Train_Set & Test_Set - 2010-2016", dt_end - start2, "seconds")

    from sklearn.naive_bayes import MultinomialNB

    start3 = time.time()
    Naive = MultinomialNB()
    model_nb = Naive.fit(x_train_i, y_train_i)
    probs_nb = model_nb.predict_proba(x_test_i)[:, 1]
    probs_nb_scol10.append(probs_nb)
    # predict the labels on validation dataset
    ny_pred = Naive.predict(x_test_i)
    fnb = f1_score(ny_pred, y_test_i)
    f1_nb_scol10.append(fnb)
    rocauc_nb = roc_auc_score(y_test_i, ny_pred)
    rocauc_nb_scol10.append(rocauc_nb)
    recalls_nb = recall_score(y_test_i, ny_pred)
    recall_nb_scol10.append(recalls_nb)
    precisions_nb = precision_score(y_test_i, ny_pred)
    precision_nb_scol10.append(precisions_nb)
    accuracys_nb = accuracy_score(y_test_i, ny_pred)
    accuracy_nb_scol10.append(accuracys_nb)
    nb_end = time.time()
    # Use accuracy_score function to get the accuracy
    print("===Naive Bayes with TfidfVectorizer Imabalanced ROS Train_Set & Test_Set - 2010-2016", y_train_i)
    print('Naive F1-score', fnb * 100)
    print('Naive ROCAUC score:', rocauc_nb * 100)
    print('Naive Recall score:', recalls_nb * 100)
    print('Naive Precision Score:', precisions_nb * 100)
    print('Naive Confusion Matrix', confusion_matrix(y_test_i, ny_pred), "\n")
    print('Naive Classification', classification_report(y_test_i, ny_pred), "\n")
    print('Naive Accuracy Score', accuracys_nb * 100)
    print("Execution Time for Naive Bayes ROS Train_Set & Test_Set - 2010-2016", nb_end - start3, "seconds")

    # XGBoost Classifier

    start4 = time.time()
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    xgb_model = XGBClassifier().fit(x_train_i, y_train_i)
    probs_xg = xgb_model.predict_proba(x_test_i)[:, 1]
    probs_xg_scol10.append(probs_xg)
    # predict
    xgb_y_predict = xgb_model.predict(x_test_i)
    fxg = f1_score(xgb_y_predict, y_test_i)
    f1_xg_scol10.append(fxg)
    rocauc_xg = roc_auc_score(xgb_y_predict, y_test_i)
    rocauc_xg_scol10.append(rocauc_xg)
    recall_xg = recall_score(xgb_y_predict, y_test_i)
    recall_xg_scol10.append(recall_xg)
    precisions_xg = precision_score(xgb_y_predict, y_test_i)
    precision_xg_scol10.append(precisions_xg)
    accuracys_xg = accuracy_score(xgb_y_predict, y_test_i)
    accuracy_xg_scol10.append(accuracys_xg)
    xg_end = time.time()
    print("===XGB with TfidfVectorizer ROS Train_Set & Test_Set - 2010-2016", y_train_i)
    print('XGB F1-Score', fxg * 100)
    print('XGB ROCAUC Score:', rocauc_xg * 100)
    print('XGB Recall score:', recall_xg * 100)
    print('XGB Precision Score:', precisions_xg * 100)
    print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, y_test_i), "\n")
    print('XGB Classification', classification_report(xgb_y_predict, y_test_i), "\n")
    print('XGB Accuracy Score', accuracys_nb * 100)
    print("Execution Time for XGBoost Classifier ROS Train_Set & Test_Set - 2010-2016", xg_end - start4, "seconds")

    # Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier

    start5 = time.time()
    rfc_model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(x_train_i, y_train_i)
    probs_rf = rfc_model.predict_proba(x_test_i)[:, 1]
    probs_rf_scol10.append(probs_rf)
    rfc_pred = rfc_model.predict(x_test_i)
    frfc = f1_score(rfc_pred, y_test_i)
    f1_rf_scol10.append(frfc)
    rocauc_rf = roc_auc_score(y_test_i, rfc_pred)
    rocauc_rf_scol10.append(rocauc_rf)
    recalls_rf = recall_score(rfc_pred, y_test_i)
    recall_rf_scol10.append(recalls_rf)
    precisions_rf = precision_score(rfc_pred, y_test_i)
    precision_rf_scol10.append(precisions_rf)
    accuracys_rf = accuracy_score(rfc_pred, y_test_i)
    accuracy_rf_scol10.append(accuracys_rf)
    rf_end = time.time()
    print("====RandomForest with Tfidf ROS Train_Set & Test_Set - 2010-2016", y_train_i)
    print('RFC F1 score', frfc * 100)
    print('RFC ROCAUC Score:', rocauc_rf * 100)
    print('RFC Recall score:', recalls_rf * 100)
    print('RFC Precision Score:', precisions_rf * 100)
    print('RFC Confusion Matrix', confusion_matrix(y_test_i, rfc_pred), "\n")
    print('RFC Classification', classification_report(y_test_i, rfc_pred), "\n")
    print('RFC Accuracy Score', accuracys_rf * 100)
    print("Execution Time for Random Forest Classifier ROS Train_Set & Test_Set - 2010-2016 ", rf_end - start5, "seconds")

    print("Array of Prob Scores LR-Sam ROS", ":", probs_lr_scol10)
    print("Array of F1 Scores LR-Sam ROS", ":", f1_lr_scol10)
    print("Array of ROCAUC Scores LR-Sam ROS", ":", rocauc_lr_scol10)
    print("Array of Recall Scores LR-Sam ROS", ":", recall_lr_scol10)
    print("Array of Precision Scores LR-Sam ROS", ":", precision_lr_scol10)
    print("Array of Accuracy Scores LR-Sam ROS", ":", accuracy_lr_scol10)

    print("Array of Prob Scores DT-Sam ROS", ":", probs_dt_scol10)
    print("Array of F1 Scores DT-Sam ROS", ":", f1_dt_scol10)
    print("Array of ROCAUC Scores DT-Sam ROS", ":", rocauc_dt_scol10)
    print("Array of Recall Scores DT-Sam ROS", ":", recall_dt_scol10)
    print("Array of Precision Scores DT-Sam ROS", ":", precision_dt_scol10)
    print("Array of Accuracy Scores DT-Sam ROS", ":", accuracy_dt_scol10)

    print("Array of Prob Scores NB-Sam ROS", ":", probs_nb_scol10)
    print("Array of F1 Scores NB-Sam ROS", ":", f1_nb_scol10)
    print("Array of ROCAUC Scores NB-Sam ROS", ":", rocauc_nb_scol10)
    print("Array of Recall Scores NB-Sam ROS", ":", recall_nb_scol10)
    print("Array of Precision Scores NB-Sam ROS", ":", precision_nb_scol10)
    print("Array of Accuracy Scores NB-Sam ROS", ":", accuracy_nb_scol10)

    print("Array of Prob Scores XG-Sam ROS", ":", probs_xg_scol10)
    print("Array of F1 Scores XG-Sam ROS", ":", f1_xg_scol10)
    print("Array of ROCAUC Scores XG-Sam ROS", ":", rocauc_xg_scol10)
    print("Array of Recall Scores XG-Sam ROS", ":", recall_xg_scol10)
    print("Array of Precision Scores XG-Sam ROS", ":", precision_xg_scol10)
    print("Array of Accuracy Scores XG-Sam ROS", ":", accuracy_xg_scol10)

    print("Array of Prob Scores RF-Sam ROS", ":", probs_rf_scol10)
    print("Array of F1 Scores RF-Sam ROS", ":", f1_rf_scol10)
    print("Array of ROCAUC Scores RF-Sam ROS", ":", rocauc_rf_scol10)
    print("Array of Recall Scores RF-Sam ROS", ":", recall_rf_scol10)
    print("Array of Precision Scores RF-Sam ROS", ":", precision_rf_scol10)
    print("Array of Accuracy Scores RF-Sam ROS", ":", accuracy_rf_scol10)

from itertools import chain

subset10 = ['Train_Set' for t in range(50)]
other_set10 = ['Test_Set' for t in range(50)]
sampling10 = ['Undersampling' for t in range(50)]
technique10 = ['ROS' for t in range(50)]
classifier_names10 = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
#train_sizes_num10 = [0.95]
# v10 = [0, 1, 2, 3, 4]
# precision_csv_num10 = [precision_lr_scol[z], precision_dt_scol[z], precision_nb_scol[z], precision_xg_scol[z], precision_rf_scol[z]]
# recall_csv_num10 = [recall_lr_scol[z], recall_dt_scol[z], recall_nb_scol[z], recall_xg_scol[z], recall_rf_scol[z]]
# auc_csv_num10 = [rocauc_lr_scol[z], rocauc_dt_scol[z], rocauc_nb_scol[z], rocauc_xg_scol[z], rocauc_rf_scol[z]]
# accuracy_csv_num10 = [accuracy_lr_scol[z], accuracy_dt_scol[z], accuracy_nb_scol[z], accuracy_xg_scol[z], accuracy_rf_scol[z]]
precision_csv_num10 = [precision_lr_scol10, precision_dt_scol10, precision_nb_scol10, precision_xg_scol10, precision_rf_scol10]
recall_csv_num10 = [recall_lr_scol10, recall_dt_scol10, recall_nb_scol10, recall_xg_scol10, recall_rf_scol10]
auc_csv_num10 = [rocauc_lr_scol10, rocauc_dt_scol10, rocauc_nb_scol10, rocauc_xg_scol10, rocauc_rf_scol10]
accuracy_csv_num10 = [accuracy_lr_scol10, accuracy_dt_scol10, accuracy_nb_scol10, accuracy_xg_scol10, accuracy_rf_scol10]
import itertools
rounds = 5
# rounds1 = 1
p10 = itertools.cycle(classifier_names10)
o10 = itertools.cycle(subsample_num)
#k10 = itertools.cycle(train_sizes_num)
# v10 = itertools.cycle(score_location)
# pr10 = itertools.cycle(precision_num)
# y10 = itertools.cycle(iteration_csv)
classifier_csv10 = [next(p10) for _ in range(rounds)] * 10
subsample_csv10 = [r for r in subsample_num for _ in range(rounds)]
#test_size_csv10 = [next(o) for _ in range(rounds)] * 1
#train_size_csv10 = [next(k) for _ in range(rounds)] * 1
# split_csv10 = ['1' for t in range(25)]
# train_csv10 = ['0.5' for t in range(25)]
precision_csv10 = list(chain(*precision_csv_num10))
recall_csv10 = list(chain(*recall_csv_num10))
auc_csv10 = list(chain(*auc_csv_num10))
accuracy_csv10 = list(chain(*accuracy_csv_num10))
csv_data10 = [subset10, sampling10, technique10, classifier_csv10, subsample_csv10, precision_csv10, recall_csv10, auc_csv10, accuracy_csv10]
export_data10 = zip_longest(*csv_data10, fillvalue='')
filename = "Subset_Iterations.csv"
with open(filename, 'a', newline='') as file:
    write = csv.writer(file)
    write.writerows(export_data10)
    
    
# RUS
probs_lr_scol11 = []
f1_lr_scol11 = []
rocauc_lr_scol11 = []
recall_lr_scol11 = []
precision_lr_scol11 = []
accuracy_lr_scol11 = []

probs_dt_scol11 = []
f1_dt_scol11 = []
rocauc_dt_scol11 = []
recall_dt_scol11 = []
precision_dt_scol11 = []
accuracy_dt_scol11 = []

probs_nb_scol11 = []
f1_nb_scol11 = []
rocauc_nb_scol11 = []
recall_nb_scol11 = []
precision_nb_scol11 = []
accuracy_nb_scol11 = []

probs_xg_scol11 = []
f1_xg_scol11 = []
rocauc_xg_scol11 = []
recall_xg_scol11 = []
precision_xg_scol11 = []
accuracy_xg_scol11 = []

probs_rf_scol11 = []
f1_rf_scol11 = []
rocauc_rf_scol11 = []
recall_rf_scol11 = []
precision_rf_scol11 = []
accuracy_rf_scol11 = []

x_rus11, y_rus11 = rus.fit_resample(subsample11, sublabel11)
x_rus12, y_rus12 = rus.fit_resample(subsample12, sublabel12)
x_rus13, y_rus13 = rus.fit_resample(subsample13, sublabel13)
x_rus14, y_rus14 = rus.fit_resample(subsample14, sublabel14)
x_rus15, y_rus15 = rus.fit_resample(subsample15, sublabel15)
x_rus16, y_rus16 = rus.fit_resample(subsample16, sublabel16)
x_rus17, y_rus17 = rus.fit_resample(subsample17, sublabel17)
x_rus18, y_rus18 = rus.fit_resample(subsample18, sublabel18)
x_rus19, y_rus19 = rus.fit_resample(subsample19, sublabel19)
x_rus20, y_rus20 = rus.fit_resample(subsample20, sublabel20)

sample_list9 = [x_rus11, x_rus12, x_rus13, x_rus14, x_rus15, x_rus16, x_rus17, x_rus18, x_rus19, x_rus20]
label_list9 = [y_rus11, y_rus12, y_rus13, y_rus14, y_rus15, y_rus16, y_rus17, y_rus18, y_rus19, y_rus20]

#for i, j in zip(sample_list, label_list):

for i, j, z in zip(sample_list9, label_list9, subsample_num):
#for h, j, k, l in zip(tl_xtrain_list, tl_xtest_list, tl_ytrain_list, tl_ytest_list):
    x_train_i, x_test_i, y_train_i, y_test_i = train_test_split(i, j, train_size=0.5, test_size=0.5)

    start1 = time.time()
    log = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
    model_lr = log.fit(x_train_i, y_train_i)
    probs_lr = model_lr.predict_proba(x_test_i)[:, 1]
    probs_lr_scol11.append(probs_lr)
    ly_prediction = log.predict(x_test_i)
    fly = f1_score(ly_prediction, y_test_i)
    f1_lr_scol11.append(fly)
    rocauc_lr = roc_auc_score(y_test_i, ly_prediction)
    rocauc_lr_scol11.append(rocauc_lr)
    recalls_lr = recall_score(y_test_i, ly_prediction)
    recall_lr_scol11.append(recalls_lr)
    precisions_lr = precision_score(y_test_i, ly_prediction)
    precision_lr_scol11.append(precisions_lr)
    accuracys_lr = accuracy_score(y_test_i, ly_prediction)
    accuracy_lr_scol11.append(accuracys_lr)
    print("===Logistic Regression with TfidfVectorizer RUS Train_Set & Test_Set - 2010-2016", y_train_i)
    lr_end = time.time()
    print('Logistic F1-score', fly * 100)
    print('Logistic ROCAUC score:', rocauc_lr * 100)
    print('Logistic Recall score:', recalls_lr * 100)
    print('Logistic Precision Score:', precisions_lr * 100)
    print('Logistic Confusion Matrix', confusion_matrix(y_test_i, ly_prediction), "\n")
    print('Logistic Classification', classification_report(y_test_i, ly_prediction), "\n")
    print('Logistic Accuracy Score', accuracys_lr * 100)
    print("Execution Time for Logistic Regression RUS Train_Set & Test_Set - 2010-2016", lr_end - start1, "seconds")

    start2 = time.time()
    from sklearn.tree import DecisionTreeClassifier

    DCT = DecisionTreeClassifier()
    model_dt = DCT.fit(x_train_i, y_train_i)
    probs_dt = model_dt.predict_proba(x_test_i)[:, 1]
    probs_dt_scol11.append(probs_dt)
    dct_pred = DCT.predict(x_test_i)
    fdct = f1_score(dct_pred, y_test_i)
    f1_dt_scol11.append(fdct)
    rocauc_dt = roc_auc_score(y_test_i, dct_pred)
    rocauc_dt_scol11.append(rocauc_dt)
    recalls_dt = recall_score(y_test_i, dct_pred)
    recall_dt_scol11.append(recalls_dt)
    precisions_dt = precision_score(y_test_i, dct_pred)
    precision_dt_scol11.append(precisions_dt)
    accuracys_dt = accuracy_score(y_test_i, dct_pred)
    accuracy_dt_scol11.append(accuracys_dt)
    print("===DecisionTreeClassifier with TfidfVectorizer RUS Train_Set & Test_Set - 2010-2016", y_train_i)
    dt_end = time.time()
    print('DCT F1-score', fdct * 100)
    print('DCT ROCAUC score:', rocauc_dt * 100)
    print('DCT Recall score:', recalls_dt * 100)
    print('DCT Precision Score:', precisions_dt * 100)
    print('DCT Confusion Matrix', confusion_matrix(y_test_i, dct_pred), "\n")
    print('DCT Classification', classification_report(y_test_i, dct_pred), "\n")
    print('DCT Accuracy Score', accuracys_dt * 100)
    print("Execution Time for Decision Tree RUS Train_Set & Test_Set - 2010-2016", dt_end - start2, "seconds")

    from sklearn.naive_bayes import MultinomialNB

    start3 = time.time()
    Naive = MultinomialNB()
    model_nb = Naive.fit(x_train_i, y_train_i)
    probs_nb = model_nb.predict_proba(x_test_i)[:, 1]
    probs_nb_scol11.append(probs_nb)
    # predict the labels on validation dataset
    ny_pred = Naive.predict(x_test_i)
    fnb = f1_score(ny_pred, y_test_i)
    f1_nb_scol11.append(fnb)
    rocauc_nb = roc_auc_score(y_test_i, ny_pred)
    rocauc_nb_scol11.append(rocauc_nb)
    recalls_nb = recall_score(y_test_i, ny_pred)
    recall_nb_scol11.append(recalls_nb)
    precisions_nb = precision_score(y_test_i, ny_pred)
    precision_nb_scol11.append(precisions_nb)
    accuracys_nb = accuracy_score(y_test_i, ny_pred)
    accuracy_nb_scol11.append(accuracys_nb)
    nb_end = time.time()
    # Use accuracy_score function to get the accuracy
    print("===Naive Bayes with TfidfVectorizer Imabalanced RUS Train_Set & Test_Set - 2010-2016", y_train_i)
    print('Naive F1-score', fnb * 100)
    print('Naive ROCAUC score:', rocauc_nb * 100)
    print('Naive Recall score:', recalls_nb * 100)
    print('Naive Precision Score:', precisions_nb * 100)
    print('Naive Confusion Matrix', confusion_matrix(y_test_i, ny_pred), "\n")
    print('Naive Classification', classification_report(y_test_i, ny_pred), "\n")
    print('Naive Accuracy Score', accuracys_nb * 100)
    print("Execution Time for Naive Bayes RUS Train_Set & Test_Set - 2010-2016", nb_end - start3, "seconds")

    # XGBoost Classifier

    start4 = time.time()
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    xgb_model = XGBClassifier().fit(x_train_i, y_train_i)
    probs_xg = xgb_model.predict_proba(x_test_i)[:, 1]
    probs_xg_scol11.append(probs_xg)
    # predict
    xgb_y_predict = xgb_model.predict(x_test_i)
    fxg = f1_score(xgb_y_predict, y_test_i)
    f1_xg_scol11.append(fxg)
    rocauc_xg = roc_auc_score(xgb_y_predict, y_test_i)
    rocauc_xg_scol11.append(rocauc_xg)
    recall_xg = recall_score(xgb_y_predict, y_test_i)
    recall_xg_scol11.append(recall_xg)
    precisions_xg = precision_score(xgb_y_predict, y_test_i)
    precision_xg_scol11.append(precisions_xg)
    accuracys_xg = accuracy_score(xgb_y_predict, y_test_i)
    accuracy_xg_scol11.append(accuracys_xg)
    xg_end = time.time()
    print("===XGB with TfidfVectorizer RUS Train_Set & Test_Set - 2010-2016", y_train_i)
    print('XGB F1-Score', fxg * 100)
    print('XGB ROCAUC Score:', rocauc_xg * 100)
    print('XGB Recall score:', recall_xg * 100)
    print('XGB Precision Score:', precisions_xg * 100)
    print('XGB Confusion Matrix', confusion_matrix(xgb_y_predict, y_test_i), "\n")
    print('XGB Classification', classification_report(xgb_y_predict, y_test_i), "\n")
    print('XGB Accuracy Score', accuracys_nb * 100)
    print("Execution Time for XGBoost Classifier RUS Train_Set & Test_Set - 2010-2016", xg_end - start4, "seconds")

    # Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier

    start5 = time.time()
    rfc_model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(x_train_i, y_train_i)
    probs_rf = rfc_model.predict_proba(x_test_i)[:, 1]
    probs_rf_scol11.append(probs_rf)
    rfc_pred = rfc_model.predict(x_test_i)
    frfc = f1_score(rfc_pred, y_test_i)
    f1_rf_scol11.append(frfc)
    rocauc_rf = roc_auc_score(y_test_i, rfc_pred)
    rocauc_rf_scol11.append(rocauc_rf)
    recalls_rf = recall_score(rfc_pred, y_test_i)
    recall_rf_scol11.append(recalls_rf)
    precisions_rf = precision_score(rfc_pred, y_test_i)
    precision_rf_scol11.append(precisions_rf)
    accuracys_rf = accuracy_score(rfc_pred, y_test_i)
    accuracy_rf_scol11.append(accuracys_rf)
    rf_end = time.time()
    print("====RandomForest with Tfidf RUS Train_Set & Test_Set - 2010-2016", y_train_i)
    print('RFC F1 score', frfc * 100)
    print('RFC ROCAUC Score:', rocauc_rf * 100)
    print('RFC Recall score:', recalls_rf * 100)
    print('RFC Precision Score:', precisions_rf * 100)
    print('RFC Confusion Matrix', confusion_matrix(y_test_i, rfc_pred), "\n")
    print('RFC Classification', classification_report(y_test_i, rfc_pred), "\n")
    print('RFC Accuracy Score', accuracys_rf * 100)
    print("Execution Time for Random Forest Classifier RUS Train_Set & Test_Set - 2010-2016 ", rf_end - start5, "seconds")

    print("Array of Prob Scores LR-Sam RUS", ":", probs_lr_scol11)
    print("Array of F1 Scores LR-Sam RUS", ":", f1_lr_scol11)
    print("Array of ROCAUC Scores LR-Sam RUS", ":", rocauc_lr_scol11)
    print("Array of Recall Scores LR-Sam RUS", ":", recall_lr_scol11)
    print("Array of Precision Scores LR-Sam RUS", ":", precision_lr_scol11)
    print("Array of Accuracy Scores LR-Sam RUS", ":", accuracy_lr_scol11)

    print("Array of Prob Scores DT-Sam RUS", ":", probs_dt_scol11)
    print("Array of F1 Scores DT-Sam RUS", ":", f1_dt_scol11)
    print("Array of ROCAUC Scores DT-Sam RUS", ":", rocauc_dt_scol11)
    print("Array of Recall Scores DT-Sam RUS", ":", recall_dt_scol11)
    print("Array of Precision Scores DT-Sam RUS", ":", precision_dt_scol11)
    print("Array of Accuracy Scores DT-Sam RUS", ":", accuracy_dt_scol11)

    print("Array of Prob Scores NB-Sam RUS", ":", probs_nb_scol11)
    print("Array of F1 Scores NB-Sam RUS", ":", f1_nb_scol11)
    print("Array of ROCAUC Scores NB-Sam RUS", ":", rocauc_nb_scol11)
    print("Array of Recall Scores NB-Sam RUS", ":", recall_nb_scol11)
    print("Array of Precision Scores NB-Sam RUS", ":", precision_nb_scol11)
    print("Array of Accuracy Scores NB-Sam RUS", ":", accuracy_nb_scol11)

    print("Array of Prob Scores XG-Sam RUS", ":", probs_xg_scol11)
    print("Array of F1 Scores XG-Sam RUS", ":", f1_xg_scol11)
    print("Array of ROCAUC Scores XG-Sam RUS", ":", rocauc_xg_scol11)
    print("Array of Recall Scores XG-Sam RUS", ":", recall_xg_scol11)
    print("Array of Precision Scores XG-Sam RUS", ":", precision_xg_scol11)
    print("Array of Accuracy Scores XG-Sam RUS", ":", accuracy_xg_scol11)

    print("Array of Prob Scores RF-Sam RUS", ":", probs_rf_scol11)
    print("Array of F1 Scores RF-Sam RUS", ":", f1_rf_scol11)
    print("Array of ROCAUC Scores RF-Sam RUS", ":", rocauc_rf_scol11)
    print("Array of Recall Scores RF-Sam RUS", ":", recall_rf_scol11)
    print("Array of Precision Scores RF-Sam RUS", ":", precision_rf_scol11)
    print("Array of Accuracy Scores RF-Sam RUS", ":", accuracy_rf_scol11)

from itertools import chain

subset11 = ['Train_Set' for t in range(50)]
other_set11 = ['Test_Set' for t in range(50)]
sampling11 = ['Undersampling' for t in range(50)]
technique11 = ['RUS' for t in range(50)]
classifier_names11 = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
#train_sizes_num = [0.95]
# v = [0, 1, 2, 3, 4]
# precision_csv_num = [precision_lr_scol[z], precision_dt_scol[z], precision_nb_scol[z], precision_xg_scol[z], precision_rf_scol[z]]
# recall_csv_num = [recall_lr_scol[z], recall_dt_scol[z], recall_nb_scol[z], recall_xg_scol[z], recall_rf_scol[z]]
# auc_csv_num = [rocauc_lr_scol[z], rocauc_dt_scol[z], rocauc_nb_scol[z], rocauc_xg_scol[z], rocauc_rf_scol[z]]
# accuracy_csv_num = [accuracy_lr_scol[z], accuracy_dt_scol[z], accuracy_nb_scol[z], accuracy_xg_scol[z], accuracy_rf_scol[z]]
precision_csv_num11 = [precision_lr_scol11, precision_dt_scol11, precision_nb_scol11, precision_xg_scol11, precision_rf_scol11]
recall_csv_num11 = [recall_lr_scol11, recall_dt_scol11, recall_nb_scol11, recall_xg_scol11, recall_rf_scol11]
auc_csv_num11 = [rocauc_lr_scol11, rocauc_dt_scol11, rocauc_nb_scol11, rocauc_xg_scol11, rocauc_rf_scol11]
accuracy_csv_num11 = [accuracy_lr_scol11, accuracy_dt_scol11, accuracy_nb_scol11, accuracy_xg_scol11, accuracy_rf_scol11]
import itertools
rounds = 5
# rounds1 = 1
p11 = itertools.cycle(classifier_names11)
o11 = itertools.cycle(subsample_num)
#k = itertools.cycle(train_sizes_num)
# v = itertools.cycle(score_location)
# pr = itertools.cycle(precision_num)
# y = itertools.cycle(iteration_csv)
classifier_csv11 = [next(p11) for _ in range(rounds)] * 10
subsample_csv11 = [r for r in subsample_num for _ in range(rounds)]
#test_size_csv = [next(o) for _ in range(rounds)] * 1
#train_size_csv = [next(k) for _ in range(rounds)] * 1
# split_csv = ['1' for t in range(25)]
# train_csv = ['0.5' for t in range(25)]
precision_csv11 = list(chain(*precision_csv_num11))
recall_csv11 = list(chain(*recall_csv_num11))
auc_csv11 = list(chain(*auc_csv_num11))
accuracy_csv11 = list(chain(*accuracy_csv_num11))
csv_data11 = [subset11, other_set11, sampling11, technique11, classifier_csv11, subsample_csv11, precision_csv11, recall_csv11, auc_csv11, accuracy_csv11]
export_data11 = zip_longest(*csv_data11, fillvalue='')
filename = "Subset_Iterations.csv"
with open(filename, 'a', newline='') as file:
    write = csv.writer(file)
    write.writerows(export_data11)
    
writef.close()