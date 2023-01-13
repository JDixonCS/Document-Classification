import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


text_data = load_files(r"C:\Users\z3696\Documents\Document-Classification\classifier\NIST_TEXT\2010")
X, y = text_data.data, text_data.target

documents = []

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

#Term frequency = (Number of Occurrences of a word)/(Total words in the document)
#IDF(word) = Log((Total number of documents)/(Number of documents containing the word))

from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(documents).toarray()
#X_train_tfidf = tfidfconverter.transform(X_train)
#X_test_tfidf = tfidfconverter.transform(X_test)

from sklearn.preprocessing import LabelEncoder
Encoder = LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_test = Encoder.fit_transform(y_test)

print("2010")
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFCclassifier = RandomForestClassifier(n_estimators=1000, random_state=0)
RFCclassifier.fit(X_train, y_train)
ry_pred = RFCclassifier.predict(X_test)
# Use accuracy_score function to get the accuracy
print("RFC Accuracy Score -> ",accuracy_score(ry_pred, y_test))
print("RFC Confusion Matrix -> ", confusion_matrix(ry_pred, y_test))
print("RFC Classification -> ", classification_report(ry_pred, y_test))


# SVM Classifier
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
from sklearn import svm
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_train, y_train)
# predict the labels on validation dataset
svy_pred = SVM.predict(X_test)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(svy_pred, y_test))
print("SVM Confusion Matrix -> ", confusion_matrix(svy_pred, y_test))
print("SVM Classification -> ", classification_report(svy_pred, y_test))
'''
'''
# Naives_Bayes Classifier
# fit the training dataset on the NB classifier
from sklearn.naive_bayes import MultinomialNB
Naive = MultinomialNB()
Naive.fit(X_train,y_train)
# predict the labels on validation dataset
ny_pred = Naive.predict(X_test)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(ny_pred, y_test))
print("Naive Confusion Matrix -> ", confusion_matrix(ny_pred, y_test))
print("Naive Classification -> ", classification_report(ny_pred, y_test))

# LogisticRegression Classifier
# fit the training dataset on the LogisticRegression Classifier
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
ly_pred = log.predict(X_test)
#f1 = f1_score(ly_prediction,y_test)
#print('LC score',f1*100)
print('LC Accuracy Score', accuracy_score(ly_pred, y_test))
print('LC Confusion Matrix', confusion_matrix(ly_pred, y_test), "\n")
print('Logistic Classification', classification_report(ly_pred, y_test), "\n")


'''
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
'''