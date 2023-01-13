import nltk
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import movie_reviews

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, featureset):
        votes = []
        for classifier in self._classifiers:
            v = classifier.classify(featureset)
            votes.append(v)
        return mode(votes)

    def confidence(self, featureset):
        votes = []
        for c in self._classifiers:
            v = c.classify(featureset)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]


# Positive training set (with no random)
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# Negative training set (with no random)
training_set = featuresets[100:]
testing_set = featuresets[:100]

# Naive Bayes
# posterior = prior x likelihood/evidence


# classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier_f = open("naivebayes.pickle",'rb')
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Original Naive Bayes Algo Accuracy: ", nltk.classify.accuracy(classifier, testing_set)*100)

classifier.show_most_informative_features(15)

# save_classifier = open("naivebayes.pickle","wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB Algo Accuracy: ", nltk.classify.accuracy(MNB_classifier, testing_set)*100)

# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(training_set)
# print("Gaussian Algo Accuracy: ", nltk.classify.accuracy(GaussianNB_classifier, testing_set)*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB Algo Accuracy: ", nltk.classify.accuracy(BernoulliNB_classifier, testing_set)*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression Algo Accuracy: ", nltk.classify.accuracy(LogisticRegression_classifier, testing_set)*100)


SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier Algo Accuracy: ", nltk.classify.accuracy(SGDClassifier_classifier, testing_set)*100)


# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC Algo Accuracy: ", nltk.classify.accuracy(SVC_classifier, testing_set)*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC Algo Accuracy: ", nltk.classify.accuracy(LinearSVC_classifier, testing_set)*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC Algo Accuracy: ", nltk.classify.accuracy(NuSVC_classifier, testing_set)*100)

voted_classifier = VoteClassifier(classifier,MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier,
                                  SGDClassifier_classifier, LinearSVC_classifier, NuSVC_classifier)

print("voted_classifier Algo Accuracy: ", nltk.classify.accuracy(voted_classifier, testing_set)*100)

# for i in range(5):
#     print("Classification: ", voted_classifier.classify(testing_set[i][0]), "\nConfidence: ", voted_classifier.confidence(testing_set[i][0]))


