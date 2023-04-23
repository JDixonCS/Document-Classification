from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import time
import itertools 
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from subsample import set_iteration_labels
import argparse as args
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, recall_score, precision_score

class ClassifyTasks:
    tfidf_vect = TfidfVectorizer(ngram_range=(1,2))
    log = LogisticRegression(penalty='l2',random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
    DCT = DecisionTreeClassifier()
    NB = MultinomialNB()
    xgb = XGBClassifier()
    rfc = RandomForestClassifier(n_estimators=1000, random_state=0)
    def __init__(self, df):
        self.df = df
        '''
        self.models = models
        self.classifier_names = classifier_names
        self.train_sizes_num = train_sizes_num
        self.test_sizes_num = test_sizes_num
        self.rounds = rounds
        '''
        self.models = models
        self.classifier_names = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'XGBoost', 'Random Forest']
        self.test_sizes_num = [0.17, 0.33, 0.50, 0.67, 0.83]
        self.train_sizes_num = [0.83, 0.67, 0.50, 0.33, 0.17]
        self.rounds = 5
        
    def extract_features(self):
        x_tfidf = self.tfidf_vect.fit_transform(self.df["lemmatized"])
        # add other feature extraction methods here
        return x_tfidf

    def get_labels(df):
        y_labels = df["label"]
        return y_labels

    def get_train_values():
        train_values = np.array([0.16, 0.33, 0.50, 0.67, 0.83])
        return train_values

    def train_subsample(x_tfidf, y_labels, train_values):
        subsamples = {}
        for i in train_values:
            x_train, x_test, y_train, y_test = train_test_split(x_tfidf, y_labels, train_size=i, stratify=y_labels)
            subsamples[i] = {
                "x_train": x_train,
                "x_test": x_test,
                "y_train": y_train,
                "y_test": y_test
            }
        return subsamples

    @classmethod
    def train_classifier(cls, classifier, x_train, y_train, x_test, y_test, train_values, labels):
        start = time.time()
        model = classifier.fit(x_train, y_train)
        probs = model.predict_proba(x_test)[:, 1]
        prediction = model.predict(x_test)
        f1 = f1_score(prediction, y_test)
        rocauc = roc_auc_score(y_test, prediction)
        recall = recall_score(y_test, prediction)
        precision = precision_score(y_test, prediction)
        accuracy = accuracy_score(y_test, prediction)
        confusion = confusion_matrix(y_test, prediction)
        report = classification_report(y_test, prediction)
        end = time.time()
        print(f"=== {type(classifier).__name__} with TfidfVectorizer Imbalanced - {labels} {train_values}")
        print(f"{type(classifier).__name__} F1-score {f1*100}")
        print(f"{type(classifier).__name__} ROCAUC score: {rocauc*100}")
        print(f"{type(classifier).__name__} Recall score: {recall*100}")
        print(f"{type(classifier).__name__} Precision Score: {precision*100}")
        print(f"{type(classifier).__name__} Confusion Matrix {confusion} /n")
        print(f"{type(classifier).__name__} Classification {report} /n")
        print(f"{type(classifier).__name__} Accuracy Score {accuracy*100}")
        print(f"Execution Time for {type(classifier).__name__} Imbalanced: {end - start} seconds")

        return probs, f1, rocauc, recall, precision, accuracy

    @staticmethod
    def initialize_results_dict(dfs, models):
        results_dict = {}
        for df_name, df in dfs.items():
            results_dict[df_name] = {}
            for model in models:
                results_dict[df_name][model] = {
                    'probs': [],
                    'f1': [],
                    'rocauc': [],
                    'recall': [],
                    'precision': [],
                    'accuracy': []
                }
        return results_dict

    def create_csv_data(self, precision_csv_num, recall_csv_num, , auc_csv_num, accuracy_csv_num):
            year = [args.group for t in range(25)]
            sampling = ['Imbalanced' for t in range(25)]
            technique = ['Imbalanced' for t in range(25)]
            precision_csv = list(itertools.chain(*precision_csv_num))
            recall_csv = list(itertools.chain(*recall_csv_num))
            auc_csv = list(itertools.chain(*auc_csv_num))
            accuracy_csv = list(itertools.chain(accuracy_csv_num))
            classifier_csv = list(itertools.islice(itertools.cycle(self.classifier_names), self.rounds5))
            test_size_csv = [a for b in self.test_sizes_num for a in (b,)*5]
            train_size_csv = [c for d in self.train_sizes_num for c in (d,)*5]
            split_csv = ['1' for t in range(25)]
            train_csv = ['5%' for t in range(25)]
            csv_data = [year, sampling, technique, classifier_csv, test_size_csv, train_size_csv, split_csv, train_csv, precision_csv, recall_csv, auc_csv, accuracy_csv]
            return csv_data

    def export_data_to_csv(self, csv_data, filename):
        with open(filename, 'w', newline='') as file:
            write = csv.writer(file)
            write.writerow(("Year", "Sampling", "Technique", "Classifier", "Test Split Size", "Train Split Size", "Iteration", "Training Data", "Precision", "Recall", "AUC", "Accuracy"))
            write.writerows(itertools.zip_longest(*csv_data, fillvalue=''))


'''
    def create_csv_data(self, precision_csv_num, recall_csv_num, auc_csv_num, accuracy_csv_num):
        year = [args.group for t in range(25)]
        sampling = ['Imbalanced' for t in range(25)]
        technique = ['Imbalanced' for t in range(25)]
        precision_csv = list(itertools.chain(*precision_csv_num))
        recall_csv = list(itertools.chain(*recall_csv_num))
        auc_csv = list(itertools.chain(*auc_csv_num))
        accuracy_csv = list(itertools.chain(*accuracy_csv_num))
        classifier_csv = list(itertools.islice(itertools.cycle(self.classifier_names), self.rounds*5))
        test_size_csv = [a for b in self.test_sizes_num for a in (b,)*5]
        train_size_csv = [c for d in self.train_sizes_num for c in (d,)*5]
        split_csv = ['1' for t in range(25)]
        train_csv = ['5%' for t in range(25)]
        csv_data = [year, sampling, technique, classifier_csv, test_size_csv, train_size_csv, split_csv, train_csv, precision_csv, recall_csv, auc_csv, accuracy_csv]
        return csv_data

    def export_data_to_csv(self, csv_data, filename):
        with open(filename, 'w', newline='') as file:
            write = csv.writer(file)
            write.writerow(("Year", "Sampling", "Technique", "Classifier", "Test Split Size", "Train Split Size", "Iteration", "Training Data", "Precision", "Recall", "AUC", "Accuracy"))
            write.writerows(itertools.zip_longest(*csv_data, fillvalue=''))
'''