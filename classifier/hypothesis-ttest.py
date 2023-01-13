# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:43:56 2018

@author: phuongnh
"""
from math import gcd
import numpy as np
#import dataloader as dtl
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyClassifier
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from timeit import default_timer as timer

