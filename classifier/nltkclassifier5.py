from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import tokenize
from operator import itemgetter
import math
from os import path
from PIL import Image
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
#% matplotlib inline
import re
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pandas as pd
# do the nlp stuff
text = open('C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\NIST_FULL\\2010-neg.txt').read()
#text = open('C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\NIST_FULL\\2010-pos.txt').read()
'''
text = open('C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\problems_in_china.txt').read()
text = open('C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\risk_factors.txt').read()
text = open('C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\testing.txt').read()
text = open('C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\transmission.txt').read()
'''
print(text)
lines = text #read all lines

#tokenized = nltk.word_tokenize(lines)'
sentences=nltk.sent_tokenize(lines)
#strings=word_tokenize(sentence)
stop_words = set(stopwords.words("english"))
##Creating a list of custom stopwords
new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown", "https",
             "version", "peer", "funder", "review", "holder", "doi", "license", "copyright", "preprint", "et", "al", "www"]
stop_words = stop_words.union(new_words)
sentences = text.lower()
sentences = re.sub('[^A-Za-z0-9]+', ' ', sentences)  #Remove punctuations
#Convert to lowercase
#lines = re.sub("(\\d|\\W)+"," ",text) #remove special characters and digits
sentences = re.sub('\W+',' ', sentences) # Remove special characters
sentences = re.sub('https?://(www.)?\w+\.\w+(/\w+)*/?', ' ', sentences) # remove hyperlinks
sentences = re.sub('@(\w+)', ' ', sentences) # remove mentions
alpha_num_re = re.compile("^[a-z0-9_.]+$")
list_pos = 0
cleaned_str = ''
for word in lines.split():
    if list_pos == 0:
        if alpha_num_re.match(word) and len(word) > 2:
            cleaned_str = word
        else:
            cleaned_str = ' '
    else:
        if alpha_num_re.match(word) and len(word) > 2:
            cleaned_str = cleaned_str + ' ' + word
        else:
            cleaned_str += ' '
    list_pos += 1
sentences = cleaned_str.split() ##Convert to list from string

print(sentences)
'''
nouns
is_noun_nn = lambda pos: pos[:2] == 'NN'
is_noun_nnp = lambda pos: pos[:3] == 'NNP'
is_noun_nns = lambda pos: pos[:4] == 'NNS'
is_noun_nnps = lambda pos: pos[:5] == 'NNPS'
'''
descriptive = []
for sentence in sentences:
    for (word, pos) in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
        if word not in stop_words:
            if (pos == 'NN'
                    or pos == 'NNP'
                    or pos == 'NNS'
                    or pos == 'NNPS'
                    or pos == 'JJ'
                    or pos == 'JJR'
                    or pos == 'JJS'
                    or pos == 'RB'
                    or pos == 'RBR'
                    or pos == 'RBS'
                    or pos == 'VB'
                    or pos == 'VBD'
                    or pos == 'VBG'
                    or pos == 'VBN'
                    or pos == 'VBP'
                    or pos == 'VBZ'):
                descriptive.append(word)
print(descriptive)
fdist = FreqDist(descriptive)
print(fdist)
print("Most frequent descriptive phrases for 2010-neg.txt")
#print("Most frequent descriptive phrases for 2010-pos.txt")
print(fdist.most_common(30))
fdist.plot(30,cumulative=False)
plt.show()