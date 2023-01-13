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

f = open("C:\\Users\\Predator\\Documents\\Document-Classification\\backend\\combined.txt", 'r')
rawtext = f.read()
print(rawtext)
print(type(rawtext), "\n")
# Sent tokenize
tokenized_text=sent_tokenize(rawtext)
print(tokenized_text)
# Tokenize the word
tokens = word_tokenize(rawtext)
print(type(tokens), "\n")
words = [w.lower() for w in tokens]
print(type(words), "\n")
vocab = sorted(set(words))
print(type(vocab), "\n")
#vocab.append('blog')
#rawtext.append('blog')
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
(wordlist)
#abstract = [w for w in wordlist if re.search('^[aA][bB][sS][tT][rR][aA][cC][tT](.*)', w)]
#print (abstract)
wsj = sorted(set(nltk.corpus.treebank.words()))
#introduction = [w for w in wordlist if re.search('^[iI][nN][tT][rR][oO][dD][uU][cC][tT][iI][oO][nN](.*)', w)]
raw = open('')