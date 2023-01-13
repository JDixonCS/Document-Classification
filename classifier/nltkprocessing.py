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
with open("C:\\Users\\Predator\\Documents\\Document-Classification\\backend\\combined.txt", 'r') as f:
   rawtext = f.read()
   print(rawtext)
from nltk.tokenize import sent_tokenize
tokenized_text=sent_tokenize(rawtext)
print(tokenized_text)
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(rawtext)
print(tokenized_word)
##Creating a list of stop words and adding custom stopwords
stop_words = set(stopwords.words("english"))
##Creating a list of custom stopwords
new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown"]
stop_words = stop_words.union(new_words)
corpus = []
# Remove punctuations
rawtext = re.sub('[^a-zA-Z]', ' ', rawtext)
#Convert to lowercase
rawtext = rawtext.lower()
# remove special characters and digits
rawtext=re.sub("(\\d|\\W)+"," ",rawtext)

##Convert to list from string
rawtext = rawtext.split()
##Stemming
ps=PorterStemmer()
#Lemmatisation
lem = WordNetLemmatizer()
rawtext = [lem.lemmatize(word) for word in rawtext if not word in
            stop_words]
rawtext = " ".join(rawtext)
corpus.append(rawtext)
'''
fdist = FreqDist(tokenized_word)
print(fdist)
print(fdist.most_common(30))
'''
# Frequency Distribution Plot
fdist.plot(30,cumulative=False)
plt.show()