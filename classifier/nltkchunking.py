from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk import RegexpParser
from nltk import Tree
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import re
from nltk.probability import FreqDist

text = open('C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\NIST_FULL\\2010-pos.txt').read()

def get_noun_phrases(text):
    pos = pos_tag(word_tokenize(text))
    count = 0
    half_chunk = ""
    for word, tag in pos:
        if re.match(r"NN.*", tag):
            count+=1
            if count>=1:
                half_chunk = half_chunk + word + " "
        else:
            half_chunk = half_chunk+"---"
            count = 0
    half_chunk = re.sub(r"-+","?",half_chunk).split("?")
    half_chunk = [x.strip() for x in half_chunk if x!=""]
    return half_chunk

print(get_noun_phrases(text))