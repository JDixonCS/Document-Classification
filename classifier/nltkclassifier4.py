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
'''
text_path = input('Enter file name: ')

text = open(text_path).read()
'''
text = open('C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\NIST_FULL\\2010-pos.txt').read()
'''
text = open('C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\NIST_FULL\\2010-pos.txt').read()
text = open('C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\NIST_FULL\\2012-neg.txt').read()
text = open('C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\NIST_FULL\\2012-pos.txt').read()
'''
'''
text = open('C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\problems_in_china.txt').read()
text = open('C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\risk_factors.txt').read()
text = open('C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\testing.txt').read()
text = open('C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\transmission.txt').read()
'''
print(text)
lines = text #read all lines
#sentences = nltk.sent_tokenize(lines) #tokenize sentences
#nouns = [] #empty to array to hold all nouns


# do the nlp stuff
#tokenized = nltk.word_tokenize(lines)
tokenized_text=sent_tokenize(lines)
tokenized=word_tokenize(lines)
stop_words = set(stopwords.words("english"))
##Creating a list of custom stopwords
new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown", "https",
             "version", "peer", "funder", "review", "holder", "doi", "license", "copyright", "preprint", "et", "al", "www"]
stop_words = stop_words.union(new_words)
lines = text.lower()
lines = re.sub('[^A-Za-z0-9]+', ' ', lines)  #Remove punctuations
#Convert to lowercase
#lines = re.sub("(\\d|\\W)+"," ",text) #remove special characters and digits
lines = re.sub('\W+',' ', lines) # Remove special characters
lines = re.sub('https?://(www.)?\w+\.\w+(/\w+)*/?', ' ', lines) # remove hyperlinks
lines = re.sub('@(\w+)', ' ', lines) # remove mentions
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
lines = cleaned_str.split() ##Convert to list from string
#ps=PorterStemmer() ##Stemming
#lem = WordNetLemmatizer() #Lemmatisation
#is_noun = lambda pos: pos[:2] == 'NN'
#lines = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos) if not word in stop_words]
#lines = [token for token, pos in nltk.pos_tag(word_tokenize(lines)) if pos.startswith('N')]
#lines = [lem.lemmatize(word) for word in text if not word in  stop_words]
#lines =" ".join(lines)
#corpus.append(lines)
print(lines)
nouns = []
is_noun_nn = lambda pos: pos[:2] == 'NN'
is_noun_nnp = lambda pos: pos[:3] == 'NNP'
#is_noun_nns = lambda pos: pos[:4] == 'NNS'
#is_noun_nnps = lambda pos: pos[:5] == 'NNPS'
noun_nn = [word for (word, pos) in nltk.pos_tag(lines) if is_noun_nn(pos) if not word in stop_words]
noun_nnp = [word for (word, pos) in nltk.pos_tag(lines) if is_noun_nnp(pos) if not word in stop_words]
#noun_nns = [word for (word, pos) in nltk.pos_tag(lines) if is_noun_nns(pos) if not word in stop_words]
#noun_nnps = [word for (word, pos) in nltk.pos_tag(lines) if is_noun_nnps(pos) if not word in stop_words]
nouns = is_noun_nn + is_noun_nnp
#nouns = nouns.append(is_noun_nns)
#nouns = nouns.append(is_noun_nnps)
print(nouns)
fdist = FreqDist(nouns)
print(fdist)
print("Most frequent descriptive phrases for 2010-neg.txt")
print(fdist.most_common(30))
fdist.plot(30,cumulative=False)
plt.show()

'''
import pandas as pd
from nltk import FreqDist, word_tokenize

#df = pd.read_csv('./SECParse3.csv')
#words = word_tokenize(' '.join([line for line in df['text'].to_numpy()]))

common = FreqDist(nouns).most_common(100)
pd.DataFrame(common, columns=['word', 'count']).to_csv('immune.csv', index=False)
'''
'''
#Word cloud

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
#% matplotlib inline
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stop_words,
                          max_words=100,
                          max_font_size=50,
                          random_state=42
                         ).generate(str(lines))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)

from sklearn.feature_extraction.text import CountVectorizer
import re
cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
X=cv.fit_transform(lines)
print(X)

cvlist =list(cv.vocabulary_.keys())[:10]
print(cvlist)

#Most frequently occuring words
def get_top_n_words(lines, n=None):
    vec = CountVectorizer().fit(lines)
    #print(vec)
    bag_of_words = vec.transform(lines)
    #print(bag_of_words)
    sum_words = bag_of_words.sum(axis=0)
    #print(sum_words)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                   vec.vocabulary_.items()]
    #print(words_freq)
    words_freq =sorted(words_freq, key = lambda x: x[1],
                       reverse=True)
    return words_freq[:n]
#Convert most freq words to dataframe for plotting bar plot
top_words = get_top_n_words(nouns, n=20)
print(top_words)
top_df = pd.DataFrame(top_words)
top_df.columns=["Word", "Freq"]

#Barplot of most freq words
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
g = sns.barplot(x="Word", y="Freq", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=30)

#Most frequently occuring Bi-grams
def get_top_n2_words(lines, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),
            max_features=2000).fit(lines)
    #print(vec1)
    bag_of_words = vec1.transform(lines)
    #print(bag_of_words)
    sum_words = bag_of_words.sum(axis=0)
    #print(sum_words)

    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1],
                reverse=True)
    return words_freq[:n]
    #print(words_freq)
    top2_words = get_top_n2_words(lines, n=20)
    print(top2_words)
    top2_df = pd.DataFrame(top2_words)
    top2_df.columns=["Bi-gram", "Freq"]
    #print(top2_df)
    #Barplot of most freq Bi-grams
    import seaborn as sns
    sns.set(rc={'figure.figsize':(13,8)})
    h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
    h.set_xticklabels(h.get_xticklabels(), rotation=45)

#Most frequently occuring Tri-grams
def get_top_n3_words(lines, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3),
           max_features=2000).fit(lines)
    #print(vec1)
    bag_of_words = vec1.transform(lines)
    #print(bag_of_words)
    sum_words = bag_of_words.sum(axis=0)
    #print(sum_words)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1],
                reverse=True)
    #print(words_freq)
    return words_freq[:n]
    top3_words = get_top_n3_words(lines, n=20)
    print(top3_words)
    top3_df = pd.DataFrame(top3_words)
    #print(top3_df)
    top3_df.columns=["Tri-gram", "Freq"]
    #print(top3_df)
    #Barplot of most freq Tri-grams
    import seaborn as sns
    sns.set(rc={'figure.figsize':(13,8)})
    j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df)
    j.set_xticklabels(j.get_xticklabels(), rotation=45)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(X)
# get feature names
feature_names = cv.get_feature_names()

# fetch document for which keywords needs to be extracted
doc = nouns[532]

# generate tf-idf for the given document
tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

# Function for sorting tf_idf in descending order
from scipy.sparse import coo_matrix


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

'''
'''
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


# sort the tf-idf vectors by descending order of scores
sorted_items = sort_coo(tf_idf_vector.tocoo())
# extract only the top n; n here is 10
keywords = extract_topn_from_vector(feature_names, sorted_items, 100)

# now print the results
print("\nAbstract:")
print(doc)
print("\nKeywords:")
for k in keywords:
    print(k, keywords[k])
'''