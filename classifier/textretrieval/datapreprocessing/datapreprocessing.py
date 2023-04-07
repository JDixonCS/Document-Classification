import string
import re
import nltk

class DataPreprocessing:
    
    def __init__(self, stopwords):
        self.stopwords = stopwords
        
    def remove_punc(self, text):
        punc_set = string.punctuation
        return "".join([x.lower() for x in text if x not in punc_set])
    
    def tokenize(self, text):
        return re.split("/W+",text)
    
    def remove_stopwords(self, tokenized_words):
        return [word for word in tokenized_words if word not in self.stopwords]
    
    def lemmatize(self, tokenized_text):
        wnl = nltk.wordnet.WordNetLemmatizer()
        return [wnl.lemmatize(word) for word in tokenized_text]
    
    def preprocess(self, df):
        df['no_punc'] = df['sentence'].apply(lambda z: self.remove_punc(z))
        df['tokenized_Data'] = df['no_punc'].apply(lambda z: self.tokenize(z))
        df['no_stop'] = df["tokenized_Data"].apply(lambda z: self.remove_stopwords(z))
        df['lemmatized'] = df['no_stop'].apply(lambda z: self.lemmatize(z))
        df['lemmatized'] = [" ".join(review) for review in df['lemmatized'].values]
        return df