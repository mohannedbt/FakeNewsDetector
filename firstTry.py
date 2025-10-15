import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import string
import re
import inflect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

#Remove stopwords from the text (not ncessary words "this that ...")
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower()
                     not in stop_words]
    return filtered_text

#Remove punctionation (although it's a bit necessary for fake or not fake news)
p = inflect.engine()
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

#lowercasing the string to remove some of the redundant words
def text_lowercase(text):
    return text.lower()

#applying stemming (plural to single ,past to present)
stemmer = PorterStemmer()
def stem_words(text):
    word_tokens = word_tokenize(text)
    stems = [stemmer.stem(word) for word in word_tokens]
    return stems
#leminization (making the word back to its root)

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
def lemma_words(text):
    word_tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word) for word in word_tokens]
    return lemmas

# converting numbers inside the string to reduce complexity
def convert_number(text):
    temp_str = text.split()
    new_string = []

    for word in temp_str:
        if word.isdigit():
            new_string.append(p.number_to_words(word))
        else:
            new_string.append(word)

    return ' '.join(new_string)

#combining these functions into one big text filter (returns a string)

def textfilter(text):
    text=text_lowercase(text)
    text=convert_number(text)
    text=remove_punctuation(text)
    final=stem_words(text)
    return ''.join(final)

#Familiarizing with the data
fakedf=pd.read_csv("fake.csv")

truedf=pd.read_csv("true.csv")

print(fakedf.columns)

print(truedf.columns) 

fakenadf=fakedf.isna()

truenadf=truedf.isna()

sns.heatmap(fakenadf)

sns.heatmap(truenadf)


plt.show()
