import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import nltk
import string
import math
import inflect
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
model = SentenceTransformer('all-MiniLM-L6-v2')  # or any embedding model

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

def chunk(text,chunk_size=100):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def embedd(chunk_size=100):
    embeddings = []
    labels = []

    # Loop over datasets
    for df, label, desc in [(truedf, 1, "Embedding True Articles"),
                            (fakedf, 0, "Embedding Fake Articles")]:
        for index, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            cleaned = textfilter(row['text'])
            chunks = chunk(cleaned, chunk_size)
            # Show progress for chunks
            chunk_embeds = model.encode(chunks, batch_size=chunk_size, show_progress_bar=True)
            article_embed = np.mean(chunk_embeds, axis=0)
            embeddings.append(article_embed)
            labels.append(label)

    return embeddings, labels
    
     

#Familiarizing with the data
fakedf=pd.read_csv("fake.csv").iloc[0:2000,:]

truedf=pd.read_csv("true.csv").iloc[0:2000,:]
#
# test for data null values and other test 
#
# print(fakedf.columns)

# print(truedf.columns) 

# fakenadf=fakedf.isna()

# truenadf=truedf.isna()

# sns.heatmap(fakenadf)

# sns.heatmap(truenadf)

# numb=math.floor(100*random.randint(0,1))
# text=truedf['text'][numb]
# print(text[0:50])
# new_text=textfilter(text)
# print(new_text[0:50])
if os.path.exists("embeddings.npy") and os.path.exists("labels.npy"):
    embeddings = np.load("embeddings.npy", allow_pickle=True)
    labels = np.load("labels.npy", allow_pickle=True)
    print("Embeddings and labels loaded from disk.")
else:
    print("Embeddings file not found. Generating embeddings...")
    embeddings, labels = embedd()  # now embeddings are defined
    np.save("embeddings.npy", np.array(embeddings))
    np.save("labels.npy", np.array(labels))
    print("Embeddings and labels saved.")

X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

# Define the hyperparameter settings you want to compare
configs = [
    {"learning_rate": "adaptive", "eta0": 0.02, "max_it":1,"label": "constant 0.01 1"},
    {"learning_rate": "adaptive", "eta0": 0.02, "max_it":10,"label": "adaptive 0.01 10"},
    {"learning_rate": "adaptive", "eta0": 0.02, "max_it":20,"label": "adaptive 0.01 1"},
    {"learning_rate": "adaptive", "eta0": 0.05, "max_it":50, "label": "invscaling 0.05 1"}
]

# Store all loss curves
all_losses = {}

for config in configs:
    clf = SGDClassifier(
        loss='log_loss',
        learning_rate=config["learning_rate"],
        eta0=config["eta0"],
        max_iter=config["max_it"],
        warm_start=True
    )
    
    losses = []
    for i in range(1000):
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_train)
        loss = log_loss(y_train, y_prob)
        # if(loss<=0.1):
        #     break
        losses.append(loss)
    
    all_losses[config["label"]] = losses

# Plot all loss curves on the same figure
for label, losses in all_losses.items():
    plt.plot(losses, label=label)

plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.title("SGDClassifier Log Loss vs Iteration (Different Learning Rates)")
plt.legend()
plt.show()
