import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import nltk
import string
import subprocess
import inflect
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
import subprocess
import json
import ollama
# Initialize Ollama client

# Use your local embedding model
model_name = "twine/mxbai-embed-xsmall-v1:latest"
#Remove stopwords from the text (not ncessary words "this that ...")
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower()
                     not in stop_words]
    return filtered_text

#Remove punctionation (modified to conserve !! and ??)
p = inflect.engine()
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation.replace('.','').replace('!','').replace('?',''))
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
    text=remove_stopwords(text)
    return ' '.join(text)

def chunk(text,chunk_size=100):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def embedd_simple(model, truedf, fakedf):
        
    embeddings = []
    labels = []
    batch_size = 32

    for df, label, desc in [(truedf, 1, "Embedding True Articles"),
                            (fakedf, 0, "Embedding Fake Articles")]:
        texts = [textfilter(t) for t in tqdm(df['text'], desc=desc)]

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Batches {desc}"):
            batch = texts[i:i+batch_size]
            batch_emb = [ollama.embed(model=model_name, input=t)["embeddings"] for t in batch]
            embeddings.extend(batch_emb)
            labels.extend([label] * len(batch_emb))

            # optional: save cache
            if len(embeddings) % 500 == 0:
                np.save("embeddings.npy", np.array(embeddings))
                np.save("labels.npy", np.array(labels))
    return embeddings,labels
    
    
     

#Reading the data(5000 row)
fakedf=pd.read_csv("fake.csv").iloc[0:3000,:]
truedf=pd.read_csv("true.csv").iloc[0:3000,:]

# embedding check if exists or not
if os.path.exists("embeddings.npy") and os.path.exists("labels.npy"):
    embeddings = np.load("embeddings.npy", allow_pickle=True)
    labels = np.load("labels.npy", allow_pickle=True)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    labels = labels.ravel()
    print("Embeddings and labels loaded from disk.")
else:
    print("Embeddings file not found. Generating embeddings...")
    embeddings, labels = embedd_simple("mxbai-embed-large",truedf,fakedf) # now embeddings are defined
    np.save("embeddings.npy", np.array(embeddings))
    np.save("labels.npy", np.array(labels))
    print("Embeddings and labels saved.")

# preparing the testing process

X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Define the hyperparameter settings you want to compare (testing different configurations for best fit)
configs = [
    {"learning_rate": "adaptive", "eta0": 0.05, "max_it":100,"label": "adaptive 0.05 100"}, #chose config 1
    # {"learning_rate": "adaptive", "eta0": 0.02, "max_it":200,"label": "adaptive 0.02 20"},
    # {"learning_rate": "adaptive", "eta0": 0.05, "max_it":300,"label": "adaptive 0.05 20"},
    # {"learning_rate": "adaptive", "eta0": 0.05, "max_it":50, "label": "adaptive 0.05 50"}
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
    for i in range(500):
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_train)
        loss = log_loss(y_train, y_prob)
        # if(loss<=0.1):
        #     break
        losses.append(loss)
        print("it number"+str(i))
    
    all_losses[config["label"]] = losses

# Plot all loss curves on the same figure
for label, losses in all_losses.items():
    plt.plot(losses, label=label)

plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.title("SGDClassifier Log Loss vs Iteration (Different Learning Rates)")
plt.legend()
plt.show()
