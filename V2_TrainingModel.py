import pandas as pd
import numpy as np
import os
import nltk
import string
import inflect
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import ollama

# ------------------ NLTK downloads ------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ------------------ Text preprocessing ------------------
p = inflect.engine()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def text_lowercase(text):
    return text.lower()

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    filtered_text = [word for word in tokens if word.lower() not in stop_words]
    return filtered_text

def convert_number(text):
    words = text.split()
    new_words = [p.number_to_words(w) if w.isdigit() else w for w in words]
    return ' '.join(new_words)

# Combining functions into one big text filter (returns a string)
def textfilter(text):
    text = text_lowercase(text)
    text = convert_number(text)
    text = remove_stopwords(text)
    # Stem words or lemmatize if you want (currently not used)
    # text = stem_words(text)
    # text = lemma_words(text)
    return ' '.join(text)

# Optional: chunking text (not used in current code)
def chunk(text, chunk_size=100):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# ------------------ Embedding function ------------------
model_name = "mxbai-embed-large"

def embedd_simple(model, truedf, fakedf):
    embeddings = []
    labels = []
    batch_size = 32

    for df, label, desc in [(truedf, 1, "Embedding True Articles"),
                            (fakedf, 0, "Embedding Fake Articles")]:
        texts = [textfilter(t) for t in tqdm(df['text'], desc=desc)]
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Batches {desc}"):
            batch = texts[i:i+batch_size]
            batch_emb = [ollama.embed(model=model, input=t)["embeddings"] for t in batch]
            embeddings.extend(batch_emb)
            labels.extend([label] * len(batch_emb))

            # optional: save cache to avoid recomputing embeddings
            if len(embeddings) % 700 == 0:
                np.save("embeddings.npy", np.array(embeddings))
                np.save("labels.npy", np.array(labels))
    return np.array(embeddings), np.array(labels)

# ------------------ Load data ------------------
fakedf = pd.read_csv("fake.csv").iloc[0:3000,:]
truedf = pd.read_csv("true.csv").iloc[0:3000,:]

# ------------------ Preprocess all texts for alignment ------------------
all_texts = np.array([textfilter(t) for t in pd.concat([truedf['text'], fakedf['text']])])

# ------------------ Load or generate embeddings ------------------
if os.path.exists("embeddings.npy") and os.path.exists("labels.npy"):
    embeddings = np.load("embeddings.npy", allow_pickle=True)
    labels = np.load("labels.npy", allow_pickle=True)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    labels = labels.ravel()
    print("Embeddings and labels loaded from disk.")
else:
    print("Generating embeddings...")
    embeddings, labels = embedd_simple(model_name, truedf, fakedf)
    
    np.save("embeddings.npy", embeddings)
    np.save("labels.npy", labels)
    print("Embeddings and labels saved.")

# ------------------ Train-test split (keep texts aligned) ------------------
X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
    embeddings, labels, all_texts, test_size=0.2, random_state=42
)

# ------------------ Define hyperparameters ------------------
configs = [
    {"learning_rate": "adaptive", "eta0": 0.05, "max_it":100,"label": "adaptive 0.05 100"},
    # {"learning_rate": "adaptive", "eta0": 0.02, "max_it":200,"label": "adaptive 0.02 200"}, # tried config
    # {"learning_rate": "adaptive", "eta0": 0.05, "max_it":300,"label": "adaptive 0.05 300"}, # tried config
    # {"learning_rate": "adaptive", "eta0": 0.05, "max_it":50, "label": "adaptive 0.05 50"}   # tried config
]

# ------------------ Training ------------------
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
    for i in range(100):
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_train)
        loss = log_loss(y_train, y_prob)
        losses.append(loss)
        print(f"Iteration {i+1}, Log Loss: {loss:.4f}")
        # optional early stopping
        # if(loss <= 0.1):
        #     break
    
    all_losses[config["label"]] = losses

# ------------------ Plot training loss ------------------
for label, losses in all_losses.items():
    plt.plot(losses, label=label)

plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.title("SGDClassifier Training Loss")
plt.legend()
plt.show()

# ------------------ Evaluate on test set ------------------
y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)
correct = (y_pred == y_test).sum()
total = len(y_test)

print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Correct predictions: {correct}/{total}")
print(f"Wrong predictions: {total-correct}/{total}")

# ------------------ Print predictions with text and confidence ------------------
for text, pred, prob in zip(texts_test, y_pred, y_pred_prob):
    print(f"Prediction: {pred}, Confidence: {prob[pred]:.2f}")
    print("Text:", text[:100], "...\n")

# ------------------ Save the trained model ------------------
joblib.dump(clf, "sgd_classifier_model.joblib")
print("Model saved as sgd_classifier_model.joblib")
