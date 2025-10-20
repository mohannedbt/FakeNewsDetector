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
model_name = "twine/mxbai-embed-xsmall-v1:latest"

def embedd_simple(model, truedf, fakedf, batch_size=32, cache_dir="cache"):
    """
    Compute embeddings for true and fake articles with auto-resume support.
    Saves progress every 100 embeddings and resumes automatically after a crash.
    """
    os.makedirs(cache_dir, exist_ok=True)

    embeddings_path = os.path.join(cache_dir, "embeddings.npy")
    labels_path = os.path.join(cache_dir, "labels.npy")

    # --- Load cached progress if exists ---
    if os.path.exists(embeddings_path) and os.path.exists(labels_path):
        embeddings = np.load(embeddings_path, allow_pickle=True).tolist()
        labels = np.load(labels_path, allow_pickle=True).tolist()
        print(f"Resuming from {len(embeddings)} cached embeddings.")
    else:
        embeddings, labels = [], []
        print("Starting new embedding session.")

    total_done = len(embeddings)

    # --- Combine both datasets with their labels ---
    data = [
        (truedf, 1, "True Articles"),
        (fakedf, 0, "Fake Articles")
    ]

    for df, label, desc in data:
        texts = [textfilter(t) for t in df["text"]]
        print(f"\nProcessing {desc}: {len(texts)} samples")

        # Count already done for this label
        done_for_label = sum(1 for l in labels if l == label)
        start_index = done_for_label

        if start_index >= len(texts):
            print(f" {desc} already embedded, skipping.")
            continue

        for i in tqdm(range(start_index, len(texts), batch_size), desc=f"Batches {desc}"):
            batch = texts[i:i + batch_size]

            try:
                response = ollama.embed(model=model, input=batch)
                batch_emb = response["embeddings"]
            except Exception:
                batch_emb = [ollama.embed(model=model, input=t)["embeddings"] for t in batch]

            embeddings.extend(batch_emb)
            labels.extend([label] * len(batch_emb))

            # Save progress every 100 embeddings
            if len(embeddings) % 100 == 0:
                np.save(embeddings_path, np.array(embeddings, dtype=object))
                np.save(labels_path, np.array(labels, dtype=object))
                print(f" Progress saved: {len(embeddings)} total embeddings")

    # Final save
    np.save(os.path.join(cache_dir, "embeddings_final.npy"), np.array(embeddings, dtype=object))
    np.save(os.path.join(cache_dir, "labels_final.npy"), np.array(labels, dtype=object))
    print(f"\n Finished embedding! Total: {len(embeddings)} samples.")

    return np.array(embeddings, dtype=object), np.array(labels, dtype=object)

# ------------------ Load data ------------------
fakedf = pd.read_csv("fake.csv").iloc[0:20000,:]
truedf = pd.read_csv("true.csv").iloc[0:20000,:]

# ------------------ Preprocess all texts for alignment ------------------
all_texts = np.array([textfilter(t) for t in pd.concat([truedf['text'], fakedf['text']])])

# ------------------ Load or generate embeddings ------------------
if os.path.exists(os.path.join("cache", "embeddings_final.npy")) and os.path.exists(os.path.join("cache", "labels_final.npy")):
    embeddings = np.load((os.path.join("cache", "embeddings_final.npy")), allow_pickle=True)
    labels = np.load((os.path.join("cache", "labels_final.npy")), allow_pickle=True)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    labels = labels.ravel()
    labels = np.array(labels, dtype=np.int32)
    print("label is raveled")
    print("Embeddings and labels loaded from disk.")
else:
    print("Generating embeddings...")
    embeddings, labels = embedd_simple(model_name, truedf, fakedf)
    print("Embeddings and labels generated and saved.")
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    labels = labels.ravel()
    labels = np.array(labels, dtype=np.int32)
    

# ------------------ Train-test split (keep texts aligned) ------------------
print("start test spliting")
X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
    embeddings, labels, all_texts, test_size=0.2, random_state=42
)
print("test splitting successful")
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
    print("start training process")

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
joblib.dump(clf, "sgd_classifier_model1.joblib")
print("Model saved as sgd_classifier_model.joblib")
