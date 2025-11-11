import pandas as pd
import numpy as np
import os
import nltk
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, log_loss, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
)
from datetime import datetime
import os
import inflect
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sentence_transformers import SentenceTransformer

# Use GPU
model_name = "all-MiniLM-L6-v2"  # or any other SentenceTransformer model
embedder = SentenceTransformer(model_name, device="cuda")  # automatically uses GPU

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
def embedd_simple_st(model, truedf, fakedf, batch_size=32, cache_dir="cache"):
    import os, numpy as np
    os.makedirs(cache_dir, exist_ok=True)
    embeddings_path = os.path.join(cache_dir, "embeddings.npy")
    labels_path = os.path.join(cache_dir, "labels.npy")

    if os.path.exists(embeddings_path) and os.path.exists(labels_path):
        embeddings = np.load(embeddings_path, allow_pickle=True)
        labels = np.load(labels_path, allow_pickle=True)
        print(f"Resuming from {len(embeddings)} cached embeddings.")
    else:
        embeddings, labels = [], []

    data = [(truedf, 1), (fakedf, 0)]
    for df, label in data:
        texts = [textfilter(t) for t in df["text"]]
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            batch_emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            embeddings.extend(batch_emb)
            labels.extend([label]*len(batch_emb))

            # Save every 500 embeddings
            if len(embeddings) % 500 == 0:
                np.save(embeddings_path, np.array(embeddings))
                np.save(labels_path, np.array(labels))
                print(f"Saved {len(embeddings)} embeddings.")

    # Final save
    np.save(os.path.join(cache_dir, "embeddings_final.npy"), np.array(embeddings))
    np.save(os.path.join(cache_dir, "labels_final.npy"), np.array(labels))
    return np.array(embeddings), np.array(labels)

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
    embeddings, labels = embedd_simple_st(embedder, truedf, fakedf)
    print("Embeddings and labels generated and saved.")
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    labels = labels.ravel()
    labels = np.array(labels, dtype=np.int32)


# ------------------ Setup ------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"model_results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# ------------------ Split ------------------
print(" Splitting dataset ...")
X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
    embeddings, labels, all_texts, test_size=0.2, random_state=42, stratify=labels
)
print(" Split done. Train:", len(X_train), "Test:", len(X_test))

# ------------------ Define hyperparameter grid ------------------
param_grid = {
    "learning_rate": ["adaptive", "optimal"],
    "eta0": [0.001, 0.01, 0.05, 0.1],
    "max_iter": [100, 200, 300],
    "alpha": [0.0001, 0.001, 0.01]
}

print(" Starting hyperparameter search ...")
clf_base = SGDClassifier(loss="log_loss", random_state=42)
grid_search = GridSearchCV(
    clf_base, param_grid, scoring="accuracy", cv=5, n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f" Best parameters: {best_params}")
print(f" Best CV accuracy: {best_score:.4f}")

# ------------------ Train best model ------------------
best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)

# Save model
model_path = os.path.join(output_dir, "best_model.joblib")
joblib.dump(best_clf, model_path)
print(f" Model saved to: {model_path}")

# ------------------ Evaluate ------------------
y_pred = best_clf.predict(X_test)
y_pred_prob = best_clf.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted"
)
try:
    roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class="ovr")
except:
    roc_auc = None

print("\n Evaluation Metrics")
print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1-score : {f1:.3f}")
if roc_auc:
    print(f"ROC-AUC  : {roc_auc:.3f}")

print("\ Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# ------------------ Confusion Matrix ------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# ------------------ ROC Curve ------------------
try:
    RocCurveDisplay.from_estimator(best_clf, X_test, y_test)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()
except:
    pass

# ------------------ Save metrics ------------------
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "cv_best_score": best_score,
    **best_params
}
metrics_path = os.path.join(output_dir, "metrics.csv")
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
print(f" Metrics saved to: {metrics_path}")

# ------------------ Save predictions ------------------
preds_df = pd.DataFrame({
    "text": texts_test,
    "true_label": y_test,
    "pred_label": y_pred,
    "confidence": np.max(y_pred_prob, axis=1)
})
preds_path = os.path.join(output_dir, "predictions.csv")
preds_df.to_csv(preds_path, index=False)
print(f" Predictions saved to: {preds_path}")

print("\n All results saved in folder:", output_dir)

