# fake_news_ai.py
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import inflect
from sentence_transformers import SentenceTransformer
import os

# ------------------ NLTK setup ------------------
nltk.download('punkt')
nltk.download('stopwords')

p = inflect.engine()

# ------------------ Text preprocessing ------------------
def textfilter(text):
    """
    Lowercase, convert numbers to words, remove stopwords.
    """
    text = text.lower()
    # Convert numbers to words
    text = ' '.join([p.number_to_words(w) if w.isdigit() else w for w in text.split()])
    # Remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered = [w for w in tokens if w.lower() not in stop_words]
    return ' '.join(filtered)

# ------------------ Load trained ML model ------------------
# Replace this path with your V2 model path
MODEL_PATH = os.path.join("model_results_<timestamp>", "best_model.joblib")
clf = joblib.load(MODEL_PATH)

# ------------------ Load SentenceTransformer embedding model ------------------
EMBEDDER_MODEL = "all-MiniLM-L6-v2"  # same as used in V2
embedder = SentenceTransformer(EMBEDDER_MODEL, device="cuda")

# ------------------ Prediction function ------------------
def predict_news(text):
    """
    Input: raw news text
    Output: label ('Real'/'Fake') and confidence %
    """
    filtered_text = textfilter(text)
    # Generate embedding
    embedding = embedder.encode([filtered_text], convert_to_numpy=True)
    
    # Predict using trained classifier
    pred = clf.predict(embedding)[0]
    prob = clf.predict_proba(embedding)[0]
    confidence = prob[pred] * 100

    if prob[pred] > 0.65:
        label = "Real" if pred == 1 else "Fake"
    else:
        label = "Uncertain - needs human intervention"

    return label, confidence


