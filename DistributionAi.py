# fake_news_ai.py
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import inflect
import ollama

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')

p = inflect.engine()

def textfilter(text):
    # Lowercase
    text = text.lower()
    # Convert numbers to words
    text = ' '.join([p.number_to_words(w) if w.isdigit() else w for w in text.split()])
    # Remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered = [w for w in tokens if w.lower() not in stop_words]
    return ' '.join(filtered)

# Load trained model
clf = joblib.load("sgd_classifier_model.joblib")
MODEL_NAME = "mxbai-embed-large"

def predict_news(text):
    """
    Input: raw news text
    Output: label ('Real'/'Fake') and confidence %
    """
    filtered_text = textfilter(text)
    # Get embedding from LLM
    embedding = ollama.embed(model=MODEL_NAME, input=filtered_text)["embeddings"]
    embedding = np.array(embedding).reshape(1, -1)
    
    pred = clf.predict(embedding)[0]
    prob = clf.predict_proba(embedding)[0]
    confidence = prob[pred] * 100
    if (prob[pred]>0.65):
        label = "Real" if pred==1 else "Fake"
    else:
        label = "uncertain needs human intervention"
    return label, confidence
