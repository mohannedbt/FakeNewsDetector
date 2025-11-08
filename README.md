FakeNewsDetector

FakeNewsDetector is an AI-powered tool that classifies news articles as real or fake. It leverages text preprocessing, embeddings, and machine learning models, with both a minimal web interface and programmatic API for predictions.

The project has evolved through multiple versions:

V1: Initial prototype with basic preprocessing.

V2: Uses SGDClassifier with semantic embeddings (Ollama).

V3: Advanced deep learning model using TensorFlow/Keras with local Ollama embeddings.

🚀 Features

Text preprocessing: Lowercasing, stopword removal, number-to-word conversion.

Embeddings: Ollama embeddings for semantic representation (local).

Models:

V2: SGDClassifier for classical ML detection.

V3: Neural network for improved detection accuracy.

UI & API:

Flask interface for web predictions.

FastAPI server for programmatic usage.

Training visualization: Log-loss curves and metrics for monitoring training.

Experimentation: Multiple model versions for iterative improvement.

📂 Folder Structure
Project
│   DistributionAi.py
│   embeddings.npy
│   Fake.csv
│   labels.npy
│   README.md
│   Server.py
│   sgd_classifier_model.joblib
│   True.csv
│   V1_model.py
│   V2_TrainingModel.py
│   V3_TrainingModel.py
│
├── templates
│       index.html
└── docs

🔑 Key Files

V1_model.py – Early version, basic preprocessing and training.

V2_TrainingModel.py – Preprocesses text, generates Ollama embeddings, trains SGDClassifier.

V3_TrainingModel.py – Preprocesses text, uses local Ollama embeddings and trains deep learning model with TensorFlow.

DistributionAi.py – Loads a trained model (V2 or V3) and makes predictions.

Server.py – Runs FastAPI server for API-based predictions.

sgd_classifier_model.joblib – Trained V2 SGDClassifier model file.

embeddings.npy – Precomputed embeddings for fast model training.

⚙️ Getting Started
1. Install dependencies:
pip install pandas numpy nltk inflect tqdm scikit-learn matplotlib joblib ollama fastapi uvicorn flask tensorflow

2. Prepare datasets

Place Fake.csv and True.csv in the project root.

3. Generate embeddings & train model

V2 (SGDClassifier):

python V2_TrainingModel.py


V3 (Neural Network with TensorFlow):

python V3_TrainingModel.py


V3 uses local Ollama embeddings and a deep learning model for higher accuracy.

4. Run the server
python Server.py

5. Make predictions programmatically
from DistributionAi import predict_article

result, confidence = predict_article("Some news text")
print(result, confidence)

💡 Notes

Embeddings are stored in embeddings.npy to avoid recomputation.

Current dataset contains ~3,000 articles per class. Increasing dataset size may improve accuracy.

Multiple versions are provided to track progress and experiments.

V3 is in active development with deep learning.

Using local Ollama embeddings ensures faster and offline embedding generation.

📝 Future Improvements

Expand the dataset for better generalization.

Fine-tune deep learning model hyperparameters in V3.

Add more advanced text preprocessing (e.g., named entity recognition, contextual embeddings).

Integrate real-time web scraping for live news detection.
