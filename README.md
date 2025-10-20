# FakeNewsDetector

**FakeNewsDetector** is an AI-powered tool that classifies news articles as **real** or **fake**. It combines **text preprocessing, embedding generation**, and a **machine learning classifier**, with a minimal **web interface** for predictions.

------

### Folder Structure

```
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
│
├── templates
│       index.html
└── docs
```

------

### Key Files

- `V2_TrainingModel.py` – Preprocesses text, generates embeddings, and trains the model.
- `DistributionAi.py` – Loads the trained model and makes predictions.
- `Server.py` – Runs a FastAPI server for programmatic predictions.
- `sgd_classifier_model.joblib` – Trained model file.

------

### Features

- Preprocessing: lowercasing, stopwords removal, number-to-word conversion.
- Embeddings: uses Ollama embeddings for semantic representation.
- Model: SGDClassifier for fake news detection.
- UI & API: Flask interface and FastAPI server for predictions.
- Training visualization: plots log-loss curves.

------

### Getting Started

1. **Install dependencies:**

```
pip install pandas numpy nltk inflect tqdm scikit-learn matplotlib joblib ollama fastapi uvicorn flask
```

1. **Prepare datasets:** `Fake.csv` and `True.csv` in project root.
2. **Generate embeddings & train model:**

```
python V2_TrainingModel.py
```

1. **Run server:**

```
python Server.py
```

1. **Make predictions in Python:**

```
from DistributionAi import predict_article
result, confidence = predict_article("Some news text")
print(result, confidence)
```

------

### Notes

- Embeddings are saved in `embeddings.npy` to avoid recomputation.

- Current dataset uses 3000 articles per class; increasing data may improve accuracy.

- Multiple versions are included to track progress and experiments.

- **In progress of making V3_trainingModel **using **Deep learning**

  