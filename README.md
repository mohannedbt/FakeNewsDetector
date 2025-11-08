- - # FakeNewsDetector

    [Python 3.10+](https://www.python.org/) | [MIT License](LICENSE) | [GitHub Issues](https://github.com/mohannedbt/FakeNewsDetector/issues)
  
    FakeNewsDetector is an AI-powered tool for classifying news articles as real or fake. It combines text preprocessing, embeddings, and machine learning or deep learning models, along with a web interface and API server for predictions.
  
    ---
  
    ## Project Structure
  
    FakeNewsDetector/
    │   DistributionAi.py          # Loads trained model and predicts
    │   Server.py                  # FastAPI server for predictions
    │   README.md                  # Project documentation
    │   V1_model.py                # Initial ML prototype
    │   V2_TrainingModel.py        # Preprocessing + embeddings + ML training (SGDClassifier)
    │   V3_TrainingModel.py        # Deep learning model (TensorFlow)
    │
    ├── templates/
    │       index.html             # Minimal web interface template
    └── __pycache__/               # Python compiled files (ignored)
  
    ---
  
    ## Features
  
    - Preprocessing: lowercasing, stopword removal, number-to-word conversion
    - Embeddings: uses Ollama local embeddings (mxbai-small) — tiny and fast
    - Models:
      - V2: SGDClassifier (traditional ML)
      - V3: Deep learning neural network (TensorFlow)
    - API & UI: FastAPI server + minimal web interface
    - Training visualization: plots log-loss and performance metrics
  
    ---
  
    ## Model Comparison
  
    | Version | Type               | Notes                                                        |
    | ------- | ------------------ | ------------------------------------------------------------ |
    | V2      | ML (SGDClassifier) | Lightweight ML model, good for fast training on embeddings   |
    | V3      | Deep Learning (NN) | TensorFlow neural network, handles complex patterns better, early stopping enabled |
  
    ---
  
    ## Quick Setup
  
    1. Install dependencies:
  
    pip install pandas numpy nltk inflect tqdm scikit-learn matplotlib joblib ollama fastapi uvicorn flask tensorflow
  
    2. Prepare datasets: Place `Fake.csv` and `True.csv` in the project root.
  
    3. Train Models:
  
    # Train ML model
    python V2_TrainingModel.py
  
    # Train Deep Learning model
    python V3_TrainingModel.py
  
    > Note: Make sure Ollama is installed locally. The embedding model `mxbai-small` is fast and lightweight. You can replace it with another Ollama embedding if desired.
  
    4. Run Server:
  
    python Server.py
  
    5. Make Predictions in Python:
  
    from DistributionAi import predict_article
  
    result, confidence = predict_article("Some news text here")
    print(result, confidence)
  
    ---
  
    ## Notes
  
    - `embeddings.npy` caches embeddings for faster retraining.
    - Multiple versions (`V1`, `V2`, `V3`) track experimentation.
    - `.pyc` and `__pycache__/` are ignored in Git.
    - Large datasets or embeddings should use Git LFS if exceeding GitHub’s 100MB limit.
  
    ---
  
    ## Recommendations
  
    - Use separate branches for experimentation.
    - Switch embedding models for larger datasets if needed.
    - Ensure Ollama is installed and accessible before training or inference.