from flask import Flask, render_template, request
from DistributionAi import predict_news  # import your AI functions

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news_text']
    label, confidence = predict_news(news_text)
    
    return render_template('index.html',
                           news_text=news_text,
                           prediction=label,
                           confidence=f"{confidence:.2f}%")

if __name__ == "__main__":
    app.run(debug=True)
