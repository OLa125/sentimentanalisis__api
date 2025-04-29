from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from transformers import pipeline
from langdetect import detect
from deep_translator import GoogleTranslator
import re
import os

app = Flask(__name__)
CORS(app)

# تحميل موديل التصنيف
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# التحقق من صلاحية الجملة
def is_valid_sentence(text):
    words = text.split()
    if len(words) < 3:
        return False
    if not re.search(r'[a-zA-Zأ-ي]', text):
        return False
    short_words = [w for w in words if len(w) <= 2]
    if len(short_words) > len(words) * 0.7:
        return False
    return True

# تحويل التقييم إلى مشاعر
def convert_rating_to_sentiment(rating):
    if rating == 1:
        return "angry"
    elif rating == 2:
        return "sad"
    elif rating == 3:
        return "normal"
    else:
        return "happy"

# HTML لواجهة المستخدم
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        .container { max-width: 600px; margin: auto; }
        textarea {
            width: 100%; height: 150px; padding: 10px; font-size: 16px;
        }
        button {
            padding: 10px 20px; font-size: 16px; margin-top: 10px;
            cursor: pointer; background-color: #4CAF50; color: white;
            border: none; border-radius: 5px;
        }
        .result, .error {
            margin-top: 20px; padding: 15px; border-radius: 5px;
        }
        .result { background-color: #f0f0f0; }
        .error { background-color: #ffe5e5; color: #b30000; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sentiment Analysis Tool</h1>
        <textarea id="text-input" placeholder="Enter text here"></textarea>
        <button onclick="getPrediction()">Get Sentiment</button>

        <div class="error" id="error-message" style="display:none;"></div>

        <div class="result" id="result" style="display:none;">
            <h3>Results:</h3>
            <p><strong>Original Text:</strong> <span id="original-text"></span></p>
            <p><strong>Translated Text:</strong> <span id="translated-text"></span></p>
            <p><strong>Sentiment:</strong> <span id="sentiment"></span></p>
            <p><strong>Rating:</strong> <span id="rating"></span></p>
            <p><strong>Raw Label:</strong> <span id="raw-label"></span></p>
            <p><strong>Score:</strong> <span id="score"></span></p>
        </div>
    </div>

    <script>
        async function getPrediction() {
            const textInput = document.getElementById('text-input').value;
            const resultDiv = document.getElementById('result');
            const errorDiv = document.getElementById('error-message');

            resultDiv.style.display = 'none';
            errorDiv.style.display = 'none';
            errorDiv.innerText = '';

            if (!textInput.trim()) {
                errorDiv.style.display = 'block';
                errorDiv.innerText = 'Please enter some text!';
                return;
            }

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: textInput })
            });

            const data = await response.json();

            if (data.error) {
                errorDiv.style.display = 'block';
                errorDiv.innerText = data.error;
                return;
            }

            document.getElementById('original-text').innerText = data.original_text;
            document.getElementById('translated-text').innerText = data.translated_text;
            document.getElementById('sentiment').innerText = data.sentiment;
            document.getElementById('rating').innerText = data.rating;
            document.getElementById('raw-label').innerText = data.raw_label;
            document.getElementById('score').innerText = data.score;

            resultDiv.style.display = 'block';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_template)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    original_text = data["text"]

    if not original_text.strip():
        return jsonify({"error": "Text is empty"}), 400

    if not is_valid_sentence(original_text):
        return jsonify({"error": "Please enter a meaningful sentence in English or any natural language."}), 400

    try:
        detected_language = detect(original_text)
        translated_text = (
            GoogleTranslator(source='auto', target='en').translate(original_text)
            if detected_language != "en"
            else original_text
        )
    except Exception as e:
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500

    try:
        result = classifier(translated_text)[0]
        star_rating = int(result['label'].split()[0])
        sentiment = convert_rating_to_sentiment(star_rating)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "original_text": original_text,
        "translated_text": translated_text,
        "sentiment": sentiment,
        "rating": star_rating,
        "raw_label": result['label'],
        "score": result['score']
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

