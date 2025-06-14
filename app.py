import joblib
import re
import nltk
from flask import Flask, request, render_template, jsonify
from nltk.corpus import stopwords
from scrape_google_rss import get_google_news  # Import RSS news scraper

# Ensure stopwords are downloaded
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower().strip()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Load the trained model and vectorizer
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("✅ Model and Vectorizer Loaded Successfully!")
except Exception as e:
    print("❌ Error loading model/vectorizer:", e)
    exit(1)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Handle both JSON and Form input
        if request.is_json:
            data = request.get_json()
            title = data.get("title", "")
        else:
            title = request.form.get("news", "")

        if not title:
            return render_template("result.html", prediction="Error: No input provided", probability=0)

        # Preprocess input
        cleaned_text = clean_text(title)
        transformed_text = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(transformed_text)[0]
        probability = model.predict_proba(transformed_text)[0][prediction]
        label = "Real" if prediction == 1 else "Fake"

        return render_template("result.html", prediction=label, probability=probability * 100)

    except Exception as e:
        return render_template("result.html", prediction="Error", probability=0)

# 🔥 New Route to Predict RSS News
@app.route('/predict_rss', methods=['GET'])
def predict_rss():
    try:
        news_list = get_google_news()  # Fetch latest RSS news
        predictions = []

        for news in news_list:
            cleaned_text = clean_text(news["title"])
            transformed_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(transformed_text)[0]
            probability = model.predict_proba(transformed_text)[0][prediction]
            label = "Real" if prediction == 1 else "Fake"

            predictions.append({
                "title": news["title"],
                "source": news["source"],
                "result": label,
                "confidence": round(probability * 100, 2)
            })

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask app
if __name__ == "__main__"
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)))
