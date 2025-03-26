import pickle
from scrape_google_rss import get_google_news

# 📌 Load Trained Model & Vectorizer
with open("model/vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("model/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

def predict_news(news_text):
    """News ko model se analyze karke Fake ya Real check karega."""
    transformed_text = vectorizer.transform([news_text])
    prediction = model.predict(transformed_text)[0]
    probability = model.predict_proba(transformed_text)[0][prediction]
    
    return "Fake" if prediction == 0 else "Real", probability * 100

if __name__ == "_main_":
    news_list = get_google_news()

    print("\n📌 Checking Google News:")
    for news in news_list:
        result, confidence = predict_news(news["title"])
        print(f"- {news['title']} ({news['source']}) → {result} ({confidence:.2f}% confidence)")