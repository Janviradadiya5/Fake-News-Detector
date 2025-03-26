import os
import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 📌 Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 📌 Function to clean text
def clean_text(text):
    if pd.isna(text):  # Handle NaN values
        return ""
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower().strip()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# 📌 Set dataset directory
DATASET_DIR = os.path.join("templates", "dataset")

# 📌 Load all four datasets
real_df_1 = pd.read_csv(os.path.join(DATASET_DIR, "gossipcop_real.csv"))
fake_df_1 = pd.read_csv(os.path.join(DATASET_DIR, "gossipcop_fake.csv"))
real_df_2 = pd.read_csv(os.path.join(DATASET_DIR, "politifact_real.csv"))
fake_df_2 = pd.read_csv(os.path.join(DATASET_DIR, "politifact_fake.csv"))

# 📌 Select the 'title' column and drop NaN values
real_df_1 = real_df_1[["title"]].dropna()
fake_df_1 = fake_df_1[["title"]].dropna()
real_df_2 = real_df_2[["title"]].dropna()
fake_df_2 = fake_df_2[["title"]].dropna()

# 📌 Clean titles
real_df_1["title"] = real_df_1["title"].apply(clean_text)
fake_df_1["title"] = fake_df_1["title"].apply(clean_text)
real_df_2["title"] = real_df_2["title"].apply(clean_text)
fake_df_2["title"] = fake_df_2["title"].apply(clean_text)

# 📌 Add labels (1 = real, 0 = fake)
real_df_1["label"] = 1
real_df_2["label"] = 1
fake_df_1["label"] = 0
fake_df_2["label"] = 0

# 📌 Merge all datasets
df = pd.concat([
    real_df_1, fake_df_1, real_df_2, fake_df_2
], ignore_index=True)

# 📌 Split dataset
X_train, X_test, y_train, y_test = train_test_split(df["title"], df["label"], test_size=0.2, random_state=42)

# 📌 Train model
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 📌 Save model & vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")