import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from preprocess import clean_text

df = pd.read_csv("data/all_tickets.csv")

df["clean_text"] = df["Document"].apply(clean_text)

X = df["clean_text"]
y = df["Topic_group"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
joblib.dump(model, "models/ticket_classifier.pkl")

print("Model and vectorizer saved successfully.")