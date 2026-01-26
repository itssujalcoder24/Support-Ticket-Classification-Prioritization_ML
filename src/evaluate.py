import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from preprocess import clean_text

df = pd.read_csv("data/all_tickets.csv")

df["clean_text"] = df["Document"].apply(clean_text)

X = df["clean_text"]
y = df["Topic_group"]

tfidf = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/ticket_classifier.pkl")

X_tfidf = tfidf.transform(X)
y_pred = model.predict(X_tfidf)

print(classification_report(y, y_pred))

ConfusionMatrixDisplay.from_predictions(
    y, y_pred, xticks_rotation=45
)
plt.title("Confusion Matrix - Ticket Classification")
plt.show()
