import joblib
import sys
import os
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import clean_text
from src.priority_logic import assign_priority



tfidf = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/ticket_classifier.pkl")

print("Support Ticket Classification System")
print("-----------------------------------")

while True:
    ticket = input("\nEnter support ticket (or 'exit'): ")

    if ticket.lower() == "exit":
        break

    cleaned = clean_text(ticket)
    vector = tfidf.transform([cleaned])

    category = model.predict(vector)[0]
    priority = assign_priority(ticket)

    print("Predicted Category:", category)
    print("Predicted Priority:", priority)