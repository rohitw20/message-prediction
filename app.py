from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import string
import numpy as np
import spacy

# Load the trained models
status_model = joblib.load('sms_classifier.pkl')
category_model = joblib.load("sms_category_model.pkl")
amount_model = joblib.load("amount_predictor_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
sender_nlp_model = spacy.load("sender_ner_model")

# Define the app
app = FastAPI()

# Define the request model
class Message(BaseModel):
    message: str

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Keyword booster function
def keyword_boost(text):
    debit_keywords = [
        'removed', 'debited', 'deducted', 'withdrawn', 'purchased', 'spent',
        'paid', 'payment made', 'txn', 'transferred', 'upi payment', 'imps'
    ]

    credit_keywords = [
        'credited', 'received', 'deposited', 'added', 'rewarded',
        'refunded', 'reversed', 'cashback', 'payment received',
        'amount added', 'settled', 'salary', 'tpt', 'neft'
    ]

    text = text.lower()
    boost = []

    for word in debit_keywords:
        if word in text:
            boost.extend(['debitkeyword'] * 5)

    for word in credit_keywords:
        if word in text:
            boost.extend(['creditkeyword'] * 5)

    return text + " " + " ".join(boost)

# Regex extractor for amount
def extract_amount_regex(text):
    matches = re.findall(r'\b(?:inr|rs|mrp)?\s?([0-9]+(?:\.[0-9]{1,2})?)\b', text.lower())
    return float(matches[0]) if matches else 0.0

# Predict transaction category
def predict_category(message):
    return category_model.predict([message])[0]

# Predict transaction amount
def predict_amount(text):
    regex_amount = extract_amount_regex(text)
    if regex_amount > 0:
        return regex_amount
    else:
        cleaned = clean_text(text)
        transformed = tfidf_vectorizer.transform([cleaned])
        return round(amount_model.predict(transformed)[0], 2)

def predict_sender(text):
    doc = sender_nlp_model(text)
    sender = [ent.text for ent in doc.ents]
    print(sender)
    if len(sender)>0:
        return sender[0]
    else:
        return "Unknown"

# Define the prediction route
@app.post("/predict")
def predict(message: Message):
    current_message = message.message
    boosted_message = keyword_boost(current_message)
    prediction_status = status_model.predict([boosted_message])[0]
    prediction_category = predict_category(current_message)
    prediction_amount = predict_amount(current_message)
    prediction_sender = predict_sender(current_message)

    return {
        "status": prediction_status,
        "category": prediction_category,
        "amount": prediction_amount,
        "sender": prediction_sender
    }
