from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the trained model
status_model = joblib.load('sms_classifier.pkl')

# Loading the category prediction model
category_model = joblib.load("sms_category_model.pkl")
# Define the app
app = FastAPI()

# Define the request model
class Message(BaseModel):
    message: str

# Your custom keyword booster function
def keyword_boost(text):
    debit_keywords = ['removed', 'debited', 'deducted', 'withdrawn', 'purchased', 'spent', 'paid']
    credit_keywords = ['credited', 'received', 'deposited', 'added']
    
    text = text.lower()
    extra_tokens = []
    for word in debit_keywords:
        if word in text:
            extra_tokens.append('debitkeyword')
    for word in credit_keywords:
        if word in text:
            extra_tokens.append('creditkeyword')
    return text + " " + " ".join(extra_tokens)

def predict_category(message): 
    return category_model.predict([message])[0]

# Define the prediction route
@app.post("/predict")
def predict(message: Message):
    current_message = message.message
    boosted_message = keyword_boost(current_message)
    prediction_message = status_model.predict([boosted_message])[0]
    prediction_category = predict_category(current_message) 
    return {
        "status": prediction_message,
        "category": prediction_category
        }
