from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the trained model
model = joblib.load('sms_classifier.pkl')

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

# Define the prediction route
@app.post("/predict")
def predict(message: Message):
    boosted_message = keyword_boost(message.message)
    prediction = model.predict([boosted_message])[0]
    return {"prediction": prediction}
