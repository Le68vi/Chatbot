import json
import random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocessing import preprocess_text

# Load dataset
with open('training.json') as file:
    data = json.load(file)

texts = []
labels = []
response_map = {}

# Extract training data
for intent in data['intents']:
    tag = intent['tag']
    response_map[tag] = intent['responses']
    for pattern in intent['patterns']:
        texts.append(preprocess_text(pattern))
        labels.append(tag)

# Auto-load all known intent tags from training.json
KNOWN_INTENTS = list(response_map.keys())

# Convert text into vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

# Save model and vectorizer
joblib.dump(model, 'chatbot_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("NLP model trained successfully")
print(f"Known intents: {KNOWN_INTENTS}")

# Chat loop
while True:

    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Bot: Goodbye! Have a great day.")
        break

    processed = preprocess_text(user_input)
    vector_input = vectorizer.transform([processed])

    prediction = model.predict(vector_input)[0]
    confidence = max(model.predict_proba(vector_input)[0])

    # Decision logic
    if prediction in KNOWN_INTENTS and confidence >= 0.25:
        # Known intent with decent confidence → respond normally
        response = random.choice(response_map[prediction])
        print(f"Bot: {response}")

    elif confidence < 0.20:
        # Truly unknown → escalate
        print("Bot: I'm connecting you to a human support agent. Please wait.")

    else:
        # Low confidence but not worth escalating → polite fallback
        print("Bot: I'm not sure I understood that. Could you rephrase? I can help with orders, refunds, shipping, and more.")