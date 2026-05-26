import json
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import preprocess_text

# Load dataset
with open('training.json') as file:
    data = json.load(file)

texts = []
labels = []

for intent in data['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        processed = preprocess_text(pattern)
        texts.append(processed)
        labels.append(tag)

print(f"Total training samples: {len(texts)}")
print(f"Total intents: {len(set(labels))}")

# ✅ Better vectorizer — captures word pairs and more features
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),      # unigrams + bigrams
    max_features=3000,
    sublinear_tf=True,       # dampens high frequency terms
    min_df=1                 # keep all terms (small dataset)
)

X = vectorizer.fit_transform(texts)
y = labels

# ✅ Cross-validation — more reliable than single split on small data
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # handles intents with fewer patterns
    C=2.0                     # slightly less regularization for small data
)

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"\n=== 5-Fold Cross-Validation ===")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}")
print(f"  Mean Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ✅ Train on full data for best final model
model.fit(X, y)

# Quick sanity check
print("\n=== Sanity Check ===")
test_cases = [
    "hi",
    "goodbye",
    "see you later",
    "I want to return my order",
    "where is my package",
    "I need help with payment",
]
for text in test_cases:
    vec = vectorizer.transform([preprocess_text(text)])
    pred = model.predict(vec)[0]
    conf = max(model.predict_proba(vec)[0])
    print(f"  '{text}' → {pred} ({conf*100:.1f}%)")

# Save
joblib.dump(model, 'chatbot_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("\nTraining completed successfully.")