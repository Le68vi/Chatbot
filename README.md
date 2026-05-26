# 💬 AI Customer Support Chatbot

An end-to-end AI-powered customer support chatbot that understands natural language queries and responds intelligently — with automatic escalation to a human agent when needed. Built using NLP, Machine Learning, and a real-time Streamlit interface.

---

## 📁 Project Structure

```
chatbot/
│
├── training.json                # Intent patterns and responses dataset
│
├── preprocessing.py             # Text cleaning and tokenization
├── task1_nlp_model.py           # Task 1: Train model + CLI chat loop
├── task2_app.py                 # Task 2: Streamlit chatbot dashboard
├── task3_training.py            # Task 3: Model training with evaluation
├── t4_escalation.py             # Task 4: Escalation logic and logger
│
├── chatbot_model.pkl            # Saved trained model (generated)
├── vectorizer.pkl               # Saved TF-IDF vectorizer (generated)
├── escalation_log.json          # Escalation event log (generated)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK resources (first run only)

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

---

## ⚙️ How to Run

### Task 1 — Train Model + CLI Chat

Trains the NLP model from `training.json` and launches a terminal-based chat loop.

```bash
python task1_nlp_model.py
```

**Example:**
```
You: Hi
Bot: Hello! How can I help you?

You: Where is my order
Bot: Please share your order ID so I can check the status for you.

You: quit
Bot: Goodbye! Have a great day.
```

---

### Task 3 — Retrain Model with Evaluation

Retrains the model and saves `chatbot_model.pkl` and `vectorizer.pkl` with accuracy metrics and sanity checks.

```bash
python task3_training.py
```

**Expected output:**
```
Total training samples: ~650
Total intents: 26
5-Fold CV Mean Accuracy: 80–88%

Sanity Check:
  ✅ 'hi' → greeting
  ✅ 'goodbye' → goodbye
  ✅ 'I want a refund' → refund
```

> ⚠️ Always run Task 3 before Task 2 to ensure the latest model is saved.

---

### Task 2 — Streamlit Chatbot Dashboard

Launches the interactive web-based chatbot interface.

```bash
streamlit run task2_app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🧠 How It Works

```
User types a message
        │
        ▼
preprocessing.py
   - Lowercase
   - Tokenize
   - Remove stopwords & punctuation
        │
        ▼
TF-IDF Vectorizer
   - Converts cleaned text to numeric vector
        │
        ▼
Logistic Regression Model
   - Predicts intent tag
   - Returns confidence score
        │
        ▼
t4_escalation.py — Decision Engine
   ├── Known intent + confidence ≥ 0.25 → Bot responds
   ├── Confidence < 0.35 → Polite fallback response
   ├── Confidence < 0.20 → Escalate to human
   ├── Negative signal words + low confidence → Escalate
   └── Human keyword detected → Always escalate
        │
        ▼
Response from training.json or Escalation message
```

---

## 🤖 Supported Intent Categories (26 total)

| Intent | Example Query |
|---|---|
| greeting | "Hi", "Hello", "Good morning" |
| goodbye | "Bye", "See you later", "I'm done" |
| thanks | "Thank you", "Much appreciated" |
| password_reset | "Forgot my password", "Can't login" |
| order_status | "Where is my order", "Track shipment" |
| refund | "I want a refund", "Return my money" |
| payment_issue | "Payment failed", "Card declined" |
| account_creation | "How to register", "Sign up problem" |
| delivery_issue | "Wrong item delivered", "Package damaged" |
| product_info | "Product details", "Is this in stock" |
| human_support | "I want to talk to a human", "Get me an agent" |
| cancel_order | "Cancel my order", "Stop my shipment" |
| change_address | "Update shipping address", "Wrong address" |
| coupon_discount | "Coupon not working", "Apply promo code" |
| invoice_billing | "I need my invoice", "GST receipt" |
| subscription | "Cancel subscription", "Upgrade my plan" |
| app_technical_issue | "App not working", "Website is down" |
| product_review | "How to write a review", "Rate a product" |
| loyalty_rewards | "My reward points", "Redeem points" |
| seller_support | "How to become a seller", "Seller account" |
| privacy_data | "Delete my account", "Privacy policy" |
| gift_card | "Use gift card", "Gift card balance" |
| warranty_claim | "Product is defective", "Claim warranty" |
| contact_info | "Customer care number", "Support hours" |
| negative_feedback | "Very bad experience", "I am frustrated" |
| bot_capabilities | "What can you do", "Are you a bot" |

---

## 🛡️ Escalation Logic (Task 4)

Escalation is triggered **only** when:

| Condition | Action |
|---|---|
| User explicitly asks for a human | Always escalate |
| Confidence < 0.20 on unknown intent | Escalate |
| Negative signal words + confidence < 0.35 | Escalate |
| Known intent + confidence ≥ 0.25 | Respond normally |
| Low confidence but known intent | Polite fallback |

All escalation events are logged to `escalation_log.json` with timestamp, user input, predicted intent, and confidence score.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| ML Model | Logistic Regression (scikit-learn) |
| Vectorizer | TF-IDF (unigrams + bigrams) |
| NLP | NLTK (tokenization, stopwords) |
| Dashboard | Streamlit |
| Serialization | joblib |

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Training Samples | ~650 |
| Intent Categories | 26 |
| CV Mean Accuracy | 80–88% |
| Escalation Threshold | confidence < 0.20 |

---

## 📌 Notes

- Run `task3_training.py` first to generate `chatbot_model.pkl` and `vectorizer.pkl` before launching the dashboard
- NLTK may show download warnings on systems without internet — non-critical if packages are already cached
- Escalation log resets on each session; `escalation_log.json` is appended, not overwritten
- To add new intents, add entries to `training.json` and retrain via `task3_training.py`
