import streamlit as st
import joblib
import json
import random

from preprocessing import preprocess_text
from t4_escalation import (
    should_escalate,
    get_escalation_message,
    log_escalation,
    simulate_human_transfer
)

# LOAD MODEL
model = joblib.load("chatbot_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

with open("training.json") as file:
    data = json.load(file)

response_map = {i["tag"]: i["responses"] for i in data["intents"]}

# UI CONFIG
st.set_page_config(page_title="Support Bot", page_icon="💬")
st.title("💬 AI Customer Support Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

# DISPLAY CHAT
for sender, msg in st.session_state.chat:
    if sender == "user":
        st.markdown(f"🧑‍💻 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Bot:** {msg}")

# USER INPUT
user_input = st.text_input("Type your message")

if st.button("Send") and user_input:

    st.session_state.chat.append(("user", user_input))

    processed = preprocess_text(user_input)
    vector = vectorizer.transform([processed])

    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0])

    # ESCALATION CHECK ← everything below must be indented inside this block
    if should_escalate(user_input, confidence, prediction):
        log_escalation(user_input, prediction, confidence)
        bot_reply = get_escalation_message()
        st.session_state.chat.append(("bot", bot_reply))
        st.session_state.chat.append(("bot", simulate_human_transfer()))

    elif confidence < 0.35:
        bot_reply = "🤔 I'm not sure I understood that. Could you rephrase? I can help with orders, refunds, shipping, and more."
        st.session_state.chat.append(("bot", bot_reply))

    else:
        bot_reply = random.choice(response_map[prediction])
        st.session_state.chat.append(("bot", bot_reply))

    st.rerun()  # ← also inside the button block