import time
import json
from datetime import datetime

# 1. ESCALATION KEYWORDS — only explicit human requests
HUMAN_KEYWORDS = [
    "human", "agent", "real person", "customer care",
    "talk to support", "live chat", "representative",
    "call me", "support executive", "help me now",
    "i want a person", "transfer me"
]

NEGATIVE_SIGNAL_WORDS = [
    "not working", "worst", "useless", "angry", "frustrated"
]

# ✅ Tags the model CAN handle — don't escalate these
KNOWN_INTENTS = [
    "greeting", "goodbye", "thanks", "hours", "order_status",
    "refund", "complaint", "product_info", "shipping", "payment"
    # Add all tags from your training.json here
]

# 2. ESCALATION DECISION ENGINE
def should_escalate(user_input, confidence, prediction):
    user_input_lower = user_input.lower()

    # Case 1: user directly asks for human → always escalate
    if any(word in user_input_lower for word in HUMAN_KEYWORDS):
        return True

    # Case 2: known intent with reasonable confidence → never escalate
    if prediction in KNOWN_INTENTS and confidence >= 0.30:
        return False

    # Case 3: truly unknown prediction → escalate
    if prediction in ["unknown", None, ""]:
        return True

    # Case 4: very low confidence AND negative signal → escalate
    if any(word in user_input_lower for word in NEGATIVE_SIGNAL_WORDS) and confidence < 0.35:
        return True

    # Case 5: extremely low confidence on unknown intent → escalate
    if confidence < 0.20:
        return True

    return False

# 3. ESCALATION RESPONSE
def get_escalation_message():
    return (
        "👨‍💼 I'm connecting you to a human support agent...\n"
        "⏳ Please wait while we transfer your request."
    )

# 4. ESCALATION LOGGER
def log_escalation(user_input, prediction, confidence):
    log_data = {
        "time": str(datetime.now()),
        "user_input": user_input,
        "prediction": prediction,
        "confidence": float(confidence)
    }
    try:
        with open("escalation_log.json", "a") as f:
            f.write(json.dumps(log_data) + "\n")
    except:
        pass

# 5. SIMULATED TRANSFER DELAY
def simulate_human_transfer():
    time.sleep(1.5)
    return "🔄 You are now connected to a human agent."