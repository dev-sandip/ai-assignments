import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os


@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(__file__)
    model = joblib.load(os.path.join(base_dir, 'spam_ham_model.joblib'))
    vectorizer = joblib.load(os.path.join(base_dir, 'count_vectorizer.joblib'))
    return model, vectorizer

model, vectorizer = load_assets()


le = LabelEncoder()
le.classes_ = np.array(['ham', 'spam'])


st.set_page_config(
    page_title="Spam / Ham Classifier",
    page_icon="📧",
    layout="centered"
)


st.title("📧 Spam or Ham Classifier")
st.markdown("Simple tool to detect whether a message is **spam** or **ham** (legitimate) using Naive Bayes.")

st.divider()


user_message = st.text_area(
    "Enter the message you want to check:",
    height=130,
    placeholder="Example: Congratulations! You've won $1000 gift card! Click here...",
    help="Paste any suspicious or normal message here"
)

classify_button = st.button("🔍 Classify", type="primary", use_container_width=True)


if classify_button:
    if user_message.strip():
        with st.spinner("Analyzing..."):
            
            X_input = vectorizer.transform([user_message])
            prediction = model.predict(X_input)
            label = le.inverse_transform(prediction)[0]
            
            
            try:
                probs = model.predict_proba(X_input)[0]
                spam_prob = probs[1] * 100 if len(probs) > 1 else None
            except:
                spam_prob = None

        
        if label == "spam":
            st.error(f"**SPAM** detected!")
            if spam_prob is not None:
                st.write(f"Spam confidence: **{spam_prob:.1f}%**")
        else:
            st.success(f"**HAM** (normal message)")
            if spam_prob is not None:
                st.write(f"Spam confidence: **{spam_prob:.1f}%**")

        st.info(f"Raw prediction class: {label}")

    else:
        st.warning("Please enter a message first!")


st.divider()
st.subheader("Quick Test Examples")

examples = [
    "Congratulations! You've won a $1,000 Walmart gift card! Claim now.",
    "Hey, are we still on for lunch tomorrow at 1?",
    "URGENT! Your account has been compromised. Reset password now.",
    "Meeting at 3pm - don't forget to bring the project report",
    "You've been selected as lucky winner! Call now: 1800-123-4567"
]

for i, msg in enumerate(examples, 1):
    with st.expander(f"Example {i}", expanded=(i==1)):
        st.write(msg)
        if st.button("Test this message", key=f"test_{i}"):
            X_test = vectorizer.transform([msg])
            pred = model.predict(X_test)
            lbl = le.inverse_transform(pred)[0]
            
            if lbl == "spam":
                st.error("→ **SPAM**")
            else:
                st.success("→ **HAM**")


st.divider()
st.caption("Powered by Multinomial Naive Bayes • Trained on SMS Spam Collection dataset")
st.caption("Updated: January 2026")
st.caption("Developed by Sandip Sapkota")