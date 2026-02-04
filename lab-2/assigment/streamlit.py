import streamlit as st
import joblib

st.set_page_config(
    page_title="Nepali News Classifier",
    page_icon="📰",
    layout="centered"
)

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model = joblib.load(os.path.join(base_dir, 'model.joblib'))
    return model

model = load_model()
st.title("📰 Nepali News Classification")
st.markdown(
    "नेपाली समाचार पाठ प्रविष्ट गर्नुहोस् र यसको वर्ग (category) थाहा पाउनुहोस्।"
)

st.subheader("⚡ Quick Test Examples")
examples = {
    "देश": "लामो समयपछि शनिबार राति परेको पानीसँगै दार्चुलाका उच्च पहाडी भेगमा हिमपात भएको छ  आइतबार बिहानैबाट हिमपात भएको स्थानीयले बताएका छन् ।",
    "खेलकुद": "विकास दोस्रो चरणमापुलिसका विकास श्रेष्ठ आठौं राष्ट्रव्यापी कृष्णमोहन स्मृति खुला ब्याडमिन्टनको दोस्रो चरणमा सोमबार प्रवेश गरेका छन् ।",
    "अर्थ": "शेयर बजार आज उच्च अंकले बढेर बन्द भएको छ।",
    "मनोरञ्जन": "नयाँ नेपाली चलचित्रले बक्स अफिसमा राम्रो व्यापार गरेको छ।",
}
cols = st.columns(len(examples))
for col, (label, text) in zip(cols, examples.items()):
    if col.button(label):
        st.session_state["news_text"] = text

news_text = st.text_area(
    "✍️ Nepali News Text",
    height=220,
    key="news_text",
    placeholder="यहाँ नेपाली समाचार लेख्नुहोस्..."
)




# spell-checker: disable
if st.button("🔍 Predict Category"):
    if not news_text.strip():
        st.warning("कृपया समाचार पाठ प्रविष्ट गर्नुहोस्।")
    else:
        prediction = model.predict([news_text])[0]

        st.success(f"🗂️ Predicted Category: **{prediction}**")

        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba([news_text]).max()
            st.info(f"📊 Confidence: **{confidence:.2%}**")
            st.caption("Calculated as the highest probability from the model's `predict_proba` output.")

# spell-checker: enable




st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 0.9em;">
        👨‍💻 Developed by <b>Sandip Sapkota</b> <br>
        🤖 Using <b>Naive Bayes Classifier & TF-IDF Vectorizer</b> <br>
        🔗 <a href="https://github.com/dev-sandip" target="_blank">GitHub Profile</a> 📱
    </div>
    """,
    unsafe_allow_html=True
)
