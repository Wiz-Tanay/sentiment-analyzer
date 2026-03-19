import streamlit as st
import pickle
import re
import emoji
import numpy as np
import time

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

# 🎨 Custom CSS
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.block-container {
    max-width: 700px;
    padding-top: 2rem;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    color: #aaa;
    margin-bottom: 30px;
}
textarea {
    border-radius: 12px !important;
    border: 1px solid #333 !important;
}
button[kind="primary"] {
    border-radius: 10px;
    width: 100%;
    height: 50px;
    font-size: 18px;
}
.result {
    text-align: center;
    padding: 30px;
    border-radius: 16px;
    font-size: 30px;
    font-weight: 600;
    margin-top: 25px;
    transition: all 0.3s ease-in-out;
}
.history-card {
    background-color: #1c1f26;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

def clean(t):
    t = t.lower()
    t = emoji.demojize(t)
    t = re.sub(r'http\S+', '', t)
    t = re.sub(r'@\w+', '', t)
    t = re.sub(r'#(\w+)', r'\1', t)
    t = t.replace(":", " ")
    t = re.sub(r'(.)\1+', r'\1\1', t)
    t = re.sub(r'[^a-z\s]', '', t)
    return t

def predict(text):
    cleaned = clean(text)
    vec = tfidf.transform([cleaned])

    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    confidence = round(np.max(prob) * 100, 2)

    if pred == -1:
        return "Negative 😡", "linear-gradient(135deg, #ff4d4d, #b30000)", confidence, cleaned
    elif pred == 0:
        return "Neutral 😐", "linear-gradient(135deg, #888, #555)", confidence, cleaned
    else:
        return "Positive 😊", "linear-gradient(135deg, #00c853, #007e33)", confidence, cleaned

# Header
st.markdown("<div class='title'>💬 Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Understand emotions instantly</div>", unsafe_allow_html=True)

# History
if "history" not in st.session_state:
    st.session_state.history = []

# Input
text = st.text_area("", placeholder="Type your sentence here...", height=140)

# Button-based UX
if st.button("Analyze 🚀"):

    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(0.6)

        label, gradient, confidence, cleaned = predict(text)

        # ✨ Result with gradient + glow
        st.markdown(f"""
        <div class="result" style="
            background: {gradient};
            color: white;
            box-shadow: 0 0 25px rgba(255,255,255,0.1);
        ">
            {label}
        </div>
        """, unsafe_allow_html=True)

        # Confidence
        st.markdown("#### Confidence")
        st.progress(confidence / 100)
        st.markdown(f"<p style='text-align:center'>{confidence}%</p>", unsafe_allow_html=True)

        # Save history
        st.session_state.history.append({
            "text": text,
            "label": label,
            "confidence": confidence
        })

        # Expandable cleaned text
        with st.expander("🔍 View processed text"):
            st.write(cleaned)

# History display
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🧠 Recent Predictions")

    for item in reversed(st.session_state.history[-5:]):
        st.markdown(f"""
        <div class="history-card">
            <b>Text:</b> {item['text']}<br>
            <b>Result:</b> {item['label']} ({item['confidence']}%)
        </div>
        """, unsafe_allow_html=True)