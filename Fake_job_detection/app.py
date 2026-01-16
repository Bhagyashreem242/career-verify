import streamlit as st
import joblib
import json
import pandas as pd

# Load model and vectorizer
model = joblib.load("model/fake_job_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

with open("model/metrics.json") as f:
    metrics = json.load(f)

st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🕵️",
    layout="centered"
)

st.title("🕵️ Fake Job Detection System")

# Metrics
st.subheader("📊 Model Performance")
st.write(f"**Accuracy:** {metrics['accuracy']*100:.2f}%")
st.write(f"**Macro F1-score:** {metrics['f1_macro']*100:.2f}%")

conf_df = pd.DataFrame(
    metrics["confusion_matrix"],
    columns=["Predicted Real", "Predicted Fake"],
    index=["Actual Real", "Actual Fake"]
)
st.table(conf_df)

st.caption("Metrics evaluated on real-world inspired data subset")

st.divider()

# Input
job_text = st.text_area(
    "Enter Job Details",
    height=220,
    placeholder="Paste job title, company, location and description"
)

if st.button("Check Job"):
    if job_text.strip() == "":
        st.warning("Please enter job details")
    else:
        vec = vectorizer.transform([job_text])
        probs = model.predict_proba(vec)[0]

        real_prob = probs[0]
        fake_prob = probs[1]

        # Show probabilities
        st.markdown("### 🔍 Prediction Confidence")
        st.markdown(
            f"""
            **Real:** {real_prob*100:.2f}%  
            **Fake:** {fake_prob*100:.2f}%
            """
        )

        st.progress(fake_prob)

        # Limited scam override
        scam_phrases = [
            "pay fees",
            "registration fee",
            "deposit required",
            "certificate fee"
        ]

        if any(p in job_text.lower() for p in scam_phrases):
            st.error("🚨 Fake Job Detected (Payment requested)")
        elif fake_prob >= 0.5:
            st.error("🚨 Fake Job Detected")
        else:
            st.success("✅ Real Job Posting")
