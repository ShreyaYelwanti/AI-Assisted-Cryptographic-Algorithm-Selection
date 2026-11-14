"""CryptoSelect AI - entry point. Run with:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="CryptoSelect AI", layout="wide")

navigation = st.navigation({
    "Use it": [
        st.Page("pages/home.py", title="Home", default=True),
        st.Page("pages/encrypt_decrypt.py", title="Encrypt & Decrypt"),
    ],
    "Under the hood": [
        st.Page("pages/performance.py", title="Performance"),
        st.Page("pages/train_ai.py", title="Train the AI"),
        st.Page("pages/security_checks.py", title="Security Checks"),
        st.Page("pages/ai_accuracy.py", title="AI Accuracy"),
    ],
    "Info": [st.Page("pages/about.py", title="About & Limits")],
})
navigation.run()
