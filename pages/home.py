import streamlit as st

from core.config import MODEL_PATH
from core.ui import inject_css

inject_css()
st.title("CryptoSelect AI")
st.subheader("Encrypt messages and files, with an AI that picks the right cipher for your situation.")
st.write(
    "Tell the app about your device and what matters to you (speed, battery, or maximum security). "
    "It recommends one of three trusted ciphers, explains why, and encrypts your data."
)

c1, c2, c3 = st.columns(3)
for col, (title, text) in zip((c1, c2, c3), [
    ("1. Add your data", "Type a message or drag in any file (up to 200 MB)."),
    ("2. Describe your situation", "Device type, battery, network and your priority."),
    ("3. Get the best cipher", "See the recommendation and why, then encrypt and download."),
]):
    with col:
        with st.container(border=True):
            st.markdown(f"### {title}")
            st.write(text)

st.page_link("pages/encrypt_decrypt.py", label="Start encrypting or decrypting")

st.divider()
left, right = st.columns(2)
with left:
    st.markdown("#### What stays the same")
    st.write(
        "The encryption itself is standard and unmodified: AES-128-GCM, AES-256-GCM or "
        "ChaCha20-Poly1305 from the trusted `cryptography` library. The AI only **chooses** between them. "
        "Decryption never uses the AI: it reads the cipher from the file."
    )
with right:
    st.markdown("#### Status")
    st.write("AI model: " + ("ready" if MODEL_PATH.exists() else "will be set up automatically on first use"))
    st.page_link("pages/about.py", label="How it works and its limits")
