import streamlit as st

from core.ui import page_header

page_header("About & Limits", "What this app does, and what it does not claim.")
st.markdown("""
### What it does
You describe your situation. A machine-learning model (a Random Forest) **chooses** one of three standard ciphers
(AES-128-GCM, AES-256-GCM, ChaCha20-Poly1305) and the app encrypts your data with it.

### What is *not* new
- The ciphers are **standard and unmodified**, taken from the `cryptography` library. No custom cryptography is used.
- The AI **only selects**. It never touches your data or keys, and it is **never used for decryption**:
  the cipher is read from the encrypted file's header.
- A security filter runs before and after the model: if you need 256-bit security, AES-128 can never be returned.

### How files are protected
Every encryption uses a fresh random 12-byte nonce. Keys are either random (shown once, downloadable) or derived from
your password with scrypt and a random 16-byte salt. The file header is authenticated, so changing it is detected.
The original file name is stored inside the encrypted part.

### How the priority setting works
The cost model has one adjustable weight: how much energy counts versus speed, which normally follows the battery level.
**Fastest** sets the energy weight to 10%, **Energy-saving** sets it to 70%, **Max security** allows only 256-bit
ciphers, and **Balanced** follows your battery level.

### Limits you should know about
- **Device profiles are simulated.** Slow CPUs, missing AES hardware and power draw are modelled with documented factors, not measured on real devices.
- **Network delay is modelled** from bandwidth; no real internet transfer is measured.
- **Speed results come from one machine** (the one running the tests). Other computers will differ.
- **ChaCha20's speed ratio and AES-256's factor are assumptions**, not measurements. The sensitivity check on the AI Accuracy page shows how much the conclusions depend on them.
- **Labels come from a cost formula.** The model learns to approximate the cheapest cipher without benchmarking every cipher on every message. Its accuracy shows how well it learns that formula.
- A simple hand-written rule list performs close to the AI in many situations. The AI's advantage is small in absolute cost, though statistically significant.
- This is a research prototype, not an audited security product. Do not rely on it for life-critical secrets.
""")
