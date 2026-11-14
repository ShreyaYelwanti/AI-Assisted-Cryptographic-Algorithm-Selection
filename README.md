#  CryptoSelect AI

An AI-assisted cryptographic algorithm selection application that encrypts messages and files using suitable encryption algorithms.

A Random Forest machine learning model recommends one of three cryptographic algorithms — AES-128-GCM, AES-256-GCM, or ChaCha20-Poly1305 — based on the user's situation. The application explains its recommendation and performs encryption using the Python `cryptography` library.

**Note:** AI is used only for algorithm selection. The actual encryption and decryption are handled by the cryptographic library.

##  Features

- AI-based cryptographic algorithm recommendation.
- Text and file encryption and decryption.
- Supports three authenticated encryption algorithms.
- Explanation of algorithm recommendations.
- Encryption performance benchmarking.
- AI accuracy evaluation and visualizations.
- Security checks for all three algorithms.
- Downloadable encrypted files and keys.

##  Technologies Used

- Python
- Random Forest
- Scikit-learn
- Streamlit
- Cryptography
- NumPy
- Pandas
- Matplotlib
- Pytest

##  Application Pages

| Page | Description |
|---|---|
| Home | Project introduction and navigation |
| Encrypt & Decrypt | Encrypt and decrypt text or files |
| Performance | Speed and memory benchmarking |
| Train the AI | Model training and cost analysis |
| Security Checks | PASS/FAIL security tests |
| AI Accuracy | Classification metrics and model evaluation |
| About & Limits | Project explanation and limitations |

##  Installation and Setup

**1. Clone the repository**

```bash
git clone <your-repository-url>
cd <repository-folder>
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate the virtual environment**

Windows:
```bash
.venv\Scripts\activate
```

Mac/Linux:
```bash
source .venv/bin/activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Run the application**

```bash
streamlit run app.py
```

The Random Forest model is trained automatically the first time you open the Encrypt & Decrypt page (approximately 15 seconds).

##  Supported Encryption Algorithms

- AES-128-GCM
- AES-256-GCM
- ChaCha20-Poly1305

The application recommends an algorithm based on factors such as CPU capability, AES hardware availability, battery level, payload size, security requirements, and network conditions.

##  Testing

Run the automated test suite using:

```bash
pytest
```

##  Optional: Regenerating Evaluation Results

The project includes a script for regenerating evaluation tables and figures.

Full execution:

```bash
python scripts/generate_evidence.py
```

Quick execution:

```bash
python scripts/generate_evidence.py --quick
```

Generated files are stored in the `results/` directory.

##  Project Structure

```text
CryptoSelectAI/
│
├── app.py
├── requirements.txt
│
├── core/
├── pages/
├── tests/
├── scripts/
├── models/
├── results/
└── .streamlit/
```

##  Project

Developed as a collaborative team project.
