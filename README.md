# CryptoSelect AI

Encrypt messages and files. A machine-learning model (Random Forest) recommends the best of three standard
ciphers (AES-128-GCM, AES-256-GCM, ChaCha20-Poly1305) for your situation, explains why, and the app encrypts your
data. The AI only selects; all cryptography comes from the `cryptography` library, unmodified.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The AI is trained automatically the first time you open Encrypt & Decrypt (about 15 seconds).

## Pages

| Page | What it shows |
|---|---|
| Home | What the app is and how to start |
| Encrypt & Decrypt | Add text or a file, describe your situation, see the recommendation and reasons, encrypt, download file and key. Decrypt reads the cipher from the file header. |
| Performance | The three speed tests from the research notebook: speed and memory table with throughput chart, median timings, local-socket transfer with modelled network delay |
| Train the AI | Cost formula, device factors and networks, label distribution, cross-validated training |
| Security Checks | PASS/FAIL tests for all three ciphers |
| AI Accuracy | Classification metrics, confusion matrix, feature importance, outcome table, brand-new scenarios, cost by network, sensitivity check |
| About & Limits | What is and is not claimed |

## Matches the research notebook

The scenario generator, cost formula, Random Forest settings, tables and figures follow the research notebook
exactly. With the built-in notebook measurements, seed 42 and 4000 scenarios, the website reproduces the notebook's
printed results (label shares, best parameters, cross-validated F1 0.9766, accuracy 0.9812, the outcome table and
its Wilcoxon p-values, the seed-777 table, cost by network, and the sensitivity table). `pytest` checks this.

If you run the speed tests on your own computer and train on those measurements, the numbers will differ slightly,
because the scenarios are built from your measured timings.

Priority setting: the cost model has one adjustable weight (energy versus speed, normally set by the battery level).
Fastest sets the energy weight to 10%, Energy-saving to 70%, Max security allows only 256-bit ciphers, and Balanced
follows the battery slider.

## Encrypted file format

`"CSAI"`, version, cipher id, key mode, [16-byte scrypt salt in password mode], 12-byte nonce, ciphertext + tag.
The whole header is passed as AEAD associated data, so changing it is detected. The original file name and type are
stored inside the encrypted payload.

## Tests

```bash
pytest
```

## Regenerating every table and figure for the report (optional)

The website itself does not export anything for a paper. To regenerate all tables, figures and a summary file:

```bash
python scripts/generate_evidence.py            # full run
python scripts/generate_evidence.py --quick    # quick check
```

Files go into `results/` with the notebook's names (fig1_throughput.png, table_outcome.csv, and so on). By default
the scenarios use the notebook's measurements so the numbers match the notebook; add `--profile local` to use the
speed test the script has just run. You can delete the `scripts/` folder if you do not need it.

## Project layout

```
app.py              entry point and navigation
pages/              one file per page
core/               crypto, fileformat, benchmark, dataset, selectors, explain, evaluate, figures, security_tests, ui
scripts/            optional evidence generator
tests/              pytest suite
models/  results/   generated at run time
```
