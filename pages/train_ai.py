"""Build the simulated scenarios and train the cipher-selection model."""
import pandas as pd
import streamlit as st

from core.config import CPU_CLASSES, NETWORK_INFO
from core.dataset import (
    CPU_SLOW, HW_PROB, NET_MBPS, NOTEBOOK_MEDIANS, NOTEBOOK_SOURCE, POWER_W, Assumptions, chacha_ratio,
    generate_dataset, load_model, local_profile, notebook_profile, save_model, train_selector,
)
from core.ui import page_header, show_table

page_header("Train the AI", "Build the practice scenarios and teach the model to pick the cheapest cipher.")

st.markdown("#### How a scenario is scored")
st.write("Every simulated situation gets a **cost** for each allowed cipher. The cipher with the lowest cost is the "
         "correct answer the AI learns from. 1.0 means as good as the best possible choice. "
         "AES-128 is not allowed when 256-bit security is required.")
st.latex(r"\text{cost} = w_{time}\cdot\frac{time}{time_{min}} + w_{energy}\cdot\frac{energy}{energy_{min}}")
st.write("Energy weight by battery level: **0.7** below 20%, **0.4** below 50%, otherwise **0.1**. "
         "The time weight is 1 minus the energy weight.")
a = Assumptions()
l, r = st.columns(2)
with l:
    st.markdown("**Device factors (assumptions)**")
    show_table(pd.DataFrame([{"Processor": c, "Slower than the test laptop": f"{CPU_SLOW[c]}x",
                              "Power (W)": POWER_W[c], "Chance of AES hardware": f"{HW_PROB[c]:.0%}"}
                             for c in CPU_CLASSES]))
    st.caption(f"No AES hardware: AES runs {a.aes_sw_penalty}x slower (ChaCha20 unaffected). "
               f"Random noise: {a.noise_sigma:.0%} on every timing.")
with r:
    st.markdown("**Networks (modelled from bandwidth, not measured)**")
    show_table(pd.DataFrame([{"Network": v[0], "Bandwidth": f"{NET_MBPS[k]} Mbit/s"} for k, v in NETWORK_INFO.items()]))
    st.caption(f"ChaCha20 time = AES time x {chacha_ratio(1024):.2f} for small data, rising to "
               f"{chacha_ratio(10**6):.2f} for large data (assumed). AES-256 time = AES-128 time x {a.aes256_factor} (assumed).")

st.divider()
st.markdown("#### Train")
local = local_profile()
choices = [NOTEBOOK_SOURCE] + (["Measured on this computer (latest speed test)"] if local else [])
source = st.radio("Speed data to build the scenarios from", choices,
                  help="The notebook measurements reproduce the research notebook's numbers exactly.")
profile = local if (local and source != NOTEBOOK_SOURCE) else notebook_profile()
if not local:
    st.caption("Run the speed tests on the Performance page to unlock your own computer's measurements.")

c1, c2, c3 = st.columns(3)
n_rows = c1.number_input("Practice scenarios", 500, 50_000, 4000, step=500)
seed = c2.number_input("Random seed", 0, 10_000, 42)
compare = c3.checkbox("Also compare Decision Tree and Logistic Regression", value=False)

if st.button("Train the AI", type="primary"):
    with st.spinner("Generating scenarios and training (about 15 to 40 seconds)..."):
        df = generate_dataset(int(n_rows), int(seed), profile)
        bundle = train_selector(df, int(seed), profile, compare_models=compare)
        save_model(bundle)
    st.session_state["train_out"] = (df, bundle)
    st.success("Trained and saved. The Encrypt & Decrypt page now uses this model.")

if "train_out" in st.session_state:
    df, bundle = st.session_state["train_out"]
    st.markdown("**Which cipher was best (share of scenarios)**")
    show_table(df["label"].value_counts(normalize=True).round(3).rename("share").reset_index()
               .rename(columns={"label": "Best cipher"}))
    st.markdown("**Best cipher by AES hardware and required security (counts)**")
    show_table(pd.crosstab([df["has_aes_hw"], df["required_security"]], df["label"]), index=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Cross-validated macro-F1", f"{bundle['cv_macro_f1']:.4f}")
    m2.metric("Training scenarios", f"{bundle['n_train']:,}")
    m3.metric("Test scenarios", f"{bundle['n_test']:,}")
    st.caption(f"Best Random Forest settings: {bundle['best_params']}. Tuned with 5-fold cross-validation "
               "on the training 80% only; the other 20% is never seen during tuning.")
    if len(bundle["comparison"]):
        st.markdown("**Model comparison**")
        show_table(bundle["comparison"])
    with st.expander("Preview the scenarios"):
        show_table(df[["cpu_class", "has_aes_hw", "battery_pct", "payload_size", "required_security",
                       "network", "label"]].head(20))
else:
    existing = load_model()
    if existing:
        st.caption(f"A saved model exists (trained on {existing['n_rows']:,} scenarios, seed {existing['seed']}).")

with st.expander("Speed measurements the notebook's scenarios are built from"):
    show_table(NOTEBOOK_MEDIANS)
