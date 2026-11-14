"""Run the security verification suite."""
import streamlit as st

from core.dataset import load_model
from core.security_tests import run_all_security_tests
from core.ui import page_header, show_table

page_header("Security Checks", "Proof that the encryption behaves safely: it rejects wrong keys, tampering and more.")
st.write("These tests run real encryption and decryption for all three ciphers. A test passes only if the "
         "system does the safe thing, for example refusing to open a file that was altered.")

if st.button("Run all security tests", type="primary"):
    with st.spinner("Running tests (a few seconds)..."):
        table, rate = run_all_security_tests(load_model())
    st.session_state["sec_out"] = (table, rate)

if "sec_out" not in st.session_state:
    st.info("Click the button to run the tests.")
    st.stop()

table, rate = st.session_state["sec_out"]
c1, c2, c3 = st.columns(3)
c1.metric("Pass rate", f"{rate:.0f}%")
c2.metric("Tests passed", f"{int((table['Result'] == 'PASS').sum())} / {len(table)}")
c3.metric("Failures", int((table["Result"] == "FAIL").sum()))
if rate == 100:
    st.success("Every test passed.")
else:
    st.error("Some tests failed. See the table below.")

st.markdown("#### Summary by cipher")
per = table[table["Cipher"] != "All selectors"]
wanted = {"Round trip (random key)": "Round-trip OK", "Corrupted ciphertext rejected": "Tamper rejected",
          "10,000 unique nonces": "Unique nonces", "Overhead is 28 bytes": "Overhead = 28 B"}
pivot = per[per["Test"].isin(wanted)].pivot(index="Cipher", columns="Test", values="Result")
pivot = pivot[[t for t in wanted]].rename(columns=wanted).reset_index()
show_table(pivot)

st.markdown("#### All tests")
show_table(table)
st.download_button("Download results (CSV)", table.to_csv(index=False), "security_tests.csv", "text/csv")
