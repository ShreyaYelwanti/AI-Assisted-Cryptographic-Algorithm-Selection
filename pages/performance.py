"""Speed tests: the same three tests as the research notebook."""
import pandas as pd
import streamlit as st

from core.benchmark import get_system_specs, run_all, save_results
from core.config import RESULTS_DIR
from core.figures import fig_throughput
from core.ui import page_header, show_fig, show_table

page_header("Performance", "How fast does each cipher run on this computer?")
st.write("Three tests: (1) encrypt + decrypt speed and memory, (2) median timings that the AI's training data is "
         "built from, (3) encrypt, send over a local socket, receive and decrypt, with the network delay modelled "
         "for slow, medium and fast links. One warm-up is discarded before timing.")

quick = st.checkbox("Quick test (5 runs instead of 30/100/30)", value=False)
if st.button("Run speed tests", type="primary"):
    bar = st.progress(0.0, text="Starting...")
    res = run_all(quick=quick, progress=lambda done, total, msg: bar.progress(done / total, text=f"{msg}  ({done}/{total})"))
    bar.empty()
    save_results(res)
    st.session_state["bench"] = res
    st.success("Done and saved. To train the AI on these measurements, open Train the AI and choose "
               "'This computer's latest speed test'.")

res = st.session_state.get("bench")
if res is None and (RESULTS_DIR / "benchmark_summary.csv").exists():
    try:
        res = {"micro_summary": pd.read_csv(RESULTS_DIR / "benchmark_summary.csv"),
               "median_summary": pd.read_csv(RESULTS_DIR / "benchmark_summary_v2.csv"),
               "transfer_summary": None}
        st.caption("Showing results saved from an earlier run.")
    except Exception:
        res = None
if res is None:
    st.info("No results yet. Click **Run speed tests**.")
    st.stop()

st.markdown("#### Encrypt + decrypt speed (mean, spread and 95% confidence interval)")
micro = res["micro_summary"]
show_table(micro.round(5))
show_fig(fig_throughput(micro))

st.markdown("#### Median timings (used to build the AI's training data)")
show_table(res["median_summary"].round(2))

if res.get("transfer_summary") is not None:
    st.markdown("#### Local-socket transfer with modelled network delay (seconds, median and spread)")
    show_table(res["transfer_summary"])

with st.expander("About this computer"):
    st.table(pd.DataFrame(get_system_specs().items(), columns=["", "Value"]).set_index(""))
