"""How good is the AI compared with simpler approaches? Same tables and figures as the research notebook."""
import streamlit as st

from core.dataset import load_model
from core.evaluate import evaluate_all, new_seed_tables, sensitivity_table
from core.figures import fig_confusion, fig_cost_by_network, fig_importance
from core.ui import page_header, show_fig, show_table

page_header("AI Accuracy", "Does the AI really choose better than simple alternatives?")
bundle = load_model()
if bundle is None:
    st.warning("No trained AI yet. Open **Train the AI** first, or use Encrypt & Decrypt once, which sets it up.")
    st.stop()

st.write("Four ways of choosing a cipher are compared: always AES-256, always ChaCha20, a hand-written rule list, and "
         "the Random Forest (the AI). First on the 20% of scenarios held back from training, then on brand-new scenarios "
         "from a different random seed.")
c1, c2 = st.columns(2)
new_seed = c1.number_input("Seed for the brand-new scenarios (must differ from training)", 0, 100_000, 777)
n_new = c2.number_input("Number of brand-new scenarios", 500, 20_000, 4000, step=500)

if st.button("Run evaluation", type="primary"):
    if int(new_seed) == int(bundle["seed"]):
        st.error("Please use a different seed from training, otherwise this is not a fair test.")
    else:
        with st.spinner("Evaluating..."):
            st.session_state["eval_out"] = (evaluate_all(bundle), new_seed_tables(bundle, int(new_seed), int(n_new)))

out = st.session_state.get("eval_out")
if out is None:
    st.info("Click **Run evaluation**.")
    st.stop()
res, (new_table, by_net, _) = out

oc = res["outcome"].set_index("Selector")
k1, k2, k3 = st.columns(3)
k1.metric("Random Forest picks the best cipher", f"{oc.loc['Random Forest', '% optimal']:.1f}%")
k2.metric("Cost saving vs always AES-256", f"{oc.loc['Fixed AES-256', 'Cost reduction by RF (%)']:.1f}%")
k3.metric("Security violations (with filter)", f"{new_table.set_index('Selector').loc['RF + security filter', '% security violations']:.1f}%",
          help="How often a cipher weaker than required was returned. Must be 0%.")
st.caption("The best cipher for each scenario is defined by the cost formula, so accuracy shows how well the model "
           "learns that formula. The mean cost (1.0 is perfect) is the more meaningful measure.")

st.markdown("#### 1. Held-out 20% of the training scenarios")
st.markdown("**Classification metrics**")
show_table(res["classification"])
l, r = st.columns(2)
with l:
    show_fig(fig_confusion(res["confusion"]))
with r:
    show_fig(fig_importance(res["importance"]))
st.markdown("**Outcome: cost, time and security**")
show_table(res["outcome"])
st.caption("Mean cost with a 95% confidence interval. Wilcoxon p-values are paired tests of Random Forest against each "
           "baseline on per-scenario cost. Cost reduction is how much cheaper the Random Forest is than that baseline.")

st.markdown("#### 2. Brand-new scenarios (different seed)")
show_table(new_table)
st.markdown("**Mean cost by network**")
show_table(by_net, index=True)
show_fig(fig_cost_by_network(by_net))

st.markdown("#### 3. How much do the results depend on the assumptions?")
st.write("Re-runs the whole pipeline with different values for the ChaCha20 speed ratio and the AES-256 factor "
         "(these two are assumptions, not measurements).")
if st.button("Run sensitivity check"):
    with st.spinner("Re-training 9 models (about 20 seconds)..."):
        st.session_state["sens_out"] = sensitivity_table(bundle["profile"])
if "sens_out" in st.session_state:
    show_table(st.session_state["sens_out"])
