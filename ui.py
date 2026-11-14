"""Small Streamlit helpers shared by all pages (the only core module that imports streamlit)."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from core.dataset import generate_dataset, load_model, notebook_profile, save_model, train_selector

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.cs-big {font-size: 1.9rem; font-weight: 700; margin: 0;}
.cs-step {font-weight: 600; font-size: 1.15rem; margin-top: 1.2rem;}
footer {visibility: hidden;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    inject_css()
    st.title(title)
    st.caption(subtitle)


def step(n: int, text: str) -> None:
    st.markdown(f'<div class="cs-step">Step {n}: {text}</div>', unsafe_allow_html=True)


def show_table(df: pd.DataFrame, index: bool = False, **kw: Any) -> None:
    """st.dataframe that works on both old and new Streamlit versions."""
    try:
        st.dataframe(df, hide_index=not index, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, hide_index=not index, use_container_width=True, **kw)


def show_fig(fig: Any) -> None:
    """Show a matplotlib figure, then free it."""
    st.pyplot(fig)
    plt.close(fig)


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def human_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def ensure_model() -> dict[str, Any]:
    """Load the saved model, or train the notebook's model (seed 42, 4000 scenarios) on first use."""
    bundle = load_model()
    if bundle is None:
        with st.spinner("Setting up the AI for first use (about 15 seconds)..."):
            bundle = train_selector(generate_dataset(4000, 42, notebook_profile()), 42)
            save_model(bundle)
    return bundle
