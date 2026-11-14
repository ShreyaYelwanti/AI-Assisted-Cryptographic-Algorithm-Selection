"""Main page: encrypt (with an AI cipher recommendation) or decrypt."""
from __future__ import annotations

import hashlib
import mimetypes

import streamlit as st

from core.config import CIPHER_BLURB, CPU_CLASSES, CPU_LABELS, NETWORK_INFO, NETWORKS, OVERHEAD_BYTES, PRIORITIES
from core.crypto import CryptoAuthenticationError, key_from_base64, key_from_hex, key_to_base64
from core.dataset import Assumptions, network_delay_s
from core.explain import comparison_table, explain_decision, grouped_importances
from core.figures import fig_confidence, fig_factor_importance
from core.fileformat import (
    InvalidContainerError, KEY_MODE_PASSWORD, decrypt_file_or_data, encrypt_file_or_data, parse_container_header,
)
from core.selectors import Scenario, allowed_ciphers, apply_priority, rf_proba, rf_secure, select_all
from core.ui import ensure_model, human_size, human_time, page_header, show_fig, show_table, step

page_header("Encrypt & Decrypt", "Protect a message or file, or open one that was protected here.")
mode = st.radio("What would you like to do?", ["Encrypt", "Decrypt"], horizontal=True)


def data_signature(data: bytes, *extra: object) -> str:
    """Cheap fingerprint so a stale result is never shown next to changed input."""
    h = hashlib.blake2b(digest_size=12)
    h.update(str(len(data)).encode() + data[:65536] + data[-65536:])
    for e in extra:
        h.update(repr(e).encode())
    return h.hexdigest()


def password_box(label: str, key: str) -> str:
    show = st.checkbox("Show password", key=f"{key}_show")
    return st.text_input(label, type="default" if show else "password", key=key)


def parse_key(text: str) -> bytes:
    """Accept a Base64 or hex key (first word of the text)."""
    token = text.strip().split()[0] if text.strip() else ""
    if not token:
        raise ValueError("Please enter or upload the key.")
    for fn in (key_from_base64, key_from_hex):
        try:
            key = fn(token)
            if len(key) in (16, 32):
                return key
        except ValueError:
            continue
    raise ValueError("That does not look like a valid key (expected a Base64 or hex key from the encrypt step).")


# ======================================================================================
# ENCRYPT
# ======================================================================================
PRESETS = {
    "Custom (set it yourself)": None,
    "Phone on mobile data": dict(cpu="mid", hw=True, batt=40, net="medium", sec=128, prio="Balanced"),
    "Old phone, low battery": dict(cpu="low", hw=False, batt=12, net="slow", sec=128, prio="Balanced"),
    "Laptop on Wi-Fi": dict(cpu="high", hw=True, batt=90, net="fast", sec=128, prio="Fastest"),
    "Sensitive data on any device": dict(cpu="mid", hw=True, batt=80, net="fast", sec=256, prio="Max security"),
    "Small IoT sensor": dict(cpu="low", hw=False, batt=60, net="slow", sec=256, prio="Energy-saving"),
}


def apply_preset() -> None:
    values = PRESETS[st.session_state["preset"]]
    for k, v in (values or {}).items():
        st.session_state[f"sit_{k}"] = v


def encrypt_ui() -> None:
    # ---- Step 1: data ------------------------------------------------------------
    step(1, "Add your data")
    tab_text, tab_file = st.tabs(["Type a message", "Upload a file"])
    with tab_text:
        text = st.text_area("Your message", height=130, key="enc_text", placeholder="Type or paste text here...")
    with tab_file:
        up = st.file_uploader("Drag a file here (any type, up to 200 MB)", key="enc_file")
    if up is not None:
        data, fname = up.getvalue(), up.name
        mime = up.type or mimetypes.guess_type(up.name)[0] or "application/octet-stream"
        st.caption("Using the uploaded file. Remove it to use the typed message instead.")
    elif text:
        data, fname, mime = text.encode("utf-8"), "message.txt", "text/plain"
    else:
        st.info("Type a message or upload a file to continue.")
        return
    st.caption(f"Size to encrypt: **{human_size(len(data))}**")

    # ---- Step 2: situation ---------------------------------------------------------
    step(2, "Describe your situation")
    for k, v in dict(cpu="mid", hw=True, batt=60, net="medium", sec=128, prio="Balanced").items():
        st.session_state.setdefault(f"sit_{k}", v)
    st.selectbox("Quick start (optional)", list(PRESETS), key="preset", on_change=apply_preset)
    a, b = st.columns(2)
    with a:
        st.selectbox("Device processor", CPU_CLASSES, format_func=CPU_LABELS.get, key="sit_cpu")
        st.radio("Does the device have AES hardware acceleration?", [True, False], horizontal=True,
                 format_func=lambda x: "Yes" if x else "No", key="sit_hw",
                 help="Most phones and computers made in the last decade do. Small sensors often do not.")
        st.slider("Battery level (%)", 1, 100, key="sit_batt")
    with b:
        st.selectbox("Network", NETWORKS, format_func=lambda n: NETWORK_INFO[n][0], key="sit_net")
        st.radio("Security level you need", [128, 256], horizontal=True, key="sit_sec",
                 format_func=lambda x: f"{x}-bit " + ("(standard)" if x == 128 else "(maximum)"))
        st.radio("What matters most?", PRIORITIES, horizontal=True, key="sit_prio",
                 help="Balanced follows your battery level. Fastest and Energy-saving set how much speed vs "
                      "energy counts. Max security allows only 256-bit ciphers.")
    if st.session_state["sit_prio"] in ("Fastest", "Energy-saving"):
        st.caption("The battery level is not used while Fastest or Energy-saving is selected.")

    entered = Scenario(st.session_state["sit_cpu"], bool(st.session_state["sit_hw"]), float(st.session_state["sit_batt"]),
                       len(data), int(st.session_state["sit_sec"]), st.session_state["sit_net"])
    priority = st.session_state["sit_prio"]
    scen = apply_priority(entered, priority)   # what the cost model actually sees

    # ---- Step 3: recommendation ----------------------------------------------------
    step(3, "The AI's recommendation")
    bundle = ensure_model()
    frame = scen.to_frame()
    proba = rf_proba(bundle, frame).iloc[0]
    ai_choice = str(rf_secure(bundle, frame).iloc[0])
    confidence = float(proba[ai_choice]) * 100
    with st.container(border=True):
        st.caption("Recommended cipher")
        st.markdown(f'<p class="cs-big">{ai_choice}</p>', unsafe_allow_html=True)
        st.write(CIPHER_BLURB[ai_choice])
        st.markdown("**Why this one?**")
        for line in explain_decision(scen, ai_choice, priority, confidence, bundle["profile"],
                                     Assumptions(**bundle["assumptions"])):
            st.markdown(f"- {line}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**How sure is the AI?**")
        show_fig(fig_confidence(proba, ai_choice))
    with c2:
        st.markdown("**What the AI pays attention to (overall)**")
        imp = grouped_importances(bundle)
        if not imp.empty:
            show_fig(fig_factor_importance(imp))
    with st.expander("What would other methods choose for this situation?", expanded=True):
        picks = {k: str(v.iloc[0]) for k, v in select_all(frame, bundle).items()}
        show_table(comparison_table(scen, picks))

    # ---- Step 4: protect -----------------------------------------------------------
    step(4, "Protect your data")
    allowed = allowed_ciphers(scen.required_security)
    cipher = st.selectbox("Cipher to use (the AI's pick is pre-selected; change it to compare)", allowed,
                          index=allowed.index(ai_choice), key=f"cipher_{ai_choice}_{scen.required_security}")
    if cipher != ai_choice:
        st.caption(f"You are overriding the AI's recommendation ({ai_choice}).")
    how = st.radio("How should the data be locked?", ["Create a random key for me", "Use my password"], horizontal=True)
    password = None
    if how == "Use my password":
        password = password_box("Password", "enc_pw")
        if password and len(password) < 8:
            st.warning("Short passwords are easy to guess. Use 8 or more characters.")

    sig = data_signature(data, cipher, how, scen, priority)
    if st.button("Encrypt", type="primary"):
        if how == "Use my password" and not password:
            st.error("Please enter a password.")
        else:
            try:
                with st.spinner("Encrypting..."):
                    blob, key, enc_s = encrypt_file_or_data(
                        data, cipher, key_mode="password" if password else "random", password=password,
                        filename=fname, mime_type=mime)
                st.session_state["enc_result"] = dict(
                    sig=sig, blob=blob, key_b64=None if password else key_to_base64(key), cipher=cipher,
                    enc_s=enc_s, size=len(data), fname=fname, network=entered.network)
            except Exception as exc:
                st.error(f"Encryption failed: {exc}")

    res = st.session_state.get("enc_result")
    if res and res["sig"] == sig:
        st.success(f"Encrypted with {res['cipher']}.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Encryption time", human_time(res["enc_s"]),
                  help="In password mode this includes turning the password into a key.")
        m2.metric("Speed", f"{res['size'] / 1e6 / max(res['enc_s'], 1e-9):.1f} MB/s")
        m3.metric("Cipher overhead", f"{OVERHEAD_BYTES} bytes", help="12-byte nonce + 16-byte authentication tag.")
        m4.metric("Est. send time", human_time(network_delay_s(len(res["blob"]), res["network"])),
                  help=f"Modelled for: {NETWORK_INFO[res['network']][0]}")
        st.caption(f"File size: {human_size(res['size'])} becomes {human_size(len(res['blob']))} "
                   "(includes the file header and the hidden original filename).")
        d1, d2 = st.columns(2)
        d1.download_button("Download encrypted file", res["blob"], file_name=f"{res['fname']}.csai",
                           mime="application/octet-stream")
        if res["key_b64"]:
            st.warning("Save your key now. It is shown only here and cannot be recovered. "
                       "Without it the file can never be opened.")
            st.code(res["key_b64"], language=None)
            d2.download_button("Download key", res["key_b64"] + "\n", file_name=f"{res['fname']}.key.txt",
                               mime="text/plain")
        else:
            st.info("Password mode: there is no key file. Remember your password; it cannot be recovered.")


# ======================================================================================
# DECRYPT
# ======================================================================================
def decrypt_ui() -> None:
    step(1, "Upload the encrypted file")
    up = st.file_uploader("Encrypted file (.csai)", key="dec_file")
    if up is None:
        st.info("Upload a file that was encrypted with CryptoSelect AI.")
        return
    blob = up.getvalue()
    try:
        header, _ = parse_container_header(blob)
    except InvalidContainerError as exc:
        st.error(f"This is not a valid CryptoSelect AI file. {exc}")
        return
    st.success(f"Detected cipher: **{header.cipher_name}**. Locked with: **{header.key_mode_name}**")
    st.caption("The cipher is read from the file itself. The AI is not involved in decryption.")

    step(2, "Enter the key or password")
    password, key_text = None, ""
    if header.key_mode == KEY_MODE_PASSWORD:
        password = password_box("Password", "dec_pw")
    else:
        key_text = st.text_input("Key (paste it here)", key="dec_key")
        kf = st.file_uploader("...or upload the key file", key="dec_keyfile")
        if kf is not None:
            key_text = kf.getvalue().decode("utf-8", errors="ignore")

    sig = data_signature(blob, password, key_text)
    if st.button("Decrypt", type="primary"):
        try:
            key = None if password is not None else parse_key(key_text)
            if password is not None and not password:
                raise ValueError("Please enter the password.")
            with st.spinner("Decrypting..."):
                raw, meta, hdr, dec_s = decrypt_file_or_data(blob, password=password, key=key)
            st.session_state["dec_result"] = dict(sig=sig, raw=raw, meta=meta, cipher=hdr.cipher_name, dec_s=dec_s)
        except CryptoAuthenticationError:
            st.session_state.pop("dec_result", None)
            st.error("Authentication failed: the key or password is wrong, or the file was changed or damaged. "
                     "Nothing was decrypted.")
        except (InvalidContainerError, ValueError) as exc:
            st.session_state.pop("dec_result", None)
            st.error(str(exc))
        except Exception:
            st.session_state.pop("dec_result", None)
            st.error("Could not decrypt this file. Please check the file, key or password and try again.")

    res = st.session_state.get("dec_result")
    if res and res["sig"] == sig:
        meta = res["meta"]
        st.success("Decrypted and verified. The data is authentic and unchanged.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Cipher", res["cipher"])
        m2.metric("Decryption time", human_time(res["dec_s"]))
        m3.metric("Size", human_size(len(res["raw"])))
        name = meta.get("filename", "decrypted.bin")
        st.caption(f"Original file name: **{name}**")
        if str(meta.get("mime_type", "")).startswith("text/") and len(res["raw"]) < 1_000_000:
            try:
                st.text_area("Message", res["raw"].decode("utf-8"), height=200, disabled=True)
            except UnicodeDecodeError:
                pass
        st.download_button("Download decrypted file", res["raw"], file_name=name, mime=meta.get("mime_type"))


encrypt_ui() if mode == "Encrypt" else decrypt_ui()
