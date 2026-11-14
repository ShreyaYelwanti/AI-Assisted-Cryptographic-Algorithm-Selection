"""Speed tests, following the three procedures in the research notebook.

1. micro      : encrypt + decrypt time, throughput and peak memory (30 runs, 1 warm-up)
2. medians    : encrypt / decrypt medians (100 runs, 5 warm-ups); the simulation is built from these
3. transfer   : encrypt -> send over a local socket -> receive -> decrypt, with modelled network delay

Only the `cryptography` library does the encryption; nothing here is custom cryptography.
"""
from __future__ import annotations

import os
import platform
import socket
import threading
import time
import tracemalloc
from typing import Any, Callable

import cryptography
import numpy as np
import pandas as pd
import sklearn
from scipy import stats

from core.config import CIPHERS, RESULTS_DIR
from core.crypto import make_cipher, raw_decrypt, raw_encrypt

SIZES: dict[str, int] = {"1KB": 1_024, "10KB": 10_240, "100KB": 102_400, "1MB": 1_048_576, "10MB": 10_485_760}
TRANSFER_NETWORKS: dict[str, float] = {"slow": 1.0, "medium": 10.0, "fast": 100.0}   # Mbit/s, simulated
Progress = Callable[[int, int, str], None]


def get_system_specs() -> dict[str, Any]:
    """Details of the computer that ran the tests."""
    return {
        "Operating system": f"{platform.system()} {platform.release()}",
        "Python": platform.python_version(),
        "Processor": platform.processor() or platform.machine(),
        "CPU cores (logical)": os.cpu_count() or 1,
        "cryptography library": cryptography.__version__,
        "scikit-learn": sklearn.__version__,
    }


def _ci(x: pd.Series) -> tuple[float, float]:
    if len(x) < 2 or x.std() == 0:
        return float(x.mean()), float(x.mean())
    lo, hi = stats.t.interval(0.95, len(x) - 1, loc=np.mean(x), scale=stats.sem(x))
    return float(lo), float(hi)


# ---------------------------------------------------------------------------------------
# 1. micro benchmark
# ---------------------------------------------------------------------------------------
def run_micro(runs: int, tick: Callable[[str], None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for name in CIPHERS:
        c = make_cipher(name)
        for label, size in SIZES.items():
            data = os.urandom(size)
            raw_encrypt(c, data)                                   # warm-up, not recorded
            tracemalloc.start()                                    # peak memory, measured once separately
            raw_decrypt(c, raw_encrypt(c, data))
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            for r in range(runs):
                t0 = time.perf_counter()
                blob = raw_encrypt(c, data)
                t1 = time.perf_counter()
                out = raw_decrypt(c, blob)
                t2 = time.perf_counter()
                assert out == data
                rows.append(dict(cipher=name, size_label=label, size=size, run=r, enc_s=t1 - t0, dec_s=t2 - t1,
                                 thr_MBps=(size / 1e6) / (t2 - t0), overhead_bytes=len(blob) - size,
                                 peak_mem_bytes=peak))
            if tick:
                tick(f"{name}, {label}")
    raw = pd.DataFrame(rows)
    g = raw.groupby(["cipher", "size_label", "size"])
    summary = g.agg(enc_mean=("enc_s", "mean"), enc_std=("enc_s", "std"), dec_mean=("dec_s", "mean"),
                    dec_std=("dec_s", "std"), thr_mean=("thr_MBps", "mean"), thr_std=("thr_MBps", "std"),
                    peak_mem=("peak_mem_bytes", "first")).reset_index()
    ci = raw.groupby(["cipher", "size_label", "size"])["thr_MBps"].apply(_ci)
    summary["thr_ci_low"] = [ci[(r.cipher, r.size_label, r.size)][0] for r in summary.itertuples()]
    summary["thr_ci_high"] = [ci[(r.cipher, r.size_label, r.size)][1] for r in summary.itertuples()]
    cols = ["cipher", "size_label", "size", "enc_mean", "enc_std", "dec_mean", "dec_std", "thr_mean", "thr_std",
            "thr_ci_low", "thr_ci_high", "peak_mem"]
    return raw, summary.sort_values(["cipher", "size"])[cols]


# ---------------------------------------------------------------------------------------
# 2. medians (feeds the simulation)
# ---------------------------------------------------------------------------------------
def run_medians(runs: int, warmups: int, tick: Callable[[str], None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for name in CIPHERS:
        c = make_cipher(name)
        for label, size in SIZES.items():
            data = os.urandom(size)
            for _ in range(warmups):
                raw_encrypt(c, data)
            for r in range(runs):
                t0 = time.perf_counter()
                blob = raw_encrypt(c, data)
                t1 = time.perf_counter()
                out = raw_decrypt(c, blob)
                t2 = time.perf_counter()
                assert out == data
                rows.append(dict(cipher=name, size_label=label, size=size, run=r, enc_us=(t1 - t0) * 1e6,
                                 dec_us=(t2 - t1) * 1e6, thr_MBps=(size / 1e6) / (t2 - t0)))
            if tick:
                tick(f"{name}, {label}")
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["cipher", "size_label", "size"]).agg(
        enc_median_us=("enc_us", "median"), dec_median_us=("dec_us", "median"), thr_median=("thr_MBps", "median"),
        thr_mean=("thr_MBps", "mean"), thr_std=("thr_MBps", "std")).reset_index().sort_values(["cipher", "size"])
    return raw, summary


# ---------------------------------------------------------------------------------------
# 3. local-socket transfer
# ---------------------------------------------------------------------------------------
def _recv_all(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("connection closed early")
        buf.extend(chunk)
    return bytes(buf)


def _transfer(name: str, size: int, runs: int) -> tuple[list[tuple[float, int]], list[float]]:
    c = make_cipher(name)
    data = os.urandom(size)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    dec_times: list[float] = []
    errors: list[str] = []

    def server() -> None:
        try:
            conn, _ = srv.accept()
            with conn:
                for _ in range(runs):
                    n = int.from_bytes(_recv_all(conn, 4), "big")
                    blob = _recv_all(conn, n)
                    t = time.perf_counter()
                    pt = raw_decrypt(c, blob)
                    dec_times.append(time.perf_counter() - t)
                    assert pt == data
                    conn.sendall(b"k")
        except Exception as exc:  # reported to the caller instead of hanging
            errors.append(repr(exc))

    th = threading.Thread(target=server, daemon=True)
    th.start()
    cli = socket.create_connection(srv.getsockname())
    out = []
    try:
        for _ in range(runs):
            t0 = time.perf_counter()
            blob = raw_encrypt(c, data)
            t_enc = time.perf_counter() - t0
            cli.sendall(len(blob).to_bytes(4, "big") + blob)
            cli.recv(1)
            out.append((t_enc, len(blob)))
    finally:
        cli.close()
        th.join(timeout=15)
        srv.close()
    if errors:
        raise RuntimeError(f"Transfer test failed: {errors[0]}")
    return out, dec_times


def run_transfer(runs: int, tick: Callable[[str], None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for name in CIPHERS:
        for label, size in SIZES.items():
            res, dec = _transfer(name, size, runs)
            for (t_enc, wire), t_dec in zip(res, dec):
                for net, mbps in TRANSFER_NETWORKS.items():
                    delay = (wire * 8) / (mbps * 1e6)
                    rows.append(dict(cipher=name, size_label=label, size=size, network=net, enc_s=t_enc, dec_s=t_dec,
                                     wire_bytes=wire, total_s=t_enc + delay + t_dec))
            if tick:
                tick(f"{name}, {label}")
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["network", "size_label", "cipher"])["total_s"].agg(["median", "std"]).round(6).reset_index()
    return raw, summary


# ---------------------------------------------------------------------------------------
# everything
# ---------------------------------------------------------------------------------------
def run_all(quick: bool = False, progress: Progress | None = None) -> dict[str, pd.DataFrame]:
    """Run the three tests. Full = 30 / 100 / 30 runs; quick = 5 / 10 / 5 runs."""
    micro_runs, med_runs, med_warm, xfer_runs = (5, 10, 2, 5) if quick else (30, 100, 5, 30)
    per = len(CIPHERS) * len(SIZES)
    total, done = 3 * per, 0

    def tick(msg: str) -> None:
        nonlocal done
        done += 1
        if progress:
            progress(done, total, msg)

    raw1, sum1 = run_micro(micro_runs, tick)
    raw2, sum2 = run_medians(med_runs, med_warm, tick)
    raw3, sum3 = run_transfer(xfer_runs, tick)
    return {"micro_raw": raw1, "micro_summary": sum1, "median_raw": raw2, "median_summary": sum2,
            "transfer_raw": raw3, "transfer_summary": sum3,
            "runs": pd.DataFrame([{"micro": micro_runs, "medians": med_runs, "transfer": xfer_runs}])}


def save_results(res: dict[str, pd.DataFrame], out_dir: str | os.PathLike = RESULTS_DIR) -> None:
    """Write the CSV files (names match the notebook)."""
    os.makedirs(out_dir, exist_ok=True)
    names = {"micro_raw": "benchmark_raw.csv", "micro_summary": "benchmark_summary.csv",
             "median_raw": "benchmark_raw_v2.csv", "median_summary": "benchmark_summary_v2.csv",
             "transfer_raw": "e2e_raw.csv"}
    for key, fname in names.items():
        res[key].to_csv(os.path.join(out_dir, fname), index=False)
