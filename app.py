from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from genesis.engine import run_optimization
from genesis.model import ModelUnavailable, propose_configs


st.set_page_config(page_title="GENESIS GPU", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #080b10; color: #e8edf5; }
[data-testid="stHeader"] { background: transparent; }
.hero { border: 1px solid #202a37; border-radius: 18px; padding: 28px 32px; background: radial-gradient(ellipse at 8% 5%, #163929 0%, #101923 45%, #0c1119 100%); margin-bottom: 22px; }
.eyebrow { color: #7ee787; font-family: 'DM Mono', monospace; font-size: 12px; letter-spacing: .14em; text-transform: uppercase; }
.hero h1 { font-size: 38px; margin: 8px 0; letter-spacing: -.04em; color: #f4f7fb; }
.hero p { color: #aab7c8; margin: 0; max-width: 720px; }
.metric { border: 1px solid #202a37; border-radius: 13px; padding: 16px; background: #0d131c; min-height: 88px; }
.metric-label { color: #91a0b2; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
.metric-value { color: #f4f7fb; font-size: 24px; font-weight: 700; margin-top: 5px; }
code, pre { font-family: 'DM Mono', monospace !important; }
</style>
<div class="hero"><div class="eyebrow">NVIDIA × Nebius · Evidence-driven kernel search</div>
<h1>GENESIS GPU <span style="color:#7ee787">/</span></h1>
<p>An AI GPU engineer that measures its way to better kernels. Nemotron proposes launch configurations; the GPU decides correctness and speed.</p></div>
""", unsafe_allow_html=True)

api_key = os.getenv("NEBIUS_API_KEY")
gpu_available = False
gpu_name = "Not detected"
try:
    import torch
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "No CUDA GPU detected"
except Exception:
    pass

cols = st.columns(3)
for col, label, value, ok in [
    (cols[0], "Nebius Token Factory", "Connected" if api_key else "API key missing", bool(api_key)),
    (cols[1], "GPU runner", gpu_name, gpu_available),
    (cols[2], "Model", os.getenv("NEBIUS_MODEL", "nvidia/Nemotron-3_5-Lightning"), True),
]:
    col.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value" style="font-size:18px;color:{"#7ee787" if ok else "#ffb86b"}">{value}</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1, 2], gap="large")
with left:
    st.subheader("Optimization run")
    size = st.select_slider("Vector length", options=[4096, 16384, 65536, 262144, 1048576], value=262144)
    attempts = st.slider("Nemotron proposals", 1, 4, 3)
    repeats = st.slider("Benchmark sample duration (ms)", 20, 100, 50, step=10)
    run = st.button("⚡  Start measured run", type="primary", use_container_width=True)
    st.caption("Only GPU-executed, numerically correct candidates may be selected. No demo data is generated.")
    if not api_key:
        st.warning("Set NEBIUS_API_KEY to enable the required live Nemotron call.")
    if not gpu_available:
        st.warning("A CUDA GPU is required for real kernel compilation and timing. CPU timings are never reported as GPU results.")

with right:
    st.subheader("Run evidence")
    if run:
        if not api_key:
            st.error("Run stopped: NEBIUS_API_KEY is not configured. No model call was made.")
        elif not gpu_available:
            st.error("Run stopped: CUDA GPU is unavailable. No GPU benchmark was claimed.")
        else:
            try:
                with st.spinner("Asking Nemotron for bounded Triton launch configurations…"):
                    proposals = propose_configs(size=size, attempts=attempts)
                with st.spinner("Compiling, validating, and measuring on the detected GPU…"):
                    result = run_optimization(size=size, proposals=proposals, repeats=repeats)
                st.session_state["last_run"] = result
                st.success("Run completed on the detected GPU.")
            except ModelUnavailable as exc:
                st.error(f"Nemotron call failed: {exc}")
            except Exception as exc:
                st.error(f"Evaluation stopped: {type(exc).__name__}: {exc}")

    result = st.session_state.get("last_run")
    if result:
        best = result.get("best")
        if best:
            a, b, c = st.columns(3)
            a.metric("Best Triton latency", f'{best["latency_us"]:.2f} μs')
            b.metric("Verified candidates", f'{result["verified_count"]}/{len(result["candidates"])}')
            c.metric("Speedup vs PyTorch", f'{result["speedup_vs_torch"]:.2f}×' if result.get("speedup_vs_torch") else "n/a")
        st.dataframe(result["candidates"], use_container_width=True, hide_index=True)
        with st.expander("Benchmark methodology"):
            st.json(result["hardware"])
            st.write(f'GPU: {result["hardware"]["gpu"]}. PyTorch baseline and correct Triton candidates use Triton do_bench with a {repeats} ms measurement window. Correctness is checked before a candidate is timed.')
        st.download_button("Download run record", json.dumps(result, indent=2), file_name="genesis-run.json", mime="application/json")
    else:
        st.info("No run recorded in this session. Start a run after configuring both Nebius access and a CUDA GPU.")

st.divider()
st.caption("GENESIS GPU · Nemotron is the proposal engine. Correctness checks and timings come from the independent evaluator.")
