from __future__ import annotations

import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path


def run_optimization(*, size: int, proposals: list[dict], repeats: int) -> dict:
    import torch
    import triton
    import triton.language as tl

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable; refusing to run CPU timings as GPU results.")

    @triton.jit
    def add_kernel(X, Y, Z, N: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(X + offsets, mask=mask, other=0)
        y = tl.load(Y + offsets, mask=mask, other=0)
        tl.store(Z + offsets, x + y, mask=mask)

    torch.manual_seed(7)
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    y = torch.randn(size, device="cuda", dtype=torch.float32)
    reference = x + y
    reference_latency_ms = triton.testing.do_bench(lambda: torch.add(x, y), rep=repeats)
    candidates: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for index, proposal in enumerate(proposals, start=1):
        block, warps = int(proposal["block_size"]), int(proposal["num_warps"])
        entry = {"candidate": index, "block_size": block, "num_warps": warps, "rationale": proposal.get("rationale", ""), "correct": False, "latency_us": None, "status": "proposed"}
        if (block, warps) in seen:
            entry["status"] = "duplicate configuration"
            candidates.append(entry)
            continue
        seen.add((block, warps))
        out = torch.empty_like(x)
        try:
            add_kernel[(triton.cdiv(size, block),)](x, y, out, size, block, num_warps=warps)
            torch.cuda.synchronize()
            entry["correct"] = bool(torch.allclose(out, reference, rtol=1e-5, atol=1e-6))
            if not entry["correct"]:
                entry["status"] = "incorrect; excluded"
            else:
                samples = triton.testing.do_bench(lambda: add_kernel[(triton.cdiv(size, block),)](x, y, out, size, block, num_warps=warps), rep=repeats)
                entry["latency_us"] = round(float(samples) * 1000, 4)
                entry["status"] = "verified and measured"
        except Exception as exc:
            entry["status"] = f"compile/execute failed: {type(exc).__name__}: {exc}"[:240]
        candidates.append(entry)

    verified = [c for c in candidates if c["correct"] and c["latency_us"] is not None]
    best = min(verified, key=lambda c: c["latency_us"]) if verified else None
    reference_latency_us = round(float(reference_latency_ms) * 1000, 4)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workload": "vector_add_fp32",
        "size": size,
        "benchmark_window_ms": repeats,
        "pytorch_reference_latency_us": reference_latency_us,
        "speedup_vs_torch": round(reference_latency_us / best["latency_us"], 4) if best and best["latency_us"] else None,
        "model": __import__("os").getenv("NEBIUS_MODEL", "nvidia/Nemotron-3_5-Lightning"),
        "hardware": {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "torch": torch.__version__, "triton": triton.__version__, "platform": platform.platform()},
        "candidates": candidates,
        "verified_count": len(verified),
        "best": best,
        "selection_rule": "minimum measured latency among numerically correct candidates",
    }
    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    record["record_path"] = str(path)
    return record
