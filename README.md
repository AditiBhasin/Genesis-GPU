# GENESIS GPU

**An AI GPU engineer that measures its way to better kernels.**

GENESIS GPU is a hackathon prototype for evidence-driven kernel search. NVIDIA Nemotron, called through Nebius Token Factory, proposes a small set of Triton launch configurations. The application compiles each configuration on a CUDA GPU, checks results against PyTorch, measures correct candidates, records the run, and selects the fastest measured candidate.

## Current scope and honest status

- Workload: FP32 vector addition.
- Nemotron proposes bounded launch parameters only. Model-generated source code is never executed.
- The evaluator refuses to report GPU timings without a CUDA device.
- The repository currently contains the implementation scaffold; no live inference, GPU run, deployment, or performance result is claimed until you configure access and execute it.

## Requirements

- Python 3.10 or newer.
- A Linux environment with a compatible NVIDIA GPU, CUDA-enabled PyTorch, and Triton. A GPU notebook or Nebius GPU machine can be used.
- Nebius Token Factory access and a `NEBIUS_API_KEY`.

Windows machines without a CUDA-compatible Triton environment should run this project in a Linux GPU notebook or GPU cloud instance. The Streamlit UI can run on another host, but the evaluator process must have access to the GPU.

## Setup

### Recommended for the deadline: Linux GPU machine

Use a Linux machine with an NVIDIA GPU (a GPU cloud machine is fine). Open the extracted project folder in VS Code, open its terminal, and run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the CUDA-enabled PyTorch build that matches this GPU environment
pip install torch
pip install -r requirements.txt
export NEBIUS_API_KEY="your-token-factory-key"  # keep this private; do not commit it
streamlit run app.py
```

The `pip install torch` command above is suitable for many standard Linux CUDA environments, but if PyTorch recommends a specific CUDA wheel for this machine, use the command from the [official PyTorch installer selector](https://pytorch.org/get-started/locally/) instead. Then run `pip install -r requirements.txt`.

Before opening the dashboard, verify GPU access:

```bash
python -c "import torch, triton; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"
```

Continue only if it prints `CUDA: True` and the actual GPU model. The dashboard starts at the local URL printed in the terminal, usually `http://localhost:8501`.

### If using Windows

The shortest supported route is to run the project in **WSL 2 Ubuntu** with an NVIDIA GPU. Install/update the NVIDIA driver on Windows, then install WSL 2 and Ubuntu. In WSL, do not install a Linux NVIDIA display driver; the Windows driver supplies GPU access. NVIDIA's [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/) has the current steps. Open the project folder from VS Code using the WSL extension, then follow the Linux steps above. If `nvidia-smi` or the PyTorch check cannot see the GPU, switch to a Linux GPU cloud machine rather than spending the remaining hackathon time on driver setup.

In PowerShell, an API key for a process running directly in Windows can be set with `$env:NEBIUS_API_KEY="your-token-factory-key"`. For WSL, use `export NEBIUS_API_KEY="your-token-factory-key"` inside the Ubuntu terminal.

The default model ID is `nvidia/Nemotron-3_5-Lightning`; override it with `NEBIUS_MODEL` if your account's live catalog lists a different eligible model. The Token Factory endpoint is configurable via `NEBIUS_BASE_URL`.

## Reproduce a run

1. Confirm the dashboard reports a connected Token Factory key and the actual CUDA GPU model.
2. Choose vector length, model proposal count, and repeated benchmark count.
3. Start a run. Nemotron returns JSON launch settings, which are restricted to a fixed allowlist.
4. The evaluator compiles each candidate, validates output against PyTorch, and times only correct candidates.
5. Inspect the candidate table and download the run JSON. Local run records are stored under `runs/`.

Do not copy sample/demo timings into a submission. Submit only a run record produced on real GPU hardware.

## Architecture

- `genesis/model.py`: OpenAI-compatible Nebius Token Factory call and strict config validation.
- `genesis/engine.py`: Triton candidate compilation, PyTorch correctness check, repeated timing, and best-candidate selection.
- `app.py`: Streamlit dashboard, access checks, actual run results, and JSON download.

## Hackathon fit

The intended track is **Coding and Agentic Engineering**. The official challenge requires a working application that makes a runtime call to Nebius Token Factory or runs on Nebius AI Cloud, and uses at least one NVIDIA open-source model. GENESIS uses NVIDIA Nemotron 3.5 Lightning through Token Factory; actual availability and successful inference must be verified using the team account before submission.

## Limitations

- No live call or GPU measurements are included in this repository yet.
- Candidate search is limited to launch parameters for one workload; this is a safe, bounded prototype rather than arbitrary AI-authored kernel execution.
- Performance varies with GPU, clocks, driver, CUDA, PyTorch, Triton, and input size. Always report hardware and versions from the run record.
- A public deployment, video, and Devpost submission still need to be created by the team.

## References

- [Hackathon overview and submission requirements](https://nebiusglobalaihackathon.devpost.com/)
- [Official rules](https://nebiusglobalaihackathon.devpost.com/rules)
- [FAQ](https://nebiusglobalaihackathon.devpost.com/details/faqs)
- [NVIDIA Nemotron models on Nebius Token Factory](https://nebius.com/services/token-factory/models/nvidia-nemotron-models-inference)
- [Nebius Token Factory documentation](https://docs.tokenfactory.nebius.com/)

## License

Apache-2.0. See [LICENSE](LICENSE).
