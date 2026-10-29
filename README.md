# GENESIS GPU

**An AI GPU engineer that measures its way to better kernels.**

GENESIS GPU is a hackathon prototype for evidence-driven GPU kernel search. NVIDIA Nemotron, accessed through Nebius Token Factory, is intended to propose a small set of Triton launch configurations. On a compatible CUDA GPU, the evaluator compiles each configuration, checks its output against PyTorch, measures correct candidates, records the run, and selects the fastest measured candidate.

## Current Scope and Status

- **Workload:** FP32 vector addition.
- **Bounded generation:** The model proposes launch parameters only. Model-generated source code is never executed.
- **Correctness first:** Candidates must pass numerical validation before their performance is considered.
- **Hardware-aware evaluation:** The evaluator does not claim GPU timings without actual GPU execution.
- **Interactive demo:** The Streamlit dashboard provides illustrative configurations and local validation when live services or compatible hardware are unavailable.

**Important:** Demo configurations are illustrative examples. They are not live Nemotron output and have not been benchmarked on a GPU. Live inference and real GPU measurements must be verified by running the application with the required credentials and hardware.

## Requirements

- Python 3.10 or newer.
- A compatible Linux environment with an NVIDIA GPU, CUDA-enabled PyTorch, and Triton.
- Nebius Token Factory access and a `NEBIUS_API_KEY` for live model inference.

A Linux GPU machine or compatible GPU cloud environment is recommended. Windows machines without a supported CUDA/Triton environment should use a Linux GPU machine instead of attempting local GPU benchmarking.

## Setup

### Recommended: Linux GPU machine

Open the project folder in VS Code and run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the CUDA-enabled PyTorch build appropriate for the machine using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

Then install the project dependencies:

```bash
pip install -r requirements.txt
```

Set your Token Factory API key privately:

```bash
export NEBIUS_API_KEY="your-token-factory-key"
```

Never commit API keys or other secrets to GitHub.

Verify GPU access:

```bash
python -c "import torch, triton; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"
```

Continue with measured benchmarking only if CUDA is available and the expected GPU is detected.

Launch the dashboard:

```bash
streamlit run app.py
```

Streamlit normally prints a local URL such as `http://localhost:8501`.

### Windows

For GPU benchmarking, use WSL 2 with a compatible NVIDIA GPU and driver, or use a Linux GPU cloud machine. Follow the [official NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/) for current setup instructions.

For a process running directly in Windows PowerShell, an API key can be set with:

```powershell
$env:NEBIUS_API_KEY="your-token-factory-key"
```

For WSL Ubuntu, set the key inside the Ubuntu terminal:

```bash
export NEBIUS_API_KEY="your-token-factory-key"
```

If the GPU is not detected by PyTorch, do not report benchmark results.

## Model Configuration

The default model ID is `nvidia/Nemotron-3_5-Lightning`.

The model can be overridden with `NEBIUS_MODEL` if a different eligible model is available to your account. The Token Factory endpoint can be configured using `NEBIUS_BASE_URL`.

Successful model access depends on account permissions, model availability, and a valid API key.

## Reproduce a Real Run

1. Confirm that the application has access to a valid Token Factory API key and a compatible CUDA GPU.
2. Choose the vector length, model proposal count, and benchmark repeat count.
3. Start a run. The model returns JSON launch settings restricted to an allowlist.
4. The evaluator compiles candidates, checks their outputs against a PyTorch reference, and times only correct candidates.
5. Inspect the candidate results and download the run JSON.
6. Review the recorded hardware and benchmark details before reporting performance.

Local run records are stored under `runs/`, where applicable.

**Do not present demo configurations or illustrative values as measured performance.** Report only results obtained from actual GPU execution.

## Architecture

- `app.py` — Streamlit dashboard, access checks, demo explorer, run results, and JSON download.
- `genesis/model.py` — Nebius Token Factory client and launch-configuration validation.
- `genesis/engine.py` — Triton candidate execution, PyTorch correctness checks, benchmarking, and best-candidate selection.

## Intended Hackathon Track

**Coding and Agentic Engineering**

The intended workflow uses NVIDIA Nemotron through Nebius Token Factory to propose bounded kernel launch configurations, followed by programmatic validation and measurement.

The live API integration, model availability, and GPU execution must be tested before claiming that the complete workflow has been demonstrated.

See the [hackathon overview and submission requirements](https://nebiusglobalaihackathon.devpost.com/) and [official rules](https://nebiusglobalaihackathon.devpost.com/rules).

## Limitations

- Live model inference and GPU measurements have not been verified merely by opening the demo.
- The candidate search currently focuses on launch parameters for one workload.
- The project is a bounded prototype, not a system for executing arbitrary AI-generated kernel code.
- Performance depends on the GPU, driver, CUDA, PyTorch, Triton, and input size.
- A public deployment, demo video, and final submission must be completed separately if required by the hackathon.

## References

- [Nebius Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/)
- [Hackathon rules](https://nebiusglobalaihackathon.devpost.com/rules)
- [Hackathon FAQ](https://nebiusglobalaihackathon.devpost.com/details/faqs)
- [NVIDIA Nemotron models on Nebius Token Factory](https://nebius.com/services/token-factory/models/nvidia-nemotron-models-inference)
- [Nebius Token Factory documentation](https://docs.tokenfactory.nebius.com/)
- [PyTorch installation selector](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/)

## License

See the repository's `LICENSE` file for the applicable license.
