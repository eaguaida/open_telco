# Open Telco

Collection of evals on telecommunications tasks.

Read more: [GSMA Benchmarks Blog Post](https://huggingface.co/blog/otellm/gsma-benchmarks-02)

## Dataset Access Requirements

Before using this repository, you must request permission for the benchmark datasets on HuggingFace:

- [TeleQnA Dataset](https://huggingface.co/datasets/netop/TeleQnA)
- [TeleMath Dataset](https://huggingface.co/datasets/netop/TeleMath)
- [TeleLogs Dataset](https://huggingface.co/datasets/netop/TeleLogs)

**HuggingFace Configuration:**

1. Get your access token from your HuggingFace account
2. Add the above repositories to "Repositories permissions"
3. Click "read access to contents of selected repos"

## Prerequisites

**Docker or OrbStack** (required for sandbox execution)

- Docker: [https://www.docker.com/get-started](https://www.docker.com/get-started)
- OrbStack (Mac): [https://orbstack.dev](https://orbstack.dev)

**uv package manager**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

1. **Install dependencies:**

```bash
uv sync
```

2. **Configure environment variables:**

Create a `.env` file in the root folder with your API credentials:

```bash
# Required: HuggingFace token for dataset access
HF_TOKEN=your_huggingface_token_here

# Add API keys for the models you want to use
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

List of all available models: [https://inspect.aisi.org.uk/models.html](https://inspect.aisi.org.uk/models.html)

## Usage

Run evals from the command line:

```bash
# TeleQnA
uv run inspect eval src/open_telco/teleqna/teleqna.py
#TeleMath
uv run inspect eval src/open_telco/telemath/telemath.py
#TeleLogs
uv run inspect eval src/open_telco/telelogs/telelogs.py
```

With options:

```bash
# Specific model
uv run inspect eval src/open_telco/telemath/telemath.py --model openai/gpt-4o

# Limit samples
uv run inspect eval src/open_telco/telemath/telemath.py --limit 10
```

**Alternative**: Use the [Inspect VS Code Extension](https://marketplace.visualstudio.com/items?itemName=inspect-ai.inspect) or run the Web UI with `python ui/app.py`

## Evals

### Knowledge & QA

- **TeleQnA: Benchmark Dataset to Assess Large Language Models for Telecommunications**
  A benchmark dataset of 10,000 question-answer pairs sourced from telecommunications standards and research articles. Evaluates LLMs' knowledge across general telecom inquiries and complex standards-related questions.
  [Paper](https://arxiv.org/abs/2310.15051) | [Dataset](https://huggingface.co/datasets/netop/TeleQnA)

  ```bash
  uv run inspect eval src/open_telco/teleqna/teleqna.py
  ```

### Mathematical Reasoning

- **TeleMath: Evaluating Mathematical Reasoning in Telecom Domain**
  500 mathematically intensive problems covering signal processing, network optimization, and performance analysis. Implemented as a ReAct agent using bash and python tools to solve domain-specific mathematical computations.
  [Paper](https://arxiv.org/abs/2506.10674) | [Dataset](https://huggingface.co/datasets/netop/TeleMath)

  ```bash
  uv run inspect eval src/open_telco/telemath/telemath.py
  ```

  *Metrics: pass@1, const@16 (majority voting over 16 answers)*

### Network Operations & Diagnostics

- **TeleLogs: Root Cause Analysis in 5G Networks**
  A synthetic dataset for root cause analysis (RCA) in 5G networks. Given network configuration parameters and user-plane data (throughput, RSRP, SINR), models must identify which of 8 predefined root causes explain throughput degradation below 600 Mbps. Use `-T <N>` to specify epochs for pass@1 and maj@4 metrics.
  [Paper](https://arxiv.org/abs/2507.21974) | [Dataset](https://huggingface.co/datasets/netop/TeleLogs)

  ```bash
  uv run inspect eval src/open_telco/telelogs/telelogs.py -T 4
  ```

  *Metrics: pass@1 (averaged over N epochs), maj@4 (majority voting)*
