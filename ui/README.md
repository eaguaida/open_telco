# Telecom Bench UI

A minimalist neobrutalism-style web interface for running Telecom Bench evaluations.

## Quick Start

### Prerequisites
- Python 3.13+ (for native type hint support)

### Setup

1. Create and activate a virtual environment:
```bash
python3.13 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python app.py
```

4. Open your browser to:
```
http://localhost:5001
```

> **Note:** Port 5001 is used instead of 5000 to avoid conflicts with macOS AirPlay Receiver (ControlCenter)

## Features

- 🎨 Clean neobrutalism design
- 🚀 Real-time streaming output
- 🎯 **Two Task Types**:
  - **📐 TeleMath (Agent)**: Mathematical problems solved with agent-based reasoning
  - **🎯 TeleQnA (Multiple Choice)**: Knowledge assessment with multiple choice questions
- ⚙️ Configure evaluation parameters:
  - Provider selection (Anthropic, OpenAI, Fireworks, Google)
  - Model selection
  - Difficulty level (for TeleMath only)
  - Sample limits
  - Max connections
  - Max tokens
  - Temperature
- 📊 Live status updates
- 📁 **Advanced Log Viewer** to inspect past evaluations:
  - View all evaluation runs grouped by type
  - Navigate through trajectories (individual samples)
  - Visualize agent reasoning step-by-step
  - See tool calls and responses in separate boxes
  - View submissions and test results
  - Navigate between trajectories with prev/next buttons
  - Neobrutalism-styled trajectory visualization

## Configuration Options

Based on [Inspect AI Options](https://inspect.aisi.org.uk/options.html):

- **Model**: Model identifier (e.g., `anthropic/claude-3-5-sonnet-20241022`)
- **Difficulty**: Filter by difficulty level (basic, intermediate, advanced, or full) - *Only applies to TeleMath*
- **Sample Limit**: Maximum number of samples to evaluate
- **Max Connections**: Concurrent connections to model provider (default: 10)
- **Max Tokens**: Maximum tokens per completion
- **Temperature**: Sampling temperature (0-2)

## Tasks

### TeleMath (Agent-based)
- Uses ReAct agent with tools (bash, python)
- Solves mathematical telecommunications problems
- Supports difficulty filtering
- Runs in Docker sandbox

### TeleQnA (Multiple Choice)
- Uses multiple_choice solver with Chain of Thought
- Tests telecommunications knowledge
- Multiple choice format (4-5 options)
- No difficulty filtering (uses all categories)

## How It Works

The UI runs `inspect eval` commands in the background and streams the output to your browser in real-time. All parameters are passed as command-line arguments to the Inspect AI CLI. Select which task to run using the two buttons below the configuration form.

## Log Format

All evaluations are automatically run with `--log-format=json` to enable the advanced trajectory visualization feature. The logs are stored in both `.json` format (for new detailed visualizations) and `.eval` format (for backward compatibility).

### Trajectory Visualization

The log viewer provides a multi-level navigation:

1. **Runs List**: View all evaluation runs grouped by task type
2. **Run Detail**: See all trajectories (samples) within a run with pass/fail badges
3. **Trajectory Detail**: Dive deep into each trajectory showing:
   - Task description
   - Agent reasoning (each thought in its own box)
   - Tool calls (Python, bash, submit, etc.) with syntax highlighting
   - Tool responses with output
   - Step-by-step navigation
   
The visualization maintains the neobrutalism aesthetic with bold borders, bright colors, and clear separation between different types of actions.

## 📚 Documentation

For frontend engineers continuing this project, comprehensive documentation is available:

- **[QUICK_REFERENCE.md](../QUICK_REFERENCE.md)** - Quick reference guide with cheat sheets and common tasks
- **[FRONTEND_ARCHITECTURE.md](../FRONTEND_ARCHITECTURE.md)** - Complete architecture guide with diagrams and API reference
- **[FLOWCHARTS.md](../FLOWCHARTS.md)** - Visual flowcharts and sequence diagrams
- **[IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md)** - Feature documentation and usage guide
- **[BUGFIXES.md](../BUGFIXES.md)** - Known issues and fixes

### Quick Links

**Getting Started:**
1. Read [QUICK_REFERENCE.md](../QUICK_REFERENCE.md) for immediate productivity
2. Review [FRONTEND_ARCHITECTURE.md](../FRONTEND_ARCHITECTURE.md) for system understanding
3. Study [FLOWCHARTS.md](../FLOWCHARTS.md) for visual flow understanding

**Common Tasks:**
- Adding a new provider → See [QUICK_REFERENCE.md](../QUICK_REFERENCE.md#adding-a-new-provider)
- Adding a new benchmark → See [QUICK_REFERENCE.md](../QUICK_REFERENCE.md#adding-a-new-benchmark)
- Debugging issues → See [QUICK_REFERENCE.md](../QUICK_REFERENCE.md#debugging-tips)

