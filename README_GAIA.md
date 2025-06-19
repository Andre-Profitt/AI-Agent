# GAIA-Enhanced FSMReActAgent

A production-ready AI agent that combines:
- **Finite State Machine (FSM)** for reliable execution flow
- **Advanced Reasoning** with multiple strategies
- **Persistent Memory** with vector search
- **Adaptive Tool Selection** using ML
- **Multi-Agent Orchestration** for complex tasks

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements_gaia.txt
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

3. **Run the example:**
   ```bash
   python examples/gaia_usage_example.py
   ```

## Features

- 🚀 **High Performance**: Optimized for GAIA benchmark tasks
- 🧠 **Smart Memory**: Learns from past interactions
- 🔧 **Adaptive Tools**: Improves tool selection over time
- 👥 **Multi-Agent**: Handles complex queries with specialized agents
- 🛡️ **Fault Tolerant**: Comprehensive error handling and recovery

## Architecture

The agent uses a Finite State Machine with the following states:
- **PLANNING**: Create execution plan
- **TOOL_EXECUTION**: Execute tools
- **SYNTHESIZING**: Combine results
- **VERIFYING**: Validate answer
- **FINISHED**: Complete

## Components

- **Advanced Reasoning Engine**: Multi-strategy reasoning
- **Enhanced Memory System**: Vector-based memory with consolidation
- **Adaptive Tool System**: ML-based tool selection
- **Multi-Agent Orchestrator**: Coordinate specialized agents

## Usage

```python
from examples.gaia_usage_example import GAIAEnhancedAgent

# Initialize agent
agent = GAIAEnhancedAgent()

# Solve queries
result = await agent.solve("What is 2 + 2?")
print(result["answer"])
```

## Testing

```bash
pytest tests/
```

## Documentation

See the `docs/` directory for detailed documentation.
