# GAIA-Enhanced FSMReActAgent Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Component Details](#component-details)
4. [Integration Guide](#integration-guide)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)
8. [GAIA Benchmark Optimization](#gaia-benchmark-optimization)

## Overview

The GAIA-Enhanced FSMReActAgent is a production-ready AI agent that combines:
- **Finite State Machine (FSM)** for reliable execution flow
- **Advanced Reasoning** with multiple strategies
- **Persistent Memory** with vector search
- **Adaptive Tool Selection** using ML
- **Multi-Agent Orchestration** for complex tasks

### Key Benefits
- 🚀 **High Performance**: Optimized for GAIA benchmark tasks
- 🧠 **Smart Memory**: Learns from past interactions
- 🔧 **Adaptive Tools**: Improves tool selection over time
- 👥 **Multi-Agent**: Handles complex queries with specialized agents
- 🛡️ **Fault Tolerant**: Comprehensive error handling and recovery

## Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        FSMReActAgent                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐       │
│  │   Planning   │  │     Tool     │  │   Synthesizing  │       │
│  │    (ARE)     │─▶│  Execution   │─▶│    (Memory)     │       │
│  └─────────────┘  │   (ATS)      │  └─────────────────┘       │
│         │          └──────────────┘           │                 │
│         │                 │                    │                 │
│         ▼                 ▼                    ▼                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │          Multi-Agent Orchestrator (MAO)              │       │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     │       │
│  │  │Research││Analyze││Execute││Validate││Synthesize│ │       │
│  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘     │       │
│  └─────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘

ARE: Advanced Reasoning Engine
ATS: Adaptive Tool System  
MAO: Multi-Agent Orchestrator
```

### FSM State Diagram
```
┌─────────┐
│ PLANNING│
└────┬────┘
     │
     ▼
┌──────────────────┐     ┌───────────────┐
│AWAITING_PLAN_RESP├────▶│VALIDATING_PLAN│
└──────────────────┘     └───────┬───────┘
                                 │
                                 ▼
                         ┌──────────────┐
                         │TOOL_EXECUTION│◀─┐
                         └──────┬───────┘  │
                                │          │
                                ▼          │
                         ┌─────────────┐   │
                         │SYNTHESIZING │───┘
                         └──────┬──────┘
                                │
                                ▼
                         ┌─────────────┐
                         │  VERIFYING  │
                         └──────┬──────┘
                                │
                                ▼
                         ┌─────────────┐
                         │  FINISHED   │
                         └─────────────┘
```

## Component Details

### 1. Advanced Reasoning Engine (ARE)

**Purpose**: Multi-strategy reasoning with self-reflection and parallel processing

**Key Features**:
- **Multiple Reasoning Strategies**: Chain-of-thought, tree-of-thoughts, self-consistency
- **Parallel Processing**: Execute multiple reasoning paths simultaneously
- **Self-Reflection**: Evaluate and improve reasoning quality
- **Confidence Scoring**: Assess reasoning confidence

**Usage**:
```python
from src.gaia_components.advanced_reasoning_engine import AdvancedReasoningEngine

reasoning_engine = AdvancedReasoningEngine(
    llm_client=api_client,
    enable_parallel=True,
    confidence_threshold=0.7
)

reasoning_path = await reasoning_engine.reason("What is 2 + 2?")
```

### 2. Enhanced Memory System (EMS)

**Purpose**: Vector-based memory with semantic search and consolidation

**Key Features**:
- **Episodic Memory**: Store interaction memories
- **Semantic Memory**: Store factual knowledge
- **Working Memory**: Current context
- **Memory Consolidation**: Optimize memory usage
- **Vector Search**: Semantic similarity search

**Usage**:
```python
from src.gaia_components.enhanced_memory_system import EnhancedMemorySystem

memory_system = EnhancedMemorySystem(
    embedding_model=embedding_model,
    persist_path=Path("data/agent_memories")
)

# Store memory
memory_id = await memory_system.store_episodic(
    query="What is the capital of France?",
    response="Paris",
    tools_used=["web_search"],
    success=True
)

# Retrieve relevant memories
memories = await memory_system.retrieve_relevant("France capital", k=5)
```

### 3. Adaptive Tool System (ATS)

**Purpose**: ML-based tool selection that improves over time

**Key Features**:
- **Tool Performance Tracking**: Monitor success rates and execution times
- **ML-Based Selection**: Use machine learning for tool recommendation
- **Failure Recovery**: Automatic fallback strategies
- **Tool Composition**: Learn effective tool combinations

**Usage**:
```python
from src.gaia_components.adaptive_tool_system import AdaptiveToolSystem

adaptive_tools = AdaptiveToolSystem(
    tools=tools,
    learning_path=Path("data/tool_learning"),
    enable_ml=True,
    confidence_threshold=0.7
)

# Select best tool
tool_name, confidence = await adaptive_tools.select_tool(query, context)

# Execute tool
result = await adaptive_tools.execute_tool(tool_name, parameters, context)
```

### 4. Multi-Agent Orchestrator (MAO)

**Purpose**: Coordinate multiple specialized agents for complex problem-solving

**Key Features**:
- **Specialized Agents**: Researcher, Analyzer, Executor, Validator, Synthesizer
- **Task Distribution**: Automatically assign tasks to appropriate agents
- **Consensus Building**: Reach agreement among agents
- **Error Recovery**: Handle agent failures gracefully

**Usage**:
```python
from src.gaia_components.multi_agent_orchestrator import MultiAgentGAIASystem

multi_agent = MultiAgentGAIASystem(
    tools=tools,
    max_agents=10
)

# Solve complex query
result = await multi_agent.solve_query(
    "Analyze the impact of climate change on global agriculture"
)
```

## Integration Guide

### Step 1: Install Dependencies

```bash
pip install -r requirements_gaia.txt
```

### Step 2: Set Up Environment

Create a `.env` file:
```env
GROQ_API_KEY=your-groq-api-key-here
LOG_LEVEL=INFO
MEMORY_PERSIST_PATH=./data/agent_memories
TOOL_LEARNING_PATH=./data/tool_learning
```

### Step 3: Initialize Agent

```python
from src.agents.advanced_agent_fsm import FSMReActAgent
from src.tools.web_researcher import WebResearcher
from src.tools.python_interpreter import PythonInterpreter

# Initialize tools
tools = [
    WebResearcher(),
    PythonInterpreter()
]

# Initialize agent
agent = FSMReActAgent(
    tools=tools,
    model_name="llama-3.3-70b-versatile",
    quality_level=DataQualityLevel.THOROUGH,
    reasoning_type=ReasoningType.ADVANCED
)
```

### Step 4: Run Queries

```python
# Simple query
result = await agent.run("What is 2 + 2?")

# Complex query
result = await agent.run(
    "Calculate the population density of New York City"
)
```

## Performance Optimization

### 1. Memory Optimization

**Memory Consolidation**:
```python
# Run memory consolidation periodically
await memory_system.consolidate_memories()
```

**Memory Insights**:
```python
stats = memory_system.get_memory_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2f}")
```

### 2. Tool Performance

**Tool Insights**:
```python
insights = adaptive_tools.get_tool_insights()
print(f"Best tool: {insights['best_performing_tool']['name']}")
```

**Training**:
```python
# Train the recommendation model
await adaptive_tools.train_recommendation_model()
```

### 3. Multi-Agent Optimization

**Agent Insights**:
```python
insights = multi_agent.get_agent_insights()
print(f"System efficiency: {insights['system_efficiency']:.2f}")
```

**Load Balancing**:
```python
# Monitor agent performance
status = multi_agent.get_system_status()
print(f"Active tasks: {status['active_tasks']}")
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem**: `ModuleNotFoundError` for GAIA components
**Solution**: Ensure all GAIA component files are in `src/gaia_components/`

#### 2. API Key Issues
**Problem**: `GROQ_API_KEY not set`
**Solution**: Add your API key to the `.env` file

#### 3. Memory Issues
**Problem**: High memory usage
**Solution**: Run memory consolidation more frequently

#### 4. Tool Failures
**Problem**: Tools failing frequently
**Solution**: Check tool reliability metrics and adjust thresholds

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

Monitor system performance:
```python
# Get comprehensive stats
memory_stats = agent.memory_system.get_memory_stats()
tool_stats = agent.adaptive_tools.get_system_stats()
agent_stats = agent.multi_agent.get_system_status()
```

## Advanced Usage

### 1. Custom Reasoning Strategies

```python
from src.gaia_components.advanced_reasoning_engine import ReasoningStrategy

# Use specific reasoning strategy
reasoning_path = await reasoning_engine.reason(
    query,
    strategy=ReasoningStrategy.TREE_OF_THOUGHTS
)
```

### 2. Custom Memory Types

```python
from src.gaia_components.enhanced_memory_system import MemoryType

# Store semantic memory
await memory_system.store_semantic(
    fact="The Earth orbits the Sun",
    source="astronomy_textbook",
    confidence=0.95
)
```

### 3. Custom Tool Compositions

```python
# Learn from tool composition
await adaptive_tools.learn_from_composition(
    tool_sequence=["web_search", "python_interpreter"],
    success=True,
    total_time=2.5,
    context={"query_type": "calculation"}
)
```

### 4. Custom Agent Roles

```python
from src.gaia_components.multi_agent_orchestrator import AgentRole

# Create custom agent
class CustomAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentRole.SPECIALIST, {"custom_capability"})
    
    async def process_task(self, task: Task) -> Any:
        # Custom task processing logic
        pass
```

## GAIA Benchmark Optimization

### 1. Question Type Analysis

```python
def analyze_question_type(query: str) -> Dict[str, str]:
    """Analyze GAIA question type for optimization"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["how many", "count", "number"]):
        return {"type": "counting", "strategy": "extract_and_count"}
    elif any(word in query_lower for word in ["calculate", "compute", "math"]):
        return {"type": "calculation", "strategy": "step_by_step_calculation"}
    elif any(word in query_lower for word in ["when", "date", "year"]):
        return {"type": "temporal", "strategy": "timeline_extraction"}
    else:
        return {"type": "factual", "strategy": "research_and_verify"}
```

### 2. Confidence Thresholds

```python
# Adjust confidence thresholds based on question type
confidence_thresholds = {
    "counting": 0.95,
    "calculation": 0.98,
    "factual": 0.85,
    "temporal": 0.90
}
```

### 3. Tool Selection Optimization

```python
# Optimize tool selection for GAIA tasks
tool_preferences = {
    "counting": ["web_search", "python_interpreter"],
    "calculation": ["python_interpreter", "calculator"],
    "factual": ["web_search", "database"],
    "temporal": ["web_search", "timeline_tool"]
}
```

### 4. Verification Strategies

```python
# Enhanced verification for GAIA tasks
def verify_gaia_answer(answer: str, question_type: str) -> bool:
    if question_type == "counting":
        return answer.isdigit() and int(answer) > 0
    elif question_type == "calculation":
        try:
            float(answer)
            return True
        except ValueError:
            return False
    elif question_type == "temporal":
        # Check for date format
        return bool(re.match(r'\d{4}', answer))
    return True
```

## Best Practices

### 1. Memory Management
- Run consolidation every 100 interactions
- Monitor memory usage and clean up old memories
- Use appropriate memory types for different data

### 2. Tool Selection
- Train the ML model regularly with new data
- Monitor tool performance and adjust thresholds
- Use tool compositions for complex tasks

### 3. Error Handling
- Implement graceful degradation
- Use circuit breakers for external services
- Log errors with correlation IDs

### 4. Performance Monitoring
- Track response times and success rates
- Monitor memory and tool usage
- Set up alerts for system issues

## Conclusion

The GAIA-Enhanced FSMReActAgent provides a comprehensive solution for building production-ready AI agents. By combining advanced reasoning, persistent memory, adaptive tool selection, and multi-agent orchestration, it delivers high performance and reliability for complex AI tasks.

For more information, see the individual component documentation and the example files in the `examples/` directory. 