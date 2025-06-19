# Enhanced FSM Implementation

## Overview

The Enhanced FSM (Finite State Machine) implementation provides advanced state management capabilities for your AI Agent system. It extends the existing FSMReActAgent with hierarchical states, probabilistic transitions, and dynamic state discovery.

## Key Features

### 1. Hierarchical FSM (HFSM)
- **Composite States**: Group related states into logical phases
- **Atomic States**: Simple states with single responsibilities
- **Nested Execution**: Execute child states within parent states
- **Better Organization**: Improved structure and maintainability

### 2. Probabilistic Transitions
- **Context-Aware**: Transitions adapt based on execution context
- **Learning Capabilities**: Probabilities adjust based on success rates
- **Context Modifiers**: Fine-tune transitions with conditions
- **History Tracking**: Learn from past transition patterns

### 3. Dynamic State Discovery
- **ML-Powered**: Uses pattern analysis to discover new states
- **Feature Extraction**: Analyzes tool usage, errors, and performance
- **Similarity Detection**: Identifies recurring patterns
- **Automatic Integration**: Seamlessly adds discovered states

### 4. Visualization & Debugging
- **State Diagrams**: Visual representation of FSM structure
- **Metrics Tracking**: Comprehensive performance monitoring
- **Transition Logs**: Detailed history of state changes
- **Debug Reports**: In-depth analysis of FSM behavior

## Quick Start

### 1. Basic Usage

```python
from src.migrated_enhanced_fsm_agent import MigratedEnhancedFSMAgent
from src.tools_enhanced import get_enhanced_tools

# Get your existing tools
tools = get_enhanced_tools()

# Create enhanced agent
agent = MigratedEnhancedFSMAgent(
    tools=tools,
    enable_hierarchical=True,
    enable_probabilistic=True,
    enable_discovery=True
)

# Run a query
result = agent.run({
    "input": "What is the weather in Tokyo?",
    "correlation_id": "test-123"
})

print(f"Answer: {result['output']}")
print(f"Execution time: {result['execution_time']:.2f}s")
```

### 2. Visualize FSM State

```python
# Get visual representation
visualization = agent.visualize_current_state()
print(visualization)

# Export metrics
metrics = agent.export_metrics()
print(f"State transitions: {len(metrics['transition_log'])}")
print(f"Discovered states: {len(metrics.get('discovered_states', []))}")
```

### 3. Monitor Performance

```python
# Get state metrics
state_metrics = agent.hfsm.get_state_metrics()
for state_name, metrics in state_metrics.items():
    success_rate = metrics.success_count / max(1, metrics.exit_count)
    print(f"{state_name}: {success_rate:.2%} success rate")
```

## Architecture

### State Hierarchy

```
EnhancedAgentFSM
├── PLANNING_PHASE
│   ├── PLANNING
│   ├── AWAITING_PLAN_RESPONSE
│   └── VALIDATING_PLAN
├── EXECUTION_PHASE
│   └── TOOL_EXECUTION
├── SYNTHESIS_PHASE
│   ├── SYNTHESIZING
│   └── VERIFYING
└── FAILURE_PHASE
    ├── TRANSIENT_API_FAILURE
    └── PERMANENT_API_FAILURE
```

### Transition Types

1. **Deterministic**: Traditional if-then transitions
2. **Probabilistic**: Probability-based with context modifiers
3. **Learning**: Adapts based on historical success rates

### Discovery Process

1. **Feature Extraction**: Analyze execution context
2. **Pattern Matching**: Compare with known patterns
3. **Significance Testing**: Determine if pattern is worth creating
4. **State Integration**: Add new state to FSM

## Configuration

### Enable/Disable Features

```python
agent = MigratedEnhancedFSMAgent(
    tools=tools,
    enable_hierarchical=True,    # Use hierarchical states
    enable_probabilistic=True,   # Use probabilistic transitions
    enable_discovery=True        # Enable state discovery
)
```

### Discovery Settings

```python
# Adjust discovery sensitivity
discovery_engine = StateDiscoveryEngine(
    similarity_threshold=0.85,    # Higher = more strict matching
    min_pattern_frequency=3       # Minimum occurrences to create state
)
```

### Transition Probabilities

```python
# Create probabilistic transition
transition = ProbabilisticTransition(
    "PLANNING", "EXECUTION", 
    base_probability=0.9
)

# Add context modifiers
transition.add_context_modifier("plan_confidence<0.5", 0.5)
transition.add_context_modifier("retry_count>2", 0.6)
```

## Best Practices

### State Design
- Keep atomic states focused on single responsibilities
- Use composite states for logical grouping
- Avoid deep nesting (max 3-4 levels)
- Document state purposes clearly

### Transition Design
- Start with deterministic transitions
- Add probabilistic features gradually
- Monitor transition performance
- Provide fallback paths

### Discovery Management
- Set appropriate similarity thresholds
- Validate discovered states before use
- Monitor discovery patterns
- Clean up unused patterns periodically

## Troubleshooting

### Common Issues

1. **FSM gets stuck in loops**
   - Check for circular transitions
   - Add stagnation detection
   - Implement iteration limits

2. **Low transition probabilities**
   - Review context modifiers
   - Check transition history
   - Adjust base probabilities

3. **Too many discovered states**
   - Increase similarity threshold
   - Increase minimum frequency
   - Review discovery criteria

### Debug Tools

```python
# Generate debug report
debug_info = agent.hfsm.export_metrics()
print(f"Current state: {debug_info['current_state']}")
print(f"Available transitions: {agent.hfsm.get_available_transitions()}")

# Visualize current state
print(agent.visualize_current_state())
```

## Integration with Existing Code

The Enhanced FSM is designed to be backward compatible with your existing FSMReActAgent:

```python
# Original usage still works
agent = FSMReActAgent(tools=tools)
result = agent.run(inputs)

# Enhanced usage with new features
enhanced_agent = MigratedEnhancedFSMAgent(tools=tools)
result = enhanced_agent.run(inputs)  # Enhanced features enabled
```

## Performance Considerations

- **Memory Usage**: State history and patterns consume memory
- **Computation**: Probabilistic calculations add overhead
- **Discovery**: Pattern analysis requires CPU cycles
- **Optimization**: Monitor and adjust based on your use case

## Future Enhancements

- **Async State Execution**: Parallel state processing
- **Distributed FSM**: Multi-agent coordination
- **Advanced Visualization**: Interactive state diagrams
- **Machine Learning**: Predictive state transitions
- **Custom Metrics**: Domain-specific performance tracking

## Contributing

To extend the Enhanced FSM:

1. **Add New State Types**: Extend BaseState class
2. **Custom Transitions**: Implement new transition logic
3. **Discovery Algorithms**: Add new pattern detection methods
4. **Visualization**: Create custom visualization tools

## Support

For issues and questions:

1. Check the implementation guide: `ENHANCED_FSM_IMPLEMENTATION_GUIDE.md`
2. Run the test suite: `python test_enhanced_fsm.py`
3. Review the example: `enhanced_fsm_example.py`
4. Check the source code: `src/enhanced_fsm.py`

## License

This implementation is part of the AI Agent project and follows the same licensing terms. 