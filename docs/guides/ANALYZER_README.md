# AI Agent Codebase Analyzer

A comprehensive static analysis tool designed specifically for AI/Agent codebases to identify upgrade opportunities in monitoring, orchestration, testing, and agent-specific patterns.

## 🚀 Features

### Core Analysis Categories
- **Monitoring & Metrics**: OpenTelemetry, Prometheus, structured logging
- **Tool Integration & Orchestration**: Workflow engines, parallel execution, circuit breakers
- **Testing Infrastructure**: Integration tests, load testing, mocking
- **AI/Agent-Specific Patterns**: LangGraph, FSM, tool registration, async patterns

### Performance & Scalability
- **Parallel File Analysis**: Uses ThreadPoolExecutor for fast scanning
- **Configurable Workers**: Adjust thread count for your system
- **Smart File Filtering**: Skips irrelevant directories and files

### Extensibility
- **YAML Configuration**: All patterns configurable via `analyzer_patterns.yaml`
- **Plugin-Ready**: Easy to add custom patterns and checks
- **Fallback Patterns**: Works even without config file

### Multiple Output Formats
- **Console**: Pretty-printed summary with emojis
- **JSON**: Machine-readable structured data
- **Markdown**: Perfect for PRs and documentation
- **HTML**: Beautiful web reports with styling

## 📦 Installation

```bash
# Clone or download the analyzer
# Ensure you have Python 3.7+ and required packages
pip install pyyaml
```

## 🎯 Usage

### Basic Analysis
```bash
# Analyze current directory
python ai_codebase_analyzer.py

# Analyze specific path
python ai_codebase_analyzer.py /path/to/your/codebase

# Use custom patterns file
python ai_codebase_analyzer.py --patterns my_patterns.yaml
```

### Output Formats
```bash
# JSON output
python ai_codebase_analyzer.py --json

# Markdown report
python ai_codebase_analyzer.py --markdown

# HTML report
python ai_codebase_analyzer.py --html

# Save to file
python ai_codebase_analyzer.py --markdown --output analysis_report.md
```

### Performance Tuning
```bash
# Use more worker threads (default: 4)
python ai_codebase_analyzer.py --workers 8

# Self-test mode (analyzes the analyzer itself)
python ai_codebase_analyzer.py --self-test
```

## 📋 Configuration

### Pattern Configuration (`analyzer_patterns.yaml`)

The analyzer uses a YAML configuration file to define patterns to look for:

```yaml
monitoring_patterns:
  missing_metrics:
    - pattern: "def\\s+(\\w+)\\s*\\([^)]*\\):"
      description: "Functions without performance metrics"
  
orchestration_patterns:
  sequential_execution:
    - pattern: "for\\s+tool\\s+in\\s+tools:"
      description: "Sequential tool execution"

testing_patterns:
  missing_tests:
    - pattern: "class\\s+(\\w+)(?!Test)"
      description: "Classes without corresponding tests"

agent_patterns:
  missing_tool_registration:
    - pattern: "class\\s+\\w+Tool"
      description: "Tools not registered with orchestrator"
```

### Pattern Structure
- **pattern**: Regex pattern to match in code
- **description**: Human-readable description of the issue

## 🔍 Analysis Categories

### 1. Monitoring & Metrics
**High Priority Issues:**
- Missing OpenTelemetry instrumentation
- No metrics collection in critical functions
- Tools without telemetry

**Recommendations:**
- Implement distributed tracing with OpenTelemetry
- Add Prometheus metrics for all critical operations
- Set up centralized logging with ELK or similar
- Create Grafana dashboards for real-time monitoring

### 2. Tool Integration & Orchestration
**High Priority Issues:**
- Missing workflow orchestration
- External API calls without circuit breakers
- Sequential tool execution

**Recommendations:**
- Implement workflow orchestration with Temporal or Airflow
- Add circuit breakers for external services
- Use async/parallel execution for tool calls
- Implement saga pattern for distributed transactions

### 3. Testing Infrastructure
**High Priority Issues:**
- Missing integration tests
- Tests using real APIs
- No load testing setup

**Recommendations:**
- Add integration tests for all major workflows
- Implement contract testing for APIs
- Set up load testing with Locust
- Add mutation testing for critical components

### 4. AI/Agent-Specific Patterns
**High Priority Issues:**
- LangGraph without proper state management
- FSM without error handling
- Tools not registered with orchestrator

**Recommendations:**
- Register all tools with the integration hub orchestrator
- Implement proper error handling in FSM workflows
- Use parallel execution for tool calls
- Add comprehensive agent testing with mock tools

## 📊 Sample Output

### Console Output
```
🔍 CODEBASE ANALYSIS REPORT
================================================================================

📊 SUMMARY
   Total upgrade points: 15
   Files analyzed: 42
   By category:
      - Monitoring: 6
      - Orchestration: 4
      - Testing: 3
      - Agent_specific: 2

📌 MONITORING UPGRADE POINTS
------------------------------------------------------------

🔴 HIGH Priority (3 items):
   📁 src/tools/weather.py:15
      Issue: Missing OpenTelemetry instrumentation
      Fix: Add OpenTelemetry spans for distributed tracing

💡 Recommendations:
   • Implement distributed tracing with OpenTelemetry
   • Add Prometheus metrics for all critical operations
   • Set up centralized logging with ELK or similar
   • Create Grafana dashboards for real-time monitoring
```

### JSON Output
```json
{
  "summary": {
    "total_upgrade_points": 15,
    "files_analyzed": 42,
    "by_category": {
      "monitoring": 6,
      "orchestration": 4,
      "testing": 3,
      "agent_specific": 2
    }
  },
  "monitoring": {
    "high_priority": [
      {
        "file": "src/tools/weather.py",
        "line": 15,
        "description": "Missing OpenTelemetry instrumentation",
        "suggestion": "Add OpenTelemetry spans for distributed tracing"
      }
    ]
  }
}
```

## 🛠️ Customization

### Adding Custom Patterns

1. Edit `analyzer_patterns.yaml`
2. Add new pattern categories or modify existing ones
3. Use regex patterns to match code patterns
4. Provide clear descriptions and suggestions

### Example Custom Pattern
```yaml
custom_patterns:
  security_issues:
    - pattern: "password.*=.*['\"]\\w+['\"]"
      description: "Hardcoded passwords in code"
```

### Extending the Analyzer

The analyzer is designed to be extensible. You can:

1. **Add new analysis methods** to the `CodebaseAnalyzer` class
2. **Create custom pattern categories** in the YAML config
3. **Add new output formats** by implementing report generators
4. **Integrate with CI/CD** using the JSON output

## 🔧 Integration Examples

### GitHub Actions
```yaml
name: Code Analysis
on: [push, pull_request]
jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Code Analysis
        run: |
          python ai_codebase_analyzer.py --json --output analysis.json
      - name: Comment Results
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const analysis = JSON.parse(fs.readFileSync('analysis.json'));
            const highPriority = analysis.monitoring.high_priority.length;
            if (highPriority > 0) {
              core.setFailed(`Found ${highPriority} high priority issues`);
            }
```

### Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit
python ai_codebase_analyzer.py --self-test
if [ $? -ne 0 ]; then
    echo "Self-test failed!"
    exit 1
fi
```

## 🧪 Testing

### Self-Test Mode
```bash
python ai_codebase_analyzer.py --self-test
```
This analyzes the analyzer itself and ensures it finds at least one upgrade point.

### Test Your Codebase
```bash
# Quick analysis
python ai_codebase_analyzer.py

# Detailed report
python ai_codebase_analyzer.py --markdown --output analysis.md

# Performance analysis
python ai_codebase_analyzer.py --workers 8 --json --output performance_analysis.json
```

## 📈 Performance Tips

1. **Adjust worker count** based on your CPU cores
2. **Use JSON output** for large codebases (faster processing)
3. **Skip irrelevant directories** by modifying `_should_skip_file()`
4. **Cache results** for repeated analysis

## 🤝 Contributing

1. **Add new patterns** to `analyzer_patterns.yaml`
2. **Extend analysis methods** in the `CodebaseAnalyzer` class
3. **Add new output formats** by implementing report generators
4. **Improve documentation** and examples

## 📄 License

This tool is designed for AI/Agent codebases and can be freely used and modified.

## 🆘 Troubleshooting

### Common Issues

1. **No patterns found**: Check if `analyzer_patterns.yaml` exists and is valid YAML
2. **Slow performance**: Reduce worker count or skip more directories
3. **False positives**: Adjust regex patterns in the config file
4. **Missing dependencies**: Install required packages with `pip install pyyaml`

### Debug Mode
```bash
# Enable debug logging
export PYTHONPATH=.
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from ai_codebase_analyzer import CodebaseAnalyzer
analyzer = CodebaseAnalyzer('.')
report = analyzer.analyze()
print(report['summary'])
"
```

---

**Happy analyzing! 🚀** 