# 🎯 GAIA Benchmark Agent - Advanced AI Agent

An enhanced AI agent for the GAIA benchmark that combines sophisticated reasoning capabilities with the official GAIA evaluation framework.

## 🚀 Key Features

### Advanced Reasoning Capabilities
- **Strategic Planning**: Analyzes questions and creates multi-step execution plans
- **Cross-Validation**: Verifies answers through multiple independent sources
- **Adaptive Intelligence**: Adjusts strategy based on question complexity and confidence levels
- **Reflection & Learning**: Self-assessment and error recovery mechanisms

### Comprehensive Tool Suite
- **Web Research**: Wikipedia, Tavily search, general web scraping
- **Document Analysis**: PDF, Word, Excel, text files with advanced parsing
- **Multimedia Processing**: Image analysis, audio transcription, video analysis
- **Computation**: Python interpreter for calculations and data processing
- **Semantic Search**: GPU-accelerated vector search through knowledge bases

### Performance Optimization
- **Multi-Model Architecture**: Specialized models for different task types
- **Parallel Processing**: Multiple worker threads with intelligent rate limiting
- **Response Caching**: Intelligent caching with TTL management
- **GPU Acceleration**: CUDA-enabled processing when available

## 📁 File Structure

```
AI-Agent/
├── agent.py                 # GAIA-compatible agent wrapper
├── gaia_app.py             # Enhanced Gradio interface with GAIA evaluation
├── app.py                  # Original advanced agent interface
├── requirements-gaia.txt   # GAIA-specific dependencies
├── src/
│   ├── advanced_agent.py   # Core advanced agent implementation
│   ├── tools.py           # Comprehensive tool suite
│   └── database.py        # Supabase integration for logging
└── GAIA_README.md         # This file
```

## 🛠️ Setup Instructions

### 1. Environment Setup

```bash
# Install GAIA-specific requirements
pip install -r requirements-gaia.txt

# Or install full requirements for all features
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file with the following variables:

```env
# Required for AI models
GROQ_API_KEY=your_groq_api_key_here

# Optional for enhanced features
TAVILY_API_KEY=your_tavily_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional for logging and knowledge base
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
SUPABASE_DB_PASSWORD=your_db_password_here

# For HuggingFace Space deployment
SPACE_ID=your_space_id_here
SPACE_HOST=your_space_host_here
```

### 3. Quick Test

```python
# Test the GAIA agent locally
from agent import build_graph

# Initialize the agent
agent_graph = build_graph()

# Test with a sample question
result = agent_graph.invoke({
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
})

print(result['messages'][-1].content)
```

## 🎯 GAIA Benchmark Usage

### Running the Evaluation

1. **Deploy to HuggingFace Space** or run locally
2. **Set environment variables** (especially `GROQ_API_KEY`)
3. **Launch the application**:
   ```bash
   python gaia_app.py
   ```
4. **Log in** with your HuggingFace account
5. **Click "Run GAIA Evaluation"** to start the benchmark

### Expected Performance

The agent is designed to handle various GAIA question types:

- **Factual Questions**: Uses cross-validated web research
- **Numerical Calculations**: Employs Python interpreter with verification
- **Document Analysis**: Processes uploaded files with specialized tools
- **Multimedia Analysis**: Handles images, audio, and video content
- **Complex Reasoning**: Applies strategic planning and reflection

## 🔧 Advanced Configuration

### Model Selection

The agent automatically selects optimal models for different tasks:

```python
# Configure model preferences in advanced_agent.py
REASONING_MODELS = {
    "primary": "llama-3.3-70b-versatile",     # Complex reasoning
    "fast": "llama-3.1-8b-instant",           # Quick responses
    "deep": "deepseek-r1-distill-llama-70b"   # Deep analysis
}
```

### Tool Customization

Add or modify tools in `src/tools.py`:

```python
@tool
def custom_analysis_tool(query: str) -> str:
    """Your custom tool implementation."""
    # Implementation here
    return result
```

### Verification Levels

Adjust verification intensity based on requirements:

- **basic**: Single source verification
- **thorough**: Multiple source cross-validation (default for GAIA)
- **exhaustive**: Comprehensive verification with alternative approaches

## 📊 Monitoring & Analytics

The agent provides comprehensive monitoring:

- **Real-time Performance**: Processing times, success rates, tool usage
- **Tool Analytics**: Individual tool performance and reliability
- **Error Tracking**: Detailed error logs and recovery statistics
- **Cache Efficiency**: Response caching effectiveness

## 🚀 Deployment Options

### Local Development
```bash
python gaia_app.py
```

### HuggingFace Space
1. Create a new Space on HuggingFace
2. Upload all files to the Space
3. Set environment variables in Space settings
4. The Space will automatically deploy

### Docker Deployment
```dockerfile
# Create a Dockerfile for containerized deployment
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements-gaia.txt

EXPOSE 7860
CMD ["python", "gaia_app.py"]
```

## 🎯 GAIA-Specific Features

### Answer Extraction
The agent includes sophisticated answer extraction for GAIA submission:
- Removes reasoning prefixes and explanations
- Extracts concise, factual answers
- Handles various response formats

### Question Analysis
Automatic question type detection and strategy selection:
- Counting/numerical questions → Research + Python verification
- Chess analysis → Specialized chess reasoning
- Country/code lookup → Official source verification
- Music/discography → Cross-referenced database search

### Performance Optimization for GAIA
- **Quality Model Selection**: Uses highest quality models for accuracy
- **Thorough Verification**: Default to comprehensive verification
- **Clean Answer Extraction**: Optimized for GAIA submission format
- **Error Recovery**: Robust handling of complex questions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Test with GAIA questions
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues or questions:
1. Check the logs for detailed error information
2. Verify environment variables are set correctly
3. Ensure all dependencies are installed
4. Test individual tools for functionality

---

**Happy benchmarking with your advanced GAIA agent!** 🚀 