# 🚀 Advanced AI Agent with GAIA Benchmark Evaluation

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/your-username/ai-agent)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A production-ready AI agent that combines powerful strategic reasoning with GAIA benchmark evaluation capabilities. Built with LangChain, LangGraph, and Gradio, featuring advanced error handling, parallel processing, and multi-modal tool orchestration.

## 🎯 Key Features

### Advanced Reasoning Capabilities
- **Strategic Planning**: Analyzes questions and creates multi-step execution plans
- **Cross-Validation**: Verifies information through multiple independent sources
- **Adaptive Intelligence**: Dynamically adjusts strategy based on complexity
- **Error Recovery**: Intelligent retry strategies with alternative approaches

### Tool Orchestration
- 🌐 **Web Research**: Wikipedia, Tavily search, web scraping
- 📄 **Document Analysis**: PDF, Word, Excel, text files
- 🖼️ **Multimedia Processing**: Image analysis, audio transcription, video analysis
- 🐍 **Code Execution**: Python interpreter for calculations and data processing
- 🔍 **Semantic Search**: GPU-accelerated vector search through knowledge bases

### Production Features
- **Rate Limiting**: Respects API limits with intelligent throttling
- **Parallel Processing**: Multi-threaded execution with API-safe concurrency
- **Response Caching**: Intelligent caching with TTL management
- **Session Management**: User session tracking and analytics
- **Comprehensive Logging**: Structured logging with correlation IDs

### GAIA Benchmark Integration
- Automated question processing and answer submission
- Performance analytics and success tracking
- Enhanced reasoning optimized for GAIA challenges
- Real-time progress monitoring

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for enhanced performance)
- API Keys:
  - **GROQ_API_KEY** (required)
  - **TAVILY_API_KEY** (recommended for web search)
  - **OPENAI_API_KEY** (optional for embeddings)
  - **SUPABASE_URL** and **SUPABASE_KEY** (optional for logging)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-agent.git
cd ai-agent
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
OPENAI_API_KEY=your_openai_api_key  # Optional
SUPABASE_URL=your_supabase_url      # Optional
SUPABASE_KEY=your_supabase_key      # Optional
```

## 🚀 Usage

### Running Locally

```bash
python app.py
```

The application will be available at `http://localhost:7860`

### Deploying to Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Set the SDK to "Gradio"
3. Upload all files from this repository
4. Add your API keys as Space secrets:
   - `GROQ_API_KEY`
   - `TAVILY_API_KEY`
   - etc.

### Using the Interface

#### Interactive Chat
1. Navigate to the "Advanced Agent Chat" tab
2. Type your question or upload a file
3. Watch the agent's reasoning process in real-time
4. View performance analytics

#### GAIA Evaluation
1. Navigate to the "GAIA Evaluation" tab
2. Log in with your Hugging Face account
3. Click "Run GAIA Evaluation"
4. Monitor progress and view results

## 🏗️ Architecture

### Modular Design

```
├── app.py                 # Main application entry point
├── config.py              # Centralized configuration
├── ui.py                  # Gradio UI components
├── session.py             # Session and cache management
├── gaia_logic.py          # GAIA evaluation logic
├── src/
│   ├── advanced_agent_fsm.py  # FSM-based agent implementation
│   ├── tools.py               # Tool implementations
│   ├── tools_enhanced.py      # Enhanced GAIA tools
│   ├── database.py            # Database connections
│   └── knowledge_ingestion.py # Knowledge base management
└── tests/                     # Unit and integration tests
```

### Key Components

- **FSM Agent**: Deterministic finite state machine for reliable execution
- **Parallel Pool**: Thread pool with API rate limiting
- **Response Cache**: LRU cache with TTL for fast responses
- **Session Manager**: Tracks user sessions and analytics

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Run specific tests:

```bash
pytest tests/test_config.py -v
pytest tests/test_session.py -v
pytest tests/test_tools.py -v
```

## ⚙️ Configuration

The application uses a centralized configuration system. Key settings:

- **Performance**:
  - `MAX_PARALLEL_WORKERS`: 8 (reduced for API limits)
  - `REQUEST_SPACING`: 0.5 seconds
  - `CACHE_TTL_SECONDS`: 3600 (1 hour)

- **Models**:
  - Reasoning: `llama-3.3-70b-versatile`
  - Function Calling: `llama-3.3-70b-versatile`
  - Text Generation: `llama-3.3-70b-versatile`
  - Vision: `meta-llama/llama-4-scout-17b-16e-instruct`

## 🐛 Troubleshooting

### Common Issues

1. **GroqError on Startup**
   - Ensure `GROQ_API_KEY` is set correctly
   - Check that `.env` file is not overriding environment variables

2. **Rate Limiting Errors**
   - The system automatically handles rate limits
   - Reduce `MAX_PARALLEL_WORKERS` if needed

3. **FSM State Errors**
   - Ensure all state transitions are defined
   - Check logs for detailed error messages

### Debug Mode

Enable debug logging:

```python
# In config.py
self.logging.LOG_LEVEL = "DEBUG"
```

## 📊 Performance Optimization

- **Parallel Processing**: Up to 8 concurrent workers
- **Response Caching**: Reduces API calls by 30-50%
- **GPU Acceleration**: Automatic when CUDA is available
- **Adaptive Timeouts**: Dynamic based on task complexity

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LangChain and LangGraph teams for the excellent frameworks
- Groq for providing fast inference
- Hugging Face for the deployment platform
- GAIA benchmark creators for the evaluation framework

## 📧 Contact

For questions or support, please open an issue on GitHub or contact [your-email@example.com]

---

**Note**: This is an active research project. Features and APIs may change. Always refer to the latest documentation.