# AI Agent SaaS Platform

A modern, beautiful web interface for your AI Agent system with real-time chat, agent management, and monitoring capabilities.

## Features

- **🤖 Multi-Agent Chat Interface**: Chat with multiple specialized AI agents in real-time
- **📊 Dashboard & Analytics**: Monitor agent performance, system health, and usage metrics
- **🎨 Modern UI/UX**: Beautiful dark/light theme with smooth animations
- **🔄 Real-time Updates**: WebSocket support for live agent responses and status updates
- **📱 Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- **🛠️ Agent Management**: Create, configure, and manage multiple AI agents
- **📈 Performance Monitoring**: Track CPU, memory, and task metrics

## Tech Stack

### Frontend
- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Framer Motion** for animations
- **Recharts** for data visualization
- **Socket.io** for real-time communication
- **Zustand** for state management

### Backend
- **FastAPI** for REST API and WebSockets
- **Multi-Agent Architecture** with specialized agents
- **Advanced Tool System** for agent capabilities
- **Real-time Monitoring** and health checks

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### One-Command Start

```bash
./start_saas.sh
```

This will:
1. Set up Python virtual environment
2. Install all backend dependencies
3. Start the FastAPI server on http://localhost:8000
4. Install frontend dependencies
5. Start the React app on http://localhost:3000

### Manual Setup

#### Backend Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
python src/api_server.py
```

#### Frontend Setup
```bash
# Navigate to frontend directory
cd saas-ui

# Install dependencies
npm install

# Start development server
npm run dev
```

## Usage

1. **Access the Platform**: Open http://localhost:3000 in your browser

2. **Select an Agent**: Choose from available AI agents in the chat interface

3. **Start Chatting**: Send messages and receive intelligent responses

4. **Monitor Performance**: Check the dashboard for real-time metrics

5. **Manage Agents**: Create and configure agents from the Agents page

## Project Structure

```
AI-Agent/
├── saas-ui/                 # React frontend
│   ├── src/
│   │   ├── components/      # Reusable UI components
│   │   ├── pages/           # Page components
│   │   ├── services/        # API and WebSocket services
│   │   ├── store/           # State management
│   │   └── types/           # TypeScript types
│   ├── package.json
│   └── vite.config.ts
├── src/
│   ├── api_server.py        # FastAPI backend
│   ├── agents/              # Agent implementations
│   └── core/                # Core functionality
├── start_saas.sh            # Startup script
└── requirements.txt         # Python dependencies
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# API Keys
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Database
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Server Config
API_HOST=0.0.0.0
API_PORT=8000
```

### Customization

#### Theme Colors
Edit `saas-ui/src/index.css` to customize the color scheme.

#### Agent Configuration
Modify agent settings in `src/agents/` directory.

## Development

### Frontend Development
```bash
cd saas-ui
npm run dev     # Start dev server
npm run build   # Build for production
npm run lint    # Run linter
```

### Backend Development
```bash
# Run with auto-reload
uvicorn src.api_server:app --reload

# Run tests
pytest tests/
```

## Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d
```

### Cloud Deployment
- Frontend: Deploy to Vercel, Netlify, or AWS S3
- Backend: Deploy to AWS ECS, Google Cloud Run, or Heroku

## API Endpoints

- `GET /api/v1/agents` - List all agents
- `POST /api/v1/agents` - Create new agent
- `GET /api/v1/tasks` - List tasks
- `POST /api/v1/messages` - Send message
- `WS /api/v1/ws/{client_id}` - WebSocket connection
- `GET /api/v1/health` - System health metrics

## Troubleshooting

### Backend Issues
- Ensure Python 3.8+ is installed
- Check all required API keys are set
- Verify port 8000 is available

### Frontend Issues
- Ensure Node.js 16+ is installed
- Clear npm cache: `npm cache clean --force`
- Delete node_modules and reinstall

### Connection Issues
- Check if backend is running on http://localhost:8000
- Verify WebSocket connection in browser console
- Check for CORS errors

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is for personal use as specified by the user.

## Support

For issues or questions, check the logs in:
- Backend: Console output from FastAPI
- Frontend: Browser developer console

---

Built with ❤️ for an amazing AI Agent experience!