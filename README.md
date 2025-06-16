# Production-Ready ReAct Agent

## Introduction

This repository contains the complete source code for a sophisticated, multi-tool AI agent built on a ReAct (Reasoning and Acting) framework. The agent is designed for high performance and modularity, making it an excellent blueprint for production-level agentic systems.

It leverages a powerful technology stack to achieve its capabilities:
- **LLM Inference**: Groq for ultra-low-latency responses.
- **Stateful Logic**: LangGraph for managing the agent's reasoning loop.
- **Knowledge Base**: LlamaIndex connected to a [Supabase](https://supabase.com/) PostgreSQL database with the pgvector extension.
- **Real-Time Search**: [Tavily](https://tavily.com/) for accessing up-to-date information.
- **UI**: Gradio for an interactive chat interface.
- **Deployment**: Designed for easy deployment on [Hugging Face Spaces](https://huggingface.co/spaces).

## Prerequisites

Before you begin, you will need to create accounts with the following services and obtain API keys/credentials:
- **Hugging Face**: To deploy the application on a Space.
- **Supabase**: For the PostgreSQL database and vector store.
- **Groq**: For LLM inference.
- **Tavily AI**: For the real-time search tool.
- **OpenAI**: For generating text embeddings to store in the knowledge base.

## Environment Variable Setup

This project requires several environment variables to connect to the various services. You must set these as "Repository secrets" in your Hugging Face Space settings for deployment. For local development, create a `.env` file in the root of the project directory by copying `.env.example` and filling in the values.

| Variable Name | Description | Example (for .env.example) |
|---------------|-------------|---------------------------|
| GROQ_API_KEY | API key for Groq inference. | gsk_... |
| TAVILY_API_KEY | API key for Tavily Search. | tvly-... |
| OPENAI_API_KEY | API key for OpenAI (used for text embeddings). | sk-... |
| SUPABASE_URL | URL for your Supabase project. | https://[your-project-ref].supabase.co |
| SUPABASE_KEY | The service_role key for your Supabase project. | ey... |
| SUPABASE_DB_PASSWORD | The database password for your Supabase project. | [your-db-password] |

## Supabase Project Setup (Step-by-Step)

1. **Create a Supabase Project**:
   - Go to [supabase.com](https://supabase.com) and create a new project.
   - When creating the project, make sure to save the Database Password you set. You will need it for the `SUPABASE_DB_PASSWORD` environment variable.

2. **Get Project Credentials**:
   - Navigate to your project's dashboard.
   - Go to Project Settings (the gear icon) > API.
   - Find and copy your Project URL. This is your `SUPABASE_URL`.
   - Under Project API Keys, find and copy the service_role secret key. This is your `SUPABASE_KEY`.

3. **Set up Database Tables**:
   - In your project's dashboard, navigate to the SQL Editor (the icon with SQL).
   - Click "New query".
   - Copy and paste the following SQL commands into the editor and click "RUN". This will enable the necessary pgvector extension and create the required tables for the knowledge base and agent trajectory logging.

```sql
-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the table to store document chunks and their embeddings
CREATE TABLE knowledge_base (
    id UUID PRIMARY KEY,
    node_id TEXT UNIQUE NOT NULL,
    embedding VECTOR(1536) NOT NULL, -- OpenAI 'text-embedding-3-small' produces 1536-dim vectors
    text TEXT,
    metadata_ JSONB
);

-- Create an HNSW index for efficient similarity search
CREATE INDEX ON knowledge_base USING hnsw (embedding vector_cosine_ops);

-- Create the table for logging agent trajectories
CREATE TABLE agent_trajectory_logs (
    log_id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    step_type TEXT NOT NULL, -- e.g., 'REASON', 'ACTION', 'OBSERVATION', 'FINAL_ANSWER'
    payload JSONB
);

-- Create an index on run_id for efficient querying of trajectories
CREATE INDEX idx_agent_trajectory_logs_run_id ON agent_trajectory_logs(run_id);
```

## Local Setup and Data Ingestion

1. **Clone the Repository**:
```bash
git clone <repository-url>
cd multi-tool-agent
```

2. **Set up Python Environment**:
   It is highly recommended to use a virtual environment.
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure Environment**:
   - Create a file named `.env` in the project root.
   - Copy the contents of `.env.example` into `.env`.
   - Fill in the values with your actual credentials obtained in the previous steps.

5. **Add Knowledge Documents**:
   - Place any text files (.txt, .md, etc.) you want the agent to have in its internal knowledge base into the `/data` directory. A sample file `knowledge_base_content.txt` is included.

6. **Run Data Ingestion Script**:
   This script will read the documents from the `/data` directory, generate embeddings, and store them in your Supabase database.
```bash
python scripts/ingest_data.py
```
   You should see progress bars and a success message upon completion.

7. **Run the Application Locally**:
```bash
python app.py
```
   The application will be available at a local URL (e.g., http://127.0.0.1:7860).

## Deployment to Hugging Face Spaces

1. **Create a New Space**:
   - On [huggingface.co](https://huggingface.co), click your profile icon and select "New Space".
   - Give your Space a name.
   - Select "Gradio" as the Space SDK.
   - Choose a hardware configuration (the free CPU tier is sufficient to start).
   - Click "Create Space".

2. **Upload Project Files**:
   - You can upload your files via the web interface or by cloning the Space repository locally using Git.
   - Upload all files and directories from this project (`src/`, `data/`, `scripts/`, `app.py`, `requirements.txt`, etc.).

3. **Add Repository Secrets**:
   - This is the most critical step for deployment.
   - In your Space, navigate to the Settings tab.
   - In the left sidebar, click on "Secrets and variables".
   - Click the "New secret" button.
   - Add each of the environment variables listed in the table above, one by one. The Name of the secret must match the variable name exactly (e.g., `GROQ_API_KEY`). Paste your key into the Value field.
   - After adding all secrets, the Space will automatically restart and build the application.

4. **Access Your Agent**:
   - Once the build process is complete, your Gradio application will be live and accessible at your Space's public URL.

## Usage Examples

You can test the agent's different capabilities with the following types of questions:
- **Real-Time Search (Tavily Tool)**: "What are the latest developments in generative AI this week?"
- **Knowledge Base Retrieval (LlamaIndex Tool)**: "What were the key results of Project Orion in Q2?"
- **Code Execution (Python REPL Tool)**: "What is the square root of 1521?"
- **Complex Query (Multi-Tool)**: "Based on the Project Orion report, what were the main challenges? Also, search for common solutions to database connection pool issues."
