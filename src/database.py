import logging
import os
from typing import List

from dotenv import load_dotenv
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.supabase import SupabaseVectorStore
from supabase import Client, create_client

# Load environment variables from.env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Supabase Client and Table Schemas ---

# These are the SQL commands to set up your Supabase database.
# Execute these in the Supabase SQL Editor for your project.

KNOWLEDGE_BASE_TABLE_SCHEMA = """
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

-- Create a function for similarity search
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding VECTOR(1536),
  match_count INT,
  filter JSONB DEFAULT '{}'
) RETURNS TABLE (
  id UUID,
  node_id TEXT,
  text TEXT,
  metadata_ JSONB,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    id,
    node_id,
    text,
    metadata_,
    1 - (knowledge_base.embedding <=> query_embedding) AS similarity
  FROM knowledge_base
  WHERE metadata_ @> filter
  ORDER BY knowledge_base.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Create an HNSW index for efficient similarity search
CREATE INDEX ON knowledge_base USING hnsw (embedding vector_cosine_ops);
"""

AGENT_LOG_TABLE_SCHEMA = """
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
"""


def get_supabase_client() -> Client:
    """
    Creates and returns a Supabase client instance.
    Requires SUPABASE_URL and SUPABASE_KEY environment variables.
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("Supabase URL and Key must be set in environment variables.")
    return create_client(supabase_url, supabase_key)


class SupabaseLogHandler(logging.Handler):
    """
    A custom logging handler that sends log records to a Supabase table.
    """
    def __init__(self, supabase_client: Client, table_name: str = "agent_trajectory_logs"):
        super().__init__()
        self.client = supabase_client
        self.table_name = table_name

    def emit(self, record: logging.LogRecord):
        """
        Emits a log record to the Supabase table.
        Expects the log record's message to be a dictionary with 'run_id', 'step_type', and 'payload'.
        """
        try:
            log_data = record.msg
            if isinstance(log_data, dict) and all(k in log_data for k in ['run_id', 'step_type', 'payload']):
                self.client.table(self.table_name).insert({
                    "run_id": str(log_data['run_id']),
                    "step_type": log_data['step_type'],
                    "payload": log_data['payload']
                }).execute()
            else:
                # Fallback for non-structured messages if needed, though not the primary use case
                pass
        except Exception as e:
            # Use print to avoid recursive logging loop
            print(f"Error in SupabaseLogHandler: {e}")


# --- LlamaIndex Knowledge Base Setup ---

def get_vector_store() -> SupabaseVectorStore:
    """
    Initializes and returns a SupabaseVectorStore for LlamaIndex.
    """
    db_password = os.getenv("SUPABASE_DB_PASSWORD")
    supabase_url = os.getenv("SUPABASE_URL")
    if not supabase_url or not db_password:
        raise ValueError("SUPABASE_URL and SUPABASE_DB_PASSWORD must be set in environment variables.")
    
    db_url_parts = supabase_url.replace("https://", "").split(".")
    if len(db_url_parts) < 3:
        raise ValueError("Unexpected SUPABASE_URL format. Expected 'https://<project_ref>.supabase.co'")

    project_ref = db_url_parts[0]
    rest_domain = '.'.join(db_url_parts[1:])
    db_host = f"db.{project_ref}.{rest_domain}"
    db_user = "postgres"
    db_name = "postgres"

    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}"

    return SupabaseVectorStore(
        postgres_connection_string=connection_string,
        collection_name="knowledge_base",
        dimension=1536  # Matches OpenAI's text-embedding-3-small
    )

def setup_knowledge_base(documents: List):
    """
    Sets up the knowledge base by ingesting documents into the Supabase vector store.
    """
    logger.info("Setting up LlamaIndex and embedding model...")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    logger.info(f"Creating VectorStoreIndex with {len(documents)} documents...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    logger.info("Knowledge base setup complete. Documents ingested into Supabase.") 