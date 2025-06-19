# Supabase SQL Setup Guide for AI Agent

This guide provides all the SQL commands needed to set up your Supabase database for the AI Agent with resilience patterns implementation.

## Prerequisites

1. A Supabase project (create one at https://supabase.com)
2. Access to the SQL Editor in your Supabase dashboard
3. Your Supabase URL and API keys

## Required Environment Variables

Add these to your `.env` file:

```bash
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-public-key
SUPABASE_DB_PASSWORD=your-database-password
```

## SQL Tables Setup

Execute these SQL commands in your Supabase SQL Editor in the following order:

### 1. Enable Required Extensions

```sql
-- Enable pgvector for semantic search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

### 2. Core Knowledge Base Table

```sql
-- Create the table to store document chunks and their embeddings
CREATE TABLE knowledge_base (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id TEXT UNIQUE NOT NULL,
    embedding VECTOR(1536) NOT NULL, -- OpenAI 'text-embedding-3-small' produces 1536-dim vectors
    text TEXT,
    metadata_ JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create an HNSW index for efficient similarity search
CREATE INDEX ON knowledge_base USING hnsw (embedding vector_cosine_ops);

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
```

### 3. Agent Trajectory Logging Table

```sql
-- Create the table for logging agent trajectories
CREATE TABLE agent_trajectory_logs (
    log_id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    correlation_id UUID,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    step_type TEXT NOT NULL, -- e.g., 'REASON', 'ACTION', 'OBSERVATION', 'FINAL_ANSWER'
    fsm_state TEXT, -- Current FSM state
    payload JSONB,
    error_category TEXT,
    recovery_strategy TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for efficient querying
CREATE INDEX idx_agent_trajectory_logs_run_id ON agent_trajectory_logs(run_id);
CREATE INDEX idx_agent_trajectory_logs_correlation_id ON agent_trajectory_logs(correlation_id);
CREATE INDEX idx_agent_trajectory_logs_timestamp ON agent_trajectory_logs(timestamp);
CREATE INDEX idx_agent_trajectory_logs_step_type ON agent_trajectory_logs(step_type);
```

### 4. Tool Reliability Metrics Table

```sql
-- Track tool performance and reliability
CREATE TABLE tool_reliability_metrics (
    tool_name TEXT PRIMARY KEY,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    total_calls INTEGER DEFAULT 0,
    average_latency_ms REAL DEFAULT 0.0,
    last_used_at TIMESTAMP WITH TIME ZONE,
    last_error TEXT,
    error_patterns JSONB DEFAULT '[]'::jsonb,
    fallback_tools JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for last_used_at for cleanup queries
CREATE INDEX idx_tool_reliability_last_used ON tool_reliability_metrics(last_used_at);
```

### 5. Clarification Patterns Table

```sql
-- Store patterns of clarification requests for learning
CREATE TABLE clarification_patterns (
    id TEXT PRIMARY KEY,
    original_query TEXT NOT NULL,
    query_embedding VECTOR(1536),  -- For similarity search
    clarification_question TEXT NOT NULL,
    user_response TEXT NOT NULL,
    query_category TEXT NOT NULL,
    frequency INTEGER DEFAULT 1,
    effectiveness_score REAL DEFAULT 0.5,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for efficient pattern matching
CREATE INDEX idx_clarification_patterns_category ON clarification_patterns(query_category);
CREATE INDEX idx_clarification_patterns_embedding ON clarification_patterns USING hnsw (query_embedding vector_cosine_ops);
```

### 6. Plan Corrections Table

```sql
-- Record user corrections to agent plans for improvement
CREATE TABLE plan_corrections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT NOT NULL,
    original_plan JSONB NOT NULL,
    corrected_plan JSONB NOT NULL,
    correction_type TEXT NOT NULL, -- 'steps_added', 'steps_removed', 'parameters_changed', etc.
    user_feedback TEXT,
    applied_to_future_plans BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for query similarity matching
CREATE INDEX idx_plan_corrections_query ON plan_corrections USING gin(to_tsvector('english', query));
```

### 7. Knowledge Lifecycle Table

```sql
-- Track document freshness and validation needs
CREATE TABLE knowledge_lifecycle (
    document_id TEXT PRIMARY KEY,
    source_url TEXT,
    document_type TEXT NOT NULL, -- 'news', 'documentation', 'research', etc.
    content_hash TEXT NOT NULL,
    ingested_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_validated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    validation_status TEXT NOT NULL, -- 'valid', 'stale', 'expired', 'source_unavailable'
    update_frequency_days INTEGER NOT NULL,
    importance_score REAL DEFAULT 0.5,
    validation_failures INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for lifecycle management
CREATE INDEX idx_knowledge_lifecycle_expires ON knowledge_lifecycle(expires_at);
CREATE INDEX idx_knowledge_lifecycle_validation ON knowledge_lifecycle(last_validated_at);
CREATE INDEX idx_knowledge_lifecycle_status ON knowledge_lifecycle(validation_status);
```

### 8. Resilience Tracking Tables

```sql
-- Track GraphRecursionError occurrences and resolutions
CREATE TABLE recursion_error_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    correlation_id UUID,
    query TEXT NOT NULL,
    state_hash TEXT NOT NULL,
    loop_count INTEGER,
    stagnation_score INTEGER,
    resolution_strategy TEXT, -- 'force_termination', 'alternative_plan', 'user_clarification'
    final_answer TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Track state corruption incidents
CREATE TABLE state_corruption_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    correlation_id UUID,
    corrupting_node TEXT NOT NULL,
    failed_field TEXT NOT NULL,
    bad_value TEXT,
    expected_type TEXT,
    stack_trace TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### 9. Human-in-the-Loop Approvals

```sql
-- Track human approval requests and decisions
CREATE TABLE human_approval_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    correlation_id UUID,
    action_type TEXT NOT NULL, -- 'send_email', 'execute_code', 'modify_data', 'api_call'
    action_description TEXT NOT NULL,
    action_parameters JSONB NOT NULL,
    risk_level TEXT NOT NULL, -- 'low', 'medium', 'high', 'critical'
    reasoning TEXT,
    alternatives JSONB,
    approval_status TEXT DEFAULT 'pending', -- 'pending', 'approved', 'rejected', 'timeout'
    approver_id TEXT,
    approval_timestamp TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for pending approvals
CREATE INDEX idx_human_approval_pending ON human_approval_requests(approval_status) WHERE approval_status = 'pending';
```

### 10. Session Management Table

```sql
-- Track user sessions and conversation history
CREATE TABLE user_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT,
    conversation_history JSONB DEFAULT '[]'::jsonb,
    total_queries INTEGER DEFAULT 0,
    successful_queries INTEGER DEFAULT 0,
    failed_queries INTEGER DEFAULT 0,
    average_steps_per_query REAL DEFAULT 0.0,
    last_active_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for user lookup
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_last_active ON user_sessions(last_active_at);
```

### 11. Create Update Triggers

```sql
-- Auto-update updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to tables with updated_at
CREATE TRIGGER update_knowledge_base_updated_at BEFORE UPDATE ON knowledge_base
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tool_reliability_updated_at BEFORE UPDATE ON tool_reliability_metrics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_lifecycle_updated_at BEFORE UPDATE ON knowledge_lifecycle
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_sessions_updated_at BEFORE UPDATE ON user_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

### 12. Row Level Security (RLS) Setup

```sql
-- Enable RLS for security
ALTER TABLE knowledge_base ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_trajectory_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE tool_reliability_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;

-- Create policies (adjust based on your authentication setup)
-- Example: Allow authenticated users to read knowledge base
CREATE POLICY "Allow authenticated read access" ON knowledge_base
    FOR SELECT
    TO authenticated
    USING (true);

-- Example: Allow service role full access
CREATE POLICY "Service role full access" ON knowledge_base
    TO service_role
    USING (true)
    WITH CHECK (true);
```

## Verification Queries

After running all the setup SQL, verify your tables are created correctly:

```sql
-- Check all tables are created
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;

-- Check pgvector extension is enabled
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Test vector similarity function
SELECT match_documents(
    array_fill(0.1, ARRAY[1536])::vector,
    5
);
```

## Maintenance Queries

### Clean up old logs (run periodically)

```sql
-- Delete logs older than 30 days
DELETE FROM agent_trajectory_logs 
WHERE timestamp < NOW() - INTERVAL '30 days';

-- Delete unused tool metrics
DELETE FROM tool_reliability_metrics 
WHERE last_used_at < NOW() - INTERVAL '90 days' 
AND total_calls < 10;
```

### Performance monitoring

```sql
-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

## Next Steps

1. Run these SQL commands in your Supabase SQL Editor
2. Update your `.env` file with your Supabase credentials
3. Test the connection with:
   ```python
   from src.database import get_supabase_client
   client = get_supabase_client()
   print("Connection successful!")
   ```
4. Consider setting up database backups in Supabase dashboard
5. Monitor usage and costs in your Supabase project settings

## Troubleshooting

- **pgvector not available**: Make sure you're on a Supabase plan that supports pgvector
- **Permission denied**: Check that your API key has the correct permissions
- **Connection errors**: Verify your SUPABASE_URL format and network connectivity
- **Performance issues**: Consider adding more specific indexes based on your query patterns 