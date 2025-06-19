-- Supabase Setup SQL for AI Agent
-- Run this entire script in your Supabase SQL Editor

-- 1. Enable Required Extensions (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Tool Reliability Metrics Table
CREATE TABLE IF NOT EXISTS tool_reliability_metrics (
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

CREATE INDEX IF NOT EXISTS idx_tool_reliability_last_used ON tool_reliability_metrics(last_used_at);

-- 3. Clarification Patterns Table
CREATE TABLE IF NOT EXISTS clarification_patterns (
    id TEXT PRIMARY KEY,
    original_query TEXT NOT NULL,
    query_embedding VECTOR(1536),
    clarification_question TEXT NOT NULL,
    user_response TEXT NOT NULL,
    query_category TEXT NOT NULL,
    frequency INTEGER DEFAULT 1,
    effectiveness_score REAL DEFAULT 0.5,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_clarification_patterns_category ON clarification_patterns(query_category);
CREATE INDEX IF NOT EXISTS idx_clarification_patterns_embedding ON clarification_patterns USING hnsw (query_embedding vector_cosine_ops);

-- 4. Plan Corrections Table
CREATE TABLE IF NOT EXISTS plan_corrections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT NOT NULL,
    original_plan JSONB NOT NULL,
    corrected_plan JSONB NOT NULL,
    correction_type TEXT NOT NULL,
    user_feedback TEXT,
    applied_to_future_plans BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_plan_corrections_query ON plan_corrections USING gin(to_tsvector('english', query));

-- 5. Knowledge Lifecycle Table
CREATE TABLE IF NOT EXISTS knowledge_lifecycle (
    document_id TEXT PRIMARY KEY,
    source_url TEXT,
    document_type TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    ingested_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_validated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    validation_status TEXT NOT NULL,
    update_frequency_days INTEGER NOT NULL,
    importance_score REAL DEFAULT 0.5,
    validation_failures INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_lifecycle_expires ON knowledge_lifecycle(expires_at);
CREATE INDEX IF NOT EXISTS idx_knowledge_lifecycle_validation ON knowledge_lifecycle(last_validated_at);
CREATE INDEX IF NOT EXISTS idx_knowledge_lifecycle_status ON knowledge_lifecycle(validation_status);

-- 6. Recursion Error Logs Table
CREATE TABLE IF NOT EXISTS recursion_error_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    correlation_id UUID,
    query TEXT NOT NULL,
    state_hash TEXT NOT NULL,
    loop_count INTEGER,
    stagnation_score INTEGER,
    resolution_strategy TEXT,
    final_answer TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 7. State Corruption Logs Table
CREATE TABLE IF NOT EXISTS state_corruption_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    correlation_id UUID,
    corrupting_node TEXT NOT NULL,
    failed_field TEXT NOT NULL,
    bad_value TEXT,
    expected_type TEXT,
    stack_trace TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 8. Human Approval Requests Table
CREATE TABLE IF NOT EXISTS human_approval_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    correlation_id UUID,
    action_type TEXT NOT NULL,
    action_description TEXT NOT NULL,
    action_parameters JSONB NOT NULL,
    risk_level TEXT NOT NULL,
    reasoning TEXT,
    alternatives JSONB,
    approval_status TEXT DEFAULT 'pending',
    approver_id TEXT,
    approval_timestamp TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_human_approval_pending ON human_approval_requests(approval_status) WHERE approval_status = 'pending';

-- 9. User Sessions Table
CREATE TABLE IF NOT EXISTS user_sessions (
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

CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_last_active ON user_sessions(last_active_at);

-- 10. Create Update Triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to tables with updated_at
CREATE TRIGGER update_tool_reliability_updated_at BEFORE UPDATE ON tool_reliability_metrics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_lifecycle_updated_at BEFORE UPDATE ON knowledge_lifecycle
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_sessions_updated_at BEFORE UPDATE ON user_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 11. Verify all tables were created
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN (
    'knowledge_base',
    'agent_trajectory_logs',
    'tool_reliability_metrics',
    'clarification_patterns',
    'plan_corrections',
    'knowledge_lifecycle',
    'recursion_error_logs',
    'state_corruption_logs',
    'human_approval_requests',
    'user_sessions'
)
ORDER BY table_name; 