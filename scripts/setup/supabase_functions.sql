-- Hybrid search function for enhanced search capabilities
CREATE OR REPLACE FUNCTION hybrid_match_documents(
    query_embedding vector(1536),
    match_count int,
    metadata_filter jsonb DEFAULT '{}',
    query_text text DEFAULT ''
) RETURNS TABLE (
    id uuid,
    node_id text,
    text text,
    metadata_ jsonb,
    similarity float,
    source text
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        kb.id,
        kb.node_id,
        kb.text,
        kb.metadata_,
        1 - (kb.embedding <=> query_embedding) AS similarity,
        COALESCE(kb.metadata_->>'source', 'unknown') AS source
    FROM knowledge_base kb
    WHERE 
        CASE 
            WHEN metadata_filter = '{}' THEN true
            ELSE kb.metadata_ @> metadata_filter
        END
    ORDER BY kb.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Basic vector similarity search function (fallback)
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1536),
    match_count int
) RETURNS TABLE (
    id uuid,
    node_id text,
    text text,
    metadata_ jsonb,
    similarity float
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        kb.id,
        kb.node_id,
        kb.text,
        kb.metadata_,
        1 - (kb.embedding <=> query_embedding) AS similarity
    FROM knowledge_base kb
    ORDER BY kb.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Knowledge base table creation
CREATE TABLE IF NOT EXISTS knowledge_base (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id TEXT UNIQUE NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    text TEXT,
    metadata_ JSONB DEFAULT '{}'::jsonb,
    
    -- Link to your lifecycle tracking
    lifecycle_id TEXT REFERENCES knowledge_lifecycle(document_id),
    
    -- Performance tracking
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS knowledge_base_embedding_idx 
ON knowledge_base USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index for metadata filtering
CREATE INDEX IF NOT EXISTS knowledge_base_metadata_idx 
ON knowledge_base USING gin (metadata_);

-- Create index for node_id lookups
CREATE INDEX IF NOT EXISTS knowledge_base_node_id_idx 
ON knowledge_base (node_id);

-- Create the optimized HNSW index
CREATE INDEX idx_kb_embedding_hnsw ON knowledge_base 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 24, ef_construction = 200);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_knowledge_base_updated_at 
    BEFORE UPDATE ON knowledge_base 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Supabase SQL Functions for AI Agent System
-- These functions provide the missing RPC endpoints referenced in database_enhanced.py

-- Enable vector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Create knowledge_base table if it doesn't exist
CREATE TABLE IF NOT EXISTS knowledge_base (
    id BIGSERIAL PRIMARY KEY,
    node_id TEXT UNIQUE NOT NULL,
    text TEXT NOT NULL,
    embedding vector(1536), -- OpenAI embedding dimension
    metadata_ JSONB DEFAULT '{}',
    source TEXT DEFAULT 'unknown',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS knowledge_base_embedding_idx ON knowledge_base USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS knowledge_base_text_idx ON knowledge_base USING gin(to_tsvector('english', text));
CREATE INDEX IF NOT EXISTS knowledge_base_metadata_idx ON knowledge_base USING gin(metadata_);

-- Function for simple vector similarity search
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1536),
    match_count int DEFAULT 5,
    filter_metadata jsonb DEFAULT '{}'
)
RETURNS TABLE (
    id bigint,
    text text,
    metadata_ jsonb,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        kb.id,
        kb.text,
        kb.metadata_,
        1 - (kb.embedding <=> query_embedding) as similarity
    FROM knowledge_base kb
    WHERE kb.embedding IS NOT NULL
        AND (filter_metadata = '{}' OR kb.metadata_ @> filter_metadata)
    ORDER BY kb.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Function for hybrid search combining vector similarity and text search
CREATE OR REPLACE FUNCTION hybrid_match_documents(
    query_embedding vector(1536),
    query_text text,
    match_count int DEFAULT 5,
    metadata_filter jsonb DEFAULT '{}'
)
RETURNS TABLE (
    id bigint,
    text text,
    metadata_ jsonb,
    similarity float,
    source text
)
LANGUAGE plpgsql
AS $$
DECLARE
    vector_weight float := 0.7;
    text_weight float := 0.3;
BEGIN
    RETURN QUERY
    SELECT
        kb.id,
        kb.text,
        kb.metadata_,
        (vector_weight * (1 - (kb.embedding <=> query_embedding)) + 
         text_weight * ts_rank(to_tsvector('english', kb.text), plainto_tsquery('english', query_text))) as similarity,
        kb.source
    FROM knowledge_base kb
    WHERE kb.embedding IS NOT NULL
        AND (metadata_filter = '{}' OR kb.metadata_ @> metadata_filter)
        AND to_tsvector('english', kb.text) @@ plainto_tsquery('english', query_text)
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

-- Function for BM25-style text search
CREATE OR REPLACE FUNCTION bm25_search(
    query_text text,
    match_count int DEFAULT 5,
    metadata_filter jsonb DEFAULT '{}'
)
RETURNS TABLE (
    id bigint,
    text text,
    metadata_ jsonb,
    score float,
    source text
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        kb.id,
        kb.text,
        kb.metadata_,
        ts_rank_cd(to_tsvector('english', kb.text), plainto_tsquery('english', query_text)) as score,
        kb.source
    FROM knowledge_base kb
    WHERE (metadata_filter = '{}' OR kb.metadata_ @> metadata_filter)
        AND to_tsvector('english', kb.text) @@ plainto_tsquery('english', query_text)
    ORDER BY score DESC
    LIMIT match_count;
END;
$$;

-- Function to insert or update documents with embeddings
CREATE OR REPLACE FUNCTION upsert_document(
    p_node_id text,
    p_text text,
    p_embedding vector(1536),
    p_metadata jsonb DEFAULT '{}',
    p_source text DEFAULT 'unknown'
)
RETURNS bigint
LANGUAGE plpgsql
AS $$
DECLARE
    doc_id bigint;
BEGIN
    INSERT INTO knowledge_base (node_id, text, embedding, metadata_, source)
    VALUES (p_node_id, p_text, p_embedding, p_metadata, p_source)
    ON CONFLICT (node_id) 
    DO UPDATE SET
        text = EXCLUDED.text,
        embedding = EXCLUDED.embedding,
        metadata_ = EXCLUDED.metadata_,
        source = EXCLUDED.source,
        updated_at = NOW()
    RETURNING id INTO doc_id;
    
    RETURN doc_id;
END;
$$;

-- Function to batch insert documents
CREATE OR REPLACE FUNCTION batch_insert_documents(
    documents jsonb
)
RETURNS int
LANGUAGE plpgsql
AS $$
DECLARE
    doc jsonb;
    inserted_count int := 0;
BEGIN
    FOR doc IN SELECT * FROM jsonb_array_elements(documents)
    LOOP
        INSERT INTO knowledge_base (
            node_id, 
            text, 
            embedding, 
            metadata_, 
            source
        )
        VALUES (
            (doc->>'node_id')::text,
            (doc->>'text')::text,
            (doc->>'embedding')::vector(1536),
            COALESCE(doc->'metadata_', '{}'::jsonb),
            COALESCE(doc->>'source', 'unknown')
        )
        ON CONFLICT (node_id) 
        DO UPDATE SET
            text = EXCLUDED.text,
            embedding = EXCLUDED.embedding,
            metadata_ = EXCLUDED.metadata_,
            source = EXCLUDED.source,
            updated_at = NOW();
        
        inserted_count := inserted_count + 1;
    END LOOP;
    
    RETURN inserted_count;
END;
$$;

-- Function to get document statistics
CREATE OR REPLACE FUNCTION get_document_stats()
RETURNS TABLE (
    total_documents bigint,
    total_embeddings bigint,
    avg_text_length float,
    sources text[]
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*) as total_documents,
        COUNT(embedding) as total_embeddings,
        AVG(LENGTH(text)) as avg_text_length,
        ARRAY_AGG(DISTINCT source) as sources
    FROM knowledge_base;
END;
$$;

-- Function to search by metadata only
CREATE OR REPLACE FUNCTION search_by_metadata(
    metadata_filter jsonb,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id bigint,
    text text,
    metadata_ jsonb,
    source text
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        kb.id,
        kb.text,
        kb.metadata_,
        kb.source
    FROM knowledge_base kb
    WHERE kb.metadata_ @> metadata_filter
    ORDER BY kb.created_at DESC
    LIMIT match_count;
END;
$$;

-- Function to delete documents by metadata
CREATE OR REPLACE FUNCTION delete_documents_by_metadata(
    metadata_filter jsonb
)
RETURNS int
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_count int;
BEGIN
    DELETE FROM knowledge_base
    WHERE metadata_ @> metadata_filter;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;

-- Create tool_metrics table for monitoring
CREATE TABLE IF NOT EXISTS tool_metrics (
    id BIGSERIAL PRIMARY KEY,
    tool_name TEXT NOT NULL,
    execution_time_ms INTEGER,
    success BOOLEAN,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for tool metrics
CREATE INDEX IF NOT EXISTS tool_metrics_tool_name_idx ON tool_metrics(tool_name);
CREATE INDEX IF NOT EXISTS tool_metrics_created_at_idx ON tool_metrics(created_at);

-- Function to record tool execution metrics
CREATE OR REPLACE FUNCTION record_tool_metric(
    p_tool_name text,
    p_execution_time_ms integer,
    p_success boolean,
    p_error_message text DEFAULT NULL,
    p_metadata jsonb DEFAULT '{}'
)
RETURNS bigint
LANGUAGE plpgsql
AS $$
DECLARE
    metric_id bigint;
BEGIN
    INSERT INTO tool_metrics (
        tool_name, 
        execution_time_ms, 
        success, 
        error_message, 
        metadata
    )
    VALUES (
        p_tool_name,
        p_execution_time_ms,
        p_success,
        p_error_message,
        p_metadata
    )
    RETURNING id INTO metric_id;
    
    RETURN metric_id;
END;
$$;

-- Function to get tool reliability statistics
CREATE OR REPLACE FUNCTION get_tool_reliability_stats(
    days_back integer DEFAULT 7
)
RETURNS TABLE (
    tool_name text,
    total_executions bigint,
    successful_executions bigint,
    success_rate float,
    avg_execution_time float,
    last_execution timestamp with time zone
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        tm.tool_name,
        COUNT(*) as total_executions,
        COUNT(*) FILTER (WHERE tm.success) as successful_executions,
        ROUND(
            COUNT(*) FILTER (WHERE tm.success)::float / COUNT(*)::float * 100, 
            2
        ) as success_rate,
        ROUND(AVG(tm.execution_time_ms), 2) as avg_execution_time,
        MAX(tm.created_at) as last_execution
    FROM tool_metrics tm
    WHERE tm.created_at >= NOW() - INTERVAL '1 day' * days_back
    GROUP BY tm.tool_name
    ORDER BY success_rate DESC;
END;
$$;

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO anon, authenticated;

-- ============================================
-- WORLD-CLASS SEARCH FUNCTION WITH YOUR FEATURES
-- ============================================

CREATE OR REPLACE FUNCTION intelligent_search(
    query_text TEXT,
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 5,
    filter JSONB DEFAULT '{}',
    use_clarification BOOLEAN DEFAULT TRUE,
    session_id UUID DEFAULT NULL
) RETURNS TABLE (
    -- Search results
    id UUID,
    node_id TEXT,
    text TEXT,
    metadata_ JSONB,
    similarity FLOAT,
    rank INT,
    
    -- Intelligence features
    needs_clarification BOOLEAN,
    suggested_clarification TEXT,
    reliability_score FLOAT,
    freshness_score FLOAT
) 
LANGUAGE plpgsql
AS $$
DECLARE
    clarification_needed BOOLEAN := FALSE;
    clarification_text TEXT;
    query_category TEXT;
BEGIN
    -- Step 1: Check if we need clarification
    IF use_clarification THEN
        SELECT 
            cp.clarification_question,
            cp.query_category
        INTO clarification_text, query_category
        FROM clarification_patterns cp
        WHERE 1 - (cp.query_embedding <=> query_embedding) > 0.85
        ORDER BY cp.effectiveness_score DESC, cp.frequency DESC
        LIMIT 1;
        
        clarification_needed := (clarification_text IS NOT NULL);
    END IF;
    
    -- Step 2: Log query to session if provided
    IF session_id IS NOT NULL THEN
        UPDATE user_sessions
        SET 
            total_queries = total_queries + 1,
            last_active_at = NOW(),
            conversation_history = conversation_history || 
                jsonb_build_object(
                    'query', query_text,
                    'timestamp', NOW(),
                    'category', COALESCE(query_category, 'general')
                )
        WHERE session_id = session_id;
    END IF;
    
    -- Step 3: Perform intelligent search with freshness and reliability
    RETURN QUERY
    WITH base_results AS (
        SELECT 
            kb.id,
            kb.node_id,
            kb.text,
            kb.metadata_,
            1 - (kb.embedding <=> query_embedding) AS base_similarity,
            kb.lifecycle_id,
            kb.access_count
        FROM knowledge_base kb
        WHERE 
            CASE 
                WHEN filter = '{}' THEN TRUE
                ELSE kb.metadata_ @> filter
            END
            AND 1 - (kb.embedding <=> query_embedding) > 0.5  -- Minimum threshold
        ORDER BY kb.embedding <=> query_embedding
        LIMIT match_count * 3  -- Get extra for reranking
    ),
    enhanced_results AS (
        SELECT 
            br.*,
            -- Calculate freshness score
            CASE 
                WHEN kl.validation_status = 'valid' THEN 1.0
                WHEN kl.validation_status = 'stale' THEN 0.7
                WHEN kl.validation_status = 'expired' THEN 0.3
                ELSE 0.5
            END * GREATEST(0, 1 - (EXTRACT(EPOCH FROM (NOW() - kl.last_validated_at)) / 86400 / 30)) AS freshness,
            
            -- Calculate reliability based on source
            CASE 
                WHEN br.metadata_->>'source_type' = 'official' THEN 1.0
                WHEN br.metadata_->>'source_type' = 'verified' THEN 0.9
                WHEN br.metadata_->>'source_type' = 'community' THEN 0.7
                ELSE 0.5
            END AS source_reliability,
            
            -- Boost for frequently accessed
            1 + (0.1 * LN(GREATEST(1, br.access_count))) AS popularity_boost
            
        FROM base_results br
        LEFT JOIN knowledge_lifecycle kl ON br.lifecycle_id = kl.document_id
    ),
    final_scored AS (
        SELECT 
            er.*,
            -- Combine all factors for final score
            er.base_similarity * 
            er.freshness * 
            er.source_reliability * 
            er.popularity_boost AS final_score
        FROM enhanced_results er
    )
    SELECT 
        fs.id,
        fs.node_id,
        fs.text,
        fs.metadata_,
        fs.final_score AS similarity,
        ROW_NUMBER() OVER (ORDER BY fs.final_score DESC)::INT AS rank,
        clarification_needed AS needs_clarification,
        clarification_text AS suggested_clarification,
        fs.source_reliability AS reliability_score,
        fs.freshness AS freshness_score
    FROM final_scored fs
    ORDER BY fs.final_score DESC
    LIMIT match_count;
    
    -- Update access counts
    UPDATE knowledge_base 
    SET 
        access_count = access_count + 1,
        last_accessed_at = NOW()
    WHERE id IN (
        SELECT id FROM final_scored 
        ORDER BY final_score DESC 
        LIMIT match_count
    );
END;
$$;

-- ============================================
-- TOOL RELIABILITY ENHANCEMENT
-- ============================================

-- Add function to get best tool for a task
CREATE OR REPLACE FUNCTION get_reliable_tool(
    tool_category TEXT,
    min_success_rate FLOAT DEFAULT 0.7
) RETURNS TABLE (
    tool_name TEXT,
    success_rate FLOAT,
    avg_latency_ms FLOAT,
    is_fallback BOOLEAN
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH tool_stats AS (
        SELECT 
            t.tool_name,
            CASE 
                WHEN t.total_calls > 0 
                THEN t.success_count::FLOAT / t.total_calls 
                ELSE 0.5  -- Default for new tools
            END AS success_rate,
            t.average_latency_ms,
            FALSE AS is_fallback
        FROM tool_reliability_metrics t
        WHERE t.tool_name LIKE tool_category || '%'
        AND t.last_used_at > NOW() - INTERVAL '30 days'
    )
    SELECT * FROM tool_stats
    WHERE success_rate >= min_success_rate
    
    UNION ALL
    
    -- Include fallback tools if main tools are unreliable
    SELECT 
        ft.tool_name,
        0.5 AS success_rate,
        1000.0 AS avg_latency_ms,
        TRUE AS is_fallback
    FROM tool_reliability_metrics t
    CROSS JOIN LATERAL jsonb_array_elements_text(t.fallback_tools) AS ft(tool_name)
    WHERE t.tool_name LIKE tool_category || '%'
    AND NOT EXISTS (
        SELECT 1 FROM tool_stats ts 
        WHERE ts.success_rate >= min_success_rate
    )
    
    ORDER BY is_fallback ASC, success_rate DESC, avg_latency_ms ASC
    LIMIT 1;
END;
$$;

-- ============================================
-- CLARIFICATION LEARNING SYSTEM
-- ============================================

CREATE OR REPLACE FUNCTION learn_from_clarification(
    p_original_query TEXT,
    p_query_embedding VECTOR(1536),
    p_clarification_question TEXT,
    p_user_response TEXT,
    p_was_helpful BOOLEAN
) RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    pattern_id TEXT;
BEGIN
    pattern_id := encode(digest(p_original_query || p_clarification_question, 'sha256'), 'hex');
    
    INSERT INTO clarification_patterns (
        id,
        original_query,
        query_embedding,
        clarification_question,
        user_response,
        query_category,
        effectiveness_score
    ) VALUES (
        pattern_id,
        p_original_query,
        p_query_embedding,
        p_clarification_question,
        p_user_response,
        'general',  -- You might want to classify this
        CASE WHEN p_was_helpful THEN 0.8 ELSE 0.2 END
    )
    ON CONFLICT (id) DO UPDATE
    SET 
        frequency = clarification_patterns.frequency + 1,
        effectiveness_score = 
            (clarification_patterns.effectiveness_score * clarification_patterns.frequency + 
             CASE WHEN p_was_helpful THEN 1.0 ELSE 0.0 END) / 
            (clarification_patterns.frequency + 1),
        last_seen_at = NOW();
END;
$$;

-- ============================================
-- KNOWLEDGE LIFECYCLE AUTOMATION
-- ============================================

-- Function to automatically mark stale documents
CREATE OR REPLACE FUNCTION update_stale_documents()
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    UPDATE knowledge_lifecycle
    SET 
        validation_status = 
            CASE 
                WHEN expires_at < NOW() THEN 'expired'
                WHEN last_validated_at < NOW() - (update_frequency_days || ' days')::INTERVAL THEN 'stale'
                ELSE validation_status
            END,
        updated_at = NOW()
    WHERE validation_status IN ('valid', 'stale')
    AND (
        expires_at < NOW() 
        OR last_validated_at < NOW() - (update_frequency_days || ' days')::INTERVAL
    );
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$;

-- ============================================
-- SESSION ANALYTICS
-- ============================================

CREATE OR REPLACE VIEW session_analytics AS
SELECT 
    user_id,
    COUNT(DISTINCT session_id) as total_sessions,
    SUM(total_queries) as total_queries,
    AVG(CASE WHEN total_queries > 0 
        THEN successful_queries::FLOAT / total_queries 
        ELSE 0 END) as avg_success_rate,
    AVG(average_steps_per_query) as avg_steps,
    MAX(last_active_at) as last_seen
FROM user_sessions
GROUP BY user_id;

-- ============================================
-- ERROR PATTERN DETECTION
-- ============================================

CREATE OR REPLACE FUNCTION detect_error_patterns()
RETURNS TABLE (
    error_type TEXT,
    frequency INTEGER,
    common_queries TEXT[],
    suggested_fix TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH error_analysis AS (
        SELECT 
            CASE 
                WHEN state_hash IN (
                    SELECT state_hash 
                    FROM recursion_error_logs 
                    GROUP BY state_hash 
                    HAVING COUNT(*) > 3
                ) THEN 'recurring_loop'
                WHEN loop_count > 10 THEN 'excessive_iteration'
                WHEN stagnation_score > 5 THEN 'stagnation_pattern'
                ELSE 'other'
            END AS error_type,
            query,
            resolution_strategy
        FROM recursion_error_logs
        WHERE created_at > NOW() - INTERVAL '7 days'
    )
    SELECT 
        ea.error_type,
        COUNT(*)::INTEGER as frequency,
        ARRAY_AGG(DISTINCT LEFT(ea.query, 50)) as common_queries,
        CASE ea.error_type
            WHEN 'recurring_loop' THEN 'Implement state diversity check'
            WHEN 'excessive_iteration' THEN 'Add iteration limits to planning'
            WHEN 'stagnation_pattern' THEN 'Force tool switching after stagnation'
            ELSE 'Review agent logs for pattern'
        END as suggested_fix
    FROM error_analysis ea
    GROUP BY ea.error_type
    HAVING COUNT(*) > 2
    ORDER BY COUNT(*) DESC;
END;
$$;

-- ============================================
-- PERFORMANCE MONITORING DASHBOARD
-- ============================================

CREATE OR REPLACE VIEW agent_performance_dashboard AS
WITH recent_metrics AS (
    SELECT * FROM tool_reliability_metrics
    WHERE last_used_at > NOW() - INTERVAL '24 hours'
)
SELECT 
    'Active Tools' as metric,
    COUNT(DISTINCT tool_name)::TEXT as value,
    'tools' as unit
FROM recent_metrics
UNION ALL
SELECT 
    'Average Success Rate',
    ROUND(AVG(CASE WHEN total_calls > 0 
        THEN success_count::FLOAT / total_calls 
        ELSE 0 END) * 100, 1)::TEXT,
    '%'
FROM recent_metrics
UNION ALL
SELECT 
    'Pending Approvals',
    COUNT(*)::TEXT,
    'requests'
FROM human_approval_requests
WHERE approval_status = 'pending'
UNION ALL
SELECT 
    'Knowledge Freshness',
    ROUND(100.0 * COUNT(CASE WHEN validation_status = 'valid' THEN 1 END) / 
          NULLIF(COUNT(*), 0), 1)::TEXT,
    '%'
FROM knowledge_lifecycle; 