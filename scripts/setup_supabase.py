#!/usr/bin/env python3
from src.agents.advanced_agent_fsm import Agent
# TODO: Fix undefined variables: Client, Optional, client, e, env_ok, existing_tables, exists, expected_tables, ext_db, f, logging, metrics, missing_tables, missing_vars, os, required_vars, response, sys, table, table_status, template, var
from src.infrastructure.database import get_supabase_client

# TODO: Fix undefined variables: Client, client, e, env_ok, existing_tables, exists, expected_tables, ext_db, f, get_supabase_client, load_dotenv, metrics, missing_tables, missing_vars, required_vars, response, table, table_status, template, var

"""
Supabase Setup Script for AI Agent
This script helps verify and set up your Supabase database for the AI Agent.
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv

import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

try:
    from supabase import create_client, Client
    from src.database import get_supabase_client
    from src.database_extended import ExtendedDatabase
except ImportError as e:
    logger.info("Error importing required modules: {}", extra={"e": e})
    logger.info("Please install required packages: pip install supabase python-dotenv")
    sys.exit(1)

def check_environment_variables() -> tuple[bool, list[str]]:
    """Check if all required environment variables are set."""
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_DB_PASSWORD"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    return len(missing_vars) == 0, missing_vars

def test_connection() -> Optional[Client]:
    """Test the Supabase connection."""
    try:
        client = get_supabase_client()
        # Try a simple query to test connection
        result = client.table("knowledge_base").select("count", count="exact").execute()
        logger.info("✅ Successfully connected to Supabase!")
        return client
    except Exception as e:
        logger.info("❌ Failed to connect to Supabase: {}", extra={"e": e})
        return None

def check_tables(self, client: Client) -> dict[str, bool]:
    """Check which tables exist in the database."""
    expected_tables = [
        "knowledge_base",
        "agent_trajectory_logs",
        "tool_reliability_metrics",
        "clarification_patterns",
        "plan_corrections",
        "knowledge_lifecycle",
        "recursion_error_logs",
        "state_corruption_logs",
        "human_approval_requests",
        "user_sessions"
    ]

    table_status = {}

    for table in expected_tables:
        try:
            # Try to select from table (will fail if doesn't exist)
            client.table(table).select("count", count="exact").limit(0).execute()
            table_status[table] = True
        except Exception:
            table_status[table] = False

    return table_status

def check_extensions(self, client: Client) -> dict[str, bool]:
    """Check if required PostgreSQL extensions are enabled."""
    # Note: This requires direct SQL access which Supabase client doesn't provide
    # You'll need to check these manually in the Supabase SQL Editor
    logger.info("\n⚠️  Please verify these extensions are enabled in your Supabase SQL Editor:")
    logger.info("  - pgvector (for semantic search)")
    logger.info("  - uuid-ossp (for UUID generation)")
    logger.info("\nRun this SQL to check: SELECT * FROM pg_extension;")

    return {"pgvector": "unknown", "uuid-ossp": "unknown"}

def create_sample_data(self, client: Client) -> bool:
    """Create some sample data for testing."""
    try:
        # Insert a sample tool metric
        client.table("tool_reliability_metrics").upsert({
            "tool_name": "test_tool",
            "success_count": 10,
            "failure_count": 2,
            "total_calls": 12,
            "average_latency_ms": 150.5
        }).execute()

        logger.info("✅ Sample data created successfully!")
        return True
    except Exception as e:
        logger.info("❌ Failed to create sample data: {}", extra={"e": e})
        return False

def generate_env_template():
    """Generate a template .env file."""
    template = """# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-public-key
SUPABASE_DB_PASSWORD=your-database-password

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key

# Groq Configuration (for LLM)
GROQ_API_KEY=your-groq-api-key

# Tavily Configuration (for web search)
TAVILY_API_KEY=your-tavily-api-key

# Optional: CrewAI Configuration
CREWAI_API_KEY=your-crewai-api-key
"""

    with open(".env.template", "w") as f:
        f.write(template)

    logger.info("✅ Created .env.template file")

def main():
    """Main setup function."""
    logger.info("🚀 Supabase Setup Script for AI Agent")
    print("=" * 50)

    # Step 1: Check environment variables
    logger.info("\n1. Checking environment variables...")
    env_ok, missing_vars = check_environment_variables()

    if not env_ok:
        logger.info("❌ Missing environment variables: {}", extra={"_____join_missing_vars_": ', '.join(missing_vars)})
        logger.info("\nPlease create a .env file with the following variables:")
        for var in missing_vars:
            logger.info("  {}=your-value-here", extra={"var": var})

        generate_env_template()
        logger.info("\nRefer to .env.template for a complete example.")
        return

    logger.info("✅ All required environment variables are set")

    # Step 2: Test connection
    logger.info("\n2. Testing Supabase connection...")
    client = test_connection()

    if not client:
        logger.info("\n❌ Could not connect to Supabase.")
        logger.info("Please check your SUPABASE_URL and SUPABASE_KEY.")
        return

    # Step 3: Check tables
    logger.info("\n3. Checking database tables...")
    table_status = check_tables(client)

    missing_tables = [table for table, exists in table_status.items() if not exists]
    existing_tables = [table for table, exists in table_status.items() if exists]

    if existing_tables:
        logger.info("\n✅ Found {} existing tables:", extra={"len_existing_tables_": len(existing_tables)})
        for table in existing_tables:
            logger.info("  - {}", extra={"table": table})

    if missing_tables:
        logger.info("\n⚠️  Missing {} tables:", extra={"len_missing_tables_": len(missing_tables)})
        for table in missing_tables:
            logger.info("  - {}", extra={"table": table})
        logger.info("\nPlease run the SQL commands from SUPABASE_SQL_SETUP.md in your Supabase SQL Editor.")

    # Step 4: Check extensions
    logger.info("\n4. Checking PostgreSQL extensions...")
    check_extensions(client)

    # Step 5: Extended database setup
    logger.info("\n5. Setting up extended database features...")
    try:
        ext_db = ExtendedDatabase()
        if ext_db.client:
            logger.info("✅ Extended database initialized")

            # Test tool metrics
            metrics = ext_db.get_tool_metrics()
            logger.info("  - Found {} tool metrics", extra={"len_metrics_": len(metrics)})
        else:
            logger.info("⚠️  Extended database features not available (missing credentials)")
    except Exception as e:
        logger.info("❌ Error with extended database: {}", extra={"e": e})

    # Step 6: Summary
    print("\n" + "=" * 50)
    logger.info("📊 Setup Summary:")

    if not missing_tables:
        logger.info("✅ All tables are properly set up!")
        logger.info("\n🎉 Your Supabase database is ready for the AI Agent!")

        # Optional: Create sample data
        response = input("\nWould you like to create sample data for testing? (y/n): ")
        if response.lower() == 'y':
            create_sample_data(client)
    else:
        logger.info("⚠️  {} tables need to be created.", extra={"len_missing_tables_": len(missing_tables)})
        logger.info("\nNext steps:")
        logger.info("1. Open your Supabase project SQL Editor")
        logger.info("2. Run the SQL commands from SUPABASE_SQL_SETUP.md")
        logger.info("3. Re-run this script to verify setup")

    logger.info("\n📚 For complete setup instructions, see SUPABASE_SQL_SETUP.md")

if __name__ == "__main__":
    main()