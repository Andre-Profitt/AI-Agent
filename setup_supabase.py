#!/usr/bin/env python3
"""
Supabase Setup Script for AI Agent
This script helps verify and set up your Supabase database for the AI Agent.
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

try:
    from supabase import create_client, Client
    from src.database import get_supabase_client
    from src.database_extended import ExtendedDatabase
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install required packages: pip install supabase python-dotenv")
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
        print("✅ Successfully connected to Supabase!")
        return client
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        return None


def check_tables(client: Client) -> dict[str, bool]:
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


def check_extensions(client: Client) -> dict[str, bool]:
    """Check if required PostgreSQL extensions are enabled."""
    # Note: This requires direct SQL access which Supabase client doesn't provide
    # You'll need to check these manually in the Supabase SQL Editor
    print("\n⚠️  Please verify these extensions are enabled in your Supabase SQL Editor:")
    print("  - pgvector (for semantic search)")
    print("  - uuid-ossp (for UUID generation)")
    print("\nRun this SQL to check: SELECT * FROM pg_extension;")
    
    return {"pgvector": "unknown", "uuid-ossp": "unknown"}


def create_sample_data(client: Client) -> bool:
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
        
        print("✅ Sample data created successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
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
    
    print("✅ Created .env.template file")


def main():
    """Main setup function."""
    print("🚀 Supabase Setup Script for AI Agent")
    print("=" * 50)
    
    # Step 1: Check environment variables
    print("\n1. Checking environment variables...")
    env_ok, missing_vars = check_environment_variables()
    
    if not env_ok:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease create a .env file with the following variables:")
        for var in missing_vars:
            print(f"  {var}=your-value-here")
        
        generate_env_template()
        print("\nRefer to .env.template for a complete example.")
        return
    
    print("✅ All required environment variables are set")
    
    # Step 2: Test connection
    print("\n2. Testing Supabase connection...")
    client = test_connection()
    
    if not client:
        print("\n❌ Could not connect to Supabase.")
        print("Please check your SUPABASE_URL and SUPABASE_KEY.")
        return
    
    # Step 3: Check tables
    print("\n3. Checking database tables...")
    table_status = check_tables(client)
    
    missing_tables = [table for table, exists in table_status.items() if not exists]
    existing_tables = [table for table, exists in table_status.items() if exists]
    
    if existing_tables:
        print(f"\n✅ Found {len(existing_tables)} existing tables:")
        for table in existing_tables:
            print(f"  - {table}")
    
    if missing_tables:
        print(f"\n⚠️  Missing {len(missing_tables)} tables:")
        for table in missing_tables:
            print(f"  - {table}")
        print("\nPlease run the SQL commands from SUPABASE_SQL_SETUP.md in your Supabase SQL Editor.")
    
    # Step 4: Check extensions
    print("\n4. Checking PostgreSQL extensions...")
    check_extensions(client)
    
    # Step 5: Extended database setup
    print("\n5. Setting up extended database features...")
    try:
        ext_db = ExtendedDatabase()
        if ext_db.client:
            print("✅ Extended database initialized")
            
            # Test tool metrics
            metrics = ext_db.get_tool_metrics()
            print(f"  - Found {len(metrics)} tool metrics")
        else:
            print("⚠️  Extended database features not available (missing credentials)")
    except Exception as e:
        print(f"❌ Error with extended database: {e}")
    
    # Step 6: Summary
    print("\n" + "=" * 50)
    print("📊 Setup Summary:")
    
    if not missing_tables:
        print("✅ All tables are properly set up!")
        print("\n🎉 Your Supabase database is ready for the AI Agent!")
        
        # Optional: Create sample data
        response = input("\nWould you like to create sample data for testing? (y/n): ")
        if response.lower() == 'y':
            create_sample_data(client)
    else:
        print(f"⚠️  {len(missing_tables)} tables need to be created.")
        print("\nNext steps:")
        print("1. Open your Supabase project SQL Editor")
        print("2. Run the SQL commands from SUPABASE_SQL_SETUP.md")
        print("3. Re-run this script to verify setup")
    
    print("\n📚 For complete setup instructions, see SUPABASE_SQL_SETUP.md")


if __name__ == "__main__":
    main() 