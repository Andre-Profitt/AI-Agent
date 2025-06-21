from fix_import_hierarchy import file_path
from setup_environment import value

from src.collaboration.realtime_collaboration import updates
from src.config.integrations import is_valid
from src.config.settings import issues
from src.core.monitoring import key
from src.infrastructure.config_cli import api_keys
from src.infrastructure.config_cli import config_dict
from src.infrastructure.config_cli import env_vars
from src.tools_introspection import description
from src.tools_introspection import name

from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent
# TODO: Fix undefined variables: Any, api_keys, config_dict, description, env_var, env_vars, file_path, is_valid, issue, issues, json, key, logging, name, section, updates, value, var

"""
Configuration CLI Tool for AI Agent
Provides command-line interface for managing integration configuration,
validation, and file operations.
"""

from typing import Any

import click
import json


import logging


try:
    from .config.integrations import integration_config
except ImportError:
    try:
        from config.integrations import integration_config
    except ImportError:
        # Fallback for when running as standalone script
        integration_config = None
        logging.warning("Could not import integration_config - using defaults")

@click.group()
def cli() -> Any:
    """Integration configuration management"""
    pass

@cli.command()
def validate() -> bool:
    """Validate current configuration"""
    is_valid, issues = integration_config.validate()
    if is_valid:
        click.echo("✅ Configuration is valid!")
    else:
        click.echo("❌ Configuration issues found:")
        for issue in issues:
            click.echo(f"  - {issue}")

@cli.command()
def show() -> Any:
    """Show current configuration (without sensitive data)"""
    config_dict = integration_config.to_dict()
    click.echo("Current Configuration:")
    click.echo(json.dumps(config_dict, indent=2))

@cli.command()
@click.argument('file_path')
def save(file_path) -> bool:
    """Save configuration to file"""
    if integration_config.save_to_file(file_path):
        click.echo(f"✅ Configuration saved to {file_path}")
    else:
        click.echo("❌ Failed to save configuration")

@cli.command()
@click.argument('file_path')
def load(file_path) -> Any:
    """Load configuration from file"""
    if integration_config.load_from_file(file_path):
        click.echo(f"✅ Configuration loaded from {file_path}")
    else:
        click.echo("❌ Failed to load configuration")

@cli.command()
def env() -> Any:
    """Show environment variables for configuration"""
    click.echo("Environment Variables for Configuration:")
    click.echo("=" * 50)
    
    env_vars = {
        "SUPABASE_URL": "Supabase project URL",
        "SUPABASE_KEY": "Supabase anon key",
        "SUPABASE_SERVICE_KEY": "Supabase service role key",
        "SUPABASE_DB_PASSWORD": "Database password",
        "OPENAI_API_KEY": "OpenAI API key",
        "ANTHROPIC_API_KEY": "Anthropic API key",
        "GROQ_API_KEY": "Groq API key",
        "LANGSMITH_TRACING": "Enable LangSmith tracing (true/false)",
        "LANGSMITH_PROJECT": "LangSmith project name",
        "LANGSMITH_API_KEY": "LangSmith API key",
        "CREWAI_ENABLED": "Enable CrewAI (true/false)",
        "CREWAI_MAX_AGENTS": "Maximum number of CrewAI agents",
        "LLAMAINDEX_STORAGE_PATH": "LlamaIndex storage path",
        "LLAMAINDEX_CHUNK_SIZE": "LlamaIndex chunk size",
        "GAIA_TOOLS_ENABLED": "Enable GAIA tools (true/false)",
        "GAIA_TIMEOUT": "GAIA timeout in seconds"
    }
    
    for var, description in env_vars.items():
        value = "***" if "KEY" in var or "PASSWORD" in var else "not set"
        click.echo(f"{var}: {description}")
        click.echo(f"  Current value: {value}")
        click.echo()

@cli.command()
@click.option('--section', help='Configuration section to update')
@click.option('--key', help='Configuration key to update')
@click.option('--value', help='New value')
def update(section, key, value) -> bool:
    """Update configuration values"""
    if not all([section, key, value]):
        click.echo("❌ Please provide section, key, and value")
        return
    
    updates = {section: {key: value}}
    if integration_config.update_config(updates):
        click.echo(f"✅ Updated {section}.{key} = {value}")
    else:
        click.echo("❌ Failed to update configuration")

@cli.command()
def test() -> Any:
    """Test all integrations"""
    click.echo("Testing integrations...")
    
    # Test Supabase
    if integration_config.supabase.is_configured():
        click.echo("✅ Supabase configured")
    else:
        click.echo("⚠️ Supabase not configured")
    
    # Test API keys
    api_keys = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY", 
        "Groq": "GROQ_API_KEY"
    }
    
    for name, env_var in api_keys.items():
        if integration_config._load_from_environment.__globals__['os'].getenv(env_var):
            click.echo(f"✅ {name} API key available")
        else:
            click.echo(f"⚠️ {name} API key not available")
    
    # Validate config
    is_valid, issues = integration_config.validate()
    if is_valid:
        click.echo("✅ Configuration validation passed")
    else:
        click.echo("❌ Configuration validation failed:")
        for issue in issues:
            click.echo(f"  - {issue}")

if __name__ == "__main__":
    cli() 