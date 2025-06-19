"""
Health Check Module for AI Agent
Provides comprehensive health monitoring for all integrations including
Supabase, LangChain, CrewAI, LlamaIndex, and GAIA components.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
import os

try:
    from .config.integrations import integration_config
except ImportError:
    try:
        from config.integrations import integration_config
    except ImportError:
        # Fallback for when running as standalone script
        integration_config = None
        logging.warning("Could not import integration_config - using defaults")

logger = logging.getLogger(__name__)

async def check_integrations_health() -> Dict[str, Any]:
    """Check health of all integrations"""
    
    health = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "unknown",
        "integrations": {}
    }
    
    # Check Supabase
    if integration_config.supabase.is_configured():
        try:
            from .database_enhanced import connection_pool
            if connection_pool and connection_pool._initialized:
                # Test connection
                async with connection_pool.get_client() as client:
                    await client.table("knowledge_base").select("count").limit(1).execute()
                health["integrations"]["supabase"] = {
                    "status": "healthy",
                    "details": "Connection pool active and database accessible"
                }
            else:
                health["integrations"]["supabase"] = {
                    "status": "unhealthy",
                    "details": "Connection pool not initialized"
                }
        except Exception as e:
            health["integrations"]["supabase"] = {
                "status": "unhealthy",
                "details": f"Connection failed: {str(e)}"
            }
    else:
        health["integrations"]["supabase"] = {
            "status": "not_configured",
            "details": "Supabase not configured"
        }
    
    # Check LlamaIndex
    try:
        from .llamaindex_enhanced import LLAMAINDEX_AVAILABLE
        if LLAMAINDEX_AVAILABLE:
            health["integrations"]["llamaindex"] = {
                "status": "available",
                "details": "LlamaIndex package installed and ready"
            }
        else:
            health["integrations"]["llamaindex"] = {
                "status": "not_installed",
                "details": "LlamaIndex package not available"
            }
    except Exception as e:
        health["integrations"]["llamaindex"] = {
            "status": "error",
            "details": f"Error checking LlamaIndex: {str(e)}"
        }
    
    # Check LangChain
    try:
        import langchain
        health["integrations"]["langchain"] = {
            "status": "available",
            "details": f"LangChain version {langchain.__version__} installed"
        }
    except ImportError:
        health["integrations"]["langchain"] = {
            "status": "not_installed",
            "details": "LangChain package not available"
        }
    except Exception as e:
        health["integrations"]["langchain"] = {
            "status": "error",
            "details": f"Error checking LangChain: {str(e)}"
        }
    
    # Check CrewAI
    try:
        import crewai
        health["integrations"]["crewai"] = {
            "status": "available",
            "details": f"CrewAI version {crewai.__version__} installed"
        }
    except ImportError:
        health["integrations"]["crewai"] = {
            "status": "not_installed",
            "details": "CrewAI package not available"
        }
    except Exception as e:
        health["integrations"]["crewai"] = {
            "status": "error",
            "details": f"Error checking CrewAI: {str(e)}"
        }
    
    # Check API Keys
    api_keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY"
    }
    
    health["api_keys"] = {}
    for name, env_var in api_keys.items():
        key_value = os.getenv(env_var)
        if key_value:
            health["api_keys"][name] = {
                "status": "configured",
                "details": f"API key available (length: {len(key_value)})"
            }
        else:
            health["api_keys"][name] = {
                "status": "not_configured",
                "details": "API key not set"
            }
    
    # Check Configuration
    is_valid, issues = integration_config.validate()
    health["configuration"] = {
        "status": "valid" if is_valid else "invalid",
        "issues": issues if issues else []
    }
    
    # Determine overall status
    healthy_count = 0
    total_count = 0
    
    for integration in health["integrations"].values():
        total_count += 1
        if integration["status"] in ["healthy", "available"]:
            healthy_count += 1
    
    if healthy_count == total_count:
        health["overall_status"] = "healthy"
    elif healthy_count > 0:
        health["overall_status"] = "degraded"
    else:
        health["overall_status"] = "unhealthy"
    
    return health

async def check_specific_integration(integration_name: str) -> Dict[str, Any]:
    """Check health of a specific integration"""
    
    health_checks = {
        "supabase": _check_supabase_health,
        "llamaindex": _check_llamaindex_health,
        "langchain": _check_langchain_health,
        "crewai": _check_crewai_health
    }
    
    if integration_name not in health_checks:
        return {
            "status": "error",
            "details": f"Unknown integration: {integration_name}"
        }
    
    try:
        return await health_checks[integration_name]()
    except Exception as e:
        return {
            "status": "error",
            "details": f"Health check failed: {str(e)}"
        }

async def _check_supabase_health() -> Dict[str, Any]:
    """Detailed Supabase health check"""
    if not integration_config.supabase.is_configured():
        return {
            "status": "not_configured",
            "details": "Supabase URL and key not configured"
        }
    
    try:
        from .database_enhanced import connection_pool
        
        if not connection_pool or not connection_pool._initialized:
            return {
                "status": "unhealthy",
                "details": "Connection pool not initialized"
            }
        
        # Test database connection
        async with connection_pool.get_client() as client:
            # Test basic query
            result = await client.table("knowledge_base").select("count").limit(1).execute()
            
            # Test vector search function
            await client.rpc(
                'match_documents',
                {
                    'query_embedding': [0.0] * 1536,  # Dummy embedding
                    'match_count': 1
                }
            ).execute()
        
        return {
            "status": "healthy",
            "details": "Database connection and functions working",
            "pool_size": connection_pool.pool_size,
            "pool_initialized": connection_pool._initialized
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "details": f"Database connection failed: {str(e)}"
        }

async def _check_llamaindex_health() -> Dict[str, Any]:
    """Detailed LlamaIndex health check"""
    try:
        from .llamaindex_enhanced import LLAMAINDEX_AVAILABLE
        
        if not LLAMAINDEX_AVAILABLE:
            return {
                "status": "not_installed",
                "details": "LlamaIndex package not available"
            }
        
        # Test basic imports
        from llama_index import VectorStoreIndex, Document
        from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding
        
        # Check if embedding models are available
        embedding_status = "local_only"
        if os.getenv("OPENAI_API_KEY"):
            embedding_status = "openai_available"
        
        return {
            "status": "available",
            "details": "LlamaIndex package installed and imports working",
            "embedding_status": embedding_status
        }
        
    except Exception as e:
        return {
            "status": "error",
            "details": f"LlamaIndex health check failed: {str(e)}"
        }

async def _check_langchain_health() -> Dict[str, Any]:
    """Detailed LangChain health check"""
    try:
        import langchain
        from langchain.schema import Document
        
        return {
            "status": "available",
            "details": "LangChain package installed and imports working",
            "version": langchain.__version__
        }
        
    except ImportError:
        return {
            "status": "not_installed",
            "details": "LangChain package not available"
        }
    except Exception as e:
        return {
            "status": "error",
            "details": f"LangChain health check failed: {str(e)}"
        }

async def _check_crewai_health() -> Dict[str, Any]:
    """Detailed CrewAI health check"""
    try:
        import crewai
        from crewai import Agent, Task, Crew
        
        return {
            "status": "available",
            "details": "CrewAI package installed and imports working",
            "version": crewai.__version__
        }
        
    except ImportError:
        return {
            "status": "not_installed",
            "details": "CrewAI package not available"
        }
    except Exception as e:
        return {
            "status": "error",
            "details": f"CrewAI health check failed: {str(e)}"
        }

def get_health_summary() -> Dict[str, Any]:
    """Get a quick health summary without async operations"""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config_valid": integration_config.validate()[0],
        "supabase_configured": integration_config.supabase.is_configured(),
        "api_keys_available": {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "groq": bool(os.getenv("GROQ_API_KEY"))
        }
    }
    
    return summary 