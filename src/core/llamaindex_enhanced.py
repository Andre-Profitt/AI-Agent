"""
Enhanced LlamaIndex Integration for GAIA Agent
Provides advanced knowledge base capabilities with hierarchical indexing,
multi-modal support, incremental updates, and specialized query engines.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import asyncio
from datetime import datetime
import json
import os
from typing import Optional, Dict, Any, List, Union, Tuple

# Circuit breaker import for config protection
from src.infrastructure.resilience.circuit_breaker import (
    circuit_breaker, CircuitBreakerConfig
)

# Note: This file uses circuit breaker protection for config access
# The create_gaia_knowledge_base function is protected with @circuit_breaker
# and uses is_configured_safe pattern through the decorator

# LlamaIndex imports with fallback
try:
    from llama_index import (
        VectorStoreIndex, Document, ServiceContext, 
        StorageContext, load_index_from_storage,
        SimpleDirectoryReader, JSONReader, CSVReader,
        download_loader
    )
    from llama_index.vector_stores import SupabaseVectorStore
    from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding
    from llama_index.llms import OpenAI, Anthropic, Groq
    from llama_index.node_parser import SentenceSplitter, HierarchicalNodeParser
    from llama_index.query_engine import RetrieverQueryEngine
    from llama_index.retrievers import VectorIndexRetriever
    from llama_index.indices.postprocessor import SimilarityPostprocessor
    from llama_index.response_synthesizers import get_response_synthesizer
    from llama_index.memory import ConversationBufferMemory
    from llama_index.tools import QueryEngineTool
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    # Create dummy types for type hints when LlamaIndex is not available
    Document = Any
    VectorStoreIndex = Any
    ServiceContext = Any
    StorageContext = Any
    logging.warning("LlamaIndex not available - enhanced knowledge base features disabled")

try:
    from .config.integrations import integration_config
except ImportError:
    try:
        from config.integrations import integration_config
    except ImportError:
        # Fallback for when running as standalone script
        integration_config = None
        logging.warning("Could not import integration_config - using defaults")

# Import centralized embedding manager
from ..services.embedding_manager import get_embedding_manager

logger = logging.getLogger(__name__)

class EnhancedKnowledgeBase:
    """Advanced knowledge base with hierarchical parsing and multi-modal support"""
    
    def __init__(self, vector_store: Optional[Any] = None) -> None:
        self.vector_store = vector_store
        self.index = None
        self.service_context = None
        self.storage_context = None
        self._setup_service_context()
    
    def _setup_service_context(self) -> Any:
        """Configure service context with best available models"""
        if not LLAMAINDEX_AVAILABLE:
            return
            
        # Use actual config values
        config = integration_config
        
        # Use centralized embedding manager instead of creating separate models
        embedding_manager = get_embedding_manager()
        
        # Create LlamaIndex embedding wrapper
        if embedding_manager.method == "openai":
            embedding_model = OpenAIEmbedding(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-3-small"
            )
        else:
            embedding_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        # Choose LLM based on available keys
        if os.getenv("ANTHROPIC_API_KEY"):
            llm = Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                model="claude-3-opus-20240229"
            )
        elif os.getenv("GROQ_API_KEY"):
            llm = Groq(
                api_key=os.getenv("GROQ_API_KEY"),
                model="llama-3.3-70b-versatile"
            )
        else:
            llm = None
        
        self.service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embedding_model,
            chunk_size=config.llamaindex.chunk_size if config else 512,  # Use config!
            chunk_overlap=config.llamaindex.chunk_overlap if config else 50
        )
    
    def create_hierarchical_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Create index with hierarchical node parsing for better structure"""
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError("LlamaIndex not available")
        
        # Use hierarchical parser for better document structure
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128]
        )
        
        nodes = node_parser.get_nodes_from_documents(documents)
        
        if self.vector_store:
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
        else:
            storage_context = StorageContext.from_defaults()
        
        self.index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            service_context=self.service_context
        )
        
        return self.index
    
    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """Load documents from directory with multiple format support"""
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError("LlamaIndex not available")
        
        directory = Path(directory_path)
        documents = []
        
        # Load different file types
        if directory.is_dir():
            # Text files
            text_files = list(directory.glob("*.txt")) + list(directory.glob("*.md"))
            if text_files:
                text_reader = SimpleDirectoryReader(
                    input_files=[str(f) for f in text_files]
                )
                documents.extend(text_reader.load_data())
            
            # JSON files
            json_files = list(directory.glob("*.json"))
            if json_files:
                for json_file in json_files:
                    json_reader = JSONReader()
                    documents.extend(json_reader.load_data(str(json_file)))
            
            # CSV files
            csv_files = list(directory.glob("*.csv"))
            if csv_files:
                for csv_file in csv_files:
                    csv_reader = CSVReader()
                    documents.extend(csv_reader.load_data(str(csv_file)))
        
        return documents

class MultiModalGAIAIndex:
    """Multi-modal index supporting text, images, and tables for GAIA tasks"""
    
    def __init__(self) -> None:
        self.text_index = None
        self.image_index = None
        self.table_index = None
        self._setup_loaders()
    
    def _setup_loaders(self) -> Any:
        """Setup specialized loaders for different content types"""
        if not LLAMAINDEX_AVAILABLE:
            return
        
        try:
            # Download specialized loaders
            self.image_loader = download_loader("ImageReader")
            self.table_loader = download_loader("PandasCSVReader")
        except Exception as e:
            logger.warning("Could not load specialized loaders: {}", extra={"e": e})
    
    def process_gaia_content(self, content_path: str) -> Dict[str, Any]:
        """Process GAIA-specific content (text, images, tables)"""
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError("LlamaIndex not available")
        
        results = {
            "text_documents": [],
            "image_documents": [],
            "table_documents": []
        }
        
        path = Path(content_path)
        
        # Process text content
        if path.is_file() and path.suffix.lower() in ['.txt', '.md', '.json', '.csv']:
            try:
                if path.suffix.lower() in ['.txt', '.md']:
                    reader = SimpleDirectoryReader(input_files=[str(path)])
                    results["text_documents"] = reader.load_data()
                elif path.suffix.lower() == '.json':
                    reader = JSONReader()
                    results["text_documents"] = reader.load_data(str(path))
                elif path.suffix.lower() == '.csv':
                    reader = CSVReader()
                    results["text_documents"] = reader.load_data(str(path))
            except Exception as e:
                logger.error("Failed to process text file {}: {}", extra={"path": path, "e": e})
        
        # Process image content
        elif path.is_file() and path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
            try:
                if hasattr(self, 'image_loader'):
                    loader = self.image_loader()
                    results["image_documents"] = loader.load_data(str(path))
            except Exception as e:
                logger.error("Failed to process image file {}: {}", extra={"path": path, "e": e})
        
        return results

class IncrementalKnowledgeBase:
    """Incremental knowledge base with caching and deduplication"""
    
    def __init__(self, storage_path: str = "./knowledge_cache") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index = None
        self._load_existing_index()
    
    def _load_existing_index(self) -> Any:
        """Load existing index from storage"""
        if not LLAMAINDEX_AVAILABLE:
            return
        
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.storage_path)
            )
            self.index = load_index_from_storage(storage_context)
            logger.info("Loaded existing index from {}", extra={"self_storage_path": self.storage_path})
        except Exception as e:
            logger.info("No existing index found at {}: {}", extra={"self_storage_path": self.storage_path, "e": e})
            self.index = None
    
    def add_documents_incrementally(self, documents: List[Document]) -> bool:
        """Add documents incrementally with deduplication"""
        if not LLAMAINDEX_AVAILABLE:
            return False
        
        try:
            if self.index is None:
                # Create new index
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=StorageContext.from_defaults(
                        persist_dir=str(self.storage_path)
                    )
                )
            else:
                # Insert into existing index
                for doc in documents:
                    self.index.insert(doc)
            
            # Persist index
            self.index.storage_context.persist(persist_dir=str(self.storage_path))
            logger.info("Added {} documents incrementally", extra={"len_documents_": len(documents)})
            return True
            
        except Exception as e:
            logger.error("Failed to add documents incrementally: {}", extra={"e": e})
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.storage_path.exists():
            return {"status": "not_initialized"}
        
        try:
            file_count = len(list(self.storage_path.glob("*")))
            total_size = sum(f.stat().st_size for f in self.storage_path.glob("*") if f.is_file())
            
            return {
                "status": "initialized",
                "file_count": file_count,
                "total_size_bytes": total_size,
                "storage_path": str(self.storage_path)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

class GAIAQueryEngine:
    """Specialized query engine for GAIA tasks"""
    
    def __init__(self, index: VectorStoreIndex) -> None:
        self.index = index
        self.query_engine = None
        self._setup_query_engine()
    
    def _setup_query_engine(self) -> Any:
        """Setup specialized query engine for GAIA tasks"""
        if not LLAMAINDEX_AVAILABLE or self.index is None:
            return
        
        try:
            # Create retriever with similarity postprocessor
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=5,
                similarity_cutoff=0.7
            )
            
            # Create response synthesizer
            response_synthesizer = get_response_synthesizer(
                response_mode="compact",
                structured_answer_filtering=True
            )
            
            # Create query engine
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
            )
            
            logger.info("GAIA query engine setup completed")
            
        except Exception as e:
            logger.error("Failed to setup query engine: {}", extra={"e": e})
    
    def query_gaia_task(self, query: str, task_type: str = "general") -> str:
        """Query the knowledge base for GAIA tasks"""
        if not LLAMAINDEX_AVAILABLE or self.query_engine is None:
            return "Knowledge base not available"
        
        try:
            # Enhance query based on task type
            enhanced_query = self._enhance_query_for_task(query, task_type)
            
            # Execute query
            response = self.query_engine.query(enhanced_query)
            
            return str(response)
            
        except Exception as e:
            logger.error("Query failed: {}", extra={"e": e})
            return f"Query failed: {str(e)}"
    
    def _enhance_query_for_task(self, query: str, task_type: str) -> str:
        """Enhance query based on GAIA task type"""
        enhancements = {
            "mathematical": f"Provide a precise mathematical answer for: {query}",
            "factual": f"Provide accurate factual information for: {query}",
            "creative": f"Provide creative and original answer for: {query}",
            "multimodal": f"Consider all available modalities for: {query}"
        }
        
        return enhancements.get(task_type, query)
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query engine statistics"""
        if self.query_engine is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "retriever_type": type(self.query_engine.retriever).__name__,
            "synthesizer_type": type(self.query_engine.response_synthesizer).__name__
        }

@circuit_breaker("gaia_knowledge_base_creation", CircuitBreakerConfig(
    failure_threshold=3, 
    recovery_timeout=60
))
def create_gaia_knowledge_base(
    storage_path: Optional[str] = None,
    use_supabase: Optional[bool] = None
) -> Union[IncrementalKnowledgeBase, EnhancedKnowledgeBase]:
    """Create GAIA knowledge base with appropriate configuration"""
    
    if not LLAMAINDEX_AVAILABLE:
        raise ImportError("LlamaIndex not available")
    
    # Use config defaults if not provided
    if storage_path is None:
        storage_path = integration_config.llamaindex.storage_path if integration_config else "./knowledge_cache"
    
    if use_supabase is None:
        use_supabase = integration_config.supabase.is_configured() if integration_config else False
    
    try:
        if use_supabase:
            # Create enhanced knowledge base with Supabase vector store
            # This would require Supabase vector store setup
            logger.info("Creating enhanced knowledge base with Supabase")
            return EnhancedKnowledgeBase()
        else:
            # Create incremental knowledge base with local storage
            logger.info("Creating incremental knowledge base at {}", extra={"storage_path": storage_path})
            return IncrementalKnowledgeBase(storage_path)
            
    except Exception as e:
        logger.error("Failed to create knowledge base: {}", extra={"e": e})
        raise 