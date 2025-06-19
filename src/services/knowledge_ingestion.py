import os
import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import time
import requests
from bs4 import BeautifulSoup

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.file import PDFReader, DocxReader, UnstructuredReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import database for vector store access
from src.database import get_vector_store, get_embedding_model
from typing import Optional, Dict, Any, List, Union, Tuple
from src.shared.types.di_types import (
    ConfigurationService, DatabaseClient, CacheClient, LoggingService

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing and embedding generation."""
    
    def __init__(self) -> None:
        self.vector_store = get_vector_store()
        self.embedding_model = get_embedding_model()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.processed_hashes = set()  # Track processed documents
        
    def _compute_hash(self, content: str) -> str:
        """Compute SHA256 hash of content to detect duplicates."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def process_file(self, file_path: Path) -> bool:
        """Process a single file and add to vector store."""
        try:
            logger.info("Processing file: {}", extra={"file_path": file_path})
            
            # Read file content based on type
            content = self._read_file(file_path)
            if not content:
                return False
            
            # Check if already processed
            content_hash = self._compute_hash(content)
            if content_hash in self.processed_hashes:
                logger.info("File already processed: {}", extra={"file_path": file_path})
                return False
            
            # Create document
            metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix,
                "ingestion_time": datetime.now().isoformat(),
                "content_hash": content_hash
            }
            
            # Split into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create LlamaIndex documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    text=chunk,
                    metadata={
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                )
                documents.append(doc)
            
            # Add to vector store
            index = VectorStoreIndex.from_documents(
                documents,
                vector_store=self.vector_store,
                embed_model=self.embedding_model,
                show_progress=True
            )
            
            # Mark as processed
            self.processed_hashes.add(content_hash)
            logger.info("Successfully ingested {} chunks from {}", extra={"len_documents_": len(documents), "file_path": file_path})
            return True
            
        except Exception as e:
            logger.error("Error processing file {}: {}", extra={"file_path": file_path, "e": e})
            return False
    
    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read content from various file types."""
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == '.pdf':
                reader = PDFReader()
                docs = reader.load_data(file_path)
                return "\n".join([doc.text for doc in docs])
            
            elif suffix in ['.docx', '.doc']:
                reader = DocxReader()
                docs = reader.load_data(file_path)
                return "\n".join([doc.text for doc in docs])
            
            elif suffix in ['.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml']:
                return file_path.read_text(encoding='utf-8')
            
            else:
                # Try unstructured reader for other formats
                reader = UnstructuredReader()
                docs = reader.load_data(file_path)
                return "\n".join([doc.text for doc in docs])
                
        except Exception as e:
            logger.error("Failed to read file {}: {}", extra={"file_path": file_path, "e": e})
            return None
    
    def process_url(self, url: str) -> bool:
        """Process content from a URL."""
        try:
            logger.info("Processing URL: {}", extra={"url": url})
            
            # Fetch content
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text content
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = ' '.join(chunk for chunk in chunks if chunk)
            
            if not content:
                return False
            
            # Check if already processed
            content_hash = self._compute_hash(content)
            if content_hash in self.processed_hashes:
                logger.info("URL already processed: {}", extra={"url": url})
                return False
            
            # Create document
            metadata = {
                "source": url,
                "source_type": "web",
                "title": soup.title.string if soup.title else "Untitled",
                "ingestion_time": datetime.now().isoformat(),
                "content_hash": content_hash
            }
            
            # Split and process similar to files
            chunks = self.text_splitter.split_text(content)
            documents = []
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    text=chunk,
                    metadata={
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                )
                documents.append(doc)
            
            # Add to vector store
            index = VectorStoreIndex.from_documents(
                documents,
                vector_store=self.vector_store,
                embed_model=self.embedding_model
            )
            
            self.processed_hashes.add(content_hash)
            logger.info("Successfully ingested {} chunks from {}", extra={"len_documents_": len(documents), "url": url})
            return True
            
        except Exception as e:
            logger.error("Error processing URL {}: {}", extra={"url": url, "e": e})
            return False

class KnowledgeLifecycleManager:
    """Manages knowledge base lifecycle including reindexing and cache invalidation"""
    
    def __init__(self, db_client: Optional[DatabaseClient] = None, cache: Optional[CacheClient] = None) -> None:
        self.db_client = db_client
        self.cache = cache
        self.last_reindex = datetime.now()
        self.reindex_threshold = 10  # Reindex after N new documents
        
    async def update_knowledge_lifecycle(self, doc_id: str, doc_metadata: Dict[str, Any]) -> bool:
        """Update knowledge lifecycle tracking"""
        if not self.db_client:
            return
        
        try:
            # Update knowledge_lifecycle table
            await self.db_client.table("knowledge_lifecycle").upsert({
                "doc_id": doc_id,
                "last_updated": datetime.now().isoformat(),
                "metadata": doc_metadata,
                "status": "active"
            }).execute()
            
            logger.info("Updated knowledge lifecycle for document {}", extra={"doc_id": doc_id})
            
        except Exception as e:
            logger.error("Failed to update knowledge lifecycle: {}", extra={"e": e})
    
    async def trigger_reindex(self, force: bool = False) -> Any:
        """Trigger vector store reindexing"""
        try:
            # Check if reindex is needed
            if not force:
                doc_count = await self._get_recent_document_count()
                if doc_count < self.reindex_threshold:
                    logger.debug("Reindex not needed yet ({} recent docs)", extra={"doc_count": doc_count})
                    return
            
            logger.info("Triggering vector store reindex...")
            
            # This would trigger the actual reindexing
            # In a real implementation, this would rebuild the vector index
            await self._perform_reindex()
            
            self.last_reindex = datetime.now()
            logger.info("Vector store reindex completed")
            
        except Exception as e:
            logger.error("Failed to trigger reindex: {}", extra={"e": e})
    
    async def invalidate_cache(self, pattern: str = "knowledge_base:*") -> Any:
        """Invalidate knowledge base cache"""
        if not self.cache:
            return
        
        try:
            self.cache.invalidate(pattern)
            logger.info("Invalidated cache pattern: {}", extra={"pattern": pattern})
            
        except Exception as e:
            logger.error("Failed to invalidate cache: {}", extra={"e": e})
    
    async def _get_recent_document_count(self) -> int:
        """Get count of recently added documents"""
        if not self.db_client:
            return 0
        
        try:
            # Query recent documents from knowledge_lifecycle table
            result = await self.db_client.table("knowledge_lifecycle").select(
                "doc_id"
            ).gte("last_updated", self.last_reindex.isoformat()).execute()
            
            return len(result.data) if result.data else 0
            
        except Exception as e:
            logger.error("Failed to get recent document count: {}", extra={"e": e})
            return 0
    
    async def _perform_reindex(self) -> Any:
        """Perform the actual reindexing operation"""
        # This is a placeholder for the actual reindexing logic
        # In a real implementation, this would:
        # 1. Rebuild the vector index
        # 2. Update metadata
        # 3. Optimize storage
        logger.info("Performing vector store reindex...")
        await asyncio.sleep(1)  # Simulate reindexing time

class KnowledgeIngestionService:
    """Enhanced knowledge ingestion service with lifecycle management."""
    
    def __init__(self, ,

    
            watch_directories: List[str] = None        poll_urls: List[str] = None        db_client: Optional[DatabaseClient] = None        cache: Optional[CacheClient] = None) -> None:
        self.processor = DocumentProcessor()
        self.lifecycle_manager = KnowledgeLifecycleManager(db_client, cache)
        self.watch_directories = watch_directories or []
        self.poll_urls = poll_urls or []
        self.running = False
        self.db_client = db_client
        self.cache = cache
        
    async def ingest_document(self, doc_path: str) -> str:
        """Enhanced document ingestion with lifecycle management"""
        try:
            file_path = Path(doc_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Document not found: {doc_path}")
            
            # Process document
            success = self.processor.process_file(file_path)
            if not success:
                raise Exception(f"Failed to process document: {doc_path}")
            
            # Generate document ID
            doc_id = f"doc_{int(time.time())}_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}"
            
            # Update lifecycle
            doc_metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix,
                "ingestion_time": datetime.now().isoformat()
            }
            
            await self.lifecycle_manager.update_knowledge_lifecycle(doc_id, doc_metadata)
            
            # Trigger reindex if needed
            await self.lifecycle_manager.trigger_reindex()
            
            # Invalidate cache
            await self.lifecycle_manager.invalidate_cache()
            
            logger.info("Successfully ingested document {} from {}", extra={"doc_id": doc_id, "doc_path": doc_path})
            return doc_id
            
        except Exception as e:
            logger.error("Failed to ingest document {}: {}", extra={"doc_path": doc_path, "e": e})
            raise
    
    async def ingest_url(self, url: str) -> str:
        """Enhanced URL ingestion with lifecycle management"""
        try:
            # Process URL
            success = self.processor.process_url(url)
            if not success:
                raise Exception(f"Failed to process URL: {url}")
            
            # Generate document ID
            doc_id = f"url_{int(time.time())}_{hashlib.md5(url.encode()).hexdigest()[:8]}"
            
            # Update lifecycle
            doc_metadata = {
                "source": url,
                "source_type": "web",
                "ingestion_time": datetime.now().isoformat()
            }
            
            await self.lifecycle_manager.update_knowledge_lifecycle(doc_id, doc_metadata)
            
            # Trigger reindex if needed
            await self.lifecycle_manager.trigger_reindex()
            
            # Invalidate cache
            await self.lifecycle_manager.invalidate_cache()
            
            logger.info("Successfully ingested URL {} from {}", extra={"doc_id": doc_id, "url": url})
            return doc_id
            
        except Exception as e:
            logger.error("Failed to ingest URL {}: {}", extra={"url": url, "e": e})
            raise
    
    def start(self) -> None:
        """Start the ingestion service"""
        if self.running:
            logger.warning("Ingestion service already running")
            return
        
        self.running = True
        logger.info("Starting knowledge ingestion service...")
        
        # Process existing directories
        for directory in self.watch_directories:
            self._process_directory(directory)
        
        # Start URL polling in background
        if self.poll_urls:
            asyncio.create_task(self._poll_urls())
    
    def stop(self) -> None:
        """Stop the ingestion service"""
        self.running = False
        logger.info("Stopped knowledge ingestion service")
    
    def _process_directory(self, directory: str) -> Any:
        """Process all files in a directory"""
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                logger.warning("Directory not found: {}", extra={"directory": directory})
                return
            
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    self.processor.process_file(file_path)
                    
        except Exception as e:
            logger.error("Error processing directory {}: {}", extra={"directory": directory, "e": e})
    
    async def _poll_urls(self) -> Any:
        """Poll URLs for updates"""
        while self.running:
            try:
                for url in self.poll_urls:
                    self.processor.process_url(url)
                
                # Wait before next poll
                await asyncio.sleep(3600)  # Poll every hour
                
            except Exception as e:
                logger.error("Error polling URLs: {}", extra={"e": e})
                await asyncio.sleep(60)  # Wait before retry
    
    def add_watch_directory(self, directory: str) -> Any:
        """Add a directory to watch for new documents"""
        self.watch_directories.append(directory)
        logger.info("Added watch directory: {}", extra={"directory": directory})
    
    def add_poll_url(self, url: str) -> Any:
        """Add a URL to poll for updates"""
        self.poll_urls.append(url)
        logger.info("Added poll URL: {}", extra={"url": url})

def run_ingestion_service(config: Dict[str, Any]) -> Any:
    """Run the knowledge ingestion service with configuration"""
    try:
        # Extract configuration
        watch_dirs = config.get("watch_directories", [])
        poll_urls = config.get("poll_urls", [])
        db_client = config.get("db_client")
        cache = config.get("cache")
        
        # Create service
        service = KnowledgeIngestionService(
            watch_directories=watch_dirs,
            poll_urls=poll_urls,
            db_client=db_client,
            cache=cache
        )
        
        # Start service
        service.start()
        
        return service
        
    except Exception as e:
        logger.error("Failed to run ingestion service: {}", extra={"e": e})
        raise

if __name__ == "__main__":
    # Example configuration
    config = {
        "watch_directories": [
            "./documents",
            "./knowledge_base",
            os.path.expanduser("~/Documents/AI_Agent_Knowledge")
        ],
        "poll_urls": [
            # Add URLs to periodically check for updates
        ]
    }
    
    logging.basicConfig(level=logging.INFO)
    run_ingestion_service(config) 