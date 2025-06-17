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

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing and embedding generation."""
    
    def __init__(self):
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
            logger.info(f"Processing file: {file_path}")
            
            # Read file content based on type
            content = self._read_file(file_path)
            if not content:
                return False
            
            # Check if already processed
            content_hash = self._compute_hash(content)
            if content_hash in self.processed_hashes:
                logger.info(f"File already processed: {file_path}")
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
            logger.info(f"Successfully ingested {len(documents)} chunks from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
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
            logger.error(f"Failed to read file {file_path}: {e}")
            return None
    
    def process_url(self, url: str) -> bool:
        """Process content from a URL."""
        try:
            logger.info(f"Processing URL: {url}")
            
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
                logger.info(f"URL already processed: {url}")
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
            logger.info(f"Successfully ingested {len(documents)} chunks from {url}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            return False

class KnowledgeIngestionService:
    """Simplified knowledge ingestion service for Hugging Face Spaces."""
    
    def __init__(self, watch_directories: List[str] = None, poll_urls: List[str] = None):
        self.processor = DocumentProcessor()
        self.watch_directories = watch_directories or []
        self.poll_urls = poll_urls or []
        self.is_running = False
        self._task = None
    
    def start(self):
        """Start the ingestion service."""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Starting knowledge ingestion service")
        
        # Process initial files
        for directory in self.watch_directories:
            self._process_directory(directory)
        
        # Start URL polling if configured
        if self.poll_urls:
            self._task = asyncio.create_task(self._poll_urls())
    
    def stop(self):
        """Stop the ingestion service."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self._task:
            self._task.cancel()
        logger.info("Stopped knowledge ingestion service")
    
    def _process_directory(self, directory: str):
        """Process all files in a directory."""
        try:
            path = Path(directory)
            if not path.exists():
                logger.warning(f"Directory does not exist: {directory}")
                return
            
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    self.processor.process_file(file_path)
        except Exception as e:
            logger.error(f"Error processing directory {directory}: {e}")
    
    async def _poll_urls(self):
        """Poll configured URLs periodically."""
        while self.is_running:
            for url in self.poll_urls:
                try:
                    self.processor.process_url(url)
                except Exception as e:
                    logger.error(f"Error polling URL {url}: {e}")
            await asyncio.sleep(3600)  # Poll every hour
    
    def add_watch_directory(self, directory: str):
        """Add a directory to watch."""
        if directory not in self.watch_directories:
            self.watch_directories.append(directory)
            if self.is_running:
                self._process_directory(directory)
    
    def add_poll_url(self, url: str):
        """Add a URL to poll."""
        if url not in self.poll_urls:
            self.poll_urls.append(url)

def run_ingestion_service(config: Dict[str, Any]):
    """Run the knowledge ingestion service with the given configuration."""
    service = KnowledgeIngestionService(
        watch_directories=config.get("watch_directories", []),
        poll_urls=config.get("poll_urls", [])
    )
    service.start()
    return service

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