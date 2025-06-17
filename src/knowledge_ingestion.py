import os
import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
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


class KnowledgeIngestionHandler(FileSystemEventHandler):
    """Handles file system events for automatic ingestion."""
    
    def __init__(self, processor: DocumentProcessor):
        self.processor = processor
        self.pending_files = {}  # Track files being written
        
    def on_created(self, event):
        if not event.is_directory:
            self._handle_file(event.src_path)
    
    def on_modified(self, event):
        if not event.is_directory:
            # Debounce modifications to avoid processing incomplete writes
            self.pending_files[event.src_path] = time.time()
    
    def _handle_file(self, file_path: str):
        """Handle a new or modified file."""
        path = Path(file_path)
        
        # Check if it's a supported file type
        supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml'}
        if path.suffix.lower() not in supported_extensions:
            return
        
        # Process the file
        self.processor.process_file(path)
    
    def process_pending(self):
        """Process files that have been stable for 2 seconds."""
        current_time = time.time()
        stable_files = []
        
        for file_path, mod_time in list(self.pending_files.items()):
            if current_time - mod_time > 2:  # File stable for 2 seconds
                stable_files.append(file_path)
                del self.pending_files[file_path]
        
        for file_path in stable_files:
            self._handle_file(file_path)


class KnowledgeIngestionService:
    """Main service for automated knowledge base ingestion."""
    
    def __init__(self, watch_directories: List[str], poll_urls: List[str] = None):
        self.watch_directories = [Path(d) for d in watch_directories]
        self.poll_urls = poll_urls or []
        self.processor = DocumentProcessor()
        self.observer = Observer()
        self.handler = KnowledgeIngestionHandler(self.processor)
        
    def start(self):
        """Start the ingestion service."""
        logger.info("Starting Knowledge Ingestion Service")
        
        # Initial scan of directories
        self._initial_scan()
        
        # Set up file system monitoring
        for directory in self.watch_directories:
            if directory.exists():
                self.observer.schedule(self.handler, str(directory), recursive=True)
                logger.info(f"Watching directory: {directory}")
            else:
                logger.warning(f"Directory does not exist: {directory}")
        
        self.observer.start()
        
        # Start async event loop for URL polling
        asyncio.create_task(self._poll_urls())
        
        logger.info("Knowledge Ingestion Service started")
    
    def stop(self):
        """Stop the ingestion service."""
        logger.info("Stopping Knowledge Ingestion Service")
        self.observer.stop()
        self.observer.join()
    
    def _initial_scan(self):
        """Perform initial scan of watched directories."""
        logger.info("Performing initial scan of directories")
        
        for directory in self.watch_directories:
            if not directory.exists():
                continue
                
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    # Check supported extensions
                    supported = {'.pdf', '.docx', '.doc', '.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml'}
                    if file_path.suffix.lower() in supported:
                        self.processor.process_file(file_path)
    
    async def _poll_urls(self):
        """Periodically poll URLs for new content."""
        while True:
            for url in self.poll_urls:
                try:
                    self.processor.process_url(url)
                except Exception as e:
                    logger.error(f"Error polling URL {url}: {e}")
            
            # Wait 1 hour before next poll
            await asyncio.sleep(3600)
    
    def add_watch_directory(self, directory: str):
        """Add a new directory to watch."""
        path = Path(directory)
        if path.exists() and path not in self.watch_directories:
            self.watch_directories.append(path)
            self.observer.schedule(self.handler, str(path), recursive=True)
            logger.info(f"Added watch directory: {path}")
    
    def add_poll_url(self, url: str):
        """Add a new URL to poll."""
        if url not in self.poll_urls:
            self.poll_urls.append(url)
            logger.info(f"Added poll URL: {url}")
            # Process immediately
            self.processor.process_url(url)


# Standalone runner
def run_ingestion_service(config: Dict[str, Any]):
    """Run the ingestion service with the given configuration."""
    service = KnowledgeIngestionService(
        watch_directories=config.get("watch_directories", ["./documents"]),
        poll_urls=config.get("poll_urls", [])
    )
    
    try:
        service.start()
        
        # Run pending file processor in a loop
        while True:
            service.handler.process_pending()
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        service.stop()


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