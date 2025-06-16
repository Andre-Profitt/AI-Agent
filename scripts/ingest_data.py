import os
import logging
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from src.database import setup_knowledge_base

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to load data from the '/data' directory and ingest it
    into the Supabase vector store.
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    if not data_dir.exists() or not any(data_dir.iterdir()):
        logger.warning(f"Data directory '{data_dir}' is empty or does not exist.")
        logger.warning("Please add documents to the /data folder and run again.")
        return

    logger.info(f"Loading documents from: {data_dir}")
    
    try:
        # Load all documents from the data directory
        documents = SimpleDirectoryReader(str(data_dir)).load_data()
        
        if not documents:
            logger.error("No documents were loaded. Please check the files in the /data directory.")
            return

        logger.info(f"Loaded {len(documents)} document(s). Starting ingestion process...")
        
        # Set up the knowledge base and ingest the documents
        setup_knowledge_base(documents)
        
        logger.info("Successfully ingested data into the knowledge base.")

    except Exception as e:
        logger.error(f"An error occurred during the ingestion process: {e}", exc_info=True)

if __name__ == "__main__":
    main() 