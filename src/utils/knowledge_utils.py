from agent import query
from benchmarks.cot_performance import end
from benchmarks.cot_performance import start
from examples.parallel_execution_example import results
from fix_import_hierarchy import file_path

from src.agents.enhanced_fsm import source
from src.core.optimized_chain_of_thought import query_words
from src.core.optimized_chain_of_thought import words
from src.database.models import text
from src.meta_cognition import query_lower
from src.meta_cognition import score
from src.utils.knowledge_utils import best_position
from src.utils.knowledge_utils import doc_data
from src.utils.knowledge_utils import doc_id
from src.utils.knowledge_utils import doc_scores
from src.utils.knowledge_utils import doc_text
from src.utils.knowledge_utils import idf
from src.utils.knowledge_utils import matching_docs
from src.utils.knowledge_utils import snippet
from src.utils.knowledge_utils import sorted_docs
from src.utils.knowledge_utils import text_lower
from src.utils.knowledge_utils import tf
from src.utils.knowledge_utils import word

from src.tools.base_tool import Tool
# TODO: Fix undefined variables: Any, Dict, List, Path, best_position, cache_dir, context_words, datetime, defaultdict, doc_data, doc_id, doc_scores, doc_text, e, end, f, file_path, i, idf, json, logging, matching_docs, query, query_lower, query_words, qw, results, score, snippet, sorted_docs, source, start, text, text_lower, top_k, word, words, x
import tensorflow as tf

# TODO: Fix undefined variables: best_position, cache_dir, context_words, doc_data, doc_id, doc_scores, doc_text, e, end, f, file_path, i, idf, matching_docs, math, query, query_lower, query_words, qw, results, score, self, snippet, sorted_docs, source, start, text, text_lower, tf, top_k, word, words, x

"""
Knowledge utilities to avoid circular imports
"""

from typing import Any
from typing import List

import json
import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class LocalKnowledgeTool:
    """Local fallback knowledge tool when vector store is unavailable"""

    def __init__(self, cache_dir: str = "./knowledge_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.local_docs = {}
        self.inverted_index = defaultdict(set)  # word -> doc_ids
        self._load_local_docs()
        self._build_index()

    def _load_local_docs(self) -> Any:
        """Load documents from local cache"""
        try:
            for file_path in self.cache_dir.glob("*.json"):
                with open(file_path, 'r') as f:
                    doc_data = json.load(f)
                    self.local_docs[doc_data["id"]] = doc_data
            logger.info("Loaded {} local documents", extra={"len_self_local_docs_": len(self.local_docs)})
        except Exception as e:
            logger.warning("Failed to load local docs: {}", extra={"e": e})

    def _build_index(self) -> Any:
        """Build inverted index for better search"""
        for doc_id, doc_data in self.local_docs.items():
            text = doc_data.get("text", "").lower()
            words = set(text.split())

            for word in words:
                # Remove punctuation
                word = word.strip('.,!?;:"')
                if len(word) > 2:  # Skip very short words
                    self.inverted_index[word].add(doc_id)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Improved search using inverted index and TF-IDF-like scoring"""
        query_words = set(query.lower().split())
        doc_scores = defaultdict(float)

        # Score documents based on word matches
        for word in query_words:
            word = word.strip('.,!?;:"')
            matching_docs = self.inverted_index.get(word, set())

            # IDF-like scoring: rarer words get higher weight
            idf = math.log(len(self.local_docs) / (len(matching_docs) + 1))

            for doc_id in matching_docs:
                # TF scoring: count occurrences
                doc_text = self.local_docs[doc_id].get("text", "").lower()
                tf = doc_text.count(word)
                doc_scores[doc_id] += tf * idf

        # Sort by score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            doc_data = self.local_docs[doc_id]

            # Extract relevant snippet
            snippet = self._extract_snippet(doc_data.get("text", ""), query)

            results.append({
                "id": doc_id,
                "text": snippet,
                "source": doc_data.get("source", "local"),
                "similarity": min(score / 10.0, 1.0),  # Normalize score
                "full_text": doc_data.get("text", "")
            })

        return results

    def _extract_snippet(self, text: str, query: str, context_words: int = 50) -> str:
        """Extract relevant snippet around query terms"""
        text_lower = text.lower()
        query_lower = query.lower()

        # Find first occurrence of any query word
        words = text.split()
        query_words = query_lower.split()

        best_position = 0
        for i, word in enumerate(words):
            if any(qw in word.lower() for qw in query_words):
                best_position = i
                break

        # Extract context around position
        start = max(0, best_position - context_words // 2)
        end = min(len(words), best_position + context_words // 2)

        snippet = " ".join(words[start:end])

        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(words):
            snippet = snippet + "..."

        return snippet

    def add_document(self, text: str, source: str = "local") -> str:
        """Add document to local cache"""
        doc_id = f"local_{len(self.local_docs) + 1}"
        doc_data = {
            "id": doc_id,
            "text": text,
            "source": source,
            "created_at": datetime.now().isoformat()
        }

        self.local_docs[doc_id] = doc_data

        # Update inverted index
        text_lower = text.lower()
        words = set(text_lower.split())
        for word in words:
            word = word.strip('.,!?;:"')
            if len(word) > 2:
                self.inverted_index[word].add(doc_id)

        # Save to file
        file_path = self.cache_dir / f"{doc_id}.json"
        with open(file_path, 'w') as f:
            json.dump(doc_data, f, indent=2)

        return doc_id

def create_local_knowledge_tool() -> LocalKnowledgeTool:
    """Create local knowledge tool as fallback"""
    return LocalKnowledgeTool()