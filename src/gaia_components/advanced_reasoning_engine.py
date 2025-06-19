"""
Advanced Reasoning Engine for GAIA-enhanced FSMReActAgent
Implements sophisticated reasoning capabilities with vector store integration
"""

import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
from pathlib import Path

# Vector store imports
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import OpenAIEmbeddings
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    Chroma = None
    OpenAIEmbeddings = None

# LangChain imports
try:
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_groq import ChatGroq
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    HumanMessage = None
    AIMessage = None
    ChatGroq = None

logger = logging.getLogger(__name__)

class ReasoningType(str, Enum):
    """Types of reasoning approaches"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CRITICAL = "critical"
    CREATIVE = "creative"

class ReasoningStep:
    """Represents a single reasoning step"""
    
    def __init__(self, step_type: str, content: str, confidence: float = 0.0):
        self.step_type = step_type
        self.content = content
        self.confidence = confidence
        self.timestamp = datetime.now()
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_type": self.step_type,
            "content": self.content,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningStep':
        step = cls(data["step_type"], data["content"], data["confidence"])
        step.timestamp = datetime.fromisoformat(data["timestamp"])
        step.metadata = data.get("metadata", {})
        return step

class ReasoningPath:
    """Represents a complete reasoning path"""
    
    def __init__(self, query: str, reasoning_type: ReasoningType):
        self.query = query
        self.reasoning_type = reasoning_type
        self.steps: List[ReasoningStep] = []
        self.confidence = 0.0
        self.completion_time = None
        self.metadata = {}
    
    def add_step(self, step: ReasoningStep):
        """Add a reasoning step to the path"""
        self.steps.append(step)
        self._update_confidence()
    
    def _update_confidence(self):
        """Update overall confidence based on step confidences"""
        if self.steps:
            confidences = [step.confidence for step in self.steps]
            self.confidence = sum(confidences) / len(confidences)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "reasoning_type": self.reasoning_type.value,
            "steps": [step.to_dict() for step in self.steps],
            "confidence": self.confidence,
            "completion_time": self.completion_time.isoformat() if self.completion_time else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningPath':
        path = cls(data["query"], ReasoningType(data["reasoning_type"]))
        path.steps = [ReasoningStep.from_dict(step_data) for step_data in data["steps"]]
        path.confidence = data["confidence"]
        if data["completion_time"]:
            path.completion_time = datetime.fromisoformat(data["completion_time"])
        path.metadata = data.get("metadata", {})
        return path

class VectorStore:
    """Vector store for storing and retrieving reasoning patterns"""
    
    def __init__(self, persist_directory: str = "./vector_store"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        self.vectorstore = None
        self.embeddings = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize the vector store with embeddings"""
        if not VECTOR_STORE_AVAILABLE:
            logger.warning("Vector store dependencies not available. Using in-memory storage.")
            return
        
        try:
            # Use OpenAI embeddings (fallback to a simple embedding if not available)
            api_key = self._get_openai_key()
            if api_key:
                self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            else:
                # Fallback to a simple embedding function
                self.embeddings = self._create_simple_embeddings()
            
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self.vectorstore = None
    
    def _get_openai_key(self) -> Optional[str]:
        """Get OpenAI API key from environment"""
        import os
        return os.getenv("OPENAI_API_KEY")
    
    def _create_simple_embeddings(self):
        """Create a simple embedding function as fallback"""
        class SimpleEmbeddings:
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                # Simple hash-based embedding
                embeddings = []
                for text in texts:
                    # Create a simple hash-based embedding
                    hash_val = hash(text) % 1000
                    embedding = [float(hash_val + i) / 1000.0 for i in range(384)]
                    embeddings.append(embedding)
                return embeddings
            
            def embed_query(self, text: str) -> List[float]:
                hash_val = hash(text) % 1000
                return [float(hash_val + i) / 1000.0 for i in range(384)]
        
        return SimpleEmbeddings()
    
    def store_reasoning_path(self, path: ReasoningPath):
        """Store a reasoning path in the vector store"""
        if not self.vectorstore:
            logger.warning("Vector store not available")
            return
        
        try:
            # Create document from reasoning path
            content = f"Query: {path.query}\nReasoning Type: {path.reasoning_type.value}\n"
            content += "\n".join([f"Step {i+1}: {step.content}" for i, step in enumerate(path.steps)])
            
            metadata = {
                "query": path.query,
                "reasoning_type": path.reasoning_type.value,
                "confidence": path.confidence,
                "step_count": len(path.steps),
                "timestamp": datetime.now().isoformat()
            }
            
            self.vectorstore.add_texts(
                texts=[content],
                metadatas=[metadata]
            )
            logger.info(f"Stored reasoning path with {len(path.steps)} steps")
        except Exception as e:
            logger.error(f"Failed to store reasoning path: {e}")
    
    def retrieve_similar_paths(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar reasoning paths for a query"""
        if not self.vectorstore:
            logger.warning("Vector store not available")
            return []
        
        try:
            results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in results
            ]
        except Exception as e:
            logger.error(f"Failed to retrieve similar paths: {e}")
            return []

class AdvancedReasoningEngine:
    """Advanced reasoning engine with multiple reasoning strategies"""
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        self.vector_store = VectorStore()
        self.llm = self._initialize_llm()
        self.reasoning_strategies = self._initialize_strategies()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_llm(self):
        """Initialize the language model"""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available. Using mock LLM.")
            return None
        
        try:
            api_key = self._get_groq_key()
            if api_key:
                return ChatGroq(
                    groq_api_key=api_key,
                    model_name=self.model_name
                )
            else:
                logger.warning("Groq API key not found. Using mock LLM.")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return None
    
    def _get_groq_key(self) -> Optional[str]:
        """Get Groq API key from environment"""
        import os
        return os.getenv("GROQ_API_KEY")
    
    def _initialize_strategies(self) -> Dict[str, str]:
        """Initialize reasoning strategies"""
        return {
            ReasoningType.DEDUCTIVE: """
            Use deductive reasoning to reach a logical conclusion from given premises.
            Start with general principles and apply them to specific cases.
            Ensure each step follows logically from the previous one.
            """,
            ReasoningType.INDUCTIVE: """
            Use inductive reasoning to form general conclusions from specific observations.
            Look for patterns and trends in the data.
            Consider multiple examples to strengthen the conclusion.
            """,
            ReasoningType.ABDUCTIVE: """
            Use abductive reasoning to form the best explanation for given observations.
            Consider multiple hypotheses and choose the most plausible one.
            Focus on explanatory power and simplicity.
            """,
            ReasoningType.ANALOGICAL: """
            Use analogical reasoning to solve problems by drawing parallels.
            Identify similarities between the current problem and known cases.
            Adapt solutions from similar situations.
            """,
            ReasoningType.CRITICAL: """
            Use critical reasoning to evaluate arguments and evidence.
            Question assumptions and identify logical fallacies.
            Consider alternative viewpoints and counterarguments.
            """,
            ReasoningType.CREATIVE: """
            Use creative reasoning to generate novel solutions and insights.
            Think outside conventional frameworks.
            Combine ideas in unexpected ways.
            """
        }
    
    def analyze_query(self, query: str) -> ReasoningType:
        """Analyze query to determine the best reasoning approach"""
        query_lower = query.lower()
        
        # Keyword-based analysis
        if any(word in query_lower for word in ["prove", "deduce", "therefore", "conclude"]):
            return ReasoningType.DEDUCTIVE
        elif any(word in query_lower for word in ["pattern", "trend", "usually", "typically"]):
            return ReasoningType.INDUCTIVE
        elif any(word in query_lower for word in ["explain", "why", "cause", "reason"]):
            return ReasoningType.ABDUCTIVE
        elif any(word in query_lower for word in ["similar", "like", "compare", "analogy"]):
            return ReasoningType.ANALOGICAL
        elif any(word in query_lower for word in ["evaluate", "criticize", "assess", "analyze"]):
            return ReasoningType.CRITICAL
        elif any(word in query_lower for word in ["create", "design", "invent", "innovate"]):
            return ReasoningType.CREATIVE
        
        # Default to deductive reasoning
        return ReasoningType.DEDUCTIVE
    
    def generate_reasoning_path(self, query: str, context: Dict[str, Any] = None) -> ReasoningPath:
        """Generate a reasoning path for the given query"""
        reasoning_type = self.analyze_query(query)
        path = ReasoningPath(query, reasoning_type)
        
        # Retrieve similar reasoning patterns
        similar_paths = self.vector_store.retrieve_similar_paths(query, k=3)
        
        # Generate reasoning steps
        steps = self._generate_reasoning_steps(query, reasoning_type, similar_paths, context)
        
        for step in steps:
            path.add_step(step)
        
        path.completion_time = datetime.now()
        
        # Store the reasoning path for future reference
        self.vector_store.store_reasoning_path(path)
        
        return path
    
    def _generate_reasoning_steps(self, query: str, reasoning_type: ReasoningType, 
                                similar_paths: List[Dict[str, Any]], 
                                context: Dict[str, Any] = None) -> List[ReasoningStep]:
        """Generate reasoning steps using the LLM"""
        if not self.llm:
            # Fallback to rule-based reasoning
            return self._generate_rule_based_steps(query, reasoning_type)
        
        try:
            # Create prompt for reasoning
            strategy = self.reasoning_strategies[reasoning_type]
            
            # Include similar paths as examples
            examples = ""
            if similar_paths:
                examples = "\n\nSimilar reasoning patterns:\n"
                for i, path_data in enumerate(similar_paths[:2]):
                    examples += f"Example {i+1}:\n{path_data['content'][:200]}...\n"
            
            prompt = f"""
            {strategy}
            
            Query: {query}
            
            Context: {context or 'No additional context provided'}
            
            {examples}
            
            Generate a step-by-step reasoning path to answer this query.
            Each step should be clear, logical, and build upon the previous steps.
            
            Format your response as a JSON array of steps, where each step has:
            - step_type: The type of reasoning used in this step
            - content: The reasoning content
            - confidence: A confidence score between 0 and 1
            
            Response:
            """
            
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            # Parse the response
            try:
                steps_data = json.loads(response.content)
                steps = []
                for step_data in steps_data:
                    step = ReasoningStep(
                        step_type=step_data.get("step_type", "general"),
                        content=step_data.get("content", ""),
                        confidence=step_data.get("confidence", 0.5)
                    )
                    steps.append(step)
                return steps
            except json.JSONDecodeError:
                # Fallback to parsing text response
                return self._parse_text_response(response.content, reasoning_type)
                
        except Exception as e:
            self.logger.error(f"Failed to generate reasoning steps with LLM: {e}")
            return self._generate_rule_based_steps(query, reasoning_type)
    
    def _generate_rule_based_steps(self, query: str, reasoning_type: ReasoningType) -> List[ReasoningStep]:
        """Generate reasoning steps using rule-based approach"""
        steps = []
        
        # Initial analysis step
        steps.append(ReasoningStep(
            step_type="analysis",
            content=f"Analyzing query: {query}",
            confidence=0.7
        ))
        
        # Reasoning type specific step
        steps.append(ReasoningStep(
            step_type=reasoning_type.value,
            content=f"Applying {reasoning_type.value} reasoning to the query",
            confidence=0.6
        ))
        
        # Conclusion step
        steps.append(ReasoningStep(
            step_type="conclusion",
            content="Formulating conclusion based on reasoning",
            confidence=0.5
        ))
        
        return steps
    
    def _parse_text_response(self, response: str, reasoning_type: ReasoningType) -> List[ReasoningStep]:
        """Parse text response into reasoning steps"""
        lines = response.strip().split('\n')
        steps = []
        
        for i, line in enumerate(lines):
            if line.strip():
                steps.append(ReasoningStep(
                    step_type=f"step_{i+1}",
                    content=line.strip(),
                    confidence=0.6
                ))
        
        if not steps:
            steps.append(ReasoningStep(
                step_type="fallback",
                content="Generated reasoning path",
                confidence=0.5
            ))
        
        return steps
    
    def evaluate_reasoning_quality(self, path: ReasoningPath) -> Dict[str, Any]:
        """Evaluate the quality of a reasoning path"""
        evaluation = {
            "overall_quality": 0.0,
            "step_count": len(path.steps),
            "average_confidence": 0.0,
            "reasoning_type_appropriateness": 0.0,
            "logical_flow": 0.0,
            "completeness": 0.0
        }
        
        if not path.steps:
            return evaluation
        
        # Calculate average confidence
        confidences = [step.confidence for step in path.steps]
        evaluation["average_confidence"] = sum(confidences) / len(confidences)
        
        # Evaluate logical flow (check if steps build upon each other)
        logical_flow_score = 0.0
        for i in range(1, len(path.steps)):
            if path.steps[i].content and path.steps[i-1].content:
                logical_flow_score += 0.2
        evaluation["logical_flow"] = min(logical_flow_score, 1.0)
        
        # Evaluate completeness (check if query is addressed)
        query_terms = set(path.query.lower().split())
        content_terms = set()
        for step in path.steps:
            content_terms.update(step.content.lower().split())
        
        overlap = len(query_terms.intersection(content_terms))
        evaluation["completeness"] = min(overlap / max(len(query_terms), 1), 1.0)
        
        # Overall quality score
        evaluation["overall_quality"] = (
            evaluation["average_confidence"] * 0.3 +
            evaluation["logical_flow"] * 0.3 +
            evaluation["completeness"] * 0.4
        )
        
        return evaluation
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get statistics about reasoning patterns"""
        # This would typically query the vector store
        # For now, return mock statistics
        return {
            "total_paths": 0,
            "reasoning_type_distribution": {},
            "average_confidence": 0.0,
            "most_common_patterns": []
        }