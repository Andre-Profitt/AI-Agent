"""
Extended Database Schema for Persistent Learning
Adds tables for tool reliability, clarification patterns, and knowledge lifecycle
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict

from supabase import create_client, Client
from config import get_api_key


@dataclass
class ToolReliabilityMetric:
    """Tool performance metrics"""
    tool_name: str
    success_count: int = 0
    failure_count: int = 0
    total_calls: int = 0
    average_latency_ms: float = 0.0
    last_used_at: Optional[str] = None
    last_error: Optional[str] = None
    error_patterns: List[str] = None
    
    def __post_init__(self):
        if self.error_patterns is None:
            self.error_patterns = []
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_calls == 0:
            return 0.0
        return self.success_count / self.total_calls
    
    @property
    def reliability_score(self) -> float:
        """Composite reliability score (0-1)"""
        if self.total_calls < 5:  # Not enough data
            return 0.5
        
        # Weighted score: 70% success rate, 30% latency factor
        latency_factor = min(1.0, 1000 / (self.average_latency_ms + 1))
        return 0.7 * self.success_rate + 0.3 * latency_factor


@dataclass
class ClarificationPattern:
    """Pattern of clarification requests"""
    id: str
    original_query: str
    query_embedding: Optional[List[float]]  # For similarity search
    clarification_question: str
    user_response: str
    query_category: str
    frequency: int = 1
    created_at: str = None
    last_seen_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.last_seen_at is None:
            self.last_seen_at = datetime.now().isoformat()


@dataclass
class PlanCorrection:
    """User corrections to agent plans"""
    id: str
    query: str
    original_plan: Dict[str, Any]
    corrected_plan: Dict[str, Any]
    correction_type: str  # "steps_added", "steps_removed", "parameters_changed", etc.
    user_feedback: Optional[str]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class KnowledgeLifecycle:
    """Knowledge document lifecycle management"""
    document_id: str
    source_url: Optional[str]
    document_type: str  # "news", "documentation", "research", etc.
    content_hash: str
    ingested_at: str
    last_validated_at: str
    expires_at: str
    validation_status: str  # "valid", "stale", "expired", "source_unavailable"
    update_frequency_days: int
    importance_score: float = 0.5
    
    @property
    def is_expired(self) -> bool:
        """Check if document has expired"""
        return datetime.fromisoformat(self.expires_at) < datetime.now()
    
    @property
    def needs_validation(self) -> bool:
        """Check if document needs re-validation"""
        next_validation = datetime.fromisoformat(self.last_validated_at) + \
                         timedelta(days=self.update_frequency_days)
        return next_validation < datetime.now()


class ExtendedDatabase:
    """Extended database operations for learning and lifecycle management"""
    
    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        """Initialize extended database client"""
        self.url = supabase_url or get_api_key("SUPABASE_URL")
        self.key = supabase_key or get_api_key("SUPABASE_SERVICE_KEY")
        
        if self.url and self.key:
            self.client: Client = create_client(self.url, self.key)
        else:
            self.client = None
            print("Warning: Supabase credentials not found. Database features disabled.")
    
    # Tool Reliability Methods
    
    def update_tool_metric(
        self,
        tool_name: str,
        success: bool,
        latency_ms: float,
        error_message: Optional[str] = None
    ) -> bool:
        """Update tool reliability metrics after a call"""
        if not self.client:
            return False
        
        try:
            # Get existing metric or create new
            result = self.client.table("tool_reliability_metrics").select("*").eq(
                "tool_name", tool_name
            ).single().execute()
            
            if result.data:
                metric = ToolReliabilityMetric(**result.data)
            else:
                metric = ToolReliabilityMetric(tool_name=tool_name)
            
            # Update metrics
            metric.total_calls += 1
            if success:
                metric.success_count += 1
            else:
                metric.failure_count += 1
                if error_message:
                    metric.last_error = error_message
                    # Track error patterns
                    if error_message not in metric.error_patterns:
                        metric.error_patterns.append(error_message)
                        if len(metric.error_patterns) > 10:  # Keep last 10
                            metric.error_patterns.pop(0)
            
            # Update latency (moving average)
            if metric.average_latency_ms == 0:
                metric.average_latency_ms = latency_ms
            else:
                # Exponential moving average
                alpha = 0.1
                metric.average_latency_ms = (
                    alpha * latency_ms + (1 - alpha) * metric.average_latency_ms
                )
            
            metric.last_used_at = datetime.now().isoformat()
            
            # Upsert metric
            data = asdict(metric)
            self.client.table("tool_reliability_metrics").upsert(data).execute()
            
            return True
            
        except Exception as e:
            print(f"Error updating tool metric: {e}")
            return False
    
    def get_tool_metrics(self, tool_names: Optional[List[str]] = None) -> List[ToolReliabilityMetric]:
        """Get reliability metrics for tools"""
        if not self.client:
            return []
        
        try:
            query = self.client.table("tool_reliability_metrics").select("*")
            
            if tool_names:
                query = query.in_("tool_name", tool_names)
            
            result = query.execute()
            
            return [ToolReliabilityMetric(**row) for row in result.data]
            
        except Exception as e:
            print(f"Error getting tool metrics: {e}")
            return []
    
    # Clarification Pattern Methods
    
    def add_clarification_pattern(
        self,
        original_query: str,
        clarification_question: str,
        user_response: str,
        query_category: str,
        query_embedding: Optional[List[float]] = None
    ) -> bool:
        """Add or update a clarification pattern"""
        if not self.client:
            return False
        
        try:
            import hashlib
            # Create pattern ID from hash of query + question
            pattern_id = hashlib.md5(
                f"{original_query}:{clarification_question}".encode()
            ).hexdigest()
            
            # Check if pattern exists
            result = self.client.table("clarification_patterns").select("*").eq(
                "id", pattern_id
            ).single().execute()
            
            if result.data:
                # Update frequency and last seen
                pattern = ClarificationPattern(**result.data)
                pattern.frequency += 1
                pattern.last_seen_at = datetime.now().isoformat()
                data = asdict(pattern)
            else:
                # Create new pattern
                pattern = ClarificationPattern(
                    id=pattern_id,
                    original_query=original_query,
                    query_embedding=query_embedding,
                    clarification_question=clarification_question,
                    user_response=user_response,
                    query_category=query_category
                )
                data = asdict(pattern)
            
            self.client.table("clarification_patterns").upsert(data).execute()
            return True
            
        except Exception as e:
            print(f"Error adding clarification pattern: {e}")
            return False
    
    def find_similar_clarifications(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        threshold: float = 0.7
    ) -> List[ClarificationPattern]:
        """Find similar clarification patterns"""
        if not self.client:
            return []
        
        try:
            if query_embedding:
                # Use vector similarity search if embeddings available
                # This would use pgvector extension in Supabase
                # For now, fallback to text search
                pass
            
            # Simple text search fallback
            result = self.client.table("clarification_patterns").select("*").execute()
            
            patterns = []
            query_words = set(query.lower().split())
            
            for row in result.data:
                pattern = ClarificationPattern(**row)
                pattern_words = set(pattern.original_query.lower().split())
                
                # Jaccard similarity
                similarity = len(query_words & pattern_words) / len(query_words | pattern_words)
                
                if similarity >= threshold:
                    patterns.append(pattern)
            
            # Sort by frequency and recency
            patterns.sort(
                key=lambda p: (p.frequency, p.last_seen_at),
                reverse=True
            )
            
            return patterns[:5]  # Top 5
            
        except Exception as e:
            print(f"Error finding similar clarifications: {e}")
            return []
    
    # Plan Correction Methods
    
    def add_plan_correction(
        self,
        query: str,
        original_plan: Dict[str, Any],
        corrected_plan: Dict[str, Any],
        user_feedback: Optional[str] = None
    ) -> bool:
        """Record a user correction to an agent plan"""
        if not self.client:
            return False
        
        try:
            import uuid
            
            # Determine correction type
            correction_type = self._analyze_correction_type(original_plan, corrected_plan)
            
            correction = PlanCorrection(
                id=str(uuid.uuid4()),
                query=query,
                original_plan=original_plan,
                corrected_plan=corrected_plan,
                correction_type=correction_type,
                user_feedback=user_feedback
            )
            
            data = asdict(correction)
            # Convert dicts to JSON strings for storage
            data["original_plan"] = json.dumps(data["original_plan"])
            data["corrected_plan"] = json.dumps(data["corrected_plan"])
            
            self.client.table("plan_corrections").insert(data).execute()
            return True
            
        except Exception as e:
            print(f"Error adding plan correction: {e}")
            return False
    
    def _analyze_correction_type(
        self,
        original: Dict[str, Any],
        corrected: Dict[str, Any]
    ) -> str:
        """Analyze what type of correction was made"""
        original_steps = original.get("steps", [])
        corrected_steps = corrected.get("steps", [])
        
        if len(corrected_steps) > len(original_steps):
            return "steps_added"
        elif len(corrected_steps) < len(original_steps):
            return "steps_removed"
        elif original_steps != corrected_steps:
            return "steps_modified"
        else:
            return "parameters_changed"
    
    # Knowledge Lifecycle Methods
    
    def add_knowledge_document(
        self,
        document_id: str,
        source_url: Optional[str],
        document_type: str,
        content: str,
        ttl_days: Optional[int] = None
    ) -> bool:
        """Add a document with lifecycle metadata"""
        if not self.client:
            return False
        
        try:
            import hashlib
            
            # Determine TTL based on document type
            if ttl_days is None:
                ttl_map = {
                    "news": 30,
                    "blog": 90,
                    "documentation": 365,
                    "research": 180,
                    "reference": 730
                }
                ttl_days = ttl_map.get(document_type, 90)
            
            # Calculate content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            now = datetime.now()
            lifecycle = KnowledgeLifecycle(
                document_id=document_id,
                source_url=source_url,
                document_type=document_type,
                content_hash=content_hash,
                ingested_at=now.isoformat(),
                last_validated_at=now.isoformat(),
                expires_at=(now + timedelta(days=ttl_days)).isoformat(),
                validation_status="valid",
                update_frequency_days=max(7, ttl_days // 10)  # Check 10% of lifetime
            )
            
            data = asdict(lifecycle)
            self.client.table("knowledge_lifecycle").upsert(data).execute()
            return True
            
        except Exception as e:
            print(f"Error adding knowledge document: {e}")
            return False
    
    def get_documents_needing_validation(self) -> List[KnowledgeLifecycle]:
        """Get documents that need re-validation"""
        if not self.client:
            return []
        
        try:
            result = self.client.table("knowledge_lifecycle").select("*").execute()
            
            documents = []
            for row in result.data:
                doc = KnowledgeLifecycle(**row)
                if doc.needs_validation or doc.is_expired:
                    documents.append(doc)
            
            # Sort by importance and age
            documents.sort(
                key=lambda d: (d.importance_score, d.last_validated_at),
                reverse=True
            )
            
            return documents
            
        except Exception as e:
            print(f"Error getting documents for validation: {e}")
            return []
    
    def update_document_validation(
        self,
        document_id: str,
        validation_status: str,
        new_content_hash: Optional[str] = None
    ) -> bool:
        """Update document validation status"""
        if not self.client:
            return False
        
        try:
            updates = {
                "validation_status": validation_status,
                "last_validated_at": datetime.now().isoformat()
            }
            
            if new_content_hash:
                updates["content_hash"] = new_content_hash
            
            if validation_status == "valid":
                # Extend expiration
                result = self.client.table("knowledge_lifecycle").select(
                    "update_frequency_days"
                ).eq("document_id", document_id).single().execute()
                
                if result.data:
                    days = result.data["update_frequency_days"] * 10
                    updates["expires_at"] = (
                        datetime.now() + timedelta(days=days)
                    ).isoformat()
            
            self.client.table("knowledge_lifecycle").update(updates).eq(
                "document_id", document_id
            ).execute()
            
            return True
            
        except Exception as e:
            print(f"Error updating document validation: {e}")
            return False
    
    # Schema Creation
    
    def create_extended_schema(self):
        """Create all extended tables (run once during setup)"""
        if not self.client:
            print("Cannot create schema: no database connection")
            return
        
        # SQL commands to create tables
        sql_commands = [
            """
            CREATE TABLE IF NOT EXISTS tool_reliability_metrics (
                tool_name TEXT PRIMARY KEY,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                total_calls INTEGER DEFAULT 0,
                average_latency_ms REAL DEFAULT 0.0,
                last_used_at TIMESTAMP,
                last_error TEXT,
                error_patterns JSONB DEFAULT '[]'::jsonb,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS clarification_patterns (
                id TEXT PRIMARY KEY,
                original_query TEXT NOT NULL,
                query_embedding VECTOR(1536),  -- Assuming OpenAI embeddings
                clarification_question TEXT NOT NULL,
                user_response TEXT NOT NULL,
                query_category TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT NOW(),
                last_seen_at TIMESTAMP DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS plan_corrections (
                id UUID PRIMARY KEY,
                query TEXT NOT NULL,
                original_plan JSONB NOT NULL,
                corrected_plan JSONB NOT NULL,
                correction_type TEXT NOT NULL,
                user_feedback TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS knowledge_lifecycle (
                document_id TEXT PRIMARY KEY,
                source_url TEXT,
                document_type TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                ingested_at TIMESTAMP NOT NULL,
                last_validated_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                validation_status TEXT NOT NULL,
                update_frequency_days INTEGER NOT NULL,
                importance_score REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            """
        ]
        
        print("Note: Execute these SQL commands in your Supabase dashboard:")
        for sql in sql_commands:
            print(f"\n{sql}")


# Example usage
if __name__ == "__main__":
    db = ExtendedDatabase()
    
    # Example: Update tool metric
    db.update_tool_metric(
        tool_name="web_search",
        success=True,
        latency_ms=250.5
    )
    
    # Example: Get tool metrics
    metrics = db.get_tool_metrics()
    for metric in metrics:
        print(f"Tool: {metric.tool_name}, Success Rate: {metric.success_rate:.2%}")
    
    # Example: Add clarification pattern
    db.add_clarification_pattern(
        original_query="Book a flight to Paris",
        clarification_question="What dates would you like to travel?",
        user_response="Next Friday to Sunday",
        query_category="travel_booking"
    ) 