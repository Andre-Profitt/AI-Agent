from agent import answer
from agent import query
from agent import response
from agent import tools
from examples.basic.simple_hybrid_demo import reason

from src.collaboration.realtime_collaboration import session_id
from src.database.models import action
from src.database.models import query_id
from src.database_extended import interaction
from src.meta_cognition import confidence
from src.tools_interactive import approval_id
from src.tools_interactive import clarification_id
from src.tools_interactive import confirmation_id
from src.tools_interactive import cutoff
from src.tools_interactive import to_remove
from src.utils.tools_introspection import field

from src.tools.base_tool import Tool

from src.tools.base_tool import BaseTool
from src.shared.types.di_types import BaseTool
# TODO: Fix undefined variables: Any, Callable, Dict, Enum, List, Optional, action, answer, approval_id, callbacks, clarification, clarification_id, confidence, confirmation_id, cutoff, dataclass, datetime, field, interaction, interaction_id, k, logging, max_age_hours, options, query, query_id, reason, response, session_id, to_remove, tools, v
# TODO: Fix undefined variables: action, answer, approval_id, callbacks, clarification, clarification_id, confidence, confirmation_id, cutoff, interaction, interaction_id, k, max_age_hours, options, query, query_id, reason, response, self, session_id, to_remove, tools, v

"""
Interactive Tools Module
Provides interactive tools for clarification, approval, and user feedback
"""

from typing import Optional
from dataclasses import field
from typing import Any
from typing import List
from typing import Callable

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class InteractionType(str, Enum):
    """Types of interactive interactions"""
    CLARIFICATION = "clarification"
    APPROVAL = "approval"
    CONFIRMATION = "confirmation"
    FEEDBACK = "feedback"

@dataclass
class InteractiveState:
    """State for interactive tools"""
    session_id: str
    pending_interactions: List[Dict[str, Any]] = field(default_factory=list)
    user_responses: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

class ClarificationTracker:
    """Tracks clarification requests and responses"""

    def __init__(self):
        self.clarifications = {}
        self.responses = {}
        self.pending = {}

    def add_clarification(self, query_id: str, clarification: str, options: List[str] = None) -> str:
        """Add a clarification request"""
        clarification_id = f"clar_{query_id}_{len(self.clarifications)}"

        self.clarifications[clarification_id] = {
            "query_id": query_id,
            "clarification": clarification,
            "options": options or [],
            "timestamp": datetime.now(),
            "status": "pending"
        }

        self.pending[clarification_id] = True

        logger.info(f"Added clarification request: {clarification_id}")
        return clarification_id

    def add_response(self, clarification_id: str, response: str) -> bool:
        """Add a response to a clarification"""
        if clarification_id not in self.clarifications:
            logger.warning(f"Clarification {clarification_id} not found")
            return False

        self.responses[clarification_id] = {
            "response": response,
            "timestamp": datetime.now()
        }

        self.clarifications[clarification_id]["status"] = "responded"
        self.pending.pop(clarification_id, None)

        logger.info(f"Added response to clarification {clarification_id}: {response}")
        return True

    def get_pending_clarifications(self) -> List[Dict[str, Any]]:
        """Get all pending clarifications"""
        return [
            {"id": k, **v}
            for k, v in self.clarifications.items()
            if v["status"] == "pending"
        ]

    def get_response(self, clarification_id: str) -> Optional[str]:
        """Get response for a clarification"""
        return self.responses.get(clarification_id, {}).get("response")

    def clear_old_clarifications(self, max_age_hours: int = 24) -> None:
        """Clear old clarifications"""
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)

        to_remove = []
        for clarification_id, clarification in self.clarifications.items():
            if clarification["timestamp"].timestamp() < cutoff:
                to_remove.append(clarification_id)

        for clarification_id in to_remove:
            self.clarifications.pop(clarification_id, None)
            self.responses.pop(clarification_id, None)
            self.pending.pop(clarification_id, None)

        if to_remove:
            logger.info(f"Cleared {len(to_remove)} old clarifications")

# Global state
interactive_state = InteractiveState(session_id="default")
clarification_tracker = ClarificationTracker()

def get_interactive_tools() -> List[Any]:
    """Get list of interactive tools"""
    tools = []

    try:
        from langchain.tools import BaseTool

        class ClarificationTool(BaseTool):
            name = "request_clarification"
            description = "Request clarification from the user when the query is ambiguous or unclear"

            def _run(self, query: str, clarification: str, options: List[str] = None) -> str:
                """Request clarification from user"""
                clarification_id = clarification_tracker.add_clarification(
                    query, clarification, options
                )

                response = f"Clarification requested: {clarification}"
                if options:
                    response += f"\nOptions: {', '.join(options)}"

                logger.info(f"Requested clarification: {clarification}")
                return response

            def _arun(self, query: str, clarification: str, options: List[str] = None) -> str:
                return self._run(query, clarification, options)

        class ApprovalTool(BaseTool):
            name = "request_approval"
            description = "Request user approval for a proposed action or answer"

            def _run(self, action: str, reason: str = None) -> str:
                """Request approval from user"""
                approval_id = f"approval_{len(interactive_state.pending_interactions)}"

                interaction = {
                    "id": approval_id,
                    "type": InteractionType.APPROVAL,
                    "action": action,
                    "reason": reason,
                    "timestamp": datetime.now(),
                    "status": "pending"
                }

                interactive_state.pending_interactions.append(interaction)

                response = f"Approval requested for: {action}"
                if reason:
                    response += f"\nReason: {reason}"

                logger.info(f"Requested approval: {action}")
                return response

            def _arun(self, action: str, reason: str = None) -> str:
                return self._run(action, reason)

        class ConfirmationTool(BaseTool):
            name = "request_confirmation"
            description = "Request user confirmation for a proposed answer or solution"

            def _run(self, answer: str, confidence: float = 0.8) -> str:
                """Request confirmation for an answer"""
                confirmation_id = f"confirm_{len(interactive_state.pending_interactions)}"

                interaction = {
                    "id": confirmation_id,
                    "type": InteractionType.CONFIRMATION,
                    "answer": answer,
                    "confidence": confidence,
                    "timestamp": datetime.now(),
                    "status": "pending"
                }

                interactive_state.pending_interactions.append(interaction)

                response = f"Confirmation requested for answer (confidence: {confidence:.2f}):\n{answer}"

                logger.info(f"Requested confirmation for answer with {confidence:.2f} confidence")
                return response

            def _arun(self, answer: str, confidence: float = 0.8) -> str:
                return self._run(answer, confidence)

        tools.extend([
            ClarificationTool(),
            ApprovalTool(),
            ConfirmationTool()
        ])

    except ImportError:
        logger.warning("LangChain BaseTool not available, skipping interactive tools")

    return tools

def setup_interactive_callbacks(self, callbacks: Dict[str, Callable]) -> None:
    """Setup callbacks for interactive tools"""
    global interactive_state

    # Store callbacks in interactive state
    interactive_state.callbacks = callbacks

    logger.info("Interactive callbacks set up")

def get_pending_interactions() -> List[Dict[str, Any]]:
    """Get all pending interactions"""
    return interactive_state.pending_interactions

def add_user_response(self, interaction_id: str, response: str) -> bool:
    """Add a user response to an interaction"""
    # Find the interaction
    for interaction in interactive_state.pending_interactions:
        if interaction["id"] == interaction_id:
            interaction["status"] = "responded"
            interaction["user_response"] = response
            interaction["response_timestamp"] = datetime.now()

            # Move to history
            interactive_state.interaction_history.append(interaction)
            interactive_state.pending_interactions.remove(interaction)

            logger.info(f"Added user response to interaction {interaction_id}: {response}")
            return True

    logger.warning(f"Interaction {interaction_id} not found")
    return False

def clear_pending_interactions() -> None:
    """Clear all pending interactions"""
    interactive_state.pending_interactions.clear()
    logger.info("Cleared all pending interactions")

def get_interaction_history() -> List[Dict[str, Any]]:
    """Get interaction history"""
    return interactive_state.interaction_history

def create_interactive_session(self, session_id: str) -> InteractiveState:
    """Create a new interactive session"""
    global interactive_state
    interactive_state = InteractiveState(session_id=session_id)
    logger.info(f"Created new interactive session: {session_id}")
    return interactive_state