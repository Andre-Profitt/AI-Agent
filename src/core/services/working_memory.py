from agent import response
from examples.parallel_execution_example import tool_calls
from tests.load_test import data

from src.config.integrations import api_key
from src.core.services.working_memory import combined_text
from src.core.services.working_memory import compressed
from src.core.services.working_memory import compressed_json
from src.core.services.working_memory import compression_prompt
from src.core.services.working_memory import date_pattern
from src.core.services.working_memory import dates
from src.core.services.working_memory import last_tool
from src.core.services.working_memory import new_state
from src.core.services.working_memory import question_words
from src.core.services.working_memory import state_json
from src.core.services.working_memory import summary_parts
from src.core.services.working_memory import tool_details
from src.core.services.working_memory import tool_summary
from src.core.services.working_memory import update_prompt
from src.core.services.working_memory import updated_json
from src.core.services.working_memory import user_lower
from src.gaia_components.advanced_reasoning_engine import prompt
from src.query_classifier import entities
from src.services.integration_hub import parts
from src.utils.error_category import call
from src.utils.knowledge_utils import word
from src.utils.tools_introspection import field

from src.tools.base_tool import Tool
# TODO: Fix undefined variables: Any, Dict, List, Optional, api_key, asdict, assistant_response, call, cls, combined_text, compressed, compressed_json, compression_prompt, data, dataclass, date, date_pattern, dates, datetime, e, entities, entity_type, field, json, json_str, last_tool, logging, max_tokens, new_state, os, parts, prompt, question_words, re, response, state_json, summary_parts, tool_calls, tool_details, tool_summary, update_prompt, updated_json, user_lower, user_message, v, word

"""
from dataclasses import asdict
from typing import Optional
# TODO: Fix undefined variables: anthropic, api_key, assistant_response, call, cls, combined_text, compressed, compressed_json, compression_prompt, data, date_pattern, dates, e, entities, entity_type, json_str, last_tool, max_tokens, new_state, parts, prompt, question_words, response, self, state_json, summary_parts, tool_calls, tool_details, tool_summary, update_prompt, updated_json, user_lower, user_message, v, word

from unittest.mock import call
Working Memory Module
Implements structured conversation summarization for token-efficient context
"""

from dataclasses import field
from typing import Any
from typing import List

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

import os
import anthropic

import logging

logger = logging.getLogger(__name__)

@dataclass
class WorkingMemoryState:
    """Structured state object for working memory"""
    conversation_summary: str = ""
    identified_entities: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    inferred_user_intent: str = ""
    current_task: str = ""
    task_progress: Dict[str, Any] = field(default_factory=dict)
    last_task_status: Dict[str, Any] = field(default_factory=dict)
    key_facts: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'WorkingMemoryState':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls(**data)

    def get_context_string(self) -> str:
        """Get a concise context string for prompts"""
        parts = []

        if self.conversation_summary:
            parts.append(f"Summary: {self.conversation_summary}")

        if self.inferred_user_intent:
            parts.append(f"Intent: {self.inferred_user_intent}")

        if self.current_task:
            parts.append(f"Current Task: {self.current_task}")

        if self.key_facts:
            parts.append(f"Key Facts: {'; '.join(self.key_facts[:5])}")

        if self.open_questions:
            parts.append(f"Open Questions: {'; '.join(self.open_questions[:3])}")

        return " | ".join(parts)

class WorkingMemoryManager:
    """Manages working memory updates and compression"""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize with optional API key"""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None

        self.current_state = WorkingMemoryState()
        self.update_history: List[Dict[str, Any]] = []

    def update_memory(
        self,
        user_message: str,
        assistant_response: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> WorkingMemoryState:
        """
        Update working memory after each interaction

        Args:
            user_message: The user's latest message
            assistant_response: The assistant's response
            tool_calls: Any tool calls made during this turn

        Returns:
            Updated working memory state
        """
        if not self.client:
            # Fallback to simple heuristic update
            return self._heuristic_update(user_message, assistant_response, tool_calls)

        # Prepare the update prompt
        update_prompt = self._create_update_prompt(
            user_message, assistant_response, tool_calls
        )

        try:
            # Use Claude to update the structured state
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": update_prompt
                }]
            )

            # Parse the response
            updated_json = response.content[0].text
            new_state = WorkingMemoryState.from_json(updated_json)

            # Record the update
            self.update_history.append({
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message,
                "previous_state": asdict(self.current_state),
                "new_state": asdict(new_state)
            })

            self.current_state = new_state
            return new_state

        except Exception as e:
            logger.info("Error updating working memory: {}", extra={"e": e})
            return self._heuristic_update(user_message, assistant_response, tool_calls)

    def _create_update_prompt(
        self,
        user_message: str,
        assistant_response: str,
        tool_calls: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Create prompt for updating working memory"""

        # Format tool calls summary
        tool_summary = ""
        if tool_calls:
            tool_details = []
            for call in tool_calls:
                tool_details.append(
                    f"- {call.get('tool_name', 'unknown')}: "
                    f"{call.get('status', 'unknown')} - "
                    f"{str(call.get('output', 'no output'))[:100]}"
                )
            tool_summary = "\n".join(tool_details)

        prompt = f"""Update the working memory based on this conversation turn.

Current Working Memory:
{self.current_state.to_json()}

Latest Exchange:
User: {user_message}
Assistant: {assistant_response}

Tool Calls Made:
{tool_summary if tool_summary else "None"}

Instructions:
1. Update the conversation_summary to include key points from this exchange
2. Extract any new entities (people, places, dates, etc.)
3. Update or confirm the user's intent
4. Update the current task and progress
5. Add any important facts learned
6. Note any questions that remain open
7. Record any user preferences expressed

Keep the summary concise and focused on actionable information.
The total JSON should be under 500 tokens.

Respond with ONLY the updated JSON object."""

        return prompt

    def _heuristic_update(
        self,
        user_message: str,
        assistant_response: str,
        tool_calls: Optional[List[Dict[str, Any]]]
    ) -> WorkingMemoryState:
        """Simple heuristic update when LLM is unavailable"""

        # Update conversation summary (keep last 200 chars + new)
        summary_parts = []
        if self.current_state.conversation_summary:
            summary_parts.append(self.current_state.conversation_summary[-200:])
        summary_parts.append(f"User asked: {user_message[:100]}")
        summary_parts.append(f"Assistant: {assistant_response[:100]}")

        self.current_state.conversation_summary = " → ".join(summary_parts)

        # Extract entities (simple pattern matching)
        import re

        # Look for potential entities
        combined_text = f"{user_message} {assistant_response}"

        # Dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        dates = re.findall(date_pattern, combined_text)
        if dates:
            if "dates" not in self.current_state.identified_entities:
                self.current_state.identified_entities["dates"] = []
            for date in dates[:3]:  # Limit to 3
                self.current_state.identified_entities["dates"].append({
                    "value": date,
                    "context": "mentioned in conversation"
                })

        # Update task status if tools were used
        if tool_calls:
            last_tool = tool_calls[-1]
            self.current_state.last_task_status = {
                "tool": last_tool.get("tool_name", "unknown"),
                "status": "success" if not last_tool.get("error") else "failed",
                "timestamp": datetime.now().isoformat()
            }

        # Infer intent from question words
        question_words = ["what", "who", "where", "when", "why", "how", "can", "will", "should"]
        user_lower = user_message.lower()
        for word in question_words:
            if user_lower.startswith(word):
                self.current_state.inferred_user_intent = f"Seeking information ({word})"
                break

        return self.current_state

    def compress_memory(self, max_tokens: int = 500) -> WorkingMemoryState:
        """
        Compress working memory to fit within token limit

        Args:
            max_tokens: Maximum tokens for compressed state

        Returns:
            Compressed working memory state
        """
        if not self.client:
            # Simple truncation
            return self._simple_compress(max_tokens)

        compression_prompt = f"""Compress this working memory to fit within {max_tokens} tokens while preserving the most important information.

Current Working Memory:
{self.current_state.to_json()}

Prioritize:
1. Current task and intent
2. Recent key facts
3. Open questions
4. Essential entities

Respond with ONLY the compressed JSON object."""

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=max_tokens,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": compression_prompt
                }]
            )

            compressed_json = response.content[0].text
            return WorkingMemoryState.from_json(compressed_json)

        except Exception as e:
            logger.info("Error compressing memory: {}", extra={"e": e})
            return self._simple_compress(max_tokens)

    def _simple_compress(self, max_tokens: int) -> WorkingMemoryState:
        """Simple compression by truncation"""
        compressed = WorkingMemoryState()

        # Copy most important fields with truncation
        compressed.conversation_summary = self.current_state.conversation_summary[-200:]
        compressed.inferred_user_intent = self.current_state.inferred_user_intent
        compressed.current_task = self.current_state.current_task
        compressed.last_task_status = self.current_state.last_task_status

        # Keep only recent facts and questions
        compressed.key_facts = self.current_state.key_facts[-5:]
        compressed.open_questions = self.current_state.open_questions[-3:]

        # Keep minimal entities
        for entity_type, entities in self.current_state.identified_entities.items():
            compressed.identified_entities[entity_type] = entities[-2:]

        return compressed

    def get_memory_for_prompt(self) -> str:
        """Get formatted memory for inclusion in prompts"""
        return self.current_state.get_context_string()

    def clear_memory(self) -> Any:
        """Clear working memory and start fresh"""
        self.current_state = WorkingMemoryState()
        self.update_history.clear()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        state_json = self.current_state.to_json()

        return {
            "state_size_bytes": len(state_json.encode()),
            "estimated_tokens": len(state_json) // 4,  # Rough estimate
            "num_entities": sum(len(v) for v in self.current_state.identified_entities.values()),
            "num_facts": len(self.current_state.key_facts),
            "num_open_questions": len(self.current_state.open_questions),
            "update_count": len(self.update_history)
        }

# Example usage
if __name__ == "__main__":
    memory = WorkingMemoryManager()

    # Simulate a conversation turn
    memory.update_memory(
        user_message="What's the weather like in Tokyo today?",
        assistant_response="I'll check the weather in Tokyo for you.",
        tool_calls=[{
            "tool_name": "weather_api",
            "status": "success",
            "output": {"temperature": 22, "condition": "sunny"}
        }]
    )

    logger.info("Working Memory State:")
    logger.info("Result: %s", memory.current_state.to_json())

    logger.info("\nMemory Context String:")
    logger.info("Result: %s", memory.get_memory_for_prompt())

    logger.info("\nMemory Stats:")
    logger.info("Result: %s", memory.get_memory_stats())