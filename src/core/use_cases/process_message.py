from examples.enhanced_unified_example import execution_time
from examples.enhanced_unified_example import fsm_agents
from examples.enhanced_unified_example import metrics
from examples.enhanced_unified_example import start_time
from migrations.env import config

from src.api_server import message
from src.collaboration.realtime_collaboration import session_id
from src.core.entities.agent import Agent
from src.core.use_cases.process_message import malicious_patterns
from src.core.use_cases.process_message import saved_message
from src.core.use_cases.process_message import saved_response
from src.database.models import user_id
from src.gaia_components.multi_agent_orchestrator import available_agents
from src.main import agent_executor
from src.main import logging_service
from src.templates.template_factory import pattern
from src.unified_architecture.communication import response_message

from src.agents.advanced_agent_fsm import AgentType

from src.agents.advanced_agent_fsm import Agent

from src.agents.advanced_agent_fsm import AgentConfig
# TODO: Fix undefined variables: Any, Dict, Optional, UUID, a, agent_executor, agent_repository, available_agents, config, context, e, execution_time, fsm_agents, logging, logging_service, malicious_patterns, message, message_repository, metrics, re, response_message, result, saved_message, saved_response, session_id, start_time, time, user_id, user_message
from tests.test_gaia_agent import agent
import pattern

from src.core.entities.agent import AgentType
from src.infrastructure.config import AgentConfig


"""
from typing import Dict
from src.gaia_components.multi_agent_orchestrator import Agent
from src.infrastructure.agents.agent_factory import AgentType
from src.infrastructure.agents.concrete_agents import AgentConfig
from src.infrastructure.logging.logging_service import LoggingService
from src.shared.exceptions import ValidationException
from src.unified_architecture.communication import MessageType
# TODO: Fix undefined variables: a, agent, agent_executor, agent_repository, available_agents, config, context, e, execution_time, fsm_agents, logging_service, malicious_patterns, message, message_repository, metrics, pattern, response_message, result, saved_message, saved_response, self, session_id, start_time, user_id, user_message

from langchain.agents import AgentExecutor
Use case for processing user messages through the AI agent system.
"""

from typing import Optional
from typing import Any

from uuid import UUID
import time
import logging

from src.core.entities.message import Message, MessageType
from src.core.interfaces.agent_repository import AgentRepository
from src.core.interfaces.message_repository import MessageRepository
from src.core.interfaces.agent_executor import AgentExecutor
from src.core.interfaces.logging_service import LoggingService
from src.shared.exceptions import DomainException, ValidationException
from src.shared.types import AgentConfig

class ProcessMessageUseCase:
    """
    Use case for processing user messages through the AI agent system.

    This use case orchestrates the entire message processing workflow,
    including agent selection, execution, and response generation.
    """

    def __init__(
        self,
        agent_repository: AgentRepository,
        message_repository: MessageRepository,
        agent_executor: AgentExecutor,
        logging_service: LoggingService,
        config: AgentConfig
    ):
        self.agent_repository = agent_repository
        self.message_repository = message_repository
        self.agent_executor = agent_executor
        self.logging_service = logging_service
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def execute(
        self,
        user_message: str,
        session_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the message processing workflow.

        Args:
            user_message: The user's input message
            session_id: Optional session identifier
            user_id: Optional user identifier
            context: Optional context information

        Returns:
            Dictionary containing the processing result

        Raises:
            ValidationException: If input validation fails
            DomainException: If business rules are violated
        """
        start_time = time.time()

        try:
            # 1. Validate input
            self._validate_input(user_message)

            # 2. Create message entity
            message = Message(
                content=user_message,
                message_type=MessageType.USER,
                session_id=session_id,
                user_id=user_id,
                context=context or {}
            )

            # 3. Save message
            saved_message = await self.message_repository.save(message)

            # 4. Select appropriate agent
            agent = await self._select_agent(user_message, context)

            # 5. Execute agent
            result = await self._execute_agent(agent, saved_message)

            # 6. Create response message
            response_message = Message(
                content=result["response"],
                message_type=MessageType.AGENT,
                session_id=session_id,
                user_id=user_id,
                context={"agent_id": str(agent.id), "execution_time": result["execution_time"]}
            )

            # 7. Save response
            saved_response = await self.message_repository.save(response_message)

            # 8. Update agent metrics
            await self._update_agent_metrics(agent, result)

            # 9. Log interaction
            await self._log_interaction(saved_message, saved_response, result)

            execution_time = time.time() - start_time

            return {
                "success": True,
                "response": result["response"],
                "agent_id": str(agent.id),
                "agent_type": agent.agent_type.value,
                "execution_time": execution_time,
                "message_id": str(saved_message.id),
                "response_id": str(saved_response.id),
                "confidence": result.get("confidence", 0.0),
                "tools_used": result.get("tools_used", []),
                "metadata": result.get("metadata", {})
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error("Message processing failed: {}", exc_info=True)

            # Log error
            await self.logging_service.log_error(
                "message_processing_failed",
                str(e),
                {}
            )

            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }

    def _validate_input(self, user_message: str) -> None:
        """Validate user input."""
        if not user_message or not user_message.strip():
            raise ValidationException("Message cannot be empty", extra={"str_e_": str(e), "_execution_time_": "execution_time", "_success_": "success"})

        if len(user_message) > self.config.max_input_length:
            raise ValidationException(
                f"Message too long. Maximum length is {self.config.max_input_length} characters"
            )

        # Check for potentially malicious content
        if self._contains_malicious_content(user_message):
            raise ValidationException("Message contains potentially malicious content")

    def _contains_malicious_content(self, message: str) -> bool:
        """Check for potentially malicious content."""
        malicious_patterns = [
            r"ignore\s+previous\s+instructions",
            r"disregard\s+all\s+prior",
            r"system\s*:\s*you\s+are",
            r"<\|im_start\|>",
            r"<\|im_end\|>",
            r"\[INST\]",
            r"\[/INST\]"
        ]

        import re
        for pattern in malicious_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return True

        return False

    async def _select_agent(self, message: str, context: Optional[Dict[str, Any]]) -> Agent:
        """Select the most appropriate agent for the message."""
        # For now, get the first available agent
        # In the future, this could implement more sophisticated selection logic
        available_agents = await self.agent_repository.find_available()

        if not available_agents:
            raise DomainException("No available agents found")

        # Simple selection: prefer FSM_REACT agents
        fsm_agents = [a for a in available_agents if a.agent_type == AgentType.FSM_REACT]
        if fsm_agents:
            return fsm_agents[0]

        return available_agents[0]

    async def _execute_agent(self, agent: Agent, message: Message) -> Dict[str, Any]:
        """Execute the agent with the given message."""
        # Update agent state
        agent.start_task(f"Processing message: {message.content[:100]}...")
        await self.agent_repository.update_state(agent.id, agent.state)

        try:
            # Execute agent
            result = await self.agent_executor.execute(agent, message)

            # Update agent state
            agent.complete_task(success=True)
            await self.agent_repository.update_state(agent.id, agent.state)

            return result

        except Exception as e:
            # Update agent state
            agent.enter_error_state(str(e))
            await self.agent_repository.update_state(agent.id, agent.state)
            raise

    async def _update_agent_metrics(self, agent: Agent, result: Dict[str, Any]) -> None:
        """Update agent performance metrics."""
        metrics = {
            "total_requests": agent.total_requests,
            "successful_requests": agent.successful_requests,
            "failed_requests": agent.failed_requests,
            "average_response_time": agent.average_response_time,
            "last_active": agent.last_active.isoformat()
        }

        await self.agent_repository.update_performance_metrics(agent.id, metrics)

    async def _log_interaction(
        self,
        user_message: Message,
        response_message: Message,
        result: Dict[str, Any]
    ) -> None:
        """Log the interaction for analytics."""
        await self.logging_service.log_interaction(
            user_message_id=str(user_message.id),
            response_message_id=str(response_message.id),
            session_id=str(user_message.session_id) if user_message.session_id else None,
            user_id=str(user_message.user_id) if user_message.user_id else None,
            execution_time=result.get("execution_time", 0.0),
            agent_id=result.get("agent_id"),
            tools_used=result.get("tools_used", []),
            confidence=result.get("confidence", 0.0)
        )