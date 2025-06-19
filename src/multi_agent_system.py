from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, Process
from langchain.tools import BaseTool
from langchain_core.messages import BaseMessage

# --- Multi-Agent System Architecture ---

class AgentRole(str, Enum):
    """Enumeration of specialized agent roles"""
    PLANNER = "planner"           # Strategic planning and task decomposition
    RESEARCHER = "researcher"     # Information gathering and analysis
    EXECUTOR = "executor"         # Tool execution and action taking
    VERIFIER = "verifier"         # Fact checking and validation
    SYNTHESIZER = "synthesizer"   # Answer synthesis and presentation

class AgentCapability(BaseModel):
    """Schema for agent capabilities"""
    role: AgentRole
    description: str
    tools: List[str] = Field(default_factory=list)
    model_config: Dict[str, Any] = Field(default_factory=dict)

@dataclass
class AgentState:
    """State shared between agents"""
    query: str
    plan: Optional[List[Dict[str, Any]]] = None
    findings: Dict[str, Any] = None
    verification_results: Dict[str, Any] = None
    final_answer: Optional[str] = None
    errors: List[str] = None

class MultiAgentSystem:
    """Orchestrates a team of specialized agents"""
    
    def __init__(self, tools: List[BaseTool], model_config: Dict[str, Any] = None):
        self.tools = tools
        self.model_config = model_config or {}
        self.state = AgentState(query="", findings={}, errors=[])
        
        # Initialize specialized agents
        self.agents = self._create_agent_team()
        
    def _create_agent_team(self) -> Dict[AgentRole, Agent]:
        """Create a team of specialized agents"""
        return {
            AgentRole.PLANNER: Agent(
                role="Strategic Planner",
                goal="Create detailed, step-by-step plans for complex tasks",
                backstory="Expert at breaking down complex problems into manageable steps",
                tools=[t for t in self.tools if t.name in ["search", "calculator"]],
                verbose=True
            ),
            AgentRole.RESEARCHER: Agent(
                role="Information Researcher",
                goal="Gather and analyze relevant information from multiple sources",
                backstory="Expert at finding and validating information from diverse sources",
                tools=[t for t in self.tools if t.name in ["search", "wikipedia"]],
                verbose=True
            ),
            AgentRole.EXECUTOR: Agent(
                role="Task Executor",
                goal="Execute specific tasks using appropriate tools",
                backstory="Expert at using tools effectively to accomplish tasks",
                tools=self.tools,
                verbose=True
            ),
            AgentRole.VERIFIER: Agent(
                role="Fact Checker",
                goal="Verify information accuracy and consistency",
                backstory="Expert at validating facts and identifying inconsistencies",
                tools=[t for t in self.tools if t.name in ["search", "calculator"]],
                verbose=True
            ),
            AgentRole.SYNTHESIZER: Agent(
                role="Answer Synthesizer",
                goal="Create clear, accurate, and well-structured answers",
                backstory="Expert at synthesizing information into coherent responses",
                tools=[],  # No tools needed for synthesis
                verbose=True
            )
        }
        
    def _create_planning_task(self, query: str) -> Task:
        """Create a planning task"""
        return Task(
            description=f"Create a detailed plan to answer: {query}",
            agent=self.agents[AgentRole.PLANNER],
            expected_output="A list of specific steps to accomplish the task"
        )
        
    def _create_research_task(self, query: str) -> Task:
        """Create a research task"""
        return Task(
            description=f"Research information relevant to: {query}",
            agent=self.agents[AgentRole.RESEARCHER],
            expected_output="Comprehensive research findings with sources"
        )
        
    def _create_execution_task(self, step: Dict[str, Any]) -> Task:
        """Create an execution task for a specific step"""
        return Task(
            description=f"Execute step: {step['description']}",
            agent=self.agents[AgentRole.EXECUTOR],
            expected_output="Results of executing the step"
        )
        
    def _create_verification_task(self, findings: Dict[str, Any]) -> Task:
        """Create a verification task"""
        return Task(
            description="Verify the accuracy and consistency of findings",
            agent=self.agents[AgentRole.VERIFIER],
            expected_output="Verification results and any identified issues"
        )
        
    def _create_synthesis_task(self, query: str, findings: Dict[str, Any]) -> Task:
        """Create a synthesis task"""
        return Task(
            description=f"Synthesize findings into a clear answer for: {query}",
            agent=self.agents[AgentRole.SYNTHESIZER],
            expected_output="A clear, accurate, and well-structured answer"
        )
        
    def process_query(self, query: str) -> str:
        """Process a user query using the multi-agent system"""
        try:
            # Create the crew
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=[],  # Will be populated based on plan
                process=Process.sequential,
                verbose=True
            )
            
            # Create initial planning task
            planning_task = self._create_planning_task(query)
            crew.tasks.append(planning_task)
            
            # Execute planning
            plan_result = crew.kickoff()
            self.state.plan = plan_result
            
            # Create and execute research task
            research_task = self._create_research_task(query)
            crew.tasks.append(research_task)
            research_result = crew.kickoff()
            self.state.findings = research_result
            
            # Create and execute verification task
            verification_task = self._create_verification_task(research_result)
            crew.tasks.append(verification_task)
            verification_result = crew.kickoff()
            self.state.verification_results = verification_result
            
            # Create and execute synthesis task
            synthesis_task = self._create_synthesis_task(query, research_result)
            crew.tasks.append(synthesis_task)
            final_answer = crew.kickoff()
            self.state.final_answer = final_answer
            
            return final_answer
            
        except Exception as e:
            error_msg = f"Error in multi-agent system: {str(e)}"
            self.state.errors.append(error_msg)
            logging.error(error_msg)
            raise
            
    def get_state(self) -> AgentState:
        """Get the current state of the multi-agent system"""
        return self.state 