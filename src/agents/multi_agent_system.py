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
    """Orchestrates a team of specialized agents with unified tool management"""
    
    def __init__(self, tools: List[BaseTool], model_config: Dict[str, Any] = None):
        self.tools = tools
        self.model_config = model_config or {}
        self.state = AgentState(query="", findings={}, errors=[])
        
        # Initialize unified tool registry integration
        try:
            from src.integration_hub import get_unified_registry, get_tool_orchestrator
            self.unified_registry = get_unified_registry()
            self.tool_orchestrator = get_tool_orchestrator()
            
            # Register tools with unified registry
            for tool in tools:
                self.unified_registry.register(tool)
            
            logger.info("Multi-agent system registered {} tools with unified registry", extra={"len_tools_": len(tools)})
            
        except ImportError:
            logger.warning("Unified tool registry not available, using local tools")
            self.unified_registry = None
            self.tool_orchestrator = None
        
        # Initialize tool introspection
        try:
            from src.tools_introspection import tool_introspector
            self.tool_introspector = tool_introspector
            
            # Register tools with introspector
            for tool in tools:
                if hasattr(tool, 'name'):
                    self.tool_introspector.tool_registry[tool.name] = tool
            
            logger.info("Multi-agent system initialized with tool introspection")
            
        except ImportError:
            logger.warning("Tool introspection not available")
            self.tool_introspector = None
        
        # Initialize specialized agents
        self.agents = self._create_agent_team()
        
    def _create_agent_team(self) -> Dict[AgentRole, Agent]:
        """Create a team of specialized agents with tool introspection"""
        agents = {}
        
        # Define agent configurations with tool assignments
        agent_configs = {
            AgentRole.PLANNER: {
                "role": "Strategic Planner",
                "goal": "Create detailed, step-by-step plans for complex tasks",
                "backstory": "Expert at breaking down complex problems into manageable steps",
                "tool_categories": ["search", "calculator", "planning"],
                "verbose": True
            },
            AgentRole.RESEARCHER: {
                "role": "Information Researcher",
                "goal": "Gather and analyze relevant information from multiple sources",
                "backstory": "Expert at finding and validating information from diverse sources",
                "tool_categories": ["search", "wikipedia", "web_research", "semantic_search"],
                "verbose": True
            },
            AgentRole.EXECUTOR: {
                "role": "Task Executor",
                "goal": "Execute specific tasks using appropriate tools",
                "backstory": "Expert at using tools effectively to accomplish tasks",
                "tool_categories": ["execution", "calculation", "file_processing", "code_execution"],
                "verbose": True
            },
            AgentRole.VERIFIER: {
                "role": "Fact Checker",
                "goal": "Verify information accuracy and consistency",
                "backstory": "Expert at validating facts and identifying inconsistencies",
                "tool_categories": ["search", "calculator", "verification", "fact_checking"],
                "verbose": True
            },
            AgentRole.SYNTHESIZER: {
                "role": "Answer Synthesizer",
                "goal": "Create clear, accurate, and well-structured answers",
                "backstory": "Expert at synthesizing information into coherent responses",
                "tool_categories": [],  # No tools needed for synthesis
                "verbose": True
            }
        }
        
        # Create agents with appropriate tools
        for role, config in agent_configs.items():
            # Get tools for this agent role
            agent_tools = self._get_tools_for_role(role, config["tool_categories"])
            
            agents[role] = Agent(
                role=config["role"],
                goal=config["goal"],
                backstory=config["backstory"],
                tools=agent_tools,
                verbose=config["verbose"]
            )
            
            logger.info("Created {} agent with {} tools", extra={"role": role, "len_agent_tools_": len(agent_tools)})
        
        return agents
    
    def _get_tools_for_role(self, role: AgentRole, tool_categories: List[str]) -> List[BaseTool]:
        """Get tools suitable for a specific agent role using introspection and reliability"""
        if not tool_categories:
            return []
        
        # Use unified registry if available
        if self.unified_registry:
            # Get reliable tools for this role
            reliable_tools = self.unified_registry.get_tools_by_reliability(min_success_rate=0.7)
            
            # Filter by role-specific categories
            role_tools = []
            for tool in reliable_tools:
                if hasattr(tool, 'name'):
                    # Check if tool matches any category for this role
                    if any(category in tool.name.lower() for category in tool_categories):
                        role_tools.append(tool)
            
            if role_tools:
                logger.info("Found {} reliable tools for {}", extra={"len_role_tools_": len(role_tools), "role": role})
                return role_tools
        
        # Fallback to tool introspection
        if self.tool_introspector:
            try:
                # Use introspection to find suitable tools
                suitable_tools = []
                for tool in self.tools:
                    if hasattr(tool, 'name'):
                        # Analyze tool capabilities for this role
                        tool_schema = self.tool_introspector.get_tool_schema(tool.name)
                        if tool_schema and any(category in tool_schema.get("description", "").lower() 
                                             for category in tool_categories):
                            suitable_tools.append(tool)
                
                if suitable_tools:
                    logger.info("Found {} suitable tools for {} via introspection", extra={"len_suitable_tools_": len(suitable_tools), "role": role})
                    return suitable_tools
                    
            except Exception as e:
                logger.warning("Tool introspection failed for {}: {}", extra={"role": role, "e": e})
        
        # Final fallback: return tools that match category names
        fallback_tools = []
        for tool in self.tools:
            if hasattr(tool, 'name'):
                if any(category in tool.name.lower() for category in tool_categories):
                    fallback_tools.append(tool)
        
        logger.info("Using {} fallback tools for {}", extra={"len_fallback_tools_": len(fallback_tools), "role": role})
        return fallback_tools
    
    def _filter_by_reliability(self, tools: List[BaseTool]) -> List[BaseTool]:
        """Filter tools by reliability score"""
        if not self.unified_registry:
            return tools
        
        reliable_tools = []
        for tool in tools:
            if hasattr(tool, 'name'):
                reliability = self.unified_registry.tool_reliability.get(tool.name, {})
                total_calls = reliability.get("total_calls", 0)
                success_count = reliability.get("success_count", 0)
                
                # Include tools with good reliability or new tools
                if total_calls == 0 or (success_count / total_calls >= 0.7):
                    reliable_tools.append(tool)
        
        return reliable_tools
    
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
        """Process a user query using the multi-agent system with enhanced tool management"""
        try:
            # Update state
            self.state.query = query
            
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
            
            # Update tool reliability metrics if orchestrator is available
            if self.tool_orchestrator:
                self._update_tool_metrics()
            
            return final_answer
            
        except Exception as e:
            error_msg = f"Error in multi-agent system: {str(e)}"
            self.state.errors.append(error_msg)
            logging.error(error_msg)
            raise
    
    def _update_tool_metrics(self):
        """Update tool reliability metrics based on execution results"""
        if not self.tool_orchestrator or not self.unified_registry:
            return
        
        try:
            # This would update metrics based on tool usage during execution
            # For now, this is a placeholder for the actual implementation
            logger.debug("Tool metrics would be updated here")
            
        except Exception as e:
            logger.warning("Failed to update tool metrics: {}", extra={"e": e})
            
    def get_state(self) -> AgentState:
        """Get the current state of the multi-agent system"""
        return self.state
    
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics for the multi-agent system"""
        if not self.unified_registry:
            return {}
        
        stats = {}
        for tool_name, reliability in self.unified_registry.tool_reliability.items():
            total_calls = reliability.get("total_calls", 0)
            success_count = reliability.get("success_count", 0)
            
            if total_calls > 0:
                stats[tool_name] = {
                    "total_calls": total_calls,
                    "success_rate": success_count / total_calls,
                    "avg_latency": reliability.get("avg_latency", 0.0),
                    "last_used": reliability.get("last_used")
                }
        
        return stats 