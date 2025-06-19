from crewai import Crew, Agent, Task, Process
from crewai.tools import tool
from langchain.tools import BaseTool
from typing import List, Dict, Any, Optional
import logging
import asyncio
from .database_enhanced import vector_store
from .langchain_enhanced import enhanced_agent

logger = logging.getLogger(__name__)

class GAIACrewOrchestrator:
    """Enhanced CrewAI orchestration for GAIA tasks"""
    
    def __init__(self, tools: List[BaseTool]):
        self.tools = {tool.name: tool for tool in tools}
        self.agents = self._create_specialized_agents()
        
    def _get_model(self, model_type: str):
        """Get appropriate model for different agent types"""
        # TODO: Implement model selection based on type
        # For now, return a default model
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.1
        )
        
    def _create_specialized_agents(self) -> Dict[str, Agent]:
        """Create GAIA-optimized specialist agents"""
        
        return {
            "strategist": Agent(
                role="Strategic Planner",
                goal="Analyze GAIA questions and create optimal execution strategies",
                backstory="Expert in question decomposition and planning with deep understanding of GAIA question patterns",
                tools=[self.tools.get('question_analyzer', self._create_dummy_tool('question_analyzer'))],
                llm=self._get_model("strategic"),
                max_iter=3,
                memory=True,
                verbose=True
            ),
            
            "researcher": Agent(
                role="Research Specialist", 
                goal="Find accurate information from multiple sources with cross-verification",
                backstory="Expert at cross-referencing, verification, and finding authoritative sources",
                tools=[
                    self.tools.get('web_researcher', self._create_dummy_tool('web_researcher')),
                    self.tools.get('wikipedia_tool', self._create_dummy_tool('wikipedia_tool')),
                    self.tools.get('semantic_search', self._create_dummy_tool('semantic_search'))
                ],
                llm=self._get_model("research"),
                parallel_tool_calls=True,  # New feature
                verbose=True
            ),
            
            "calculator": Agent(
                role="Computation Specialist",
                goal="Perform accurate calculations and data analysis with verification",
                backstory="Mathematical precision expert with strong background in verification and error checking",
                tools=[
                    self.tools.get('python_interpreter', self._create_dummy_tool('python_interpreter')),
                    self.tools.get('wolfram_alpha', self._create_dummy_tool('wolfram_alpha'))
                ],
                llm=self._get_model("analytical"),
                verbose=True
            ),
            
            "validator": Agent(
                role="Answer Validator",
                goal="Verify and format final answers for GAIA submission",
                backstory="Quality assurance specialist with expertise in GAIA answer formatting and validation",
                tools=[self.tools.get('answer_formatter', self._create_dummy_tool('answer_formatter'))],
                llm=self._get_model("validation"),
                verbose=True
            )
        }
    
    def _create_dummy_tool(self, name: str):
        """Create dummy tool for missing dependencies"""
        @tool
        def dummy_tool(query: str) -> str:
            return f"Dummy {name} tool - not implemented"
        dummy_tool.name = name
        return dummy_tool
    
    def create_gaia_crew(self, question_type: str) -> Crew:
        """Create dynamic crew based on question type"""
        
        # Select agents based on question type
        if question_type == "calculation":
            selected_agents = [
                self.agents["strategist"],
                self.agents["calculator"],
                self.agents["validator"]
            ]
        elif question_type == "research":
            selected_agents = [
                self.agents["strategist"],
                self.agents["researcher"],
                self.agents["validator"]
            ]
        elif question_type == "factual_lookup":
            selected_agents = [
                self.agents["researcher"],
                self.agents["validator"]
            ]
        else:
            selected_agents = list(self.agents.values())
        
        return Crew(
            agents=selected_agents,
            process=Process.hierarchical,  # Better for GAIA
            manager_llm=self._get_model("manager"),
            memory=True,
            cache=True,
            max_rpm=100,
            share_crew=False,  # Isolate for GAIA
            verbose=True
        )

class GAIATaskFactory:
    """Create optimized tasks for GAIA questions"""
    
    @staticmethod
    def create_tasks(question: str, question_analysis: Dict, agents: Dict[str, Agent]) -> List[Task]:
        """Generate task chain for GAIA question"""
        
        tasks = []
        
        # 1. Planning task
        tasks.append(Task(
            description=f"Create detailed execution plan for: {question}",
            expected_output="Step-by-step plan with tool selections and verification steps",
            agent=agents.get('strategist'),
            async_execution=False,
            context=[]
        ))
        
        # 2. Execution tasks (can be parallel)
        if question_analysis.get('requires_search', False):
            tasks.append(Task(
                description=f"Search for accurate information about: {question}",
                expected_output="Relevant facts with authoritative sources and cross-verification",
                agent=agents.get('researcher'),
                async_execution=True,
                context=[tasks[0]] if tasks else []
            ))
        
        if question_analysis.get('requires_calculation', False):
            tasks.append(Task(
                description=f"Perform calculations for: {question}",
                expected_output="Numerical result with verification and error checking",
                agent=agents.get('calculator'),
                async_execution=True,
                context=[tasks[0]] if tasks else []
            ))
        
        # 3. Synthesis task
        tasks.append(Task(
            description="Synthesize all findings into final GAIA answer",
            expected_output="Concise, accurate answer in GAIA format",
            agent=agents.get('validator'),
            context=tasks[:-1] if len(tasks) > 1 else tasks,  # All previous tasks
            async_execution=False
        ))
        
        return tasks

class EnhancedCrewExecutor:
    """Enhanced crew execution with monitoring and optimization"""
    
    def __init__(self, orchestrator: GAIACrewOrchestrator):
        self.orchestrator = orchestrator
        self.execution_history = []
        
    def execute_gaia_question(self, question: str, question_analysis: Dict) -> Dict[str, Any]:
        """Execute GAIA question with enhanced crew - FIXED: Now sync"""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create crew
            crew = self.orchestrator.create_gaia_crew(question_analysis.get('type', 'general'))
            
            # Create tasks
            tasks = GAIATaskFactory.create_tasks(question, question_analysis, crew.agents)
            
            # Execute crew - FIXED: CrewAI.kickoff() is sync, not async
            result = crew.kickoff()
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Record execution
            execution_record = {
                'question': question,
                'question_type': question_analysis.get('type', 'general'),
                'execution_time': execution_time,
                'result': result,
                'success': True
            }
            self.execution_history.append(execution_record)
            
            return {
                'success': True,
                'result': result,
                'execution_time': execution_time,
                'crew_size': len(crew.agents),
                'tasks_executed': len(tasks)
            }
            
        except Exception as e:
            logger.error(f"Crew execution failed: {e}")
            # FIXED: Proper error propagation instead of swallowing
            raise RuntimeError(f"CrewAI execution failed: {e}")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {}
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for record in self.execution_history if record.get('success', False))
        avg_execution_time = sum(record.get('execution_time', 0) for record in self.execution_history) / total_executions
        
        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': successful_executions / total_executions if total_executions > 0 else 0,
            'average_execution_time': avg_execution_time
        }

def initialize_crew_enhanced(tools: List[BaseTool]) -> EnhancedCrewExecutor:
    """Initialize enhanced crew with proper error handling"""
    try:
        orchestrator = GAIACrewOrchestrator(tools)
        executor = EnhancedCrewExecutor(orchestrator)
        logger.info("Enhanced CrewAI initialized successfully")
        return executor
    except Exception as e:
        logger.error(f"Failed to initialize CrewAI: {e}")
        raise

class EnhancedKnowledgeBase:
    """Enhanced knowledge base for GAIA tasks"""
    
    def __init__(self):
        self.index = None
        self.query_engine = None

class MultiModalGAIAIndex:
    """Multi-modal index for GAIA content"""
    
    def __init__(self):
        self.text_index = None
        self.image_index = None
        self.table_index = None

class IncrementalKnowledgeBase:
    """Incremental knowledge base updates"""
    
    def __init__(self, storage_path: str = "./knowledge_cache"):
        self.storage_path = storage_path
        self.index = None

class GAIAQueryEngine:
    """Specialized query engine for GAIA tasks"""
    
    def __init__(self, index):
        self.index = index
        self.query_engine = None 