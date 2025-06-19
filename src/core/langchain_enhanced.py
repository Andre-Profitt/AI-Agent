from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.cache import SQLiteCache
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import BaseTool
from langchain_core.messages import ToolCall
from langchain.callbacks.base import BaseCallbackHandler
import langchain
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
import time
import logging
import asyncio
from typing import Optional, Dict, Any, List, Union, Tuple

logger = logging.getLogger(__name__)

# Enable caching for faster responses
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

class CustomMetricsCallback(BaseCallbackHandler):
    """Custom callback for tracking metrics"""
    
    def __init__(self) -> None:
        self.token_count = 0
        self.start_time = None
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> Any:
        """Track LLM start"""
        self.start_time = time.time()
        logger.info("LLM started")
        
    def on_llm_end(self, response, **kwargs) -> Any:
        """Track LLM end and metrics"""
        if self.start_time:
            duration = time.time() - self.start_time
            logger.info("LLM completed in {}s", extra={"duration": duration})
            
    def on_llm_error(self, error: str, **kwargs) -> Any:
        """Track LLM errors"""
        logger.error("LLM error: {}", extra={"error": error})

class ErrorRecoveryCallback(BaseCallbackHandler):
    """Callback for error recovery strategies"""
    
    def __init__(self) -> None:
        self.error_count = 0
        self.last_error_time = None
        
    def on_llm_error(self, error: str, **kwargs) -> Any:
        """Handle LLM errors with recovery strategies"""
        self.error_count += 1
        self.last_error_time = time.time()
        
        if "context length" in error.lower():
            logger.warning("Context length exceeded, reducing context")
            # TODO: Implement context reduction
        elif "timeout" in error.lower():
            logger.warning("LLM timeout, switching to faster model")
            # TODO: Implement model switching
        else:
            logger.error("Unhandled LLM error: {}", extra={"error": error})

class ParallelToolExecutor:
    """Execute compatible tools in parallel"""
    
    def __init__(self, tools: List[BaseTool]) -> None:
        self.tools = {tool.name: tool for tool in tools}
        self.executor = ThreadPoolExecutor(max_workers=5)
        
    def _group_compatible_tools(self, tool_calls: List[ToolCall]) -> List[List[ToolCall]]:
        """Group tools by compatibility for parallel execution"""
        # Simple grouping: tools that don't share resources can run in parallel
        # For now, group by tool type
        groups = {}
        for tool_call in tool_calls:
            tool_name = tool_call.get('name', 'unknown')
            tool_type = tool_name.split('_')[0]  # Extract tool type
            
            if tool_type not in groups:
                groups[tool_type] = []
            groups[tool_type].append(tool_call)
        
        return list(groups.values())
        
    async def execute_parallel(
        self, 
        tool_calls: List[ToolCall]
    ) -> Dict[str, Any]:
        """Execute non-conflicting tools in parallel"""
        # Group tools by compatibility
        groups = self._group_compatible_tools(tool_calls)
        
        results = {}
        for group in groups:
            if len(group) == 1:
                # Single tool, execute normally
                tool_call = group[0]
                results[tool_call.get('id', 'unknown')] = await self._execute_single(tool_call)
            else:
                # Multiple compatible tools, execute in parallel
                futures = []
                for tool_call in group:
                    future = self.executor.submit(
                        self._execute_single_sync, 
                        tool_call
                    )
                    futures.append((tool_call.get('id', 'unknown'), future))
                
                # Collect results
                for tool_id, future in futures:
                    try:
                        results[tool_id] = future.result(timeout=30)
                    except Exception as e:
                        logger.error("Tool execution failed for {}: {}", extra={"tool_id": tool_id, "e": e})
                        results[tool_id] = f"Error: {str(e)}"
        
        return results
    
    async def _execute_single(self, tool_call: ToolCall) -> Any:
        """Execute a single tool call"""
        tool_name = tool_call.get('name', 'unknown')
        tool_args = tool_call.get('args', {})
        
        if tool_name in self.tools:
            try:
                tool = self.tools[tool_name]
                result = await tool.ainvoke(tool_args)
                return result
            except Exception as e:
                logger.error("Tool {} failed: {}", extra={"tool_name": tool_name, "e": e})
                return f"Error: {str(e)}"
        else:
            return f"Tool {tool_name} not found"
    
    def _execute_single_sync(self, tool_call: ToolCall) -> Any:
        """Execute a single tool call synchronously"""
        tool_name = tool_call.get('name', 'unknown')
        tool_args = tool_call.get('args', {})
        
        if tool_name in self.tools:
            try:
                tool = self.tools[tool_name]
                result = tool.invoke(tool_args)
                return result
            except Exception as e:
                logger.error("Tool {} failed: {}", extra={"tool_name": tool_name, "e": e})
                return f"Error: {str(e)}"
        else:
            return f"Tool {tool_name} not found"

class EnhancedLangChainAgent:
    """Optimized LangChain agent with advanced features"""
    
    def __init__(self, llm: LLM, tools: List[BaseTool]) -> None:
        self.llm = llm
        self.tools = tools
        self.tool_executor = ParallelToolExecutor(tools)
        
        # Use summary buffer memory for long conversations
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=2000,
            return_messages=True
        )
        
        # Custom prompt optimization
        self.system_prompt = PromptTemplate(
            input_variables=["context", "question", "tools"],
            template="""You are an expert GAIA agent. 

Context from previous conversation:
{context}

Available tools:
{tools}

Question: {question}

Instructions:
1. Analyze the question type
2. Select appropriate tools
3. Execute step-by-step
4. Verify results
5. Provide ONLY the final answer

Answer:"""
        )
        
        # Create optimized chain
        self.chain = self.create_optimized_chain()
    
    def create_optimized_chain(self) -> Any:
        """Create chain with streaming and callbacks"""
        return LLMChain(
            llm=self.llm,
            prompt=self.system_prompt,
            memory=self.memory,
            callbacks=[
                StreamingStdOutCallbackHandler(),
                CustomMetricsCallback(),
                ErrorRecoveryCallback()
            ]
        )
    
    async def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the enhanced agent"""
        try:
            # Add tools to context
            tools_info = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
            inputs["tools"] = tools_info
            
            # Run the chain
            result = await self.chain.arun(inputs)
            
            return {
                "result": result,
                "memory_summary": self.get_memory_summary()
            }
            
        except Exception as e:
            logger.error("Enhanced agent error: {}", extra={"e": e})
            return {
                "error": str(e),
                "memory_summary": self.get_memory_summary()
            }
    
    def get_memory_summary(self) -> str:
        """Get memory summary for debugging"""
        try:
            return self.memory.moving_summary_buffer
        except:
            return "Memory summary not available"

def initialize_enhanced_agent(llm, tools: List[BaseTool]) -> Any:
    """Initialize enhanced LangChain agent"""
    return EnhancedLangChainAgent(llm, tools)

# Export for compatibility
enhanced_agent = initialize_enhanced_agent 