import operator
import logging
import time
import random
from typing import Annotated, List, TypedDict
from uuid import UUID

from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Configure logging
logger = logging.getLogger(__name__)

# --- Enhanced Agent State Definition ---

class EnhancedAgentState(TypedDict):
    """
    Enhanced agent state for sophisticated reasoning.

    Attributes:
        messages: The history of messages in the conversation.
        run_id: A unique identifier for the agent run.
        log_to_db: A flag to control database logging for this run.
        plan: Current strategic plan for solving the problem.
        step_count: Number of reasoning steps taken.
        confidence: Confidence level in current answer (0-1).
        reflection_notes: Self-reflection on progress and potential issues.
        sub_goals: List of sub-goals to accomplish.
        completed_goals: List of completed sub-goals.
        error_history: Track of errors encountered and how they were handled.
    """
    messages: Annotated[List[AnyMessage], operator.add]
    run_id: UUID
    log_to_db: bool
    plan: str
    step_count: int
    confidence: float
    reflection_notes: str
    sub_goals: List[str]
    completed_goals: List[str]
    error_history: List[str]

# --- World-Class ReAct Agent Implementation ---

class WorldClassReActAgent:
    """
    A sophisticated ReAct agent with advanced planning, reflection, 
    and multi-step reasoning capabilities for complex GAIA questions.
    """
    def __init__(self, tools: list, log_handler: logging.Handler = None):
        self.tools = tools
        self.log_handler = log_handler
        self.graph = self._build_graph()
        self.max_reasoning_steps = 20  # Allow for complex multi-step problems

    def _get_llm(self):
        """Initializes and returns the Groq LLM optimized for complex reasoning."""
        return ChatGroq(
            temperature=0.2,  # Slight temperature for creative problem solving
            model_name="llama-3.1-8b-instant",
            max_tokens=3072,  # Increased for detailed reasoning
            max_retries=1,
            request_timeout=90  # Longer timeout for complex reasoning
        )
    
    def _get_system_prompt(self):
        """Returns a sophisticated system prompt for world-class ReAct reasoning."""
        return """You are Orion, a world-class AI research assistant with sophisticated reasoning capabilities.

🎯 MISSION: Solve complex, multi-step problems through systematic reasoning, strategic planning, and intelligent tool use.

🧠 SOPHISTICATED REASONING FRAMEWORK:

**PHASE 1: DEEP ANALYSIS**
- Understand the question thoroughly
- Identify the type of problem (research, calculation, analysis, etc.)
- Recognize implicit requirements and constraints
- Consider what makes this question challenging

**PHASE 2: STRATEGIC PLANNING**
- Break down into logical sub-problems
- Map out information dependencies
- Plan tool usage sequence
- Identify potential failure points and alternatives
- Set measurable success criteria

**PHASE 3: SYSTEMATIC EXECUTION**
- Execute plan step-by-step
- Validate each result before proceeding
- Cross-reference information from multiple sources
- Handle errors gracefully with alternative approaches

**PHASE 4: CONTINUOUS REFLECTION**
- Monitor progress against plan
- Assess confidence in current approach
- Identify when to pivot or adjust strategy
- Learn from failures and adapt

**PHASE 5: SYNTHESIS & VERIFICATION**
- Integrate all gathered information
- Verify answer against original question
- Check for logical consistency
- Ensure completeness

🛠️ SOPHISTICATED TOOL ORCHESTRATION:

**For Research Questions:**
1. web_researcher (Wikipedia) → specific topic research
2. tavily_search → current information, verification
3. semantic_search_tool → knowledge base queries

**For Multimedia Analysis:**
1. video_analyzer → YouTube content analysis
2. audio_transcriber → extract speech/audio content
3. image_analyzer → visual content analysis

**For Data Processing:**
1. file_reader → text files, CSV data
2. advanced_file_reader → Excel, PDF, Word documents
3. python_interpreter → calculations, data manipulation

**For Complex Problems:**
- Chain multiple tools systematically
- Use earlier results to inform later tool choices
- Validate findings across different sources
- Build evidence step-by-step

🔍 ADVANCED REFLECTION TRIGGERS:
- "Am I answering the right question?"
- "Do I have sufficient evidence?"
- "Are there contradictions in my findings?"
- "What assumptions am I making?"
- "How confident am I in each piece of information?"
- "What could I be missing?"
- "Should I verify this from another source?"

⚡ EXECUTION EXCELLENCE:
- **Be methodical**: Follow your plan systematically
- **Be adaptive**: Adjust when you hit obstacles
- **Be thorough**: Don't skip verification steps
- **Be efficient**: Don't repeat unnecessary work
- **Be precise**: Distinguish between facts and inferences

🎯 GAIA-LEVEL PERFORMANCE STANDARDS:
- Handle multi-step reasoning chains
- Process multimedia content accurately
- Cross-reference historical/temporal information
- Perform mathematical and logical analysis
- Navigate complex file formats
- Validate information across sources

⚠️ CRITICAL SUCCESS FACTORS:
1. **Plan before acting** - Don't jump into tool use
2. **Validate everything** - Verify critical information
3. **Think incrementally** - Build understanding step by step
4. **Handle failures gracefully** - Try alternative approaches
5. **Be complete** - Address all aspects of the question

🎯 FINAL OUTPUT RULE:
When you have the complete, verified answer, respond with ONLY the direct answer. No explanations, no reasoning steps, no prefixes.

EXAMPLES:
- Question: "How many studio albums?" → Response: "3"
- Question: "What chess move guarantees win?" → Response: "Rd5"
- Question: "Which country IOC code?" → Response: "EGY"
- Question: "What NASA award number?" → Response: "NNX15AK56G"

Remember: Excellence through systematic thinking, strategic planning, and rigorous execution."""

    def _build_graph(self):
        """
        Builds a sophisticated LangGraph with planning, reasoning, and reflection.
        """
        llm = self._get_llm()
        model_with_tools = llm.bind_tools(self.tools)

        # Planning Node - Strategic problem decomposition
        def planning_node(state: EnhancedAgentState):
            """Strategic planning node for complex multi-step problems."""
            messages = state["messages"]
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self._get_system_prompt())] + messages

            # Enhanced planning prompt for GAIA-level questions
            planning_prompt = f"""
STRATEGIC PLANNING PHASE for this question:

Analyze this question and create a comprehensive plan:

1. **QUESTION TYPE ANALYSIS:**
   - What type of problem is this? (research, calculation, multimedia analysis, etc.)
   - What makes this question challenging?
   - What are the key requirements and constraints?

2. **INFORMATION REQUIREMENTS:**
   - What specific information do I need to find?
   - Are there temporal constraints (dates, time periods)?
   - Do I need to cross-reference multiple sources?

3. **TOOL STRATEGY:**
   - Which tools should I use and in what sequence?
   - How will I validate the information I gather?
   - What are backup approaches if primary tools fail?

4. **SUB-GOAL DECOMPOSITION:**
   - Break this into 3-5 specific sub-goals
   - Define success criteria for each sub-goal
   - Identify dependencies between sub-goals

5. **RISK ASSESSMENT:**
   - What could go wrong with this approach?
   - What are alternative strategies?
   - How will I verify my final answer?

Provide your detailed plan, then proceed with systematic execution.
Current step: {state.get('step_count', 0)}
"""
            
            if not state.get("plan"):
                messages.append(SystemMessage(content=planning_prompt))

            response = model_with_tools.invoke(messages)
            
            # Extract structured information from planning response
            plan_content = response.content
            
            # Extract sub-goals if mentioned in the response
            sub_goals = []
            if "sub-goal" in plan_content.lower() or "step" in plan_content.lower():
                lines = plan_content.split('\n')
                for line in lines:
                    if any(keyword in line.lower() for keyword in ["step", "goal", "1.", "2.", "3.", "4.", "5."]):
                        if len(line.strip()) > 0 and len(line) < 200:  # Reasonable sub-goal length
                            sub_goals.append(line.strip())

            return {
                "messages": [response],
                "plan": plan_content,
                "step_count": 1,
                "confidence": 0.3,  # Start with low confidence
                "reflection_notes": "Strategic planning completed",
                "sub_goals": sub_goals[:5],  # Limit to 5 sub-goals
                "completed_goals": [],
                "error_history": []
            }

        # Reasoning Node - Enhanced with reflection and adaptation
        def reasoning_node(state: EnhancedAgentState):
            """Enhanced reasoning with sophisticated reflection and adaptation."""
            messages = state["messages"]
            step_count = state.get("step_count", 0)
            
            # Add reflection context for complex problems
            if step_count > 3:
                reflection_context = f"""
🔍 REFLECTION CHECKPOINT (Step {step_count}):

**CURRENT STATUS:**
- Plan: {state.get('plan', 'No plan')[:200]}...
- Completed Goals: {len(state.get('completed_goals', []))} of {len(state.get('sub_goals', []))}
- Confidence: {state.get('confidence', 0.0):.1f}
- Errors Encountered: {len(state.get('error_history', []))}

**REFLECTION QUESTIONS:**
1. Am I making progress toward the answer?
2. Do I have enough information to be confident?
3. Should I verify my findings from another source?
4. Is my current approach still optimal?
5. What critical information might I be missing?

**DECISION POINT:**
Continue with systematic execution or provide final answer if you have sufficient confidence and completeness.
"""
                messages.append(SystemMessage(content=reflection_context))

            # Intelligent context management for long conversations
            if len(messages) > 20:
                # Keep system prompts, planning, and recent messages
                system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
                recent_msgs = messages[-12:]
                messages = system_msgs[:3] + recent_msgs  # Keep up to 3 system messages

            response = model_with_tools.invoke(messages)
            
            # Update reasoning state
            new_confidence = state.get("confidence", 0.0)
            
            # Increase confidence based on successful tool use and information gathering
            if "found" in response.content.lower() or "according" in response.content.lower():
                new_confidence = min(0.95, new_confidence + 0.15)
            elif response.tool_calls:
                new_confidence = min(0.9, new_confidence + 0.1)
            
            return {
                "messages": [response],
                "step_count": step_count + 1,
                "confidence": new_confidence,
                "reflection_notes": f"Step {step_count + 1} reasoning completed"
            }

        # Enhanced Tool Node with sophisticated error handling
        def enhanced_tool_node(state: EnhancedAgentState):
            """Enhanced tool execution with error recovery and learning."""
            tool_node = ToolNode(self.tools)
            
            try:
                tool_output = tool_node.invoke(state)
                
                # Log successful tool execution
                if state.get('log_to_db', False) and self.log_handler:
                    last_message = state['messages'][-1]
                    if last_message.tool_calls:
                        tool_name = last_message.tool_calls[0].get('name', 'unknown')
                        log_payload = {
                            "run_id": state['run_id'],
                            "step_type": "SUCCESSFUL_ACTION",
                            "payload": {
                                "tool_name": tool_name,
                                "step_count": state.get("step_count", 0),
                                "confidence": state.get("confidence", 0.0)
                            }
                        }
                        self.log_handler.emit(logging.makeLogRecord({'msg': log_payload}))

                # Update completed goals if tool execution was successful
                tool_result = tool_output['messages'][-1].content
                if not tool_result.startswith("Error") and len(tool_result) > 10:
                    completed_goals = state.get("completed_goals", [])
                    step_description = f"Step {state.get('step_count', 0)}: Tool execution successful"
                    if step_description not in completed_goals:
                        completed_goals.append(step_description)
                    
                    new_state = dict(tool_output)
                    new_state["completed_goals"] = completed_goals
                    return new_state

                return tool_output
                
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                
                # Add to error history and suggest recovery
                error_history = state.get("error_history", [])
                error_msg = f"Step {state.get('step_count', 0)}: {str(e)}"
                error_history.append(error_msg)
                
                recovery_message = f"""
TOOL EXECUTION ERROR: {str(e)}

RECOVERY STRATEGY:
1. Try an alternative tool if available
2. Modify the approach or parameters
3. Search for the information using a different method
4. If critical, acknowledge the limitation and proceed with available information

Continue with error recovery or alternative approaches.
"""
                
                return {
                    "messages": [SystemMessage(content=recovery_message)],
                    "error_history": error_history,
                    "confidence": max(0.1, state.get("confidence", 0.5) - 0.2)  # Reduce confidence after errors
                }

        # Sophisticated decision logic
        def should_continue(state: EnhancedAgentState):
            """Advanced decision logic for world-class performance."""
            last_message = state["messages"][-1]
            step_count = state.get("step_count", 0)
            confidence = state.get("confidence", 0.0)
            
            # Continue if there are tool calls to execute
            if last_message.tool_calls:
                return "tools"
            
            # Stop if max steps reached
            if step_count >= self.max_reasoning_steps:
                logger.warning(f"Max reasoning steps ({self.max_reasoning_steps}) reached")
                return END
            
            # Check for final answer indicators
            content = last_message.content.lower()
            final_indicators = [
                "final answer", "the answer is", "conclusion", "therefore",
                "result:", "solution:", "based on my analysis"
            ]
            
            # High confidence + final indicators = end
            if confidence > 0.8 and any(indicator in content for indicator in final_indicators):
                return END
            
            # Very high confidence regardless of indicators
            if confidence > 0.95:
                return END
                
            # Check if we've completed most sub-goals
            completed = len(state.get("completed_goals", []))
            total = len(state.get("sub_goals", []))
            if total > 0 and completed >= total * 0.8 and confidence > 0.7:
                return END
            
            # Continue reasoning
            return "reasoning"

        # Build sophisticated graph
        graph_builder = StateGraph(EnhancedAgentState)
        graph_builder.add_node("planning", planning_node)
        graph_builder.add_node("reasoning", reasoning_node)
        graph_builder.add_node("tools", enhanced_tool_node)

        # Sophisticated flow control
        graph_builder.set_entry_point("planning")
        graph_builder.add_edge("planning", "reasoning")
        graph_builder.add_conditional_edges(
            "reasoning",
            should_continue,
            {
                "tools": "tools",
                "reasoning": "reasoning",
                "end": END
            }
        )
        graph_builder.add_edge("tools", "reasoning")

        return graph_builder.compile()

    def run(self, inputs: dict):
        """Execute the world-class ReAct agent."""
        # Initialize enhanced state
        enhanced_inputs = dict(inputs)
        enhanced_inputs.setdefault("plan", "")
        enhanced_inputs.setdefault("step_count", 0)
        enhanced_inputs.setdefault("confidence", 0.0)
        enhanced_inputs.setdefault("reflection_notes", "")
        enhanced_inputs.setdefault("sub_goals", [])
        enhanced_inputs.setdefault("completed_goals", [])
        enhanced_inputs.setdefault("error_history", [])
            
        return self.graph.invoke(enhanced_inputs)

    def stream(self, inputs: dict):
        """Stream the sophisticated execution steps."""
        # Initialize enhanced state
        enhanced_inputs = dict(inputs)
        enhanced_inputs.setdefault("plan", "")
        enhanced_inputs.setdefault("step_count", 0)
        enhanced_inputs.setdefault("confidence", 0.0)
        enhanced_inputs.setdefault("reflection_notes", "")
        enhanced_inputs.setdefault("sub_goals", [])
        enhanced_inputs.setdefault("completed_goals", [])
        enhanced_inputs.setdefault("error_history", [])
            
        return self.graph.stream(enhanced_inputs) 