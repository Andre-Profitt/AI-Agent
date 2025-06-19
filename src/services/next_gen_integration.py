"""
Next-Generation Agent Integration Module
Integrates all advanced features into the FSM agent
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import new modules
from src.query_classifier import QueryClassifier, QueryCategory, OperationalParameters
from src.meta_cognition import MetaCognition, MetaCognitiveRouter
from src.tools_interactive import (
from typing import Optional, Dict, Any, List, Union, Tuple
    interactive_state, 
    get_interactive_tools,
    clarification_tracker
)
from src.tools_introspection import (
    ToolIntrospector,
    INTROSPECTION_TOOLS
)
from src.database_extended import ExtendedDatabase

# Import existing modules
from src.advanced_agent_fsm import FSMReActAgent, EnhancedAgentState
from src.tools_enhanced import get_enhanced_tools

logger = logging.getLogger(__name__)


class NextGenFSMAgent(FSMReActAgent):
    """
    Enhanced FSM Agent with all next-generation features:
    - Query classification and dynamic parameters
    - Meta-cognitive self-awareness
    - Interactive tools and clarification
    - Tool introspection and self-correction
    - Persistent learning and metrics
    """
    
    def __init__(
        self,
        tools: list = None,
        use_query_classification: bool = True,
        use_meta_cognition: bool = True,
        use_interactive_tools: bool = True,
        use_tool_introspection: bool = True,
        use_persistent_learning: bool = True,
        **kwargs
    ):
        """
        Initialize the next-gen agent with optional features
        
        Args:
            tools: Base tools to use
            use_query_classification: Enable dynamic parameter adjustment
            use_meta_cognition: Enable self-awareness of capabilities
            use_interactive_tools: Enable clarification and approval tools
            use_tool_introspection: Enable self-correction on tool errors
            use_persistent_learning: Enable tool performance tracking
        """
        # Initialize components
        self.query_classifier = QueryClassifier() if use_query_classification else None
        self.meta_cognition = MetaCognition() if use_meta_cognition else None
        self.meta_router = MetaCognitiveRouter(self.meta_cognition) if use_meta_cognition else None
        self.extended_db = ExtendedDatabase() if use_persistent_learning else None
        
        # Get base tools
        if tools is None:
            tools = get_enhanced_tools()
        
        # Add interactive tools
        if use_interactive_tools:
            tools.extend(get_interactive_tools())
        
        # Add introspection tools
        if use_tool_introspection:
            tools.extend(INTROSPECTION_TOOLS)
            # Create tool registry for introspection
            self.tool_registry = {tool.name: tool for tool in tools}
            self.tool_introspector = ToolIntrospector(self.tool_registry)
        
        # Store feature flags
        self.use_query_classification = use_query_classification
        self.use_meta_cognition = use_meta_cognition
        self.use_interactive_tools = use_interactive_tools
        self.use_tool_introspection = use_tool_introspection
        self.use_persistent_learning = use_persistent_learning
        
        # Initialize parent with enhanced tools
        super().__init__(tools=tools, **kwargs)
        
        logger.info(
            f"NextGenFSMAgent initialized with features: "
            f"query_classification={use_query_classification}, "
            f"meta_cognition={use_meta_cognition}, "
            f"interactive_tools={use_interactive_tools}, "
            f"tool_introspection={use_tool_introspection}, "
            f"persistent_learning={use_persistent_learning}"
        )
    
    def run(self, inputs: dict) -> Any:
        """
        Enhanced run method with all next-gen features
        """
        query = inputs.get("query", "")
        correlation_id = inputs.get("correlation_id", str(datetime.now().timestamp()))
        
        logger.info("NextGen agent processing query: {}...", extra={"query_": query[})
        
        # Step 1: Query Classification (if enabled)
        if self.use_query_classification and self.query_classifier:
            classification, params = self.query_classifier.classify_query(query)
            logger.info(
                f"Query classified as: {classification.category} "
                f"(confidence: {classification.confidence})"
            )
            
            # Apply dynamic parameters
            self._apply_operational_parameters(params)
            
            # Store classification in inputs
            inputs["query_classification"] = classification
            inputs["operational_params"] = params
        
        # Step 2: Meta-Cognitive Assessment (if enabled)
        if self.use_meta_cognition and self.meta_router:
            available_tools = [tool.name for tool in self.tools]
            should_use_tools, meta_score = self.meta_router.should_enter_tool_loop(
                query, available_tools
            )
            
            logger.info(
                f"Meta-cognitive assessment - Confidence: {meta_score.confidence}, "
                f"Should use tools: {should_use_tools}"
            )
            
            # Store meta-cognitive assessment
            inputs["meta_cognitive_score"] = meta_score
            inputs["should_use_tools"] = should_use_tools
        
        # Step 3: Set up interactive callbacks (if enabled)
        if self.use_interactive_tools:
            self._setup_interactive_callbacks()
        
        # Step 4: Run the base FSM with enhancements
        result = super().run(inputs)
        
        # Step 5: Track tool performance (if enabled)
        if self.use_persistent_learning and self.extended_db:
            self._track_tool_performance(result)
        
        # Step 6: Learn from clarifications (if enabled)
        if self.use_interactive_tools:
            self._learn_from_clarifications(query, result)
        
        return result
    
    def _apply_operational_parameters(self, params: OperationalParameters) -> Any:
        """Apply dynamic operational parameters from query classification"""
        # Update model preference
        if hasattr(params, 'model_preference'):
            self.model_preference = params.model_preference
            logger.info("Updated model preference to: {}", extra={"params_model_preference": params.model_preference})
        
        # Update verification level
        if hasattr(params, 'verification_level'):
            self.verification_level = params.verification_level
            logger.info("Updated verification level to: {}", extra={"params_verification_level": params.verification_level})
        
        # Update max reasoning steps
        if hasattr(params, 'max_reasoning_steps'):
            self.max_reasoning_steps = params.max_reasoning_steps
            logger.info("Updated max reasoning steps to: {}", extra={"params_max_reasoning_steps": params.max_reasoning_steps})
    
    def _setup_interactive_callbacks(self) -> Any:
        """Set up callbacks for interactive tools"""
        # This would be connected to the UI layer
        # For now, log that callbacks would be set up
        logger.info("Interactive tool callbacks would be set up here")
    
    def _track_tool_performance(self, result: dict) -> Any:
        """Track tool performance metrics for learning"""
        if not self.extended_db:
            return
        
        # Extract tool calls from result
        tool_calls = result.get("tool_calls", [])
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool_name")
            success = not bool(tool_call.get("error"))
            # Calculate latency (would need timing in actual implementation)
            latency_ms = 100.0  # Placeholder
            error_msg = tool_call.get("error")
            
            # Update metrics in database
            self.extended_db.update_tool_metric(
                tool_name=tool_name,
                success=success,
                latency_ms=latency_ms,
                error_message=error_msg
            )
            
            logger.debug(
                f"Tracked tool performance: {tool_name}, "
                f"success={success}, latency={latency_ms}ms"
            )
    
    def _learn_from_clarifications(self, original_query: str, result: dict) -> Any:
        """Learn from clarification patterns"""
        # Check if clarification tool was used
        tool_calls = result.get("tool_calls", [])
        
        for tool_call in tool_calls:
            if tool_call.get("tool_name") == "ask_user_for_clarification":
                question = tool_call.get("tool_input", {}).get("question")
                response = tool_call.get("output")
                
                if question and response and self.extended_db:
                    # Get query category from classification
                    category = "general"
                    if hasattr(self, '_last_classification'):
                        category = self._last_classification.category.value
                    
                    # Store clarification pattern
                    self.extended_db.add_clarification_pattern(
                        original_query=original_query,
                        clarification_question=question,
                        user_response=response,
                        query_category=category
                    )
                    
                    logger.info(
                        f"Learned clarification pattern: {question[:50]}... -> {response[:50]}..."
                    )
    
    def handle_tool_error(self, tool_name: str, error: Exception, attempted_params: dict) -> Optional[dict]:
        """
        Enhanced error handling with introspection and self-correction
        
        Returns:
            Corrected parameters if possible, None otherwise
        """
        if not self.use_tool_introspection or not self.tool_introspector:
            return None
        
        logger.info("Attempting self-correction for tool error: {}", extra={"tool_name": tool_name})
        
        # Analyze the error
        error_analysis = self.tool_introspector.analyze_tool_error(
            tool_name=tool_name,
            error_message=str(error),
            attempted_params=attempted_params
        )
        
        # Log the analysis
        logger.info(
            f"Error analysis - Type: {error_analysis.error_type}, "
            f"Suggestion: {error_analysis.suggestion}"
        )
        
        # Attempt to correct based on suggestion
        if error_analysis.error_type == "parameter_error":
            # Get tool schema
            schema = self.tool_introspector.get_tool_schema(tool_name)
            
            if "error" not in schema:
                # Try to build corrected parameters
                corrected_params = self._build_corrected_params(
                    schema, attempted_params, error_analysis
                )
                
                if corrected_params:
                    logger.info("Self-corrected parameters: {}", extra={"corrected_params": corrected_params})
                    return corrected_params
        
        return None
    
    def _build_corrected_params(
        self,
        schema: dict,
        attempted_params: dict,
        error_analysis: Any
    ) -> Optional[dict]:
        """Build corrected parameters based on schema and error analysis"""
        corrected = {}
        
        # Add required parameters
        for param in schema.get("required_parameters", []):
            if param in attempted_params:
                corrected[param] = attempted_params[param]
            else:
                # Try to infer from schema
                param_info = schema["parameters"].get(param, {})
                if param_info.get("type") == "string":
                    corrected[param] = ""  # Placeholder
                elif param_info.get("type") == "integer":
                    corrected[param] = 0
                elif param_info.get("type") == "boolean":
                    corrected[param] = False
        
        # Remove unknown parameters
        known_params = set(schema.get("parameters", {}).keys())
        for param in list(corrected.keys()):
            if param not in known_params:
                del corrected[param]
        
        return corrected if corrected else None
    
    def suggest_clarification(self, query: str) -> Optional[str]:
        """
        Suggest a clarification question based on past patterns
        
        Args:
            query: The user's query
            
        Returns:
            Suggested clarification question or None
        """
        if not self.extended_db:
            return None
        
        # Find similar clarification patterns
        similar_patterns = self.extended_db.find_similar_clarifications(query)
        
        if similar_patterns:
            # Use the most frequent pattern
            best_pattern = similar_patterns[0]
            logger.info(
                f"Found similar clarification pattern: "
                f"{best_pattern.clarification_question}"
            )
            return best_pattern.clarification_question
        
        return None
    
    def get_tool_recommendations(self, task_description: str) -> List[str]:
        """
        Get tool recommendations based on reliability metrics
        
        Args:
            task_description: Description of the task
            
        Returns:
            List of recommended tool names
        """
        if not self.extended_db:
            return []
        
        # Get all tool metrics
        metrics = self.extended_db.get_tool_metrics()
        
        # Sort by reliability score
        sorted_tools = sorted(
            metrics,
            key=lambda m: m.reliability_score,
            reverse=True
        )
        
        # Return top tools
        recommendations = [tool.tool_name for tool in sorted_tools[:5]]
        logger.info("Tool recommendations: {}", extra={"recommendations": recommendations})
        
        return recommendations


# Integration helper functions

def create_next_gen_agent(
    enable_all_features: bool = True,
    **kwargs
) -> NextGenFSMAgent:
    """
    Create a next-gen agent with specified features
    
    Args:
        enable_all_features: Enable all next-gen features
        **kwargs: Individual feature flags and other parameters
        
    Returns:
        Configured NextGenFSMAgent instance
    """
    if enable_all_features:
        return NextGenFSMAgent(
            use_query_classification=True,
            use_meta_cognition=True,
            use_interactive_tools=True,
            use_tool_introspection=True,
            use_persistent_learning=True,
            **kwargs
        )
    else:
        return NextGenFSMAgent(**kwargs)


def setup_interactive_ui_callbacks(agent: NextGenFSMAgent, ui_callbacks: dict) -> None:
    """
    Connect UI callbacks to the agent's interactive tools
    
    Args:
        agent: The NextGenFSMAgent instance
        ui_callbacks: Dictionary of UI callback functions
    """
    if agent.use_interactive_tools:
        # Set clarification callback
        if "clarification" in ui_callbacks:
            interactive_state.set_clarification_callback(ui_callbacks["clarification"])
        
        # Set approval callback
        if "approval" in ui_callbacks:
            interactive_state.approval_callback = ui_callbacks["approval"]
        
        # Set feedback callback
        if "feedback" in ui_callbacks:
            interactive_state.feedback_callback = ui_callbacks["feedback"]
        
        logger.info("Interactive UI callbacks configured")


# Example usage
if __name__ == "__main__":
    # Create a fully-featured next-gen agent
    agent = create_next_gen_agent(enable_all_features=True)
    
    # Test with a sample query
    result = agent.run({
        "query": "What is the latest news about AI developments in 2024?"
    })
    
    logger.info("Result: {}", extra={"result_get__final_answer____No_answer_generated__": result.get('final_answer', 'No answer generated')}) 