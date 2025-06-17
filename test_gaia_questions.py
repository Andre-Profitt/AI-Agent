#!/usr/bin/env python3
"""
GAIA Question Testing Framework for Advanced ReAct Agent
Tests sophisticated reasoning capabilities across multiple domains.
"""

import asyncio
import json
import time
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class GAIAQuestion:
    """Represents a GAIA-style test question."""
    id: str
    question: str
    expected_answer: str
    category: str  # "factual", "mathematical", "chess", "multimedia", "temporal"
    difficulty: str  # "easy", "medium", "hard"
    requires_tools: List[str]
    reasoning_steps: Optional[int] = None
    time_limit: Optional[int] = None

@dataclass
class TestResult:
    """Results from testing a single question."""
    question_id: str
    question: str
    expected_answer: str
    agent_answer: str
    correct: bool
    confidence: float
    reasoning_steps: int
    execution_time: float
    tools_used: List[str]
    error_messages: List[str]
    timestamp: datetime

class GAIATestSuite:
    """Comprehensive test suite for GAIA-style questions."""
    
    def __init__(self):
        self.questions = self._create_test_questions()
        self.results: List[TestResult] = []
        
    def _create_test_questions(self) -> List[GAIAQuestion]:
        """Create a comprehensive set of GAIA-style test questions."""
        return [
            # Factual Questions
            GAIAQuestion(
                id="fact_001",
                question="How many studio albums did Mercedes Sosa release between 2000 and 2009?",
                expected_answer="0",
                category="factual",
                difficulty="medium",
                requires_tools=["web_researcher", "semantic_search_tool"],
                reasoning_steps=3,
                time_limit=120
            ),
            
            GAIAQuestion(
                id="fact_002", 
                question="What is the three-letter country code for Egypt according to ISO 3166-1 alpha-3?",
                expected_answer="EGY",
                category="factual",
                difficulty="easy",
                requires_tools=["web_researcher"],
                reasoning_steps=2,
                time_limit=60
            ),
            
            # Mathematical Questions
            GAIAQuestion(
                id="math_001",
                question="What is 1729 divided by 7?",
                expected_answer="247",
                category="mathematical", 
                difficulty="easy",
                requires_tools=["python_interpreter"],
                reasoning_steps=2,
                time_limit=30
            ),
            
            GAIAQuestion(
                id="math_002",
                question="If a compound grows at 12% annually, what will $1000 become after 5 years? (Round to nearest dollar)",
                expected_answer="1762",
                category="mathematical",
                difficulty="medium",
                requires_tools=["python_interpreter"],
                reasoning_steps=3,
                time_limit=60
            ),
            
            # Chess Questions  
            GAIAQuestion(
                id="chess_001",
                question="In a chess position where White has Rook on d1, Black King on h8, what is the best move for White?",
                expected_answer="Rd8+",
                category="chess",
                difficulty="medium", 
                requires_tools=["image_analyzer", "web_researcher"],
                reasoning_steps=4,
                time_limit=180
            ),
            
            # Temporal/Historical Questions
            GAIAQuestion(
                id="temporal_001",
                question="How many years passed between the fall of the Berlin Wall and the 9/11 attacks?",
                expected_answer="11", 
                category="temporal",
                difficulty="medium",
                requires_tools=["web_researcher", "python_interpreter"],
                reasoning_steps=3,
                time_limit=90
            ),
            
            # Complex Research Questions
            GAIAQuestion(
                id="research_001",
                question="What was the population of Tokyo in the year the Nintendo Game Boy was first released?",
                expected_answer="11855563",
                category="factual",
                difficulty="hard",
                requires_tools=["web_researcher", "semantic_search_tool"],
                reasoning_steps=5,
                time_limit=240
            ),
            
            # Multimedia Questions (hypothetical - would need actual files)
            GAIAQuestion(
                id="multimedia_001", 
                question="What is the dominant color in the provided image?",
                expected_answer="blue",
                category="multimedia",
                difficulty="easy",
                requires_tools=["image_analyzer"],
                reasoning_steps=2,
                time_limit=60
            ),
            
            # Cross-domain Verification Questions
            GAIAQuestion(
                id="verify_001",
                question="What is the surname of the current President of France (as of 2024)?",
                expected_answer="Macron",
                category="factual", 
                difficulty="easy",
                requires_tools=["web_researcher", "tavily_search"],
                reasoning_steps=2,
                time_limit=60
            ),
            
            # Multi-step Reasoning
            GAIAQuestion(
                id="multistep_001",
                question="If the Eiffel Tower was built in 1889, and the Statue of Liberty was dedicated 3 years earlier, in what year was the Statue of Liberty dedicated?",
                expected_answer="1886",
                category="mathematical",
                difficulty="medium",
                requires_tools=["web_researcher", "python_interpreter"],
                reasoning_steps=4,
                time_limit=120
            )
        ]
    
    async def test_agent(self, agent, max_concurrent=3) -> Dict[str, Any]:
        """Test the agent against all GAIA questions with detailed analysis."""
        print("🧪 Starting GAIA Test Suite")
        print(f"📊 Testing {len(self.questions)} questions across {len(set(q.category for q in self.questions))} categories")
        print("=" * 80)
        
        # Run tests in batches to avoid overwhelming the system
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [self._test_single_question(agent, question, semaphore) for question in self.questions]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Process results
        successful_results = [r for r in results if isinstance(r, TestResult)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        # Analyze performance
        analysis = self._analyze_results(successful_results, total_time)
        
        # Print detailed results
        self._print_detailed_results(successful_results, failed_results, analysis)
        
        return {
            "results": successful_results,
            "failed": failed_results,
            "analysis": analysis,
            "total_time": total_time
        }
    
    async def _test_single_question(self, agent, question: GAIAQuestion, semaphore) -> TestResult:
        """Test a single question with detailed tracking."""
        async with semaphore:
            print(f"\n🔍 Testing {question.id}: {question.question[:50]}...")
            
            start_time = time.time()
            tools_used = []
            error_messages = []
            reasoning_steps = 0
            confidence = 0.0
            
            try:
                # Prepare agent input
                agent_input = {
                    "messages": [{"role": "user", "content": question.question}],
                    "run_id": uuid.uuid4(),
                    "log_to_db": False
                }
                
                # Track agent execution
                agent_response = None
                if hasattr(agent, 'stream'):
                    # Stream execution for detailed tracking
                    for chunk in agent.stream(agent_input):
                        reasoning_steps += 1
                        last_message = chunk.get("messages", [])[-1] if chunk.get("messages") else None
                        
                        if last_message:
                            # Track tool usage
                            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                                for tool_call in last_message.tool_calls:
                                    tools_used.append(tool_call.get('name', 'unknown'))
                            
                            # Update confidence
                            confidence = chunk.get("confidence", confidence)
                            
                            # Check for completion
                            if chunk.get("reasoning_complete", False):
                                agent_response = last_message.content if hasattr(last_message, 'content') else str(last_message)
                                break
                else:
                    # Fallback for non-streaming agents
                    result = agent.run(agent_input)
                    agent_response = result.get("messages", [])[-1].content if result.get("messages") else "No response"
                    reasoning_steps = 1
                
                execution_time = time.time() - start_time
                
                # Clean up agent response
                if agent_response:
                    agent_response = self._extract_final_answer(str(agent_response))
                else:
                    agent_response = "No response generated"
                    error_messages.append("Agent failed to generate response")
                
                # Check correctness
                correct = self._check_answer_correctness(agent_response, question.expected_answer)
                
                # Create result
                result = TestResult(
                    question_id=question.id,
                    question=question.question,
                    expected_answer=question.expected_answer,
                    agent_answer=agent_response,
                    correct=correct,
                    confidence=confidence,
                    reasoning_steps=reasoning_steps,
                    execution_time=execution_time,
                    tools_used=list(set(tools_used)),
                    error_messages=error_messages,
                    timestamp=datetime.now()
                )
                
                # Print immediate result
                status = "✅ PASS" if correct else "❌ FAIL"
                print(f"   {status} | Expected: '{question.expected_answer}' | Got: '{agent_response}' | Time: {execution_time:.1f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_messages.append(str(e))
                
                print(f"   💥 ERROR | {str(e)[:50]}... | Time: {execution_time:.1f}s")
                
                return TestResult(
                    question_id=question.id,
                    question=question.question,
                    expected_answer=question.expected_answer,
                    agent_answer=f"ERROR: {str(e)}",
                    correct=False,
                    confidence=0.0,
                    reasoning_steps=reasoning_steps,
                    execution_time=execution_time,
                    tools_used=tools_used,
                    error_messages=error_messages,
                    timestamp=datetime.now()
                )
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract clean final answer from agent response."""
        if not response:
            return ""
        
        # Remove common prefixes
        prefixes = ["final answer:", "the answer is:", "answer:", "result:", "therefore:"]
        response_lower = response.lower().strip()
        
        for prefix in prefixes:
            if prefix in response_lower:
                idx = response_lower.rfind(prefix)
                if idx != -1:
                    extracted = response[idx + len(prefix):].strip()
                    if extracted:
                        response = extracted
                        break
        
        # Clean up
        response = response.strip()
        response = response.strip('.,!?')
        
        # If multiline, take the last non-empty line
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            response = lines[-1]
        
        return response
    
    def _check_answer_correctness(self, agent_answer: str, expected_answer: str) -> bool:
        """Check if agent answer matches expected answer with smart comparison."""
        if not agent_answer or not expected_answer:
            return False
        
        # Normalize both answers
        agent_norm = agent_answer.lower().strip()
        expected_norm = expected_answer.lower().strip()
        
        # Exact match
        if agent_norm == expected_norm:
            return True
        
        # Numeric comparison for math questions
        try:
            agent_num = float(agent_norm)
            expected_num = float(expected_norm)
            return abs(agent_num - expected_num) < 0.01  # Small tolerance for floating point
        except (ValueError, TypeError):
            pass
        
        # Partial match for longer answers
        if len(expected_norm) > 5:
            return expected_norm in agent_norm or agent_norm in expected_norm
        
        return False
    
    def _analyze_results(self, results: List[TestResult], total_time: float) -> Dict[str, Any]:
        """Analyze test results comprehensively."""
        if not results:
            return {"error": "No successful test results"}
        
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r.correct)
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        # Category analysis
        category_stats = {}
        for result in results:
            question = next((q for q in self.questions if q.id == result.question_id), None)
            if question:
                category = question.category
                if category not in category_stats:
                    category_stats[category] = {"total": 0, "correct": 0}
                category_stats[category]["total"] += 1
                if result.correct:
                    category_stats[category]["correct"] += 1
        
        # Performance metrics
        avg_time = sum(r.execution_time for r in results) / len(results)
        avg_steps = sum(r.reasoning_steps for r in results) / len(results)
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        # Tool usage analysis
        all_tools = []
        for result in results:
            all_tools.extend(result.tools_used)
        
        tool_usage = {}
        for tool in set(all_tools):
            tool_usage[tool] = all_tools.count(tool)
        
        return {
            "overall_accuracy": accuracy,
            "correct_answers": correct_answers,
            "total_questions": total_questions,
            "category_performance": {
                cat: stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                for cat, stats in category_stats.items()
            },
            "avg_execution_time": avg_time,
            "avg_reasoning_steps": avg_steps,
            "avg_confidence": avg_confidence,
            "tool_usage": tool_usage,
            "total_execution_time": total_time
        }
    
    def _print_detailed_results(self, results: List[TestResult], failed: List[Exception], analysis: Dict[str, Any]):
        """Print comprehensive test results."""
        print("\n" + "=" * 80)
        print("📊 GAIA TEST RESULTS SUMMARY")
        print("=" * 80)
        
        # Overall performance
        accuracy = analysis.get("overall_accuracy", 0)
        print(f"🎯 Overall Accuracy: {accuracy:.1%} ({analysis.get('correct_answers', 0)}/{analysis.get('total_questions', 0)})")
        print(f"⏱️  Average Time: {analysis.get('avg_execution_time', 0):.1f}s")
        print(f"🧠 Average Steps: {analysis.get('avg_reasoning_steps', 0):.1f}")
        print(f"📈 Average Confidence: {analysis.get('avg_confidence', 0):.1%}")
        
        # Category breakdown
        print(f"\n📋 Category Performance:")
        for category, accuracy in analysis.get("category_performance", {}).items():
            print(f"   {category.title()}: {accuracy:.1%}")
        
        # Tool usage
        print(f"\n🛠️  Tool Usage:")
        for tool, count in sorted(analysis.get("tool_usage", {}).items(), key=lambda x: x[1], reverse=True):
            print(f"   {tool}: {count} times")
        
        # Failed tests
        if failed:
            print(f"\n💥 Failed Tests: {len(failed)}")
            for i, error in enumerate(failed[:3]):  # Show first 3 errors
                print(f"   {i+1}. {str(error)[:100]}...")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if accuracy < 0.5:
            print("   🔴 Critical: Accuracy below 50% - fundamental reasoning issues")
        elif accuracy < 0.7:
            print("   🟡 Moderate: Accuracy below 70% - needs refinement")
        else:
            print("   🟢 Good: Accuracy above 70% - fine-tuning recommended")
        
        low_categories = [cat for cat, acc in analysis.get("category_performance", {}).items() if acc < 0.5]
        if low_categories:
            print(f"   📌 Focus improvement on: {', '.join(low_categories)}")
        
        print("=" * 80)

def test_agent_with_gaia():
    """Main function to test an agent with GAIA questions."""
    print("🚀 GAIA Agent Testing Framework")
    print("Testing sophisticated reasoning capabilities...\n")
    
    # Initialize test suite
    test_suite = GAIATestSuite()
    
    # Note: This is a demonstration framework
    # To actually test, you would pass your agent instance:
    # results = await test_suite.test_agent(your_agent_instance)
    
    print("📝 Test Suite Created Successfully!")
    print(f"📊 Questions: {len(test_suite.questions)}")
    print(f"📂 Categories: {len(set(q.category for q in test_suite.questions))}")
    print(f"🎯 Difficulty Levels: {len(set(q.difficulty for q in test_suite.questions))}")
    
    # Show sample questions
    print("\n🔍 Sample Questions:")
    for i, question in enumerate(test_suite.questions[:3]):
        print(f"   {i+1}. [{question.category.title()}] {question.question}")
        print(f"      Expected: {question.expected_answer}")
    
    print("\n✅ Ready for agent testing!")
    print("   To run: test_suite.test_agent(your_agent)")

if __name__ == "__main__":
    test_agent_with_gaia() 