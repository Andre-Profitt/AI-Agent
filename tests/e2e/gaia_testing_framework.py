#!/usr/bin/env python3
"""
GAIA Testing Framework for Advanced ReAct Agent Performance
Comprehensive evaluation and iteration system for sophisticated reasoning.
"""

import asyncio
import json
import time
import uuid
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

@dataclass
class GAIAQuestion:
    """GAIA-style test question with metadata."""
    id: str
    question: str
    expected_answer: str
    category: str
    difficulty: str
    requires_tools: List[str]
    reasoning_steps: Optional[int] = None
    time_limit: Optional[int] = None

@dataclass
class AgentTestResult:
    """Comprehensive test result for a single question."""
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
    validation_score: float
    timestamp: datetime

class GAIATestSuite:
    """Advanced testing framework for GAIA-style evaluation."""
    
    def __init__(self):
        self.questions = self._create_comprehensive_test_set()
        self.results: List[AgentTestResult] = []
        
        # Add regression test suite
        self.regression_tests = [
            {
                "id": "edge_case_1",
                "question": "What is the result of 0 divided by 0?",
                "expected_behavior": "handle_gracefully",
                "category": "error_handling"
            },
            {
                "id": "edge_case_2", 
                "question": "Parse this JSON: {invalid json}",
                "expected_behavior": "return_error_message",
                "category": "error_handling"
            },
            {
                "id": "performance_1",
                "question": "Calculate the 50th Fibonacci number",
                "expected_behavior": "complete_within_30s",
                "category": "performance"
            }
        ]
        
    def _create_comprehensive_test_set(self) -> List[GAIAQuestion]:
        """Create comprehensive GAIA-style test questions across domains."""
        return [
            # Factual Research Questions
            GAIAQuestion(
                id="fact_mercedes_albums",
                question="How many studio albums did Mercedes Sosa release between 2000 and 2009?",
                expected_answer="0",
                category="factual_research",
                difficulty="medium",
                requires_tools=["web_researcher", "semantic_search_tool"],
                reasoning_steps=4,
                time_limit=180
            ),
            
            GAIAQuestion(
                id="fact_country_code",
                question="What is the three-letter country code for Egypt according to ISO 3166-1 alpha-3?",
                expected_answer="EGY",
                category="factual_lookup",
                difficulty="easy",
                requires_tools=["web_researcher"],
                reasoning_steps=2,
                time_limit=60
            ),
            
            # Mathematical Computation
            GAIAQuestion(
                id="math_division",
                question="What is 1729 divided by 7?",
                expected_answer="247",
                category="mathematical",
                difficulty="easy",
                requires_tools=["python_interpreter"],
                reasoning_steps=2,
                time_limit=30
            ),
            
            GAIAQuestion(
                id="math_compound_interest",
                question="If $1000 grows at 12% annually compounded, what will it become after 5 years? (Round to nearest dollar)",
                expected_answer="1762",
                category="mathematical",
                difficulty="medium",
                requires_tools=["python_interpreter"],
                reasoning_steps=3,
                time_limit=60
            ),
            
            # Temporal Analysis
            GAIAQuestion(
                id="temporal_berlin_911",
                question="How many years passed between the fall of the Berlin Wall and the 9/11 attacks?",
                expected_answer="11",
                category="temporal_analysis",
                difficulty="medium",
                requires_tools=["web_researcher", "python_interpreter"],
                reasoning_steps=4,
                time_limit=120
            ),
            
            # Complex Cross-Reference
            GAIAQuestion(
                id="complex_tokyo_gameboy",
                question="What was the population of Tokyo in the year the Nintendo Game Boy was first released?",
                expected_answer="11855563",
                category="complex_research",
                difficulty="hard",
                requires_tools=["web_researcher", "semantic_search_tool", "python_interpreter"],
                reasoning_steps=6,
                time_limit=300
            ),
            
            # Current Events
            GAIAQuestion(
                id="current_france_president",
                question="What is the surname of the current President of France?",
                expected_answer="Macron",
                category="current_events",
                difficulty="easy",
                requires_tools=["web_researcher", "tavily_search"],
                reasoning_steps=2,
                time_limit=60
            ),
            
            # Multi-step Reasoning
            GAIAQuestion(
                id="multistep_statue_liberty",
                question="If the Eiffel Tower was built in 1889, and the Statue of Liberty was dedicated 3 years earlier, in what year was the Statue of Liberty dedicated?",
                expected_answer="1886",
                category="multi_step_reasoning",
                difficulty="medium",
                requires_tools=["web_researcher", "python_interpreter"],
                reasoning_steps=4,
                time_limit=120
            ),
            
            # Scientific Facts
            GAIAQuestion(
                id="science_speed_light",
                question="What is the speed of light in kilometers per second? (Round to nearest thousand)",
                expected_answer="300000",
                category="scientific_facts",
                difficulty="easy",
                requires_tools=["web_researcher", "python_interpreter"],
                reasoning_steps=3,
                time_limit=90
            ),
            
            # Historical Analysis
            GAIAQuestion(
                id="history_ww2_duration",
                question="How many years did World War II last (from start to end)?",
                expected_answer="6",
                category="historical_analysis",
                difficulty="medium",
                requires_tools=["web_researcher", "python_interpreter"],
                reasoning_steps=3,
                time_limit=120
            )
        ]
    
    def test_agent_performance(self, agent, verbose=True) -> Dict[str, Any]:
        """
        Test agent performance across all GAIA questions.
        Returns comprehensive analysis and recommendations.
        """
        if verbose:
            print("🚀 GAIA Testing Framework - Advanced Agent Evaluation")
            print("=" * 70)
            print(f"📊 Questions: {len(self.questions)}")
            print(f"📂 Categories: {len(set(q.category for q in self.questions))}")
            print(f"🎯 Difficulty Levels: {len(set(q.difficulty for q in self.questions))}")
            print("=" * 70)
        
        # Initialise CSV log
        self._init_results_csv()

        results = []
        start_time = time.time()
        
        for i, question in enumerate(self.questions, 1):
            if verbose:
                print(f"\n🔍 Test {i}/{len(self.questions)}: {question.id}")
                print(f"   Question: {question.question[:80]}{'...' if len(question.question) > 80 else ''}")
            
            result = self._test_single_question(agent, question, verbose)

            # Optional: LLM grading overlay
            result = self._grade_with_llm_if_available(result)

            results.append(result)

            # Persist incremental result to CSV
            self._append_result_csv(result)
            
            if verbose:
                status = "✅ PASS" if result.correct else "❌ FAIL"
                print(f"   {status} | Expected: '{question.expected_answer}' | Got: '{result.agent_answer}'")
                print(f"   ⏱️  Time: {result.execution_time:.1f}s | 🧠 Steps: {result.reasoning_steps} | 📈 Confidence: {result.confidence:.0%}")
        
        total_time = time.time() - start_time
        analysis = self._analyze_comprehensive_results(results, total_time, verbose)
        
        if verbose:
            self._print_comprehensive_analysis(analysis)
        
        return {
            "results": results,
            "analysis": analysis,
            "total_time": total_time,
            "recommendations": self._generate_improvement_recommendations(analysis)
        }
    
    def _test_single_question(self, agent, question: GAIAQuestion, verbose=False) -> AgentTestResult:
        """Test a single question with comprehensive tracking."""
        start_time = time.time()
        tools_used = []
        error_messages = []
        reasoning_steps = 0
        confidence = 0.0
        validation_score = 0.0
        
        try:
            # Prepare agent input
            agent_input = {
                "messages": [{"role": "user", "content": question.question}],
                "run_id": uuid.uuid4(),
                "log_to_db": False,
                "plan": "",
                "step_count": 0,
                "confidence": 0.3,
                "reasoning_complete": False
            }
            
            # Execute agent
            agent_response = None
            final_confidence = 0.3
            
            try:
                if hasattr(agent, 'stream'):
                    # Stream execution for detailed tracking
                    for chunk in agent.stream(agent_input):
                        reasoning_steps += 1
                        last_message = chunk.get("messages", [])[-1] if chunk.get("messages") else None
                        
                        if last_message:
                            # Track tool usage
                            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                                for tool_call in last_message.tool_calls:
                                    if isinstance(tool_call, dict):
                                        tools_used.append(tool_call.get('name', 'unknown'))
                                    else:
                                        tools_used.append(getattr(tool_call, 'name', 'unknown'))
                            
                            # Update confidence
                            final_confidence = chunk.get("confidence", final_confidence)
                            
                            # Check for completion
                            if chunk.get("reasoning_complete", False):
                                agent_response = getattr(last_message, 'content', str(last_message))
                                break
                            
                        # Safety timeout
                        if reasoning_steps > 25:
                            break
                
                elif hasattr(agent, 'run'):
                    # Fallback for non-streaming agents
                    result = agent.run(agent_input)
                    messages = result.get("messages", [])
                    if messages:
                        agent_response = getattr(messages[-1], 'content', str(messages[-1]))
                    final_confidence = result.get("confidence", 0.5)
                    reasoning_steps = result.get("step_count", 1)
                
                else:
                    # Direct function call fallback
                    agent_response = str(agent(question.question))
                    reasoning_steps = 1
                    final_confidence = 0.5
                
            except Exception as e:
                error_messages.append(f"Agent execution error: {str(e)}")
                agent_response = f"ERROR: {str(e)}"
            
            execution_time = time.time() - start_time
            
            # Extract and clean final answer
            if agent_response:
                clean_answer = self._extract_clean_answer(str(agent_response))
            else:
                clean_answer = "No response"
                error_messages.append("Agent failed to generate response")
            
            # Check correctness
            correct = self._check_answer_correctness(clean_answer, question.expected_answer)
            
            # Calculate validation score (simple heuristic)
            validation_score = self._calculate_validation_score(
                correct, final_confidence, len(set(tools_used)), reasoning_steps
            )
            
            return AgentTestResult(
                question_id=question.id,
                question=question.question,
                expected_answer=question.expected_answer,
                agent_answer=clean_answer,
                correct=correct,
                confidence=final_confidence,
                reasoning_steps=reasoning_steps,
                execution_time=execution_time,
                tools_used=list(set(tools_used)),
                error_messages=error_messages,
                validation_score=validation_score,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_messages.append(str(e))
            
            return AgentTestResult(
                question_id=question.id,
                question=question.question,
                expected_answer=question.expected_answer,
                agent_answer=f"CRITICAL_ERROR: {str(e)}",
                correct=False,
                confidence=0.0,
                reasoning_steps=reasoning_steps,
                execution_time=execution_time,
                tools_used=tools_used,
                error_messages=error_messages,
                validation_score=0.0,
                timestamp=datetime.now()
            )
    
    def _extract_clean_answer(self, response: str) -> str:
        """Extract clean final answer from complex agent response."""
        if not response:
            return ""
        
        # Remove common prefixes
        prefixes = [
            "final answer:", "the answer is:", "answer:", "result:", 
            "therefore:", "conclusion:", "my final answer is:",
            "based on my analysis,", "after analyzing,"
        ]
        
        response_lower = response.lower().strip()
        
        for prefix in prefixes:
            if prefix in response_lower:
                idx = response_lower.rfind(prefix)
                if idx != -1:
                    extracted = response[idx + len(prefix):].strip()
                    if extracted:
                        response = extracted
                        break
        
        # Clean up formatting
        response = response.strip()
        response = response.strip('.,!?')
        
        # If multiline, prefer the last substantive line
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines and len(lines) > 1:
            # Look for short, factual final line
            for line in reversed(lines):
                if len(line) < 50 and any(c.isalnum() for c in line):
                    response = line
                    break
        
        return response
    
    def _check_answer_correctness(self, agent_answer: str, expected_answer: str) -> bool:
        """Smart answer comparison with multiple matching strategies."""
        if not agent_answer or not expected_answer:
            return False
        
        # Normalize
        agent_norm = agent_answer.lower().strip()
        expected_norm = expected_answer.lower().strip()
        
        # Exact match
        if agent_norm == expected_norm:
            return True
        
        # Numeric comparison
        try:
            agent_num = float(agent_norm.replace(',', ''))
            expected_num = float(expected_norm.replace(',', ''))
            return abs(agent_num - expected_num) < 0.01
        except (ValueError, TypeError):
            pass
        
        # Partial match for longer answers
        if len(expected_norm) > 3:
            return expected_norm in agent_norm or agent_norm in expected_norm
        
        # Fuzzy match for short answers
        if len(expected_norm) <= 3:
            return agent_norm.startswith(expected_norm) or expected_norm.startswith(agent_norm)
        
        return False
    
    def _calculate_validation_score(self, correct: bool, confidence: float, 
                                  tool_diversity: int, reasoning_steps: int) -> float:
        """Calculate comprehensive validation score."""
        score = 0.0
        
        # Base score for correctness
        if correct:
            score += 0.6
        
        # Confidence calibration
        if correct and confidence > 0.7:
            score += 0.2  # Well-calibrated high confidence
        elif not correct and confidence < 0.5:
            score += 0.1  # At least uncertain when wrong
        
        # Tool diversity bonus
        if tool_diversity > 2:
            score += 0.1
        elif tool_diversity > 1:
            score += 0.05
        
        # Reasoning depth
        if reasoning_steps > 3:
            score += 0.1
        elif reasoning_steps > 1:
            score += 0.05
        
        return min(1.0, score)
    
    def _analyze_comprehensive_results(self, results: List[AgentTestResult], 
                                     total_time: float, verbose=False) -> Dict[str, Any]:
        """Comprehensive analysis of test results."""
        if not results:
            return {"error": "No results to analyze"}
        
        # Basic metrics
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r.correct)
        accuracy = correct_answers / total_questions
        
        # Category analysis
        category_stats = {}
        difficulty_stats = {}
        
        for result in results:
            question = next((q for q in self.questions if q.id == result.question_id), None)
            if question:
                # Category performance
                cat = question.category
                if cat not in category_stats:
                    category_stats[cat] = {"total": 0, "correct": 0, "avg_time": 0, "avg_confidence": 0}
                category_stats[cat]["total"] += 1
                category_stats[cat]["avg_time"] += result.execution_time
                category_stats[cat]["avg_confidence"] += result.confidence
                if result.correct:
                    category_stats[cat]["correct"] += 1
                
                # Difficulty analysis
                diff = question.difficulty
                if diff not in difficulty_stats:
                    difficulty_stats[diff] = {"total": 0, "correct": 0}
                difficulty_stats[diff]["total"] += 1
                if result.correct:
                    difficulty_stats[diff]["correct"] += 1
        
        # Finalize category stats
        for cat_data in category_stats.values():
            if cat_data["total"] > 0:
                cat_data["avg_time"] /= cat_data["total"]
                cat_data["avg_confidence"] /= cat_data["total"]
        
        # Performance metrics
        avg_time = sum(r.execution_time for r in results) / len(results)
        avg_steps = sum(r.reasoning_steps for r in results) / len(results)
        avg_confidence = sum(r.confidence for r in results) / len(results)
        avg_validation = sum(r.validation_score for r in results) / len(results)
        
        # Tool usage analysis
        all_tools = []
        for result in results:
            all_tools.extend(result.tools_used)
        
        tool_usage = {}
        for tool in set(all_tools):
            tool_usage[tool] = all_tools.count(tool)
        
        # Error analysis
        errors = []
        for result in results:
            errors.extend(result.error_messages)
        
        error_types = {}
        for error in errors:
            error_type = error.split(':')[0] if ':' in error else 'general'
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "overall_accuracy": accuracy,
            "correct_answers": correct_answers,
            "total_questions": total_questions,
            "category_performance": {
                cat: {
                    "accuracy": stats["correct"] / stats["total"],
                    "avg_time": stats["avg_time"],
                    "avg_confidence": stats["avg_confidence"]
                }
                for cat, stats in category_stats.items()
            },
            "difficulty_performance": {
                diff: stats["correct"] / stats["total"]
                for diff, stats in difficulty_stats.items()
            },
            "avg_execution_time": avg_time,
            "avg_reasoning_steps": avg_steps,
            "avg_confidence": avg_confidence,
            "avg_validation_score": avg_validation,
            "tool_usage": tool_usage,
            "error_analysis": error_types,
            "total_execution_time": total_time
        }
    
    def _print_comprehensive_analysis(self, analysis: Dict[str, Any]):
        """Print detailed analysis with actionable insights."""
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE GAIA EVALUATION RESULTS")
        print("=" * 70)
        
        # Overall Performance
        accuracy = analysis.get("overall_accuracy", 0)
        print(f"🎯 Overall Accuracy: {accuracy:.1%} ({analysis.get('correct_answers', 0)}/{analysis.get('total_questions', 0)})")
        print(f"⏱️  Average Time: {analysis.get('avg_execution_time', 0):.1f}s")
        print(f"🧠 Average Reasoning Steps: {analysis.get('avg_reasoning_steps', 0):.1f}")
        print(f"📈 Average Confidence: {analysis.get('avg_confidence', 0):.1%}")
        print(f"✅ Average Validation Score: {analysis.get('avg_validation_score', 0):.1%}")
        
        # Category Performance
        print(f"\n📋 Category Performance:")
        for category, stats in analysis.get("category_performance", {}).items():
            print(f"   {category.replace('_', ' ').title()}: {stats['accuracy']:.1%} "
                  f"(⏱️ {stats['avg_time']:.1f}s, 📈 {stats['avg_confidence']:.1%})")
        
        # Difficulty Analysis
        print(f"\n🎚️  Difficulty Analysis:")
        for difficulty, accuracy in analysis.get("difficulty_performance", {}).items():
            print(f"   {difficulty.title()}: {accuracy:.1%}")
        
        # Tool Usage
        print(f"\n🛠️  Tool Usage (Top 5):")
        tool_usage = analysis.get("tool_usage", {})
        for tool, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {tool}: {count} times")
        
        # Error Analysis
        error_analysis = analysis.get("error_analysis", {})
        if error_analysis:
            print(f"\n🚨 Error Analysis:")
            for error_type, count in sorted(error_analysis.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"   {error_type}: {count} occurrences")
        
        print("=" * 70)
    
    def _generate_improvement_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for agent improvement."""
        recommendations = []
        accuracy = analysis.get("overall_accuracy", 0)
        avg_confidence = analysis.get("avg_confidence", 0)
        
        # Overall performance recommendations
        if accuracy < 0.4:
            recommendations.append("🔴 CRITICAL: Overall accuracy below 40% - fundamental reasoning system needs overhaul")
        elif accuracy < 0.6:
            recommendations.append("🟡 MODERATE: Accuracy below 60% - significant improvements needed in core reasoning")
        elif accuracy < 0.8:
            recommendations.append("🟢 GOOD: Accuracy above 60% - focus on fine-tuning and edge case handling")
        
        # Confidence calibration
        if avg_confidence > 0.8 and accuracy < 0.6:
            recommendations.append("⚠️ OVERCONFIDENCE: High confidence with low accuracy - improve uncertainty quantification")
        elif avg_confidence < 0.5 and accuracy > 0.7:
            recommendations.append("📈 UNDERCONFIDENCE: Low confidence with good accuracy - boost confidence in correct answers")
        
        # Category-specific recommendations
        category_performance = analysis.get("category_performance", {})
        weak_categories = [cat for cat, stats in category_performance.items() if stats["accuracy"] < 0.5]
        if weak_categories:
            recommendations.append(f"📌 FOCUS AREAS: Improve performance in {', '.join(weak_categories)}")
        
        # Tool usage recommendations
        tool_usage = analysis.get("tool_usage", {})
        if not tool_usage:
            recommendations.append("🛠️ TOOL USAGE: Agent not using tools effectively - improve tool selection logic")
        elif len(tool_usage) < 3:
            recommendations.append("🔧 TOOL DIVERSITY: Increase tool diversity for better cross-validation")
        
        # Error-specific recommendations
        error_analysis = analysis.get("error_analysis", {})
        if "timeout" in str(error_analysis):
            recommendations.append("⏱️ TIMEOUT ISSUES: Implement better time management and early stopping")
        if "rate limit" in str(error_analysis):
            recommendations.append("🔄 RATE LIMITING: Improve rate limit handling and request pacing")
        
        return recommendations

    # --------------------------------------------------
    # Lightweight LLM grader and CSV utilities
    # --------------------------------------------------

    _CSV_HEADERS = [
        "question_id", "agent_answer", "expected_answer", "score", "confidence", "reasoning_steps", "execution_time", "validation_score", "tools_used", "justification"
    ]

    def _init_results_csv(self, path: str = "test_results.csv"):
        import csv, os
        self._csv_path = path
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self._CSV_HEADERS)

    def _append_result_csv(self, res: AgentTestResult):
        import csv
        justification = next((m.replace("LLM_GRADE:", "").strip() for m in res.error_messages if m.startswith("LLM_GRADE:")), "")
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                res.question_id,
                res.agent_answer,
                res.expected_answer,
                "Pass" if res.correct else "Fail",
                f"{res.confidence:.2f}",
                res.reasoning_steps,
                f"{res.execution_time:.2f}",
                f"{res.validation_score:.2f}",
                ";".join(res.tools_used),
                justification
            ])

    def _get_grader_llm(self):
        try:
            from langchain_groq import ChatGroq
            # Use Gemma-7b for grading as it's optimized for evaluation tasks
            return ChatGroq(
                model_name="gemma-7b-it",
                temperature=0,
                max_retries=1
            )
        except Exception:
            return None

    def _grade_with_llm_if_available(self, result: AgentTestResult) -> AgentTestResult:
        """Adds pass/fail grading from LLM if API available."""
        if getattr(self, "_grader", None) is False:
            return result  # disabled previously

        if not hasattr(self, "_grader"):
            self._grader = self._get_grader_llm() or False

        grader = self._grader if self._grader else None
        if grader is None:
            return result  # skip if not available

        system_prompt = (
            "You are an automated grader. You will get a question, correct answer, and agent answer. "
            'Return strictly JSON as {"score": "Pass|Fail", "justification": "..."}.'
        )
        user_prompt = f"Question: {result.question}\nCorrect: {result.expected_answer}\nAgent: {result.agent_answer}"

        try:
            llm_resp = grader.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            import json as _json
            parsed = _json.loads(llm_resp.content.strip())
            score = parsed.get("score", "Fail").lower()
            justification = parsed.get("justification", "")
            result.correct = score == "pass"
            result.error_messages.append(f"LLM_GRADE:{justification}")
        except Exception as e:
            # on failure disable further grading
            self._grader = False
            result.error_messages.append(f"GraderError:{e}")
        return result

    def run_regression_tests(self, agent) -> Dict[str, Any]:
        """Run regression tests to ensure stability."""
        results = []
        
        for test in self.regression_tests:
            start_time = time.time()
            try:
                result = agent(test["question"])
                passed = self._check_expected_behavior(
                    result, test["expected_behavior"], time.time() - start_time
                )
                results.append({
                    "test_id": test["id"],
                    "passed": passed,
                    "category": test["category"]
                })
            except Exception as e:
                results.append({
                    "test_id": test["id"],
                    "passed": False,
                    "error": str(e)
                })
                
        return {
            "total_tests": len(results),
            "passed": sum(1 for r in results if r.get("passed", False)),
            "results": results
        }
    
    def _check_expected_behavior(self, result, expected_behavior, duration):
        """Check if result matches expected behavior."""
        if expected_behavior == "handle_gracefully":
            return "error" not in str(result).lower() or "unable" in str(result).lower()
        elif expected_behavior == "return_error_message":
            return "error" in str(result).lower() or "invalid" in str(result).lower()
        elif expected_behavior == "complete_within_30s":
            return duration < 30
        return True

def main():
    """Main function for standalone testing."""
    print("🧪 GAIA Testing Framework")
    print("Advanced evaluation system for ReAct agents")
    print("=" * 50)
    
    # Initialize test suite
    test_suite = GAIATestSuite()
    
    print(f"✅ Test suite initialized with {len(test_suite.questions)} questions")
    print(f"📂 Categories: {len(set(q.category for q in test_suite.questions))}")
    print(f"🎯 Difficulty levels: {len(set(q.difficulty for q in test_suite.questions))}")
    
    # Show sample questions
    print("\n🔍 Sample Test Questions:")
    for i, question in enumerate(test_suite.questions[:3]):
        print(f"\n{i+1}. [{question.category}] {question.question}")
        print(f"   Expected: {question.expected_answer}")
        print(f"   Tools: {', '.join(question.requires_tools)}")
    
    print("\n✅ Framework ready for agent testing!")
    print("Usage: test_suite.test_agent_performance(your_agent)")

if __name__ == "__main__":
    main() 