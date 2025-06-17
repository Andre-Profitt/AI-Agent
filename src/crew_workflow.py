from typing import Dict, Any, List
import logging

# Placeholder imports – replace with real CrewAI components when available
try:
    from crewai import Crew, Agent as CrewAgent, Task as CrewTask  # type: ignore
except ImportError:
    Crew = None  # type: ignore
    CrewAgent = object  # type: ignore
    CrewTask = object  # type: ignore

logger = logging.getLogger(__name__)


def run_crew_workflow(query: str, tools: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight wrapper that executes a 3-agent CrewAI workflow.

    1. ResearcherAgent – web & semantic search
    2. ExecutionAgent  – python / file / media tools
    3. SynthesisAgent  – synthesise final answer only

    Returns a dict with keys: output, intermediate_steps
    """

    if Crew is None:
        logger.warning("CrewAI not installed – falling back to empty result")
        return {"output": "CrewAI not available", "intermediate_steps": []}

    # --- Define Crewmates ---
    researcher = CrewAgent(
        role="ResearcherAgent",
        goal="Gather facts and sources needed to answer user questions.",
        tools=[tools.get("web_researcher"), tools.get("semantic_search_tool")],
        backstory="A diligent internet researcher with quick search skills.",
        allow_delegation=False,
    )

    executor = CrewAgent(
        role="ExecutionAgent",
        goal="Run code and parse files or media as requested by the researcher.",
        tools=[tools.get("python_interpreter"), tools.get("file_reader"), tools.get("audio_transcriber")],
        backstory="A Python REPL in agent form, validating all inputs before execution.",
        allow_delegation=False,
    )

    synthesiser = CrewAgent(
        role="SynthesisAgent",
        goal="Craft the direct, concise final answer using gathered evidence only.",
        tools=[],
        backstory="A precise writer that outputs only what the user asked for.",
        allow_delegation=False,
    )

    # --- Define tasks ---
    t1 = CrewTask(
        description=f"Research the answer to: {query}",
        agent=researcher,
        expected_output="A fact sheet with citations",
    )

    t2 = CrewTask(
        description="If any code execution or parsing is needed, perform it and store results.",
        agent=executor,
        expected_output="Parsed / computed data ready for final answer",
        context=[t1],
    )

    t3 = CrewTask(
        description="Write the final answer in the required minimal format (number, name, etc.).",
        agent=synthesiser,
        expected_output="Final answer only",
        context=[t1, t2],
    )

    crew = Crew(
        agents=[researcher, executor, synthesiser],
        tasks=[t1, t2, t3],
        verbose=False,
    )

    try:
        result = crew.kickoff()
    except Exception as e:
        logger.error(f"Crew execution failed: {e}")
        return {"output": f"Crew error: {str(e)}", "intermediate_steps": []}

    return {
        "output": result.get("task_results", {}).get(t3.id, ""),
        "intermediate_steps": result.get("task_results", {})
    } 