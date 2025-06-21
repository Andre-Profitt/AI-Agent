#!/usr/bin/env python3
"""
Simple Demo of the Super AI Agent - Works with minimal dependencies
"""

import asyncio
import os
from datetime import datetime

# Set up environment
os.environ['GROQ_API_KEY'] = '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'

async def demo_reasoning_engine():
    """Demo the reasoning engine"""
    print("\n🧠 DEMO: Advanced Reasoning Engine")
    print("="*50)
    
    try:
        from src.gaia_components.advanced_reasoning_engine import AdvancedReasoningEngine, ReasoningType
        
        engine = AdvancedReasoningEngine()
        
        # Test different reasoning types
        queries = [
            ("Why is the sky blue?", ReasoningType.DEDUCTIVE),
            ("I've seen 3 black cats today. Are all cats black?", ReasoningType.INDUCTIVE),
            ("My car won't start and the radio is dead. What's wrong?", ReasoningType.ABDUCTIVE)
        ]
        
        for query, reasoning_type in queries:
            print(f"\n📝 Query: {query}")
            result = await engine.reason(query, {"demo": True}, reasoning_type)
            print(f"💡 Response: {result['response']}")
            print(f"📊 Confidence: {result['confidence']:.2%}")
            
    except Exception as e:
        print(f"❌ Error in reasoning demo: {e}")

async def demo_memory_system():
    """Demo the memory system"""
    print("\n🧩 DEMO: Enhanced Memory System")
    print("="*50)
    
    try:
        from src.gaia_components.enhanced_memory_system import EnhancedMemorySystem, MemoryPriority
        
        memory = EnhancedMemorySystem()
        
        # Store memories
        print("\n📝 Storing memories...")
        memory.store_episodic("User asked about AI capabilities", "user_question")
        memory.store_semantic("AI can process language and images", ["AI", "capabilities"])
        memory.store_working("Current task: Demo the system", MemoryPriority.HIGH)
        
        # Retrieve memories
        ai_memories = memory.search_memories("AI")
        print(f"\n🔍 Found {len(ai_memories)} memories about 'AI'")
        
        stats = memory.get_memory_statistics()
        print(f"\n📊 Memory Stats: {stats['total_memories']} total memories")
        
    except Exception as e:
        print(f"❌ Error in memory demo: {e}")

async def demo_tool_executor():
    """Demo the tool executor"""
    print("\n🔧 DEMO: Tool Executor")
    print("="*50)
    
    try:
        from src.gaia_components.tool_executor import create_production_tool_executor
        
        executor = create_production_tool_executor()
        
        async with executor:
            # Calculator
            print("\n🧮 Calculator: 2^10 + sqrt(16)")
            result = await executor.execute("calculator", expression="2**10 + sqrt(16)")
            if result.status.value == "success":
                print(f"   Result: {result.result['result']}")
            
            # Text analysis
            print("\n📝 Text Analysis:")
            text = "AI agents are revolutionizing software development"
            result = await executor.execute("text_processor", text=text, operation="analyze")
            if result.status.value == "success":
                print(f"   Word count: {result.result['word_count']}")
                
    except Exception as e:
        print(f"❌ Error in tool demo: {e}")

async def demo_groq_integration():
    """Demo Groq LLM integration"""
    print("\n🤖 DEMO: Groq LLM Integration")
    print("="*50)
    
    try:
        from langchain_groq import ChatGroq
        from langchain.schema import HumanMessage
        
        # Initialize Groq
        llm = ChatGroq(
            temperature=0.7,
            model_name="llama3-8b-8192",  # Fast and free!
            groq_api_key=os.environ['GROQ_API_KEY']
        )
        
        # Test simple query
        print("\n💬 Testing Groq LLM...")
        response = llm.invoke([HumanMessage(content="What are the benefits of AI agents?")])
        print(f"🤖 Response: {response.content[:200]}...")
        
    except Exception as e:
        print(f"❌ Error in Groq demo: {e}")
        print("   Make sure langchain-groq is installed: pip install langchain-groq")

async def main():
    """Run all demos"""
    print("\n🚀 SUPER AI AGENT - SIMPLE DEMO")
    print("🌟" * 25)
    
    demos = [
        demo_reasoning_engine,
        demo_memory_system,
        demo_tool_executor,
        demo_groq_integration
    ]
    
    for demo in demos:
        try:
            await demo()
            await asyncio.sleep(1)
        except Exception as e:
            print(f"❌ Demo {demo.__name__} failed: {e}")
    
    print("\n✨ Demo Complete!")
    print("="*50)

if __name__ == "__main__":
    # Check minimal dependencies
    print("🔍 Checking dependencies...")
    
    required = ["numpy", "pandas", "aiohttp"]
    missing = []
    
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"⚠️  Missing dependencies: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        print("\n   For Groq support: pip install langchain-groq")
    else:
        print("✅ Core dependencies found!")
        
    # Run demo
    asyncio.run(main())