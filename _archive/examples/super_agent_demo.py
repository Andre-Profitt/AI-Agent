#!/usr/bin/env python3
"""
🚀 Super AI Agent Demo - Integrated APIs
Uses Groq LLM, Tavily Search, and Supabase Database
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List

# Set up environment
from dotenv import load_dotenv
load_dotenv()

class SuperAgent:
    """Your Super Powerful AI Agent with integrated APIs"""
    
    def __init__(self):
        # Groq LLM
        from langchain_groq import ChatGroq
        self.llm = ChatGroq(
            temperature=0.7,
            model_name="llama3-70b-8192",
            groq_api_key=os.getenv('GROQ_API_KEY')
        )
        
        # Tavily Search API key
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        
        # Initialize Supabase
        try:
            from supabase import create_client
            self.db = create_client(
                os.getenv('SUPABASE_URL'),
                os.getenv('SUPABASE_KEY')
            )
            print("✅ Database connected")
        except Exception as e:
            print(f"⚠️  Database not available: {e}")
            self.db = None
            
    async def search_web(self, query: str) -> List[Dict]:
        """Search the web using Tavily"""
        import requests
        
        try:
            response = requests.post(
                'https://api.tavily.com/search',
                json={
                    "api_key": self.tavily_api_key,
                    "query": query,
                    "max_results": 5,
                    "include_domains": [],
                    "exclude_domains": []
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
            else:
                return []
        except Exception as e:
            print(f"Search error: {e}")
            return []
            
    async def think(self, prompt: str) -> str:
        """Use Groq LLM to think about something"""
        from langchain.schema import HumanMessage
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Error thinking: {e}"
            
    async def remember(self, key: str, value: Any) -> bool:
        """Store something in memory (Supabase)"""
        if not self.db:
            return False
            
        try:
            self.db.table('agent_memory').insert({
                'key': key,
                'value': json.dumps(value),
                'timestamp': datetime.now().isoformat()
            }).execute()
            return True
        except Exception as e:
            # Table might not exist, let's create it
            try:
                # Try to create the table using raw SQL
                self.db.rpc('create_agent_memory_table', {}).execute()
            except:
                pass
            return False
            
    async def recall(self, key: str) -> Any:
        """Recall something from memory"""
        if not self.db:
            return None
            
        try:
            result = self.db.table('agent_memory').select('value').eq('key', key).execute()
            if result.data:
                return json.loads(result.data[0]['value'])
            return None
        except:
            return None
            
    async def research(self, topic: str) -> Dict[str, Any]:
        """Research a topic using search and LLM analysis"""
        print(f"\n🔬 Researching: {topic}")
        
        # Step 1: Search the web
        print("🌐 Searching the web...")
        search_results = await self.search_web(topic)
        
        if not search_results:
            print("⚠️  No search results found")
            search_summary = "No web results available."
        else:
            # Create a summary of search results
            search_summary = "\n".join([
                f"- {r.get('title', 'No title')}: {r.get('content', '')[:200]}..."
                for r in search_results[:3]
            ])
            print(f"✅ Found {len(search_results)} results")
        
        # Step 2: Analyze with LLM
        print("🧠 Analyzing information...")
        analysis_prompt = f"""
        Topic: {topic}
        
        Web Search Results:
        {search_summary}
        
        Please provide:
        1. A comprehensive summary of the topic
        2. Key insights and important facts
        3. Practical applications or implications
        4. Interesting aspects worth exploring further
        
        Be thorough but concise.
        """
        
        analysis = await self.think(analysis_prompt)
        
        # Step 3: Store in memory
        research_data = {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "search_results": search_results[:3],
            "analysis": analysis
        }
        
        await self.remember(f"research_{topic}", research_data)
        
        return research_data
        
    async def chat(self, message: str) -> str:
        """Have a conversation with the agent"""
        # Check if user is asking for research
        if any(keyword in message.lower() for keyword in ['research', 'search', 'find out about', 'tell me about']):
            # Extract topic
            topic = message.lower()
            for keyword in ['research', 'search', 'find out about', 'tell me about']:
                topic = topic.replace(keyword, '').strip()
                
            research = await self.research(topic)
            return research['analysis']
        else:
            # Regular conversation
            return await self.think(message)

async def demo():
    """Run the demo"""
    print("\n" + "🌟" * 30)
    print("🚀 SUPER AI AGENT DEMO")
    print("🌟" * 30)
    
    agent = SuperAgent()
    
    print("\n💡 Capabilities:")
    print("  - 🧠 Advanced reasoning with Groq Llama 3 70B")
    print("  - 🔍 Web search with Tavily")
    print("  - 💾 Memory storage with Supabase")
    print("  - 🔬 Comprehensive research capabilities")
    
    print("\n📝 Example Commands:")
    print("  - 'Research quantum computing'")
    print("  - 'Tell me about climate change'")
    print("  - 'What is the meaning of life?'")
    print("  - 'Help me write Python code'")
    print("\nType 'quit' to exit\n")
    
    # Demo some capabilities
    print("🎯 Quick Demo:")
    
    # 1. Simple question
    print("\n1️⃣ Testing reasoning...")
    response = await agent.think("What are the benefits of AI agents?")
    print(f"Agent: {response[:200]}...")
    
    # 2. Web search
    print("\n2️⃣ Testing web search...")
    results = await agent.search_web("latest AI developments 2024")
    if results:
        print(f"Found {len(results)} search results")
        print(f"Top result: {results[0].get('title', 'No title')}")
    
    # 3. Memory
    print("\n3️⃣ Testing memory...")
    await agent.remember("demo_time", datetime.now().isoformat())
    recalled = await agent.recall("demo_time")
    if recalled:
        print(f"Successfully stored and recalled: {recalled}")
    
    # Interactive chat
    print("\n💬 Chat Mode Active!\n")
    
    while True:
        try:
            user_input = input("You: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye!")
                break
                
            print("\n🤖 Agent: ", end="", flush=True)
            
            # Get response
            response = await agent.chat(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    # Check dependencies
    try:
        import requests
    except ImportError:
        print("Installing requests...")
        os.system("pip3 install requests")
        
    try:
        from supabase import create_client
    except ImportError:
        print("Installing supabase...")
        os.system("pip3 install supabase")
        
    # Run demo
    asyncio.run(demo())