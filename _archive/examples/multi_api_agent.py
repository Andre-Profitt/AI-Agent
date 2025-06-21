#!/usr/bin/env python3
"""
🚀 Multi-API Super Agent - Combining Multiple AI Models and Search
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

# Load environment
from dotenv import load_dotenv
load_dotenv()

class MultiAPIAgent:
    """Agent that can use multiple LLMs and search APIs"""
    
    def __init__(self):
        self.llms = {}
        self.search_tools = {}
        self.setup_apis()
        
    def setup_apis(self):
        """Setup all available APIs"""
        print("🔧 Setting up APIs...")
        
        # Groq (Free and Fast!)
        try:
            from langchain_groq import ChatGroq
            self.llms['groq'] = ChatGroq(
                temperature=0.7,
                model_name="llama3-70b-8192",
                groq_api_key=os.getenv('GROQ_API_KEY')
            )
            print("✅ Groq LLM ready (Llama 3 70B)")
        except Exception as e:
            print(f"❌ Groq setup failed: {e}")
            
        # Google Gemini
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llms['google'] = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=os.getenv('GOOGLE_API_KEY'),
                temperature=0.7
            )
            print("✅ Google Gemini ready")
        except Exception as e:
            print(f"❌ Google setup failed: {e}")
            
        # Anthropic Claude
        try:
            from langchain_anthropic import ChatAnthropic
            self.llms['anthropic'] = ChatAnthropic(
                model="claude-3-sonnet-20240229",
                anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
                temperature=0.7
            )
            print("✅ Anthropic Claude ready")
        except Exception as e:
            print(f"❌ Anthropic setup failed: {e}")
            
        # Tavily Search
        try:
            import requests
            self.search_tools['tavily'] = {
                'api_key': os.getenv('TAVILY_API_KEY'),
                'url': 'https://api.tavily.com/search'
            }
            print("✅ Tavily Search ready")
        except Exception as e:
            print(f"❌ Tavily setup failed: {e}")
            
        # Supabase Database
        try:
            from supabase import create_client
            self.db = create_client(
                os.getenv('SUPABASE_URL'),
                os.getenv('SUPABASE_KEY')
            )
            print("✅ Supabase Database ready")
        except Exception as e:
            print(f"❌ Supabase setup failed: {e}")
            self.db = None
            
    async def ask_llm(self, prompt: str, model: str = "groq") -> str:
        """Ask a specific LLM"""
        if model not in self.llms:
            return f"Model {model} not available. Available: {list(self.llms.keys())}"
            
        try:
            from langchain.schema import HumanMessage
            response = await asyncio.to_thread(
                self.llms[model].invoke,
                [HumanMessage(content=prompt)]
            )
            return response.content
        except Exception as e:
            return f"Error with {model}: {str(e)}"
            
    async def ask_all_llms(self, prompt: str) -> Dict[str, str]:
        """Ask all available LLMs the same question"""
        results = {}
        
        tasks = []
        for model_name in self.llms.keys():
            task = self.ask_llm(prompt, model_name)
            tasks.append((model_name, task))
            
        for model_name, task in tasks:
            try:
                result = await task
                results[model_name] = result
            except Exception as e:
                results[model_name] = f"Error: {str(e)}"
                
        return results
        
    async def search_web(self, query: str) -> List[Dict[str, Any]]:
        """Search the web using Tavily"""
        if 'tavily' not in self.search_tools:
            return [{"error": "Tavily search not available"}]
            
        try:
            import requests
            
            response = requests.post(
                self.search_tools['tavily']['url'],
                json={
                    "api_key": self.search_tools['tavily']['api_key'],
                    "query": query,
                    "max_results": 5
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
            else:
                return [{"error": f"Search failed: {response.status_code}"}]
                
        except Exception as e:
            return [{"error": f"Search error: {str(e)}"}]
            
    async def save_conversation(self, user_msg: str, agent_response: str):
        """Save conversation to Supabase"""
        if not self.db:
            return
            
        try:
            self.db.table('conversations').insert({
                'user_message': user_msg,
                'agent_response': agent_response,
                'timestamp': datetime.now().isoformat()
            }).execute()
        except Exception as e:
            print(f"Failed to save conversation: {e}")
            
    async def get_conversation_history(self, limit: int = 10):
        """Get recent conversation history"""
        if not self.db:
            return []
            
        try:
            result = self.db.table('conversations').select('*').order(
                'timestamp', desc=True
            ).limit(limit).execute()
            return result.data
        except Exception as e:
            print(f"Failed to get history: {e}")
            return []
            
    async def run_research_task(self, topic: str) -> Dict[str, Any]:
        """Run a comprehensive research task using all APIs"""
        print(f"\n🔬 Researching: {topic}")
        
        results = {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "search_results": [],
            "llm_analysis": {},
            "synthesis": ""
        }
        
        # Step 1: Search the web
        print("📡 Searching the web...")
        search_results = await self.search_web(topic)
        results["search_results"] = search_results[:3]  # Top 3
        
        # Step 2: Ask each LLM to analyze
        print("🧠 Analyzing with multiple LLMs...")
        
        search_summary = "\n".join([
            f"- {r.get('title', 'No title')}: {r.get('content', '')[:200]}..."
            for r in search_results[:3]
        ])
        
        analysis_prompt = f"""
        Research Topic: {topic}
        
        Web Search Results:
        {search_summary}
        
        Please provide a comprehensive analysis of this topic based on the search results.
        Include key insights, important facts, and your expert perspective.
        """
        
        llm_responses = await self.ask_all_llms(analysis_prompt)
        results["llm_analysis"] = llm_responses
        
        # Step 3: Synthesize all responses
        print("🔄 Synthesizing results...")
        
        synthesis_prompt = f"""
        Multiple AI models have analyzed the topic: {topic}
        
        Their responses:
        {json.dumps(llm_responses, indent=2)}
        
        Create a unified, comprehensive summary that combines the best insights from all models.
        """
        
        synthesis = await self.ask_llm(synthesis_prompt, "groq")
        results["synthesis"] = synthesis
        
        # Save to database
        await self.save_conversation(f"Research: {topic}", synthesis)
        
        return results

async def interactive_cli():
    """Interactive command-line interface"""
    agent = MultiAPIAgent()
    
    print("\n" + "🌟" * 30)
    print("🚀 MULTI-API SUPER AGENT")
    print("🌟" * 30)
    print("\nCommands:")
    print("  ask <model> <question> - Ask specific model (groq/google/anthropic)")
    print("  compare <question> - Compare all models")
    print("  search <query> - Search the web")
    print("  research <topic> - Full research with all APIs")
    print("  history - Show conversation history")
    print("  help - Show commands")
    print("  quit - Exit")
    print("\n")
    
    while True:
        try:
            command = input("🤖 > ").strip()
            
            if command.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
                
            elif command.lower() == 'help':
                print("\nCommands:")
                print("  ask <model> <question> - Ask specific model")
                print("  compare <question> - Compare all models")
                print("  search <query> - Search the web")
                print("  research <topic> - Full research")
                print("  history - Show conversation history")
                
            elif command.startswith('ask '):
                parts = command[4:].split(' ', 1)
                if len(parts) == 2:
                    model, question = parts
                    print(f"\n💭 Asking {model}...")
                    response = await agent.ask_llm(question, model)
                    print(f"\n{model.upper()}: {response}")
                    await agent.save_conversation(command, response)
                else:
                    print("Usage: ask <model> <question>")
                    
            elif command.startswith('compare '):
                question = command[8:]
                print("\n🔄 Comparing all models...")
                responses = await agent.ask_all_llms(question)
                
                for model, response in responses.items():
                    print(f"\n{model.upper()}:")
                    print(response[:500] + "..." if len(response) > 500 else response)
                    print("-" * 50)
                    
            elif command.startswith('search '):
                query = command[7:]
                print("\n🔍 Searching...")
                results = await agent.search_web(query)
                
                for i, result in enumerate(results[:5]):
                    print(f"\n{i+1}. {result.get('title', 'No title')}")
                    print(f"   {result.get('url', '')}")
                    print(f"   {result.get('content', '')[:200]}...")
                    
            elif command.startswith('research '):
                topic = command[9:]
                results = await agent.run_research_task(topic)
                
                print(f"\n📊 Research Complete!")
                print(f"\nSynthesis:")
                print(results['synthesis'])
                
            elif command == 'history':
                history = await agent.get_conversation_history()
                for item in history:
                    print(f"\n🕒 {item['timestamp']}")
                    print(f"You: {item['user_message']}")
                    print(f"Agent: {item['agent_response'][:200]}...")
                    
            else:
                # Default: ask Groq
                print("\n💭 Thinking...")
                response = await agent.ask_llm(command, "groq")
                print(f"\n🤖 {response}")
                await agent.save_conversation(command, response)
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # First, let's install missing dependencies
    print("📦 Checking dependencies...")
    
    try:
        import requests
    except ImportError:
        print("Installing requests...")
        os.system("pip3 install requests")
        
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        print("Installing Google Generative AI...")
        os.system("pip3 install langchain-google-genai")
        
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        print("Installing Anthropic...")
        os.system("pip3 install langchain-anthropic")
        
    try:
        from supabase import create_client
    except ImportError:
        print("Installing Supabase...")
        os.system("pip3 install supabase")
        
    # Run the agent
    asyncio.run(interactive_cli())