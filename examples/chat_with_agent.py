#!/usr/bin/env python3
"""
💬 Chat with your Super AI Agent
"""

import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
import requests
from datetime import datetime

# Load env
from dotenv import load_dotenv
load_dotenv()

class ChatAgent:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.7,
            model_name="llama3-70b-8192",
            groq_api_key=os.getenv('GROQ_API_KEY')
        )
        self.messages = [SystemMessage(content="""
        You are a super powerful AI assistant with access to:
        - Advanced reasoning capabilities
        - Web search functionality
        - Memory storage
        - Multiple knowledge domains
        
        Be helpful, accurate, and engaging.
        """)]
        
    def search_web(self, query):
        """Quick web search"""
        try:
            response = requests.post(
                'https://api.tavily.com/search',
                json={
                    "api_key": os.getenv('TAVILY_API_KEY'),
                    "query": query,
                    "max_results": 3
                }
            )
            if response.status_code == 200:
                return response.json().get('results', [])
        except:
            pass
        return []
        
    def chat(self, user_input):
        """Process user input"""
        # Check if user wants to search
        if any(word in user_input.lower() for word in ['search', 'find', 'look up']):
            print("🔍 Searching the web...")
            results = self.search_web(user_input)
            if results:
                context = "\n".join([f"- {r['title']}: {r['content'][:100]}..." for r in results])
                user_input = f"{user_input}\n\nWeb results:\n{context}"
                
        self.messages.append(HumanMessage(content=user_input))
        response = self.llm.invoke(self.messages)
        self.messages.append(response)
        
        return response.content

def main():
    agent = ChatAgent()
    
    print("\n🤖 SUPER AI AGENT CHAT")
    print("=" * 40)
    print("Commands:")
    print("  - Ask any question")
    print("  - Say 'search' to search the web")
    print("  - Type 'quit' to exit")
    print("=" * 40 + "\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
            
        print("\nAgent: ", end="", flush=True)
        response = agent.chat(user_input)
        print(response + "\n")

if __name__ == "__main__":
    main()