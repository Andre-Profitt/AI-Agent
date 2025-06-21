#!/usr/bin/env python3
"""
Your Custom AI Agent - Powered by Groq!
"""

import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# Set your Groq API key
os.environ['GROQ_API_KEY'] = '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'

def create_my_agent():
    """Create your custom AI agent"""
    
    # Initialize Groq with Llama 3
    agent = ChatGroq(
        temperature=0.7,
        model_name="llama3-70b-8192",  # Most powerful free model!
        groq_api_key=os.environ['GROQ_API_KEY']
    )
    
    return agent

def chat_with_agent():
    """Interactive chat with your agent"""
    
    agent = create_my_agent()
    
    print("🤖 Your AI Agent is ready!")
    print("Type 'quit' to exit\n")
    
    # System prompt to make it super powerful
    system_prompt = """You are a super powerful AI agent with advanced reasoning capabilities.
    You can help with coding, analysis, creative tasks, and complex problem solving.
    Be helpful, accurate, and thorough in your responses."""
    
    messages = [SystemMessage(content=system_prompt)]
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
            
        # Add user message
        messages.append(HumanMessage(content=user_input))
        
        # Get agent response
        response = agent.invoke(messages)
        print(f"\n🤖 Agent: {response.content}")
        
        # Add agent response to history
        messages.append(response)

if __name__ == "__main__":
    chat_with_agent()