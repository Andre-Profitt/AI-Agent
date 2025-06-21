#!/usr/bin/env python3
"""
🔄 Compare responses from different AI models
"""

import os
from dotenv import load_dotenv
load_dotenv()

def ask_groq(question):
    """Ask Groq Llama 3"""
    try:
        from langchain_groq import ChatGroq
        from langchain.schema import HumanMessage
        
        llm = ChatGroq(
            model_name="llama3-70b-8192",
            groq_api_key=os.getenv('GROQ_API_KEY')
        )
        response = llm.invoke([HumanMessage(content=question)])
        return response.content
    except Exception as e:
        return f"Error: {e}"

def ask_google(question):
    """Ask Google Gemini"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain.schema import HumanMessage
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
        response = llm.invoke([HumanMessage(content=question)])
        return response.content
    except Exception as e:
        return f"Error: {e}"

def compare_models(question):
    """Compare responses from available models"""
    print(f"\n📝 Question: {question}\n")
    print("=" * 60)
    
    # Groq
    print("\n🦙 GROQ (Llama 3 70B):")
    print("-" * 40)
    groq_response = ask_groq(question)
    print(groq_response[:500] + "..." if len(groq_response) > 500 else groq_response)
    
    # Google
    print("\n\n🌟 GOOGLE (Gemini Pro):")
    print("-" * 40)
    google_response = ask_google(question)
    print(google_response[:500] + "..." if len(google_response) > 500 else google_response)
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    print("🔄 AI MODEL COMPARISON TOOL")
    print("Compare responses from different AI models\n")
    
    # Example questions
    questions = [
        "What is consciousness?",
        "Write a haiku about AI",
        "Explain quantum computing in simple terms"
    ]
    
    print("Example questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
    
    print("\nOr type your own question (or 'quit' to exit)")
    
    while True:
        choice = input("\nChoice (1-3 or your question): ")
        
        if choice.lower() in ['quit', 'exit']:
            break
            
        if choice in ['1', '2', '3']:
            question = questions[int(choice) - 1]
        else:
            question = choice
            
        compare_models(question)