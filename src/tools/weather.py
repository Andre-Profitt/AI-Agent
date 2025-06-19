"""
Weather tool implementation.
"""

import os
import requests
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    """Input schema for weather tool."""
    location: str = Field(description="Location to get weather for")
    units: str = Field(default="metric", description="Units (metric/imperial)")

@tool
def get_weather(location: str, units: str = "metric") -> str:
    """
    Get current weather for a location.
    
    Args:
        location (str): Location to get weather for
        units (str): Units (metric/imperial)
        
    Returns:
        str: Weather information or error message
    """
    try:
        # Get API key from environment
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            return "Error: OPENWEATHER_API_KEY environment variable not set"
            
        # Make API request
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": location,
            "appid": api_key,
            "units": units
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Format weather information
        weather = {
            "location": data["name"],
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
        
        return str(weather)
                
    except Exception as e:
        return f"Error getting weather: {str(e)}" 