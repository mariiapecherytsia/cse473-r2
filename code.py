# Import required libraries for HTTP requests, MCP server, FastAPI, and async operations
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
import os
import json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import ollama
import logging
import asyncio

# Initialize FastAPI app for HTTP server functionality
app = FastAPI()

# Initialize FastMCP server and attach it to the FastAPI app
# This allows us to use MCP tools while maintaining FastAPI's routing capabilities
mcp = FastMCP("weather-assistant", app)

# List of example prompts to test different scenarios:
EXAMPLE_PROMPTS = [
    # Weather-related prompts
    "What's the weather like in Seattle?",
    "Will it rain in New York tomorrow?",
    "Is it sunny in Japan?",
    "What's the temperature in Tokyo?",
    
    # Non-weather prompts
    "What is the capital of France?",
    "Who wrote 'To Kill a Mockingbird'?",
    "When did World War II end?",

    
    # Complex prompts that would be challenging even with weather API
    "Compare the weather in San Francisco and Las Vegas for the next 3 days",
    "What was the weather like in Portland last week?",
    "Which city has better weather for outdoor activities this weekend: Los Angeles, Santa Barbara, or San Diego?"
]

# OpenMeteo API endpoints for geocoding and weather data
BASE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Set of weather-related keywords to identify weather-related queries
WEATHER_KEYWORDS = {
    'weather', 'temperature', 'rain', 'snow', 'sunny', 'cloudy', 'storm',
    'forecast', 'humidity', 'wind', 'precipitation', 'drizzle', 'fog',
    'thunder', 'hail', 'breeze', 'chilly', 'warm', 'cold', 'hot',
    'degrees', 'celsius', 'fahrenheit', '°C', '°F'
}

async def get_city_coordinates(city: str) -> dict[str, Any] | None:
    """
    Get coordinates for a city using OpenMeteo geocoding API.
    
    Args:
        city (str): Name of the city to get coordinates for
        
    Returns:
        dict[str, Any] | None: Dictionary containing city coordinates and metadata, or None if not found
    """
    params = {
        'name': city,
        'count': 1,
        'language': 'en',
        'format': 'json'
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get('results'):
                return data['results'][0]
            return None
        except Exception as e:
            print(f"Error getting coordinates: {str(e)}")
            return None

async def get_weather_data(city: str) -> dict[str, Any] | str:
    """
    Fetch current and forecast weather data from OpenMeteo API.
    
    Args:
        city (str): Name of the city to get weather data for
        
    Returns:
        dict[str, Any] | str: Weather data dictionary or error message string
    """
    # First get the city coordinates
    city_data = await get_city_coordinates(city)
    if not city_data:
        return f"Error: Could not find coordinates for {city}"
    
    # Set up parameters for weather API request
    params = {
        'latitude': city_data['latitude'],
        'longitude': city_data['longitude'],
        'current': 'temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m',
        'daily': 'temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code',
        'timezone': 'auto'
    }
    
    # Make the weather API request
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(WEATHER_URL, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return f"Error fetching weather data: {str(e)}"

def get_weather_description(code: int) -> str:
    """
    Convert weather code to human-readable description.
    
    Args:
        code (int): Weather code from OpenMeteo API
        
    Returns:
        str: Human-readable weather description
    """
    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }
    return weather_codes.get(code, "Unknown")

def needs_weather_data(question: str) -> bool:
    """
    Check if a question requires weather data by looking for weather-related keywords.
    
    Args:
        question (str): The user's question
        
    Returns:
        bool: True if the question is weather-related, False otherwise
    """
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in WEATHER_KEYWORDS)

def extract_city(question: str) -> str:
    """
    Extract city name from a question using the Llama model.
    
    Args:
        question (str): The user's question
        
    Returns:
        str: Extracted city name or 'NONE' if no city is found
    """
    prompt = f"""Extract the city name from this question. If no city is mentioned, respond with 'NONE'.
            Question: {question}

            Respond with ONLY the city name or 'NONE'."""
    
    response = ollama.chat(model='llama3.2:3b', messages=[
        {
            'role': 'system',
            'content': 'You are a city name extractor. Respond with ONLY the city name or "NONE".'
        },
        {
            'role': 'user',
            'content': prompt
        }
    ])
    return response['message']['content'].strip()

@mcp.tool("get-weather")
async def get_weather_tool(city: str) -> dict:
    """
    MCP tool to get weather data for a city.
    This tool is exposed through the MCP server for external use.
    
    Args:
        city (str): Name of the city to get weather data for
        
    Returns:
        dict: Weather data dictionary
    """
    return await get_weather_data(city)

@mcp.tool("process-query")
async def process_query_tool(question: str) -> str:
    """
    MCP tool to process a user query and provide appropriate response.
    This is the main processing function that handles both weather and non-weather queries.
    
    Args:
        question (str): The user's question
        
    Returns:
        str: Generated response based on the query type and available data
    """
    # Check if weather data is needed
    if not needs_weather_data(question):
        # Handle non-weather related questions directly with Ollama
        response = ollama.chat(model='llama3.2:3b', messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant that can answer various questions.'
            },
            {
                'role': 'user',
                'content': question
            }
        ])
        return response['message']['content']
    
    # For weather-related questions, extract city and get weather data
    city = extract_city(question)
    if city == 'NONE':
        return "I need a specific city to provide weather information. Could you please specify which city you're asking about?"
    
    # Get weather data using the MCP tool
    weather_data = await get_weather_tool(city)
    if isinstance(weather_data, str):  # If it's an error message
        return weather_data
    
    # Format weather data for the LLM
    current = weather_data['current']
    daily = weather_data['daily']
    weather_info = f"""Current weather in {city}:
                        Temperature: {current['temperature_2m']}°C
                        Humidity: {current['relative_humidity_2m']}%
                        Conditions: {get_weather_description(current['weather_code'])}
                        Wind Speed: {current['wind_speed_10m']} km/h

                        Forecast for tomorrow:
                        High: {daily['temperature_2m_max'][0]}°C
                        Low: {daily['temperature_2m_min'][0]}°C
                        Conditions: {get_weather_description(daily['weather_code'][0])}
                        Chance of precipitation: {daily['precipitation_probability_max'][0]}%"""
    
    # Get response from Ollama with weather context
    response = ollama.chat(model='llama3.2:3b', messages=[
        {
            'role': 'system',
            'content': 'You are a helpful weather assistant that provides accurate weather information based on real data.'
        },
        {
            'role': 'user',
            'content': f"""Based on the following real weather data for {city}:
                {weather_info}

                Please answer this question: {question}

                Provide a natural, conversational response incorporating the actual weather data."""
        }
    ])
    return response['message']['content']

async def test_prompts():
    """
    Test different prompts using MCP tools.
    This function runs through all example prompts and displays the responses.
    """
    print("\n=== Testing Example Prompts ===\n")
    
    for prompt in EXAMPLE_PROMPTS:
        print(f"\nPrompt: {prompt}")
        print("-" * 50)
        
        # Use the MCP tool to process the query
        response = await process_query_tool(prompt)
        print("\nResponse:")
        print(response)
        print("\n" + "="*50)

if __name__ == "__main__":
    # Run the test prompts
    asyncio.run(test_prompts())
