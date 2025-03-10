import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GEMINI_PASS_KEY") 

genai.configure(api_key=GOOGLE_API_KEY)

def prompting(tweet):
    emotions = ["ambiguous", "amusement", "anger", "anxiety", "belief", "confusion", "depression", "disgust", "excitement", "optimism", "panic", "surprise"]
    
    prompt = (
        f"You are an intelligent stock based tweet analyzing assistant that selects out of 12 emotions, the one that best represents the feeling towards a certain company's stock. Based on the user's tweet: '{tweet}', "
        f"select the most relevant emotion, only one singular, from the following list {emotions}. Also say whether the tweet represents a bullish or bearish market for that specific company\n\n"
        f"your response should only contain two words, the emotion and if its either bullish or bearish (market sentiment)"
    )


    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    categorizing = response.candidates[0].content.parts[0].text.strip()

    return categorizing

def start_chat():
    while True:
        user_query = input("\nEnter your tweet: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        response = prompting(user_query)       
        response_array = response.split() 
        print(response_array)

if __name__ == "__main__":
    start_chat()
