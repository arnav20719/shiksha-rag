import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
PARSE_API_KEY = os.getenv("PARSE_API_KEY")

if not PARSE_API_KEY:
    raise ValueError("PARSE_API_KEY not found in .env file")

API_URL = "https://api.parse.bot/scraper/9aadd01b-6856-442a-9002-9c6bf49e614e/search_colleges"

def fetch_colleges(query):
    """Fetch college data for a search query."""
    headers = {
        "X-API-Key": PARSE_API_KEY
    }
    params = {
        "query": query
    }
    
    response = requests.get(API_URL, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def main():
    query = "IIT"
    print(f"Fetching colleges for query: {query}")
    
    data = fetch_colleges(query)
    
    with open(f"data_{query.lower()}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Raw response saved to data_{query.lower()}.json")
    
    if data.get("status") == "success":
        colleges = data.get("data", {}).get("colleges", [])
        print(f"\nTotal colleges found: {len(colleges)}")
        print("\nFirst 3 colleges:")
        for college in colleges[:3]:
            print(f"  - {college['name']} ({college['location']})")
            print(f"    Rating: {college['rating']}, Courses: {college['course_count']}")
            print(f"    Salary: {college['median_salary']}")
    else:
        print("API returned an error:", data)

if __name__ == "__main__":
    main()