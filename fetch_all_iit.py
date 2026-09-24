import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()
PARSE_API_KEY = os.getenv("PARSE_API_KEY")

if not PARSE_API_KEY:
    raise ValueError("PARSE_API_KEY not found in .env")

# The search_colleges endpoint that worked before
API_URL = "https://api.parse.bot/scraper/9aadd01b-6856-442a-9002-9c6bf49e614e/search_colleges"

def fetch_colleges(query):
    """Fetch colleges matching a query."""
    headers = {"X-API-Key": PARSE_API_KEY}
    params = {"query": query}
    
    print(f"Fetching: {query}")
    response = requests.get(API_URL, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def main():
    # Fetch all IITs — the search endpoint returns 30 in one call
    data = fetch_colleges("IIT")
    
    if data.get("status") != "success":
        print(f"API error: {data}")
        return
    
    colleges = data.get("data", {}).get("colleges", [])
    print(f"\nFetched {len(colleges)} colleges")
    
    # Save to file in the same structure as iit_colleges.json
    output = {
        "status": "success",
        "data": {
            "query": "IIT",
            "intent": "basicinfo",
            "heading": "Check college details",
            "total_results": len(colleges),
            "redirect_url": None,
            "colleges": colleges
        }
    }
    
    with open("data/iit_colleges.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(colleges)} colleges to data/iit_colleges.json")
    
    # Print the names for verification
    print("\nColleges fetched:")
    for i, c in enumerate(colleges, 1):
        print(f"  {i}. {c['name']} ({c['location']})")

if __name__ == "__main__":
    main()