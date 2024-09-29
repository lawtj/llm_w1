from serpapi import GoogleSearch
from serpapi.google_search import GoogleSearch
from dotenv import load_dotenv
import os

load_dotenv()


async def get_citation_count(query):
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": os.getenv("SERPAPI_KEY")
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]
    return organic_results
