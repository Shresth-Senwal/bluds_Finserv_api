"""
Integration test for FastAPI /query endpoint with URL-based document.
Requires FastAPI and Node.js backend running locally.
"""
import requests

url = "http://localhost:8080/query"
from typing import Dict, Any

payload: Dict[str, Any] = {
    "query": "Does this policy cover knee surgery, and what are the conditions?",
    "urls": ["https://example.com/policy.pdf"]
}
response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())
