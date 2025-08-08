
"""
Integration test for FastAPI /query endpoint using realistic test data.

Purpose:
  - Validates end-to-end query functionality using a real PDF file.
  - Requires FastAPI and Node.js backend running locally and accessible.
  - Uses only existing test data (no hardcoded legacy paths).

Structure:
  - Sends a POST request to /query with a realistic PDF file and query string.
  - Prints response status and JSON output for verification.

Dependencies:
  - requests (Python HTTP library)
  - test data: data/test/test.pdf (must exist)

NOTE: Update context documentation after running this test.
"""

import requests

def run_integration_test():
    """
    Runs an integration test against the FastAPI /query endpoint.
    Sends a realistic PDF file and query string, prints response status and JSON.
    Raises AssertionError if the test file does not exist or response is not 200.
    """
    url = "http://localhost:8000/query"
    test_file_path = r"C:\D_Drive\Projects\Finserv\data\test\test.pdf"
    try:
        with open(test_file_path, "rb") as f:
            files = [("files", f)]
            data = {"query": "Does this policy cover knee surgery, and what are the conditions?"}
            response = requests.post(url, files=files, data=data)
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.json()}")
            assert response.status_code == 200, f"Unexpected status: {response.status_code}"
    except FileNotFoundError:
        print(f"Test file not found: {test_file_path}")
        raise

if __name__ == "__main__":
    run_integration_test()
