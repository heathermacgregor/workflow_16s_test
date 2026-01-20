# NOT WORKING !!!

import requests
import time
import os  # Good practice to get keys from environment variables

# Best practice: store your token as an environment variable
# In your terminal: export HF_TOKEN="your_token_here"
API_TOKEN = os.getenv("HF_TOKEN")
if not API_TOKEN:
    raise ValueError("Hugging Face API token not found. Please set the HF_TOKEN environment variable.")

# A 404 error typically means the model ID is incorrect or the model is "gated".
# Gated models (like Gemma and Mistral) require you to visit their Hugging Face page
# and agree to their terms of use before you can access them via the API.
#
# To debug, let's use a non-gated, publicly available model like 'gpt2'.
# If this works, it confirms your code and token are correct, and the issue
# is with accessing the specific gated models.
#
# To use a gated model later:
# 1. Go to its page on hf.co (e.g., https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
# 2. Log in with the account tied to your API token.
# 3. Accept the terms and conditions.
# 4. Try this script again with the gated model's URL.
API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload, retries=5, delay=10):
    """
    Sends a request to the Hugging Face API with robust error handling and retries.
    """
    for i in range(retries):
        # Added a timeout to the request for better network robustness
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

        # 1. Check for a successful response code
        if response.status_code == 200:
            try:
                # 2. Try to parse the JSON
                return response.json()
            except requests.exceptions.JSONDecodeError:
                print(f"Error: Received a 200 OK status, but failed to decode JSON.")
                print(f"Response text: {response.text}")
                return None

        # 3. Handle the "model is loading" state (503 error)
        elif response.status_code == 503:
            print(f"Model is loading... Retrying in {delay} seconds. (Attempt {i+1}/{retries})")
            time.sleep(delay)
            continue  # Go to the next iteration of the loop to retry

        # 4. Handle other potential errors
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response text: {response.text}")  # This will show you the actual error message
            return None

    print(f"Model failed to load after {retries} retries.")
    return None

# --- Example Usage ---
if __name__ == '__main__':
    prompt = "What is the capital of California?"
    payload = {"inputs": prompt}

    output = query(payload)

    if output:
        # The output format can vary by model, inspect it first!
        # For gpt2, the result is typically in a list.
        # Ensure the response structure matches what you expect before accessing it.
        try:
            print(output[0]['generated_text'])
        except (TypeError, KeyError, IndexError) as e:
            print(f"Error parsing the output: {e}")
            print(f"Full output received: {output}")

