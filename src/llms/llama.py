import requests
import json

from src.utils import get_env_variable

def llama_3_2_1B(prompt):
    OPENROUTER_API_KEY = get_env_variable("OPENROUTER_API_KEY")

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        },
        data=json.dumps({
            "model": "meta-llama/llama-3.2-1b-instruct:free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0
        })
    )
    return response.json()['choices'][0]['message']['content']