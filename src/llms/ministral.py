import requests
import json

from src.utils import get_env_variable

def ministral_3b(prompt):
    OPENROUTER_API_KEY = get_env_variable("OPENROUTER_API_KEY")

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        },
        data=json.dumps({
            "model": "mistralai/ministral-3b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        })
    )
    return response.json()['choices'][0]['message']['content']