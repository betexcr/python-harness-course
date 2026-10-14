import os
import sys

sys.stdout.reconfigure(encoding="utf-8")  # so emoji don't crash the Windows console

from dotenv import load_dotenv
from openai import OpenAI

# One client. Point it at any OpenAI-compatible server (Ollama here).
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

load_dotenv()  # Load environment variables from .env file 


def chat(user_message: str) -> str:
    response = client.chat.completions.create(
        model=os.getenv("MODEL"),  # or "qwen2.5:3b" or "qwen3.5:4b"
        messages=[
            {"role": "system", "content": "You are a helpful personal assistant."},
            {"role": "user", "content": user_message},
            # {"role": "assistant", "content": "Sure! I can help you with that."}, # Response from the assistant can be included here if needed
        ],
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    print(f"v0 assistant ({os.getenv('MODEL')}) - ctrl+c to quit")
    try:
        while True:
            question = input("\nyou: ")
            print("\nassistant:", chat(question))
    except (EOFError, KeyboardInterrupt):
        print("\nbye!")


