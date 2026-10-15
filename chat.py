import os
import sys
import time
import ollama
import platform


sys.stdout.reconfigure(encoding="utf-8")  # so emoji don't crash the Windows console

from dotenv import load_dotenv
from openai import OpenAI

# One client. Point it at any OpenAI-compatible server (Ollama here).
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
ollama_client = ollama.Client("http://127.0.0.1:11434")

load_dotenv()  # Load environment variables from .env file 

def native_ping():
    system = platform.system()
    if system == "Windows":
        import winsound
        winsound.MessageBeep()
    elif system == "Darwin":  # macOS
        os.system('afplay /System/Library/Sounds/Glass.aiff')
    elif system == "Linux":
        # Attempts to use standard Linux command line audio players
        os.system('aplay -q /usr/share/sounds/alsa/Front_Center.wav > /dev/null 2>&1 || speaker-test -l 1 -t sine -f 1000 > /dev/null 2>&1')


def chat(user_message: str, model: str = os.getenv("MODEL")) -> str:
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model, 
        messages=[
            {"role": "system", "content": "You are a helpful personal assistant."},
            {"role": "user", "content": user_message},
            # {"role": "assistant", "content": "Sure! I can help you with that."}, # Response from the assistant can be included here if needed
        ],
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"\n {model} response time: {elapsed_time:.2f} seconds \n")
    return response.choices[0].message.content or ""

def get_local_models():
    # Fetch the list of available local models
    response = ollama_client.list()
    
    # Extract model names from the response object
    models = [model.model for model in response.models]
    return models

def select_model():
    """Prompts the user to select an available model."""
    models = get_local_models()
    
    if not models:
        print("No local Ollama models found. Please pull a model first (e.g., 'ollama pull llama3').")
        return None

    print("\n--- Available Ollama Models ---")
    for index, model in enumerate(models, 1):
        print(f"[{index}] {model}")
    print("---------------------------------")

    while True:
        user_input = input("\nEnter the model name or number to select: ").strip()

        # Scenario A: User typed the exact model name or part of it
        if user_input in models:
            return user_input
            
        # Scenario B: User entered a number from the list
        if user_input.isdigit():
            choice_idx = int(user_input) - 1
            if 0 <= choice_idx < len(models):
                return models[choice_idx]

        # Scenario C: Case-insensitive fallback check
        matched = [m for m in models if user_input.lower() == m.lower()]
        if matched:
            return matched[0]

        print(f"\n⚠️ '{user_input}' is not valid. Will use default: {os.getenv('MODEL')}.\n")


if __name__ == "__main__":
    print(f"v0 assistant ({os.getenv('MODEL')}) - ctrl+c to quit")
    try:
        while True:
            question = input("\nyou: ")
            if question.strip() == "":
                question = "What are some interesting WW1 facts?"
                
            model_choice = select_model()
            if model_choice:
                print("\nassistant:", chat(question, model=model_choice))
                native_ping()

                
    except (EOFError, KeyboardInterrupt):
        print("\nbye!")


