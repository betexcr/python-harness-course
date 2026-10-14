import os
import sys

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")  # so emoji don't crash the Windows console

import json
from pathlib import Path

from openai import OpenAI


load_dotenv()  # Load environment variables from .env file
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
 

WORKSPACE = Path(__file__).parent / "workspace"


# --- the actual tools: plain Python functions -------------------------------

def list_files() -> str:
    """List the files in the assistant's workspace folder."""
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"


def read_file(filename: str) -> str:
    """Read a file from the workspace folder."""
    path = WORKSPACE / filename
    if not path.is_file():
        return f"error: no file named {filename}"
    return path.read_text(encoding="utf-8")


def write_file(filename: str, content: str) -> str:
    """Write data to a file in the workspace folder."""
    path = WORKSPACE / filename
    path.write_text(content, encoding="utf-8")
    return f"File written: {filename}"


TOOLS = {"list_files": list_files, "read_file": read_file, "write_file": write_file}

# --- what the model sees: a JSON description of each tool -------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in the user's workspace folder.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one file from the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write data to a file in the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}, "content": {"type": "string"}},
                "required": ["filename", "content"],
            },
        },
    },
]


def chat(user_message: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful personal assistant."},
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model=os.getenv("MODEL"), messages=messages, tools=TOOL_SCHEMAS
    )
    message = response.choices[0].message

    # The model didn't want a tool - just answer.
    if not message.tool_calls:
        return message.content or ""

    # Run the tool it asked for and give it the result... ONCE.
    messages.append(message)
    for call in message.tool_calls:
        args = json.loads(call.function.arguments or "{}")
        print(f"  [tool] {call.function.name}({args})")
        result = TOOLS[call.function.name](**args)
        messages.append(
            {"role": "tool", "tool_call_id": call.id, "content": result}
        )

    final = client.chat.completions.create(model=os.getenv("MODEL"), messages=messages)
    return final.choices[0].message.content


if __name__ == "__main__":
    print(f"v1 assistant ({os.getenv('MODEL')}) - ctrl+c to quit")
    try:
        while True:
            question = input("\nyou: ")
            print("\nassistant:", chat(question))
    except (EOFError, KeyboardInterrupt):
        print("\nbye!")
