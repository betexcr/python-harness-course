import sys

from dotenv import load_dotenv
import os
sys.stdout.reconfigure(encoding="utf-8")  # so emoji don't crash the Windows console

import inspect
import json
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
load_dotenv()  # Load environment variables from .env file

WORKSPACE = Path(__file__).parent / "workspace"


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
    """Write (or overwrite) a file in the workspace folder."""
    if "{{" in content and "}}" in content:
        return "error: content contains an unresolved template placeholder"
    (WORKSPACE / filename).write_text(content, encoding="utf-8")
    return f"wrote {filename}"


def create_file(filename: str, content: str) -> str:
    """Create a new file in the workspace folder."""
    if "{{" in content and "}}" in content:
        return "error: content contains an unresolved template placeholder"
    path = WORKSPACE / filename
    if path.exists():
        return f"error: file {filename} already exists"
    path.write_text(content, encoding="utf-8")
    return f"created {filename}"


def combine_files(filename: str) -> str:
    """Combine all top-level workspace files into a new output file."""
    output_path = WORKSPACE / filename
    source_paths = sorted(
        path for path in WORKSPACE.iterdir()
        if path.is_file() and path != output_path
    )
    if not source_paths:
        return "error: no source files to combine"

    contents = [path.read_text(encoding="utf-8") for path in source_paths]
    output_path.write_text("\n\n".join(contents), encoding="utf-8")
    return f"combined {len(source_paths)} files into {filename}"


TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "create_file": create_file,
    "combine_files": combine_files,
}

SYSTEM_PROMPT = """You are a helpful personal assistant with tools for working with files.

When the user asks you to create, write, or change a file, you must call the
appropriate file tool. Do not merely describe the intended change or say that
you are writing a file. For a new file, call create_file; to overwrite or
update a file, call write_file. If you need source contents, read them first.
Only say the file was created or written after the tool returns a success
message. If a tool returns an error, explain that error instead of claiming
the requested change succeeded.

When asked to combine all files in the workspace into one file, call
combine_files with the requested output filename. That tool reads the source
files itself; do not use placeholder values for file contents."""


def call_tool(name: str, args: dict) -> str:
    tool = TOOLS.get(name)
    if tool is None:
        return f"error: unknown tool {name}"

    try:
        inspect.signature(tool).bind(**args)
    except TypeError as exc:
        return f"error: invalid arguments for {name}: {exc}"

    return tool(**args)


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
            "description": "Write a file in the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file in the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "combine_files",
            "description": (
                "Combine the contents of all top-level files in the workspace "
                "into a new output file. Use this when asked to combine files."
            ),
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
        },
    },
]


def run_agent(user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # THE agent loop. This is the whole trick.
    while True:
        response = client.chat.completions.create(
            model=os.getenv("MODEL"), messages=messages, tools=TOOL_SCHEMAS
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or ""  # done - it answered

        messages.append(message)
        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            print(f"  [tool] {call.function.name}({args})")
            result = call_tool(call.function.name, args)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )


if __name__ == "__main__":
    print(f"v2 agent ({os.getenv('MODEL')}) - ctrl+c to quit")
    try:
        while True:
            question = input("\nyou: ")
            print("\nassistant:", run_agent(question))
    except (EOFError, KeyboardInterrupt):
        print("\nbye!")
