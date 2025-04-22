import os
import sys
import subprocess
import platform
import shutil
import webbrowser
from typing import Any, Dict, List
from datetime import datetime, timezone, timedelta

# --- Gemini imports with error handling ---
try:
    from google import genai
    from google.genai import types
    from google.genai.types import (
        GenerateContentConfig,
    )
except ImportError:
    print("[ERROR] Please install the Gemini SDK: pip install google-generativeai")
    sys.exit(1)

# --- Configuration constants ---
SYSTEM_PROMPT = (
    "You are a supercharged coding agent with full filesystem and Git integration. "
    "Automatically infer and create directories for file operations. "
    "For web projects, place HTML in 'templates/' and assets in 'static/css/', "
    "'static/js/', 'static/images/' and so on. Support linting, formatting, testing, "
    "and version control (Git). Detach long‑running servers so the chat loop remains responsive. "
    "Use 'run_script' to capture output for non‑interactive scripts, and 'start_interactive' "
    "for scripts requiring user input. Use 'open_in_browser' for quick HTML previews. "
    "Remember main.py is the file you are running on—do not delete it, rename it, nor create a new file "
    "with the same name. If anything is beyond your capability, just say 'I can't do that'."
)
MODEL_NAME = "models/gemini-2.5-flash-preview-04-17"
MAX_CHUNK_SIZE = 200 * 1024  # 200 KB

# --- Initialize Gemini client ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("[ERROR] Missing GOOGLE_API_KEY")
    sys.exit(1)

try:
    genai_client = genai.Client(api_key=api_key)
except Exception as e:
    print(f"[ERROR] Failed to initialize Gemini client: {e}")
    sys.exit(1)

# --- File‐tool implementations ---

def chunk_file(path: str, max_size: int) -> Dict[str, Any]:
    # Always use chunk reading for large files, regardless of cache
    if not os.path.isfile(path):
        return {"status": "error", "error": "file not found"}
    try:
        chunks: List[str] = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                data = f.read(max_size)
                if not data:
                    break
                chunks.append(data)
        return {"status": "success", "chunks": chunks}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def process_large_file(path: str, max_size: int) -> Dict[str, Any]:
    cr = chunk_file(path, max_size)
    if cr.get("status") != "success":
        return cr
    responses = []
    for i, chunk in enumerate(cr["chunks"], 1):
        print(f"Processing chunk {i}/{len(cr['chunks'])}…")
        resp = chat.send_message(f"Process chunk:\n{chunk}")
        responses.append(getattr(resp, 'text', str(resp)))
    return {"status": "success", "responses": responses}

def read_file(path: str) -> Dict[str, Any]:
    try:
        size = os.path.getsize(path)
        if size > MAX_CHUNK_SIZE:
            return {
                "status": "warning",
                "warning": f"File is large ({size} bytes). Use 'chunk_file'.",
                "size": size
            }
        content = open(path, "r", encoding="utf-8", errors="ignore").read()
        return {"status": "success", "content": content}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def create_file(path: str, content: str) -> Dict[str, Any]:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"status": "success", "path": path}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def update_file(path: str, content: str) -> Dict[str, Any]:
    return create_file(path, content)

def delete_file(path: str) -> Dict[str, Any]:
    def onerr(fn, p, exc):
        os.chmod(p, 0o666)
        fn(p)
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, onerror=onerr)
            return {"status": "deleted_directory", "path": path}
        os.remove(path)
        return {"status": "deleted_file", "path": path}
    except FileNotFoundError:
        return {"status": "error", "error": "not found"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def rename_file(old: str, new: str) -> Dict[str, Any]:
    try:
        os.makedirs(os.path.dirname(new) or ".", exist_ok=True)
        os.rename(old, new)
        return {"status": "renamed", "from": old, "to": new}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def move_file(src: str, dst: str) -> Dict[str, Any]:
    try:
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.move(src, dst)
        return {"status": "moved", "from": src, "to": dst}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def list_directory(path: str) -> Dict[str, Any]:
    try:
        return {"status": "success", "path": path, "items": os.listdir(path)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def search_file(keyword: str, path: str) -> Dict[str, Any]:
    matches = []
    for root, _, files in os.walk(path):
        for f in files:
            full = os.path.join(root, f)
            try:
                if keyword in open(full, "r", encoding="utf-8", errors="ignore").read():
                    matches.append(full)
            except:
                pass
    return {"status": "success", "matches": matches}

def run_script(path: str) -> Dict[str, Any]:
    """Run a script. If it requires user input or is a server, run in a new terminal; else, run in current terminal."""
    try:
        # Heuristic: check for input() or flask/django/server keywords
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        needs_interactive = (
            'input(' in content or 'raw_input(' in content or 'flask' in content or 'django' in content or 'app.run' in content or 'serve_forever' in content
        )
        if needs_interactive:
            # Start in new terminal/console
            kwargs = {}
            if platform.system() == 'Windows' and hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
            else:
                kwargs['start_new_session'] = True
            proc = subprocess.Popen([sys.executable, path], **kwargs)
            return {'status': 'started_interactive', 'pid': proc.pid}
        else:
            result = subprocess.run([sys.executable, path], capture_output=True, text=True)
            return {'status': 'completed', 'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def start_interactive(path: str) -> Dict[str, Any]:
    try:
        kwargs = {}
        if platform.system() == "Windows" and hasattr(subprocess, "CREATE_NEW_CONSOLE"):
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
        else:
            kwargs["start_new_session"] = True
        p = subprocess.Popen([sys.executable, path], **kwargs)
        return {"status": "started", "pid": p.pid}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def install_package(pkg: str) -> Dict[str, Any]:
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "install", pkg], capture_output=True, text=True)
        return {"status": "completed", "stdout": r.stdout, "stderr": r.stderr, "exit_code": r.returncode}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def open_in_browser(path: str) -> Dict[str, Any]:
    try:
        url = path if path.startswith("http") else f"file://{os.path.abspath(path)}"
        webbrowser.open(url)
        return {"status": "opened", "url": url}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def lint_code(path: str) -> Dict[str, Any]:
    try:
        r = subprocess.run(["flake8", path], capture_output=True, text=True)
        return {"status": "completed", "stdout": r.stdout, "stderr": r.stderr}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def format_code(path: str) -> Dict[str, Any]:
    try:
        r = subprocess.run(["black", path], capture_output=True, text=True)
        return {"status": "formatted", "stdout": r.stdout, "stderr": r.stderr}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def run_tests(path: str) -> Dict[str, Any]:
    try:
        r = subprocess.run(["pytest", path], capture_output=True, text=True)
        return {"status": "completed", "stdout": r.stdout, "stderr": r.stderr}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def git_commit(msg: str) -> Dict[str, Any]:
    try:
        subprocess.run(["git", "add", "."], check=True)
        r = subprocess.run(["git", "commit", "-m", msg], capture_output=True, text=True)
        return {"status": "committed", "stdout": r.stdout, "stderr": r.stderr}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def git_push(remote: str, branch: str) -> Dict[str, Any]:
    try:
        r = subprocess.run(["git", "push", remote, branch], capture_output=True, text=True)
        return {"status": "pushed", "stdout": r.stdout, "stderr": r.stderr}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def prompt_input(msg: str) -> Dict[str, Any]:
    return {"user_input": input(f"{msg} ")}

# --- Build initial chat session ---
config = GenerateContentConfig(
    tools=[
        create_file, read_file, chunk_file, update_file, delete_file,
        rename_file, move_file, list_directory, search_file,
        run_script, start_interactive, install_package, open_in_browser,
        lint_code, format_code, run_tests, git_commit, git_push, prompt_input
    ],
    temperature=0,
    max_output_tokens=1024
)
chat = genai_client.chats.create(model=MODEL_NAME, config=config)

# --- Main REPL loop ---
if __name__ == "__main__":
    print("=== Gemini Super Agent ===")
    print("[INFO] Chat session started. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            print("Agent: Goodbye!")
            break
        response = chat.send_message(user_input)
        print(f"Agent: {getattr(response, 'text', response)}")
