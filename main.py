import os
import subprocess
import sys
import platform
import shutil
import webbrowser
from typing import Any, Dict, List

# --- Gemini API import with error handling ---
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("[ERROR] The 'google.genai' package is not installed. "
          "Please install it with 'pip install google-generativeai' and try again.")
    sys.exit(1)

# Initialize Gemini client
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("[ERROR] GOOGLE_API_KEY environment variable is required. Please set it and try again.")
    sys.exit(1)

try:
    genai_client = genai.Client(api_key=api_key)
except Exception as e:
    print(f"[ERROR] Failed to initialize Gemini client: {e}")
    sys.exit(1)

# System prompt guiding the agent behavior
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

# Maximum chunk size for reading large files (in bytes)
MAX_CHUNK_SIZE = 200 * 1024

def chunk_file(path: str, max_size: int) -> Dict[str, Any]:
    """
    Read a file in chunks to avoid sending too much content at once.
    Returns {'status':'success','chunks':[...]} or {'status':'error',...}.
    """
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
    chunk_result = chunk_file(path, max_size)
    if chunk_result.get("status") != "success":
        return chunk_result
    chunks = chunk_result["chunks"]
    responses = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1} of {len(chunks)}...")
        resp = chat.send_message(f"Process this chunk of the file:\n{chunk}")
        responses.append(resp.text if hasattr(resp, 'text') else str(resp))
    return {"status": "success", "responses": responses}

def read_file(path: str) -> Dict[str, Any]:
    """
    Read and return the content of a file, with size check.
    If the file exceeds MAX_CHUNK_SIZE, returns a warning.
    """
    try:
        size = os.path.getsize(path)
        if size > MAX_CHUNK_SIZE:
            return {
                "status": "warning",
                "warning": f"File is large ({size} bytes). "
                           "Use 'chunk_file' to read it in parts.",
                "size": size
            }
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return {"status": "success", "content": f.read()}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def create_file(path: str, content: str) -> Dict[str, Any]:
    """Create or overwrite a file, auto‑creating parent directories."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {'status': 'success', 'path': path}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def update_file(path: str, content: str) -> Dict[str, Any]:
    """Alias for create_file."""
    return create_file(path, content)

def delete_file(path: str) -> Dict[str, Any]:
    """Delete a file or directory at the given path, handling permissions."""
    def onerror(func, fn, excinfo):
        try:
            os.chmod(fn, 0o666)
            func(fn)
        except Exception:
            pass
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, onerror=onerror)
            return {'status': 'deleted_directory', 'path': path}
        os.remove(path)
        return {'status': 'deleted_file', 'path': path}
    except FileNotFoundError:
        return {'status': 'error', 'error': 'path not found'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def rename_file(old_path: str, new_path: str) -> Dict[str, Any]:
    """Rename a file or directory."""
    try:
        os.makedirs(os.path.dirname(new_path) or ".", exist_ok=True)
        os.rename(old_path, new_path)
        return {'status': 'renamed', 'from': old_path, 'to': new_path}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def move_file(src: str, dest: str) -> Dict[str, Any]:
    """Move a file or directory."""
    try:
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        shutil.move(src, dest)
        return {'status': 'moved', 'from': src, 'to': dest}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def list_directory(path: str) -> Dict[str, Any]:
    """List files and directories at a given path."""
    try:
        items = os.listdir(path)
        return {'status': 'success', 'path': path, 'items': items}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def search_file(keyword: str, path: str) -> Dict[str, Any]:
    """Search for a keyword in files under the given path."""
    matches: List[str] = []
    for root, _, files in os.walk(path):
        for fname in files:
            full = os.path.join(root, fname)
            try:
                with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                    if keyword in f.read():
                        matches.append(full)
            except Exception:
                continue
    return {'status': 'success', 'matches': matches}

def run_script(path: str) -> Dict[str, Any]:
    """Run a non‑interactive script, capturing stdout and stderr."""
    try:
        result = subprocess.run([sys.executable, path], capture_output=True, text=True)
        return {'status': 'completed', 'exit_code': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def start_interactive(path: str) -> Dict[str, Any]:
    """Launch a script in a new console for interactive input."""
    try:
        kwargs = {}
        if platform.system() == 'Windows' and hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
        else:
            kwargs['start_new_session'] = True
        proc = subprocess.Popen([sys.executable, path], **kwargs)
        return {'status': 'started', 'pid': proc.pid}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def install_package(package: str) -> Dict[str, Any]:
    """Install a Python package via pip."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package],
            capture_output=True, text=True
        )
        return {'status': 'completed', 'stdout': result.stdout, 'stderr': result.stderr,
                'exit_code': result.returncode}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def open_in_browser(path: str) -> Dict[str, Any]:
    """Open a file or URL in the default web browser for preview."""
    try:
        url = path if path.startswith('http') else f'file://{os.path.abspath(path)}'
        webbrowser.open(url)
        return {'status': 'opened', 'url': url}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def lint_code(path: str) -> Dict[str, Any]:
    """Run flake8 linter on the given file or directory."""
    try:
        result = subprocess.run(['flake8', path], capture_output=True, text=True)
        return {'status': 'completed', 'stdout': result.stdout, 'stderr': result.stderr}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def format_code(path: str) -> Dict[str, Any]:
    """Format code using black on the given file or directory."""
    try:
        result = subprocess.run(['black', path], capture_output=True, text=True)
        return {'status': 'formatted', 'stdout': result.stdout, 'stderr': result.stderr}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def run_tests(path: str) -> Dict[str, Any]:
    """Run pytest on the specified directory."""
    try:
        result = subprocess.run(['pytest', path], capture_output=True, text=True)
        return {'status': 'completed', 'stdout': result.stdout, 'stderr': result.stderr}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def git_commit(message: str) -> Dict[str, Any]:
    """Commit staged changes with a commit message."""
    try:
        subprocess.run(['git', 'add', '.'], check=True)
        result = subprocess.run(['git', 'commit', '-m', message], capture_output=True, text=True)
        return {'status': 'committed', 'stdout': result.stdout, 'stderr': result.stderr}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def git_push(remote: str, branch: str) -> Dict[str, Any]:
    """Push commits to the remote repository."""
    try:
        result = subprocess.run(['git', 'push', remote, branch], capture_output=True, text=True)
        return {'status': 'pushed', 'stdout': result.stdout, 'stderr': result.stderr}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def prompt_input(message: str) -> Dict[str, Any]:
    """Prompt the user and return the input."""
    val = input(f"{message} ")
    return {'user_input': val}

# Configure chat with function‑calling, streaming, and our system prompt
try:
    config = types.GenerateContentConfig(
        tools=[
            create_file, read_file, chunk_file, update_file, delete_file,
            rename_file, move_file, list_directory, search_file,
            run_script, start_interactive, install_package, open_in_browser,
            lint_code, format_code, run_tests, git_commit, git_push, prompt_input
        ],
        temperature=0,
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=1024
    )
    chat = genai_client.chats.create(
        model='gemini-2.5-pro-preview-03-25',
        config=config
    )
except Exception as e:
    print(f"[ERROR] Failed to configure Gemini chat: {e}")
    sys.exit(1)

if __name__ == '__main__':
    print('=== Gemini Super Agent ===')
    try:
        while True:
            user_input = input('You: ').strip()
            if user_input.lower() in ('exit', 'quit'):
                print('Agent: Goodbye!')
                break
            if user_input.startswith('process_file '):
                _, file_path = user_input.split(' ', 1)
                result = process_large_file(file_path, MAX_CHUNK_SIZE)
                if result["status"] == "success":
                    for i, r in enumerate(result["responses"]):
                        print(f"Chunk {i+1} result:\n{r}\n")
                else:
                    print(f"Error: {result.get('error', result)}")
                continue
            response = chat.send_message(user_input)
            # If streaming, print as it arrives
            if hasattr(response, 'stream'):
                for part in response.stream:
                    print(part.text, end='', flush=True)
                print()
            else:
                print(f"Agent: {response.text}")
    except KeyboardInterrupt:
        print('\nAgent: Interrupted by user. Bye!')
