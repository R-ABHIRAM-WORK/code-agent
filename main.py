import os
import subprocess
import sys
import platform
import shutil
import webbrowser
from typing import Any, Dict, List
from google import genai
from google.genai import types

# Initialize Gemini client
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")
genai_client = genai.Client(api_key=api_key)

# System prompt guiding the agent behavior
SYSTEM_PROMPT = (
    "You are a supercharged coding agent with full filesystem and Git integration. "
    "Automatically infer and create directories for file operations. "
    "For web projects, place HTML in 'templates/' and assets in 'static/css/', 'static/js/', 'static/images/'. and so on "
    "Support linting, formatting, testing, and version control (Git). "
    "Detach long-running servers so the chat loop remains responsive. "
    "Use 'run_script' to capture output for non-interactive scripts, and 'start_interactive' for scripts requiring user input. "
    "Use 'open_in_browser' for quick HTML previews."
)

# Tool functions

def create_file(path: str, content: str) -> Dict[str, Any]:
    """Create or overwrite a file, auto-creating parent directories."""
    try:
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {'status': 'success', 'path': path}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def read_file(path: str) -> Dict[str, Any]:
    """Read and return the content of a file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = f.read()
        return {'status': 'success', 'content': data}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def update_file(path: str, content: str) -> Dict[str, Any]:
    """Alias for create_file."""
    return create_file(path, content)


def delete_file(path: str) -> Dict[str, Any]:
    """Delete a file or directory at the given path, handling permission issues."""
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
        else:
            os.remove(path)
            return {'status': 'deleted_file', 'path': path}
    except FileNotFoundError:
        return {'status': 'error', 'error': 'path not found'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def rename_file(old_path: str, new_path: str) -> Dict[str, Any]:
    """Rename a file or directory."""
    try:
        dirpath = os.path.dirname(new_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        os.rename(old_path, new_path)
        return {'status': 'renamed', 'from': old_path, 'to': new_path}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def move_file(src: str, dest: str) -> Dict[str, Any]:
    """Move a file or directory."""
    try:
        dirpath = os.path.dirname(dest)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
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
        for file in files:
            try:
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    if keyword in f.read():
                        matches.append(os.path.join(root, file))
            except:
                continue
    return {'status': 'success', 'matches': matches}


def run_script(path: str) -> Dict[str, Any]:
    """Run a non-interactive script, capturing stdout and stderr."""
    try:
        result = subprocess.run([sys.executable, path], capture_output=True, text=True)
        return {'status': 'completed', 'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
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
            capture_output=True,
            text=True
        )
        return {
            'status': 'completed',
            'stdout': result.stdout,
            'stderr': result.stderr,
            'exit_code': result.returncode
        }
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


def chunk_file(path: str, chunk_size: int = 100, chunk_index: int = 0) -> Dict[str, Any]:
    """Read a file in chunks. Returns the specified chunk (0-based)."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        total_chunks = (len(lines) + chunk_size - 1) // chunk_size
        if chunk_index < 0 or chunk_index >= total_chunks:
            return {'status': 'error', 'error': 'chunk_index out of range', 'total_chunks': total_chunks}
        start = chunk_index * chunk_size
        end = min(start + chunk_size, len(lines))
        chunk = ''.join(lines[start:end])
        return {
            'status': 'success',
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'chunk': chunk,
            'start_line': start,
            'end_line': end - 1
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def update_file_chunk(path: str, chunk_content: str, chunk_size: int = 100, chunk_index: int = 0) -> Dict[str, Any]:
    """Update a specific chunk of a file (0-based)."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        total_chunks = (len(lines) + chunk_size - 1) // chunk_size
        if chunk_index < 0 or chunk_index >= total_chunks:
            return {'status': 'error', 'error': 'chunk_index out of range', 'total_chunks': total_chunks}
        start = chunk_index * chunk_size
        end = min(start + chunk_size, len(lines))
        new_lines = lines[:start] + chunk_content.splitlines(keepends=True) + lines[end:]
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return {'status': 'success', 'chunk_index': chunk_index, 'total_chunks': total_chunks}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

# Configure the chat with function-calling and system instruction
config = types.GenerateContentConfig(
    tools=[
        create_file, read_file, update_file, delete_file, rename_file, move_file,
        list_directory, search_file, run_script, start_interactive, install_package,
        open_in_browser, lint_code, format_code, run_tests, git_commit, git_push, prompt_input,
        chunk_file, update_file_chunk
    ],
    temperature=0,
    system_instruction=SYSTEM_PROMPT
)

# Create the chat session using the updated model version
chat = genai_client.chats.create(
    model='gemini-2.5-pro-preview-03-25',
    config=config
)

if __name__ == '__main__':
    print('=== Gemini Super Agent ===')
    while True:
        user_input = input('You: ')
        if user_input.lower() in ('exit', 'quit'):
            print('Agent: Goodbye!')
            break
        response = chat.send_message(user_input)
        print(f"Agent: {response.text}")