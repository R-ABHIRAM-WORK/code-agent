#!/usr/bin/env python3
'''
Gemini Super Agent Wrapper
Cross-platform, environment-agnostic coding assistant with robust tooling.
'''

import os
import sys
import subprocess
import platform
import shutil
import webbrowser
import logging
import textwrap
import re
import difflib
import base64
from typing import Any, Dict, List, Optional

# Optional dependencies
try:
    import requests
except ImportError:
    requests = None
try:
    import sqlite3
except ImportError:
    sqlite3 = None

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── Gemini API Import ────────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types
except ImportError as ie:
    logger.error('Missing google.genai: pip install google-generativeai')
    sys.exit(1)

# ─── Client Initialization ────────────────────────────────────────────────────
API_KEY = os.getenv('GOOGLE_API_KEY')
if not API_KEY:
    logger.error('GOOGLE_API_KEY not set')
    sys.exit(1)
try:
    genai_client = genai.Client(api_key=API_KEY)
except Exception as e:
    logger.error(f'Failed to initialize Gemini client: {e}')
    sys.exit(1)

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = '''You are a supercharged coding agent with full filesystem, shell, and Git integration.
Automatically infer and create directories for file operations.
For web projects, place HTML in 'templates/' and assets in 'static/css/', 'static/js/', 'static/images/', etc.
Support linting, formatting, testing, and version control (Git).
Detach long-running servers so the chat loop remains responsive.
Use 'run_script' to capture output for non-interactive scripts, and 'start_interactive' for scripts requiring user input.
Use 'open_in_browser' for quick HTML previews.
Remember: main.py is the file you are running—do not delete, rename, or overwrite it.
If anything is beyond your capability, just say 'I can\'t do that'.'''

# ─── Helpers ──────────────────────────────────────────────────────────────────
def chunk_file(path: str, max_tokens: int = 2000) -> List[str]:
    '''Split a text file into approximate token-sized chunks.'''
    AVG_CHARS_PER_TOKEN = 4
    chunk_size = max_tokens * AVG_CHARS_PER_TOKEN
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    except OSError as e:
        logger.error(f'Cannot read {path}: {e}')
        return []
    return textwrap.wrap(text, width=chunk_size,
                         break_long_words=False,
                         replace_whitespace=False)

# ─── File Operations ─────────────────────────────────────────────────────────
def create_file(path: str, content: str) -> Dict[str, Any]:
    '''Create or overwrite a text file, auto-creating parent dirs.'''
    try:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {'status': 'success', 'path': path}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def read_file(path: str) -> Dict[str, Any]:
    '''Read and return file content.'''
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return {'status': 'success', 'content': f.read()}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def update_file(path: str, content: str) -> Dict[str, Any]:
    '''Alias for create_file.'''
    return create_file(path, content)

def delete_file(path: str) -> Dict[str, Any]:
    '''Delete a file or directory safely.'''
    def _onerror(func, fn, excinfo):
        try:
            os.chmod(fn, 0o666)
            func(fn)
        except Exception:
            pass
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, onerror=_onerror)
            return {'status': 'deleted_directory', 'path': path}
        else:
            os.remove(path)
            return {'status': 'deleted_file', 'path': path}
    except FileNotFoundError:
        return {'status': 'error', 'error': 'path not found'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def rename_file(old_path: str, new_path: str) -> Dict[str, Any]:
    '''Rename a file or directory.'''
    try:
        if os.path.dirname(new_path):
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
        os.rename(old_path, new_path)
        return {'status': 'renamed', 'from': old_path, 'to': new_path}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def move_file(src: str, dest: str) -> Dict[str, Any]:
    '''Move a file or directory.'''
    try:
        if os.path.dirname(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(src, dest)
        return {'status': 'moved', 'from': src, 'to': dest}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def list_directory(path: str) -> Dict[str, Any]:
    '''List files and directories at a path.'''
    try:
        return {'status': 'success', 'path': path, 'items': os.listdir(path)}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

# ─── Binary File Operations (base64) ─────────────────────────────────────────
def write_binary(path: str, data_b64: str) -> Dict[str, Any]:
    '''Write base64-encoded data to a binary file.'''
    try:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        data = base64.b64decode(data_b64)
        with open(path, 'wb') as f:
            f.write(data)
        return {'status': 'success', 'path': path}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def read_binary(path: str) -> Dict[str, Any]:
    '''Read a binary file and return its base64-encoded contents.'''
    try:
        with open(path, 'rb') as f:
            data = f.read()
        return {'status': 'success', 'data_b64': base64.b64encode(data).decode('utf-8')}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

# ─── Search Utilities ─────────────────────────────────────────────────────────
def search_file(keyword: str, root: str) -> Dict[str, Any]:
    '''Stream-search keyword in files under a directory.'''
    matches: List[str] = []
    for dirpath, _, files in os.walk(root):
        for fname in files:
            try:
                with open(os.path.join(dirpath, fname), 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if keyword in line:
                            matches.append(os.path.join(dirpath, fname))
                            break
            except OSError:
                continue
    return {'status': 'success', 'matches': matches}

def search_regex(pattern: str, root: str) -> Dict[str, Any]:
    '''Search files using a regex pattern.'''
    matches: List[str] = []
    regex = re.compile(pattern)
    for dirpath, _, files in os.walk(root):
        for fname in files:
            try:
                with open(os.path.join(dirpath, fname), 'r', encoding='utf-8', errors='ignore') as f:
                    if any(regex.search(line) for line in f):
                        matches.append(os.path.join(dirpath, fname))
            except OSError:
                continue
    return {'status': 'success', 'matches': matches}

# ─── Diff and Patch ───────────────────────────────────────────────────────────
def diff_and_patch(old: str, new: str) -> Dict[str, Any]:
    '''Generate a unified diff between a file on disk and a new content string.'''
    try:
        with open(old, 'r', encoding='utf-8') as f:
            old_lines = f.readlines()
    except Exception as e:
        return {'status': 'error', 'error': str(e)}
    new_lines = new.splitlines(keepends=True)
    diff = ''.join(difflib.unified_diff(old_lines, new_lines,
                                        fromfile=old, tofile=old,
                                        lineterm=''))
    return {'status': 'success', 'diff': diff}

# ─── Command Execution ───────────────────────────────────────────────────────
def run_command(cmd: List[str], cwd: Optional[str] = None, timeout: int = 300) -> Dict[str, Any]:
    '''Execute any shell command safely.'''
    try:
        result = subprocess.run(cmd, cwd=cwd,
                                capture_output=True, text=True,
                                timeout=timeout, shell=False)
        return {
            'status': 'completed',
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        return {'status': 'error', 'error': f'Timeout after {timeout}s'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def run_script(path: str) -> Dict[str, Any]:
    '''Run a non-interactive Python script.'''
    return run_command([sys.executable, path])

def start_interactive(path: str) -> Dict[str, Any]:
    '''Launch a script in a new console/session for interactive input.'''
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
    '''Install a Python package via pip.'''
    return run_command([sys.executable, '-m', 'pip', 'install', package])

def open_in_browser(path: str) -> Dict[str, Any]:
    '''Open a file or URL in the default browser.'''
    try:
        url = path if path.startswith('http') else f'file://{os.path.abspath(path)}'
        webbrowser.open(url)
        return {'status': 'opened', 'url': url}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

# ─── Git Primitives ───────────────────────────────────────────────────────────
def git_status() -> Dict[str, Any]:
    return run_command(['git', 'status'])

def git_diff(path: Optional[str] = None) -> Dict[str, Any]:
    cmd = ['git', 'diff'] + ([path] if path else [])
    return run_command(cmd)

def git_pull(remote: str = 'origin', branch: str = 'main') -> Dict[str, Any]:
    return run_command(['git', 'pull', remote, branch])

def git_checkout(branch: str) -> Dict[str, Any]:
    return run_command(['git', 'checkout', branch])

def git_merge(source: str, target: str) -> Dict[str, Any]:
    run_command(['git', 'checkout', target])
    return run_command(['git', 'merge', source])

def git_commit(message: str) -> Dict[str, Any]:
    run_command(['git', 'add', '.'])
    return run_command(['git', 'commit', '-m', message])

def git_push(remote: str = 'origin', branch: str = 'main') -> Dict[str, Any]:
    return run_command(['git', 'push', remote, branch])

# ─── Linting & Formatting ─────────────────────────────────────────────────────
def lint_code(path: str) -> Dict[str, Any]:
    return run_command(['flake8', path])

def format_code(path: str) -> Dict[str, Any]:
    return run_command(['black', path])

# ─── Testing & Coverage ───────────────────────────────────────────────────────
def run_tests(path: str) -> Dict[str, Any]:
    return run_command(['pytest', path])

def run_coverage(path: str) -> Dict[str, Any]:
    try:
        subprocess.run(['coverage', 'run', '-m', 'pytest', path], check=True)
        result = subprocess.run(['coverage', 'report'], capture_output=True, text=True)
        return {'status': 'completed', 'report': result.stdout}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

# ─── Environment Introspection ────────────────────────────────────────────────
def get_env() -> Dict[str, Any]:
    return {'status': 'success', 'env': dict(os.environ)}

def pip_freeze() -> Dict[str, Any]:
    return run_command([sys.executable, '-m', 'pip', 'freeze'])

def pip_show(pkg: str) -> Dict[str, Any]:
    return run_command([sys.executable, '-m', 'pip', 'show', pkg])

def which(cmd: str) -> Dict[str, Any]:
    from shutil import which as _which
    path = _which(cmd)
    return {'status': 'success', 'path': path} if path else {'status': 'error', 'error': f'{cmd} not found'}

# ─── HTTP & DB Utilities ─────────────────────────────────────────────────────
def http_request(method: str, url: str, params: Optional[Dict[str, Any]] = None,
                 data: Any = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    '''Make an HTTP request (requires `requests`).'''
    if not requests:
        return {'status': 'error', 'error': 'requests not installed'}
    try:
        resp = requests.request(method.upper(), url, params=params, data=data, headers=headers)
        return {
            'status': 'success',
            'status_code': resp.status_code,
            'headers': dict(resp.headers),
            'text': resp.text
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def db_query(conn_str: str, query: str) -> Dict[str, Any]:
    '''Execute SQL query on SQLite (conn_str='sqlite:///path/to/db').'''
    if not sqlite3:
        return {'status': 'error', 'error': 'sqlite3 not available'}
    if conn_str.startswith('sqlite:///'):
        db_path = conn_str.replace('sqlite:///', '')
    else:
        return {'status': 'error', 'error': 'only sqlite supported'}
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        conn.close()
        return {'status': 'success', 'rows': rows}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

# ─── Chat Configuration ───────────────────────────────────────────────────────
try:
    config = types.GenerateContentConfig(
        tools=[
            create_file, read_file, update_file, delete_file,
            rename_file, move_file, list_directory, read_binary,
            write_binary, search_file, search_regex, diff_and_patch,
            run_command, run_script, start_interactive, install_package,
            open_in_browser, git_status, git_diff, git_pull,
            git_checkout, git_merge, git_commit, git_push,
            lint_code, format_code, run_tests, run_coverage,
            get_env, pip_freeze, pip_show, which,
            http_request, db_query
        ],
        temperature=0,
        system_instruction=SYSTEM_PROMPT
    )
    chat = genai_client.chats.create(
        model='gemini-2.5-pro-preview-03-25',
        config=config
    )
except Exception as e:
    logger.error(f'Failed to configure chat: {e}')
    sys.exit(1)

if __name__ == '__main__':
    logger.info('=== Gemini Super Agent ===')
    while True:
        user_input = input('You: ').strip()
        if user_input.lower() in ('exit', 'quit'):
            logger.info('Goodbye!')
            break
        try:
            resp = chat.send_message(user_input)
            print(f'Agent: {resp.text}')
        except Exception as e:
            logger.error(f'Chat error: {e}')
