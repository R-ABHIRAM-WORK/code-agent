import os
import subprocess
import sys
import platform
import shutil
import webbrowser
import re
import time
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import queue
from datetime import datetime, timedelta
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TokenBucket:
    """Token bucket for rate limiting."""
    def __init__(self, tokens_per_min: int, requests_per_min: int):
        self.tokens_per_min = tokens_per_min
        self.requests_per_min = requests_per_min
        self.tokens = tokens_per_min
        self.requests = requests_per_min
        self.last_refill = datetime.now()
        self.lock = queue.Queue(maxsize=1)
        self.lock.put(True)  # Initialize lock as available

    def _refill(self):
        now = datetime.now()
        time_passed = (now - self.last_refill).total_seconds()
        if time_passed >= 60:
            self.tokens = self.tokens_per_min
            self.requests = self.requests_per_min
            self.last_refill = now

    def consume(self, tokens: int) -> bool:
        """Consume tokens if available, return True if successful."""
        try:
            self.lock.get(timeout=1)  # Acquire lock
            try:
                self._refill()
                if self.tokens >= tokens and self.requests > 0:
                    self.tokens -= tokens
                    self.requests -= 1
                    return True
                return False
            finally:
                self.lock.put(True)  # Release lock
        except queue.Empty:
            return False

class GeminiClient:
    """Wrapper for Gemini client with rate limiting and retry logic."""
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.token_bucket = TokenBucket(tokens_per_min=1000000, requests_per_min=100)
        self.max_retries = 3
        self.retry_delay = 2
        self.chat = None

    def create_chat(self):
        """Create a new chat session."""
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
        self.chat = self.client.chats.create(
            model='models/gemini-2.5-flash-preview-04-17',
            config=config
        )

    def send_message(self, message: str, max_tokens: int = 1000) -> Optional[str]:
        """Send message with rate limiting and retry logic."""
        if not self.chat:
            self.create_chat()
            
        for attempt in range(self.max_retries):
            if not self.token_bucket.consume(max_tokens):
                time.sleep(self.retry_delay)
                continue
            
            try:
                response = self.chat.send_message(message)
                return response.text
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    # Recreate chat session on failure
                    self.create_chat()
                else:
                    raise

# Initialize Gemini client with rate limiting
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required")
genai_client = GeminiClient(api_key=api_key)

# System prompt with token optimization
SYSTEM_PROMPT = (
    "You are a supercharged coding agent with full filesystem and Git integration. "
    "Automatically infer and create directories for file operations. "
    "For web projects, place HTML in 'templates/' and assets in 'static/css/', 'static/js/', 'static/images/'. "
    "Support linting, formatting, testing, and version control (Git). "
    "Detach long-running servers so the chat loop remains responsive. "
    "Use 'run_script' to capture output for non-interactive scripts, and 'start_interactive' for scripts requiring user input. "
    "Use 'open_in_browser' for quick HTML previews. "
    "Never use main.py, README.md, and requirements.txt files to write any code and never delete them in current directory. "
    "Create new files only within subdirectories."
)

@dataclass
class OperationResult:
    """Standardized operation result structure."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    status_code: int = 200

def safe_file_operation(path: str) -> Optional[Path]:
    """Safely resolve and validate file paths."""
    try:
        safe_path = Path(path).resolve()
        base_path = Path.cwd().resolve()
        if base_path in safe_path.parents:
            return safe_path
        return None
    except (ValueError, RuntimeError):
        return None

def create_file(path: str, content: str) -> OperationResult:
    """Create or overwrite a file, auto-creating parent directories."""
    try:
        safe_path = safe_file_operation(path)
        if not safe_path:
            return OperationResult(False, error="Invalid path", status_code=400)
            
        dirpath = safe_path.parent
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
            
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return OperationResult(True, data={'path': str(safe_path)})
    except Exception as e:
        logger.error(f"Error creating file {path}: {str(e)}")
        return OperationResult(False, error=str(e), status_code=500)

def read_file(path: str) -> OperationResult:
    """Read and return the content of a file."""
    try:
        safe_path = safe_file_operation(path)
        if not safe_path:
            return OperationResult(False, error="Invalid path", status_code=400)
            
        with open(safe_path, 'r', encoding='utf-8') as f:
            data = f.read()
        return OperationResult(True, data={'content': data})
    except Exception as e:
        logger.error(f"Error reading file {path}: {str(e)}")
        return OperationResult(False, error=str(e), status_code=500)

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

def chunk_file(path: str, chunk_size: int, chunk_index: int) -> Dict[str, Any]:
    """Read a file in chunks (line-based, memory efficient, resumable, robust for huge files). Returns the specified chunk (0-based)."""
    if chunk_size is None:
        chunk_size = 100
    if chunk_index is None:
        chunk_index = 0
    try:
        total_lines = 0
        chunk_lines = []
        start = chunk_index * chunk_size
        end = start + chunk_size
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= start and i < end:
                    chunk_lines.append(line)
                total_lines += 1
        total_chunks = (total_lines + chunk_size - 1) // chunk_size
        if chunk_index < 0 or chunk_index >= total_chunks:
            return {'status': 'error', 'error': 'chunk_index out of range', 'total_chunks': total_chunks}
        chunk = ''.join(chunk_lines)
        return {
            'status': 'success',
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'chunk': chunk,
            'start_line': start,
            'end_line': min(end, total_lines) - 1
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def update_file_chunk(path: str, chunk_content: str, chunk_size: int, chunk_index: int) -> Dict[str, Any]:
    """Update a specific chunk of a file (0-based, memory efficient, robust for huge files)."""
    if chunk_size is None:
        chunk_size = 100
    if chunk_index is None:
        chunk_index = 0
    try:
        temp_path = path + '.tmp'
        start = chunk_index * chunk_size
        end = start + chunk_size
        total_lines = 0
        written_lines = 0
        chunk_lines = chunk_content.splitlines(keepends=True)
        with open(path, 'r', encoding='utf-8') as src, open(temp_path, 'w', encoding='utf-8') as dst:
            for i, line in enumerate(src):
                if i == start:
                    for cl in chunk_lines:
                        dst.write(cl)
                        written_lines += 1
                if i < start or i >= end:
                    dst.write(line)
                total_lines += 1
            # If chunk is appended at the end
            if start >= total_lines:
                for cl in chunk_lines:
                    dst.write(cl)
                    written_lines += 1
        os.replace(temp_path, path)
        total_chunks = (total_lines + chunk_size - 1) // chunk_size
        if start >= total_lines:
            total_chunks = (start + written_lines + chunk_size - 1) // chunk_size
        return {'status': 'success', 'chunk_index': chunk_index, 'total_chunks': total_chunks}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def is_multifile_request(user_input: str) -> bool:
    """Detect if the user is requesting a multi-file or full project generation or edit."""
    # Exclude generic run/test/output commands
    generic_run_phrases = [
        'run', 'run the code', 'execute', 'start', 'test', 'show output', 'output', 'show result', 'show results', 'print', 'print output', 'print result', 'print results', 'display', 'display output', 'display result', 'display results', 'launch', 'open', 'open app', 'open project', 'open file', 'open folder', 'open directory', 'open website', 'open page', 'preview', 'preview app', 'preview project', 'preview file', 'preview folder', 'preview directory', 'preview website', 'preview page', 'just run', 'just execute', 'just test', 'just show', 'just print', 'just display', 'just open', 'just preview', 'run main', 'run main.py', 'run app', 'run app.py', 'run script', 'run program', 'run project', 'run this', 'run it', 'run all', 'run everything', 'run my code', 'run my project', 'run my app', 'run my script', 'run my program', 'run my file', 'run my folder', 'run my directory', 'run my website', 'run my page', 'run my preview', 'run my output', 'run my result', 'run my results', 'run my print', 'run my display', 'run my launch', 'run my open', 'run my preview', 'run code', 'run codes', 'run all code', 'run all codes', 'run all files', 'run all scripts', 'run all programs', 'run all projects', 'run all apps', 'run all websites', 'run all pages', 'run all previews', 'run all outputs', 'run all results', 'run all prints', 'run all displays', 'run all launches', 'run all opens', 'run all previews', 'run everything', 'run anything', 'run something', 'run nothing', 'run whatever', 'run whichever', 'run whichever code', 'run whichever file', 'run whichever script', 'run whichever program', 'run whichever project', 'run whichever app', 'run whichever website', 'run whichever page', 'run whichever preview', 'run whichever output', 'run whichever result', 'run whichever results', 'run whichever print', 'run whichever display', 'run whichever launch', 'run whichever open', 'run whichever preview', 'run whichever main', 'run whichever main.py', 'run whichever app', 'run whichever app.py', 'run whichever script', 'run whichever program', 'run whichever project', 'run whichever file', 'run whichever folder', 'run whichever directory', 'run whichever website', 'run whichever page', 'run whichever preview', 'run whichever output', 'run whichever result', 'run whichever results', 'run whichever print', 'run whichever display', 'run whichever launch', 'run whichever open', 'run whichever preview',
    ]
    if any(phrase == user_input.strip().lower() for phrase in generic_run_phrases):
        return False
    # Only trigger for clear project/file structure creation or multi-file edit requests
    keywords = [
        'full stack', 'full project', 'all files', 'create all', 'generate all',
        'multiple files', 'folders', 'structure', 'backend and frontend', 'full code', 'ready to deploy', 'not just a prototype',
        'folder', 'website', 'project', 'app', 'dashboard', 'study plan', 'resources', 'path', 'daily', 'monthly', 'test', 'quiz', 'assignment', 'deploy', 'ui', 'single user', 'multi-file', 'multi file', 'multi-project', 'multi project', 'subdirectory', 'subdirectories', 'sub-folder', 'subfolders', 'subfolder', 'subfolders', 'plan', 'progress', 'graph', 'graphs', 'track', 'tracking', 'visualize', 'visualization', 'visualisations', 'visualisation', 'visual', 'modern', 'production-ready', 'production ready', 'deploy-ready', 'deploy ready', 'no placeholders', 'fully implemented', 'end-to-end', 'end to end', 'comprehensive', 'complete', 'study', 'studies', 'fix all errors', 'fix errors in all files', 'fix errors in project', 'fix errors in codebase', 'fix errors in all code', 'edit all files', 'edit all', 'edit project', 'edit codebase', 'edit everything', 'edit the whole project', 'edit the entire project', 'edit the whole codebase', 'edit the entire codebase', 'edit the whole app', 'edit the entire app', 'edit the whole website', 'edit the entire website', 'edit the whole folder', 'edit the entire folder', 'edit the whole directory', 'edit the entire directory', 'edit the whole file', 'edit the entire file', 'edit the whole script', 'edit the entire script', 'edit the whole program', 'edit the entire program', 'edit the whole preview', 'edit the entire preview', 'edit the whole output', 'edit the entire output', 'edit the whole result', 'edit the entire result', 'edit the whole results', 'edit the entire results', 'edit the whole print', 'edit the entire print', 'edit the whole display', 'edit the entire display', 'edit the whole launch', 'edit the entire launch', 'edit the whole open', 'edit the entire open', 'edit the whole preview', 'edit the entire preview',
    ]
    return any(k in user_input.lower() for k in keywords)

def extract_file_list(user_input: str) -> list:
    """Extract a list of files to generate from the user input. Simple heuristic for demo purposes."""
    # You can improve this with NLP or prompt the user for confirmation
    files = []
    if 'backend' in user_input or 'flask' in user_input:
        files.append('app.py')
    if 'frontend' in user_input or 'html' in user_input:
        files.append('templates/index.html')
    if 'css' in user_input:
        files.append('static/css/style.css')
    if 'js' in user_input or 'javascript' in user_input:
        files.append('static/js/script.js')
    # Default for full stack
    if not files:
        files = ['app.py', 'templates/index.html', 'static/css/style.css', 'static/js/script.js']
    return files

def strip_code_block_markers(text: str) -> str:
    """Remove leading/trailing code block markers and clean up the content."""
    # Remove all code block markers (```python, ```html, ```, etc.)
    text = re.sub(r'^```[a-zA-Z0-9]*\s*', '', text.strip())
    text = re.sub(r'```\s*$', '', text.strip())
    
    # Remove any remaining markdown formatting
    text = re.sub(r'^\s*`{3,}.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*`{1,2}.*$', '', text, flags=re.MULTILINE)
    
    # Remove any leading/trailing whitespace
    text = text.strip()
    
    # Remove any empty lines at the start and end
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    return '\n'.join(lines)

def generate_file_content(file_path: str, user_input: str) -> str:
    """Generate content for a file with improved code block handling and error checking."""
    # Determine file type for better prompt
    file_ext = os.path.splitext(file_path)[1].lower()
    file_type = {
        '.py': 'Python',
        '.html': 'HTML',
        '.css': 'CSS',
        '.js': 'JavaScript',
        '.json': 'JSON',
        '.md': 'Markdown',
        '.txt': 'Text',
        '.sql': 'SQL'
    }.get(file_ext, 'Text')
    
    prompt = (
        f"Generate the complete code for the file '{file_path}' as part of this project: {user_input}\n"
        f"File type: {file_type}\n"
        f"Requirements:\n"
        f"1. Output only the raw code without any markdown formatting or code block markers\n"
        f"2. Do not include any explanations or comments about the code\n"
        f"3. Ensure proper indentation and formatting\n"
        f"4. Include all necessary imports and dependencies\n"
        f"5. Follow best practices for {file_type} files\n"
        f"6. Make sure the code is complete and runnable"
    )
    
    try:
        response = genai_client.send_message(prompt, max_tokens=2000)
        if response is None:
            logger.error(f"Empty response received for {file_path}")
            return ""
            
        content = strip_code_block_markers(response)
        if not content.strip():
            logger.warning(f"Empty content generated for {file_path}")
            return ""
            
        return content
    except Exception as e:
        logger.error(f"Error generating content for {file_path}: {str(e)}")
        return ""

def get_file_plan(user_input: str) -> list:
    """Ask Gemini to generate a file/folder plan for the project."""
    plan_prompt = (
        f"Given this project description, list all files and folders (with relative paths) needed. "
        f"Output as a JSON array of file paths only, no explanations.\nProject description: {user_input}"
    )
    response = genai_client.send_message(plan_prompt)
    import json
    try:
        file_list = json.loads(response)
        if isinstance(file_list, list):
            return file_list
    except Exception:
        pass
    # Fallback: try to extract file paths from text
    return [line.strip('- ').strip() for line in response.splitlines() if '.' in line]

def supervisor_generate_project(user_input: str):
    """Generate project files with improved error handling and token management."""
    logger.info('Generating file/folder plan...')
    try:
        file_list = get_file_plan(user_input)
        if not file_list:
            logger.error("No files to generate in the plan")
            return
            
        logger.info(f'File plan: {file_list}')
        
        # Create project directory if it doesn't exist
        project_dir = os.path.abspath("project")
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
            logger.info(f"Created project directory: {project_dir}")
        
        for file_path in file_list:
            # Clean and validate the file path
            clean_path = re.sub(r'^[\'"\s,]+|[\'"\s,]+$', '', file_path)
            if not clean_path:
                logger.warning(f"Skipping empty file path")
                continue
                
            # Handle paths that already include the project directory
            if clean_path.startswith("project/"):
                # Remove the project prefix to avoid double nesting
                relative_path = clean_path[8:]  # Remove "project/" prefix
            else:
                relative_path = clean_path
                
            # Ensure we only have the filename part for checking forbidden files
            base_name = os.path.basename(relative_path).lower()
            if base_name in ['main.py', 'readme.md', 'requirements.txt']:
                logger.info(f'Skipping forbidden file: {base_name}')
                continue
                
            # Create the full path within the project directory
            full_path = os.path.join(project_dir, relative_path)
            logger.info(f'Generating {full_path}...')
            
            # Create parent directories if they don't exist
            parent_dir = os.path.dirname(full_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            # Handle special file types
            if full_path.lower().endswith('.db'):
                if not os.path.exists(full_path):
                    open(full_path, 'wb').close()
                    logger.info(f'{full_path} (empty database file) created.')
                continue
                
            if full_path.lower().endswith('.json'):
                if not os.path.exists(full_path):
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write('[]')
                    logger.info(f'{full_path} (empty JSON array) created.')
                continue
                
            # Handle binary files like images
            if full_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.ico')):
                # Create an empty placeholder image
                try:
                    with open(full_path, 'wb') as f:
                        # Write a minimal transparent PNG
                        f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')
                    logger.info(f'{full_path} (placeholder image) created.')
                except Exception as e:
                    logger.error(f"Failed to create placeholder image {full_path}: {str(e)}")
                continue
                
            # Generate content for other files
            code = generate_file_content(relative_path, user_input)
            if not code:
                # Create an empty file as fallback
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write('# Generated file\n')
                logger.info(f'{full_path} (empty file) created as fallback.')
                continue
                
            # Write the content to the file
            try:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                logger.info(f'{full_path} generated successfully.')
                
                # Verify the generated content
                if '```' in code:
                    logger.warning(f'Content for {full_path} contains code block markers.')
            except Exception as e:
                logger.error(f'Failed to write {full_path}: {str(e)}')
                
    except Exception as e:
        logger.error(f'Error in project generation: {str(e)}')
        import traceback
        logger.error(traceback.format_exc())

def verify_file_content(file_path: str) -> OperationResult:
    """Verify the generated file content for common issues."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for common issues
        issues = []
        
        # Check for remaining code block markers
        if '```' in content:
            issues.append("Found remaining code block markers")
            
        # Check for incomplete code blocks
        if content.count('```') % 2 != 0:
            issues.append("Found unclosed code block")
            
        # Check for empty content
        if not content.strip():
            issues.append("File is empty")
            
        if issues:
            return OperationResult(False, error=", ".join(issues))
            
        return OperationResult(True)
    except Exception as e:
        return OperationResult(False, error=str(e))

def parse_python_traceback(traceback_str: str):
    """
    Parse a Python traceback string and extract a list of (file_path, line_number) tuples for error locations.
    Returns a list of dicts: [{'file': file_path, 'line': line_number}]
    """
    import re
    error_locations = []
    pattern = re.compile(r'File "([^"]+)", line (\d+)')
    for match in pattern.finditer(traceback_str):
        file_path, line_str = match.groups()
        error_locations.append({'file': file_path, 'line': int(line_str)})
    return error_locations

def agent_per_file_edit(error_message: str, traceback_str: str, minimal_files: bool = True):
    """
    Parse the error traceback, assign an agent to each affected file (or chunk), and fix the error.
    Only edit the exact files/lines needed. Validate fixes after each change.
    If minimal_files is True, do not generate or edit unnecessary files (like README, docs, or tests).
    """
    error_locations = parse_python_traceback(traceback_str)
    for loc in error_locations:
        file_path = loc['file']
        line_number = loc['line']
        # For large files, determine chunk
        file_size = os.path.getsize(file_path)
        if file_size > 100_000:
            chunk_size = 100
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            total_chunks = (len(lines) + chunk_size - 1) // chunk_size
            chunk_index = (line_number - 1) // chunk_size
            chunk_lines = lines[chunk_index*chunk_size:(chunk_index+1)*chunk_size]
            chunk_content = ''.join(chunk_lines)
            print(f"Agent for {file_path} [chunk {chunk_index}]: analyzing and fixing error at line {line_number}...")
            # ...agent logic to fix errors in chunk_content...
            # After fix, validate by running code/tests
        else:
            print(f"Agent for {file_path}: analyzing and fixing error at line {line_number}...")
            # ...agent logic to fix errors in the file...
            # After fix, validate by running code/tests
    if minimal_files:
        print("Minimal files mode: No extra files (README, docs, tests) will be generated or edited unless explicitly requested.")

def get_multi_line_input() -> str:
    """Get multi-line input from user until 'END' is entered on a new line."""
    print("Enter your prompt (type 'END' on a new line when finished):")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        lines.append(line)
    return '\n'.join(lines)

def main():
    """Main function with improved error handling and token management."""
    logger.info('=== Gemini Super Agent ===')
    print("Welcome to Gemini Super Agent!")
    print("To enter multi-line input, type your text and press Enter.")
    print("When finished, type 'END' on a new line and press Enter.")
    print("To exit, type 'exit' or 'quit'.")
    print()
    
    while True:
        try:
            user_input = get_multi_line_input()
            if not user_input.strip():
                continue
                
            if user_input.lower() in ('exit', 'quit'):
                logger.info('Goodbye!')
                break
                
            if is_multifile_request(user_input):
                logger.info('Detected multi-file/full-project request. Using supervisor workflow...')
                supervisor_generate_project(user_input)
            else:
                response = genai_client.send_message(user_input)
                print(f"Agent: {response}")
                
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")
            print(f"Agent: An error occurred. Please try again later.")

if __name__ == '__main__':
    main()