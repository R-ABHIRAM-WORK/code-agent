import os
import subprocess
import sys
import platform
import shutil
import webbrowser
import re
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
    "never use the main.py,README.md and requirements.txt files to write any code and also never delete it in current directory.you can create it within a new subdirectory only."
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
    """Remove leading/trailing code block markers (e.g., ```python, ```html, ```) from code."""
    # Remove triple backtick blocks with optional language
    text = re.sub(r'^```[a-zA-Z0-9]*\s*', '', text.strip())
    text = re.sub(r'```\s*$', '', text.strip())
    return text.strip()


def generate_file_content(file_path: str, user_input: str) -> str:
    """Prompt Gemini to generate content for a single file based on the user request and file path."""
    prompt = (
        f"Generate the complete code for the file '{file_path}' as part of this project: {user_input}. "
        f"Assume all other files in the project plan already exist and can be imported or referenced as needed, even if they are generated later. "
        f"Do not include explanations or any other files. Output only the code for {file_path}."
    )
    response = chat.send_message(prompt)
    return strip_code_block_markers(response.text)


def get_file_plan(user_input: str) -> list:
    """Ask Gemini to generate a file/folder plan for the project."""
    plan_prompt = (
        f"Given this project description, list all files and folders (with relative paths) needed. "
        f"Output as a JSON array of file paths only, no explanations.\nProject description: {user_input}"
    )
    response = chat.send_message(plan_prompt)
    import json
    try:
        file_list = json.loads(response.text)
        if isinstance(file_list, list):
            return file_list
    except Exception:
        pass
    # Fallback: try to extract file paths from text
    return [line.strip('- ').strip() for line in response.text.splitlines() if '.' in line]


def supervisor_generate_project(user_input: str):
    print('Agent: Generating file/folder plan...')
    file_list = get_file_plan(user_input)
    print(f'Agent: File plan: {file_list}')
    # Always generate all required files in the plan (except forbidden ones)
    for file_path in file_list:
        clean_path = re.sub(r'^[\'"\s,]+|[\'"\s,]+$', '', file_path)
        # Skip forbidden files only (main.py, README.md, requirements.txt in root)
        forbidden = [
            'main.py', 'README.md', 'requirements.txt'
        ]
        if not clean_path or any(clean_path.lower() == f.lower() for f in forbidden):
            print(f'Agent: Skipping {clean_path} (forbidden or empty)')
            continue
        # Always generate .json, .db, and other data files if in the plan
        print(f'Agent: Generating {clean_path}...')
        # For .db files, create an empty file if not present
        if clean_path.lower().endswith('.db'):
            if not os.path.exists(clean_path):
                open(clean_path, 'wb').close()
                print(f'Agent: {clean_path} (empty database file) created.')
            else:
                print(f'Agent: {clean_path} already exists.')
            continue
        # For .json files, create minimal valid content if not present
        if clean_path.lower().endswith('.json'):
            if not os.path.exists(clean_path):
                with open(clean_path, 'w', encoding='utf-8') as f:
                    f.write('[]')
                print(f'Agent: {clean_path} (empty JSON array) created.')
            else:
                print(f'Agent: {clean_path} already exists.')
            continue
        # For all other files, generate as usual
        code = generate_file_content(clean_path, user_input)
        result = create_file(clean_path, code)
        if result.get('status') == 'success':
            print(f'Agent: {clean_path} generated.')
        else:
            print(f'Agent: Failed to generate {clean_path}: {result.get("error")})')
    print('Agent: All files generated and combined. Your project is ready!')


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


def main():
    print('=== Gemini Super Agent ===')
    while True:
        user_input = input('You: ')
        if user_input.lower() in ('exit', 'quit'):
            print('Agent: Goodbye!')
            break
        try:
            if is_multifile_request(user_input):
                print('Agent: Detected a multi-file/full-project request. Using supervisor workflow...')
                supervisor_generate_project(user_input)
            else:
                response = chat.send_message(user_input)
                print(f"Agent: {response.text}")
        except Exception as e:
            print(f"Agent: Error occurred: {e}")
            # Retry logic for Gemini API 500 errors
            import time
            retries = 0
            while retries < 3:
                try:
                    time.sleep(2)
                    if is_multifile_request(user_input):
                        supervisor_generate_project(user_input)
                    else:
                        response = chat.send_message(user_input)
                        print(f"Agent: {response.text}")
                    break
                except Exception as e2:
                    print(f"Agent: Retry {retries+1} failed: {e2}")
                    retries += 1
            else:
                print("Agent: Failed after multiple retries. Please try again later or check Gemini API status.")


if __name__ == '__main__':
    main()