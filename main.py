import os
import subprocess
import sys
import platform
import shutil # Used by shutil.rmtree indirectly via env_utils, but not directly in main.py
import webbrowser # Used by env_utils.open_in_browser
import re # Used in strip_code_block_markers and _generate_project_files
import time # Used in GeminiClient and main loop
import logging
from typing import Any, Dict, List, Optional, Union # Union might be unused
from pathlib import Path # Used in supervisor_generate_project's helpers
# from dataclasses import dataclass # No longer used in main.py
# from concurrent.futures import ThreadPoolExecutor # Not used
import queue # Used by TokenBucket
from datetime import datetime, timedelta # Used by TokenBucket
from google import genai
from google.genai import types

# Import file utility functions
import file_utils # For tools used in GeminiClient
from file_utils import OperationResult, verify_file_content # verify_file_content was moved here
# Note: Other specific file_utils functions are called as file_utils.function_name

# Import environment and process utility functions
import env_utils # For tools used in GeminiClient
# Note: Specific env_utils functions are called as env_utils.function_name
# No direct calls to env_utils functions from main.py's global scope currently,
# so the broad `from env_utils import ...` is not strictly necessary here if GeminiClient handles all.
# However, keeping it doesn't harm and allows flexibility if main were to call them.
from env_utils import (
    # run_script, # Example: if main were to call these directly
    # start_interactive,
    # install_package,
    # open_in_browser,
    # lint_code,
    # format_code,
    # run_tests,
    # git_commit,
    # git_push
) # Emptying this as no direct calls from main's scope. GeminiClient uses env_utils.func_name

# Configure logging
logging.basicConfig(
    open_in_browser,
    lint_code,
    format_code,
    run_tests,
    git_commit,
    git_push
)

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
        """Create a new chat session, handling potential errors."""
        logger.info("Attempting to create a new chat session.")
        try:
            config = types.GenerateContentConfig(
                tools=[
                    file_utils.create_file, file_utils.read_file, file_utils.update_file,
                    file_utils.delete_file, file_utils.rename_file, file_utils.move_file,
                    file_utils.list_directory, file_utils.search_file,
                    file_utils.chunk_file, file_utils.update_file_chunk, # chunk_file is now in file_utils
                    env_utils.run_script, env_utils.start_interactive, env_utils.install_package,
                    env_utils.open_in_browser, env_utils.lint_code, env_utils.format_code,
                    env_utils.run_tests, env_utils.git_commit, env_utils.git_push,
                    prompt_input  # Assuming prompt_input remains in main.py or is imported
                ],
                temperature=0,
                system_instruction=SYSTEM_PROMPT
            )
            self.chat = self.client.chats.create(
                model='models/gemini-2.5-flash-preview-04-17', # Consider making model configurable
                config=config
            )
            logger.info("Successfully created a new chat session.")
        except Exception as e:
            logger.error(f"Failed to create chat session: {str(e)}", exc_info=True)
            # Depending on the error, self.chat might be None or in an invalid state.
            # The caller (send_message) should handle this.
            self.chat = None # Ensure chat is None if creation fails
            raise # Re-raise the exception so send_message knows creation failed

    def send_message(self, message: str, max_tokens: int = 1000) -> Optional[str]:
        """Send message with rate limiting and retry logic."""
        if not self.chat:
            try:
                self.create_chat()
            except Exception as e: # Catch errors from create_chat
                logger.error(f"Failed to initialize chat for send_message due to: {str(e)}")
                return None # Or raise, depending on desired behavior for critical chat init failure

        # If chat is still None after attempting creation, means create_chat failed critically.
        if not self.chat:
             logger.error("Chat session is not available. Cannot send message.")
             return None

        for attempt in range(self.max_retries):
            logger.info(f"Attempt {attempt + 1} of {self.max_retries} to send message.")
            if not self.token_bucket.consume(max_tokens):
                logger.warning(f"Rate limit exceeded. Waiting for {self.retry_delay} seconds before retrying.")
                time.sleep(self.retry_delay)
                continue
            
            try:
                response = self.chat.send_message(message)
                logger.info("Message sent and response received successfully.")
                return response.text
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} to send message failed: {str(e)}", exc_info=True)
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    # Recreate chat session on failure as a recovery strategy
                    try:
                        logger.info("Attempting to recreate chat session due to send error.")
                        self.create_chat()
                        if not self.chat: # If chat creation failed again
                            logger.error("Failed to recreate chat session. Aborting retries for this message.")
                            return None # Or raise specific error
                    except Exception as ce:
                        logger.error(f"Critical error during chat recreation: {str(ce)}. Aborting retries.")
                        return None # Or raise
                else:
                    logger.error("Maximum retries reached. Failed to send message.")
                    # Not raising the exception here to allow the application to continue if desired.
                    # Depending on requirements, `raise` could be used.
                    return None # Indicate failure after max retries

# Initialize Gemini client with rate limiting
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required")
genai_client = GeminiClient(api_key=api_key)

# System prompt with token optimization
SYSTEM_PROMPT = (
    "You are a coding agent with filesystem access and Git integration. "
    "Manage files and directories, ensuring web project assets (HTML, CSS, JS) are in 'templates/' and 'static/' respectively. "
    "Utilize tools for linting, formatting, testing, and version control (Git). "
    "Run scripts non-interactively (capturing output) or interactively. Detach long-running servers. "
    "Preview HTML using a browser. Avoid modifying 'main.py', 'README.md', 'requirements.txt'. Create new files in subdirectories."
)

# @dataclass
# class OperationResult: # This definition is now removed, using the one from file_utils.py
#     """Standardized operation result structure."""
#     success: bool
#     data: Optional[Any] = None
#     error: Optional[str] = None
#     status_code: int = 200

def prompt_input(message: str) -> Dict[str, Any]:
    """Prompt the user and return the input."""
    val = input(f"{message} ")
    return {'user_input': val}

# chunk_file function was moved to file_utils.py

def is_multifile_request(user_input: str) -> bool:
    """
    Detect if the user is requesting a multi-file or full project generation or edit.
    This function uses a keyword-based approach. For more complex scenarios,
    NLP or more sophisticated pattern matching might be considered.
    """
    # Exclude generic run/test/output commands to avoid false positives
    generic_run_phrases = [
        'run', 'run the code', 'execute', 'start', 'test', 'show output', 'output', 
        'show result', 'show results', 'print', 'print output', 'print result', 'print results', 
        'display', 'display output', 'display result', 'display results', 'launch', 'open', 
        'open app', 'open project', 'open file', 'open folder', 'open directory', 
        'open website', 'open page', 'preview', 'preview app', 'preview project', 
        'preview file', 'preview folder', 'preview directory', 'preview website', 'preview page',
        # Simplified "just run..." type phrases
        'just run', 'just execute', 'just test', 'just show', 'just print', 'just display', 
        'just open', 'just preview',
        # Simplified "run main/app..." type phrases
        'run main', 'run main.py', 'run app', 'run app.py', 'run script', 'run program', 'run project',
        'run this', 'run it', 'run all', 'run everything',
        # Simplified "run my..." type phrases
        'run my code', 'run my project', 'run my app', 'run my script', 'run my program',
        # Simplified "run all..." type phrases
        'run all code', 'run all files', 'run all scripts', 'run all programs', 'run all projects',
        # Simplified "run whichever..." type phrases
        'run whichever code', 'run whichever file', 'run whichever script', 'run whichever program'
    ]
    # Convert user input to lowercase and strip whitespace for robust comparison
    normalized_input = user_input.strip().lower()
    if any(phrase == normalized_input for phrase in generic_run_phrases):
        return False
        
    # Keywords that strongly indicate a multi-file or project-level request.
    # This list can be expanded or refined.
    project_keywords = [
        'full stack', 'full project', 'all files', 'create all', 'generate all', 
        'multiple files', 'folders', 'structure', 'backend and frontend', 'full code', 
        'ready to deploy', 'not just a prototype', 'folder', 'website', 'project', 'app', 
        'dashboard', 'study plan', 'resources', 'path', 'daily', 'monthly', 'test', 'quiz', 
        'assignment', 'deploy', 'ui', 'single user', 'multi-file', 'multi file', 
        'multi-project', 'multi project', 'subdirectory', 'subdirectories', 'sub-folder', 
        'subfolders', 'subfolder', 'subfolders', 'plan', 'progress', 'graph', 'graphs', 
        'track', 'tracking', 'visualize', 'visualization', 'visualisations', 'visualisation', 
        'visual', 'modern', 'production-ready', 'production ready', 'deploy-ready', 
        'deploy ready', 'no placeholders', 'fully implemented', 'end-to-end', 'end to end', 
        'comprehensive', 'complete', 'study', 'studies',
        # Keywords for editing multiple files or project-wide edits
        'fix all errors', 'fix errors in all files', 'fix errors in project', 
        'fix errors in codebase', 'fix errors in all code', 'edit all files', 'edit all', 
        'edit project', 'edit codebase', 'edit everything', 'edit the whole project', 
        'edit the entire project', 'edit the whole codebase', 'edit the entire codebase'
    ]
    return any(keyword in normalized_input for keyword in project_keywords)

# --- Project Generation Helper Functions ---
# These functions support the supervisor_generate_project workflow.

def strip_code_block_markers(text: str) -> str:
    """
    Removes leading/trailing code block markers (e.g., ```python, ```) and
    other markdown formatting from a string.

    Args:
        text: The input string, potentially with code block markers.

    Returns:
        The cleaned string without code block markers or common markdown.
    """
    if not isinstance(text, str):
        return "" # Or raise TypeError
    # Remove all code block markers (```python, ```html, ```, etc.)
    processed_text = re.sub(r'^```[a-zA-Z0-9]*\s*', '', text.strip())
    processed_text = re.sub(r'```\s*$', '', processed_text.strip())
    
    # Remove any remaining markdown formatting (like single or double backticks if they are on their own lines)
    # This is a simple heuristic and might need refinement for complex markdown.
    processed_text = re.sub(r'^\s*`{1,3}.*`{1,3}\s*$', '', processed_text, flags=re.MULTILINE) # Full line backticks
    
    # Remove any leading/trailing whitespace again after regex
    processed_text = processed_text.strip()
    
    # Remove any empty lines at the start and end
    lines = processed_text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    return '\n'.join(lines)

def generate_file_content(file_path: str, user_input: str) -> str:
    """
    Generates content for a specific file using the Gemini client.
    This function formulates a prompt to the AI to generate code/text for the given file path,
    considering the overall user project description.

    Args:
        file_path: The relative path of the file for which to generate content (e.g., "src/app.py").
        user_input: The original user prompt describing the project.

    Returns:
        A string containing the generated content for the file. Returns an empty string on error
        or if no content is generated.
    """
    # Determine file type for better prompt context
    file_ext = os.path.splitext(file_path)[1].lower() # Use os.path.splitext
    # More comprehensive mapping of file extensions to types for better prompting
    file_type_map = {
        '.py': 'Python', '.html': 'HTML', '.css': 'CSS', '.js': 'JavaScript',
        '.json': 'JSON', '.md': 'Markdown', '.txt': 'Text', '.sql': 'SQL',
        '.xml': 'XML', '.yaml': 'YAML', '.yml': 'YAML', '.java': 'Java',
        '.c': 'C', '.cpp': 'C++', '.h': 'C/C++ Header', '.cs': 'C#',
        '.php': 'PHP', '.rb': 'Ruby', '.go': 'Go', '.rs': 'Rust', '.swift': 'Swift',
        '.kt': 'Kotlin', '.ts': 'TypeScript', '.sh': 'Shell Script', '.pl': 'Perl'
    }
    file_type_description = file_type_map.get(file_ext, f"{file_ext} file" if file_ext else "Text")
    
    # Refined prompt for content generation
    prompt = (
        f"Please generate the complete and production-ready code for the file named '{file_path}'.\n"
        f"This file is part of a larger project described as: '{user_input}'.\n"
        f"The file type is: {file_type_description}.\n"
        f"Ensure the generated code is functional, follows best practices for {file_type_description}, "
        f"and includes all necessary imports or dependencies within the code itself.\n"
        f"Output only the raw code for '{file_path}'. Do not include any explanations, comments about the code generation process, or markdown code block markers."
    )
    
    logger.info(f"Generating content for '{file_path}' using prompt (first 100 chars): {prompt[:100]}...")
    try:
        # Assuming genai_client is a globally available Gemini client instance
        response_text = genai_client.send_message(prompt, max_tokens=4096) # Increased max_tokens
        
        if response_text is None:
            logger.error(f"No response received from AI for {file_path} generation.")
            return "" # Return empty string if no response
            
        # Clean the response to remove any surrounding explanations or markdown
        cleaned_content = strip_code_block_markers(response_text)
        
        if not cleaned_content.strip():
            logger.warning(f"Empty content generated for {file_path} after cleaning.")
            # Optionally, try a fallback prompt or return a placeholder
            return f"# Placeholder for {file_path} - AI returned empty content\n"
            
        logger.info(f"Successfully generated and cleaned content for {file_path}.")
        return cleaned_content
    except Exception as e:
        logger.error(f"Error generating content for file '{file_path}': {str(e)}", exc_info=True)
        return "" # Return empty string on error

def get_file_plan(user_input: str) -> List[str]:
    """
    Asks the Gemini client to generate a file and folder plan for the project based on user input.
    The expected output is a JSON array of relative file paths.

    Args:
        user_input: The user's description of the project.

    Returns:
        A list of strings, where each string is a relative file path.
        Returns an empty list if the plan generation fails or the response is not as expected.
    """
    plan_prompt = (
        f"Based on the following project description, provide a comprehensive list of all files and folders "
        f"(using relative paths like 'src/app.py' or 'static/css/style.css') necessary for the project.\n"
        f"Output this list as a JSON array of strings. Each string should be a file path. Do not include any explanations or surrounding text.\n"
        f"Project description: \"{user_input}\""
    )
    logger.info("Requesting file plan from AI...")
    response = genai_client.send_message(plan_prompt, max_tokens=1024) # Max tokens for plan
    
    if not response:
        logger.error("No response received from AI for file plan generation.")
        return []

    import json # Import json here as it's only used in this function
    try:
        # Attempt to strip markdown before parsing JSON, as AI might wrap JSON in ```json ... ```
        cleaned_response = strip_code_block_markers(response)
        file_list = json.loads(cleaned_response)
        if isinstance(file_list, list) and all(isinstance(item, str) for item in file_list):
            logger.info(f"Successfully parsed file plan: {file_list}")
            return file_list
        else:
            logger.error(f"Parsed JSON is not a list of strings: {file_list}")
            return []
    except json.JSONDecodeError as e_json:
        logger.error(f"JSONDecodeError parsing file plan: {e_json}. Response was: {response[:200]}...") # Log snippet
        # Fallback: try to extract file paths from plain text if JSON parsing fails
        logger.info("Attempting fallback to extract file paths from plain text response.")
        extracted_paths = [line.strip('- ').strip() for line in response.splitlines() if '.' in line and not line.startswith("```")]
        if extracted_paths:
            logger.info(f"Fallback extraction found paths: {extracted_paths}")
            return extracted_paths
        return []
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error parsing file plan: {e}", exc_info=True)
        return []


def _get_project_file_list(user_input: str) -> List[str]:
    """
    Gets the initial list of files for the project based on user input.
    Calls get_file_plan and performs basic validation.
    """
    logger.info('Getting project file list...')
    try:
        file_list = get_file_plan(user_input) # get_file_plan is already defined in main.py
        if not file_list or not isinstance(file_list, list):
            logger.error("No files to generate in the plan or plan is not a list.")
            return []
        logger.info(f'Initial file plan: {file_list}')
        return file_list
    except Exception as e:
        logger.error(f"Error getting project file list: {str(e)}", exc_info=True)
        return []

def _generate_project_files(project_dir_path: Path, file_list: List[str], user_input: str):
    """
    Generates the project files based on the provided list and user input.
    Handles path cleaning, directory creation, special file types, and content generation.
    """
    if not file_list:
        logger.warning("File list is empty. Nothing to generate in _generate_project_files.")
        return

    for file_path_str in file_list:
        # Clean and validate the file path
        clean_path = re.sub(r'^[\'"\s,]+|[\'"\s,]+$', '', file_path_str)
        if not clean_path:
            logger.warning(f"Skipping empty file path from plan: '{file_path_str}'")
            continue
            
        # Handle paths that might already include a "project/" prefix from the plan
        # Assumes project_dir_path.name is the name of the root project folder (e.g., "project")
        if clean_path.startswith(project_dir_path.name + "/"):
            relative_path_str = clean_path[len(project_dir_path.name) + 1:]
        else:
            relative_path_str = clean_path
            
        base_name = os.path.basename(relative_path_str).lower()
        if base_name in ['main.py', 'readme.md', 'requirements.txt']:
            logger.info(f'Skipping forbidden file: {base_name} (Original: {file_path_str})')
            continue
            
        full_path = project_dir_path / relative_path_str
        logger.info(f'Processing file: {full_path} (Original path in plan: {file_path_str})')
        
        try:
            parent_dir = full_path.parent
            if not parent_dir.exists():
                logger.info(f"Creating parent directory: {parent_dir}")
                os.makedirs(parent_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create parent directory for {full_path}: {str(e)}", exc_info=True)
            continue

        # Handle special file types
        if full_path.suffix.lower() == '.db':
            if not full_path.exists():
                try:
                    open(full_path, 'wb').close()
                    logger.info(f'{full_path} (empty database file) created.')
                except Exception as e:
                    logger.error(f"Failed to create empty database file {full_path}: {str(e)}", exc_info=True)
            else:
                logger.info(f'{full_path} (database file) already exists. Skipping creation.')
            continue
            
        if full_path.suffix.lower() == '.json':
            if not full_path.exists():
                try:
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write('[]')
                    logger.info(f'{full_path} (empty JSON array) created.')
                except Exception as e:
                    logger.error(f"Failed to create empty JSON file {full_path}: {str(e)}", exc_info=True)
            else:
                logger.info(f'{full_path} (JSON file) already exists. Skipping creation.')
            continue
            
        if full_path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.gif', '.ico'):
            if not full_path.exists():
                try:
                    with open(full_path, 'wb') as f:
                        f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')
                    logger.info(f'{full_path} (placeholder image) created.')
                except Exception as e:
                    logger.error(f"Failed to create placeholder image {full_path}: {str(e)}", exc_info=True)
            else:
                logger.info(f'{full_path} (image file) already exists. Skipping creation.')
            continue
            
        logger.info(f"Generating content for {relative_path_str} (Full path: {full_path})")
        code = generate_file_content(relative_path_str, user_input)
        
        if not code:
            logger.warning(f"No content generated for {relative_path_str}. Creating an empty file as fallback.")
            try:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(f'# Empty file: {relative_path_str}\n')
                logger.info(f'{full_path} (empty file from fallback) created.')
            except Exception as e:
                logger.error(f'Failed to write empty fallback file {full_path}: {str(e)}', exc_info=True)
            continue
            
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(code)
            logger.info(f'{full_path} generated successfully.')
            if '```' in code:
                logger.warning(f'Content for {full_path} may contain code block markers.')
        except Exception as e:
            logger.error(f'Failed to write content to {full_path}: {str(e)}', exc_info=True)

def supervisor_generate_project(user_input: str):
    """
    Generates project files based on user input by orchestrating helper functions.
    """
    logger.info(f"Supervisor starting project generation for input: {user_input[:100]}...")
    
    project_files = _get_project_file_list(user_input)
    
    if not project_files:
        logger.error("Project generation aborted: No file list obtained.")
        return

    try:
        project_dir_path = Path(os.path.abspath("project"))

        if not project_dir_path.exists():
            logger.info(f"Creating project directory: {project_dir_path}")
            os.makedirs(project_dir_path, exist_ok=True)
        else:
            logger.info(f"Project directory {project_dir_path} already exists.")

        _generate_project_files(project_dir_path, project_files, user_input)
        
        logger.info("Supervisor finished project generation.")
        
    except Exception as e:
        logger.error(f'Critical error during supervisor_generate_project: {str(e)}', exc_info=True)

def verify_file_content(file_path: str) -> OperationResult:
# verify_file_content was moved to file_utils.py

# --- Main Application Logic ---

def get_multi_line_input() -> str:
    """
    Prompts the user for multi-line input until they type 'END' on a new line.

    Returns:
        A string containing all lines of user input, joined by newlines.
    """
    print("\nEnter your prompt for the AI agent. Type 'END' on a new line when finished, or 'exit'/'quit' to close the agent.")
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
    # print("To exit, type 'exit' or 'quit'.") # This is now part of the prompt message
    print("-" * 30) # Separator
    
    while True:
        try:
            user_input = get_multi_line_input().strip() # Strip leading/trailing whitespace
            
            if not user_input: # Skip if input is empty after stripping
                logger.info("Empty input received.")
                continue
                
            if user_input.lower() in ('exit', 'quit'):
                logger.info("Exiting agent as per user request.")
                print("Goodbye! 👋")
                break
                
            logger.info(f"User input received (first 100 chars): {user_input[:100]}")
            if is_multifile_request(user_input):
                logger.info("Multi-file/project request detected. Engaging supervisor workflow...")
                print("🤖 Supervisor: Generating project based on your request...")
                supervisor_generate_project(user_input)
                print("🤖 Supervisor: Project generation process completed.")
                # After project generation, perhaps list top-level files or suggest next steps?
                project_dir = Path("project").resolve()
                if project_dir.exists():
                    top_level_items = [item.name for item in project_dir.iterdir()]
                    print(f"🤖 Supervisor: Project generated in '{project_dir}'. Top-level items: {top_level_items}")
            else:
                logger.info("Single message/task detected. Sending to Gemini client...")
                print("🤖 Agent: Processing your request...")
                response = genai_client.send_message(user_input)
                if response:
                    print(f"\n🤖 Agent Response:\n{'-'*20}\n{response}\n{'-'*20}")
                else:
                    print("🤖 Agent: Received no specific response or an error occurred. Please check logs.")
                    
        except KeyboardInterrupt:
            logger.info("User interrupted the agent (Ctrl+C). Exiting.")
            print("\n🤖 Agent: Exiting due to user interruption. Goodbye!")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main loop: {str(e)}", exc_info=True)
            print(f"🤖 Agent: An unexpected error occurred. Please check the 'agent.log' for details and try again.")
            # Potentially add a short delay or specific recovery logic here if needed.
            time.sleep(1) # Brief pause

if __name__ == '__main__':
    try:
        main()
    except Exception as e: # Catch-all for any error during startup before main loop's try-except
        logger.critical(f"Fatal error during agent startup: {e}", exc_info=True)
        print(f"A critical error occurred during startup: {e}. Please check agent.log. Exiting.")