# env_utils.py

import os
import subprocess
import sys
import platform
# import shutil # Removed as it's not directly used.
import webbrowser
import logging
from typing import Dict, Any, Optional # Added Optional for git_push branch hint

# Configure logging (can be configured by the main application or here if standalone)
# It's generally better if the main application configures logging.
# If this module is run standalone, this logger will not output unless configured.
logger = logging.getLogger(__name__)

def run_script(path: str) -> Dict[str, Any]:
    """
    Runs a non-interactive script using the system's Python interpreter and captures its output.

    Args:
        path: The file path of the Python script to execute.

    Returns:
        A dictionary containing the execution status, exit code, stdout, and stderr.
        Example: {'status': 'completed', 'exit_code': 0, 'stdout': '...', 'stderr': '...'}
                 {'status': 'error', 'error': 'reason'}
    """
    try:
        script_path = os.path.abspath(path)
        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            return {'status': 'error', 'error': 'Script not found', 'path': path}

        # Consider adding a timeout to subprocess.run if scripts can hang
        result = subprocess.run(
            [sys.executable, script_path], 
            capture_output=True, text=True, check=False, timeout=300 # 5 min timeout
        )
        logger.info(f"Script '{script_path}' executed. Exit code: {result.returncode}")
        return {'status': 'completed', 'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    except FileNotFoundError: # Raised if sys.executable is not found
        logger.error(f"Python interpreter '{sys.executable}' not found. Is Python installed and in PATH?", exc_info=True)
        return {'status': 'error', 'error': f"Python interpreter '{sys.executable}' not found."}
    except PermissionError:
        logger.error(f"Permission denied when trying to execute script: {path}", exc_info=True)
        return {'status': 'error', 'error': f"Permission denied for script '{path}'."}
    except subprocess.TimeoutExpired:
        logger.error(f"Script execution timed out after 300 seconds: {path}", exc_info=True)
        return {'status': 'error', 'error': f"Script execution timed out: {path}."}
    except Exception as e:
        logger.error(f"Error running script '{path}': {str(e)}", exc_info=True)
        return {'status': 'error', 'error': str(e)}

def start_interactive(path: str) -> Dict[str, Any]:
    """
    Launches a script in a new console window for interactive input.

    Args:
        path: The file path of the Python script to launch.

    Returns:
        A dictionary with the status and PID of the started process, or an error.
        Example: {'status': 'started', 'pid': 12345}
                 {'status': 'error', 'error': 'reason'}
    """
    try:
        script_path = os.path.abspath(path)
        if not os.path.exists(script_path):
            logger.error(f"Interactive script not found: {script_path}")
            return {'status': 'error', 'error': 'Script not found', 'path': path}

        kwargs = {}
        if platform.system() == 'Windows':
            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
        elif platform.system() in ['Linux', 'Darwin']: # Linux or macOS
            kwargs['start_new_session'] = True # Creates a new process group
        
        proc = subprocess.Popen([sys.executable, script_path], **kwargs)
        logger.info(f"Interactive script '{script_path}' started with PID: {proc.pid}.")
        return {'status': 'started', 'pid': proc.pid}
    except FileNotFoundError: # Raised if sys.executable is not found
        logger.error(f"Python interpreter '{sys.executable}' not found for interactive script. Is Python installed?", exc_info=True)
        return {'status': 'error', 'error': f"Python interpreter '{sys.executable}' not found."}
    except PermissionError:
        logger.error(f"Permission denied when trying to start interactive script: {path}", exc_info=True)
        return {'status': 'error', 'error': f"Permission denied for script '{path}'."}
    except Exception as e:
        logger.error(f"Error starting interactive script '{path}': {str(e)}", exc_info=True)
        return {'status': 'error', 'error': str(e)}

def install_package(package_name: str) -> Dict[str, Any]:
    """
    Installs a Python package using pip.

    Args:
        package_name: The name of the package to install.

    Returns:
        A dictionary with the installation status, pip's stdout and stderr, exit code, and package name.
    """
    if not package_name or not isinstance(package_name, str) or " " in package_name:
        logger.error(f"Invalid package name provided: '{package_name}'")
        return {'status': 'error', 'error': 'Invalid package name provided', 'package': package_name}
    
    logger.info(f"Attempting to install package: '{package_name}' using pip.")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package_name],
            capture_output=True, text=True, check=False, timeout=300 # 5 min timeout for pip install
        )
        if result.returncode == 0:
            logger.info(f"Package '{package_name}' installed successfully. Output:\n{result.stdout}")
            status = 'completed'
        else:
            logger.error(f"Failed to install package '{package_name}'. Exit code: {result.returncode}. Error:\n{result.stderr}")
            status = 'error'
        
        return {
            'status': status,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'exit_code': result.returncode,
            'package': package_name
        }
    except FileNotFoundError:
        logger.error(f"pip command failed. Ensure Python and pip are installed and in PATH. sys.executable: {sys.executable}", exc_info=True)
        return {'status': 'error', 'error': 'pip command failed. Python/pip not found?', 'package': package_name}
    except subprocess.TimeoutExpired:
        logger.error(f"Package installation timed out after 300 seconds for: {package_name}", exc_info=True)
        return {'status': 'error', 'error': f"Package installation timed out: {package_name}.", 'package': package_name}
    except Exception as e:
        logger.error(f"Error installing package '{package_name}': {str(e)}", exc_info=True)
        return {'status': 'error', 'error': str(e), 'package': package_name}

def open_in_browser(path_or_url: str) -> Dict[str, Any]:
    """
    Opens a local HTML file or a URL in the default web browser.

    Args:
        path_or_url: The local file path (e.g., "index.html") or a full URL (e.g., "http://example.com").

    Returns:
        A dictionary with the status and the URL opened, or an error.
    """
    try:
        if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
            url = path_or_url
            logger.info(f"Opening URL in browser: {url}")
        else:
            abs_path = os.path.abspath(path_or_url)
            if not os.path.exists(abs_path):
                logger.error(f"File not found for browser opening: {abs_path}")
                return {'status': 'error', 'error': 'File not found', 'path': path_or_url}
            url = f'file://{abs_path}' # Construct file URI
            logger.info(f"Opening local file in browser: {url}")
            
        if not webbrowser.open(url):
            logger.warning(f"webbrowser.open returned False for {url}. This might indicate no browser was found or the URL was blocked.")
            # Attempt to find a browser path for common browsers for better error reporting (optional)
            # This part can be platform-specific and extensive.
            return {'status': 'error', 'error': 'webbrowser.open failed. No browser found or action blocked.'}
        
        return {'status': 'opened', 'url': url}
    except Exception as e:
        logger.error(f"Error opening '{path_or_url}' in browser: {str(e)}", exc_info=True)
        return {'status': 'error', 'error': str(e)}

def lint_code(path: str) -> Dict[str, Any]:
    """
    Runs flake8 linter on the specified file or directory.

    Args:
        path: The path to the file or directory to lint.

    Returns:
        A dictionary with linting status, flake8's stdout and stderr, and exit code.
    """
    target_path = os.path.abspath(path)
    if not os.path.exists(target_path):
        logger.error(f"Path for linting not found: {target_path}")
        return {'status': 'error', 'error': 'Path not found for linting.', 'path': path}
    
    logger.info(f"Running flake8 linter on: {target_path}")
    try:
        result = subprocess.run(['flake8', target_path], capture_output=True, text=True, check=False)
        
        if result.stderr and result.returncode !=0 and not result.stdout: # Flake8 specific error
             logger.error(f"Flake8 execution error for '{target_path}'. Stderr: {result.stderr}")
             status = 'error_flake8'
        elif result.returncode != 0: # Linting issues found
             status = 'completed_with_issues'
             logger.info(f"Linting completed for '{target_path}'. Issues found.")
        else: # No linting issues
             status = 'completed_no_issues'
             logger.info(f"Linting completed for '{target_path}'. No issues found.")

        return {'status': status, 'stdout': result.stdout, 'stderr': result.stderr, 'exit_code': result.returncode}
    except FileNotFoundError:
        logger.error("flake8 command not found. Please ensure flake8 is installed and in PATH.", exc_info=True)
        return {'status': 'error', 'error': 'flake8 command not found.'}
    except Exception as e:
        logger.error(f"Error linting code at '{target_path}': {str(e)}", exc_info=True)
        return {'status': 'error', 'error': str(e)}

def format_code(path: str) -> Dict[str, Any]:
    """
    Formats code using the 'black' formatter on the specified file or directory.

    Args:
        path: The path to the file or directory to format.

    Returns:
        A dictionary with formatting status, black's stdout and stderr, and exit code.
    """
    target_path = os.path.abspath(path)
    if not os.path.exists(target_path):
        logger.error(f"Path for formatting not found: {target_path}")
        return {'status': 'error', 'error': 'Path not found for formatting.', 'path': path}

    logger.info(f"Running black formatter on: {target_path}")
    try:
        result = subprocess.run(['black', target_path], capture_output=True, text=True, check=False)
        
        if result.returncode == 0: # Black exits 0 if no changes needed or if files were reformatted.
            logger.info(f"Formatting with black completed for '{target_path}'. Output:\n{result.stdout}\nStderr:\n{result.stderr}")
            status = 'formatted' 
        else: # Black exits with non-zero for errors (e.g., internal error, invalid syntax in file)
            logger.error(f"Black formatter execution error for '{target_path}'. Exit Code: {result.returncode}. Stderr:\n{result.stderr}\nStdout:\n{result.stdout}")
            status = 'error_black'
            
        return {'status': status, 'stdout': result.stdout, 'stderr': result.stderr, 'exit_code': result.returncode}
    except FileNotFoundError:
        logger.error("black command not found. Please ensure black is installed and in PATH.", exc_info=True)
        return {'status': 'error', 'error': 'black command not found.'}
    except Exception as e:
        logger.error(f"Error formatting code at '{target_path}': {str(e)}", exc_info=True)
        return {'status': 'error', 'error': str(e)}

def run_tests(path: str = ".") -> Dict[str, Any]:
    """
    Runs pytest on the specified file or directory. Defaults to current directory.

    Args:
        path: The path to run tests on (file or directory). Defaults to ".".

    Returns:
        A dictionary with test status, pytest's stdout and stderr, and exit code.
    """
    target_path = os.path.abspath(path)
    # No os.path.exists check here, as pytest will handle non-existent paths with an error.
    logger.info(f"Running pytest on: {target_path}")
    try:
        result = subprocess.run(['pytest', target_path], capture_output=True, text=True, check=False)
        
        if result.returncode == 0: # All tests passed
            status = 'tests_passed'
            logger.info(f"Pytest run completed for '{target_path}'. All tests passed.")
        elif result.returncode == 1: # Tests failed
            status = 'tests_failed'
            logger.warning(f"Pytest run completed for '{target_path}'. Some tests failed.")
        elif result.returncode == 5: # No tests collected
             status = 'no_tests_found'
             logger.info(f"Pytest: No tests found in '{target_path}'.")
        else: # Other pytest errors (e.g., 2 for interruption, 3 for internal error, 4 for usage error)
            status = 'test_run_error'
            logger.error(f"Pytest run for '{target_path}' encountered an error. Exit code: {result.returncode}. Stderr:\n{result.stderr}")

        return {'status': status, 'stdout': result.stdout, 'stderr': result.stderr, 'exit_code': result.returncode}
    except FileNotFoundError:
        logger.error("pytest command not found. Please ensure pytest is installed and in PATH.", exc_info=True)
        return {'status': 'error', 'error': 'pytest command not found.'}
    except Exception as e:
        logger.error(f"Error running tests at '{target_path}': {str(e)}", exc_info=True)
        return {'status': 'error', 'error': str(e)}

def git_commit(message: str, path_spec: str = '.') -> Dict[str, Any]:
    """
    Stages changes in the specified pathspec (defaulting to current directory) and commits them.

    Args:
        message: The commit message.
        path_spec: The pathspec to stage (e.g., '.', 'specific_file.py'). Defaults to '.'.

    Returns:
        A dictionary with commit status, git's stdout and stderr, and the commit message.
    """
    if not message or not isinstance(message, str):
        logger.error("Commit message cannot be empty.")
        return {'status': 'error', 'error': 'Commit message required.'}
    if not path_spec or not isinstance(path_spec, str): # Basic validation for path_spec
        logger.error(f"Invalid path_spec for git add: '{path_spec}'")
        return {'status': 'error', 'error': 'Invalid path_spec for git add.'}

    logger.info(f"Attempting to stage '{path_spec}' and commit with message: '{message}'")
    try:
        # Stage changes. Ensure CWD is the repository root or use `git -C <repo_root_path> ...`
        add_result = subprocess.run(['git', 'add', path_spec], check=False, capture_output=True, text=True)
        if add_result.returncode != 0:
            logger.error(f"Error during 'git add {path_spec}': {add_result.stderr}")
            return {'status': 'error_staging', 'error': f"git add {path_spec} failed: {add_result.stderr}", 
                    'stdout': add_result.stdout, 'stderr': add_result.stderr, 'exit_code': add_result.returncode}
        
        # Commit
        commit_result = subprocess.run(['git', 'commit', '-m', message], capture_output=True, text=True, check=False)
        
        if commit_result.returncode == 0: # Successful commit
            logger.info(f"Git commit successful: '{message}'. Output:\n{commit_result.stdout}")
            return {'status': 'committed', 'stdout': commit_result.stdout, 'stderr': commit_result.stderr, 'message': message}
        # Common case: no changes to commit (git commit exits with 1)
        elif "nothing to commit" in commit_result.stdout.lower() or \
             "no changes added to commit" in commit_result.stdout.lower() or \
             (commit_result.returncode == 1 and not commit_result.stderr): # Some git versions exit 1 with no stderr for "nothing to commit"
            logger.info(f"Git commit: No changes to commit for message '{message}'.")
            return {'status': 'no_changes_to_commit', 'stdout': commit_result.stdout, 'stderr': commit_result.stderr}
        else: # Other commit errors
            logger.error(f"Git commit failed. Message: '{message}'. Exit code: {commit_result.returncode}. Stderr:\n{commit_result.stderr}\nStdout:\n{commit_result.stdout}")
            return {'status': 'error_committing', 'error': f"Git commit failed: {commit_result.stderr or commit_result.stdout}", 
                    'stdout': commit_result.stdout, 'stderr': commit_result.stderr, 'exit_code': commit_result.returncode}
            
    except FileNotFoundError:
        logger.error("git command not found. Please ensure Git is installed and in PATH.", exc_info=True)
        return {'status': 'error', 'error': 'git command not found.'}
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error during git commit: {str(e)}", exc_info=True)
        return {'status': 'error', 'error': str(e)}

def git_push(remote: str = 'origin', branch: Optional[str] = None) -> Dict[str, Any]:
    """
    Pushes commits to the specified remote and branch.
    If 'branch' is None, it attempts to push the current branch.

    Args:
        remote: The name of the remote repository (e.g., 'origin').
        branch: The name of the branch to push. If None, pushes the current branch.

    Returns:
        A dictionary with push status, git's stdout and stderr.
    """
    logger.info(f"Attempting to push to remote '{remote}'" + (f" branch '{branch}'" if branch else " (current branch)"))
    try:
        cmd = ['git', 'push', remote]
        
        target_branch = branch
        if not target_branch:
            # Determine current branch if not specified
            current_branch_proc = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True, check=True)
            target_branch = current_branch_proc.stdout.strip()
            if not target_branch:
                logger.error("Could not determine current git branch to push.")
                return {'status': 'error', 'error': 'Could not determine current git branch.'}
            logger.info(f"Current branch is '{target_branch}'. Pushing to {remote}/{target_branch}.")
        
        cmd.append(target_branch)
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            logger.info(f"Git push to {remote}/{target_branch} successful. Output:\n{result.stdout}")
            return {'status': 'pushed', 'stdout': result.stdout, 'stderr': result.stderr}
        elif "everything up-to-date" in result.stdout.lower() or \
             "everything up-to-date" in result.stderr.lower():
             logger.info(f"Git push: Everything up-to-date with {remote}/{target_branch}.")
             return {'status': 'up_to_date', 'stdout': result.stdout, 'stderr': result.stderr}
        else:
            logger.error(f"Git push to {remote}/{target_branch} failed. Exit code: {result.returncode}. Stderr:\n{result.stderr}\nStdout:\n{result.stdout}")
            return {'status': 'error_pushing', 'error': f"Git push failed: {result.stderr or result.stdout}", 
                    'stdout': result.stdout, 'stderr': result.stderr, 'exit_code': result.returncode}

    except FileNotFoundError:
        logger.error("git command not found. Please ensure Git is installed and in PATH.", exc_info=True)
        return {'status': 'error', 'error': 'git command not found.'}
    except subprocess.CalledProcessError as e: # Handles errors from `git branch --show-current`
        logger.error(f"Error determining current branch for git push: {e.stderr}", exc_info=True)
        return {'status': 'error', 'error': f"Failed to get current branch: {e.stderr}", 
                'stdout': e.stdout, 'stderr': e.stderr, 'exit_code': e.returncode}
    except Exception as e:
        logger.error(f"Unexpected error during git push: {str(e)}", exc_info=True)
        return {'status': 'error', 'error': str(e)}

# Example usage (if module is run directly, though typically not the case for utils)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO, 
#                         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                         handlers=[logging.StreamHandler()])
#     logger.info("Env utils module ready for direct testing (if applicable).")
#
#     # Example: Test install_package (be cautious with actual installs)
#     # print(install_package("non_existent_package_for_testing_error")) 
#     # print(install_package("requests")) # This would actually install requests
#
#     # Example: Test lint_code (requires a file/directory)
#     # Create a dummy file for testing
#     # with open("test_lint_file.py", "w") as f:
#     #     f.write("import os\n\ndef my_func():\n  print('hello')\n")
#     # print(lint_code("test_lint_file.py"))
#     # os.remove("test_lint_file.py")
#
#     # Note: Git commands would require a git repository context.
#     # print(git_commit("Test commit from env_utils")) # This would fail if not in a repo
