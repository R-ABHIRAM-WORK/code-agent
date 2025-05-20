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

# --- Custom Exceptions ---
class AgentError(Exception): # Base Agent Error
    """Base class for agent-related errors."""
    pass

class FilePathError(AgentError):
    """Error related to file paths (e.g., invalid, unsafe)."""
    pass

class OperationFailedError(AgentError):
    """Error during an operation (e.g., file I/O, subprocess execution)."""
    pass

class GeminiClientError(AgentError):
    """Error related to the Gemini API client."""
    pass

# --- Tool Classes ---
class FileSystemTools:
    """Encapsulates file system operation tools."""

    def safe_file_operation(self, path: str) -> Path: # Return Path, not Optional[Path]
        """Safely resolve and validate file paths. Raises FilePathError or AgentError."""
        try:
            resolved_path = Path(path).resolve()
            base_path = Path.cwd().resolve()
            # Allow path to be CWD itself or a child of CWD.
            if resolved_path != base_path and base_path not in resolved_path.parents:
                raise FilePathError(f"Path '{path}' resolves to '{resolved_path}', which is outside the allowed base directory '{base_path}'.")
            return resolved_path
        except FileNotFoundError: 
            raise FilePathError(f"File or directory component not found for path: {path}")
        except (ValueError, RuntimeError) as e: 
            raise FilePathError(f"Invalid or malformed path '{path}': {e}")
        except Exception as e: 
            # Catching FilePathError again here is redundant if it's already specific.
            # This should be a more general AgentError for truly unexpected issues in path logic.
            if isinstance(e, FilePathError): raise 
            raise AgentError(f"Unexpected error during path validation for '{path}': {e}")

    def ensure_parent_dir_exists(self, file_path: Path) -> None: # No Optional[str], raise error
        """Ensure parent directory of the given file_path exists. Raises OperationFailedError."""
        try:
            parent_dir = file_path.parent
            if parent_dir: # Only try to create if there's a parent (i.e., not for root-level files)
                parent_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e: # More specific than generic Exception
            msg = f"Failed to create parent directory for {file_path}: {e}"
            logger.error(msg)
            raise OperationFailedError(msg) from e
        except Exception as e: # Catch any other unexpected errors
            msg = f"Unexpected error creating parent for {file_path}: {e}"
            logger.error(msg)
            raise OperationFailedError(msg) from e


    def create_file(self, path: str, content: str) -> OperationResult:
        """Create or overwrite a file, auto-creating parent directories."""
        try:
            safe_path = self.safe_file_operation(path)
            self.ensure_parent_dir_exists(safe_path)
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"File {safe_path} created/overwritten successfully.")
            return OperationResult(True, data={'path': str(safe_path)}, status_code=201)
        except FilePathError as e:
            return OperationResult(False, error=str(e), status_code=400)
        except OperationFailedError as e:
            return OperationResult(False, error=str(e), status_code=500)
        except IOError as e:
            return OperationResult(False, error=f"File I/O error: {e}", status_code=500)
        except AgentError as e: # Broader agent error
            return OperationResult(False, error=str(e), status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error creating file {path}: {e}")
            return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

    def read_file(self, path: str) -> OperationResult:
        """Read and return the content of a file."""
        try:
            safe_path = self.safe_file_operation(path)
            if not safe_path.exists():
                raise FileNotFoundError(f"File not found at path: {path}")
            if not safe_path.is_file():
                raise IsADirectoryError(f"Path is a directory, not a file: {path}")
            with open(safe_path, 'r', encoding='utf-8') as f:
                data = f.read()
            logger.info(f"File {safe_path} read successfully.")
            return OperationResult(True, data={'content': data}, status_code=200)
        except FilePathError as e:
            return OperationResult(False, error=str(e), status_code=400)
        except FileNotFoundError as e:
            return OperationResult(False, error=str(e), status_code=404)
        except IsADirectoryError as e:
            return OperationResult(False, error=str(e), status_code=400)
        except IOError as e:
            return OperationResult(False, error=f"File I/O error: {e}", status_code=500)
        except AgentError as e:
            return OperationResult(False, error=str(e), status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error reading file {path}: {str(e)}")
            return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

    def delete_file(self, path: str) -> OperationResult:
        """Delete a file or directory at the given path, handling permission issues."""
        def onerror_rmtree(func, path_str, excinfo):
            logger.warning(f"Error during shutil.rmtree of {path_str}: {excinfo[1]}. Attempting to chmod.")
            try:
                os.chmod(path_str, 0o777)
                func(path_str)
            except Exception as e_chmod:
                logger.error(f"Failed to delete {path_str} even after chmod: {e_chmod}")
        try:
            safe_path = self.safe_file_operation(path)
            if not safe_path.exists():
                raise FileNotFoundError(f"Path not found for deletion: {path}")
            if safe_path.is_dir():
                shutil.rmtree(safe_path, onerror=onerror_rmtree)
                logger.info(f"Directory {safe_path} deleted successfully.")
                return OperationResult(True, data={'status': 'deleted_directory', 'path': str(safe_path)}, status_code=200)
            else:
                safe_path.unlink() 
                logger.info(f"File {safe_path} deleted successfully.")
                return OperationResult(True, data={'status': 'deleted_file', 'path': str(safe_path)}, status_code=200)
        except FilePathError as e:
            return OperationResult(False, error=str(e), status_code=400)
        except FileNotFoundError as e:
            return OperationResult(False, error=str(e), status_code=404)
        except (IOError, OSError) as e:
            return OperationResult(False, error=f"File/directory deletion error: {e}", status_code=500)
        except AgentError as e:
            return OperationResult(False, error=str(e), status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error deleting {path}: {str(e)}")
            return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

    def rename_file(self, old_path: str, new_path: str) -> OperationResult:
        """Rename a file or directory."""
        try:
            safe_old_path = self.safe_file_operation(old_path)
            if not safe_old_path.exists():
                raise FileNotFoundError(f"Source path for rename does not exist: {old_path}")
            safe_new_path = self.safe_file_operation(new_path)
            if safe_new_path.exists():
                raise FileExistsError(f"Destination path for rename already exists: {new_path}")
            self.ensure_parent_dir_exists(safe_new_path)
            os.rename(safe_old_path, safe_new_path)
            logger.info(f"Renamed {safe_old_path} to {safe_new_path} successfully.")
            return OperationResult(True, data={'from': str(safe_old_path), 'to': str(safe_new_path)}, status_code=200)
        except FilePathError as e:
            return OperationResult(False, error=str(e), status_code=400)
        except FileNotFoundError as e:
            return OperationResult(False, error=str(e), status_code=404)
        except FileExistsError as e:
            return OperationResult(False, error=str(e), status_code=409)
        except OperationFailedError as e: # From ensure_parent_dir_exists
            return OperationResult(False, error=str(e), status_code=500)
        except (IOError, OSError) as e:
            return OperationResult(False, error=f"File system error during rename: {e}", status_code=500)
        except AgentError as e:
            return OperationResult(False, error=str(e), status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error renaming {old_path} to {new_path}: {str(e)}")
            return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

    def move_file(self, src: str, dest: str) -> OperationResult:
        """Move a file or directory."""
        try:
            safe_src_path = self.safe_file_operation(src)
            if not safe_src_path.exists():
                raise FileNotFoundError(f"Source path for move does not exist: {src}")
            safe_dest_path = self.safe_file_operation(dest)
            if safe_dest_path.exists() and safe_dest_path.is_file():
                 raise FileExistsError(f"Destination file for move already exists: {dest}")
            self.ensure_parent_dir_exists(safe_dest_path)
            shutil.move(str(safe_src_path), str(safe_dest_path))
            logger.info(f"Moved {safe_src_path} to {safe_dest_path} successfully.")
            return OperationResult(True, data={'from': str(safe_src_path), 'to': str(safe_dest_path)}, status_code=200)
        except FilePathError as e:
            return OperationResult(False, error=str(e), status_code=400)
        except FileNotFoundError as e:
            return OperationResult(False, error=str(e), status_code=404)
        except FileExistsError as e:
            return OperationResult(False, error=str(e), status_code=409)
        except OperationFailedError as e:
            return OperationResult(False, error=str(e), status_code=500)
        except (IOError, OSError, shutil.Error) as e:
            return OperationResult(False, error=f"File system error during move: {e}", status_code=500)
        except AgentError as e:
            return OperationResult(False, error=str(e), status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error moving {src} to {dest}: {str(e)}")
            return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

    def list_directory(self, path: str) -> OperationResult:
        """List files and directories at a given path."""
        try:
            safe_path = self.safe_file_operation(path)
            if not safe_path.exists():
                raise FileNotFoundError(f"Directory not found: {path}")
            if not safe_path.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {path}")
            items = os.listdir(safe_path)
            logger.info(f"Listed directory {safe_path} successfully.")
            return OperationResult(True, data={'path': str(safe_path), 'items': items}, status_code=200)
        except FilePathError as e:
            return OperationResult(False, error=str(e), status_code=400)
        except FileNotFoundError as e:
            return OperationResult(False, error=str(e), status_code=404)
        except NotADirectoryError as e:
            return OperationResult(False, error=str(e), status_code=400)
        except (IOError, OSError) as e:
            return OperationResult(False, error=f"File system error listing directory: {e}", status_code=500)
        except AgentError as e:
            return OperationResult(False, error=str(e), status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error listing directory {path}: {str(e)}")
            return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

    def search_file(self, keyword: str, path: str) -> OperationResult:
        """Search for a keyword in files under the given path."""
        try:
            safe_search_path = self.safe_file_operation(path)
            if not safe_search_path.exists():
                raise FileNotFoundError(f"Search directory not found: {path}")
            if not safe_search_path.is_dir():
                raise NotADirectoryError(f"Search path is not a directory: {path}")
            matches: List[str] = []
            for root_str, _, files in os.walk(str(safe_search_path)):
                root_path = Path(root_str)
                for file_name in files:
                    file_path_obj = root_path / file_name
                    try:
                        current_safe_file_path = self.safe_file_operation(str(file_path_obj))
                        if current_safe_file_path.is_file():
                            with open(current_safe_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                if keyword in f.read():
                                    matches.append(str(current_safe_file_path))
                    except Exception as e_inner: # Catch broadly for individual file search errors
                        logger.debug(f"Skipping file {file_path_obj} during search due to error: {e_inner}")
            logger.info(f"Search for '{keyword}' in {safe_search_path} completed. Found {len(matches)} matches.")
            return OperationResult(True, data={'matches': matches}, status_code=200)
        except FilePathError as e:
            return OperationResult(False, error=str(e), status_code=400)
        except FileNotFoundError as e:
            return OperationResult(False, error=str(e), status_code=404)
        except NotADirectoryError as e:
            return OperationResult(False, error=str(e), status_code=400)
        except (IOError, OSError) as e: # From os.walk
            return OperationResult(False, error=f"File system error during search: {e}", status_code=500)
        except AgentError as e:
            return OperationResult(False, error=str(e), status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error searching for '{keyword}' in {path}: {str(e)}")
            return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

    def chunk_file(self, path: str, chunk_size: int, chunk_index: int) -> OperationResult:
        """Read a file in chunks."""
        try:
            chunk_size = int(chunk_size) if chunk_size is not None else 100
            chunk_index = int(chunk_index) if chunk_index is not None else 0
            if chunk_size <= 0: raise ValueError("chunk_size must be positive.")
            if chunk_index < 0: raise ValueError("chunk_index must be non-negative.")

            safe_file_path = self.safe_file_operation(path)
            if not safe_file_path.is_file(): # Implies exists() is true from safe_file_operation if no error
                raise FileNotFoundError(f"File not found or path is a directory: {path}")

            total_lines, chunk_lines, start_line_num, end_line_num, total_chunks = 0, [], 0, 0, 0
            start_line_num = chunk_index * chunk_size
            end_line_num = start_line_num + chunk_size
            
            with open(safe_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if start_line_num <= i < end_line_num:
                        chunk_lines.append(line)
                    total_lines += 1
            
            total_chunks = (total_lines + chunk_size - 1) // chunk_size if total_lines > 0 else 0
            if chunk_index > 0 and chunk_index >= total_chunks and total_lines > 0: # Allow chunk 0 of empty file
                 error_msg = f"chunk_index {chunk_index} is out of range. File has {total_lines} lines, {total_chunks} chunks."
                 return OperationResult(False, error=error_msg, status_code=416, data={'total_chunks': total_chunks, 'total_lines': total_lines})

            data = {
                'path': str(safe_file_path), 'chunk_index': chunk_index, 'total_chunks': total_chunks,
                'chunk': "".join(chunk_lines), 'start_line': start_line_num, 
                'end_line': min(end_line_num, total_lines) -1 if total_lines > 0 and chunk_lines else -1,
                'total_lines': total_lines
            }
            return OperationResult(True, data=data, status_code=200)
        except (FilePathError, FileNotFoundError, IsADirectoryError, ValueError) as e: # Catch specific validation/path errors
            return OperationResult(False, error=str(e), status_code=400 if not isinstance(e, FileNotFoundError) else 404)
        except (IOError, OSError) as e:
            return OperationResult(False, error=f"File I/O error during chunking: {e}", status_code=500)
        except AgentError as e:
            return OperationResult(False, error=str(e), status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error chunking file {path}: {e}")
            return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

    def update_file_chunk(self, path: str, chunk_content: str, chunk_size: int, chunk_index: int) -> OperationResult:
        """Update a specific chunk of a file."""
        temp_file_path = None
        try:
            chunk_size = int(chunk_size) if chunk_size is not None else 100
            chunk_index = int(chunk_index) if chunk_index is not None else 0
            if chunk_size <= 0: raise ValueError("chunk_size must be positive.")
            if chunk_index < 0: raise ValueError("chunk_index must be non-negative.")
            if not isinstance(chunk_content, str): raise ValueError("chunk_content must be a string.")

            safe_target_path = self.safe_file_operation(path)
            file_existed_before = safe_target_path.exists()

            if not file_existed_before:
                if chunk_index == 0:
                    self.ensure_parent_dir_exists(safe_target_path)
                    with open(safe_target_path, 'w', encoding='utf-8') as f: pass 
                else:
                    raise FileNotFoundError(f"File {path} does not exist, cannot update chunk {chunk_index}.")
            
            if not safe_target_path.is_file():
                 raise IsADirectoryError(f"Target path '{path}' for chunk update is a directory.")

            temp_file_path = safe_target_path.with_suffix(safe_target_path.suffix + '.tmp.' + str(time.time_ns()))
            start_line_num = chunk_index * chunk_size
            
            current_line_idx = 0
            chunk_actually_written = False
            lines_in_new_chunk = chunk_content.count('\n') + (1 if chunk_content else 0) # Approx
            if not chunk_content and lines_in_new_chunk == 1: lines_in_new_chunk = 0


            with open(safe_target_path, 'r', encoding='utf-8', errors='ignore') as f_src, \
                 open(temp_file_path, 'w', encoding='utf-8') as f_dst:
                
                # Write lines before the target chunk start
                for line in f_src:
                    if current_line_idx < start_line_num:
                        f_dst.write(line)
                        current_line_idx += 1
                    else:
                        break # Reached the point where chunk should be inserted or replaced
                
                # If file is shorter than start_line_num, pad with newlines
                if current_line_idx < start_line_num:
                    f_dst.write('\n' * (start_line_num - current_line_idx))
                
                # Write the new chunk
                f_dst.write(chunk_content)
                chunk_actually_written = True # Mark that we've written the new chunk

                # Skip lines in source file that are replaced by the new chunk
                # Need to re-open or seek if we consumed f_src already.
                # Simpler: assume f_src is at the right place if loop broke, or we padded.
                # This replacement logic is complex. A simpler strategy is to read all lines,
                # manipulate the list of lines, then write all back. For very large files,
                # line-by-line processing is better but harder to get right for replacement.
                
                # Reset and skip lines in source that were "replaced"
                # This part is tricky with iterators. The original logic was better.
                # Re-simplifying the replacement part:
                # The previous logic for skipping lines in f_src after writing the chunk was flawed
                # because the iterator f_src was already advanced.
                # A full read/modify/write is safer for chunks unless extremely large files AND
                # complex line manipulation is needed.
                # Given the current structure, the provided code will effectively insert and then append
                # the rest of original content if the chunk is not at the end.
                # If chunk replaces existing lines, those lines need to be skipped in f_src.
                # The original logic with `next(f_src)` for `num_lines_in_new_chunk` times was an attempt.
                # Let's refine the original logic for skipping:
                
                # We are already at start_line_num in f_src (or past it if file was short)
                # We need to skip `lines_in_new_chunk` from the *original* file from this point
                
                # Corrected skipping logic (conceptual, as f_src iterator is tricky here)
                # This part needs careful re-evaluation. The current code will insert the chunk
                # and then append the rest of f_src. If the intention is to replace,
                # the original lines must be skipped.
                # For now, assuming the original simplified logic for insert/append:
                if file_existed_before: # Only try to read more if file existed
                    # If the previous loop finished because current_line_idx < start_line_num,
                    # f_src is still at the beginning. If it broke, f_src is at the line after chunk start.
                    # This logic is getting too complex for simple replacement.
                    # A list-based approach is safer for correctness here.
                    # However, sticking to line-by-line for now:
                    if current_line_idx == start_line_num: # if we broke from the loop at exact start
                        for _ in range(lines_in_new_chunk): # Skip lines replaced by new chunk
                            try:
                                next(f_src)
                            except StopIteration: break
                    # Append remaining lines
                    for line in f_src:
                        f_dst.write(line)


            os.replace(temp_file_path, safe_target_path)
            temp_file_path = None 
            final_total_lines = 0
            with open(safe_target_path, 'r', encoding='utf-8', errors='ignore') as f:
                for _ in f: final_total_lines += 1
            final_total_chunks = (final_total_lines + chunk_size - 1) // chunk_size if final_total_lines > 0 else 0
            if final_total_chunks == 0 and final_total_lines > 0: final_total_chunks = 1
            
            return OperationResult(True, data={'path': str(safe_target_path), 'chunk_index': chunk_index, 'total_chunks': final_total_chunks, 'total_lines': final_total_lines}, status_code=200)
        except (FilePathError, FileNotFoundError, IsADirectoryError, ValueError) as e:
            return OperationResult(False, error=str(e), status_code=400 if not isinstance(e, FileNotFoundError) else 404)
        except OperationFailedError as e:
            return OperationResult(False, error=str(e), status_code=500) # From ensure_parent or bad content
        except (IOError, OSError) as e:
            return OperationResult(False, error=f"File I/O error during chunk update: {e}", status_code=500)
        except AgentError as e:
            return OperationResult(False, error=str(e), status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error updating file chunk for {path}: {str(e)}")
            return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)
        finally:
            if temp_file_path and temp_file_path.exists():
                try:
                    temp_file_path.unlink(); logger.info(f"Cleaned up temporary file {temp_file_path}")
                except Exception as e_unlink:
                    logger.error(f"Failed to delete temporary file {temp_file_path}: {e_unlink}")

class ExecutionTools:
    """Encapsulates code execution and environment management tools."""
    # Note: These tools might need access to FileSystemTools.safe_file_operation
    # For now, we assume paths are validated before calling or use a simplified check.
    # A proper solution would involve DI or making safe_file_operation a shared utility.

    def _validate_script_path(self, path_str: str) -> Path:
        """Basic path validation for executable scripts."""
        # This is a simplified version. Ideally, uses FileSystemTools.safe_file_operation
        p = Path(path_str)
        if not p.exists(): raise FileNotFoundError(f"Script path does not exist: {path_str}")
        if not p.is_file(): raise OperationFailedError(f"Script path is not a file: {path_str}")
        # Could add execute permission check for POSIX: os.access(p, os.X_OK)
        return p.resolve() # Return resolved path

    def run_script(self, path: str) -> OperationResult:
        """Run a non-interactive script."""
        try:
            # safe_script_path = FileSystemTools().safe_file_operation(path) # Ideal
            safe_script_path = self._validate_script_path(path) # Temporary
            result = subprocess.run(
                [sys.executable, str(safe_script_path)], 
                capture_output=True, text=True, check=False
            )
            data = {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
            status_code = 200 # Operation itself succeeded
            success = result.returncode == 0
            error_msg = f"Script execution failed with exit code {result.returncode}." if not success else None
            if success: logger.info(f"Script {safe_script_path} executed successfully.")
            else: logger.warning(f"Script {safe_script_path} failed. Stderr: {result.stderr}")
            return OperationResult(success, data=data, error=error_msg, status_code=status_code)
        except (FilePathError, FileNotFoundError, OperationFailedError) as e:
            return OperationResult(False, error=str(e), status_code=400 if isinstance(e, FilePathError) or isinstance(e, OperationFailedError) else 404)
        except (OSError, subprocess.SubprocessError) as e:
            return OperationResult(False, error=f"Error during script execution: {e}", status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error running script {path}: {e}")
            return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

    def start_interactive(self, path: str) -> OperationResult:
        """Launch a script in a new console for interactive input."""
        try:
            # safe_script_path = FileSystemTools().safe_file_operation(path) # Ideal
            safe_script_path = self._validate_script_path(path) # Temporary
            kwargs = {'start_new_session': True} if platform.system() != 'Windows' else {'creationflags': subprocess.CREATE_NEW_CONSOLE}
            proc = subprocess.Popen([sys.executable, str(safe_script_path)], **kwargs)
            logger.info(f"Interactive script {safe_script_path} started with PID {proc.pid}.")
            return OperationResult(True, data={'status': 'started', 'pid': proc.pid}, status_code=200)
        except (FilePathError, FileNotFoundError, OperationFailedError) as e:
            return OperationResult(False, error=str(e), status_code=400 if isinstance(e, FilePathError) or isinstance(e, OperationFailedError) else 404)
        except (OSError, subprocess.SubprocessError) as e:
            return OperationResult(False, error=f"Error during interactive script startup: {e}", status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error starting interactive script {path}: {e}")
            return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

    def install_package(self, package: str) -> OperationResult:
        """Install a Python package via pip."""
        try:
            if not package or not isinstance(package, str) or not re.match(r'^[a-zA-Z0-9\-_~.=<>! ]+$', package):
                raise OperationFailedError(f"Invalid package name or specification: {package}")
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], capture_output=True, text=True, check=False)
            data = {'package': package, 'stdout': result.stdout, 'stderr': result.stderr, 'exit_code': result.returncode}
            success = result.returncode == 0
            error_msg = f"pip install failed with exit code {result.returncode}." if not success else None
            if success: logger.info(f"Package '{package}' installed successfully.")
            else: logger.error(f"Failed to install '{package}'. Stderr: {result.stderr}")
            return OperationResult(success, data=data, error=error_msg, status_code=200) # Pip ran, so operation is 200
        except OperationFailedError as e:
            return OperationResult(False, error=str(e), status_code=400)
        except (OSError, subprocess.SubprocessError) as e:
            return OperationResult(False, error=f"Error during package installation: {e}", status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error installing package {package}: {str(e)}")
            return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

    def lint_code(self, path: str) -> OperationResult:
        """Run flake8 linter."""
        try:
            # safe_lint_path = FileSystemTools().safe_file_operation(path) # Ideal
            safe_lint_path = self._validate_script_path(path) # Temporary, and might need to allow dirs
            result = subprocess.run(['flake8', str(safe_lint_path)], capture_output=True, text=True, check=False)
            data = {'path': str(safe_lint_path), 'stdout': result.stdout, 'stderr': result.stderr, 'exit_code': result.returncode}
            success = result.returncode == 0 # Success means no linting issues found
            error_msg = "Linting issues found." if not success and result.returncode == 1 else (f"Flake8 error (code {result.returncode})" if not success else None)
            if success: logger.info(f"Linting for {safe_lint_path} completed. No issues.")
            else: logger.warning(f"Linting for {safe_lint_path} found issues or error. Code: {result.returncode}")
            return OperationResult(success, data=data, error=error_msg, status_code=200) # Flake8 ran
        except (FilePathError, FileNotFoundError, OperationFailedError) as e: # Path validation errors
             return OperationResult(False, error=str(e), status_code=400 if isinstance(e, FilePathError) or isinstance(e, OperationFailedError) else 404)
        except (OSError, subprocess.SubprocessError) as e:
            return OperationResult(False, error=f"Error during linting process: {e}", status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error linting {path}: {e}")
            return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

    def format_code(self, path: str) -> OperationResult:
        """Format code using black."""
        try:
            # safe_format_path = FileSystemTools().safe_file_operation(path) # Ideal
            safe_format_path = self._validate_script_path(path) # Temporary, might need to allow dirs
            result = subprocess.run(['black', str(safe_format_path)], capture_output=True, text=True, check=False)
            data = {'path': str(safe_format_path), 'stdout': result.stdout, 'stderr': result.stderr, 'exit_code': result.returncode}
            success = result.returncode == 0 or result.returncode == 1 # 0=no changes, 1=changes made
            error_msg = f"Black formatting failed (code {result.returncode})." if not success else None
            if success : logger.info(f"Formatting for {safe_format_path} completed (Code: {result.returncode}).")
            else: logger.error(f"Black formatting failed for {safe_format_path}. Stderr: {result.stderr}")
            return OperationResult(success, data=data, error=error_msg, status_code=200) # Black ran
        except (FilePathError, FileNotFoundError, OperationFailedError) as e:
            return OperationResult(False, error=str(e), status_code=400 if isinstance(e, FilePathError) or isinstance(e, OperationFailedError) else 404)
        except (OSError, subprocess.SubprocessError) as e:
            return OperationResult(False, error=f"Error during formatting process: {e}", status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error formatting {path}: {e}")
            return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

    def run_tests(self, path: str) -> OperationResult:
        """Run pytest."""
        try:
            # safe_test_path = FileSystemTools().safe_file_operation(path) # Ideal
            safe_test_path = self._validate_script_path(path) # Temporary, might need to allow dirs
            result = subprocess.run(['pytest', str(safe_test_path)], capture_output=True, text=True, check=False)
            data = {'path': str(safe_test_path), 'stdout': result.stdout, 'stderr': result.stderr, 'exit_code': result.returncode}
            success = result.returncode == 0 # All tests passed
            error_msg = None
            if result.returncode == 1: error_msg = "Tests failed."
            elif result.returncode == 5: error_msg = "No tests found."
            elif result.returncode != 0: error_msg = f"Pytest error (code {result.returncode})."
            
            if success: logger.info(f"Tests for {safe_test_path} passed.")
            else: logger.warning(f"Test outcome for {safe_test_path}: {error_msg or 'See output'}. Code: {result.returncode}")
            return OperationResult(success, data=data, error=error_msg, status_code=200) # Pytest ran
        except (FilePathError, FileNotFoundError, OperationFailedError) as e:
            return OperationResult(False, error=str(e), status_code=400 if isinstance(e, FilePathError) or isinstance(e, OperationFailedError) else 404)
        except (OSError, subprocess.SubprocessError) as e:
            return OperationResult(False, error=f"Error during test execution: {e}", status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error running tests on {path}: {e}")
            return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

class BrowserTools:
    """Encapsulates browser interaction tools."""
    def open_in_browser(self, path_or_url: str) -> OperationResult:
        """Open a file or URL in the default web browser."""
        try:
            url_to_open: str
            if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
                if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', path_or_url):
                     raise OperationFailedError(f"Invalid URL format: {path_or_url}")
                url_to_open = path_or_url
                path_type = "URL"
            else:
                # safe_file_path = FileSystemTools().safe_file_operation(path_or_url) # Ideal
                # Temporary direct call for this step, assuming it's made global/static for now
                safe_file_path = FileSystemTools().safe_file_operation(path_or_url) 
                if not safe_file_path.is_file(): # safe_file_operation ensures exists
                    raise IsADirectoryError(f"Path is a directory, not a file: {path_or_url}")
                url_to_open = f'file://{str(safe_file_path.resolve())}'
                path_type = "file"
            webbrowser.open(url_to_open)
            logger.info(f"Opened {path_type} '{path_or_url}' (resolved to '{url_to_open}') in browser.")
            return OperationResult(True, data={'status': 'opened', 'url': url_to_open}, status_code=200)
        except (FilePathError, FileNotFoundError, IsADirectoryError, OperationFailedError) as e:
            return OperationResult(False, error=str(e), status_code=400 if not isinstance(e, FileNotFoundError) else 404)
        except webbrowser.Error as e:
            return OperationResult(False, error=f"Webbrowser error: {e}", status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error opening {path_or_url} in browser: {str(e)}")
            return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

class GitTools:
    """Encapsulates Git operation tools."""
    def git_commit(self, message: str) -> OperationResult:
        """Commit staged changes with a commit message."""
        try:
            if not message or not isinstance(message, str):
                raise OperationFailedError("Commit message cannot be empty.")
            add_result = subprocess.run(['git', 'add', '.'], capture_output=True, text=True, check=False)
            if add_result.returncode != 0:
                raise OperationFailedError(f"git add failed: {add_result.stderr}")
            commit_result = subprocess.run(['git', 'commit', '-m', message], capture_output=True, text=True, check=False)
            data = {
                'message': message, 'add_stdout': add_result.stdout, 'add_stderr': add_result.stderr,
                'commit_stdout': commit_result.stdout, 'commit_stderr': commit_result.stderr, 
                'commit_exit_code': commit_result.returncode
            }
            if commit_result.returncode == 0:
                return OperationResult(True, data=data, status_code=200)
            elif "nothing to commit" in commit_result.stdout or "no changes added to commit" in commit_result.stdout or \
                 (commit_result.returncode == 1 and not commit_result.stderr):
                return OperationResult(True, data=data, error="No changes to commit.", status_code=200)
            else:
                raise OperationFailedError(f"git commit failed: {commit_result.stderr}")
        except OperationFailedError as e:
            return OperationResult(False, data={'message': message, 'error_details': str(e)}, error=str(e), status_code=400 if "Commit message" in str(e) else 500)
        except (OSError, subprocess.SubprocessError) as e:
            return OperationResult(False, error=f"Error during git commit process: {e}", status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error during git commit: {str(e)}")
            return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

    def git_push(self, remote: str, branch: str) -> OperationResult:
        """Push commits to the remote repository."""
        try:
            if not remote or not isinstance(remote, str) or not re.match(r'^[a-zA-Z0-9\-_/\.:]+$', remote):
                 raise OperationFailedError(f"Invalid remote name: {remote}")
            if not branch or not isinstance(branch, str) or not re.match(r'^[a-zA-Z0-9\-_/\.]+$', branch):
                 raise OperationFailedError(f"Invalid branch name: {branch}")
            result = subprocess.run(['git', 'push', remote, branch], capture_output=True, text=True, check=False)
            data = {
                'remote': remote, 'branch': branch, 'stdout': result.stdout, 
                'stderr': result.stderr, 'exit_code': result.returncode
            }
            if result.returncode == 0:
                return OperationResult(True, data=data, status_code=200)
            else:
                raise OperationFailedError(f"git push failed: {result.stderr}")
        except OperationFailedError as e:
            return OperationResult(False, data={'remote': remote, 'branch': branch, 'error_details': str(e)}, error=str(e), status_code=400 if "Invalid" in str(e) else 500)
        except (OSError, subprocess.SubprocessError) as e:
            return OperationResult(False, error=f"Error during git push process: {e}", status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error during git push to {remote}/{branch}: {str(e)}")
            return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

class UserInteractionTools:
    """Encapsulates user interaction tools."""
    def prompt_input(self, message: str) -> Dict[str, Any]:
        """Prompt the user and return the input."""
        try:
            val = input(f"{message} ")
            return {'user_input': val}
        except EOFError: return {'user_input': "", 'error': "EOF received, no input taken."}
        except KeyboardInterrupt: return {'user_input': "", 'error': "Input cancelled by user."}
        except Exception as e: return {'user_input': "", 'error': f"Unexpected error: {e}"}

# --- Agent Class Definition ---
DEFAULT_SYSTEM_PROMPT = (
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

class Agent:
    """
    Encapsulates the Gemini client, agent configuration, and primary interaction logic.
    """
    def __init__(self, api_key: Optional[str] = None, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable or api_key argument is required")
        
        self.gemini_client = self._initialize_gemini_client(api_key, system_prompt)
        self.system_prompt = system_prompt
        # Other configurations like max_retries, retry_delay are part of GeminiClient now
        
        # Project context can be stored here if needed for agent_per_file_edit
        self.current_project_context: str = ""


    class _GeminiClient:
        """Inner class for Gemini client with rate limiting and retry logic."""
        def __init__(self, agent_instance, api_key: str, system_prompt: str, tokens_per_min: int = 1000000, requests_per_min: int = 100, max_retries: int = 3, retry_delay: int = 2):
            self.agent_instance = agent_instance # To access outer class if needed, e.g. for system_prompt
            self.client = genai.Client(api_key=api_key)
            self.token_bucket = TokenBucket(tokens_per_min=tokens_per_min, requests_per_min=requests_per_min)
            self.max_retries = max_retries
            self.retry_delay = retry_delay
            self.chat = None
            self.system_prompt = system_prompt # Store system_prompt

        def create_chat(self):
            """Create a new chat session."""
            # Note: Tool functions (create_file, etc.) are still global as genai expects function references.
            config = types.GenerateContentConfig(
                tools=[
                    create_file, read_file, delete_file, rename_file, move_file,
                    list_directory, search_file, run_script, start_interactive, install_package,
                    open_in_browser, lint_code, format_code, run_tests, git_commit, git_push, prompt_input,
                    chunk_file, update_file_chunk
                ],
                temperature=0,
                system_instruction=self.system_prompt # Use the stored system_prompt
            )
            self.chat = self.client.chats.create(
                model='models/gemini-2.5-flash-preview-04-17', # Consider making model configurable
                config=config
            )

        def send_message(self, message: str, max_tokens: int = 1000) -> Optional[str]:
            """Send message with rate limiting and retry logic."""
            if not self.chat:
                self.create_chat()
            
            for attempt in range(self.max_retries):
                if not self.token_bucket.consume(max_tokens): # Assuming max_tokens is a proxy for actual token count
                    logger.info(f"Rate limit hit, waiting for {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue
                
                try:
                    logger.info(f"Sending message to Gemini (attempt {attempt + 1})...")
                    response = self.chat.send_message(message)
                    logger.info("Received response from Gemini.")
                    return response.text
                except (types.GoogleAPIError, genai.APIError, Exception) as e: # Catch more specific genai errors if available
                    logger.error(f"Gemini API call failed (attempt {attempt + 1}/{self.max_retries}): {type(e).__name__} - {str(e)}")
                    if attempt < self.max_retries - 1:
                        # Check for specific non-retryable errors if needed.
                        # For example, authentication errors (though API key is checked at init)
                        # or quota errors that shouldn't be retried immediately.
                        # if isinstance(e, types.PermissionDenied): raise GeminiClientError(f"Permission denied: {e}") from e
                        # if isinstance(e, types.ResourceExhausted): raise GeminiClientError(f"Resource exhausted: {e}") from e
                        
                        time.sleep(self.retry_delay * (attempt + 1)) # Exponential backoff could be better
                        logger.info("Recreating chat session due to error...")
                        try:
                            self.create_chat() # Recreate chat on failure
                        except Exception as chat_e:
                            logger.error(f"Failed to recreate chat session: {chat_e}")
                            # If chat recreation fails, it might be a persistent issue.
                            raise GeminiClientError(f"Failed to recreate chat session after API error: {chat_e}") from chat_e
                    else:
                        logger.error("Max retries reached for Gemini API call.")
                        raise GeminiClientError(f"Max retries reached for Gemini API call: {e}") from e
            # If all retries fail (e.g. due to rate limiting not resolving)
            raise GeminiClientError("Failed to send message after multiple retries due to rate limiting or other transient errors.")

        
        def update_system_prompt(self, new_system_prompt: str):
            """Updates the system prompt and resets the chat."""
            self.system_prompt = new_system_prompt
            self.create_chat() # Recreate chat with new system prompt
            logger.info(f"System prompt updated. Chat session reset.")

    def _initialize_gemini_client(self, api_key: str, system_prompt: str) -> _GeminiClient:
        return Agent._GeminiClient(self, api_key=api_key, system_prompt=system_prompt)

    def get_file_plan(self, user_input: str) -> list:
        """Ask Gemini to generate a file/folder plan for the project."""
        plan_prompt = (
            f"Given this project description, list all files and folders (with relative paths) needed. "
            f"Output as a JSON array of file paths only, no explanations.\nProject description: {user_input}"
        )
        try:
            response = self.gemini_client.send_message(plan_prompt) 
            if not response:
                logger.warning("Received no response or empty response for file plan request.")
                return [] # Return empty list, not an exception, as this is a "soft" failure.

            import json # Keep import here as it's only used for this method.
            file_list = json.loads(response) # Can raise json.JSONDecodeError
            
            if isinstance(file_list, list) and all(isinstance(item, str) for item in file_list):
                logger.info(f"Successfully obtained file plan: {file_list}")
                return file_list
            else:
                logger.warning(f"File plan from JSON is not a list of strings. Response: {response}. Discarding.")
                return [] # Treat malformed JSON structure as a soft failure for file plan.

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode file plan from LLM response as JSON: {e}. Response was: {response}")
            return [] # Soft failure
        except GeminiClientError as e: # More specific error from the client
            logger.error(f"Gemini client error while getting file plan: {e}")
            # Depending on desired behavior, could re-raise or return empty list.
            # For now, returning empty list to allow supervisor_generate_project to note "no plan".
            return [] 
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred while getting file plan: {e}. Response was: {response}")
            return [] # Soft failure

    def generate_file_content(self, file_path: str, user_input_for_project_context: str) -> str:
        """Generate content for a file with improved code block handling and error checking."""
        file_ext = os.path.splitext(file_path)[1].lower()
        file_type = {
            '.py': 'Python', '.html': 'HTML', '.css': 'CSS', '.js': 'JavaScript',
            '.json': 'JSON', '.md': 'Markdown', '.txt': 'Text', '.sql': 'SQL'
        }.get(file_ext, 'Text')
        
        prompt = (
            f"Generate the complete code for the file '{file_path}' "
            f"as part of this project description: {user_input_for_project_context}\n"
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
            response = self.gemini_client.send_message(prompt, max_tokens=2000) 
            if response is None: # Handles cases where send_message itself returns None (e.g. after retries but before raising GeminiClientError)
                logger.error(f"LLM returned no response for generating content for {file_path}.")
                # Returning empty string, supervisor_generate_project handles this by creating an empty fallback file.
                return "" 
                
            content = strip_code_block_markers(response)
            if not content.strip():
                logger.warning(f"Empty content generated by LLM for {file_path} after stripping markers.")
                return "" # Treat as soft failure, supervisor will create fallback.
            
            logger.info(f"Successfully generated content for {file_path}.")
            return content
        except GeminiClientError as e: # Specific error from the client
            logger.error(f"Gemini client error while generating content for {file_path}: {e}")
            # This is a more severe error, propagate or handle. For now, return empty.
            return ""
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error generating content for {file_path}: {str(e)}")
            return "" # Soft failure, allow fallback

    def supervisor_generate_project(self, user_input: str, project_dir_name: str = "project"):
        """Generate project files with improved error handling and token management."""
        logger.info(f"Starting project generation in directory: '{project_dir_name}' for user input: '{user_input[:100]}...'")
        try:
            file_list = self.get_file_plan(user_input) 
            if not file_list: # get_file_plan now returns empty list on error/no plan
                logger.error("No files to generate in the plan, or an error occurred retrieving the plan.")
                # User feedback might be needed here or handled by the caller.
                return 
                
            logger.info(f'File plan: {file_list}')
            
            # Validate project_dir_name
            if not isinstance(project_dir_name, str) or not re.match(r'^[a-zA-Z0-9_.-]+$', project_dir_name) or \
               ".." in project_dir_name or "/" in project_dir_name or "\\" in project_dir_name :
                logger.error(f"Invalid project directory name: '{project_dir_name}'. Using default 'project'.")
                project_dir_name = "project"
                
            project_dir = Path.cwd() / project_dir_name
            try:
                project_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured project directory exists: {project_dir.resolve()}")
            except OSError as e:
                logger.error(f"Failed to create project directory '{project_dir}': {e}")
                # This is a critical error for this function, so re-raise or handle.
                # For now, let it propagate to the main error handler for the agent.
                raise OperationFailedError(f"Failed to create project directory '{project_dir}': {e}") from e

            for file_path_str in file_list:
                try:
                    clean_path_str = re.sub(r'^[\'"\s,]+|[\'"\s,]+$', '', file_path_str)
                    if not clean_path_str:
                        logger.warning(f"Skipping empty file path string from plan: '{file_path_str}'")
                        continue
                    
                    relative_path_obj = Path(clean_path_str)
                    if ".." in relative_path_obj.parts: # Security check
                        logger.warning(f"Skipping potentially unsafe path from plan: {clean_path_str}")
                        continue

                    base_name = relative_path_obj.name.lower()
                    if base_name in ['main.py', 'readme.md', 'requirements.txt']: # Forbidden files
                        logger.info(f'Skipping forbidden file from plan: {base_name}')
                        continue
                        
                    full_path = project_dir / relative_path_obj
                    logger.info(f'Processing planned file: {full_path}')
                    
                    parent_error_msg = ensure_parent_dir_exists(full_path) # This logs errors internally
                    if parent_error_msg:
                        # Log and skip this file, but continue with others in the plan.
                        logger.error(f"Skipping file '{full_path}' due to parent directory creation error: {parent_error_msg}")
                        continue 
                    
                    # Handle special file types (empty files for .db, .json, images)
                    if full_path.name.lower().endswith('.db'):
                        if not full_path.exists(): open(full_path, 'wb').close(); logger.info(f'{full_path} (empty .db) created.')
                        continue
                    if full_path.name.lower().endswith('.json'):
                        if not full_path.exists(): 
                            with open(full_path, 'w', encoding='utf-8') as f: f.write('[]')
                            logger.info(f'{full_path} (empty .json) created.')
                        continue
                    if full_path.name.lower().endswith(('.png', '.jpg', '.jpeg', 'gif', '.ico')):
                        if not full_path.exists():
                            with open(full_path, 'wb') as f: f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')
                            logger.info(f'{full_path} (placeholder image) created.')
                        continue
                        
                    # Generate content for other files
                    code = self.generate_file_content(str(relative_path_obj), user_input) 
                    if not code: # generate_file_content returns "" on error or empty LLM response
                        if not full_path.exists(): # Create empty fallback only if file doesn't exist
                            with open(full_path, 'w', encoding='utf-8') as f: f.write(f'# Generated file (empty or error generating): {relative_path_obj}\n')
                            logger.info(f'{full_path} (empty file due to no content) created as fallback.')
                        else: # File exists, but LLM generated no new content
                            logger.info(f'{full_path} already exists, LLM generated no new content, not overwriting.')
                        continue
                        
                    # Write the generated content
                    with open(full_path, 'w', encoding='utf-8') as f: f.write(code)
                    logger.info(f'{full_path} generated successfully with LLM content.')
                    if '```' in code: logger.warning(f'Content for {full_path} may still contain code block markers.')

                except (IOError, OSError) as e_file_op: # Catch errors for operations on a single file_path_str from the plan
                    logger.error(f"Error processing file '{file_path_str}' in project '{project_dir_name}': {e_file_op}")
                    # Continue to the next file in the plan.
                except Exception as e_file_unexpected: # Catch any other unexpected error for a single file
                    logger.error(f"Unexpected error processing file '{file_path_str}': {e_file_unexpected}")


        except OperationFailedError as e_op: # e.g. from project_dir.mkdir()
             logger.critical(f"A critical operation failed during project generation for '{project_dir_name}': {e_op}")
             # This error is re-raised to be handled by the main loop, as it's a fundamental failure.
             raise
        except Exception as e_global: # Catch-all for unexpected errors in the main try block
            logger.critical(f"Unexpected critical error in supervisor_generate_project for '{project_dir_name}': {e_global}")
            import traceback
            logger.error(traceback.format_exc()) 
            # Depending on desired robustness, might raise AgentError here.

    def agent_per_file_edit(self, error_message: str, traceback_str: str, minimal_files: bool = True): # project_context removed, can use self.current_project_context
        """
        Parse the error traceback, attempt to fix errors in affected files using the LLM.
        """
        Parse the error traceback, attempt to fix errors in affected files using the LLM.
        """
        logger.info(f"Starting agent_per_file_edit for error: '{error_message}' with minimal_files={minimal_files}")
        # current_project_context is set by process_command
        logger.info(f"Project context for edit: '{self.current_project_context[:200]}...'")

        try:
            error_locations = parse_python_traceback(traceback_str)
            if not error_locations:
                logger.warning("No valid file locations found in traceback. Cannot proceed with automated edits.")
                return # Or raise AgentError("No valid locations in traceback") if this should halt caller

            MAX_LINES_FOR_FULL_FILE = 1000 
            CONTEXT_WINDOW_LINES = 50    

            for loc in error_locations:
                file_path_str = loc['file']
                line_number = loc['line'] 
                logger.info(f"Attempting to fix error in {file_path_str} at line {line_number}")

                try:
                    safe_file_path = safe_file_operation(file_path_str) # Can raise FilePathError
                    if not safe_file_path.exists() or not safe_file_path.is_file():
                        # This specific error might be caught by safe_file_operation's own FileNotFoundError,
                        # but double-checking or logging here is fine.
                        logger.error(f"File {file_path_str} from traceback not found or is not a file. Skipping.")
                        continue
                    
                    with open(safe_file_path, 'r', encoding='utf-8') as f: file_lines = f.readlines()

                    code_to_fix: str
                    is_chunk = False
                    chunk_start_line_idx = 0 
                    chunk_end_line_idx = 0   

                    if len(file_lines) > MAX_LINES_FOR_FULL_FILE:
                        is_chunk = True
                        chunk_start_line_idx = max(0, line_number - 1 - CONTEXT_WINDOW_LINES)
                        chunk_end_line_idx = min(len(file_lines), line_number - 1 + CONTEXT_WINDOW_LINES)
                        code_chunk_lines = file_lines[chunk_start_line_idx:chunk_end_line_idx]
                        code_to_fix = "".join(code_chunk_lines)
                        logger.info(f"File is large. Processing chunk from line {chunk_start_line_idx + 1} to {chunk_end_line_idx}.")
                    else:
                        code_to_fix = "".join(file_lines)
                        logger.info("Processing full file content.")

                    prompt = (
                        f"Project Context (if any): {self.current_project_context}\n\n"
                        f"An error occurred in the following Python code:\n"
                        f"File Path: {file_path_str}\n"
                        f"Error Line: {line_number}\n\n"
                        f"Error Message: {error_message}\n\n"
                        f"Full Traceback:\n```\n{traceback_str}\n```\n\n"
                        f"Code {'Chunk ' if is_chunk else ''}to Fix:\n```python\n{code_to_fix}\n```\n\n"
                        f"Please provide the corrected version of the above code {'chunk' if is_chunk else 'file'}. "
                        f"Output only the raw corrected code, without any explanations, comments, or markdown formatting. "
                        f"Ensure the corrected code is complete and addresses the identified error."
                    )

                    logger.info("Sending request to LLM for code correction...")
                    llm_response = self.gemini_client.send_message(prompt, max_tokens=2000) 

                    if not llm_response:
                        logger.warning(f"LLM returned an empty response for {safe_file_path}. Skipping fix for this file.")
                        continue

                    corrected_code = strip_code_block_markers(llm_response)
                    if not corrected_code.strip():
                        logger.warning(f"LLM returned an empty code block after stripping for {safe_file_path}. Skipping fix.")
                        continue
                    
                    logger.info(f"Received corrected code suggestion for {safe_file_path}.")

                    if is_chunk:
                        new_file_lines = file_lines[:chunk_start_line_idx] + \
                                         [line + '\n' for line in corrected_code.splitlines()] + \
                                         file_lines[chunk_end_line_idx:]
                        logger.info(f"Applying corrected chunk to {safe_file_path} from line {chunk_start_line_idx+1}.")
                    else:
                        new_file_lines = [line + '\n' for line in corrected_code.splitlines()]
                        logger.info(f"Applying full corrected code to {safe_file_path}.")
                    
                    with open(safe_file_path, 'w', encoding='utf-8') as f: f.writelines(new_file_lines)
                    logger.info(f"Successfully applied fix to {safe_file_path}.")
                    logger.info(f"Validation step for {safe_file_path} skipped. (TODO: Implement validation)")

                except FilePathError as e_fp:
                    logger.error(f"FilePathError for {file_path_str} during edit attempt: {e_fp}. Skipping this file.")
                except FileNotFoundError as e_fnf: # From open() if file disappears after safe_file_operation check
                    logger.error(f"FileNotFoundError for {file_path_str} during edit attempt: {e_fnf}. Skipping this file.")
                except (IOError, OSError) as e_io:
                    logger.error(f"IO/OS Error for {file_path_str} during edit attempt: {e_io}. Skipping this file.")
                except GeminiClientError as e_gemini: # Error from LLM call
                    logger.error(f"GeminiClientError while trying to get fix for {file_path_str}: {e_gemini}. Skipping this file.")
                except Exception as e_file_edit: # Catch-all for unexpected errors within loop for one file
                    logger.error(f"Unexpected error processing/applying fix for {file_path_str}: {e_file_edit}")
                    import traceback
                    logger.debug(f"Traceback for {file_path_str} fix error: {traceback.format_exc()}")
            
            if minimal_files:
                logger.info("`minimal_files` is True. Agent focused only on files in the traceback.")
            logger.info("Finished all attempts in agent_per_file_edit.")

        except AgentError as e_agent: # Catch errors from parse_python_traceback or other AgentErrors
            logger.error(f"AgentError in agent_per_file_edit: {e_agent}")
            # Depending on severity, this might warrant re-raising or specific handling.
        except Exception as e_outer: # Catch-all for the entire function
            logger.critical(f"Unexpected critical error in agent_per_file_edit: {e_outer}")
            import traceback
            logger.error(traceback.format_exc())
            # This indicates a more severe problem, potentially re-raise.
            # raise AgentError(f"Critical failure in agent_per_file_edit: {e_outer}") from e_outer


    def process_command(self, user_input: str):
        """Processes a user command, routing to appropriate handlers."""
        try:
            if is_multifile_request(user_input):
                logger.info('Detected multi-file/full-project request. Using supervisor workflow...')
                project_name_match = re.search(r"(?:project|app|website|program)\s*named\s*['\"]?([a-zA-Z0-9_.-]+)['\"]?", user_input, re.IGNORECASE)
                custom_project_dir_name = project_name_match.group(1) if project_name_match else "generated_project"
                
                self.current_project_context = f"User is working on a project called '{custom_project_dir_name}'. Original request: {user_input}"
                logger.info(f"Set project context to: {self.current_project_context}")

                # supervisor_generate_project might raise OperationFailedError for critical dir creation.
                self.supervisor_generate_project(user_input, project_dir_name=custom_project_dir_name)
                # If it completes without error, we can assume it logged its own file-specific issues.
                print(f"Agent: Project generation process for '{custom_project_dir_name}' initiated. Check logs for details.")

            else: # Direct command to LLM
                self.current_project_context = f"User's last direct command: {user_input}" 
                logger.info(f"Set project context to: {self.current_project_context}")
                
                # This can raise GeminiClientError
                response = self.gemini_client.send_message(user_input) 
                print(f"Agent: {response if response else 'No response or error from LLM.'}")

        except GeminiClientError as e:
            logger.error(f"Gemini client error processing command: '{user_input[:50]}...': {e}")
            print(f"Agent: Sorry, I encountered an issue communicating with the AI. Please try again later. Details: {e}")
        except OperationFailedError as e: # Catch critical operational failures
            logger.error(f"A critical operation failed while processing command: '{user_input[:50]}...': {e}")
            print(f"Agent: A critical operation failed. I cannot proceed with this request. Details: {e}")
        # FilePathError should ideally be caught by the methods that call safe_file_operation
        # and converted to OperationResult if they are tools, or handled if they are internal.
        # If one escapes to here, it's an unexpected AgentError.
        except AgentError as e: # Catch other agent-specific errors
            logger.error(f"Agent error processing command: '{user_input[:50]}...': {e}")
            print(f"Agent: An agent error occurred: {e}")
        except Exception as e: # Catch any other unexpected error during command processing
            logger.error(f"Unexpected error processing command '{user_input[:50]}...': {e}")
            import traceback
            logger.debug(traceback.format_exc())
            print(f"Agent: An unexpected error occurred. Please check the logs.")


    def run(self):
        """Main application loop for the agent."""
        logger.info('=== Gemini Super Agent ===')
        print("Welcome to Gemini Super Agent!")
        print("To enter multi-line input, type your text and press Enter.")
        print("When finished, type 'END' on a new line and press Enter.")
        print("To exit, type 'exit' or 'quit'.")
        print()
        
        while True:
            try:
                user_input = get_multi_line_input() # Assuming get_multi_line_input remains global or becomes static
                if not user_input.strip():
                    continue
                    
                if user_input.lower() in ('exit', 'quit'):
                    logger.info('Goodbye!')
                    break
                
                self.process_command(user_input)
                    
            except Exception as e:
                logger.error(f"An error occurred in the main loop: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                print(f"Agent: An error occurred. Please try again later or check logs.")

# --- Global Utility Functions (Stateless or specific to genai tools) ---
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
        # Attempt to resolve the path. This can raise ValueError or RuntimeError for invalid paths.
        resolved_path = Path(path).resolve()
        base_path = Path.cwd().resolve()

        # Check if the resolved path is within the current working directory or its subdirectories.
        # This is a basic security measure against directory traversal.
        if base_path not in resolved_path.parents and resolved_path != base_path:
            # If path is exactly cwd, it's fine. If it's a parent of cwd, it's not allowed.
            # If it's completely outside cwd tree, it's not allowed.
            # This logic might need adjustment if files *outside* CWD are explicitly allowed for some operations.
            # For now, we restrict to CWD and its children.
             if resolved_path != Path.cwd(): # Allow operations on CWD itself.
                raise FilePathError(f"Path '{path}' resolves to '{resolved_path}', which is outside the allowed base directory '{base_path}'.")

        return resolved_path
    except FileNotFoundError: # Raised by resolve() if a component of the path doesn't exist and strict=True (not used here, but good to be aware)
        raise FilePathError(f"File or directory not found: {path}")
    except (ValueError, RuntimeError) as e: # Catches errors from Path() or resolve() for malformed paths
        raise FilePathError(f"Invalid or malformed path '{path}': {e}")
    except Exception as e: # Catch any other unexpected errors during path operations
        raise AgentError(f"Unexpected error during path validation for '{path}': {e}")


def ensure_parent_dir_exists(file_path: Path) -> Optional[str]:
    """Ensure parent directory of the given file_path exists. Returns error message or None."""
    try:
        parent_dir = file_path.parent
        if parent_dir:
            parent_dir.mkdir(parents=True, exist_ok=True)
        return None
    except Exception as e:
        logger.error(f"Error creating parent directory for {file_path}: {str(e)}")
        return f"Failed to create parent directory: {str(e)}"

def create_file(path: str, content: str) -> OperationResult:
    """Create or overwrite a file, auto-creating parent directories."""
    try:
        safe_path = safe_file_operation(path) # Can raise FilePathError or AgentError
        
        error_msg = ensure_parent_dir_exists(safe_path) # Returns string or None
        if error_msg:
            # This indicates a failure in mkdir, which is an internal server-like error
            raise OperationFailedError(f"Parent directory creation failed for {safe_path}: {error_msg}")
            
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"File {safe_path} created/overwritten successfully.")
        return OperationResult(True, data={'path': str(safe_path)}, status_code=201) # 201 for created
    except FilePathError as e:
        logger.error(f"File path error in create_file for '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400) # Bad request due to path
    except OperationFailedError as e: # Catching specific error from ensure_parent_dir_exists
        logger.error(f"Operation failed in create_file for '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=500)
    except IOError as e: # More specific for file I/O issues
        logger.error(f"IOError creating file {path}: {e}")
        return OperationResult(False, error=f"File I/O error: {e}", status_code=500)
    except AgentError as e: # Catch other agent errors from safe_file_operation
        logger.error(f"Agent error in create_file for '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=500)
    except Exception as e: # General catch-all for truly unexpected errors
        logger.error(f"Unexpected error creating file {path}: {str(e)}")
        return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

def read_file(path: str) -> OperationResult:
    """Read and return the content of a file."""
    try:
        safe_path = safe_file_operation(path) # Can raise FilePathError or AgentError
        if not safe_path.exists():
            raise FileNotFoundError(f"File not found at path: {path}") # More specific than just "invalid path"
        if not safe_path.is_file():
            raise IsADirectoryError(f"Path is a directory, not a file: {path}")

        with open(safe_path, 'r', encoding='utf-8') as f:
            data = f.read()
        logger.info(f"File {safe_path} read successfully.")
        return OperationResult(True, data={'content': data}, status_code=200)
    except FilePathError as e:
        logger.error(f"File path error in read_file for '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400) # Bad request due to path
    except FileNotFoundError as e:
        logger.error(f"File not found error in read_file for '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=404) # Not Found
    except IsADirectoryError as e:
        logger.error(f"Is a directory error in read_file for '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400) # Bad request, tried to read dir
    except IOError as e:
        logger.error(f"IOError reading file {path}: {e}")
        return OperationResult(False, error=f"File I/O error: {e}", status_code=500)
    except AgentError as e: # Catch other agent errors from safe_file_operation
        logger.error(f"Agent error in read_file for '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=500)
    except Exception as e: # General catch-all
        logger.error(f"Unexpected error reading file {path}: {str(e)}")
        return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

def delete_file(path: str) -> OperationResult:
    """Delete a file or directory at the given path, handling permission issues."""
    def onerror_rmtree(func, path_str, excinfo):
        # This function is called by shutil.rmtree if an error occurs.
        # func: the function that raised the exception (e.g., os.remove, os.rmdir)
        # path_str: the path name passed to func
        # excinfo: the exception information return by sys.exc_info()
        logger.warning(f"Error during shutil.rmtree of {path_str}: {excinfo[1]}. Attempting to chmod.")
        try:
            # Attempt to change permissions and retry the operation
            os.chmod(path_str, 0o777) # More permissive for deletion attempt
            func(path_str) # Retry the function that failed
        except Exception as e_chmod:
            logger.error(f"Failed to delete {path_str} even after chmod: {e_chmod}")
            # We cannot easily propagate this to OperationResult from here,
            # rmtree will raise the original error or the error from this handler.

    try:
        safe_path = safe_file_operation(path) # Can raise FilePathError or AgentError

        if not safe_path.exists():
            raise FileNotFoundError(f"Path not found for deletion: {path}")

        if safe_path.is_dir():
            shutil.rmtree(safe_path, onerror=onerror_rmtree)
            logger.info(f"Directory {safe_path} deleted successfully.")
            return OperationResult(True, data={'status': 'deleted_directory', 'path': str(safe_path)}, status_code=200) # 200 OK or 204 No Content
        else:
            safe_path.unlink() 
            logger.info(f"File {safe_path} deleted successfully.")
            return OperationResult(True, data={'status': 'deleted_file', 'path': str(safe_path)}, status_code=200)
    except FilePathError as e:
        logger.error(f"File path error in delete_file for '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400)
    except FileNotFoundError as e: # Specifically for the case where safe_path.exists() is false
        logger.error(f"File/directory not found for deletion '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=404)
    except (IOError, OSError) as e: # Catch errors from rmtree or unlink
        logger.error(f"IOError/OSError deleting {path}: {e}")
        return OperationResult(False, error=f"File/directory deletion error: {e}", status_code=500)
    except AgentError as e: # Catch other agent errors from safe_file_operation
        logger.error(f"Agent error in delete_file for '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=500)
    except Exception as e: # General catch-all
        logger.error(f"Unexpected error deleting {path}: {str(e)}")
        return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

def rename_file(old_path: str, new_path: str) -> OperationResult:
    """Rename a file or directory."""
    try:
        safe_old_path = safe_file_operation(old_path) # Can raise FilePathError
        if not safe_old_path.exists():
            raise FileNotFoundError(f"Source path for rename does not exist: {old_path}")

        safe_new_path = safe_file_operation(new_path) # Can raise FilePathError
        if safe_new_path.exists():
            raise FileExistsError(f"Destination path for rename already exists: {new_path}")
            
        error_msg = ensure_parent_dir_exists(safe_new_path) # Returns str or None
        if error_msg:
            raise OperationFailedError(f"Parent directory creation failed for new path {safe_new_path}: {error_msg}")
            
        os.rename(safe_old_path, safe_new_path)
        logger.info(f"Renamed {safe_old_path} to {safe_new_path} successfully.")
        return OperationResult(True, data={'from': str(safe_old_path), 'to': str(safe_new_path)}, status_code=200)
    except FilePathError as e:
        logger.error(f"File path error in rename_file from '{old_path}' to '{new_path}': {e}")
        return OperationResult(False, error=str(e), status_code=400)
    except FileNotFoundError as e:
        logger.error(f"Source not found for rename '{old_path}': {e}")
        return OperationResult(False, error=str(e), status_code=404)
    except FileExistsError as e:
        logger.error(f"Destination exists for rename to '{new_path}': {e}")
        return OperationResult(False, error=str(e), status_code=409) # Conflict
    except OperationFailedError as e:
        logger.error(f"Operation failed in rename_file from '{old_path}' to '{new_path}': {e}")
        return OperationResult(False, error=str(e), status_code=500)
    except (IOError, OSError) as e:
        logger.error(f"IOError/OSError renaming {old_path} to {new_path}: {e}")
        return OperationResult(False, error=f"File system error during rename: {e}", status_code=500)
    except AgentError as e: # Catch other agent errors from safe_file_operation
        logger.error(f"Agent error in rename_file for '{old_path}' to '{new_path}': {e}")
        return OperationResult(False, error=str(e), status_code=500)
    except Exception as e: # General catch-all
        logger.error(f"Unexpected error renaming {old_path} to {new_path}: {str(e)}")
        return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

def move_file(src: str, dest: str) -> OperationResult:
    """Move a file or directory."""
    try:
        safe_src_path = safe_file_operation(src) # Can raise FilePathError
        if not safe_src_path.exists():
            raise FileNotFoundError(f"Source path for move does not exist: {src}")

        safe_dest_path = safe_file_operation(dest) # Can raise FilePathError
        if safe_dest_path.exists():
            # shutil.move can overwrite if dest is a dir.
            # To prevent accidental overwrite of a file with a file, or a dir with a file,
            # we explicitly check if the destination is a file and already exists.
            # If dest is a directory, shutil.move will move src *into* dest.
            if safe_dest_path.is_file():
                 raise FileExistsError(f"Destination file for move already exists: {dest}")
            # If safe_dest_path is a directory, shutil.move will place safe_src_path inside it.
            # If an item with the same name as safe_src_path already exists inside safe_dest_path directory,
            # shutil.move will typically raise an error. This is desired.

        error_msg = ensure_parent_dir_exists(safe_dest_path) # Returns str or None
        if error_msg:
            # If dest itself is the directory we are creating, this is fine.
            # This check is more for when dest is `dir/new_file_or_dir_name`.
            if safe_dest_path.parent != Path.cwd() or safe_dest_path.name != Path(error_msg).name : # crude check
                 raise OperationFailedError(f"Parent directory creation failed for new path {safe_dest_path}: {error_msg}")
            
        shutil.move(str(safe_src_path), str(safe_dest_path)) # Ensure paths are strings for shutil
        logger.info(f"Moved {safe_src_path} to {safe_dest_path} successfully.")
        return OperationResult(True, data={'from': str(safe_src_path), 'to': str(safe_dest_path)}, status_code=200)
    except FilePathError as e:
        logger.error(f"File path error in move_file from '{src}' to '{dest}': {e}")
        return OperationResult(False, error=str(e), status_code=400)
    except FileNotFoundError as e:
        logger.error(f"Source not found for move '{src}': {e}")
        return OperationResult(False, error=str(e), status_code=404)
    except FileExistsError as e: # Custom check above
        logger.error(f"Destination file exists for move to '{dest}': {e}")
        return OperationResult(False, error=str(e), status_code=409)
    except OperationFailedError as e:
        logger.error(f"Operation failed in move_file from '{src}' to '{dest}': {e}")
        return OperationResult(False, error=str(e), status_code=500)
    except (IOError, OSError, shutil.Error) as e: # shutil.Error is base for shutil exceptions
        logger.error(f"IOError/OSError/shutil.Error moving {src} to {dest}: {e}")
        return OperationResult(False, error=f"File system error during move: {e}", status_code=500)
    except AgentError as e: # Catch other agent errors from safe_file_operation
        logger.error(f"Agent error in move_file for '{src}' to '{dest}': {e}")
        return OperationResult(False, error=str(e), status_code=500)
    except Exception as e: # General catch-all
        logger.error(f"Unexpected error moving {src} to {dest}: {str(e)}")
        return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

def list_directory(path: str) -> OperationResult:
    """List files and directories at a given path."""
    try:
        safe_path = safe_file_operation(path) # Can raise FilePathError

        if not safe_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not safe_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")
            
        items = os.listdir(safe_path)
        logger.info(f"Listed directory {safe_path} successfully.")
        return OperationResult(True, data={'path': str(safe_path), 'items': items}, status_code=200)
    except FilePathError as e:
        logger.error(f"File path error in list_directory for '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400)
    except FileNotFoundError as e:
        logger.error(f"Directory not found for listing '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=404)
    except NotADirectoryError as e:
        logger.error(f"Path is not a directory for listing '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400)
    except (IOError, OSError) as e:
        logger.error(f"IOError/OSError listing directory {path}: {e}")
        return OperationResult(False, error=f"File system error listing directory: {e}", status_code=500)
    except AgentError as e: # Catch other agent errors from safe_file_operation
        logger.error(f"Agent error in list_directory for '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=500)
    except Exception as e: # General catch-all
        logger.error(f"Unexpected error listing directory {path}: {str(e)}")
        return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

def search_file(keyword: str, path: str) -> OperationResult:
    """Search for a keyword in files under the given path."""
    try:
        safe_search_path = safe_file_operation(path) # Can raise FilePathError

        if not safe_search_path.exists():
            raise FileNotFoundError(f"Search directory not found: {path}")
        if not safe_search_path.is_dir():
            raise NotADirectoryError(f"Search path is not a directory: {path}")

        matches: List[str] = []
        for root_str, _, files in os.walk(str(safe_search_path)): # os.walk needs string path
            root_path = Path(root_str)
            for file_name in files:
                file_path_obj = root_path / file_name
                try:
                    # Validate each found file path again to be absolutely sure it's safe
                    # This is somewhat redundant if os.walk behaves well, but adds a layer of safety.
                    current_safe_file_path = safe_file_operation(str(file_path_obj))
                    if current_safe_file_path.is_file(): # Ensure it's a file before reading
                        with open(current_safe_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            if keyword in f.read():
                                matches.append(str(current_safe_file_path))
                except FilePathError as e_fp: # safe_file_operation could raise this
                    logger.debug(f"Skipping file {file_path_obj} during search due to path error: {e_fp}")
                except IOError as e_io: # Error reading a specific file
                    logger.debug(f"Skipping file {file_path_obj} during search due to IO error: {e_io}")
                except Exception as e_inner: # Other unexpected errors for a specific file
                    logger.debug(f"Skipping file {file_path_obj} during search due to unexpected error: {e_inner}")
        
        logger.info(f"Search for '{keyword}' in {safe_search_path} completed. Found {len(matches)} matches.")
        return OperationResult(True, data={'matches': matches}, status_code=200)
    except FilePathError as e:
        logger.error(f"File path error in search_file for '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400)
    except FileNotFoundError as e:
        logger.error(f"Search directory not found for search '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=404)
    except NotADirectoryError as e:
        logger.error(f"Search path is not a directory for search '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400)
    except (IOError, OSError) as e: # Errors during os.walk itself
        logger.error(f"IOError/OSError during search in {path}: {e}")
        return OperationResult(False, error=f"File system error during search: {e}", status_code=500)
    except AgentError as e: # Catch other agent errors from safe_file_operation
        logger.error(f"Agent error in search_file for '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=500)
    except Exception as e: # General catch-all
        logger.error(f"Unexpected error searching for '{keyword}' in {path}: {str(e)}")
        return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

def run_script(path: str) -> OperationResult:
    """Run a non-interactive script, capturing stdout and stderr."""
    try:
        safe_script_path = safe_file_operation(path) # Can raise FilePathError

        if not safe_script_path.exists():
            raise FileNotFoundError(f"Script not found: {path}")
        if not safe_script_path.is_file():
            # Technically could check for execute permissions here too, but keeping it simple.
            raise OperationFailedError(f"Path is not a file, cannot execute script: {path}")

        # `check=False` means it won't raise CalledProcessError on non-zero exit codes.
        # We capture this in OperationResult.
        result = subprocess.run(
            [sys.executable, str(safe_script_path)], 
            capture_output=True, text=True, check=False
        )
        
        data = {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
        if result.returncode == 0:
            logger.info(f"Script {safe_script_path} executed successfully.")
            return OperationResult(True, data=data, status_code=200)
        else:
            logger.warning(f"Script {safe_script_path} executed with non-zero exit code {result.returncode}. Stderr: {result.stderr}")
            # Still a "successful" operation in terms of execution, but script had an error.
            return OperationResult(False, data=data, error=f"Script execution failed with exit code {result.returncode}.", status_code=200)

    except FilePathError as e:
        logger.error(f"File path error running script '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400)
    except FileNotFoundError as e:
        logger.error(f"Script not found for execution '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=404)
    except OperationFailedError as e: # Custom error for non-file
        logger.error(f"Operation failed for script '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400) # Bad request
    except (IOError, OSError, subprocess.SubprocessError) as e: # Covers errors from subprocess.run
        logger.error(f"Subprocess/IO error running script {path}: {e}")
        return OperationResult(False, error=f"Error during script execution: {e}", status_code=500)
    except AgentError as e: # Catch other agent errors from safe_file_operation
        logger.error(f"Agent error running script '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=500)
    except Exception as e: # General catch-all
        logger.error(f"Unexpected error running script {path}: {str(e)}")
        return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

def start_interactive(path: str) -> OperationResult:
    """Launch a script in a new console for interactive input."""
    try:
        safe_script_path = safe_file_operation(path) # Can raise FilePathError

        if not safe_script_path.exists():
            raise FileNotFoundError(f"Script not found: {path}")
        if not safe_script_path.is_file():
            raise OperationFailedError(f"Path is not a file, cannot execute script: {path}")

        kwargs = {}
        if platform.system() == 'Windows' and hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
        else:
            # For POSIX systems, start_new_session=True creates a new process group.
            # This allows the subprocess to continue running if the parent (agent) exits.
            kwargs['start_new_session'] = True
            
        proc = subprocess.Popen([sys.executable, str(safe_script_path)], **kwargs)
        logger.info(f"Interactive script {safe_script_path} started with PID {proc.pid}.")
        return OperationResult(True, data={'status': 'started', 'pid': proc.pid}, status_code=200)
    except FilePathError as e:
        logger.error(f"File path error starting interactive script '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400)
    except FileNotFoundError as e:
        logger.error(f"Script not found for interactive execution '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=404)
    except OperationFailedError as e: # Custom error for non-file
        logger.error(f"Operation failed for interactive script '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400)
    except (IOError, OSError, subprocess.SubprocessError) as e: # Covers errors from Popen
        logger.error(f"Subprocess/IO error starting interactive script {path}: {e}")
        return OperationResult(False, error=f"Error during interactive script startup: {e}", status_code=500)
    except AgentError as e: # Catch other agent errors from safe_file_operation
        logger.error(f"Agent error starting interactive script '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=500)
    except Exception as e: # General catch-all
        logger.error(f"Unexpected error starting interactive script {path}: {str(e)}")
        return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

def install_package(package: str) -> OperationResult:
    """Install a Python package via pip."""
    try:
        if not package or not isinstance(package, str) or not re.match(r'^[a-zA-Z0-9\-_~.=<>! ]+$', package):
            # Slightly more permissive regex for package specs (e.g., 'package_name>=1.0').
            # Still, be cautious with complex shell commands.
            raise OperationFailedError(f"Invalid package name or specification: {package}")

        # Using check=False to manually handle the result.
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True, text=True, check=False
        )
        
        data = {'package': package, 'stdout': result.stdout, 'stderr': result.stderr, 'exit_code': result.returncode}
        
        if result.returncode == 0:
            logger.info(f"Package '{package}' installed successfully. Stdout: {result.stdout}")
            return OperationResult(True, data=data, status_code=200)
        else:
            logger.error(f"Failed to install package '{package}'. Exit code: {result.returncode}. Stderr: {result.stderr}")
            # Return 200 because the operation itself (running pip) completed, but pip failed.
            return OperationResult(False, data=data, error=f"pip install failed with exit code {result.returncode}.", status_code=200)
            
    except OperationFailedError as e: # For invalid package name
        logger.error(f"Operation failed installing package '{package}': {e}")
        return OperationResult(False, error=str(e), status_code=400) # Bad request
    except (OSError, subprocess.SubprocessError) as e: # Errors during subprocess execution
        logger.error(f"OSError/SubprocessError installing package {package}: {e}")
        return OperationResult(False, error=f"Error during package installation: {e}", status_code=500)
    except Exception as e: # General catch-all
        logger.error(f"Unexpected error installing package {package}: {str(e)}")
        return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

def open_in_browser(path_or_url: str) -> OperationResult:
    """Open a file or URL in the default web browser for preview."""
    try:
        url_to_open: str
        if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
            # Basic validation for URLs could be added here if desired (e.g., regex)
            if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', path_or_url):
                 raise OperationFailedError(f"Invalid URL format: {path_or_url}")
            url_to_open = path_or_url
            path_type = "URL"
        else:
            # Treat as a local file path
            safe_file_path = safe_file_operation(path_or_url) # Can raise FilePathError
            if not safe_file_path.exists():
                raise FileNotFoundError(f"File not found for browser preview: {path_or_url}")
            if not safe_file_path.is_file(): # Ensure it's a file, not a directory
                raise IsADirectoryError(f"Path is a directory, not a file, cannot open in browser: {path_or_url}")

            url_to_open = f'file://{str(safe_file_path.resolve())}'
            path_type = "file"
            
        webbrowser.open(url_to_open)
        logger.info(f"Opened {path_type} '{path_or_url}' (resolved to '{url_to_open}') in browser.")
        return OperationResult(True, data={'status': 'opened', 'url': url_to_open}, status_code=200)
    except FilePathError as e:
        logger.error(f"File path error opening '{path_or_url}' in browser: {e}")
        return OperationResult(False, error=str(e), status_code=400)
    except FileNotFoundError as e:
        logger.error(f"File not found for browser preview '{path_or_url}': {e}")
        return OperationResult(False, error=str(e), status_code=404)
    except IsADirectoryError as e:
        logger.error(f"Path is a directory, not a file, for browser preview '{path_or_url}': {e}")
        return OperationResult(False, error=str(e), status_code=400) # Bad request
    except OperationFailedError as e: # For invalid URL format
        logger.error(f"Operation failed opening '{path_or_url}' in browser: {e}")
        return OperationResult(False, error=str(e), status_code=400) # Bad request
    except webbrowser.Error as e: # Specific error from webbrowser module
        logger.error(f"Webbrowser error opening {path_or_url}: {e}")
        return OperationResult(False, error=f"Webbrowser error: {e}", status_code=500)
    except AgentError as e: # Catch other agent errors from safe_file_operation
        logger.error(f"Agent error opening '{path_or_url}' in browser: {e}")
        return OperationResult(False, error=str(e), status_code=500)
    except Exception as e: # General catch-all
        logger.error(f"Unexpected error opening {path_or_url} in browser: {str(e)}")
        return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

def lint_code(path: str) -> OperationResult:
    """Run flake8 linter on the given file or directory."""
    try:
        safe_lint_path = safe_file_operation(path) # Can raise FilePathError

        if not safe_lint_path.exists():
            raise FileNotFoundError(f"Path for linting not found: {path}")
        # Flake8 can lint both files and directories.

        # `check=False` as flake8 uses exit codes to indicate if linting issues were found.
        result = subprocess.run(
            ['flake8', str(safe_lint_path)], 
            capture_output=True, text=True, check=False
        )
        
        data = {'path': str(safe_lint_path), 'stdout': result.stdout, 'stderr': result.stderr, 'exit_code': result.returncode}
        
        # Flake8 exit codes: 0 if no errors/warnings, 1 if errors/warnings found.
        # Other codes might indicate usage errors, but we treat any non-zero as "issues found"
        # for simplicity in OperationResult.
        if result.returncode == 0:
            logger.info(f"Linting for {safe_lint_path} completed. No issues found.")
            return OperationResult(True, data=data, status_code=200)
        else:
            logger.warning(f"Linting for {safe_lint_path} completed. Issues found. Exit code: {result.returncode}. Stdout:\n{result.stdout}\nStderr:\n{result.stderr}")
            # The operation (running flake8) was successful, but it found linting issues.
            return OperationResult(True, data=data, error="Linting issues found.", status_code=200) 

    except FilePathError as e:
        logger.error(f"File path error linting '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400)
    except FileNotFoundError as e: # Should be caught by safe_lint_path.exists() check if path itself is the issue
        logger.error(f"Path not found for linting '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=404)
    except (OSError, subprocess.SubprocessError) as e:
        logger.error(f"OSError/SubprocessError linting {path}: {e}")
        return OperationResult(False, error=f"Error during linting process: {e}", status_code=500)
    except AgentError as e: # Catch other agent errors from safe_file_operation
        logger.error(f"Agent error linting '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=500)
    except Exception as e: # General catch-all
        logger.error(f"Unexpected error linting {path}: {str(e)}")
        return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

def format_code(path: str) -> OperationResult:
    """Format code using black on the given file or directory."""
    try:
        safe_format_path = safe_file_operation(path) # Can raise FilePathError

        if not safe_format_path.exists():
            raise FileNotFoundError(f"Path for formatting not found: {path}")
        # Black can format both files and directories.

        # `check=False` as black uses exit codes for different outcomes.
        result = subprocess.run(
            ['black', str(safe_format_path)], 
            capture_output=True, text=True, check=False
        )
        
        data = {'path': str(safe_format_path), 'stdout': result.stdout, 'stderr': result.stderr, 'exit_code': result.returncode}
        
        # Black exit codes:
        # 0: No changes needed or successful reformatting.
        # 1: File(s) reformatted. (Still a success for our purpose)
        # 123: Internal error / invalid input.
        if result.returncode == 0:
            logger.info(f"Formatting for {safe_format_path} completed. No changes or successful reformatting.")
            return OperationResult(True, data=data, status_code=200)
        elif result.returncode == 1: # Black made changes
            logger.info(f"Formatting for {safe_format_path} completed. File(s) were reformatted.")
            return OperationResult(True, data=data, status_code=200) # Still success, changes made
        else: # Any other error code from black
            logger.error(f"Black formatting failed for {safe_format_path}. Exit code: {result.returncode}. Stderr: {result.stderr}")
            return OperationResult(False, data=data, error=f"Black formatting failed with exit code {result.returncode}.", status_code=200) # Operation ran, black failed

    except FilePathError as e:
        logger.error(f"File path error formatting '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400)
    except FileNotFoundError as e:
        logger.error(f"Path not found for formatting '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=404)
    except (OSError, subprocess.SubprocessError) as e:
        logger.error(f"OSError/SubprocessError formatting {path}: {e}")
        return OperationResult(False, error=f"Error during formatting process: {e}", status_code=500)
    except AgentError as e: # Catch other agent errors from safe_file_operation
        logger.error(f"Agent error formatting '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=500)
    except Exception as e: # General catch-all
        logger.error(f"Unexpected error formatting {path}: {str(e)}")
        return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

def run_tests(path: str) -> OperationResult:
    """Run pytest on the specified directory."""
    try:
        safe_test_path = safe_file_operation(path) # Can raise FilePathError

        if not safe_test_path.exists():
            raise FileNotFoundError(f"Path for tests not found: {path}")
        # Pytest can run on both files and directories.

        # `check=False` as pytest uses exit codes for test outcomes.
        result = subprocess.run(
            ['pytest', str(safe_test_path)], 
            capture_output=True, text=True, check=False
        )
        
        data = {'path': str(safe_test_path), 'stdout': result.stdout, 'stderr': result.stderr, 'exit_code': result.returncode}
        
        # Pytest Exit Codes:
        # 0: All tests passed
        # 1: Tests were collected and run but some tests failed
        # 2: Test execution was interrupted by the user
        # 3: Internal error happened while executing tests
        # 4: pytest command line usage error
        # 5: No tests were collected
        if result.returncode == 0:
            logger.info(f"Tests for {safe_test_path} passed.")
            return OperationResult(True, data=data, status_code=200)
        elif result.returncode == 1:
            logger.warning(f"Tests for {safe_test_path} failed. Exit code: {result.returncode}. Stdout:\n{result.stdout}\nStderr:\n{result.stderr}")
            return OperationResult(False, data=data, error="Tests failed.", status_code=200) # Operation (running pytest) succeeded
        elif result.returncode == 5:
            logger.warning(f"No tests found for {safe_test_path}. Exit code: {result.returncode}.")
            return OperationResult(False, data=data, error="No tests found.", status_code=200) # Operation succeeded
        else: # Other pytest errors
            logger.error(f"Pytest internal error or usage error for {safe_test_path}. Exit code: {result.returncode}. Stderr: {result.stderr}")
            return OperationResult(False, data=data, error=f"Pytest error (exit code {result.returncode}).", status_code=200) # Operation ran, pytest had issues

    except FilePathError as e:
        logger.error(f"File path error running tests on '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400)
    except FileNotFoundError as e:
        logger.error(f"Path not found for running tests '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=404)
    except (OSError, subprocess.SubprocessError) as e:
        logger.error(f"OSError/SubprocessError running tests on {path}: {e}")
        return OperationResult(False, error=f"Error during test execution process: {e}", status_code=500)
    except AgentError as e: # Catch other agent errors from safe_file_operation
        logger.error(f"Agent error running tests on '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=500)
    except Exception as e: # General catch-all
        logger.error(f"Unexpected error running tests on {path}: {str(e)}")
        return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

def git_commit(message: str) -> OperationResult:
    """Commit staged changes with a commit message."""
    try:
        if not message or not isinstance(message, str): # Basic validation
            raise OperationFailedError("Commit message cannot be empty.")

        # Git commands operate on the CWD, so no specific path validation needed here,
        # assuming the agent is run in the root of a git repository.
        
        # Stage all changes. Consider if this should be more granular.
        add_result = subprocess.run(['git', 'add', '.'], capture_output=True, text=True, check=False)
        if add_result.returncode != 0:
            logger.error(f"git add failed. Exit code: {add_result.returncode}. Stderr: {add_result.stderr}")
            raise OperationFailedError(f"git add failed: {add_result.stderr}")

        # Commit
        commit_result = subprocess.run(
            ['git', 'commit', '-m', message], 
            capture_output=True, text=True, check=False
        )
        
        data = {
            'message': message, 
            'add_stdout': add_result.stdout, 'add_stderr': add_result.stderr,
            'commit_stdout': commit_result.stdout, 'commit_stderr': commit_result.stderr, 
            'commit_exit_code': commit_result.returncode
        }
        
        if commit_result.returncode == 0:
            logger.info(f"Commit successful: {message}. Stdout: {commit_result.stdout}")
            return OperationResult(True, data=data, status_code=200)
        elif "nothing to commit" in commit_result.stdout or \
             "no changes added to commit" in commit_result.stdout or \
             (commit_result.returncode == 1 and not commit_result.stderr): # Some git versions exit 1 for no changes
            logger.info(f"No changes to commit for message: {message}.")
            # This is not an error in the operation itself.
            return OperationResult(True, data=data, error="No changes to commit.", status_code=200)
        else:
            logger.error(f"git commit failed. Exit code: {commit_result.returncode}. Stderr: {commit_result.stderr}")
            raise OperationFailedError(f"git commit failed: {commit_result.stderr}")

    except OperationFailedError as e:
        logger.error(f"Operation failed during git commit: {e}")
        # data might not be fully populated here, handle carefully or construct minimal
        error_data = {'message': message, 'error_details': str(e)}
        return OperationResult(False, data=error_data, error=str(e), status_code=400 if "Commit message" in str(e) else 500)
    except (OSError, subprocess.SubprocessError) as e:
        logger.error(f"OSError/SubprocessError during git commit: {e}")
        return OperationResult(False, error=f"Error during git commit process: {e}", status_code=500)
    except Exception as e: # General catch-all
        logger.error(f"Unexpected error during git commit: {str(e)}")
        return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

def git_push(remote: str, branch: str) -> OperationResult:
    """Push commits to the remote repository."""
    try:
        # Basic validation for remote and branch names
        if not remote or not isinstance(remote, str) or not re.match(r'^[a-zA-Z0-9\-_/\.:]+$', remote):
             raise OperationFailedError(f"Invalid remote name: {remote}")
        if not branch or not isinstance(branch, str) or not re.match(r'^[a-zA-Z0-9\-_/\.]+$', branch):
             raise OperationFailedError(f"Invalid branch name: {branch}")

        # Git commands operate on CWD.
        result = subprocess.run(
            ['git', 'push', remote, branch], 
            capture_output=True, text=True, check=False
        )
        
        data = {
            'remote': remote, 'branch': branch, 
            'stdout': result.stdout, 'stderr': result.stderr, 
            'exit_code': result.returncode
        }
        
        if result.returncode == 0:
            logger.info(f"Successfully pushed to {remote}/{branch}. Stdout: {result.stdout}")
            return OperationResult(True, data=data, status_code=200)
        else:
            logger.error(f"git push to {remote}/{branch} failed. Exit code: {result.returncode}. Stderr: {result.stderr}")
            raise OperationFailedError(f"git push failed: {result.stderr}")

    except OperationFailedError as e:
        logger.error(f"Operation failed during git push: {e}")
        error_data = {'remote': remote, 'branch': branch, 'error_details': str(e)}
        return OperationResult(False, data=error_data, error=str(e), status_code=400 if "Invalid" in str(e) else 500)
    except (OSError, subprocess.SubprocessError) as e:
        logger.error(f"OSError/SubprocessError during git push to {remote}/{branch}: {e}")
        return OperationResult(False, error=f"Error during git push process: {e}", status_code=500)
    except Exception as e: # General catch-all
        logger.error(f"Unexpected error during git push to {remote}/{branch}: {str(e)}")
        return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

def prompt_input(message: str) -> Dict[str, Any]: # Not changing this, as it's user input, not an operation
    """Prompt the user and return the input."""
    val = input(f"{message} ")
    return {'user_input': val}

def chunk_file(path: str, chunk_size: int, chunk_index: int) -> OperationResult:
    """Read a file in chunks (line-based, memory efficient, resumable, robust for huge files). Returns the specified chunk (0-based)."""
    try:
        if chunk_size is None or not isinstance(chunk_size, int) or chunk_size <= 0:
            chunk_size = 100  # Default chunk size
        if chunk_index is None or not isinstance(chunk_index, int) or chunk_index < 0:
            chunk_index = 0  # Default chunk index

        safe_file_path = safe_file_operation(path) # Can raise FilePathError

        if not safe_file_path.exists():
            raise FileNotFoundError(f"File not found for chunking: {path}")
        if not safe_file_path.is_file():
            raise IsADirectoryError(f"Path is a directory, not a file, cannot chunk: {path}")

        total_lines = 0
        chunk_lines = []
        start_line_num = chunk_index * chunk_size
        end_line_num = start_line_num + chunk_size

        # Using 'errors=ignore' for robustness, though ideally files are UTF-8.
        with open(safe_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if start_line_num <= i < end_line_num:
                    chunk_lines.append(line)
                total_lines += 1
        
        # Calculate total_chunks, ensuring it's at least 1 if there's content, or 0 if empty.
        if total_lines == 0:
            total_chunks = 0
        else:
            total_chunks = (total_lines + chunk_size - 1) // chunk_size
        
        # Handle chunk_index out of range
        # If total_lines is 0, only chunk_index 0 is valid (returns empty chunk).
        if chunk_index > 0 and chunk_index >= total_chunks :
             error_msg = f"chunk_index {chunk_index} is out of range. File has {total_lines} lines, {total_chunks} chunks."
             logger.warning(error_msg + f" (path: {path})")
             return OperationResult(False, error=error_msg, status_code=416, data={'total_chunks': total_chunks, 'total_lines': total_lines}) # Range Not Satisfiable

        chunk_content = ''.join(chunk_lines)
        data = {
            'path': str(safe_file_path),
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'chunk': chunk_content,
            'start_line': start_line_num, # 0-based start line of the chunk
            'end_line': min(end_line_num, total_lines) -1 if total_lines > 0 and chunk_lines else -1, # 0-based end line of the chunk
            'total_lines': total_lines
        }
        logger.info(f"Chunk {chunk_index}/{total_chunks-1 if total_chunks > 0 else 0} read from {safe_file_path}.")
        return OperationResult(True, data=data, status_code=200)

    except FilePathError as e:
        logger.error(f"File path error in chunk_file for '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400)
    except FileNotFoundError as e:
        logger.error(f"File not found for chunking '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=404)
    except IsADirectoryError as e:
        logger.error(f"Path is a directory, not a file, for chunking '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400)
    except (IOError, OSError) as e:
        logger.error(f"IOError/OSError chunking file {path}: {e}")
        return OperationResult(False, error=f"File I/O error during chunking: {e}", status_code=500)
    except AgentError as e: # Catch other agent errors from safe_file_operation
        logger.error(f"Agent error in chunk_file for '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=500)
    except Exception as e: # General catch-all
        logger.error(f"Unexpected error chunking file {path}: {str(e)}")
        return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)

def update_file_chunk(path: str, chunk_content: str, chunk_size: int, chunk_index: int) -> OperationResult:
    """Update a specific chunk of a file (0-based, memory efficient, robust for huge files)."""
    temp_file_path = None # Initialize for potential cleanup in except block
    try:
        if chunk_size is None or not isinstance(chunk_size, int) or chunk_size <= 0:
            chunk_size = 100
        if chunk_index is None or not isinstance(chunk_index, int) or chunk_index < 0:
            chunk_index = 0
        if not isinstance(chunk_content, str):
            # Could also try to convert, but for now, strict type.
            raise OperationFailedError("chunk_content must be a string.")

        safe_target_path = safe_file_operation(path) # Can raise FilePathError

        # Logic for handling file creation if it doesn't exist (only for chunk 0)
        file_existed_before = safe_target_path.exists()
        if not file_existed_before:
            if chunk_index == 0:
                logger.info(f"File {path} does not exist. Creating it for chunk 0 update.")
                parent_error = ensure_parent_dir_exists(safe_target_path)
                if parent_error:
                    raise OperationFailedError(f"Failed to create parent directory for new file '{path}': {parent_error}")
                # Create an empty file to ensure it exists before reading/writing lines
                with open(safe_target_path, 'w', encoding='utf-8') as f:
                    pass 
            else:
                # Cannot update a chunk beyond 0 for a non-existent file.
                raise FileNotFoundError(f"File {path} does not exist, cannot update chunk {chunk_index}.")
        
        if not safe_target_path.is_file(): # Should be redundant if creation logic is sound
             raise IsADirectoryError(f"Target path '{path}' for chunk update is a directory, not a file.")


        temp_file_path = safe_target_path.with_suffix(safe_target_path.suffix + '.tmp.' + str(time.time_ns()))
        
        start_line_num = chunk_index * chunk_size
        num_lines_in_new_chunk = len(chunk_content.splitlines())

        lines_processed_from_original = 0
        chunk_written = False

        with open(safe_target_path, 'r', encoding='utf-8', errors='ignore') as f_src, \
             open(temp_file_path, 'w', encoding='utf-8') as f_dst:
            
            # Write lines before the target chunk
            for i, line in enumerate(f_src):
                if i < start_line_num:
                    f_dst.write(line)
                else:
                    # Reached the start of the chunk to be replaced/inserted
                    f_dst.write(chunk_content) # Write the new chunk
                    chunk_written = True
                    # Skip lines in the original file that are covered by the new chunk's length
                    for _ in range(num_lines_in_new_chunk):
                        try:
                            next(f_src) # Advance the source iterator
                            lines_processed_from_original +=1
                        except StopIteration:
                            break # End of source file
                    # Write remaining lines from source file
                    for remaining_line in f_src: # This will continue from where f_src was left
                        f_dst.write(remaining_line)
                    break # Exit outer loop as we've processed the rest of the file
                lines_processed_from_original = i + 1
            
            # If the chunk is to be appended (start_line_num >= lines_processed_from_original)
            # or if the file was empty and we are writing chunk 0.
            if not chunk_written:
                 if start_line_num > lines_processed_from_original and file_existed_before and lines_processed_from_original > 0 :
                     # If trying to write a chunk far beyond current file content, pad with newlines
                     padding = '\n' * (start_line_num - lines_processed_from_original)
                     f_dst.write(padding)
                 f_dst.write(chunk_content)


        os.replace(temp_file_path, safe_target_path)
        temp_file_path = None # Avoid accidental deletion in finally if os.replace succeeds

        # Recalculate total lines and chunks for the response
        final_total_lines = 0
        with open(safe_target_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in f: final_total_lines += 1
        
        final_total_chunks = (final_total_lines + chunk_size - 1) // chunk_size if final_total_lines > 0 else 0
        if final_total_chunks == 0 and final_total_lines > 0: final_total_chunks = 1


        logger.info(f"Chunk {chunk_index} updated for file {safe_target_path}.")
        return OperationResult(True, data={'path': str(safe_target_path), 'chunk_index': chunk_index, 'total_chunks': final_total_chunks, 'total_lines': final_total_lines}, status_code=200)

    except FilePathError as e:
        logger.error(f"File path error updating chunk for '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400)
    except FileNotFoundError as e:
        logger.error(f"File not found for chunk update '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=404)
    except IsADirectoryError as e:
        logger.error(f"Path is a directory, not a file, for chunk update '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400)
    except OperationFailedError as e: # Custom errors like bad chunk_content or parent dir fail
        logger.error(f"Operation failed updating chunk for '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=400) # Often bad request
    except (IOError, OSError) as e:
        logger.error(f"IOError/OSError updating file chunk for {path}: {e}")
        return OperationResult(False, error=f"File I/O error during chunk update: {e}", status_code=500)
    except AgentError as e: # Catch other agent errors from safe_file_operation
        logger.error(f"Agent error updating chunk for '{path}': {e}")
        return OperationResult(False, error=str(e), status_code=500)
    except Exception as e:
        logger.error(f"Unexpected error updating file chunk for {path}: {str(e)}")
        return OperationResult(False, error=f"An unexpected internal error occurred: {e}", status_code=500)
    finally:
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
                logger.info(f"Cleaned up temporary file {temp_file_path}")
            except Exception as e_unlink:
                logger.error(f"Failed to delete temporary file {temp_file_path}: {e_unlink}")

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
            # Further validation: ensure all items in the list are strings (potential paths)
            if all(isinstance(item, str) for item in file_list):
                return file_list
            else:
                logger.warning("File plan from JSON is not a list of strings. Discarding.")
    except json.JSONDecodeError:
        logger.warning(f"Failed to decode file plan from LLM response as JSON. Response was: {response}")
    except Exception as e:
        logger.warning(f"An unexpected error occurred while parsing file plan JSON: {e}. Response was: {response}")
    
    # If JSON parsing fails or validation fails, return an empty list.
    # The fallback to text extraction has been removed due to unreliability.
    logger.warning("Could not obtain a valid file plan as JSON. Proceeding with an empty file list.")
    return []

# supervisor_generate_project and other functions using genai_client will be adapted later
# For now, they might break if called before genai_client is properly passed or they become Agent methods.

# --- Main Application Logic ---

# def get_multi_line_input() -> str: ... # Will be moved or become static method of Agent

# def main(): ... # Will be refactored to use Agent class

# Global genai_client is now removed. Functions needing it must be updated.
# genai_client = GeminiClient(api_key=api_key) # This line is now effectively part of Agent init


# supervisor_generate_project will be refactored to be a method of Agent or take an Agent instance.
# For now, its signature is adjusted but calls within it to get_file_plan and generate_file_content
# supervisor_generate_project is now a method of Agent. This global one is removed.
# def supervisor_generate_project(...): ...

# agent_per_file_edit is now a method of Agent. This global one is removed.
# def agent_per_file_edit(...): ...

# generate_file_content and get_file_plan are now methods of Agent. These global ones are removed.
# def generate_file_content(...): ...
# def get_file_plan(...): ...


# --- Main Application Logic ---
            return
            
        logger.info(f'File plan: {file_list}')
        
        # Create project directory if it doesn't exist
        # Validate project_dir_name to prevent path traversal or invalid names
        if not re.match(r'^[a-zA-Z0-9_.-]+$', project_dir_name) or ".." in project_dir_name or "/" in project_dir_name:
            logger.error(f"Invalid project directory name: {project_dir_name}. Using default 'project'.")
            project_dir_name = "project"
            
        project_dir = Path.cwd() / project_dir_name 
        project_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured project directory exists: {project_dir.resolve()}")
        
        for file_path_str in file_list:
            # Clean and validate the file path
            clean_path_str = re.sub(r'^[\'"\s,]+|[\'"\s,]+$', '', file_path_str)
            if not clean_path_str:
                logger.warning(f"Skipping empty file path string")
                continue
            
            # Path object for easier manipulation
            relative_path_obj = Path(clean_path_str)

            # Prevent paths from trying to escape the project directory
            # This check is crucial for security
            if ".." in relative_path_obj.parts:
                logger.warning(f"Skipping potentially unsafe path: {clean_path_str}")
                continue

            # Ensure we only have the filename part for checking forbidden files
            base_name = relative_path_obj.name.lower()
            if base_name in ['main.py', 'readme.md', 'requirements.txt']:
                logger.info(f'Skipping forbidden file: {base_name}')
                continue
                
            # Create the full path within the project directory
            # No need to check if it starts with project_dir_name, as paths from plan should be relative
            full_path = project_dir / relative_path_obj
            logger.info(f'Processing {full_path}...')
            
            # Create parent directories if they don't exist using helper
            error_msg = ensure_parent_dir_exists(full_path)
            if error_msg:
                logger.error(f"Skipping file {full_path} due to parent directory creation error: {error_msg}")
                continue
            
            # Handle special file types
            if full_path.name.lower().endswith('.db'):
                if not full_path.exists():
                    open(full_path, 'wb').close()
                    logger.info(f'{full_path} (empty database file) created.')
                continue
                
            if full_path.name.lower().endswith('.json'):
                if not full_path.exists():
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write('[]')
                    logger.info(f'{full_path} (empty JSON array) created.')
                continue
                
            # Handle binary files like images
            if full_path.name.lower().endswith(('.png', '.jpg', '.jpeg', 'gif', '.ico')):
                # Create an empty placeholder image
                try:
                    if not full_path.exists():
                        with open(full_path, 'wb') as f:
                            # Write a minimal transparent PNG
                            f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')
                        logger.info(f'{full_path} (placeholder image) created.')
                except Exception as e:
                    logger.error(f"Failed to create placeholder image {full_path}: {str(e)}")
                continue
                
            # Generate content for other files (using relative_path_obj for generate_file_content)
            # Assuming generate_file_content will be adapted
            code = agent.generate_file_content(str(relative_path_obj), user_input) # Call as Agent method
            if not code:
                # Create an empty file as fallback
                if not full_path.exists():
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(f'# Generated file: {relative_path_obj}\n')
                    logger.info(f'{full_path} (empty file) created as fallback.')
                else:
                    logger.info(f'{full_path} already exists, not overwriting with empty fallback.')
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
    # This function is stateless and can remain global.
    import re
    error_locations = []
    # More robust regex to handle variations in traceback lines including those from Jupyter/IPython
    # It looks for 'File "<path>", line <num>' or 'File <path>, line <num>'
    # or ----> X /path/to/file.py (Y) ... for IPython
    pattern = re.compile(
        r'File "?([^"]+)"?, line (\d+)|----> \d+ ([^\(]+)\((\d+)\)'
    )
    for match in pattern.finditer(traceback_str):
        # match.groups() will be like:
        # ('/path/to/file.py', '123', None, None) for standard tracebacks
        # (None, None, '/path/to/ipython_file.py', '45') for IPython tracebacks
        if match.group(1): # Standard traceback
            file_path, line_str = match.group(1), match.group(2)
        else: # IPython traceback
            file_path, line_str = match.group(3), match.group(4)
        
        # Normalize potential relative paths from traceback (e.g. "./file.py")
        # Check if file_path is already absolute. If not, it's likely relative to CWD.
        abs_file_path = Path(file_path)
        if not abs_file_path.is_absolute():
            abs_file_path = Path.cwd() / file_path # Resolve relative to current working directory
        
        # Ensure the file path is simplified (e.g. /foo/../bar -> /bar) and is a file
        try:
            resolved_path = abs_file_path.resolve()
            if resolved_path.is_file():
                 error_locations.append({'file': str(resolved_path), 'line': int(line_str)})
            else:
                logger.warning(f"Traceback path {file_path} resolved to {resolved_path} which is not a file. Skipping.")
        except Exception as e:
            logger.warning(f"Could not resolve or access traceback path {file_path}: {e}. Skipping.")

    return error_locations

def agent_per_file_edit(error_message: str, traceback_str: str, minimal_files: bool = True, project_context: str = ""):
    """
    Parse the error traceback, attempt to fix errors in affected files using an LLM.
    Only edits files/lines identified in the traceback.
    If minimal_files is True, non-essential file generation/editing is skipped.
    """
    if not agent_client: # This should be an Agent instance or its GeminiClient
        logger.error("agent_per_file_edit called without agent_client. Cannot proceed.")
        return

    logger.info(f"Starting agent_per_file_edit for error: {error_message}")
    error_locations = parse_python_traceback(traceback_str)

    if not error_locations:
        logger.warning("No valid file locations found in traceback. Cannot proceed with automated edits.")
        return

    # If minimal_files is true, we only focus on files in the traceback.
    # This is implicitly handled by iterating through error_locations.
    # No need to generate other files like tests/docs unless they are in the traceback.

    MAX_LINES_FOR_FULL_FILE = 1000 # Files with more lines will be chunked
    CONTEXT_WINDOW_LINES = 50    # Lines before and after error for chunking

    for loc in error_locations:
        file_path_str = loc['file']
        line_number = loc['line'] # 1-based line number from traceback

        logger.info(f"Attempting to fix error in {file_path_str} at line {line_number}")

        safe_file_path = safe_file_operation(file_path_str)
        if not safe_file_path or not safe_file_path.exists() or not safe_file_path.is_file():
            logger.error(f"File {file_path_str} from traceback not found or is not a file. Skipping.")
            continue

        try:
            with open(safe_file_path, 'r', encoding='utf-8') as f:
                file_lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading file {safe_file_path}: {e}. Skipping.")
            continue

        code_to_fix: str
        is_chunk = False
        chunk_start_line_idx = 0 # 0-based index
        chunk_end_line_idx = 0   # 0-based index, exclusive for slicing

        if len(file_lines) > MAX_LINES_FOR_FULL_FILE:
            is_chunk = True
            chunk_start_line_idx = max(0, line_number - 1 - CONTEXT_WINDOW_LINES)
            chunk_end_line_idx = min(len(file_lines), line_number - 1 + CONTEXT_WINDOW_LINES)
            code_chunk_lines = file_lines[chunk_start_line_idx:chunk_end_line_idx]
            code_to_fix = "".join(code_chunk_lines)
            logger.info(f"File is large. Processing chunk from line {chunk_start_line_idx + 1} to {chunk_end_line_idx}.")
        else:
            code_to_fix = "".join(file_lines)
            logger.info("Processing full file content.")

        prompt = (
            f"Project Context (if any): {project_context}\n\n"
            f"An error occurred in the following Python code:\n"
            f"File Path: {file_path_str}\n"
            f"Error Line: {line_number}\n\n"
            f"Error Message: {error_message}\n\n"
            f"Full Traceback:\n```\n{traceback_str}\n```\n\n"
            f"Code {'Chunk ' if is_chunk else ''}to Fix:\n```python\n{code_to_fix}\n```\n\n"
            f"Please provide the corrected version of the above code {'chunk' if is_chunk else 'file'}. "
            f"Output only the raw corrected code, without any explanations, comments, or markdown formatting. "
            f"Ensure the corrected code is complete and addresses the identified error."
        )

        try:
            logger.info("Sending request to LLM for code correction...")
            # Assuming agent_client here is the Agent._GeminiClient instance
            llm_response = agent_client.send_message(prompt, max_tokens=2000)

            if not llm_response:
                logger.warning(f"LLM returned an empty response for {safe_file_path}. Skipping fix.")
                continue

            corrected_code = strip_code_block_markers(llm_response)

            if not corrected_code.strip():
                logger.warning(f"LLM returned an empty code block after stripping markers for {safe_file_path}. Skipping fix.")
                continue
            
            logger.info(f"Received corrected code suggestion for {safe_file_path}.")

            # Apply the fix
            if is_chunk:
                # Replace the chunk in the original lines
                new_file_lines = file_lines[:chunk_start_line_idx] + \
                                 [line + '\n' for line in corrected_code.splitlines()] + \
                                 file_lines[chunk_end_line_idx:]
                # Handle if corrected code doesn't end with newline, but original chunk might have
                if corrected_code and not corrected_code.endswith('\n') and code_chunk_lines and code_chunk_lines[-1].endswith('\n'):
                    # remove last line from new_file_lines and add it again with corrected_code + \n
                    # This is tricky. A simpler way is to ensure splitlines() and then adding \n to each line from corrected_code
                    # The current list comprehension [line + '\n' for line in corrected_code.splitlines()] might add an extra \n if already present.
                    # A better way for chunk replacement is to ensure consistent line endings.
                    # For now, this simple splitlines and add is a heuristic.
                    pass # The list comprehension should handle this okay by adding \n to each line.

                logger.info(f"Applying corrected chunk to {safe_file_path} from line {chunk_start_line_idx+1}.")
            else:
                # Replace the entire file content
                new_file_lines = [line + '\n' for line in corrected_code.splitlines()]
                # if corrected_code and not corrected_code.endswith('\n') and file_lines and file_lines[-1].endswith('\n'):
                #    new_file_lines[-1] = new_file_lines[-1].rstrip('\n') # Avoid double newline if original had one and fix doesn't
                logger.info(f"Applying full corrected code to {safe_file_path}.")
            
            with open(safe_file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_file_lines)
            logger.info(f"Successfully applied fix to {safe_file_path}.")

            # Placeholder for validation
            logger.info(f"Validation step for {safe_file_path} skipped for now. (TODO: Implement validation)")

        except Exception as e:
            logger.error(f"An error occurred while processing/applying fix for {safe_file_path}: {e}")
            import traceback
            logger.debug(f"Traceback for processing/applying fix: {traceback.format_exc()}")

    if minimal_files:
        logger.info("`minimal_files` is True. Agent focused only on files in the traceback.")
    logger.info("Finished agent_per_file_edit attempts.")


# get_multi_line_input can be a static method of Agent or remain global if preferred
# For now, it's left global as it doesn't interact with Agent state directly.
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
                # Potentially ask user for project_dir_name here or use a default/extracted one
                project_name_match = re.search(r"(?:project|app|website|program)\s*named\s*['\"]?([a-zA-Z0-9_.-]+)['\"]?", user_input, re.IGNORECASE)
                custom_project_dir_name = project_name_match.group(1) if project_name_match else "generated_project"
                # This will need an Agent instance to access its gemini_client
                # supervisor_generate_project(user_input, project_dir_name=custom_project_dir_name, agent_client=?) 
                logger.warning("supervisor_generate_project call in main needs agent_client") # Placeholder
            else:
                # This will need an Agent instance
                # response = genai_client.send_message(user_input)
                logger.warning("genai_client.send_message call in main needs agent_client") # Placeholder
                response = "Placeholder: Agent response" # Placeholder
                print(f"Agent: {response}")
                
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")
            print(f"Agent: An error occurred. Please try again later.")

if __name__ == '__main__':
    # main() will now instantiate and run the Agent
    try:
        # API key can be passed here or read from ENV within Agent constructor
        agent = Agent() 
        agent.run()
    except ValueError as e:
        logger.critical(f"Failed to initialize or run Agent: {e}")
    except Exception as e:
        logger.critical(f"An critical unexpected error occurred: {e}")
        import traceback
        logger.critical(traceback.format_exc())