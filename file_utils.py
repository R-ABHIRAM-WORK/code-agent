# file_utils.py

import os
from pathlib import Path
import shutil
import logging
from typing import Any, Dict, List, Optional

# Configure logging (can be configured by the main application)
logger = logging.getLogger(__name__)

class OperationResult:
    """
    Standardized operation result structure.
    Used by file and environment utility functions to return consistent success/failure information.
    """
    def __init__(self, success: bool, message: str = "", data: Any = None, error: Optional[str] = None, status_code: int = 200):
        """
        Initializes the OperationResult.

        Args:
            success: True if the operation was successful, False otherwise.
            message: A general message about the operation (can be used for non-error info).
            data: Any data returned by the successful operation (e.g., file content, list of files).
            error: A string describing the error if the operation failed.
            status_code: An HTTP-like status code (e.g., 200 for success, 404 for not found, 500 for server error).
        """
        self.success = success
        self.message = message
        self.data = data
        self.error = error
        self.status_code = status_code

    def __repr__(self):
        return f"OperationResult(success={self.success}, message='{self.message}', data={self.data}, error='{self.error}', status_code={self.status_code})"

def safe_file_operation(path: str) -> Optional[Path]:
    """
    Safely resolves and validates a file path relative to the current working directory.

    Args:
        path: The file path to validate.

    Returns:
        A resolved Path object if the path is valid and within the current working directory's scope.
        None if the path is invalid, outside the allowed scope, or if an error occurs during resolution.
    """
    try:
        safe_path = Path(path).resolve()
        base_path = Path.cwd().resolve()
        
        if base_path in safe_path.parents or base_path == safe_path:
            return safe_path
        
        logger.warning(f"Path validation failed: '{path}' is outside the current working directory '{base_path}'.")
        return None
    except (ValueError, RuntimeError) as e:
        logger.error(f"Error resolving path '{path}': {e}", exc_info=True)
        return None

def create_file(path: str, content: str) -> OperationResult:
    """
    Creates or overwrites a file with the given content.
    Parent directories are automatically created if they don't exist.

    Args:
        path: The path to the file to be created/overwritten.
        content: The string content to write to the file.

    Returns:
        An OperationResult indicating success or failure.
        On success, data contains {'path': str(absolute_file_path)}.
    """
    safe_path = safe_file_operation(path)
    if not safe_path:
        return OperationResult(False, error="Invalid or disallowed path for create_file.", status_code=400)
            
    try:
        dirpath = safe_path.parent
        os.makedirs(dirpath, exist_ok=True)
            
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"File '{safe_path}' created/updated successfully.")
        return OperationResult(True, data={'path': str(safe_path)}, message=f"File '{safe_path}' created/updated.")
    except Exception as e:
        logger.error(f"Error creating file '{safe_path}': {str(e)}", exc_info=True)
        return OperationResult(False, error=f"Failed to create file: {str(e)}", status_code=500)

def read_file(path: str) -> OperationResult:
    """
    Reads and returns the content of a file.

    Args:
        path: The path to the file to be read.

    Returns:
        An OperationResult. On success, data contains {'content': file_content_string}.
        On failure, error details are provided.
    """
    safe_path = safe_file_operation(path)
    if not safe_path:
        return OperationResult(False, error="Invalid or disallowed path for read_file.", status_code=400)
            
    if not safe_path.is_file():
        logger.error(f"Error reading file: '{safe_path}' is not a file or does not exist.")
        return OperationResult(False, error="File not found or path is not a file.", status_code=404)

    try:
        with open(safe_path, 'r', encoding='utf-8') as f:
            data = f.read()
        logger.info(f"File '{safe_path}' read successfully.")
        return OperationResult(True, data={'content': data})
    except Exception as e:
        logger.error(f"Error reading file '{safe_path}': {str(e)}", exc_info=True)
        return OperationResult(False, error=f"Failed to read file: {str(e)}", status_code=500)

def update_file(path: str, content: str) -> OperationResult:
    """
    Updates an existing file with new content. This is an alias for create_file,
    as create_file will overwrite if the file exists.

    Args:
        path: The path to the file to be updated.
        content: The new string content for the file.

    Returns:
        An OperationResult, same as create_file.
    """
    logger.info(f"Updating file '{path}' (delegating to create_file).")
    return create_file(path, content)

def delete_file(path: str) -> OperationResult:
    """
    Deletes a file or directory at the given path.
    Handles potential permission issues on deletion, especially for directories.

    Args:
        path: The path to the file or directory to be deleted.

    Returns:
        An OperationResult. On success, data contains {'status': 'deleted_directory' or 'deleted_file', 'path': str(path)}.
    """
    safe_path = safe_file_operation(path)
    if not safe_path:
        return OperationResult(False, error="Invalid or disallowed path for delete_file.", status_code=400)

    def onerror_handler(func, fn_path_str, excinfo):
        logger.warning(f"Error during rmtree on '{fn_path_str}': {excinfo}. Attempting to change permissions.")
        try:
            # Ensure fn_path_str is a Path object for os.access
            fn_path_obj = Path(fn_path_str)
            if not os.access(fn_path_obj, os.W_OK):
                os.chmod(fn_path_obj, 0o666)
                func(fn_path_obj) # Retry the operation
            else:
                # If already writable, chmod won't help. Re-raise to indicate persistent issue.
                # This part is tricky; rmtree might continue or stop based on the exception.
                # For simplicity, we log and let shutil.rmtree decide.
                logger.info(f"'{fn_path_str}' was already writable, or chmod did not resolve the issue.")
        except Exception as e_chmod:
            logger.error(f"Failed to delete '{fn_path_str}' even after attempting to change permissions: {e_chmod}", exc_info=True)
            # Do not re-raise here to allow rmtree to attempt to continue with other files if possible.

    try:
        if not safe_path.exists():
            logger.error(f"Deletion failed: Path '{safe_path}' not found.")
            return OperationResult(False, error='Path not found for deletion.', status_code=404)

        if safe_path.is_dir():
            shutil.rmtree(safe_path, onerror=onerror_handler)
            logger.info(f"Directory '{safe_path}' deleted successfully.")
            return OperationResult(True, data={'status': 'deleted_directory', 'path': str(safe_path)})
        else:
            os.remove(safe_path)
            logger.info(f"File '{safe_path}' deleted successfully.")
            return OperationResult(True, data={'status': 'deleted_file', 'path': str(safe_path)})
    except Exception as e:
        logger.error(f"Error deleting path '{safe_path}': {str(e)}", exc_info=True)
        return OperationResult(False, error=f"Failed to delete path: {str(e)}", status_code=500)

def rename_file(old_path: str, new_path: str) -> OperationResult:
    """
    Renames a file or directory. Parent directories for the new path are created if they don't exist.

    Args:
        old_path: The current path to the file or directory.
        new_path: The new path for the file or directory.

    Returns:
        An OperationResult. On success, data contains {'status': 'renamed', 'from': old_path, 'to': new_path}.
    """
    safe_old_path = safe_file_operation(old_path)
    if not safe_old_path:
        return OperationResult(False, error=f"Invalid or disallowed old path: '{old_path}'", status_code=400)
    if not safe_old_path.exists():
        return OperationResult(False, error=f"Source path '{old_path}' does not exist for renaming.", status_code=404)

    # For new_path, we primarily care that its parent directory can be created.
    # The actual new_path itself should not exist yet.
    try:
        # Resolve to handle relative paths robustly for parent directory creation
        resolved_new_path = Path(new_path).resolve()
    except (ValueError, RuntimeError) as e:
         logger.error(f"Error resolving new path '{new_path}': {e}", exc_info=True)
         return OperationResult(False, error=f"Invalid new path format: '{new_path}'", status_code=400)

    if resolved_new_path.exists():
        return OperationResult(False, error=f"Destination path '{new_path}' already exists.", status_code=409)

    try:
        parent_dir_of_new_path = resolved_new_path.parent
        os.makedirs(parent_dir_of_new_path, exist_ok=True)
        
        os.rename(safe_old_path, resolved_new_path)
        logger.info(f"Path '{old_path}' renamed to '{resolved_new_path}' successfully.")
        return OperationResult(True, data={'status': 'renamed', 'from': str(safe_old_path), 'to': str(resolved_new_path)})
    except Exception as e:
        logger.error(f"Error renaming '{old_path}' to '{new_path}': {str(e)}", exc_info=True)
        return OperationResult(False, error=f"Failed to rename: {str(e)}", status_code=500)

def move_file(src: str, dest: str) -> OperationResult:
    """
    Moves a file or directory. This is similar to rename but uses `shutil.move`
    which can handle cross-filesystem moves. Parent directories for the destination are created.

    Args:
        src: The source path of the file or directory.
        dest: The destination path.

    Returns:
        An OperationResult. On success, data contains {'status': 'moved', 'from': src, 'to': dest}.
    """
    safe_src_path = safe_file_operation(src)
    if not safe_src_path:
        return OperationResult(False, error=f"Invalid or disallowed source path: '{src}'", status_code=400)
    if not safe_src_path.exists():
        return OperationResult(False, error=f"Source path '{src}' does not exist for moving.", status_code=404)
    
    try:
        resolved_dest_path = Path(dest).resolve()
        dest_parent_dir = resolved_dest_path.parent
    except (ValueError, RuntimeError) as e:
        logger.error(f"Error resolving destination path '{dest}': {e}", exc_info=True)
        return OperationResult(False, error=f"Invalid destination path format: '{dest}'", status_code=400)

    try:
        os.makedirs(dest_parent_dir, exist_ok=True)
        
        shutil.move(str(safe_src_path), str(resolved_dest_path))
        logger.info(f"Path '{src}' moved to '{resolved_dest_path}' successfully.")
        return OperationResult(True, data={'status': 'moved', 'from': str(safe_src_path), 'to': str(resolved_dest_path)})
    except shutil.Error as e: # Catches errors like "Destination path '...' already exists"
        logger.error(f"shutil.Error moving '{src}' to '{dest}': {str(e)}", exc_info=True)
        return OperationResult(False, error=str(e), status_code=409) 
    except Exception as e:
        logger.error(f"Error moving '{src}' to '{dest}': {str(e)}", exc_info=True)
        return OperationResult(False, error=f"Failed to move: {str(e)}", status_code=500)

def list_directory(path: str = ".") -> OperationResult:
    """
    Lists files and directories at a given path. Defaults to the current directory.

    Args:
        path: The directory path to list. Defaults to ".".

    Returns:
        An OperationResult. On success, data contains {'path': str(absolute_path), 'items': list_of_item_names}.
    """
    safe_path = safe_file_operation(path)
    if not safe_path: # safe_file_operation returns None if path is invalid or outside cwd
        # Allow listing current directory if path is empty or "."
        if path == "" or path == ".":
             safe_path = Path.cwd().resolve()
        else:
            return OperationResult(False, error=f"Invalid or disallowed path for listing: '{path}'", status_code=400)

    if not safe_path.is_dir():
        logger.error(f"Cannot list directory: '{safe_path}' is not a directory or does not exist.")
        return OperationResult(False, error="Path is not a directory or does not exist.", status_code=404)

    try:
        items = os.listdir(safe_path)
        logger.info(f"Directory '{safe_path}' listed successfully. Found {len(items)} items.")
        return OperationResult(True, data={'path': str(safe_path), 'items': items})
    except Exception as e:
        logger.error(f"Error listing directory '{safe_path}': {str(e)}", exc_info=True)
        return OperationResult(False, error=f"Failed to list directory: {str(e)}", status_code=500)

def search_file(keyword: str, path: str = ".") -> OperationResult:
    """
    Searches for a keyword in files under the given path (defaults to current directory).
    The search is case-sensitive. It attempts to read text-based files.

    Args:
        keyword: The keyword string to search for.
        path: The directory path to search within. Defaults to ".".

    Returns:
        An OperationResult. On success, data contains 
        {'matches': list_of_relative_paths_to_matching_files, 'keyword': keyword, 'path': str(search_path)}.
    """
    safe_search_path = safe_file_operation(path)
    if not safe_search_path:
        if path == "" or path == ".":
             safe_search_path = Path.cwd().resolve()
        else:
            return OperationResult(False, error=f"Invalid or disallowed path for searching: '{path}'", status_code=400)

    if not safe_search_path.is_dir():
        return OperationResult(False, error="Search path is not a directory or does not exist.", status_code=404)

    matches: List[str] = []
    try:
        for root, _, files in os.walk(safe_search_path):
            for file_name in files:
                file_path_obj = Path(root) / file_name
                try:
                    # More robust check for text-based files, can be expanded
                    text_suffixes = ['.py', '.txt', '.md', '.json', '.html', '.css', '.js', '.yaml', '.yml', '.xml', '.csv', '.log', '.ini', '.cfg', '.toml']
                    if file_path_obj.suffix.lower() in text_suffixes:
                        with open(file_path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                            # Read in chunks for very large files? For now, f.read()
                            if keyword in f.read():
                                matches.append(str(file_path_obj.relative_to(Path.cwd())))
                except Exception as e_read:
                    logger.debug(f"Could not read or search in file {file_path_obj}: {e_read}")
                    continue 
        logger.info(f"Search for '{keyword}' in '{safe_search_path}' completed. Found {len(matches)} match(es).")
        return OperationResult(True, data={'matches': matches, 'keyword': keyword, 'path': str(safe_search_path)})
    except Exception as e_walk:
        logger.error(f"Error during search in '{safe_search_path}': {str(e_walk)}", exc_info=True)
        return OperationResult(False, error=f"File search failed: {str(e_walk)}", status_code=500)

def chunk_file(path: str, chunk_size: int = 100, chunk_index: int = 0) -> OperationResult:
    """
    Reads a file in line-based chunks.

    Args:
        path: The path to the file.
        chunk_size: The number of lines per chunk. Defaults to 100.
        chunk_index: The 0-based index of the chunk to retrieve. Defaults to 0.

    Returns:
        An OperationResult. On success, data contains:
        {
            'chunk_index': int, 
            'total_chunks': int, 
            'chunk_content': str, 
            'start_line': int (0-based), 
            'end_line': int (0-based, inclusive),
            'total_lines': int
        }
        Returns error if path is invalid, not a file, or if chunk_index is out of range.
    """
    safe_path = safe_file_operation(path)
    if not safe_path:
        return OperationResult(False, error="Invalid or disallowed path for chunk_file.", status_code=400)
    if not safe_path.is_file():
        return OperationResult(False, error="Path is not a file or does not exist.", status_code=404)

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        return OperationResult(False, error="chunk_size must be a positive integer.", status_code=400)
    if not isinstance(chunk_index, int) or chunk_index < 0:
        return OperationResult(False, error="chunk_index must be a non-negative integer.", status_code=400)

    try:
        total_lines = 0
        chunk_lines_content: List[str] = []
        start_line_num = chunk_index * chunk_size
        end_line_num = start_line_num + chunk_size

        with open(safe_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if start_line_num <= i < end_line_num:
                    chunk_lines_content.append(line)
                total_lines += 1
        
        if total_lines == 0: # Empty file
            total_chunks = 1 if chunk_index == 0 else 0 # Chunk 0 of empty file can be "valid" (empty content)
        else:
            total_chunks = (total_lines + chunk_size - 1) // chunk_size
        
        if chunk_index >= total_chunks and not (chunk_index == 0 and total_lines == 0) :
            logger.warning(f"chunk_index {chunk_index} is out of range for file '{safe_path}'. Total lines: {total_lines}, Total chunks: {total_chunks}.")
            return OperationResult(False, error='chunk_index out of range', data={'total_chunks': total_chunks, 'total_lines': total_lines}, status_code=416)

        chunk_str_content = ''.join(chunk_lines_content)
        actual_end_line = -1 # For empty chunk or file
        if chunk_lines_content: # If any lines were read for the chunk
            actual_end_line = start_line_num + len(chunk_lines_content) - 1
        elif total_lines == 0 and chunk_index == 0 : # Chunk 0 of empty file
             actual_end_line = -1 # Or start_line_num -1, depending on convention
        
        logger.info(f"Read chunk {chunk_index} (lines {start_line_num}-{actual_end_line}) from '{safe_path}' successfully.")
        return OperationResult(True, data={
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'chunk_content': chunk_str_content,
            'start_line': start_line_num if total_lines > 0 or chunk_index==0 else -1,
            'end_line': actual_end_line,
            'total_lines': total_lines
        })
    except Exception as e:
        logger.error(f"Error chunking file '{safe_path}': {str(e)}", exc_info=True)
        return OperationResult(False, error=f"Failed to chunk file: {str(e)}", status_code=500)

def update_file_chunk(path: str, chunk_content: str, chunk_index: int, chunk_size: int = 100) -> OperationResult:
    """
    Updates a specific line-based chunk of a file. If the chunk_index is beyond
    the current end of the file, it pads with newlines and appends the chunk.

    Args:
        path: The path to the file.
        chunk_content: The new content for the chunk (can be multi-line).
        chunk_index: The 0-based index of the chunk to update/append.
        chunk_size: The number of lines per chunk. Defaults to 100.

    Returns:
        An OperationResult. On success, data contains:
        {
            'path': str(absolute_file_path),
            'chunk_index': int,
            'total_chunks': int (recalculated after update),
            'total_lines': int (recalculated after update)
        }
    """
    safe_path = safe_file_operation(path)
    if not safe_path:
        return OperationResult(False, error="Invalid or disallowed path for update_file_chunk.", status_code=400)
    if safe_path.exists() and not safe_path.is_file(): # If it exists, must be a file
        return OperationResult(False, error="Path exists but is not a file.", status_code=400)

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        return OperationResult(False, error="chunk_size must be a positive integer.", status_code=400)
    if not isinstance(chunk_index, int) or chunk_index < 0:
        return OperationResult(False, error="chunk_index must be a non-negative integer.", status_code=400)

    temp_file_path = safe_path.with_suffix(safe_path.suffix + '.tmp')
    start_line_to_replace = chunk_index * chunk_size
    lines_in_new_chunk = chunk_content.splitlines(keepends=True) # Keep newlines for writing
    
    try:
        current_line_num = 0
        chunk_was_placed = False

        # Open source file (even if it's os.devnull if file doesn't exist yet)
        # and temporary destination file.
        with open(safe_path if safe_path.exists() else os.devnull, 'r', encoding='utf-8') as src_file, \
             open(temp_file_path, 'w', encoding='utf-8') as dest_file:

            # Iterate through existing lines if file exists
            for line in src_file:
                # If current line is where the new chunk starts
                if current_line_num == start_line_to_replace and not chunk_was_placed:
                    for new_line_content in lines_in_new_chunk:
                        dest_file.write(new_line_content)
                    chunk_was_placed = True
                
                # If current line is within the old chunk's span (and not yet replaced)
                if not (start_line_to_replace <= current_line_num < start_line_to_replace + chunk_size):
                    dest_file.write(line) # Write original line if outside replacement zone
                
                current_line_num += 1
            
            # If chunk_index is beyond the current end of the file (appending new chunk)
            if not chunk_was_placed:
                # Add padding lines if necessary to reach the start_line_to_replace
                while current_line_num < start_line_to_replace:
                    dest_file.write('\n')
                    current_line_num += 1
                # Write the new chunk content
                for new_line_content in lines_in_new_chunk:
                    dest_file.write(new_line_content)
                # current_line_num would be updated implicitly by the writes above if needed for further logic

        os.replace(temp_file_path, safe_path) # Atomically replace original with temp file
        
        # Recalculate total lines and chunks for the updated file for the response
        final_total_lines = 0
        if safe_path.exists():
            with open(safe_path, 'r', encoding='utf-8') as f:
                final_total_lines = sum(1 for _ in f)
        
        final_total_chunks = 1 if final_total_lines == 0 else (final_total_lines + chunk_size - 1) // chunk_size
        
        logger.info(f"Chunk {chunk_index} in file '{safe_path}' updated successfully.")
        return OperationResult(True, data={
            'path': str(safe_path),
            'chunk_index': chunk_index,
            'total_chunks': final_total_chunks,
            'total_lines': final_total_lines
        })
    except Exception as e:
        logger.error(f"Error updating chunk in file '{safe_path}': {str(e)}", exc_info=True)
        if temp_file_path.exists(): # Clean up temp file on error
            try:
                os.remove(temp_file_path)
            except OSError as e_remove:
                logger.error(f"Could not remove temporary file {temp_file_path}: {e_remove}")
        return OperationResult(False, error=f"Failed to update file chunk: {str(e)}", status_code=500)

def verify_file_content(file_path: str) -> OperationResult:
    """
    Verifies the content of a generated file for common issues like
    remaining code block markers or empty content.

    Args:
        file_path: The path to the file to verify.

    Returns:
        An OperationResult. Success is True if no issues are found.
        If issues are found, success is False, and error contains a description of the issues.
    """
    safe_path = safe_file_operation(file_path)
    if not safe_path:
        return OperationResult(False, error="Invalid or disallowed path for verify_file_content.", status_code=400)
    
    if not safe_path.is_file():
        return OperationResult(False, error="Path is not a file or does not exist.", status_code=404)

    try:
        with open(safe_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        issues = []
        if '```' in content: # Basic check for markdown code blocks
            issues.append("Found remaining code block markers (```)")
        # A more robust check for unclosed blocks might be too complex for here,
        # but this catches simple cases.
        if content.count('```') % 2 != 0:
            issues.append("Found an odd number of code block markers (```), suggesting an unclosed block")
            
        if not content.strip():
            issues.append("File is empty or contains only whitespace")
            
        if issues:
            error_message = ", ".join(issues)
            logger.warning(f"Verification issues in file '{safe_path}': {error_message}")
            return OperationResult(False, error=error_message, data={'issues': issues, 'path': str(safe_path)})
            
        logger.info(f"File '{safe_path}' verified successfully. No common issues found.")
        return OperationResult(True, message="File verified successfully.")
    except Exception as e:
        logger.error(f"Error verifying file content for '{safe_path}': {str(e)}", exc_info=True)
        return OperationResult(False, error=f"Failed to verify file content: {str(e)}", status_code=500)

# Example of how logger could be configured by the main application
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     logger.info("File utils module ready for testing.")
#
#     # Create a test directory
#     test_dir = Path("file_utils_test_dir")
#     if test_dir.exists():
#         shutil.rmtree(test_dir)
#     test_dir.mkdir()
#     os.chdir(test_dir)
#     logger.info(f"Current working directory: {Path.cwd()}")
#
#     # Test create_file
#     res_create = create_file("test.txt", "Hello\nWorld\nLine 3")
#     print(f"Create File: {res_create}")
#
#     if res_create.success:
#         # Test read_file
#         res_read = read_file("test.txt")
#         print(f"Read File: {res_read}")
#
#         # Test chunk_file
#         res_chunk1 = chunk_file("test.txt", chunk_size=1, chunk_index=0)
#         print(f"Chunk 1: {res_chunk1}")
#         res_chunk2 = chunk_file("test.txt", chunk_size=1, chunk_index=1)
#         print(f"Chunk 2: {res_chunk2}")
#         res_chunk_large = chunk_file("test.txt", chunk_size=10, chunk_index=0)
#         print(f"Large Chunk: {res_chunk_large}")
#         res_chunk_out = chunk_file("test.txt", chunk_size=1, chunk_index=5)
#         print(f"Out of bounds chunk: {res_chunk_out}")
#
#         # Test update_file_chunk
#         res_update_c0 = update_file_chunk("test.txt", "UPDATED LINE 0\n", chunk_index=0, chunk_size=1)
#         print(f"Update Chunk 0: {res_update_c0}")
#         res_read_after_upd0 = read_file("test.txt")
#         print(f"Read after update 0: {res_read_after_upd0.data.get('content') if res_read_after_upd0.success else 'Error'}")
#
#         res_update_c2_append = update_file_chunk("test.txt", "NEW LINE APPENDED\n", chunk_index=3, chunk_size=1) # Appending
#         print(f"Update Chunk 3 (Append): {res_update_c2_append}")
#         res_read_after_append = read_file("test.txt")
#         print(f"Read after append: {res_read_after_append.data.get('content') if res_read_after_append.success else 'Error'}")
#
#         # Test verify_file_content
#         res_verify_ok = verify_file_content("test.txt")
#         print(f"Verify OK: {res_verify_ok}")
#         create_file("test_issues.txt", "Hello ``` unclosed")
#         res_verify_issue = verify_file_content("test_issues.txt")
#         print(f"Verify Issue: {res_verify_issue}")
#
#     # Cleanup: Change back directory and remove test_dir
#     os.chdir("..")
#     shutil.rmtree(test_dir)
#     logger.info("Test cleanup complete.")
