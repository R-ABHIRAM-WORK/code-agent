import unittest
import os
import sys
from pathlib import Path
import shutil
import logging

# Add the parent directory to sys.path to allow importing file_utils
# This assumes the tests directory is a subdirectory of the project root
# and file_utils.py is in the project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Attempt to import functions from file_utils
try:
    from file_utils import (
        OperationResult,
        safe_file_operation,
        create_file,
        read_file,
        update_file,
        delete_file,
        rename_file,
        move_file,
        list_directory,
        search_file,
        chunk_file,
        update_file_chunk,
        verify_file_content
    )
except ImportError as e:
    # Fallback if the above path adjustment doesn't work in all environments (e.g. specific CI)
    # This is a common issue in test setups.
    print(f"ImportError: {e}. Ensure file_utils.py is in the Python path.")
    # If running this test file directly, and file_utils.py is in parent dir:
    # import sys
    # sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    # from file_utils import ...
    raise  # Re-raise the error to make it clear if imports fail

# Configure logging for tests (optional, but can be helpful)
# Keep it minimal to avoid cluttering test output unless debugging.
logging.basicConfig(level=logging.ERROR) 
# logging.getLogger('file_utils').setLevel(logging.DEBUG) # For debugging file_utils specifically


class TestFileUtils(unittest.TestCase):
    """
    Test suite for file utility functions in file_utils.py.
    """
    TEST_DIR_NAME = "test_temp_dir_file_utils"
    original_cwd = None

    @classmethod
    def setUpClass(cls):
        """Set up resources for the entire test class."""
        cls.original_cwd = Path.cwd()
        # Create a temporary directory for test files at the original CWD
        # This ensures that safe_file_operation (which uses Path.cwd()) behaves as expected
        # relative to the project root, not the tests/ directory if tests are run from there.
        cls.test_dir_path = cls.original_cwd / cls.TEST_DIR_NAME
        
        if cls.test_dir_path.exists():
            shutil.rmtree(cls.test_dir_path)
        cls.test_dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests in the class have run."""
        if cls.test_dir_path.exists():
            shutil.rmtree(cls.test_dir_path)
        # No need to chdir back as each test method handles its CWD

    def setUp(self):
        """Set up environment for each test method."""
        # Change CWD to the test directory so file operations are contained
        # and safe_file_operation works relative to this test dir.
        os.chdir(self.test_dir_path)
        # Create a fresh subdirectory for each test to ensure isolation
        self.current_test_subdir_name = self.id().split('.')[-1] + "_data" # e.g., test_create_new_file_data
        self.current_test_subdir = Path(self.current_test_subdir_name)
        self.current_test_subdir.mkdir(exist_ok=True)
        os.chdir(self.current_test_subdir)


    def tearDown(self):
        """Clean up environment after each test method."""
        os.chdir(self.test_dir_path) # Move out of test-specific subdir
        if self.current_test_subdir.exists(): # Redundant if test_dir_path is removed by tearDownClass
             shutil.rmtree(self.current_test_subdir) # But good for isolating test method effects
        os.chdir(self.original_cwd) # Change back to original CWD

    # --- Test OperationResult ---
    def test_operation_result_instantiation(self):
        """Test basic instantiation of OperationResult."""
        res_success = OperationResult(success=True, message="Done", data=[1, 2], status_code=200)
        self.assertTrue(res_success.success)
        self.assertEqual(res_success.message, "Done")
        self.assertEqual(res_success.data, [1, 2])
        self.assertEqual(res_success.status_code, 200)
        self.assertIsNone(res_success.error)

        res_failure = OperationResult(success=False, error="Failed", status_code=500)
        self.assertFalse(res_failure.success)
        self.assertEqual(res_failure.error, "Failed")
        self.assertEqual(res_failure.status_code, 500)
        self.assertEqual(res_failure.message, "") # Default message
        self.assertIsNone(res_failure.data)


    # --- Test safe_file_operation ---
    def test_safe_file_operation_valid(self):
        """Test safe_file_operation with valid paths within the CWD."""
        # We are already chdired into self.test_dir_path / self.current_test_subdir by setUp
        valid_path_str = "test_file.txt"
        (Path.cwd() / valid_path_str).touch() # Create the file
        
        resolved_path = safe_file_operation(valid_path_str)
        self.assertIsNotNone(resolved_path)
        self.assertTrue(resolved_path.exists())
        self.assertTrue(str(resolved_path).endswith(valid_path_str))

        subdir_path_str = "subdir/another.txt"
        (Path.cwd() / "subdir").mkdir()
        (Path.cwd() / subdir_path_str).touch()
        resolved_subdir_path = safe_file_operation(subdir_path_str)
        self.assertIsNotNone(resolved_subdir_path)
        self.assertTrue(resolved_subdir_path.exists())

    def test_safe_file_operation_outside_cwd(self):
        """Test safe_file_operation with paths outside the CWD (should fail)."""
        # Path.cwd() is currently self.test_dir_path / self.current_test_subdir
        # We need to construct a path that is truly outside the initial project root (original_cwd)
        # For this test to be meaningful with the current safe_file_operation logic,
        # it assumes safe_file_operation checks against Path.cwd() at the time of the call.
        # Our setUp changes cwd into a temp dir. So, any path outside this temp dir is "outside".
        
        # Go up one level from current test CWD to the main test_temp_dir
        path_outside_current_test_subdir = "../some_other_file.txt"
        # This path is still within the broader test_temp_dir, so safe_file_operation should resolve it.
        # To test truly "outside", we'd need to reference something in original_cwd's parent.
        # However, safe_file_operation's check is `base_path in safe_path.parents` where base_path is CWD.
        
        # Path relative to the test_temp_dir_file_utils, but outside the current CWD (test method's subdir)
        # This should still be resolved by safe_file_operation as it's within the broader test setup.
        # (Path.cwd().parent / "sibling_file.txt").touch() # Create it in test_temp_dir
        # resolved_path = safe_file_operation("../sibling_file.txt")
        # self.assertIsNotNone(resolved_path) # This should pass as it's still within the test_dir_path

        # To test a path truly outside the main test_dir_path (which is inside original_cwd):
        # This is tricky because on some systems /tmp might be on a different device or symlinked.
        # A more reliable way to test "outside" is to try to access system files.
        # However, this can be platform-dependent and might require permissions.
        
        # The current implementation of safe_file_operation makes "outside" relative to current CWD.
        # So, "../" from within the test method's dedicated subdir is one level up.
        # If we go up enough levels to exit the initial `self.original_cwd / self.TEST_DIR_NAME`
        # then it should fail.
        
        # CWD is original_cwd / TEST_DIR_NAME / current_test_subdir_name
        # ../../ is original_cwd
        # ../../../ is original_cwd.parent
        
        # Let's assume the original_cwd is not the root of the filesystem.
        path_truly_outside_project = "../../../outside_project_file.txt"
        # This path is relative to current CWD (test_dir/test_method_data)
        # This path attempts to go three levels up.
        # Level 1 up: test_dir
        # Level 2 up: original_cwd (where test_dir resides)
        # Level 3 up: parent of original_cwd

        # The safe_file_operation is called with CWD being .../test_temp_dir_file_utils/test_method_data
        # If we pass "../../../some_os_file" it tries to resolve relative to this CWD.
        # The check `Path.cwd() in resolved_path.parents` will fail if resolved_path is outside.
        self.assertIsNone(safe_file_operation(path_truly_outside_project))
        if sys.platform != "win32": # /etc/hosts is a common outside-project path
            self.assertIsNone(safe_file_operation("/etc/hosts"))
        else: # C:\Windows is a common outside-project path on Windows
            self.assertIsNone(safe_file_operation("C:/Windows/System32/drivers/etc/hosts"))


    def test_safe_file_operation_invalid_chars(self):
        """Test safe_file_operation with invalid path characters (should cause error)."""
        # Behavior depends on OS and underlying filesystem.
        # Python's Path object might raise ValueError before safe_file_operation is deeply involved.
        # safe_file_operation catches ValueError and RuntimeError.
        if sys.platform == "win32":
            invalid_path = "file_with<invalid>char.txt" # Invalid on Windows
        else:
            invalid_path = "file_with\0nullchar.txt" # Null character is generally invalid
        
        # This should result in None due to exception during Path(invalid_path).resolve()
        self.assertIsNone(safe_file_operation(invalid_path))


    # --- Test create_file ---
    def test_create_new_file(self):
        file_path = "new_file.txt"
        content = "Hello, world!"
        result = create_file(file_path, content)
        self.assertTrue(result.success)
        self.assertTrue(Path(file_path).exists())
        self.assertEqual(Path(file_path).read_text(), content)
        self.assertIn('path', result.data)
        self.assertTrue(Path(result.data['path']).name == file_path)

    def test_create_file_overwrite(self):
        file_path = "overwrite_me.txt"
        Path(file_path).write_text("Initial content.")
        new_content = "Overwritten!"
        result = create_file(file_path, new_content)
        self.assertTrue(result.success)
        self.assertEqual(Path(file_path).read_text(), new_content)

    def test_create_file_in_new_subdir(self):
        file_path = "new_subdir/another_file.txt"
        content = "Content in subdir."
        result = create_file(file_path, content)
        self.assertTrue(result.success)
        self.assertTrue(Path(file_path).exists())
        self.assertEqual(Path(file_path).read_text(), content)
        self.assertTrue(Path("new_subdir").is_dir())
    
    def test_create_file_invalid_path_name(self):
        # Attempting to create a file with a name that's actually a directory
        dir_path = "a_directory"
        Path(dir_path).mkdir()
        result = create_file(dir_path, "content") # Trying to write content to a dir path
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        # Exact error message can be OS-dependent (e.g., IsADirectoryError on POSIX)
        # For now, just check that it failed and reported an error.
        self.assertIn("Failed to create file", result.error)


    # --- Test read_file ---
    def test_read_existing_file(self):
        file_path = "readable_file.txt"
        content = "Content to be read."
        Path(file_path).write_text(content)
        result = read_file(file_path)
        self.assertTrue(result.success)
        self.assertEqual(result.data['content'], content)

    def test_read_non_existent_file(self):
        result = read_file("non_existent_file.txt")
        self.assertFalse(result.success)
        self.assertIn("File not found", result.error)
        self.assertEqual(result.status_code, 404)

    # --- Test update_file (as alias of create_file) ---
    def test_update_file_behaves_like_create(self):
        file_path = "update_test.txt"
        content = "Initial for update."
        # Create it first (optional, as update_file should create if not exists)
        create_file(file_path, content) 
        
        new_content = "Updated content!"
        result = update_file(file_path, new_content)
        self.assertTrue(result.success)
        self.assertEqual(Path(file_path).read_text(), new_content)

        # Test creating a new file with update_file
        new_file_path = "update_creates_new.txt"
        result_new = update_file(new_file_path, new_content)
        self.assertTrue(result_new.success)
        self.assertTrue(Path(new_file_path).exists())
        self.assertEqual(Path(new_file_path).read_text(), new_content)


    # --- Test delete_file ---
    def test_delete_existing_file(self):
        file_path = "to_delete.txt"
        Path(file_path).touch()
        result = delete_file(file_path)
        self.assertTrue(result.success)
        self.assertFalse(Path(file_path).exists())

    def test_delete_existing_directory(self):
        dir_path = Path("dir_to_delete")
        dir_path.mkdir()
        (dir_path / "some_file.txt").touch()
        result = delete_file(str(dir_path)) # delete_file takes string path
        self.assertTrue(result.success)
        self.assertFalse(dir_path.exists())

    def test_delete_non_existent_path(self):
        result = delete_file("nothing_here_to_delete.txt")
        self.assertFalse(result.success)
        self.assertIn("Path not found", result.error)


    # --- Test rename_file ---
    def test_rename_file_simple(self):
        old_name = "original_name.txt"
        new_name = "renamed_file.txt"
        Path(old_name).write_text("data")
        result = rename_file(old_name, new_name)
        self.assertTrue(result.success, msg=result.error)
        self.assertFalse(Path(old_name).exists())
        self.assertTrue(Path(new_name).exists())
        self.assertEqual(Path(new_name).read_text(), "data")

    def test_rename_file_to_new_subdir(self):
        old_name = "move_me.txt"
        new_name = "renamed_subdir/moved_file.txt"
        Path(old_name).write_text("moving data")
        result = rename_file(old_name, new_name)
        self.assertTrue(result.success, msg=result.error)
        self.assertFalse(Path(old_name).exists())
        self.assertTrue(Path(new_name).exists())
        self.assertEqual(Path(new_name).read_text(), "moving data")
        self.assertTrue(Path("renamed_subdir").is_dir())
    
    def test_rename_directory(self):
        old_dir_name = "old_dir"
        new_dir_name = "new_renamed_dir"
        Path(old_dir_name).mkdir()
        (Path(old_dir_name) / "file.txt").touch()
        
        result = rename_file(old_dir_name, new_dir_name)
        self.assertTrue(result.success, msg=result.error)
        self.assertFalse(Path(old_dir_name).exists())
        self.assertTrue(Path(new_dir_name).is_dir())
        self.assertTrue((Path(new_dir_name) / "file.txt").exists())


    # --- Test move_file ---
    def test_move_file_to_existing_subdir(self):
        file_name = "file_to_move.txt"
        subdir_name = "destination_dir"
        Path(file_name).write_text("content of moved file")
        Path(subdir_name).mkdir()
        
        result = move_file(file_name, str(Path(subdir_name) / file_name))
        self.assertTrue(result.success, msg=result.error)
        self.assertFalse(Path(file_name).exists())
        self.assertTrue((Path(subdir_name) / file_name).exists())

    def test_move_file_to_new_subdir_structure(self):
        file_name = "another_file_to_move.txt"
        new_dest = "new_parent/new_child_dest/" + file_name
        Path(file_name).write_text("complex move")

        result = move_file(file_name, new_dest)
        self.assertTrue(result.success, msg=result.error)
        self.assertFalse(Path(file_name).exists())
        self.assertTrue(Path(new_dest).exists())
        self.assertEqual(Path(new_dest).read_text(), "complex move")

    def test_move_directory(self):
        src_dir = Path("source_folder_to_move")
        src_dir.mkdir()
        (src_dir / "data.txt").write_text("test data")
        
        dest_dir_parent = Path("target_parent_for_move")
        dest_dir_parent.mkdir() # Ensure parent of destination exists
        
        result = move_file(str(src_dir), str(dest_dir_parent / src_dir.name))
        self.assertTrue(result.success, msg=result.error)
        self.assertFalse(src_dir.exists())
        self.assertTrue((dest_dir_parent / src_dir.name).is_dir())
        self.assertTrue((dest_dir_parent / src_dir.name / "data.txt").exists())


    # --- Test list_directory ---
    def test_list_directory_with_items(self):
        Path("item1.txt").touch()
        Path("item2.py").touch()
        Path("sub_dir_list").mkdir()
        result = list_directory(".") # List current test method's subdir
        self.assertTrue(result.success)
        self.assertIn("items", result.data)
        self.assertIsInstance(result.data["items"], list)
        # Order is not guaranteed by os.listdir, so use sets
        self.assertSetEqual(set(result.data["items"]), {"item1.txt", "item2.py", "sub_dir_list"})

    def test_list_empty_directory(self):
        empty_dir = Path("empty_listing_dir")
        empty_dir.mkdir()
        result = list_directory(str(empty_dir))
        self.assertTrue(result.success)
        self.assertEqual(result.data["items"], [])

    def test_list_non_existent_directory(self):
        result = list_directory("no_such_dir_to_list")
        self.assertFalse(result.success)
        self.assertIn("Path is not a directory or does not exist", result.error)


    # --- Test search_file ---
    def test_search_file_keyword_exists(self):
        Path("search_target1.txt").write_text("Hello world, this is a test.")
        Path("search_target2.txt").write_text("Another file with the keyword test.")
        Path("no_keyword.txt").write_text("Nothing relevant here.")
        
        result = search_file("test", ".") # Search in current test method's subdir
        self.assertTrue(result.success)
        self.assertIn("matches", result.data)
        # search_file returns paths relative to CWD at the time of call.
        # CWD for the test is test_temp_dir/current_test_subdir.
        # So paths are relative to this subdir.
        expected_matches = sorted([str(Path("search_target1.txt")), str(Path("search_target2.txt"))])
        self.assertListEqual(sorted(result.data["matches"]), expected_matches)


    def test_search_file_keyword_not_exists(self):
        Path("search_no_keyword.txt").write_text("Some random content.")
        result = search_file("nonexistentkeyword", ".")
        self.assertTrue(result.success)
        self.assertEqual(result.data["matches"], [])

    def test_search_file_in_subdirs(self):
        subdir = Path("search_subdir")
        subdir.mkdir()
        (subdir / "sub_target.txt").write_text("Keyword is deep inside.")
        Path("top_level_no_keyword.txt").write_text("...")

        result = search_file("Keyword", ".")
        self.assertTrue(result.success)
        expected_path = str(Path(subdir.name) / "sub_target.txt") # Relative path
        self.assertIn(expected_path, result.data["matches"])
        self.assertEqual(len(result.data["matches"]), 1)


    # --- Test chunk_file ---
    def test_chunk_file_basic(self):
        file_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        Path("chunk_test.txt").write_text(file_content)
        
        # Read first chunk (2 lines)
        result_c0 = chunk_file("chunk_test.txt", chunk_size=2, chunk_index=0)
        self.assertTrue(result_c0.success, msg=result_c0.error)
        self.assertEqual(result_c0.data['chunk_content'], "Line 1\nLine 2\n")
        self.assertEqual(result_c0.data['total_chunks'], 3)
        self.assertEqual(result_c0.data['total_lines'], 5)
        
        # Read second chunk
        result_c1 = chunk_file("chunk_test.txt", chunk_size=2, chunk_index=1)
        self.assertTrue(result_c1.success)
        self.assertEqual(result_c1.data['chunk_content'], "Line 3\nLine 4\n")
        
        # Read last chunk (1 line)
        result_c2 = chunk_file("chunk_test.txt", chunk_size=2, chunk_index=2)
        self.assertTrue(result_c2.success)
        self.assertEqual(result_c2.data['chunk_content'], "Line 5")

    def test_chunk_file_empty_file(self):
        Path("empty_chunk.txt").touch()
        result = chunk_file("empty_chunk.txt", chunk_size=10, chunk_index=0)
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data['chunk_content'], "")
        self.assertEqual(result.data['total_chunks'], 1) # Chunk 0 of empty file is valid
        self.assertEqual(result.data['total_lines'], 0)

    def test_chunk_file_smaller_than_chunk_size(self):
        Path("small_chunk.txt").write_text("Line A\nLine B")
        result = chunk_file("small_chunk.txt", chunk_size=10, chunk_index=0)
        self.assertTrue(result.success)
        self.assertEqual(result.data['chunk_content'], "Line A\nLine B")
        self.assertEqual(result.data['total_chunks'], 1)
        self.assertEqual(result.data['total_lines'], 2)

    def test_chunk_file_index_out_of_bounds(self):
        Path("bound_chunk.txt").write_text("L1\nL2")
        result = chunk_file("bound_chunk.txt", chunk_size=1, chunk_index=5)
        self.assertFalse(result.success)
        self.assertIn("chunk_index out of range", result.error)


    # --- Test update_file_chunk ---
    def test_update_file_chunk_existing_middle(self):
        file_path = "update_chunk_test.txt"
        initial_content = "Line Zero\nLine One\nLine Two\nLine Three\nLine Four"
        Path(file_path).write_text(initial_content)
        
        update_content = "UPDATED LINE 1\nUPDATED LINE 2\n"
        result = update_file_chunk(file_path, update_content, chunk_index=1, chunk_size=1)
        # This means chunk 1 (which starts at line 1) of size 1 (so just line 1) will be replaced
        # by "UPDATED LINE 1\nUPDATED LINE 2\n".
        # Line 0: Line Zero
        # Line 1: UPDATED LINE 1
        # Line 2: UPDATED LINE 2
        # Line 3: Line Two  (original line 2, shifted)
        # Line 4: Line Three (original line 3, shifted)
        # Line 5: Line Four (original line 4, shifted)
        
        self.assertTrue(result.success, msg=result.error)
        expected_content = "Line Zero\nUPDATED LINE 1\nUPDATED LINE 2\nLine Two\nLine Three\nLine Four"
        self.assertEqual(Path(file_path).read_text(), expected_content)
        self.assertEqual(result.data['total_lines'], 6) # 1 original + 2 new + 3 original = 6

    def test_update_file_chunk_append(self):
        file_path = "update_chunk_append.txt"
        Path(file_path).write_text("First line\nSecond line") # 2 lines
        
        # Append by targeting chunk_index = 2 (for chunk_size=1) or chunk_index=1 (for chunk_size=2)
        # If chunk_size = 1, chunk_index=2 means start writing at line 2
        append_content = "Appended Line\n"
        result = update_file_chunk(file_path, append_content, chunk_index=2, chunk_size=1)
        self.assertTrue(result.success, msg=result.error)
        expected = "First line\nSecond line\nAppended Line\n"
        self.assertEqual(Path(file_path).read_text(), expected)

    def test_update_file_chunk_empty_file(self):
        file_path = "update_chunk_empty.txt"
        Path(file_path).touch() # Create empty file
        
        content = "New content for empty file\n"
        result = update_file_chunk(file_path, content, chunk_index=0, chunk_size=10)
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(Path(file_path).read_text(), content)


    # --- Test verify_file_content ---
    def test_verify_content_with_markers(self):
        file_path = "verify_markers.txt"
        Path(file_path).write_text("```python\nprint('hello')\n```")
        result = verify_file_content(file_path)
        self.assertFalse(result.success)
        self.assertIn("Found remaining code block markers", result.error)

    def test_verify_content_empty_file(self):
        file_path = "verify_empty.txt"
        Path(file_path).touch()
        result = verify_file_content(file_path)
        self.assertFalse(result.success) # Empty file is an "issue"
        self.assertIn("File is empty", result.error)

    def test_verify_content_valid_file(self):
        file_path = "verify_valid.txt"
        Path(file_path).write_text("This is perfectly valid content.")
        result = verify_file_content(file_path)
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.message, "File verified successfully.")


if __name__ == '__main__':
    # This allows running the tests directly from the command line
    # Change CWD to project root so that imports and safe_file_operation work as expected
    original_cwd_for_runner = os.getcwd()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)
    print(f"Changed CWD to project root: {project_root} for test runner")
    
    # Ensure logging is more verbose if run directly for debugging
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logging.getLogger('file_utils').setLevel(logging.DEBUG)
    
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestFileUtils))
    # You could add more selective tests here if needed:
    # suite.addTest(TestFileUtils('test_safe_file_operation_valid'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Change back to the original CWD after tests are done
    os.chdir(original_cwd_for_runner)

    # Exit with a non-zero code if tests failed
    if not result.wasSuccessful():
        sys.exit(1)
