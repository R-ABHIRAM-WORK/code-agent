import unittest
from unittest.mock import patch, MagicMock, call # Added call for checking multiple calls
import os
import sys
import platform
import subprocess # For CompletedProcess and Popen attributes

# Add parent directory to sys.path to allow importing env_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from env_utils import (
        run_script,
        start_interactive,
        install_package,
        open_in_browser,
        lint_code,
        format_code,
        run_tests,
        git_commit,
        git_push
    )
except ImportError:
    print("ImportError: Ensure env_utils.py is in the Python path or parent directory.")
    raise

# Basic logging setup for tests if needed, but usually keep test output clean.
# import logging
# logging.basicConfig(level=logging.DEBUG) 
# logger = logging.getLogger(__name__)

class TestEnvUtils(unittest.TestCase):
    """
    Test suite for environment and process utility functions in env_utils.py.
    """

    # --- Test run_script ---
    @patch('env_utils.subprocess.run')
    @patch('env_utils.os.path.exists', return_value=True) # Assume script path exists for these tests
    @patch('env_utils.os.path.abspath', side_effect=lambda x: '/abs/path/to/' + x) # Mock abspath
    def test_run_script_success(self, mock_abspath, mock_exists, mock_subprocess_run):
        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 0
        mock_result.stdout = "Script output"
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        result = run_script("test_script.py")
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['exit_code'], 0)
        self.assertEqual(result['stdout'], "Script output")
        self.assertEqual(result['stderr'], "")
        mock_subprocess_run.assert_called_once_with(
            [sys.executable, '/abs/path/to/test_script.py'],
            capture_output=True, text=True, check=False, timeout=300
        )

    @patch('env_utils.subprocess.run')
    @patch('env_utils.os.path.exists', return_value=True)
    @patch('env_utils.os.path.abspath', side_effect=lambda x: '/abs/path/to/' + x)
    def test_run_script_error_return_code(self, mock_abspath, mock_exists, mock_subprocess_run):
        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Script error"
        mock_subprocess_run.return_value = mock_result

        result = run_script("error_script.py")
        self.assertEqual(result['status'], 'completed') # Still 'completed' as subprocess ran
        self.assertEqual(result['exit_code'], 1)
        self.assertEqual(result['stderr'], "Script error")

    @patch('env_utils.subprocess.run', side_effect=subprocess.TimeoutExpired(cmd="cmd", timeout=300))
    @patch('env_utils.os.path.exists', return_value=True)
    @patch('env_utils.os.path.abspath', side_effect=lambda x: '/abs/path/to/' + x)
    def test_run_script_exception(self, mock_abspath, mock_exists, mock_subprocess_run):
        result = run_script("timeout_script.py")
        self.assertEqual(result['status'], 'error')
        self.assertIn("Script execution timed out", result['error'])

    @patch('env_utils.os.path.exists', return_value=False) # Script does not exist
    @patch('env_utils.os.path.abspath', side_effect=lambda x: '/abs/path/to/' + x)
    def test_run_script_not_found(self, mock_abspath, mock_exists):
        result = run_script("non_existent_script.py")
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['error'], 'Script not found')


    # --- Test start_interactive ---
    @patch('env_utils.subprocess.Popen')
    @patch('env_utils.os.path.exists', return_value=True)
    @patch('env_utils.os.path.abspath', side_effect=lambda x: '/abs/path/to/' + x)
    def test_start_interactive_success(self, mock_abspath, mock_exists, mock_subprocess_popen):
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 12345
        mock_subprocess_popen.return_value = mock_proc

        result = start_interactive("interactive_script.py")
        self.assertEqual(result['status'], 'started')
        self.assertEqual(result['pid'], 12345)
        
        expected_kwargs = {}
        if platform.system() == 'Windows' and hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
            expected_kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
        elif platform.system() in ['Linux', 'Darwin']:
            expected_kwargs['start_new_session'] = True
        
        mock_subprocess_popen.assert_called_once_with(
            [sys.executable, '/abs/path/to/interactive_script.py'], **expected_kwargs
        )

    @patch('env_utils.platform.system', return_value='Windows')
    @patch('env_utils.hasattr', return_value=True) # Mock hasattr for subprocess.CREATE_NEW_CONSOLE
    @patch('env_utils.subprocess.Popen')
    @patch('env_utils.os.path.exists', return_value=True)
    @patch('env_utils.os.path.abspath', side_effect=lambda x: '/abs/path/to/' + x)
    def test_start_interactive_windows(self, mock_abspath, mock_exists, mock_subprocess_popen, mock_hasattr, mock_platform_system):
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 54321
        mock_subprocess_popen.return_value = mock_proc

        result = start_interactive("win_script.py")
        self.assertEqual(result['status'], 'started')
        self.assertEqual(result['pid'], 54321)
        mock_subprocess_popen.assert_called_once_with(
            [sys.executable, '/abs/path/to/win_script.py'], creationflags=subprocess.CREATE_NEW_CONSOLE
        )

    @patch('env_utils.subprocess.Popen', side_effect=Exception("Popen failed"))
    @patch('env_utils.os.path.exists', return_value=True)
    @patch('env_utils.os.path.abspath', side_effect=lambda x: '/abs/path/to/' + x)
    def test_start_interactive_exception(self, mock_abspath, mock_exists, mock_subprocess_popen):
        result = start_interactive("fail_script.py")
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['error'], "Popen failed")


    # --- Test install_package ---
    @patch('env_utils.subprocess.run')
    def test_install_package_success(self, mock_subprocess_run):
        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 0
        mock_result.stdout = "Successfully installed package"
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        result = install_package("test_package")
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['stdout'], "Successfully installed package")
        mock_subprocess_run.assert_called_once_with(
            [sys.executable, '-m', 'pip', 'install', 'test_package'],
            capture_output=True, text=True, check=False, timeout=300
        )

    @patch('env_utils.subprocess.run')
    def test_install_package_failure(self, mock_subprocess_run):
        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Failed to install"
        mock_subprocess_run.return_value = mock_result

        result = install_package("bad_package")
        self.assertEqual(result['status'], 'error') # Changed to 'error' in implementation
        self.assertEqual(result['stderr'], "Failed to install")
    
    def test_install_package_invalid_name(self):
        result = install_package("invalid package name")
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['error'], 'Invalid package name provided')


    # --- Test open_in_browser ---
    @patch('env_utils.webbrowser.open', return_value=True)
    @patch('env_utils.os.path.abspath', side_effect=lambda x: '/abs/path/to/' + x)
    @patch('env_utils.os.path.exists', return_value=True)
    def test_open_in_browser_local_file(self, mock_exists, mock_abspath, mock_webbrowser_open):
        result = open_in_browser("local.html")
        self.assertEqual(result['status'], 'opened')
        self.assertEqual(result['url'], 'file:///abs/path/to/local.html')
        mock_webbrowser_open.assert_called_once_with('file:///abs/path/to/local.html')

    @patch('env_utils.webbrowser.open', return_value=True)
    def test_open_in_browser_url(self, mock_webbrowser_open):
        url = "http://example.com"
        result = open_in_browser(url)
        self.assertEqual(result['status'], 'opened')
        self.assertEqual(result['url'], url)
        mock_webbrowser_open.assert_called_once_with(url)

    @patch('env_utils.webbrowser.open', side_effect=Exception("Browser error"))
    def test_open_in_browser_exception(self, mock_webbrowser_open):
        result = open_in_browser("http://example.com")
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['error'], "Browser error")


    # --- Test lint_code ---
    @patch('env_utils.subprocess.run')
    @patch('env_utils.os.path.exists', return_value=True)
    @patch('env_utils.os.path.abspath', side_effect=lambda x: x) # Simple mock for abspath
    def test_lint_code_success_no_issues(self, mock_abspath, mock_exists, mock_subprocess_run):
        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 0
        mock_result.stdout = "" # No output means no issues for flake8
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        result = lint_code("clean_code.py")
        self.assertEqual(result['status'], 'completed_no_issues')
        mock_subprocess_run.assert_called_once_with(
            ['flake8', 'clean_code.py'], capture_output=True, text=True, check=False
        )

    @patch('env_utils.subprocess.run')
    @patch('env_utils.os.path.exists', return_value=True)
    @patch('env_utils.os.path.abspath', side_effect=lambda x: x)
    def test_lint_code_with_issues(self, mock_abspath, mock_exists, mock_subprocess_run):
        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 1 # flake8 returns 1 if issues found
        mock_result.stdout = "file.py:1:1: F821 undefined name 'x'"
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        result = lint_code("issues_code.py")
        self.assertEqual(result['status'], 'completed_with_issues')
        self.assertIn("F821", result['stdout'])


    # --- Test format_code ---
    @patch('env_utils.subprocess.run')
    @patch('env_utils.os.path.exists', return_value=True)
    @patch('env_utils.os.path.abspath', side_effect=lambda x: x)
    def test_format_code_success_no_changes(self, mock_abspath, mock_exists, mock_subprocess_run):
        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 0
        mock_result.stdout = "All files unchanged."
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        result = format_code("formatted_code.py")
        self.assertEqual(result['status'], 'formatted')
        self.assertIn("unchanged", result['stdout'])
        mock_subprocess_run.assert_called_once_with(
            ['black', 'formatted_code.py'], capture_output=True, text=True, check=False
        )

    @patch('env_utils.subprocess.run')
    @patch('env_utils.os.path.exists', return_value=True)
    @patch('env_utils.os.path.abspath', side_effect=lambda x: x)
    def test_format_code_with_changes(self, mock_abspath, mock_exists, mock_subprocess_run):
        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 0 # black returns 0 even if files changed
        mock_result.stdout = "1 file reformatted."
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        result = format_code("unformatted_code.py")
        self.assertEqual(result['status'], 'formatted')
        self.assertIn("reformatted", result['stdout'])


    # --- Test run_tests ---
    @patch('env_utils.subprocess.run')
    @patch('env_utils.os.path.abspath', side_effect=lambda x: x)
    def test_run_tests_success(self, mock_abspath, mock_subprocess_run):
        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 0 # pytest returns 0 for all tests passed
        mock_result.stdout = "All tests passed"
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        result = run_tests("tests_dir/")
        self.assertEqual(result['status'], 'tests_passed')
        mock_subprocess_run.assert_called_once_with(
            ['pytest', 'tests_dir/'], capture_output=True, text=True, check=False
        )

    @patch('env_utils.subprocess.run')
    @patch('env_utils.os.path.abspath', side_effect=lambda x: x)
    def test_run_tests_with_failures(self, mock_abspath, mock_subprocess_run):
        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 1 # pytest returns 1 for failed tests
        mock_result.stdout = "1 test failed"
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        result = run_tests("failing_tests_dir/")
        self.assertEqual(result['status'], 'tests_failed')
        self.assertIn("1 test failed", result['stdout'])


    # --- Test git_commit ---
    @patch('env_utils.subprocess.run')
    def test_git_commit_success(self, mock_subprocess_run):
        # Mock for 'git add .'
        mock_add_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_add_result.returncode = 0
        mock_add_result.stdout = ""
        mock_add_result.stderr = ""

        # Mock for 'git commit -m ...'
        mock_commit_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_commit_result.returncode = 0
        mock_commit_result.stdout = "[main 1234567] Test commit"
        mock_commit_result.stderr = ""
        
        mock_subprocess_run.side_effect = [mock_add_result, mock_commit_result]

        result = git_commit("Test commit")
        self.assertEqual(result['status'], 'committed')
        self.assertIn("Test commit", result['stdout'])
        
        expected_calls = [
            call(['git', 'add', '.'], check=False, capture_output=True, text=True), # Updated in my env_utils to check=False
            call(['git', 'commit', '-m', 'Test commit'], capture_output=True, text=True, check=False)
        ]
        mock_subprocess_run.assert_has_calls(expected_calls)

    @patch('env_utils.subprocess.run')
    def test_git_commit_failure_on_add(self, mock_subprocess_run):
        mock_add_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_add_result.returncode = 1
        mock_add_result.stderr = "Error on git add"
        mock_subprocess_run.return_value = mock_add_result # Only mock 'git add'

        result = git_commit("Test commit fail add")
        self.assertEqual(result['status'], 'error_staging') # As per updated env_utils
        self.assertIn("Error on git add", result['error'])


    # --- Test git_push ---
    @patch('env_utils.subprocess.run')
    def test_git_push_success_specific_branch(self, mock_subprocess_run):
        mock_push_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_push_result.returncode = 0
        mock_push_result.stdout = "Everything up-to-date" # Or successful push message
        mock_push_result.stderr = ""
        mock_subprocess_run.return_value = mock_push_result

        result = git_push("origin", "main")
        # Status can be 'pushed' or 'up_to_date'
        self.assertIn(result['status'], ['pushed', 'up_to_date']) 
        mock_subprocess_run.assert_called_once_with(
            ['git', 'push', 'origin', 'main'], capture_output=True, text=True, check=False
        )

    @patch('env_utils.subprocess.run')
    def test_git_push_success_current_branch(self, mock_subprocess_run):
        # Mock for 'git branch --show-current'
        mock_branch_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_branch_result.returncode = 0
        mock_branch_result.stdout = "current_branch\n" # Git outputs newline
        
        # Mock for 'git push origin current_branch'
        mock_push_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_push_result.returncode = 0
        mock_push_result.stdout = "Pushed successfully"
        mock_push_result.stderr = ""
        
        mock_subprocess_run.side_effect = [mock_branch_result, mock_push_result]

        result = git_push("origin") # No branch specified
        self.assertEqual(result['status'], 'pushed')
        
        expected_calls = [
            call(['git', 'branch', '--show-current'], capture_output=True, text=True, check=True),
            call(['git', 'push', 'origin', 'current_branch'], capture_output=True, text=True, check=False)
        ]
        mock_subprocess_run.assert_has_calls(expected_calls)

    @patch('env_utils.subprocess.run')
    def test_git_push_failure(self, mock_subprocess_run):
        mock_branch_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_branch_result.returncode = 0
        mock_branch_result.stdout = "main\n"
        
        mock_push_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_push_result.returncode = 1
        mock_push_result.stderr = "Push failed"
        mock_push_result.stdout = ""
        
        mock_subprocess_run.side_effect = [mock_branch_result, mock_push_result]

        result = git_push("origin", "main")
        self.assertEqual(result['status'], 'error_pushing') # As per updated env_utils
        self.assertIn("Push failed", result['error'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
