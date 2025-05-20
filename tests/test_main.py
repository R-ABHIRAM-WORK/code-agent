import unittest
from unittest.mock import patch, mock_open, MagicMock, call
import os
import shutil
from pathlib import Path
import sys
import subprocess # For CompletedProcess
import webbrowser # For webbrowser.Error

# Add the parent directory to sys.path to allow importing main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import (
    FileSystemTools,
    ExecutionTools, 
    BrowserTools,   
    GitTools,       
    UserInteractionTools, 
    Agent,          
    OperationResult,
    AgentError,
    FilePathError,
    OperationFailedError,
    GeminiClientError, 
    strip_code_block_markers,
    parse_python_traceback,
)

# Helper to simulate FileSystemTools.safe_file_operation for other tool tests
def mock_safe_file_operation(path_str_arg):
    # This is a simplified mock. It assumes the path is valid and returns a mock Path object.
    # For more complex path validation needs in tool tests, this might need adjustment.
    mock_path = MagicMock(spec=Path)
    # Resolve to an absolute-like path for consistency in what tools might expect
    mock_path.resolve.return_value = Path("/test_base_dir") / path_str_arg 
    mock_path.exists.return_value = True # Assume path exists for most tool tests
    mock_path.is_file.return_value = True # Assume it's a file unless test specifies otherwise
    mock_path.is_dir.return_value = False
    mock_path.__str__.return_value = str(Path("/test_base_dir") / path_str_arg)
    return mock_path

class TestFileSystemTools(unittest.TestCase):
    def setUp(self):
        self.fs_tools = FileSystemTools()
        self.mock_project_root = Path.cwd() / "temp_test_project_root_for_agent_tests"
        self.mock_project_root.mkdir(exist_ok=True)

        self.cwd_patcher = patch('main.Path.cwd') 
        self.mock_cwd = self.cwd_patcher.start()
        self.mock_cwd.return_value = self.mock_project_root

    def tearDown(self):
        self.cwd_patcher.stop()
        if self.mock_project_root.exists():
            shutil.rmtree(self.mock_project_root)

    def test_safe_file_operation_valid_path_within_cwd(self):
        test_path_str = "some_file.txt"
        expected_resolved_path = self.mock_project_root / test_path_str
        mock_path_instance = MagicMock(spec=Path)
        mock_path_instance.resolve.return_value = expected_resolved_path
        
        with patch('main.Path', return_value=mock_path_instance) as mock_path_constructor:
            mock_path_constructor.cwd = self.mock_cwd
            result_path = self.fs_tools.safe_file_operation(test_path_str)
            self.assertEqual(result_path, expected_resolved_path)
            mock_path_constructor.assert_any_call(test_path_str)
            mock_path_instance.resolve.assert_called_once()

    def test_safe_file_operation_path_is_cwd_itself(self):
        test_path_str = str(self.mock_project_root)
        expected_resolved_path = self.mock_project_root
        mock_path_instance = MagicMock(spec=Path)
        mock_path_instance.resolve.return_value = expected_resolved_path
        with patch('main.Path', return_value=mock_path_instance) as mock_path_constructor:
            mock_path_constructor.cwd = self.mock_cwd
            result_path = self.fs_tools.safe_file_operation(test_path_str)
            self.assertEqual(result_path, expected_resolved_path)

    def test_safe_file_operation_path_outside_cwd_raises_filepatherror(self):
        outside_path_str = str(self.mock_project_root.parent / "outside_file.txt")
        with self.assertRaises(FilePathError):
            self.fs_tools.safe_file_operation(outside_path_str)
            
    def test_safe_file_operation_non_existent_component_raises_filepatherror(self):
        with patch.object(Path, 'resolve', side_effect=FileNotFoundError("No such dir from resolve")):
            with self.assertRaises(FilePathError) as context:
                self.fs_tools.safe_file_operation("non_existent_dir/file.txt")
            self.assertIn("File or directory component not found", str(context.exception))

    def test_safe_file_operation_malformed_path_valueerror_raises_filepatherror(self):
        with patch.object(Path, 'resolve', side_effect=ValueError("Malformed path with null byte")):
            with self.assertRaises(FilePathError) as context:
                self.fs_tools.safe_file_operation("a\0b")
            self.assertIn("Invalid or malformed path", str(context.exception))

    @patch.object(Path, 'mkdir')
    def test_ensure_parent_dir_exists_creates_parent(self, mock_mkdir_method):
        file_path_obj = self.mock_project_root / "some_subdir" / "some_file.txt"
        self.fs_tools.ensure_parent_dir_exists(file_path_obj)
        self.assertTrue(mock_mkdir_method.called)
        args, kwargs = mock_mkdir_method.call_args
        self.assertEqual(kwargs, {'parents': True, 'exist_ok': True})

    @patch.object(Path, 'mkdir', side_effect=OSError("Permission denied"))
    def test_ensure_parent_dir_exists_oserror_raises_operationfailed(self, mock_mkdir_method):
        file_path_obj = self.mock_project_root / "another_subdir" / "another_file.txt"
        with self.assertRaises(OperationFailedError) as context:
            self.fs_tools.ensure_parent_dir_exists(file_path_obj)
        self.assertIn("Failed to create parent directory", str(context.exception))

    @patch('builtins.open', new_callable=mock_open)
    def test_create_file_success(self, mock_file_open):
        file_path_str = "test_create.txt"
        content = "dummy content"
        resolved_path = self.mock_project_root / file_path_str
        with patch.object(self.fs_tools, 'safe_file_operation', return_value=resolved_path), \
             patch.object(self.fs_tools, 'ensure_parent_dir_exists') as mock_ensure_parent:
            result = self.fs_tools.create_file(file_path_str, content)
        mock_ensure_parent.assert_called_once_with(resolved_path)
        mock_file_open.assert_called_once_with(resolved_path, 'w', encoding='utf-8')
        mock_file_open().write.assert_called_once_with(content)
        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 201)

    @patch('builtins.open', new_callable=mock_open, read_data="file data")
    def test_read_file_success(self, mock_file_open):
        file_path_str = "test_read.txt"
        resolved_path_mock = MagicMock(spec=Path)
        resolved_path_mock.exists.return_value = True
        resolved_path_mock.is_file.return_value = True
        resolved_path_mock.__str__.return_value = str(self.mock_project_root / file_path_str) 
        with patch.object(self.fs_tools, 'safe_file_operation', return_value=resolved_path_mock):
            result = self.fs_tools.read_file(file_path_str)
        mock_file_open.assert_called_once_with(resolved_path_mock, 'r', encoding='utf-8')
        self.assertTrue(result.success)
        self.assertEqual(result.data['content'], "file data")

    @patch('main.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    def test_delete_file_deletes_file(self, mock_unlink, mock_rmtree):
        file_path_str = "test_delete_me.txt"
        mock_path_obj = MagicMock(spec=Path); mock_path_obj.exists.return_value = True
        mock_path_obj.is_dir.return_value = False; mock_path_obj.unlink = mock_unlink 
        with patch.object(self.fs_tools, 'safe_file_operation', return_value=mock_path_obj):
            result = self.fs_tools.delete_file(file_path_str)
        self.assertTrue(result.success); mock_unlink.assert_called_once(); mock_rmtree.assert_not_called()

    @patch('main.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    def test_delete_file_deletes_directory(self, mock_unlink, mock_rmtree):
        dir_path_str = "test_delete_me_dir"
        mock_path_obj = MagicMock(spec=Path); mock_path_obj.exists.return_value = True; mock_path_obj.is_dir.return_value = True
        with patch.object(self.fs_tools, 'safe_file_operation', return_value=mock_path_obj):
            result = self.fs_tools.delete_file(dir_path_str)
        self.assertTrue(result.success); mock_rmtree.assert_called_once_with(mock_path_obj, onerror=unittest.mock.ANY); mock_unlink.assert_not_called()

    @patch('main.os.rename')
    def test_rename_file_success(self, mock_os_rename):
        old_f, new_f = "old.txt", "new.txt"
        mock_old = MagicMock(spec=Path); mock_old.exists.return_value = True
        mock_new = MagicMock(spec=Path); mock_new.exists.return_value = False
        with patch.object(self.fs_tools, 'safe_file_operation', side_effect=[mock_old, mock_new]), \
             patch.object(self.fs_tools, 'ensure_parent_dir_exists') as mock_ensure:
            res = self.fs_tools.rename_file(old_f, new_f)
        self.assertTrue(res.success); mock_ensure.assert_called_once_with(mock_new); mock_os_rename.assert_called_once_with(mock_old, mock_new)

    @patch('main.shutil.move')
    def test_move_file_success(self, mock_shutil_move):
        src, dest = "src.txt", "dest.txt"
        mock_src = MagicMock(spec=Path); mock_src.exists.return_value = True
        mock_dest = MagicMock(spec=Path); mock_dest.exists.return_value = False; mock_dest.is_file.return_value = True
        with patch.object(self.fs_tools, 'safe_file_operation', side_effect=[mock_src, mock_dest]), \
             patch.object(self.fs_tools, 'ensure_parent_dir_exists') as mock_ensure:
            res = self.fs_tools.move_file(src, dest)
        self.assertTrue(res.success); mock_ensure.assert_called_once_with(mock_dest); mock_shutil_move.assert_called_once_with(str(mock_src), str(mock_dest))

    @patch('main.os.listdir', return_value=["item1"])
    def test_list_directory_success(self, mock_listdir):
        dir_s = "d"; mock_p = MagicMock(spec=Path); mock_p.exists.return_value = True; mock_p.is_dir.return_value = True
        with patch.object(self.fs_tools, 'safe_file_operation', return_value=mock_p):
            res = self.fs_tools.list_directory(dir_s)
        self.assertTrue(res.success); self.assertEqual(res.data['items'], ["item1"]); mock_listdir.assert_called_once_with(mock_p)

    @patch('main.os.walk')
    @patch('builtins.open', new_callable=mock_open)
    def test_search_file_success(self, mock_f_open, mock_walk):
        s_path, kwd = "r", "kwd"; mock_s_root = self.mock_project_root / s_path
        f_root, f_sub = "f1.t", "f2.t"; sub_p = mock_s_root / "s1"
        mock_walk.return_value = [(str(mock_s_root), ["s1"], [f_root]), (str(sub_p), [], [f_sub])]
        p_f1_res, p_f2_res = mock_s_root / f_root, sub_p / f_sub
        m_p_f1, m_p_f2 = MagicMock(spec=Path), MagicMock(spec=Path)
        m_p_f1.is_file.return_value=True; m_p_f1.__str__=lambda s:str(p_f1_res)
        m_p_f2.is_file.return_value=True; m_p_f2.__str__=lambda s:str(p_f2_res)
        m_r_d_p_obj = MagicMock(spec=Path); m_r_d_p_obj.exists.return_value=True; m_r_d_p_obj.is_dir.return_value=True
        s_o_s_eff = [mock_s_root, m_p_f1, m_p_f2]
        def o_s_e(p_arg, *args, **kwargs):
            return mock_open(read_data=f"has {kwd}" if str(p_arg)==str(p_f1_res) else "other").return_value
        mock_f_open.side_effect = o_s_e_for_search
        with patch.object(self.fs_tools, 'safe_file_operation', side_effect=s_o_s_eff):
            res = self.fs_tools.search_file(kwd, s_path)
        self.assertTrue(res.success); self.assertEqual(len(res.data['matches']), 1); self.assertIn(str(p_f1_res), res.data['matches'][0])

    @patch('builtins.open', new_callable=mock_open)
    def test_chunk_file_reads_correct_chunk(self, mock_f_open_inst):
        f_p_s = "chunk.txt"; lines = [f"L{i+1}\n" for i in range(15)]; mock_f_open_inst.read_data="".join(lines)
        res_p_mock = MagicMock(spec=Path); res_p_mock.exists.return_value=True; res_p_mock.is_file.return_value=True
        res_p_mock.__str__.return_value = str(self.mock_project_root / f_p_s)
        with patch.object(self.fs_tools, 'safe_file_operation', return_value=res_p_mock):
            res = self.fs_tools.chunk_file(f_p_s, chunk_size=5, chunk_index=1)
        self.assertTrue(res.success); self.assertEqual(res.data['chunk_index'], 1); self.assertEqual(res.data['total_lines'], 15)
        self.assertEqual(res.data['total_chunks'], 3); self.assertEqual(res.data['start_line'], 5); self.assertEqual(res.data['end_line'], 9)
        self.assertEqual(res.data['chunk'], "".join(lines[5:10]))

    @patch('main.os.replace')
    @patch('builtins.open', new_callable=mock_open)
    def test_update_file_chunk_modifies_correctly(self, mock_open_m, mock_os_replace):
        f_p_s="update.txt"; orig_lines=[f"OL{i+1}\n" for i in range(10)]; new_c_cont="UpL6\nUpL7\n"
        res_p_mock=MagicMock(spec=Path); res_p_mock.exists.return_value=True; res_p_mock.is_file.return_value=True
        tmp_f_p_obj=self.mock_project_root/(f_p_s+".tmp.123"); res_p_mock.with_suffix.return_value=tmp_f_p_obj
        res_p_mock.__str__.return_value=str(self.mock_project_root/f_p_s)
        mock_open_m.side_effect=[mock_open(read_data="".join(orig_lines)).return_value, mock_open().return_value, mock_open(read_data="dummy").return_value]
        with patch.object(self.fs_tools, 'safe_file_operation', return_value=res_p_mock):
            res=self.fs_tools.update_file_chunk(f_p_s,new_c_cont,chunk_size=2,chunk_index=2)
        self.assertTrue(res.success, msg=f"Update failed: {res.error}"); mock_os_replace.assert_called_once_with(tmp_f_p_obj, res_p_mock)

class TestExecutionTools(unittest.TestCase):
    def setUp(self):
        self.exec_tools = ExecutionTools()
        # Mock FileSystemTools().safe_file_operation for ExecutionTools
        self.fs_patcher = patch('main.FileSystemTools')
        self.MockFileSystemTools = self.fs_patcher.start()
        self.mock_fs_instance = self.MockFileSystemTools.return_value
        self.mock_fs_instance.safe_file_operation.side_effect = mock_safe_file_operation

    def tearDown(self):
        self.fs_patcher.stop()

    @patch('main.subprocess.run')
    def test_run_script_success(self, mock_subprocess_run):
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="output", stderr="")
        result = self.exec_tools.run_script("valid/script.py")
        self.assertTrue(result.success)
        self.assertEqual(result.data['exit_code'], 0)
        mock_subprocess_run.assert_called_once()

    @patch('main.subprocess.Popen')
    def test_start_interactive_success(self, mock_subprocess_popen):
        mock_proc = MagicMock(); mock_proc.pid = 123
        mock_subprocess_popen.return_value = mock_proc
        result = self.exec_tools.start_interactive("valid/interactive.py")
        self.assertTrue(result.success)
        self.assertEqual(result.data['pid'], 123)
        mock_subprocess_popen.assert_called_once()

    @patch('main.subprocess.run')
    def test_install_package_success(self, mock_subprocess_run):
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="installed", stderr="")
        result = self.exec_tools.install_package("some_package")
        self.assertTrue(result.success)
        self.assertEqual(result.data['exit_code'], 0)
    
    # Add more tests for failure cases, invalid package names etc. for ExecutionTools

class TestBrowserTools(unittest.TestCase):
    def setUp(self):
        self.browser_tools = BrowserTools()
        self.fs_patcher = patch('main.FileSystemTools')
        self.MockFileSystemTools = self.fs_patcher.start()
        self.mock_fs_instance = self.MockFileSystemTools.return_value
        self.mock_fs_instance.safe_file_operation.side_effect = mock_safe_file_operation


    def tearDown(self):
        self.fs_patcher.stop()

    @patch('main.webbrowser.open')
    def test_open_in_browser_url_success(self, mock_webbrowser_open):
        result = self.browser_tools.open_in_browser("http://example.com")
        self.assertTrue(result.success)
        mock_webbrowser_open.assert_called_once_with("http://example.com")

    @patch('main.webbrowser.open')
    def test_open_in_browser_file_success(self, mock_webbrowser_open):
        # Need to ensure safe_file_operation is properly mocked for file paths
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = True
        resolved_path = Path("/test_base_dir/local_file.html").resolve()
        mock_path.resolve.return_value = resolved_path
        self.mock_fs_instance.safe_file_operation.return_value = mock_path
        
        result = self.browser_tools.open_in_browser("local_file.html")
        self.assertTrue(result.success)
        mock_webbrowser_open.assert_called_once_with(f"file://{str(resolved_path)}")


class TestGitTools(unittest.TestCase):
    def setUp(self):
        self.git_tools = GitTools()

    @patch('main.subprocess.run')
    def test_git_commit_success(self, mock_subprocess_run):
        # Simulate 'git add' and 'git commit' successes
        mock_subprocess_run.side_effect = [
            subprocess.CompletedProcess(args=['git', 'add', '.'], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=['git', 'commit', '-m', 'Test commit'], returncode=0, stdout="committed", stderr="")
        ]
        result = self.git_tools.git_commit("Test commit")
        self.assertTrue(result.success)
        self.assertEqual(mock_subprocess_run.call_count, 2)

    @patch('main.subprocess.run')
    def test_git_push_success(self, mock_subprocess_run):
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="pushed", stderr="")
        result = self.git_tools.git_push("origin", "main")
        self.assertTrue(result.success)
        mock_subprocess_run.assert_called_once()


class TestUserInteractionTools(unittest.TestCase):
    @patch('builtins.input', return_value="user says yes")
    def test_prompt_input_success(self, mock_input):
        tools = UserInteractionTools()
        result = tools.prompt_input("Proceed?")
        self.assertEqual(result['user_input'], "user says yes")
        self.assertNotIn('error', result)


class TestOperationResult(unittest.TestCase):
    def test_operation_result_instantiation(self):
        res_success = OperationResult(True, data={"key": "value"}, status_code=200)
        self.assertTrue(res_success.success); self.assertEqual(res_success.data, {"key": "value"}); self.assertIsNone(res_success.error); self.assertEqual(res_success.status_code, 200)
        res_failure = OperationResult(False, error="Test Error", status_code=500)
        self.assertFalse(res_failure.success); self.assertIsNone(res_failure.data); self.assertEqual(res_failure.error, "Test Error"); self.assertEqual(res_failure.status_code, 500)

class TestCustomExceptions(unittest.TestCase):
    def test_custom_exceptions_raising(self):
        with self.assertRaises(AgentError): raise AgentError("Base agent error")
        with self.assertRaises(FilePathError): raise FilePathError("File path specific error")
        with self.assertRaises(OperationFailedError): raise OperationFailedError("Operation failed specific error")
        with self.assertRaises(GeminiClientError): raise GeminiClientError("Gemini client error")

class TestUtils(unittest.TestCase):
    def test_strip_code_block_markers(self):
        self.assertEqual(strip_code_block_markers("```python\ncode\n```"), "code")
        self.assertEqual(strip_code_block_markers("```\ncode\n```"), "code")

    @patch('pathlib.Path.is_file') 
    @patch('pathlib.Path.resolve') 
    @patch('main.Path.cwd') 
    def test_parse_python_traceback(self, mock_main_cwd, mock_resolve, mock_is_file):
        mock_main_cwd.return_value = Path("/app")
        def resolve_s_e(p_arg_inst): return mock_main_cwd()/p_arg_inst if not p_arg_inst.is_absolute() else p_arg_inst
        mock_resolve.side_effect = resolve_s_e
        def is_file_s_e_func(p_obj_arg):
            s_p = str(p_obj_arg)
            return s_p=="/app/test.py" or s_p=="/app/project/module.py" or s_p=="/tmp/ipykernel_123/12345.py"
        mock_is_file.side_effect = is_file_s_e_func
        tb1 = """Traceback (most recent call last):
  File "test.py", line 10, in <module>
ValueError: Test error"""
        self.assertEqual(parse_python_traceback(tb1), [{'file': str(Path("/app/test.py")), 'line': 10}])
        tb_ipy = """
----> 1 /tmp/ipykernel_123/12345.py(1)<module>
NameError: name 'x' is not defined"""
        self.assertEqual(parse_python_traceback(tb_ipy), [{'file': '/tmp/ipykernel_123/12345.py', 'line': 1}])

class TestAgent(unittest.TestCase):
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"}, clear=True)
    @patch('main.Agent._GeminiClient') 
    def test_agent_instantiation(self, MockGeminiClient):
        try:
            agent = Agent(api_key="test_api_key_arg"); self.assertIsNotNone(agent); MockGeminiClient.assert_called_once()
            MockGeminiClient.reset_mock()
            agent_env = Agent(); self.assertIsNotNone(agent_env); MockGeminiClient.assert_called_once()
        except ValueError as e: self.fail(f"Agent instantiation failed: {e}")
            
    @patch.dict(os.environ, {}, clear=True) 
    def test_agent_instantiation_no_api_key_raises_valueerror(self):
        with self.assertRaises(ValueError) as ctx: Agent()
        self.assertIn("GOOGLE_API_KEY", str(ctx.exception))

if __name__ == '__main__':
    unittest.main()
