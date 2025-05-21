import unittest
from unittest.mock import patch, MagicMock, ANY, call
import os
import sys
from datetime import datetime, timedelta
import time # For patching time.sleep
import queue # For TokenBucket lock testing

# Add parent directory to sys.path to allow importing main.py components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components from main.py
try:
    from main import (
        TokenBucket,
        GeminiClient,
        is_multifile_request,
        SYSTEM_PROMPT, # Needed for verifying GeminiClient.create_chat
        prompt_input   # Tool used by GeminiClient
    )
    # Import the utility modules that are used as tools by GeminiClient
    import file_utils
    import env_utils
except ImportError as e:
    print(f"ImportError in test_main_components.py: {e}. Ensure main.py and utils are accessible.")
    # This might happen if the script is not run from the project root or if there are circular dependencies.
    raise

# For GeminiClient.create_chat, we need the actual functions for the tools list
# as they are passed as function objects.
# The genai library itself will be mocked.
try:
    from google.genai import types as genai_types # For GenerateContentConfig
except ImportError:
    # Create a dummy genai_types if google.genai is not installed in the test environment
    # This allows tests to run without the actual genai library for non-API interaction parts.
    class MockGenerateContentConfig:
        def __init__(self, tools, temperature, system_instruction):
            self.tools = tools
            self.temperature = temperature
            self.system_instruction = system_instruction
    
    class MockGenaiTypes:
        GenerateContentConfig = MockGenerateContentConfig
    
    genai_types = MockGenaiTypes()
    print("Warning: google.genai.types not found, using mock. Some tests might be affected if they rely on specific genai types.")


class TestTokenBucket(unittest.TestCase):
    """Tests for the TokenBucket class."""

    def setUp(self):
        self.tokens_per_min = 100
        self.requests_per_min = 10
        self.bucket = TokenBucket(self.tokens_per_min, self.requests_per_min)

    def test_initialization(self):
        self.assertEqual(self.bucket.tokens, self.tokens_per_min)
        self.assertEqual(self.bucket.requests, self.requests_per_min)
        self.assertIsNotNone(self.bucket.last_refill)

    def test_consume_success(self):
        consumed = self.bucket.consume(10)
        self.assertTrue(consumed)
        self.assertEqual(self.bucket.tokens, self.tokens_per_min - 10)
        self.assertEqual(self.bucket.requests, self.requests_per_min - 1)

    def test_consume_insufficient_tokens(self):
        consumed = self.bucket.consume(self.tokens_per_min + 1)
        self.assertFalse(consumed)
        self.assertEqual(self.bucket.tokens, self.tokens_per_min) # Tokens not changed

    def test_consume_no_requests_left(self):
        self.bucket.requests = 0
        consumed = self.bucket.consume(10)
        self.assertFalse(consumed)

    @patch('main.datetime') # Patch datetime within the main module where TokenBucket uses it
    def test_refill(self, mock_datetime):
        # Simulate time passing for refill
        past_time = datetime.now() - timedelta(seconds=61)
        mock_datetime.now.return_value = past_time # Initial time
        
        bucket_for_refill = TokenBucket(self.tokens_per_min, self.requests_per_min)
        bucket_for_refill.consume(50) # Consume some tokens
        self.assertEqual(bucket_for_refill.tokens, self.tokens_per_min - 50)
        
        # Advance time by more than 60 seconds for the mock
        future_time = past_time + timedelta(seconds=61)
        mock_datetime.now.return_value = future_time
        
        # Trigger refill by attempting to consume (or by direct call to _refill if it were public)
        bucket_for_refill.consume(1) 
        
        self.assertEqual(bucket_for_refill.tokens, self.tokens_per_min - 1) # Refilled then consumed 1
        self.assertEqual(bucket_for_refill.requests, self.requests_per_min -1)
        self.assertEqual(bucket_for_refill.last_refill, future_time)

    @patch.object(queue.Queue, 'get')
    def test_consume_lock_unavailable(self, mock_queue_get):
        mock_queue_get.side_effect = queue.Empty # Simulate lock not being acquired
        
        bucket_locked = TokenBucket(self.tokens_per_min, self.requests_per_min)
        consumed = bucket_locked.consume(10)
        self.assertFalse(consumed)
        mock_queue_get.assert_called_once_with(timeout=1)


@patch('main.genai.Client') # Patch the genai.Client where it's imported in main.py
class TestGeminiClient(unittest.TestCase):
    """Tests for the GeminiClient class."""

    def setUp(self):
        self.api_key = "test_api_key"
        # We will mock TokenBucket behavior per test, or use a real one if logic isn't complex

    @patch('main.TokenBucket') # Patch TokenBucket where it's used by GeminiClient
    def test_init(self, mock_token_bucket_class, mock_genai_client_constructor):
        mock_genai_client_instance = MagicMock()
        mock_genai_client_constructor.return_value = mock_genai_client_instance
        
        mock_tb_instance = MagicMock()
        mock_token_bucket_class.return_value = mock_tb_instance

        client = GeminiClient(self.api_key)
        
        mock_genai_client_constructor.assert_called_once_with(api_key=self.api_key)
        self.assertEqual(client.client, mock_genai_client_instance)
        mock_token_bucket_class.assert_called_once_with(tokens_per_min=1000000, requests_per_min=100)
        self.assertEqual(client.token_bucket, mock_tb_instance)
        self.assertIsNone(client.chat)


    @patch('main.TokenBucket') 
    def test_create_chat_success(self, mock_token_bucket_class, mock_genai_client_constructor):
        mock_genai_client_instance = MagicMock()
        mock_chats_service = MagicMock()
        mock_chat_object = MagicMock()
        mock_chats_service.create.return_value = mock_chat_object
        mock_genai_client_instance.chats = mock_chats_service
        mock_genai_client_constructor.return_value = mock_genai_client_instance

        client = GeminiClient(self.api_key)
        client.create_chat()

        # Verify that the tools list is correctly constructed
        expected_tools = [
            file_utils.create_file, file_utils.read_file, file_utils.update_file,
            file_utils.delete_file, file_utils.rename_file, file_utils.move_file,
            file_utils.list_directory, file_utils.search_file,
            file_utils.chunk_file, file_utils.update_file_chunk,
            env_utils.run_script, env_utils.start_interactive, env_utils.install_package,
            env_utils.open_in_browser, env_utils.lint_code, env_utils.format_code,
            env_utils.run_tests, env_utils.git_commit, env_utils.git_push,
            prompt_input
        ]

        mock_chats_service.create.assert_called_once()
        args, kwargs = mock_chats_service.create.call_args
        self.assertEqual(kwargs['model'], 'models/gemini-2.5-flash-preview-04-17')
        self.assertIsInstance(kwargs['config'], genai_types.GenerateContentConfig)
        self.assertListEqual(kwargs['config'].tools, expected_tools) # Check if tools match
        self.assertEqual(kwargs['config'].temperature, 0)
        self.assertEqual(kwargs['config'].system_instruction, SYSTEM_PROMPT)
        self.assertEqual(client.chat, mock_chat_object)


    @patch('main.TokenBucket')
    def test_create_chat_failure(self, mock_token_bucket_class, mock_genai_client_constructor):
        mock_genai_client_instance = MagicMock()
        mock_chats_service = MagicMock()
        mock_chats_service.create.side_effect = Exception("API Error")
        mock_genai_client_instance.chats = mock_chats_service
        mock_genai_client_constructor.return_value = mock_genai_client_instance

        client = GeminiClient(self.api_key)
        with self.assertRaises(Exception) as context:
            client.create_chat()
        self.assertIn("API Error", str(context.exception))
        self.assertIsNone(client.chat)


    @patch('main.time.sleep') # Patch time.sleep
    @patch('main.TokenBucket')
    def test_send_message_success(self, mock_token_bucket_class, mock_time_sleep, mock_genai_client_constructor):
        # Setup client and mocks
        mock_genai_client_instance = MagicMock()
        mock_chat_object = MagicMock()
        mock_chat_object.send_message.return_value = MagicMock(text="AI response")
        mock_genai_client_instance.chats.create.return_value = mock_chat_object # For create_chat
        mock_genai_client_constructor.return_value = mock_genai_client_instance
        
        mock_tb_instance = MagicMock()
        mock_tb_instance.consume.return_value = True # Allow consumption
        mock_token_bucket_class.return_value = mock_tb_instance

        client = GeminiClient(self.api_key)
        client.chat = mock_chat_object # Assume chat is already created for this specific test part

        response_text = client.send_message("Hello AI")

        self.assertEqual(response_text, "AI response")
        mock_tb_instance.consume.assert_called_once_with(1000) # Default max_tokens
        mock_chat_object.send_message.assert_called_once_with("Hello AI")
        mock_time_sleep.assert_not_called()


    @patch('main.time.sleep')
    @patch('main.TokenBucket')
    def test_send_message_chat_creation_first_call(self, mock_token_bucket_class, mock_time_sleep, mock_genai_client_constructor):
        mock_genai_client_instance = MagicMock()
        mock_chat_object = MagicMock()
        mock_chat_object.send_message.return_value = MagicMock(text="First response")
        mock_genai_client_instance.chats.create.return_value = mock_chat_object
        mock_genai_client_constructor.return_value = mock_genai_client_instance

        mock_tb_instance = MagicMock()
        mock_tb_instance.consume.return_value = True
        mock_token_bucket_class.return_value = mock_tb_instance
        
        client = GeminiClient(self.api_key) # client.chat is None initially
        self.assertIsNone(client.chat)

        response_text = client.send_message("Test message")
        
        self.assertEqual(response_text, "First response")
        mock_genai_client_instance.chats.create.assert_called_once() # create_chat was called
        self.assertIsNotNone(client.chat) # Chat should now be initialized
        mock_chat_object.send_message.assert_called_once_with("Test message")


    @patch('main.time.sleep')
    @patch('main.TokenBucket')
    def test_send_message_rate_limited_then_succeed(self, mock_token_bucket_class, mock_time_sleep, mock_genai_client_constructor):
        mock_genai_client_instance = MagicMock()
        mock_chat_object = MagicMock()
        mock_chat_object.send_message.return_value = MagicMock(text="Delayed Response")
        mock_genai_client_instance.chats.create.return_value = mock_chat_object
        mock_genai_client_constructor.return_value = mock_genai_client_instance

        mock_tb_instance = MagicMock()
        # Simulate rate limit: fail first, then succeed
        mock_tb_instance.consume.side_effect = [False, True] 
        mock_token_bucket_class.return_value = mock_tb_instance

        client = GeminiClient(self.api_key)
        client.chat = mock_chat_object # Pre-initialize chat

        response_text = client.send_message("Rate limit test")

        self.assertEqual(response_text, "Delayed Response")
        self.assertEqual(mock_tb_instance.consume.call_count, 2)
        mock_time_sleep.assert_called_once_with(client.retry_delay) # Check if sleep was called
        mock_chat_object.send_message.assert_called_once_with("Rate limit test")


    @patch('main.time.sleep')
    @patch('main.TokenBucket')
    def test_send_message_retry_then_succeed(self, mock_token_bucket_class, mock_time_sleep, mock_genai_client_constructor):
        mock_genai_client_instance = MagicMock()
        mock_chat_object = MagicMock()
        # Fail once, then succeed
        mock_chat_object.send_message.side_effect = [
            Exception("Temporary API Error"), 
            MagicMock(text="Success after retry")
        ]
        mock_genai_client_instance.chats.create.return_value = mock_chat_object # For initial and recreated chat
        mock_genai_client_constructor.return_value = mock_genai_client_instance
        
        mock_tb_instance = MagicMock()
        mock_tb_instance.consume.return_value = True # Always allow consumption for this test
        mock_token_bucket_class.return_value = mock_tb_instance

        client = GeminiClient(self.api_key)
        # client.chat will be created on first send_message attempt if it fails
        
        response_text = client.send_message("Retry test")

        self.assertEqual(response_text, "Success after retry")
        self.assertEqual(mock_chat_object.send_message.call_count, 2)
        self.assertEqual(mock_genai_client_instance.chats.create.call_count, 2) # Initial + 1 for retry
        mock_time_sleep.assert_called_once_with(client.retry_delay)


    @patch('main.time.sleep')
    @patch('main.TokenBucket')
    def test_send_message_max_retries_exceeded(self, mock_token_bucket_class, mock_time_sleep, mock_genai_client_constructor):
        mock_genai_client_instance = MagicMock()
        mock_chat_object = MagicMock()
        mock_chat_object.send_message.side_effect = Exception("Persistent API Error") # Always fail
        # Mock create to be called multiple times (initial + retries)
        mock_genai_client_instance.chats.create.return_value = mock_chat_object 
        mock_genai_client_constructor.return_value = mock_genai_client_instance

        mock_tb_instance = MagicMock()
        mock_tb_instance.consume.return_value = True
        mock_token_bucket_class.return_value = mock_tb_instance

        client = GeminiClient(self.api_key)
        client.max_retries = 2 # For faster test

        response_text = client.send_message("Max retry test")

        self.assertIsNone(response_text) # Should return None after max retries
        self.assertEqual(mock_chat_object.send_message.call_count, client.max_retries)
        # create.call_count = initial create + (max_retries -1) for send failures
        self.assertEqual(mock_genai_client_instance.chats.create.call_count, client.max_retries) 
        self.assertEqual(mock_time_sleep.call_count, client.max_retries - 1)


class TestIsMultifileRequest(unittest.TestCase):
    """Tests for the is_multifile_request function."""

    def test_multifile_keywords(self):
        multifile_inputs = [
            "Create a full project for a flask app with HTML, CSS, and JS.",
            "Generate multiple files: app.py, utils.py, and models.py.",
            "Develop a folder structure for a new web service.",
            "Need a full stack application with backend and frontend.",
            "The project requires a detailed plan for all components.",
            "Edit all files in the src directory to update the API endpoint."
        ]
        for text_input in multifile_inputs:
            with self.subTest(input=text_input):
                self.assertTrue(is_multifile_request(text_input))

    def test_single_file_or_generic_keywords(self):
        single_file_inputs = [
            "Write a Python function to sort a list.",
            "What is the capital of France?",
            "run the script",
            "show output",
            "print result",
            "run main.py",
            "Can you help me debug this code snippet?",
            "Explain how a token bucket works."
        ]
        for text_input in single_file_inputs:
            with self.subTest(input=text_input):
                self.assertFalse(is_multifile_request(text_input))
    
    def test_empty_and_whitespace_input(self):
        self.assertFalse(is_multifile_request(""))
        self.assertFalse(is_multifile_request("    "))


if __name__ == '__main__':
    unittest.main(verbosity=2)
