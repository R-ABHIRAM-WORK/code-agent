# Code Agent

Code Agent is a supercharged coding assistant powered by Google Gemini. It provides full filesystem and Git integration, linting, formatting, testing, and more, all accessible via natural language prompts.

## Features

- **File Operations**: create, read, update, delete, move, rename, and search files via simple commands.
- **Chunked File Reading**: handle large files seamlessly with the `chunk_file` tool to avoid token limits.
- **Code Execution**: run scripts non-interactively (`run_script`) or interactively (`start_interactive`).
- **Environment Management**: install Python packages on the fly with `install_package`.
- **Web Preview**: preview HTML or URLs directly in your default browser using `open_in_browser`.
- **Code Quality**: lint (`lint_code`), format (`format_code`), and test (`run_tests`) your code.
- **Version Control**: stage, commit (`git_commit`), and push (`git_push`) changes with Git.
- **User Prompts**: gather user input on demand via `prompt_input`.

## Prerequisites

- **Python**: Version 3.8 or higher.
- **Dependencies**: Install via pip:
  ```bash
  pip install google-generativeai flake8 black pytest
  ```
- **API Key**: Export your Google Generative AI API key:
  ```bash
  export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
  ```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd code-agent
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

- The main entrypoint is `main.py`. **Do not delete or rename** this file.
- All available tools are defined and registered in the script’s `SYSTEM_PROMPT` and passed to the Gemini chat client.

## Usage

Run the agent:

```bash
python main.py
```

- **Prompt**: Type your natural-language commands (e.g., "create a new file config.json with default settings").
- **Exit**: Type `exit` or `quit` to terminate the session.
- **Streaming**: Outputs stream in real time; large responses won’t exceed token limits.

## Editing Large Codebases

When working with projects that exceed token limits:

1. **Read in Chunks**
   ```python
   response = chunk_file('path/to/large_file.py')
   ```
2. **Review & Edit** each chunk sequentially via your prompts.
3. **Apply Changes** with `update_file('path/to/large_file.py', edited_content)`.
4. **Commit** changes:
   ```python
   git_commit('Refactored large_file in chunks')
   ```

## Development & Contributing

- Feel free to fork, open issues, or submit pull requests.
- Ensure code passes linting and tests before submitting.
