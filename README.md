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
- **No Caching**: The agent no longer uses Gemini server-side caching. All chat and tool operations are performed live, ensuring compatibility and reliability for all script execution and file operations.
- **Smart Script Execution**: When you ask the agent to run a script, it will automatically detect if the script requires user input or is a server. If so, it launches the script in a new terminal window for interactive use; otherwise, it runs the script in the current terminal and returns the output directly.

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

  For Windows (Command Prompt):

  ```
  set GOOGLE_API_KEY=YOUR_API_KEY_HERE
  ```

  For Windows (PowerShell):

  ```powershell
  $env:GOOGLE_API_KEY="YOUR_API_KEY_HERE"
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
- **Large File Processing**: To process a large file automatically in manageable chunks, use:
  ```
  process_file <path>
  ```
  The agent will process the file chunk by chunk and show progress.

## Chunked File Processing and Editing

When working with very large files (1,000+ lines), use the chunked file tools to avoid memory or token errors:

- **Read a file in chunks:**

  ```python
  response = chunk_file('path/to/large_file.py', chunk_size=100, chunk_index=0)
  # chunk_index is 0-based; increase to get the next chunk
  # response['chunk'] contains the chunk content
  # response['total_chunks'] gives the number of chunks
  ```

- **Edit a specific chunk:**

  ```python
  update_file_chunk('path/to/large_file.py', new_chunk_content, chunk_size=100, chunk_index=0)
  # Only the specified chunk is replaced
  ```

- **Recommended workflow:**
  1. Use `chunk_file` to read each chunk sequentially.
  2. Edit the chunk as needed.
  3. Use `update_file_chunk` to write back only the changed chunk.
  4. Repeat for all chunks.

This allows you to process and edit files of any size without exceeding memory or token limits.

## Editing Large Codebases

When working with projects that exceed token limits:

- **Recommended:** Use the `process_file <path>` command for automatic chunked processing and progress feedback.
- **Manual (Advanced):**
  1. Read in Chunks
     ```python
     response = chunk_file('path/to/large_file.py')
     ```
  2. Review & Edit each chunk sequentially via your prompts.
  3. Apply Changes with `update_file('path/to/large_file.py', edited_content)`.
  4. Commit changes:
     ```python
     git_commit('Refactored large_file in chunks')
     ```

---

# Agent Workflow & Project Structure

This project uses an advanced agent-based workflow for code generation and project management. The agent can:

- Detect single-file and multi-file/full-project requests from your prompt.
- For multi-file/full-project requests, the agent:
  1. Prints: `Agent: Detected a multi-file/full-project request. Using supervisor workflow...`
  2. Generates a file/folder plan and prints it.
  3. Generates each file (skipping `main.py`, `README.md`, `requirements.txt`).
  4. Prints: `Agent: All files generated and combined. Your project is ready!`
- For single-file or simple requests, the agent responds directly.

## Example Usage

```
You: Create an interactive Neural Network simulator web app where the user can specify the number of input and output nodes, choose the number of hidden layers and nodes per layer, select activation functions for each layer (ReLU, GELU, Softmax, etc.), and input custom data for forward and backward passes. The app should visually display the network structure with different colors for each layer and activation, simulate and animate both forward and backward passes, show how logits and softmax outputs change, and clearly illustrate how weights and biases are updated step by step. Include visual explanations, tooltips, and make the UI intuitive, educational, and visually appealing, with all updates and changes clearly shown.

Agent: Detected a multi-file/full-project request. Using supervisor workflow...
Agent: Generating file/folder plan...
Agent: File plan: ["app.py", "neural_network.py", "templates/index.html", "static/css/style.css", ...]
Agent: Generating app.py...
Agent: app.py generated.
...
Agent: All files generated and combined. Your project is ready!
```

## Project Structure

- `main.py` – The entrypoint and agent logic (do not modify or delete).
- `README.md` – This file.
- `requirements.txt` – Python dependencies.
- `example1/`, `example2/` – Example projects generated by the agent.

## How to Use

1. **Run the agent:**
   ```bash
   python main.py
   ```
2. **Type your prompt:**
   For example:
   - "Create a Flask app with a homepage and about page."
   - "Build a neural network simulator web app with interactive visualization."
3. **Exit:**
   Type `exit` or `quit` to terminate the session.

## Notes

- The agent will never write to or delete `main.py`, `README.md`, or `requirements.txt`.
- All generated files are placed in appropriate folders (e.g., `templates/`, `static/css/`, `static/js/`).
- For large files, use chunked processing commands as described in the code.

---

## Example: Quirky Excuse Generator Website

When you run the project, you’ll see:

```
=== Gemini Super Agent ===
You: Build a quirky website that generates random, funny excuses for being late to work or school. Each time the user clicks a button, display a new, hilarious excuse. Add a feature to copy the excuse to the clipboard and a button to share it on social media. Create all the files and folders accordingly
Agent: Okay, the files are created, and the website should be open in your browser.

You can now:

1. Click the "Get New Excuse!" button to see different funny excuses.
2. Click the "Copy Excuse" button to copy the current excuse to your clipboard.
3. Click the "Share (Twitter)" button to open a new Twitter tab with a pre-filled tweet containing the excuse.

Enjoy your quirky excuse generator! Let me know if you'd like any modifications.
```

**Note:**

- The `example` directory contains the `static` and `templates` folders for the website.
- All files in the `example` directory are committed to version control.

---

## Example 2: Interactive Neural Network Simulator

This example demonstrates the agent's ability to generate a full-featured, multi-file web application for neural network simulation and visualization.

**Prompt used to generate Example 2:**

```
Create an interactive Neural Network simulator web app where the user can specify the number of input and output nodes, choose the number of hidden layers and nodes per layer, select activation functions for each layer (ReLU, GELU, Softmax, etc.), and input custom data for forward and backward passes. The app should visually display the network structure with different colors for each layer and activation, simulate and animate both forward and backward passes, show how logits and softmax outputs change, and clearly illustrate how weights and biases are updated step by step. Include visual explanations, tooltips, and make the UI intuitive, educational, and visually appealing, with all updates and changes clearly shown.
```

**Agent workflow and messages:**

- Detected a multi-file/full-project request. Using supervisor workflow...
- Generating file/folder plan...
- File plan: ["app.py", "neural_network.py", "templates/index.html", "static/css/style.css", "static/css/network-visualization.css", "static/js/script.js", "static/js/network-visualization.js", "static/js/simulation.js"]
- Generating each file...
- All files generated and combined. Your project is ready!

**Features:**

- Specify number of input/output nodes, hidden layers, and nodes per layer
- Choose activation functions for each layer (ReLU, GELU, Softmax, etc.)
- Input custom data for forward and backward passes
- Visualize the network structure with color-coded layers and activations
- Animate forward and backward passes, showing how logits and softmax outputs change
- Step-by-step visualization of weight and bias updates
- Visual explanations, tooltips, and an intuitive, educational UI

**File/folder structure for Example 2:**

```
example2/
  app.py
  neural_network.py
  static/
    css/
      network-visualization.css
      style.css
    js/
      network-visualization.js
      script.js
      simulation.js
  templates/
    index.html
```

**How to run Example 2:**

1. Open a terminal in the `example2` directory:
   ```bash
   cd example2
   ```
2. Install any required dependencies (Flask, etc.) if not already installed:
   ```bash
   pip install flask
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open your browser and go to the address shown in the terminal (usually http://127.0.0.1:5000/).

---

## Development & Contributing

- Feel free to fork, open issues, or submit pull requests.
- Ensure code passes linting and tests before submitting.
