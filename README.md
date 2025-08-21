# Local Chatbot with Hugging Face

A simple command-line chatbot using Hugging Face transformers with conversation memory.

## Setup

1. **Install dependencies:**
```bash
pip install transformers torch
```

2. **Download the files:**
- `model_loader.py` - Loads the Hugging Face model
- `chat_memory.py` - Manages conversation history  
- `interface.py` - Main chat interface

3. **Get access to model:**
   - Log in to [Hugging Face](https://huggingface.co/)
   - Search for `google/gemma-2-2b-it` (recommended) or `microsoft/DialoGPT-small`
   - Click "Access repository" if prompted and follow the steps
   - Generate a Hugging Face access token:
     - Go to Settings → Access Tokens
     - Create a new token with "Read" permissions
   - Connect your local environment:
   ```bash
   huggingface-cli login
   ```
   - Enter your token when prompted
   - The model will automatically download on first use

## How to Run

```bash
python interface.py
```

## Usage

- Type your messages and press Enter
- The bot remembers the last 4 exchanges  
- Type `/exit` to quit
- Type `/clear` to reset conversation memory

## Sample Interaction

```
Chatbot ready! Type /exit to quit.
User: What is the capital of France?
Bot: The capital of France is Paris.
User: And what about Italy?  
Bot: The capital of Italy is Rome.
User: /exit
Exiting chatbot. Goodbye!
```

## Models

**Default**: `microsoft/DialoGPT-small` (117M parameters) - optimized for conversations and runs on most hardware.

**Alternative**: `google/gemma-2-2b-it` (2B parameters) - more capable but requires more RAM (8GB+ recommended).

You can change the model in `model_loader.py` if needed.

## Features

- **Sliding window memory**: Keeps last 4 conversation turns
- **Local execution**: No internet needed after initial download
- **Error handling**: Graceful failure recovery
- **Simple CLI**: Easy to use terminal interface
- **Model flexibility**: Easy to switch between different models

## System Requirements

- **Minimum**: 4GB RAM (for DialoGPT-small)
- **Recommended**: 8GB+ RAM (for larger models like Gemma)
- **Storage**: 1-5GB depending on model size
- **Python**: 3.7+