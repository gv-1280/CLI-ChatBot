from typing import List, Tuple

#def chat_memory function to store and retrieve chat history upto a certain limit
# the funcationality is as follows:
# 1. store as list of (role,text) tuples.
# 2. provide add_user(text) add_bot(text) get_context(limit=4)

def chat_memory() -> Tuple[List[Tuple[str, str]], callable, callable, callable]:
    """
    Create a chat memory to store and retrieve chat history.
    
    Returns:
        Tuple containing:
            - List of tuples representing chat history (role, text).
            - Function to add user message.
            - Function to add bot message.
            - Function to get context with a specified limit.
    """
    memory = []

    def add_user(text: str):
        """Add a user message to the chat memory."""
        memory.append(("user", text))

    def add_bot(text: str):
        """Add a bot message to the chat memory."""
        memory.append(("bot", text))

    def get_context(limit: int = 4) -> List[Tuple[str, str]]:
        """Retrieve the last 'limit' messages from the chat memory."""
        return memory[-limit:]

    return memory, add_user, add_bot, get_context