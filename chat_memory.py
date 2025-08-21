from typing import List, Tuple, Callable

def chat_memory() -> Tuple[Callable, Callable, Callable]:
    """
    Create a chat memory to store and retrieve chat history.
    
    Returns:
        Tuple containing:
            - Function to add user message.
            - Function to add bot message.
            - Function to get context with a specified limit.
    """
    memory = []

    def add_user(text: str):
        """Add a user message to the chat memory."""
        memory.append(("User", text))

    def add_bot(text: str):
        """Add a bot message to the chat memory."""
        memory.append(("Bot", text))

    def get_context(limit: int = 4) -> List[Tuple[str, str]]:
        """Retrieve the last 'limit' messages from the chat memory (sliding window)."""
        return memory[-limit:] if len(memory) > limit else memory

    return add_user, add_bot, get_context