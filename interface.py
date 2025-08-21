from model_loader import load_pipe
from chat_memory import chat_memory
import re

def generate_response():
    """
    AI-powered chatbot using Hugging Face transformers with proper memory management.
    
    This chatbot relies entirely on the language model for responses while implementing
    robust text processing and memory management for coherent conversations.
    """
    # Load the pipeline
    pipe = load_pipe()
    if pipe is None:
        print("Failed to load model. Exiting.")
        return
    
    add_user, add_bot, get_context = chat_memory()
    
    print("Chatbot ready! Type /exit to quit.")
    
    while True:
        user_input = input("User: ").strip()
        
        if user_input == "/exit":
            print("Exiting chatbot. Goodbye!")
            break
        
        if not user_input:
            continue
        
        add_user(user_input)
        
        context = get_context(2)  # Uses last 2 exchanges
        
        if context:
            # Format for instruction-tuned models
            conversation = []
            for role, text in context[:-1]:  # Exclude current user input
                if role.lower() == "user":
                    conversation.append(f"User: {text}")
                else:
                    conversation.append(f"Assistant: {text}")
            
            prompt = "\n".join(conversation) + f"\nUser: {user_input}\nAssistant:"
        else:
            prompt = f"User: {user_input}\nAssistant:"
        
        try:
            response = pipe(
                prompt,
                max_new_tokens=40,
                temperature=0.5,          # Lower for more focused responses
                top_p=0.8,               # Slightly more focused sampling
                do_sample=True,
                repetition_penalty=1.1,  
                pad_token_id=pipe.tokenizer.eos_token_id,
                eos_token_id=pipe.tokenizer.eos_token_id
            )
            # Get the generated text
            generated_text = response[0]['generated_text']
            bot_response = generated_text[len(prompt):].strip()
            
            # Post-process the response for better quality
            bot_response = clean_response(bot_response)
            
            if is_valid_response(bot_response, user_input):
                print(f"Bot: {bot_response}")
                add_bot(bot_response)
            else:
                # Retry with instruction format if response is poor
                simple_prompt = f"Please answer this question: {user_input}\n\nAnswer:"
                retry_response = pipe(
                    simple_prompt,
                    max_new_tokens=25,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=pipe.tokenizer.eos_token_id
                )
                
                bot_response = retry_response[0]['generated_text'][len(simple_prompt):].strip()
                bot_response = clean_response(bot_response)
                
                if bot_response and len(bot_response) > 5:
                    print(f"Bot: {bot_response}")
                    add_bot(bot_response)
                else:
                    fallback_response = "I'm having trouble understanding. Could you rephrase your question?"
                    print(f"Bot: {fallback_response}")
                    add_bot(fallback_response)
            
        except Exception as e:
            error_response = "Sorry, I encountered an error. Please try again."
            print(f"Bot: {error_response}")
            add_bot(error_response)

def clean_response(response):
    """Clean and post-process the generated response for Gemma models."""
    if not response:
        return ""
    
    # Remove unwanted patterns specific to instruction models
    response = re.sub(r'\bUser:\b.*', '', response)      
    response = re.sub(r'\bAssistant:\b', '', response)   
    response = re.sub(r'\bBot:\b', '', response)         
    
    # Clean up common instruction artifacts
    response = re.sub(r'^(Answer:|Response:)\s*', '', response, flags=re.IGNORECASE)
    
    # Take the first complete sentence to avoid rambling
    sentences = re.split(r'[.!?]\s+', response.strip())
    if sentences and sentences[0]:
        response = sentences[0].strip()
        if not response.endswith(('.', '!', '?')):
            response += '.'
    
    # Clean up whitespace and weird characters
    response = ' '.join(response.split())
    response = ''.join(char for char in response if char.isprintable())
    
    return response.strip()

def is_valid_response(response, user_input):
    """Validate if the generated response is appropriate."""
    if not response or len(response) < 3:
        return False
    
    words = response.lower().split()
    if len(words) > 3 and len(set(words)) < len(words) * 0.5:
        return False
    
    # Check if response is just punctuation or gibberish
    if not any(char.isalpha() for char in response):
        return False
    
    # Check for reasonable length
    if len(response) > 200:
        return False
    
    return True

if __name__ == "__main__":
    generate_response()