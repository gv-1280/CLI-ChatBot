from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Specify the model
MODEL = "google/gemma-3-270m-it"

def load_pipe(MODEL_NAME: str = MODEL):
    """
    Load the model and tokenizer for text generation.
    
    Args:
        MODEL_NAME (str): The name of the model to load.
        
    Returns:
        pipeline: A text generation pipeline using the specified model.
    """
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        
        # Create a text generation pipeline
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        
        return pipe
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None