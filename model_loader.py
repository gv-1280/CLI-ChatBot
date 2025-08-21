from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

#Specify the model name
Model = "distilbert/distilgpt2"

def load_pipe(MODEL_NAME: str = Model):
    """
    Load the model and tokenizer for text generation.
    
    Args:
        MODEL_NAME (str): The name of the model to load.
        
    Returns:
        pipeline: A text generation pipeline using the specified model.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # Create a text generation pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    return pipe