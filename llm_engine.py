"""
Minimal LLM Engine - A simple implementation for running inference with Hugging Face models.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class Request:
    """Represents a single inference request."""
    request_id: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0

@dataclass
class RequestOutput:
    """Represents the output of a processed request."""
    request_id: str
    generated_text: str
    finish_reason: str
    usage: Dict[str, int]

class LLMEngine:
    """
    A minimal LLM engine that can load and run inference with Hugging Face models.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",  # Using a small model by default
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name = model_name
        self.device = device
        
        # Initialize model and tokenizer
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        print("Model loaded successfully!")
        
        # Request management
        self.requests: List[Request] = []
    
    def add_request(self, request: Request) -> None:
        """Add a new request to the queue."""
        self.requests.append(request)
    
    def process_requests(self) -> List[RequestOutput]:
        """Process all pending requests in the queue."""
        outputs = []
        
        for request in self.requests:
            # Tokenize the prompt
            inputs = self.tokenizer(request.prompt, return_tensors="pt").to(self.device)
            
            # Generate tokens
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the generated tokens
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Create output
            output = RequestOutput(
                request_id=request.request_id,
                generated_text=generated_text,
                finish_reason="length",  # Simplified for now
                usage={
                    "prompt_tokens": len(inputs["input_ids"][0]),
                    "completion_tokens": len(output[0]) - len(inputs["input_ids"][0]),
                    "total_tokens": len(output[0])
                }
            )
            outputs.append(output)
        
        # Clear the request queue
        self.requests.clear()
        return outputs

def main():
    # Initialize the engine with a small model
    engine = LLMEngine(model_name="gpt2")
    
    # Create a test request
    request = Request(
        request_id="test_1",
        prompt="Once upon a time",
        max_tokens=50,
        temperature=0.7
    )
    
    # Add and process the request
    engine.add_request(request)
    outputs = engine.process_requests()
    
    # Print the result
    for output in outputs:
        print("\nGenerated text:")
        print(output.generated_text)
        print("\nToken usage:", output.usage)

if __name__ == "__main__":
    main()
