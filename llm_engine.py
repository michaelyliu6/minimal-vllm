"""
Minimal LLM Engine - A simple implementation for running inference with Hugging Face models.
Includes basic performance benchmarking with KV cache comparison.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

@dataclass
class Request:
    """Represents a single inference request."""
    request_id: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    use_cache: bool = True  # Whether to use KV cache

@dataclass
class RequestOutput:
    """Represents the output of a processed request."""
    request_id: str
    generated_text: str
    finish_reason: str
    usage: Dict[str, int]
    latency: float  # seconds
    tokens_per_second: float
    used_cache: bool

class LLMEngine:
    """
    A minimal LLM engine that can load and run inference with Hugging Face models.
    Includes basic performance benchmarking with KV cache comparison.
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
        
        # Set pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Request management
        self.requests: List[Request] = []
    
    def add_request(self, request: Request) -> None:
        """Add a new request to the queue."""
        self.requests.append(request)
    
    def process_requests(self) -> List[RequestOutput]:
        """Process all pending requests in the queue. Returns outputs with timing info."""
        if not self.requests:
            return []
        
        outputs = []
        total_tokens = 0
        total_time = 0.0
        
        for request in self.requests:
            # Tokenize the prompt
            inputs = self.tokenizer(request.prompt, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
            
            start_time = time.perf_counter()
            
            # Use the model's generation with or without KV caching
            with torch.no_grad():
                generated_output = self.model.generate(
                    input_ids,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=request.use_cache,  # Enable/disable KV caching based on request
                    return_dict_in_generate=True,
                    output_scores=False
                )
            
            end_time = time.perf_counter()
            latency = end_time - start_time
            
            # Extract generated tokens
            generated_ids = generated_output.sequences[0]
            completion_tokens = len(generated_ids) - len(input_ids[0])
            tokens_per_second = completion_tokens / latency if latency > 0 else float('inf')
            total_tokens += completion_tokens
            total_time += latency
            
            # Decode the generated tokens
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Create output
            output = RequestOutput(
                request_id=request.request_id,
                generated_text=generated_text,
                finish_reason="length",  # Simplified for now
                usage={
                    "prompt_tokens": len(input_ids[0]),
                    "completion_tokens": completion_tokens,
                    "total_tokens": len(generated_ids)
                },
                latency=latency,
                tokens_per_second=tokens_per_second,
                used_cache=request.use_cache
            )
            outputs.append(output)
        
        # Print benchmarking summary
        print("\n--- Benchmarking Summary ---")
        print(f"Total requests: {len(outputs)}")
        
        # Separate stats for cached and non-cached requests
        cached_outputs = [o for o in outputs if o.used_cache]
        non_cached_outputs = [o for o in outputs if not o.used_cache]
        
        if cached_outputs:
            cached_time = sum(o.latency for o in cached_outputs)
            cached_tokens = sum(o.usage["completion_tokens"] for o in cached_outputs)
            print("\nWith KV Cache:")
            print(f"Total time: {cached_time:.4f} seconds")
            print(f"Total tokens: {cached_tokens}")
            print(f"Average tokens/sec: {cached_tokens/cached_time:.2f}")
        
        if non_cached_outputs:
            non_cached_time = sum(o.latency for o in non_cached_outputs)
            non_cached_tokens = sum(o.usage["completion_tokens"] for o in non_cached_outputs)
            print("\nWithout KV Cache:")
            print(f"Total time: {non_cached_time:.4f} seconds")
            print(f"Total tokens: {non_cached_tokens}")
            print(f"Average tokens/sec: {non_cached_tokens/non_cached_time:.2f}")
        
        if cached_outputs and non_cached_outputs:
            speedup = non_cached_time / cached_time
            print(f"\nKV Cache Speedup: {speedup:.2f}x")
        
        print("---------------------------\n")
        
        # Clear the request queue
        self.requests.clear()
        return outputs

def main():
    # Initialize the engine with a small model
    engine = LLMEngine(model_name="gpt2")
    
    # Create two identical test requests, one with cache and one without
    prompt = "Once upon a time in a magical forest, there lived a wise old owl"
    
    # First request with KV cache
    request_with_cache = Request(
        request_id="test_with_cache",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        use_cache=True
    )
    
    # Second request without KV cache
    request_without_cache = Request(
        request_id="test_without_cache",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        use_cache=False
    )
    
    # Add and process both requests
    engine.add_request(request_with_cache)
    engine.add_request(request_without_cache)
    outputs = engine.process_requests()
    
    # Print detailed results for each request
    for output in outputs:
        cache_status = "with" if output.used_cache else "without"
        print(f"\nRequest {output.request_id} ({cache_status} KV cache):")
        print("Generated text:")
        print(output.generated_text)
        print("\nToken usage:", output.usage)
        print(f"Latency: {output.latency:.4f} seconds")
        print(f"Tokens/sec: {output.tokens_per_second:.2f}")

if __name__ == "__main__":
    main()
