"""
Minimal LLM Engine - A simple implementation for running inference with Hugging Face models.
Includes basic performance benchmarking with KV cache comparison.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_cache import KVCache
from custom_attention import replace_attention_layers
import time

@dataclass
class Request:
    """Represents a single inference request."""
    request_id: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    use_cache: bool = True  # Whether to use KV cache
    use_custom_cache: bool = False  # Whether to use our custom KV cache

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
    used_custom_cache: bool

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
        
        # Initialize our custom KV cache
        self.custom_kv_cache = KVCache(
            num_layers=self.model.config.n_layer,
            device=device
        )
        
        # Create a version of the model with our custom attention
        self.custom_model = None
        self._setup_custom_model()
        
        # Request management
        self.requests: List[Request] = []
    
    def _setup_custom_model(self):
        """Create a copy of the model with our custom attention layers."""
        import copy
        self.custom_model = copy.deepcopy(self.model)
        self.custom_model = replace_attention_layers(self.custom_model, self.custom_kv_cache)
        print("Custom attention model created!")
    
    def add_request(self, request: Request) -> None:
        """Add a new request to the queue."""
        self.requests.append(request)
    
    def _generate_with_custom_cache(self, input_ids: torch.Tensor, request: Request) -> torch.Tensor:
        """Generate tokens using our custom KV cache implementation."""
        batch_size, initial_seq_len = input_ids.shape
        
        # Clear and prepare our custom KV cache for this generation
        self.custom_kv_cache.clear()
        
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            # FIRST PASS: Process the full prompt to populate cache
            outputs = self.custom_model(
                input_ids,  # Process only the original prompt
                use_cache=True,
                return_dict=True
            )
            
            # INCREMENTAL GENERATION: Process one token at a time
            for step in range(request.max_tokens):
                # Get next token logits from the last position
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / request.temperature
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for end of sequence
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # For next iteration, process ONLY the new token
                if step < request.max_tokens - 1:  # Don't run on last iteration
                    outputs = self.custom_model(
                        next_token,  # Only process the new token!
                        use_cache=True,
                        return_dict=True
                    )
        
        return generated_ids
    
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
            
            if request.use_custom_cache:
                # Use our custom KV cache implementation
                generated_ids = self._generate_with_custom_cache(input_ids, request)
            else:
                # Use HF's built-in generation with or without KV caching
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
                    generated_ids = generated_output.sequences[0]
            
            end_time = time.perf_counter()
            latency = end_time - start_time
            
            # Calculate token counts
            if request.use_custom_cache:
                completion_tokens = len(generated_ids[0]) - len(input_ids[0])
            else:
                completion_tokens = len(generated_ids) - len(input_ids[0])
            
            tokens_per_second = completion_tokens / latency if latency > 0 else float('inf')
            total_tokens += completion_tokens
            total_time += latency
            
            # Decode the generated tokens
            if request.use_custom_cache:
                generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            else:
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Create output
            output = RequestOutput(
                request_id=request.request_id,
                generated_text=generated_text,
                finish_reason="length",  # Simplified for now
                usage={
                    "prompt_tokens": len(input_ids[0]),
                    "completion_tokens": completion_tokens,
                    "total_tokens": len(input_ids[0]) + completion_tokens
                },
                latency=latency,
                tokens_per_second=tokens_per_second,
                used_cache=request.use_cache,
                used_custom_cache=request.use_custom_cache
            )
            outputs.append(output)
        
        # Print benchmarking summary
        print("\n--- Benchmarking Summary ---")
        print(f"Total requests: {len(outputs)}")
        
        # Separate stats for different cache types
        hf_cached_outputs = [o for o in outputs if o.used_cache and not o.used_custom_cache]
        custom_cached_outputs = [o for o in outputs if o.used_custom_cache]
        no_cache_outputs = [o for o in outputs if not o.used_cache and not o.used_custom_cache]
        
        if hf_cached_outputs:
            hf_time = sum(o.latency for o in hf_cached_outputs)
            hf_tokens = sum(o.usage["completion_tokens"] for o in hf_cached_outputs)
            print("\nHF Built-in KV Cache:")
            print(f"Total time: {hf_time:.4f} seconds")
            print(f"Total tokens: {hf_tokens}")
            print(f"Average tokens/sec: {hf_tokens/hf_time:.2f}")
        
        if custom_cached_outputs:
            custom_time = sum(o.latency for o in custom_cached_outputs)
            custom_tokens = sum(o.usage["completion_tokens"] for o in custom_cached_outputs)
            print("\nCustom KV Cache:")
            print(f"Total time: {custom_time:.4f} seconds")
            print(f"Total tokens: {custom_tokens}")
            print(f"Average tokens/sec: {custom_tokens/custom_time:.2f}")
        
        if no_cache_outputs:
            no_cache_time = sum(o.latency for o in no_cache_outputs)
            no_cache_tokens = sum(o.usage["completion_tokens"] for o in no_cache_outputs)
            print("\nNo KV Cache:")
            print(f"Total time: {no_cache_time:.4f} seconds")
            print(f"Total tokens: {no_cache_tokens}")
            print(f"Average tokens/sec: {no_cache_tokens/no_cache_time:.2f}")
        
        # Compare speedups
        if custom_cached_outputs and no_cache_outputs:
            speedup = no_cache_time / custom_time
            print(f"\nCustom Cache Speedup vs No Cache: {speedup:.2f}x")
        
        if hf_cached_outputs and custom_cached_outputs:
            comparison = custom_time / hf_time
            if comparison < 1:
                print(f"Custom Cache is {1/comparison:.2f}x faster than HF Cache")
            else:
                print(f"HF Cache is {comparison:.2f}x faster than Custom Cache")
        
        print("---------------------------\n")
        
        # Clear the request queue
        self.requests.clear()
        return outputs

def main():
    # Initialize the engine with a small model
    engine = LLMEngine(model_name="gpt2")
    
    # Create three test requests to compare all implementations
    prompt = "Once upon a time in a magical forest, there lived a wise old owl"
    
    # Request with HF built-in cache
    request_hf_cache = Request(
        request_id="test_hf_cache",
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
        use_cache=True,
        use_custom_cache=False
    )
    
    # Request with our custom cache
    request_custom_cache = Request(
        request_id="test_custom_cache",
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
        use_cache=True,  # This will be ignored when use_custom_cache=True
        use_custom_cache=True
    )
    
    # Request without any cache
    request_no_cache = Request(
        request_id="test_no_cache",
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
        use_cache=False,
        use_custom_cache=False
    )
    
    # Add and process all requests
    engine.add_request(request_hf_cache)
    engine.add_request(request_custom_cache)
    engine.add_request(request_no_cache)
    outputs = engine.process_requests()
    
    # Print detailed results for each request
    for output in outputs:
        cache_type = "Custom" if output.used_custom_cache else ("HF" if output.used_cache else "None")
        print(f"\nRequest {output.request_id} ({cache_type} cache):")
        print("Generated text:")
        print(output.generated_text)
        print("\nToken usage:", output.usage)
        print(f"Latency: {output.latency:.4f} seconds")
        print(f"Tokens/sec: {output.tokens_per_second:.2f}")

if __name__ == "__main__":
    main()
