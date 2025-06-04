# minimal-vllm

A minimal implementation of an LLM inference engine inspired by vLLM, designed for educational purposes. This project aims to provide a simplified but functional version of a modern LLM inference engine to help developers understand the core concepts and implementation details of large language model serving.

## Project Goals

- Create a minimal but functional LLM inference engine
- Implement core features of modern LLM serving:
  - ✅ Basic model loading and inference
  - ✅ KV cache management (using built-in transformer caching)
  - ⏳ Continuous batching
  - ⏳ Paged attention
- Provide clear, well-documented code that serves as a learning resource
- Focus on understanding the fundamental concepts rather than production-level optimization

## Current Implementation Status

### ✅ Completed Features

1. **Basic LLM Engine** (`llm_engine.py`)
   - Model loading using Hugging Face transformers
   - Request queue management
   - Text generation with configurable parameters (temperature, max_tokens)
   - Support for GPT-2 and other autoregressive models

2. **Performance Benchmarking**
   - Request-level timing (latency, tokens/second)
   - Batch-level performance summaries
   - KV cache vs no-cache performance comparison
   - Detailed metrics for optimization tracking

3. **KV Cache Integration**
   - Uses transformer's built-in KV caching (`use_cache=True/False`)
   - Benchmarking shows performance improvements with caching enabled
   - Foundation for more advanced cache optimizations

4. **Basic Infrastructure**
   - Request/Response data structures
   - Error handling and device management
   - Extensible design for adding features

### 📁 Project Structure

```
minimal-vllm/
├── llm_engine.py          # Main LLM engine implementation
├── kv_cache.py           # Custom KV cache (reference implementation)
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Basic Example**
   ```bash
   python llm_engine.py
   ```
   
   This will run two identical requests (one with KV cache, one without) and show performance comparison.

## Current Performance

The current implementation includes benchmarking that shows:
- Request latency (seconds)
- Tokens generated per second
- KV cache speedup comparison
- Batch processing statistics

Example output includes metrics like:
```
With KV Cache:
Total time: 2.3456 seconds
Average tokens/sec: 42.15

Without KV Cache:
Total time: 4.7892 seconds
Average tokens/sec: 20.65

KV Cache Speedup: 2.04x
```

## Next Steps & Development Agenda

### 🎯 Immediate Next Steps

1. **Custom KV Cache Implementation**
   - Implement our own KV cache from scratch (building on `kv_cache.py`)
   - Add memory management and cache eviction strategies
   - Compare performance with built-in transformer caching

2. **Continuous Batching**
   - Implement dynamic batch processing
   - Handle requests with different lengths efficiently
   - Add request scheduling and prioritization

3. **Attention Optimizations**
   - Implement basic paged attention concepts
   - Explore memory-efficient attention patterns
   - Add attention profiling and optimization

### 🔮 Future Enhancements

4. **Advanced Features**
   - Support for different model architectures
   - Streaming response generation
   - Request cancellation and timeout handling
   - Memory usage monitoring and optimization

5. **Production Features**
   - Basic HTTP API server
   - Request rate limiting
   - Model warm-up and preloading
   - Configuration management

6. **Educational Enhancements**
   - Detailed code documentation and tutorials
   - Performance visualization tools
   - Comparison benchmarks with other engines
   - Interactive examples and notebooks

## Learning Outcomes So Far

Through this implementation, you've learned about:
- LLM inference pipeline basics
- KV cache fundamentals and performance impact
- Request/response handling in inference engines
- Performance benchmarking and optimization measurement
- Integration with Hugging Face transformer models

## Key Files to Explore

- **`llm_engine.py`** - Main engine with benchmarking
- **`kv_cache.py`** - Custom cache implementation (for future use)
- **`requirements.txt`** - Project dependencies

## Contributing & Development

This is a learning project focused on understanding LLM inference engines. As you continue development:

1. **Measure first** - Use the benchmarking system to quantify improvements
2. **Start simple** - Implement basic versions before optimizing
3. **Document learnings** - Update this README with insights and discoveries
4. **Compare approaches** - Test different implementations and measure trade-offs

## License

MIT License - feel free to use this code for learning and educational purposes.

## Acknowledgments

This project is inspired by [vLLM](https://github.com/vllm-project/vllm) and other open-source LLM serving projects. The goal is not to replicate their full functionality, but to create a minimal implementation that helps understand the core concepts.
