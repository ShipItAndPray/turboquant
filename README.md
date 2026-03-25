# TurboQuant

**Compress Any LLM Up to 6x in One Command.**

No GPU provisioning. No format headaches. No guesswork.

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 4
```

That's it. Your 16GB model is now 4GB.

## What It Does

| Input | Output |
|-------|--------|
| 16GB Llama-3.1-8B (FP16) | 4.6GB GGUF Q4_K_M |
| 140GB Llama-3.1-70B (FP16) | 40GB GGUF Q4_K_M |
| Any HuggingFace model | GGUF, GPTQ, or AWQ format |

## Why

Quantizing a model today requires:
1. Installing 3 different tools with conflicting dependencies
2. Provisioning a GPU machine
3. Writing custom scripts for each format
4. Guessing which format/bits to use
5. No quality comparison

TurboQuant does it in one command. Pick a format or run `--format all` to compare.

## Install

```bash
pip install turboquant

# With all backends:
pip install turboquant[all]

# Or pick what you need:
pip install turboquant[gguf]    # GGUF only
pip install turboquant[gptq]    # GPTQ only
pip install turboquant[awq]     # AWQ only
```

## Usage

```bash
# Compress to GGUF 4-bit (most common, works with ollama/llama.cpp)
turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 4

# Compress to GPTQ 4-bit (best for GPU inference with vLLM)
turboquant meta-llama/Llama-3.1-8B-Instruct --format gptq --bits 4

# Compress to AWQ 4-bit (fast GPU inference)
turboquant meta-llama/Llama-3.1-8B-Instruct --format awq --bits 4

# Compare ALL formats (GGUF + GPTQ + AWQ)
turboquant meta-llama/Llama-3.1-8B-Instruct --format all --bits 4

# Just check model info
turboquant meta-llama/Llama-3.1-8B-Instruct --info

# Check which backends are installed
turboquant --check any-model

# Extreme compression (2-bit)
turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 2

# High quality (8-bit)
turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 8
```

## Compression Ratios

| Bits | Compression vs FP16 | Quality | Use Case |
|------|---------------------|---------|----------|
| 2 | 8x | Degraded | Experimentation |
| 3 | 5.3x | Good for small models | Tight memory |
| 4 | 4x | Near-lossless | **Default — best balance** |
| 5 | 3.2x | Excellent | Quality-sensitive apps |
| 8 | 2x | Lossless | Maximum quality |

## Output Formats

| Format | Best For | Inference Engine |
|--------|----------|-----------------|
| **GGUF** | Local/CPU, ollama, LM Studio | llama.cpp |
| **GPTQ** | GPU serving, high throughput | vLLM, TGI |
| **AWQ** | GPU serving, fast inference | vLLM, TGI |

## How It Works

1. Downloads model from HuggingFace (or reads local files)
2. Applies the selected quantization method
3. Saves compressed model + quality report
4. That's it

Under the hood: wraps llama.cpp (GGUF), AutoGPTQ, and AutoAWQ with a unified interface, sensible defaults, and automatic calibration data.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- GPU recommended for GPTQ/AWQ (CPU works for GGUF)
- Enough RAM to load the model once

## License

MIT
