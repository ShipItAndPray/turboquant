<p align="center">
  <pre align="center">
 ████████╗██╗   ██╗██████╗ ██████╗  ██████╗  ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗
 ╚══██╔══╝██║   ██║██╔══██╗██╔══██╗██╔═══██╗██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝
    ██║   ██║   ██║██████╔╝██████╔╝██║   ██║██║   ██║██║   ██║███████║██╔██╗ ██║   ██║
    ██║   ██║   ██║██╔══██╗██╔══██╗██║   ██║██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║
    ██║   ╚██████╔╝██║  ██║██████╔╝╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║
    ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝
  </pre>
</p>

<h3 align="center">Compress Any LLM Up to 6x in One Command</h3>

<p align="center">
  <strong>No GPU provisioning. No format headaches. No guesswork.</strong>
</p>

<p align="center">
  <a href="#install">Install</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#formats">Formats</a> &bull;
  <a href="#how-it-works">How It Works</a>
</p>

---

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 4
```

That's it. Your 16GB model is now 4GB. Ship it to ollama, vLLM, or llama.cpp.

## The Problem

Quantizing a model today looks like this:

```
Step 1: Google "how to quantize llama gguf"
Step 2: Clone llama.cpp, build from source, pray
Step 3: Find out you also need AutoGPTQ for GPU serving
Step 4: Conflicting CUDA versions break everything
Step 5: Write a custom script for each format
Step 6: Give up and download someone else's quant from HF
```

TurboQuant replaces all of that with one command.

## What It Does

| Input | Output | Compression |
|-------|--------|-------------|
| 16GB Llama-3.1-8B (FP16) | **4.6GB** GGUF Q4_K_M | 3.5x |
| 140GB Llama-3.1-70B (FP16) | **40GB** GGUF Q4_K_M | 3.5x |
| 1.1GB GPT-2 (FP16) | **300MB** GGUF Q4_K_M | 3.7x |
| Any HuggingFace model | GGUF, GPTQ, or AWQ | up to 8x |

## Install

```bash
pip install turboquant

# With all backends:
pip install turboquant[all]

# Or pick what you need:
pip install turboquant[gguf]    # GGUF — for ollama, LM Studio, llama.cpp
pip install turboquant[gptq]    # GPTQ — for vLLM, TGI GPU serving
pip install turboquant[awq]     # AWQ  — for fast GPU inference
```

## Quick Start

```bash
# Compress to GGUF 4-bit (most common — works with ollama/llama.cpp)
turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 4

# Compress to GPTQ 4-bit (best for GPU inference with vLLM)
turboquant meta-llama/Llama-3.1-8B-Instruct --format gptq --bits 4

# Compress to AWQ 4-bit (fast GPU inference)
turboquant meta-llama/Llama-3.1-8B-Instruct --format awq --bits 4

# Compare ALL formats side-by-side
turboquant meta-llama/Llama-3.1-8B-Instruct --format all --bits 4

# Inspect any model before compressing
turboquant openai-community/gpt2 --info

# Check which backends you have installed
turboquant --check
```

### Example: `--info` Output

```
$ turboquant TinyLlama/TinyLlama-1.1B-Chat-v1.0 --info

  Fetching model info: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  Architecture: LlamaForCausalLM
  Parameters:   1.2B
  Size:         2.2 GB (estimated)
  Est. output:  559.2 MB (4.0x compression)
```

### Example: `--check` Output

```
$ turboquant --check

  Available backends:
    PyTorch:      YES
    CUDA GPU:     YES (NVIDIA A100, 80.0GB)
    Transformers: YES
    GGUF:         YES
    GPTQ:         YES
    AWQ:          YES
```

## Formats

### Which format should I use?

| Format | Best For | Inference Engine | GPU Required? |
|--------|----------|-----------------|---------------|
| **GGUF** | Local/CPU inference, ollama, LM Studio | llama.cpp | No |
| **GPTQ** | GPU serving, high throughput | vLLM, TGI | Yes |
| **AWQ** | GPU serving, fast inference | vLLM, TGI | Yes |

**Don't know? Use GGUF.** It works everywhere.

**Running `--format all`** compresses to every format and gives you a side-by-side comparison report so you can pick the best one for your use case.

### Compression Levels

| Bits | Compression | Quality | Use Case |
|------|-------------|---------|----------|
| **2** | 8x | Degraded | Experimentation only |
| **3** | 5.3x | Decent | Tight memory budgets |
| **4** | **4x** | **Near-lossless** | **Default. Best balance.** |
| **5** | 3.2x | Excellent | Quality-sensitive apps |
| **8** | 2x | Lossless | Maximum quality |

## How It Works

```
                    ┌─────────────────────────────────┐
                    │   turboquant model --format X   │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  1. Fetch model from HuggingFace │
                    │     (or read local files)        │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                     ▼
     ┌────────────────┐  ┌────────────────┐   ┌────────────────┐
     │  llama.cpp      │  │  AutoGPTQ      │   │  AutoAWQ       │
     │  (GGUF)         │  │  (GPTQ)        │   │  (AWQ)         │
     │                 │  │                │   │                │
     │  CPU-friendly   │  │  Calibration   │   │  Activation-   │
     │  K-quant types  │  │  128 samples   │   │  aware quant   │
     └───────┬─────────┘  └───────┬────────┘   └───────┬────────┘
              │                    │                     │
              └────────────────────┼────────────────────┘
                                   ▼
                    ┌──────────────────────────────────┐
                    │  Compressed model + JSON report   │
                    └──────────────────────────────────┘
```

Under the hood, TurboQuant wraps **llama.cpp**, **AutoGPTQ**, and **AutoAWQ** with:
- A unified CLI interface across all three backends
- Sensible defaults (Q4_K_M for GGUF, group_size=128 for GPTQ/AWQ)
- Automatic calibration data (C4 dataset, 128 samples) for GPTQ
- A compression report saved as JSON for comparison

## Supported Architectures

Works with any HuggingFace model, including:

- **LLaMA** (1, 2, 3, 3.1, 3.2, 3.3) — `meta-llama/*`
- **Mistral / Mixtral** — `mistralai/*`
- **Qwen** (1.5, 2, 2.5) — `Qwen/*`
- **Phi** (1, 2, 3, 4) — `microsoft/phi-*`
- **GPT-2 / GPT-J / GPT-NeoX** — `openai-community/gpt2`, `EleutherAI/*`
- **Gemma** — `google/gemma-*`
- **DeepSeek** — `deepseek-ai/*`
- **TinyLlama** — `TinyLlama/*`
- And anything else on HuggingFace with `.safetensors` or `.bin` weights

## Requirements

- Python 3.9+
- PyTorch 2.0+
- **GGUF**: CPU works fine. Install `llama-cpp-python`.
- **GPTQ/AWQ**: GPU recommended. Install `auto-gptq` or `autoawq`.
- Enough RAM to load the model once (e.g., ~16GB for an 8B model)

## Common Workflows

### Compress a model for ollama

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 4
# Then: ollama create mymodel -f ./turboquant-output/Modelfile
```

### Compare all formats before choosing

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct --format all --bits 4
# Check turboquant-output/turboquant-report.json for the comparison
```

### Extreme compression for edge devices

```bash
turboquant TinyLlama/TinyLlama-1.1B-Chat-v1.0 --format gguf --bits 2
# 1.1B model → ~140MB
```

### Quantize a local model (not on HuggingFace)

```bash
turboquant ./my-finetuned-model --format gguf --bits 4
```

## License

MIT
