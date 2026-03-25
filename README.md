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
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#target-platforms">Target Platforms</a> &bull;
  <a href="#publish-to-huggingface">Publish</a> &bull;
  <a href="#quality-evaluation">Eval</a> &bull;
  <a href="#smart-recommendations">Recommend</a> &bull;
  <a href="#github-action">CI/CD</a>
</p>

---

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 4
```

That's it. Your 16GB model is now 4GB. Ship it to Ollama, vLLM, or llama.cpp.

## The Problem

Quantizing a model today looks like this:

```
Step 1: Google "how to quantize llama gguf"
Step 2: Clone llama.cpp, build from source, pray
Step 3: Find out you also need AutoGPTQ for GPU serving
Step 4: Conflicting CUDA versions break everything
Step 5: Write a custom script for each format
Step 6: No way to tell if your quant is actually good
Step 7: Give up and download someone else's quant from HF
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
pip install turboquant[gguf]    # GGUF — for Ollama, LM Studio, llama.cpp
pip install turboquant[gptq]    # GPTQ — for vLLM, TGI GPU serving
pip install turboquant[awq]     # AWQ  — for fast GPU inference
```

## Quick Start

```bash
# Compress to GGUF 4-bit (most common — works with Ollama/llama.cpp)
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

---

## Target Platforms

**Don't know which format to use?** Just tell TurboQuant where you want to run it.

### Ollama (one command, ready to run)

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct --target ollama --bits 4
```

This:
1. Quantizes to GGUF (required by Ollama)
2. Auto-generates a `Modelfile` with the correct chat template (LLaMA, Mistral, Qwen, Phi, Gemma)
3. Tells you the exact `ollama create` command to run

Output:
```
  READY FOR OLLAMA
  ========================================================

  GGUF file:  ./turboquant-output/model-Q4_K_M.gguf
  Modelfile:  ./turboquant-output/Modelfile

  To import into Ollama, run:

    cd ./turboquant-output
    ollama create llama-3-1-8b-instruct-q4_k_m -f Modelfile

  Then run it:

    ollama run llama-3-1-8b-instruct-q4_k_m
```

### vLLM

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct --target vllm --bits 4
```

Auto-selects AWQ (best GPU throughput for vLLM).

### LM Studio / llama.cpp

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct --target lmstudio --bits 4
turboquant meta-llama/Llama-3.1-8B-Instruct --target llamacpp --bits 4
```

---

## Publish to HuggingFace

**Be the next bartowski.** Quantize any model and publish it to HuggingFace Hub in one command.

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct \
  --format gguf --bits 4 \
  --push-to-hub yourname/Llama-3.1-8B-Instruct-GGUF
```

This:
1. Quantizes the model
2. Generates a model card with base model info, quant method, usage instructions
3. Uploads everything to your HuggingFace repo
4. Tags the repo with `quantized`, `turboquant`, bit depth

Publish all formats at once:

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct \
  --format all --bits 4 \
  --push-to-hub yourname/Llama-3.1-8B-Instruct-Quantized
```

**Requires:** `huggingface-cli login` or `HF_TOKEN` environment variable.

---

## Quality Evaluation

**No more "quantize and pray."** TurboQuant can evaluate your quantized model's quality automatically.

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 4 --eval
```

Output:
```
  QUALITY EVALUATION
  --------------------------------------------------------

  Perplexity:  7.42
  Method:      llama-cpp-python
  Tokens:      847

  Quality:     EXCELLENT
  Assessment:  Minimal quality loss from quantization.
```

Quality grades:
| Perplexity | Grade | Meaning |
|------------|-------|---------|
| < 10 | EXCELLENT | Minimal quality loss |
| 10-20 | GOOD | Acceptable for most use cases |
| 20-50 | FAIR | Some degradation, consider higher bits |
| 50-100 | DEGRADED | Significant loss, use more bits |
| > 100 | POOR | Model may be broken |

---

## Smart Recommendations

**Don't know what format or bits to use?** TurboQuant detects your hardware and tells you.

```bash
turboquant meta-llama/Llama-3.1-8B-Instruct --recommend
```

Output on an M1 Mac:
```
  TURBOQUANT FORMAT RECOMMENDATION
  ========================================================

  Hardware Detected:
    GPU:  Apple Silicon (MPS) — 16.0GB unified memory
    RAM:  16.0GB

  Model:
    Parameters: 8.0B
    FP16 Size:  16.1 GB

  Recommendations:

  [BEST] GGUF 4-bit
    Why:  Best format for Apple Silicon. llama.cpp has Metal acceleration.
    For:  Ollama or LM Studio on Mac
    Run:  turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 4
```

Output on an A100 GPU:
```
  Hardware Detected:
    GPU:  NVIDIA A100 (80.0GB VRAM)
    RAM:  256.0GB

  [BEST] AWQ 4-bit
    Why:  Best GPU throughput. 4-bit model (~4.0GB) fits in 80.0GB VRAM.
    For:  Production GPU serving with vLLM or TGI

  [ALSO GOOD] GPTQ 4-bit
    Why:  Alternative GPU format. Wider tool support than AWQ.

  [ALTERNATIVE] GGUF 4-bit
    Why:  Universal format. Works with Ollama, LM Studio, llama.cpp.
```

---

## GitHub Action

**First-ever CI/CD pipeline for LLM quantization.** Auto-quantize after fine-tuning.

### Basic Usage

```yaml
# .github/workflows/quantize.yml
name: Quantize Model
on:
  workflow_dispatch:
    inputs:
      model:
        description: 'Model to quantize'
        required: true

jobs:
  quantize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ShipItAndPray/turboquant@master
        with:
          model: ${{ inputs.model }}
          format: gguf
          bits: 4
          eval: true
          push-to-hub: yourname/model-GGUF
          hf-token: ${{ secrets.HF_TOKEN }}
```

### Auto-Quantize After Fine-Tuning

```yaml
name: Fine-Tune and Quantize
on:
  push:
    paths: ['training/**']

jobs:
  train:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - run: python training/finetune.py --output ./finetuned-model

      - uses: ShipItAndPray/turboquant@master
        with:
          model: ./finetuned-model
          target: ollama
          bits: 4
          push-to-hub: yourname/my-finetuned-model-GGUF
          hf-token: ${{ secrets.HF_TOKEN }}
```

### Action Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `model` | Yes | — | HuggingFace model ID or local path |
| `format` | No | `gguf` | `gguf`, `gptq`, `awq`, or `all` |
| `bits` | No | `4` | `2`, `3`, `4`, `5`, or `8` |
| `target` | No | — | `ollama`, `vllm`, `llamacpp`, `lmstudio` |
| `push-to-hub` | No | — | HuggingFace repo to upload to |
| `eval` | No | `false` | Run quality evaluation |
| `hf-token` | No | — | HuggingFace API token |
| `output` | No | `./turboquant-output` | Output directory |

### Action Outputs

| Output | Description |
|--------|-------------|
| `output-dir` | Directory with quantized model files |
| `report` | Path to JSON compression report |

---

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
                    │  + Modelfile (Ollama)             │
                    │  + Model card (HuggingFace)       │
                    │  + Quality eval (--eval)          │
                    └──────────────────────────────────┘
```

## Formats

### Which format should I use?

| Format | Best For | Inference Engine | GPU Required? |
|--------|----------|-----------------|---------------|
| **GGUF** | Local/CPU inference, Ollama, LM Studio | llama.cpp | No |
| **GPTQ** | GPU serving, high throughput | vLLM, TGI | Yes |
| **AWQ** | GPU serving, fast inference | vLLM, TGI | Yes |

**Don't know?** Run `turboquant your-model --recommend` and let TurboQuant tell you.

### Compression Levels

| Bits | Compression | Quality | Use Case |
|------|-------------|---------|----------|
| **2** | 8x | Degraded | Experimentation only |
| **3** | 5.3x | Decent | Tight memory budgets |
| **4** | **4x** | **Near-lossless** | **Default. Best balance.** |
| **5** | 3.2x | Excellent | Quality-sensitive apps |
| **8** | 2x | Lossless | Maximum quality |

## Supported Architectures

Works with any HuggingFace model, including:

- **LLaMA** (1, 2, 3, 3.1, 3.2, 3.3) -- `meta-llama/*`
- **Mistral / Mixtral** -- `mistralai/*`
- **Qwen** (1.5, 2, 2.5) -- `Qwen/*`
- **Phi** (1, 2, 3, 4) -- `microsoft/phi-*`
- **GPT-2 / GPT-J / GPT-NeoX** -- `openai-community/gpt2`, `EleutherAI/*`
- **Gemma** -- `google/gemma-*`
- **DeepSeek** -- `deepseek-ai/*`
- And anything else on HuggingFace with `.safetensors` or `.bin` weights

## Requirements

- Python 3.9+
- PyTorch 2.0+
- **GGUF**: CPU works fine. Install `llama-cpp-python`.
- **GPTQ/AWQ**: GPU recommended. Install `auto-gptq` or `autoawq`.
- Enough RAM to load the model once (e.g., ~16GB for an 8B model)

## All CLI Options

```
turboquant MODEL [OPTIONS]

Positional:
  MODEL                     HuggingFace model ID or local path

Formats:
  --format, -f FORMAT       gguf, gptq, awq, or all (default: gguf)
  --bits, -b BITS           2, 3, 4, 5, or 8 (default: 4)
  --output, -o DIR          Output directory (default: ./turboquant-output)

Target Platforms:
  --target, -t TARGET       ollama, vllm, llamacpp, lmstudio
                            Auto-selects format + generates platform-specific files

Publishing:
  --push-to-hub REPO        Upload to HuggingFace Hub (e.g. user/model-GGUF)

Quality:
  --eval                    Run perplexity evaluation after quantization
  --recommend               Show hardware-aware format recommendation

Info:
  --info                    Show model details without quantizing
  --check                   Show available backends and hardware
```

## License

MIT
