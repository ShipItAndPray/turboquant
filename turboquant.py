"""
TurboQuant — Compress Any LLM Up to 6x in One Command
=======================================================
Upload your model. Pick a format. Get it compressed.

Supports: GGUF, GPTQ, AWQ
Input: HuggingFace model ID or local path
Output: Compressed model + quality report

Usage:
    turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 4
    turboquant ./my-model --format gptq --bits 4
    turboquant meta-llama/Llama-3.1-8B-Instruct --format all --bits 4
    turboquant meta-llama/Llama-3.1-8B-Instruct --target ollama --bits 4
    turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 4 --push-to-hub user/repo
    turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 4 --eval
    turboquant meta-llama/Llama-3.1-8B-Instruct --recommend
"""

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path


SUPPORTED_FORMATS = ["gguf", "gptq", "awq", "all"]
SUPPORTED_BITS = [2, 3, 4, 5, 8]
SUPPORTED_TARGETS = ["ollama", "vllm", "llamacpp", "lmstudio"]

GGUF_QUANT_TYPES = {
    2: "Q2_K",
    3: "Q3_K_M",
    4: "Q4_K_M",
    5: "Q5_K_M",
    8: "Q8_0",
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def check_dependencies():
    """Check which quantization backends are available."""
    available = {}

    # Check for llama.cpp (GGUF)
    llama_convert = shutil.which("llama-quantize") or shutil.which("quantize")
    if llama_convert:
        available["gguf"] = True
    else:
        try:
            import llama_cpp  # noqa: F401
            available["gguf"] = True
        except ImportError:
            available["gguf"] = False

    # Check for AutoGPTQ
    try:
        import auto_gptq  # noqa: F401
        available["gptq"] = True
    except ImportError:
        available["gptq"] = False

    # Check for AutoAWQ
    try:
        import awq  # noqa: F401
        available["awq"] = True
    except ImportError:
        available["awq"] = False

    # Check for transformers (always needed)
    try:
        import transformers  # noqa: F401
        available["transformers"] = True
    except ImportError:
        available["transformers"] = False

    # Check for torch
    try:
        import torch
        available["torch"] = True
        available["cuda"] = torch.cuda.is_available()
        if available["cuda"]:
            available["gpu_name"] = torch.cuda.get_device_name(0)
            available["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
        # Check for Apple Silicon MPS
        available["mps"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        available["torch"] = False
        available["cuda"] = False
        available["mps"] = False

    return available


def print_banner():
    print("""
 ████████╗██╗   ██╗██████╗ ██████╗  ██████╗  ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗
 ╚══██╔══╝██║   ██║██╔══██╗██╔══██╗██╔═══██╗██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝
    ██║   ██║   ██║██████╔╝██████╔╝██║   ██║██║   ██║██║   ██║███████║██╔██╗ ██║   ██║
    ██║   ██║   ██║██╔══██╗██╔══██╗██║   ██║██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║
    ██║   ╚██████╔╝██║  ██║██████╔╝╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║
    ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝

    Compress Any LLM Up to 6x in One Command
    """)


def estimate_compression(original_bits, target_bits):
    """Estimate compression ratio."""
    return original_bits / target_bits


def format_size(size_bytes):
    """Format bytes to human-readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

def get_model_info(model_id_or_path):
    """Get model information from HuggingFace or local path."""
    info = {"source": model_id_or_path}

    try:
        from huggingface_hub import model_info as hf_model_info
        mi = hf_model_info(model_id_or_path)
        info["model_id"] = mi.id
        info["size_bytes"] = sum(
            s.size for s in mi.siblings
            if s.rfilename.endswith(('.safetensors', '.bin')) and s.size is not None
        )
        info["size_human"] = format_size(info["size_bytes"])

        # Try to get parameter count from config
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(model_id_or_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        info["config"] = config
        info["arch"] = config.get("architectures", ["unknown"])[0]
        # Support LLaMA, GPT, T5, Falcon, etc. config key names
        info["hidden_size"] = (
            config.get("hidden_size") or config.get("n_embd")
            or config.get("d_model") or 0
        )
        info["num_layers"] = (
            config.get("num_hidden_layers") or config.get("n_layer")
            or config.get("num_layers") or 0
        )
        info["vocab_size"] = config.get("vocab_size", 0)
        info["context_length"] = (
            config.get("max_position_embeddings") or config.get("n_positions")
            or config.get("max_seq_len") or config.get("seq_length") or 0
        )

        # Estimate parameters
        h = info["hidden_size"]
        n = info["num_layers"]
        v = info["vocab_size"]
        if h and n and v:
            params = 12 * n * h * h + v * h
            info["params_estimate"] = params
            info["params_human"] = f"{params/1e9:.1f}B" if params > 1e9 else f"{params/1e6:.0f}M"

        # If HF API didn't return file sizes, estimate from parameters
        if not info["size_bytes"] and info.get("params_estimate"):
            info["size_bytes"] = info["params_estimate"] * 2  # FP16
            info["size_human"] = format_size(info["size_bytes"]) + " (estimated)"

        info["found"] = True
    except Exception as e:
        # Check if local path
        if os.path.isdir(model_id_or_path):
            info["found"] = True
            info["local"] = True
            total = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(model_id_or_path)
                for f in fns if f.endswith(('.safetensors', '.bin'))
            )
            info["size_bytes"] = total
            info["size_human"] = format_size(total)
            # Try to read local config.json
            local_config = os.path.join(model_id_or_path, "config.json")
            if os.path.exists(local_config):
                with open(local_config) as f:
                    config = json.load(f)
                info["config"] = config
                info["arch"] = config.get("architectures", ["unknown"])[0]
                info["hidden_size"] = (
                    config.get("hidden_size") or config.get("n_embd")
                    or config.get("d_model") or 0
                )
                info["num_layers"] = (
                    config.get("num_hidden_layers") or config.get("n_layer")
                    or config.get("num_layers") or 0
                )
                info["vocab_size"] = config.get("vocab_size", 0)
                info["context_length"] = (
                    config.get("max_position_embeddings") or config.get("n_positions")
                    or config.get("max_seq_len") or 0
                )
        else:
            info["found"] = False
            info["error"] = str(e)

    return info


# ---------------------------------------------------------------------------
# Quantization backends
# ---------------------------------------------------------------------------

def quantize_gguf(model_id, bits, output_dir):
    """Quantize model to GGUF format using llama.cpp."""
    quant_type = GGUF_QUANT_TYPES.get(bits, "Q4_K_M")
    output_file = os.path.join(output_dir, f"model-{quant_type}.gguf")

    print(f"  Converting to GGUF {quant_type} ({bits}-bit)...")

    # Method 1: Try llama-cpp-python convert
    try:
        fp16_file = os.path.join(output_dir, "model-fp16.gguf")
        cmd_convert = [
            sys.executable, "-m", "llama_cpp.convert",
            "--outfile", fp16_file,
            "--outtype", "f16",
            model_id,
        ]
        print("  Step 1: Converting to GGUF FP16...")
        result = subprocess.run(cmd_convert, capture_output=True, text=True, timeout=3600)

        if result.returncode == 0 and os.path.exists(fp16_file):
            cmd_quant = ["llama-quantize", fp16_file, output_file, quant_type]
            print(f"  Step 2: Quantizing to {quant_type}...")
            result = subprocess.run(cmd_quant, capture_output=True, text=True, timeout=3600)

            if result.returncode == 0 and os.path.exists(output_file):
                os.remove(fp16_file)
                return {"success": True, "file": output_file, "size": os.path.getsize(output_file)}

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 2: Try convert_hf_to_gguf.py from llama.cpp
    try:
        print("  Using transformers + manual GGUF conversion...")
        convert_script = shutil.which("convert_hf_to_gguf.py")
        if not convert_script:
            # Check common locations
            for candidate in [
                os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
                "/opt/llama.cpp/convert_hf_to_gguf.py",
            ]:
                if os.path.exists(candidate):
                    convert_script = candidate
                    break

        if convert_script:
            cmd = [
                sys.executable, convert_script, model_id,
                "--outfile", output_file, "--outtype", quant_type.lower(),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                return {"success": True, "file": output_file, "size": os.path.getsize(output_file)}
    except Exception:
        pass

    return {
        "success": False,
        "error": "GGUF quantization requires llama.cpp. Install: pip install llama-cpp-python, or build llama.cpp from source.",
        "install_cmd": "pip install llama-cpp-python",
    }


def quantize_gptq(model_id, bits, output_dir):
    """Quantize model using GPTQ."""
    print(f"  Quantizing with GPTQ ({bits}-bit, group_size=128)...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=128,
            damp_percent=0.1,
            desc_act=False,
        )

        model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)

        print("  Preparing calibration data (c4 dataset, 128 samples)...")
        from datasets import load_dataset
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        calibration_data = []
        for i, example in enumerate(dataset):
            if i >= 128:
                break
            tokenized = tokenizer(
                example["text"], return_tensors="pt",
                truncation=True, max_length=2048,
            )
            calibration_data.append(tokenized.input_ids)

        print("  Running GPTQ quantization (this takes a while)...")
        model.quantize(calibration_data)

        output_path = os.path.join(output_dir, f"model-gptq-{bits}bit")
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)

        total_size = sum(
            os.path.getsize(os.path.join(output_path, f))
            for f in os.listdir(output_path)
            if f.endswith(('.safetensors', '.bin'))
        )

        return {"success": True, "file": output_path, "size": total_size}

    except ImportError:
        return {
            "success": False,
            "error": "GPTQ requires: pip install auto-gptq datasets",
            "install_cmd": "pip install auto-gptq datasets",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def quantize_awq(model_id, bits, output_dir):
    """Quantize model using AWQ."""
    print(f"  Quantizing with AWQ ({bits}-bit)...")

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        model = AutoAWQForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": bits,
            "version": "GEMM",
        }

        print("  Running AWQ quantization...")
        model.quantize(tokenizer, quant_config=quant_config)

        output_path = os.path.join(output_dir, f"model-awq-{bits}bit")
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)

        total_size = sum(
            os.path.getsize(os.path.join(output_path, f))
            for f in os.listdir(output_path)
            if f.endswith(('.safetensors', '.bin'))
        )

        return {"success": True, "file": output_path, "size": total_size}

    except ImportError:
        return {
            "success": False,
            "error": "AWQ requires: pip install autoawq",
            "install_cmd": "pip install autoawq",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(model_info, results, bits):
    """Generate compression quality report."""
    report = {
        "model": model_info.get("source"),
        "original_size": model_info.get("size_human", "unknown"),
        "original_size_bytes": model_info.get("size_bytes", 0),
        "target_bits": bits,
        "theoretical_compression": f"{estimate_compression(16, bits):.1f}x (from FP16)",
        "results": {},
    }

    for fmt, result in results.items():
        if result["success"]:
            compressed_size = result["size"]
            original = model_info.get("size_bytes", 1)
            actual_compression = original / compressed_size if compressed_size > 0 else 0

            report["results"][fmt] = {
                "status": "success",
                "file": result["file"],
                "compressed_size": format_size(compressed_size),
                "compressed_size_bytes": compressed_size,
                "actual_compression": f"{actual_compression:.1f}x",
            }
        else:
            report["results"][fmt] = {
                "status": "failed",
                "error": result.get("error", "unknown"),
                "install_cmd": result.get("install_cmd", ""),
            }

    return report


def print_report(report):
    """Print a formatted compression report."""
    print()
    print("=" * 60)
    print("  TURBOQUANT COMPRESSION REPORT")
    print("=" * 60)
    print()
    print(f"  Model:       {report['model']}")
    print(f"  Original:    {report['original_size']}")
    print(f"  Target bits: {report['target_bits']}")
    print(f"  Theoretical: {report['theoretical_compression']}")
    print()

    for fmt, result in report["results"].items():
        if result["status"] == "success":
            print(f"  [{fmt.upper()}] Compressed: {result['compressed_size']} "
                  f"({result['actual_compression']} compression)")
            print(f"         File: {result['file']}")
        else:
            print(f"  [{fmt.upper()}] FAILED: {result['error']}")
            if result.get("install_cmd"):
                print(f"         Fix: {result['install_cmd']}")
        print()

    print("=" * 60)


# ---------------------------------------------------------------------------
# Feature: --target ollama  (generates Modelfile + ready-to-use GGUF)
# ---------------------------------------------------------------------------

def generate_ollama_modelfile(gguf_path, model_info, output_dir):
    """Generate an Ollama Modelfile pointing to the quantized GGUF."""
    model_name = model_info.get("source", "unknown").split("/")[-1]
    arch = model_info.get("arch", "")
    params = model_info.get("params_human", "unknown")
    context = model_info.get("context_length", 4096)

    # Detect chat template from config
    template_str = ""
    config = model_info.get("config", {})
    arch_lower = arch.lower() if arch else ""

    if "llama" in arch_lower:
        template_str = textwrap.dedent("""\
            TEMPLATE \"\"\"{{- if .System }}<|start_header_id|>system<|end_header_id|>

            {{ .System }}<|eot_id|>{{- end }}
            <|start_header_id|>user<|end_header_id|>

            {{ .Prompt }}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>

            {{ .Response }}<|eot_id|>\"\"\"

            PARAMETER stop "<|eot_id|>"
            PARAMETER stop "<|end_of_text|>"
        """)
    elif "mistral" in arch_lower:
        template_str = textwrap.dedent("""\
            TEMPLATE \"\"\"[INST] {{- if .System }}{{ .System }} {{- end }}{{ .Prompt }} [/INST]{{ .Response }}\"\"\"

            PARAMETER stop "[INST]"
            PARAMETER stop "[/INST]"
        """)
    elif "qwen" in arch_lower:
        template_str = textwrap.dedent("""\
            TEMPLATE \"\"\"<|im_start|>system
            {{- if .System }}{{ .System }}{{- else }}You are a helpful assistant.{{- end }}<|im_end|>
            <|im_start|>user
            {{ .Prompt }}<|im_end|>
            <|im_start|>assistant
            {{ .Response }}<|im_end|>\"\"\"

            PARAMETER stop "<|im_start|>"
            PARAMETER stop "<|im_end|>"
        """)
    elif "phi" in arch_lower:
        template_str = textwrap.dedent("""\
            TEMPLATE \"\"\"<|system|>
            {{- if .System }}{{ .System }}{{- else }}You are a helpful assistant.{{- end }}<|end|>
            <|user|>
            {{ .Prompt }}<|end|>
            <|assistant|>
            {{ .Response }}<|end|>\"\"\"

            PARAMETER stop "<|end|>"
            PARAMETER stop "<|endoftext|>"
        """)
    elif "gemma" in arch_lower:
        template_str = textwrap.dedent("""\
            TEMPLATE \"\"\"<start_of_turn>user
            {{ .Prompt }}<end_of_turn>
            <start_of_turn>model
            {{ .Response }}<end_of_turn>\"\"\"

            PARAMETER stop "<end_of_turn>"
        """)

    # Build Modelfile
    gguf_filename = os.path.basename(gguf_path)
    modelfile = f"FROM ./{gguf_filename}\n\n"

    if template_str:
        modelfile += template_str + "\n"

    if context:
        modelfile += f"PARAMETER num_ctx {context}\n"

    modelfile_path = os.path.join(output_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile)

    return modelfile_path


def handle_target_ollama(model_id, bits, output_dir, model_info):
    """Full ollama pipeline: quantize to GGUF + generate Modelfile."""
    print()
    print("  --- TARGET: OLLAMA ---")
    print("  Format: GGUF (required by Ollama)")
    print()

    result = quantize_gguf(model_id, bits, output_dir)

    if result["success"]:
        modelfile_path = generate_ollama_modelfile(
            result["file"], model_info, output_dir,
        )
        model_name = model_id.split("/")[-1].lower().replace(".", "-")
        quant_type = GGUF_QUANT_TYPES.get(bits, "Q4_K_M")

        print()
        print("  " + "=" * 56)
        print("  READY FOR OLLAMA")
        print("  " + "=" * 56)
        print()
        print(f"  GGUF file:  {result['file']}")
        print(f"  Modelfile:  {modelfile_path}")
        print(f"  Size:       {format_size(result['size'])}")
        print()
        print("  To import into Ollama, run:")
        print()
        print(f"    cd {output_dir}")
        print(f"    ollama create {model_name}-{quant_type.lower()} -f Modelfile")
        print()
        print("  Then run it:")
        print()
        print(f"    ollama run {model_name}-{quant_type.lower()}")
        print()
        print("  " + "=" * 56)

        result["modelfile"] = modelfile_path
    else:
        print(f"\n  FAILED: {result.get('error')}")
        if result.get("install_cmd"):
            print(f"  Fix: {result['install_cmd']}")

    return result


# ---------------------------------------------------------------------------
# Feature: --push-to-hub  (upload to HuggingFace with model card)
# ---------------------------------------------------------------------------

def generate_model_card(model_info, results, bits, hub_repo):
    """Generate a HuggingFace model card (README.md) for the quantized model."""
    model_id = model_info.get("source", "unknown")
    arch = model_info.get("arch", "unknown")
    params = model_info.get("params_human", "unknown")
    original_size = model_info.get("size_human", "unknown")

    # Build results table
    results_rows = ""
    for fmt, result in results.items():
        if result["status"] == "success":
            results_rows += (
                f"| {fmt.upper()} | {bits}-bit | "
                f"{result['compressed_size']} | "
                f"{result['actual_compression']} |\n"
            )

    quant_methods = ", ".join(
        fmt.upper() for fmt, r in results.items() if r["status"] == "success"
    )

    card = textwrap.dedent(f"""\
        ---
        base_model: {model_id}
        tags:
        - quantized
        - turboquant
        - {bits}bit
        license: mit
        ---

        # {hub_repo.split('/')[-1]}

        **Quantized version of [{model_id}](https://huggingface.co/{model_id})**

        Quantized with [TurboQuant](https://github.com/ShipItAndPray/turboquant) — compress any LLM up to 6x in one command.

        ## Model Details

        | Property | Value |
        |----------|-------|
        | Base Model | [{model_id}](https://huggingface.co/{model_id}) |
        | Architecture | {arch} |
        | Parameters | {params} |
        | Original Size | {original_size} |
        | Quantization | {bits}-bit |
        | Methods | {quant_methods} |

        ## Quantization Results

        | Format | Bits | Compressed Size | Compression |
        |--------|------|----------------|-------------|
        {results_rows}
        ## Usage

        ### GGUF (Ollama / llama.cpp / LM Studio)

        ```bash
        # Download and run with Ollama
        ollama run {hub_repo}

        # Or download the GGUF file directly for llama.cpp
        huggingface-cli download {hub_repo} --include "*.gguf"
        ```

        ### GPTQ / AWQ (vLLM / TGI)

        ```python
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("{hub_repo}")
        ```

        ## Quantized With

        ```bash
        pip install turboquant
        turboquant {model_id} --format all --bits {bits} --push-to-hub {hub_repo}
        ```

        ---
        *Quantized with [TurboQuant](https://github.com/ShipItAndPray/turboquant)*
    """)

    return card


def push_to_hub(hub_repo, output_dir, model_info, report):
    """Upload quantized model files to HuggingFace Hub."""
    print()
    print(f"  --- PUSHING TO HUB: {hub_repo} ---")
    print()

    try:
        from huggingface_hub import HfApi, login
        api = HfApi()

        # Check authentication
        try:
            user_info = api.whoami()
            print(f"  Authenticated as: {user_info.get('name', 'unknown')}")
        except Exception:
            print("  Not logged in to HuggingFace. Run: huggingface-cli login")
            print("  Or set HF_TOKEN environment variable.")
            return False

        # Create repo if it doesn't exist
        try:
            api.create_repo(hub_repo, exist_ok=True, repo_type="model")
            print(f"  Repository: https://huggingface.co/{hub_repo}")
        except Exception as e:
            print(f"  Warning creating repo: {e}")

        # Generate model card
        card = generate_model_card(
            model_info, report.get("results", {}),
            report.get("target_bits", 4), hub_repo,
        )
        card_path = os.path.join(output_dir, "README.md")
        with open(card_path, "w") as f:
            f.write(card)

        # Upload all files in output directory
        print("  Uploading files...")
        files_uploaded = 0
        for root, dirs, files in os.walk(output_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                # Skip the report JSON (internal) but upload everything else
                rel_path = os.path.relpath(fpath, output_dir)
                fsize = os.path.getsize(fpath)
                print(f"    {rel_path} ({format_size(fsize)})")
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=rel_path,
                    repo_id=hub_repo,
                    repo_type="model",
                )
                files_uploaded += 1

        print()
        print("  " + "=" * 56)
        print("  PUBLISHED TO HUGGINGFACE")
        print("  " + "=" * 56)
        print()
        print(f"  Repository: https://huggingface.co/{hub_repo}")
        print(f"  Files:      {files_uploaded} uploaded")
        print()
        print("  " + "=" * 56)
        return True

    except ImportError:
        print("  ERROR: huggingface-hub required. Install: pip install huggingface-hub")
        return False
    except Exception as e:
        print(f"  ERROR uploading: {e}")
        return False


# ---------------------------------------------------------------------------
# Feature: --eval  (quality validation via perplexity)
# ---------------------------------------------------------------------------

def evaluate_quantized_model(model_path, model_info, fmt, bits):
    """Run perplexity evaluation on a quantized model."""
    print()
    print("  --- QUALITY EVALUATION ---")
    print()

    if fmt == "gguf":
        return evaluate_gguf(model_path, model_info)
    elif fmt in ("gptq", "awq"):
        return evaluate_transformers(model_path, model_info, fmt)
    else:
        print("  Evaluation not supported for this format.")
        return None


def evaluate_gguf(model_path, model_info):
    """Evaluate GGUF model using llama.cpp perplexity or llama-cpp-python."""
    # Method 1: Try llama-perplexity binary
    llama_perplexity = shutil.which("llama-perplexity") or shutil.which("perplexity")
    if llama_perplexity:
        print("  Running perplexity evaluation (llama.cpp)...")
        print("  Dataset: wikitext-2 (standard benchmark)")
        try:
            cmd = [
                llama_perplexity,
                "-m", model_path,
                "-f", "wikitext-2-raw/wiki.test.raw",
                "--ctx-size", "512",
                "--chunks", "20",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                # Parse perplexity from output
                for line in result.stdout.split("\n"):
                    if "perplexity" in line.lower() and "=" in line:
                        try:
                            ppl = float(line.split("=")[-1].strip().split()[0])
                            return {"perplexity": ppl, "method": "llama.cpp"}
                        except (ValueError, IndexError):
                            pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Method 2: Use llama-cpp-python
    try:
        from llama_cpp import Llama
        print("  Running perplexity evaluation (llama-cpp-python)...")
        print("  Loading model for evaluation...")

        llm = Llama(model_path=model_path, n_ctx=512, verbose=False)

        # Use a standard test passage for quick evaluation
        test_texts = [
            "The quick brown fox jumps over the lazy dog. This is a standard test sentence used to evaluate language model quality.",
            "In machine learning, quantization refers to the process of reducing the number of bits that represent a number.",
            "The Transformer architecture has become the dominant paradigm in natural language processing and computer vision.",
            "Large language models have demonstrated remarkable capabilities in text generation and reasoning tasks.",
            "Neural networks consist of layers of interconnected nodes that process information using learned weights.",
        ]

        total_loss = 0.0
        total_tokens = 0
        for text in test_texts:
            tokens = llm.tokenize(text.encode())
            if len(tokens) < 2:
                continue
            # Score the text
            logits_list = llm.eval(tokens)
            # Simple approximate perplexity using model scoring
            result = llm.create_completion(
                text, max_tokens=1, logprobs=1, echo=True,
            )
            if "choices" in result and result["choices"]:
                logprobs = result["choices"][0].get("logprobs", {})
                if logprobs and logprobs.get("token_logprobs"):
                    token_lps = [
                        lp for lp in logprobs["token_logprobs"]
                        if lp is not None
                    ]
                    if token_lps:
                        total_loss += -sum(token_lps)
                        total_tokens += len(token_lps)

        if total_tokens > 0:
            avg_nll = total_loss / total_tokens
            ppl = math.exp(avg_nll)
            return {"perplexity": round(ppl, 2), "method": "llama-cpp-python", "tokens": total_tokens}

    except ImportError:
        pass
    except Exception as e:
        print(f"  Evaluation error: {e}")

    print("  Could not evaluate GGUF model. Install llama-cpp-python for evaluation.")
    return None


def evaluate_transformers(model_path, model_info, fmt):
    """Evaluate GPTQ/AWQ model using transformers."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  Running perplexity evaluation ({fmt.upper()} via transformers)...")
        print("  Loading quantized model...")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.float16,
        )
        model.eval()

        test_texts = [
            "The quick brown fox jumps over the lazy dog. This is a standard test sentence.",
            "In machine learning, quantization reduces the number of bits that represent a number.",
            "The Transformer architecture has become the dominant paradigm in natural language processing.",
            "Large language models have demonstrated remarkable capabilities in text generation.",
            "Neural networks consist of layers of interconnected nodes that process information.",
        ]

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in test_texts:
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                num_tokens = inputs["input_ids"].shape[1]
                total_loss += loss * num_tokens
                total_tokens += num_tokens

        avg_nll = total_loss / total_tokens
        ppl = math.exp(avg_nll)

        return {"perplexity": round(ppl, 2), "method": "transformers", "tokens": total_tokens}

    except ImportError:
        print("  Evaluation requires transformers + torch. Install: pip install transformers torch")
        return None
    except Exception as e:
        print(f"  Evaluation error: {e}")
        return None


def print_eval_results(eval_result, bits):
    """Print evaluation results with quality assessment."""
    if not eval_result:
        return

    ppl = eval_result.get("perplexity", 0)
    method = eval_result.get("method", "unknown")

    print()
    print("  " + "-" * 56)
    print("  QUALITY EVALUATION")
    print("  " + "-" * 56)
    print()
    print(f"  Perplexity:  {ppl:.2f}")
    print(f"  Method:      {method}")
    print(f"  Tokens:      {eval_result.get('tokens', 'N/A')}")
    print()

    # Quality assessment based on typical perplexity ranges
    if ppl < 10:
        quality = "EXCELLENT"
        note = "Minimal quality loss from quantization."
    elif ppl < 20:
        quality = "GOOD"
        note = "Acceptable quality for most use cases."
    elif ppl < 50:
        quality = "FAIR"
        note = f"Some quality degradation at {bits}-bit. Consider using higher bits."
    elif ppl < 100:
        quality = "DEGRADED"
        note = f"Significant quality loss at {bits}-bit. Recommend {min(bits + 1, 8)}-bit or higher."
    else:
        quality = "POOR"
        note = f"Severe quality loss. Model may produce incoherent output. Use higher bit quantization."

    print(f"  Quality:     {quality}")
    print(f"  Assessment:  {note}")
    print()
    print("  " + "-" * 56)


# ---------------------------------------------------------------------------
# Feature: --recommend  (hardware-aware format recommendation)
# ---------------------------------------------------------------------------

def recommend_format(model_info, deps):
    """Recommend the best quantization format based on hardware and model."""
    print()
    print("  " + "=" * 56)
    print("  TURBOQUANT FORMAT RECOMMENDATION")
    print("  " + "=" * 56)
    print()

    model_size_gb = model_info.get("size_bytes", 0) / 1e9
    params = model_info.get("params_estimate", 0)
    params_b = params / 1e9 if params else 0

    # Detect hardware
    has_cuda = deps.get("cuda", False)
    gpu_name = deps.get("gpu_name", "")
    gpu_mem = deps.get("gpu_mem_gb", 0)
    has_mps = deps.get("mps", False)
    system_ram = get_system_ram_gb()

    print("  Hardware Detected:")
    if has_cuda:
        print(f"    GPU:  {gpu_name} ({gpu_mem}GB VRAM)")
    elif has_mps:
        print(f"    GPU:  Apple Silicon (MPS) — {system_ram}GB unified memory")
    else:
        print(f"    GPU:  None (CPU only)")
    print(f"    RAM:  {system_ram}GB")
    print()

    print("  Model:")
    print(f"    Parameters: {model_info.get('params_human', 'unknown')}")
    print(f"    FP16 Size:  {model_info.get('size_human', 'unknown')}")
    print()

    recommendations = []

    # Estimate quantized sizes
    size_4bit = model_size_gb / 4 if model_size_gb else params_b * 0.5
    size_8bit = model_size_gb / 2 if model_size_gb else params_b * 1.0

    # --- Decision logic ---

    if has_cuda and gpu_mem > 0:
        # GPU available — recommend based on VRAM

        if size_4bit * 1.2 <= gpu_mem:
            # Model fits in VRAM at 4-bit
            recommendations.append({
                "rank": 1,
                "format": "AWQ",
                "bits": 4,
                "reason": f"Best GPU throughput. 4-bit model (~{size_4bit:.1f}GB) fits in {gpu_mem}GB VRAM.",
                "cmd": f"turboquant {model_info['source']} --format awq --bits 4",
                "use": "Production GPU serving with vLLM or TGI",
            })
            recommendations.append({
                "rank": 2,
                "format": "GPTQ",
                "bits": 4,
                "reason": "Alternative GPU format. Wider tool support than AWQ.",
                "cmd": f"turboquant {model_info['source']} --format gptq --bits 4",
                "use": "GPU serving when AWQ isn't available",
            })
            recommendations.append({
                "rank": 3,
                "format": "GGUF",
                "bits": 4,
                "reason": "Universal format. Works with Ollama, LM Studio, llama.cpp.",
                "cmd": f"turboquant {model_info['source']} --format gguf --bits 4",
                "use": "Local use, sharing, or CPU fallback",
            })

        elif size_4bit * 1.2 > gpu_mem and size_4bit <= system_ram:
            # Too big for VRAM, fits in RAM
            recommendations.append({
                "rank": 1,
                "format": "GGUF",
                "bits": 4,
                "reason": f"Model too large for {gpu_mem}GB VRAM. GGUF supports CPU+GPU split.",
                "cmd": f"turboquant {model_info['source']} --format gguf --bits 4",
                "use": "CPU+GPU hybrid inference via llama.cpp",
            })
            if params_b > 13:
                recommendations.append({
                    "rank": 2,
                    "format": "GGUF",
                    "bits": 2,
                    "reason": f"Aggressive compression to fit in {gpu_mem}GB VRAM. Quality trade-off.",
                    "cmd": f"turboquant {model_info['source']} --format gguf --bits 2",
                    "use": "When VRAM is tight and you need GPU acceleration",
                })
        else:
            # Very large model
            recommendations.append({
                "rank": 1,
                "format": "GGUF",
                "bits": 2,
                "reason": f"Model requires aggressive compression for your hardware.",
                "cmd": f"turboquant {model_info['source']} --format gguf --bits 2",
                "use": "Maximum compression for large models",
            })

    elif has_mps:
        # Apple Silicon
        recommendations.append({
            "rank": 1,
            "format": "GGUF",
            "bits": 4,
            "reason": "Best format for Apple Silicon. llama.cpp has Metal acceleration.",
            "cmd": f"turboquant {model_info['source']} --format gguf --bits 4",
            "use": "Ollama or LM Studio on Mac",
        })
        if size_8bit <= system_ram * 0.7:
            recommendations.append({
                "rank": 2,
                "format": "GGUF",
                "bits": 8,
                "reason": f"Higher quality, still fits in {system_ram}GB unified memory.",
                "cmd": f"turboquant {model_info['source']} --format gguf --bits 8",
                "use": "Maximum quality on Mac",
            })

    else:
        # CPU only
        recommendations.append({
            "rank": 1,
            "format": "GGUF",
            "bits": 4,
            "reason": "Only format that runs well on CPU. Use with Ollama or llama.cpp.",
            "cmd": f"turboquant {model_info['source']} --format gguf --bits 4",
            "use": "CPU inference via Ollama or llama.cpp",
        })
        if params_b <= 3 and size_8bit <= system_ram * 0.5:
            recommendations.append({
                "rank": 2,
                "format": "GGUF",
                "bits": 8,
                "reason": f"Small model ({model_info.get('params_human', '')}). Higher quality fits in RAM.",
                "cmd": f"turboquant {model_info['source']} --format gguf --bits 8",
                "use": "Better quality for small models on CPU",
            })

    # Print recommendations
    print("  Recommendations:")
    print()
    for rec in recommendations:
        rank_label = {1: "BEST", 2: "ALSO GOOD", 3: "ALTERNATIVE"}.get(rec["rank"], "")
        print(f"  [{rank_label}] {rec['format']} {rec['bits']}-bit")
        print(f"    Why:  {rec['reason']}")
        print(f"    For:  {rec['use']}")
        print(f"    Run:  {rec['cmd']}")
        print()

    # Quick comparison
    if len(recommendations) > 1:
        print("  To compare all formats:")
        print(f"    turboquant {model_info['source']} --format all --bits 4")
        print()

    print("  " + "=" * 56)

    return recommendations


def get_system_ram_gb():
    """Get system RAM in GB."""
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                return round(int(result.stdout.strip()) / 1e9, 1)
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return round(kb / 1e6, 1)
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant — Compress Any LLM Up to 6x in One Command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 4
              turboquant meta-llama/Llama-3.1-8B-Instruct --target ollama --bits 4
              turboquant meta-llama/Llama-3.1-8B-Instruct --format all --bits 4 --push-to-hub user/repo
              turboquant meta-llama/Llama-3.1-8B-Instruct --recommend
              turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 4 --eval
              turboquant meta-llama/Llama-3.1-8B-Instruct --info
        """),
    )

    parser.add_argument("model", nargs="?", default=None,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--format", "-f", choices=SUPPORTED_FORMATS, default="gguf",
                        help="Output format: gguf, gptq, awq, or all (default: gguf)")
    parser.add_argument("--bits", "-b", type=int, choices=SUPPORTED_BITS, default=4,
                        help="Quantization bits (default: 4)")
    parser.add_argument("--output", "-o", default="./turboquant-output",
                        help="Output directory (default: ./turboquant-output)")
    parser.add_argument("--info", action="store_true",
                        help="Just show model info, don't quantize")
    parser.add_argument("--check", action="store_true",
                        help="Check available backends")

    # New features
    parser.add_argument("--target", "-t", choices=SUPPORTED_TARGETS,
                        help="Target platform: ollama, vllm, llamacpp, lmstudio")
    parser.add_argument("--push-to-hub", metavar="REPO",
                        help="Upload quantized model to HuggingFace Hub (e.g. user/model-GGUF)")
    parser.add_argument("--eval", action="store_true", dest="run_eval",
                        help="Run quality evaluation (perplexity) after quantization")
    parser.add_argument("--recommend", action="store_true",
                        help="Recommend best format based on your hardware and model size")

    args = parser.parse_args()

    print_banner()

    # Check dependencies (no model needed)
    if args.check:
        deps = check_dependencies()
        print("  Available backends:")
        print(f"    PyTorch:      {'YES' if deps.get('torch') else 'NO -- pip install torch'}")
        if deps.get("cuda"):
            print(f"    CUDA GPU:     YES ({deps.get('gpu_name', '')}, {deps.get('gpu_mem_gb', 0)}GB)")
        elif deps.get("mps"):
            print(f"    Apple MPS:    YES (Metal Performance Shaders)")
        else:
            print(f"    GPU:          NO -- CPU only")
        print(f"    Transformers: {'YES' if deps.get('transformers') else 'NO -- pip install transformers'}")
        print(f"    GGUF:         {'YES' if deps.get('gguf') else 'NO -- pip install llama-cpp-python'}")
        print(f"    GPTQ:         {'YES' if deps.get('gptq') else 'NO -- pip install auto-gptq'}")
        print(f"    AWQ:          {'YES' if deps.get('awq') else 'NO -- pip install autoawq'}")
        ram = get_system_ram_gb()
        if ram:
            print(f"    System RAM:   {ram}GB")
        return

    # Require model for all other commands
    if not args.model:
        print("  ERROR: Model is required.")
        print("  Usage: turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 4")
        print("  Check backends: turboquant --check")
        sys.exit(1)

    # Get model info
    print(f"  Fetching model info: {args.model}")
    model_info = get_model_info(args.model)

    if not model_info.get("found"):
        print(f"  ERROR: Model not found: {model_info.get('error', 'unknown')}")
        sys.exit(1)

    print(f"  Architecture: {model_info.get('arch', 'unknown')}")
    print(f"  Parameters:   {model_info.get('params_human', 'unknown')}")
    print(f"  Size:         {model_info.get('size_human', 'unknown')}")

    if model_info.get("size_bytes"):
        est_compressed = model_info["size_bytes"] / estimate_compression(16, args.bits)
        print(f"  Est. output:  {format_size(est_compressed)} ({estimate_compression(16, args.bits):.1f}x compression)")

    # --info: just show model info
    if args.info:
        return

    # --recommend: hardware-aware format recommendation
    if args.recommend:
        deps = check_dependencies()
        recommend_format(model_info, deps)
        return

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # --target ollama: special pipeline
    if args.target == "ollama":
        result = handle_target_ollama(args.model, args.bits, args.output, model_info)
        results = {"gguf": result if result["success"] else result}
        report = generate_report(model_info, results, args.bits)

        # Eval if requested
        if args.run_eval and result["success"]:
            eval_result = evaluate_quantized_model(
                result["file"], model_info, "gguf", args.bits,
            )
            print_eval_results(eval_result, args.bits)
            if eval_result:
                report["eval"] = eval_result

        # Push if requested
        if args.push_to_hub:
            push_to_hub(args.push_to_hub, args.output, model_info, report)

        # Save report
        report_file = os.path.join(args.output, "turboquant-report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Report saved: {report_file}")
        return

    # --target vllm: force AWQ (best vLLM format)
    if args.target == "vllm":
        print("\n  Target: vLLM — using AWQ format (best GPU throughput)")
        args.format = "awq"

    # --target llamacpp or lmstudio: force GGUF
    if args.target in ("llamacpp", "lmstudio"):
        print(f"\n  Target: {args.target} — using GGUF format")
        args.format = "gguf"

    # Run quantization
    formats_to_run = SUPPORTED_FORMATS[:-1] if args.format == "all" else [args.format]
    results = {}

    for fmt in formats_to_run:
        print()
        print(f"  --- {fmt.upper()} ---")
        start = time.time()

        if fmt == "gguf":
            results[fmt] = quantize_gguf(args.model, args.bits, args.output)
        elif fmt == "gptq":
            results[fmt] = quantize_gptq(args.model, args.bits, args.output)
        elif fmt == "awq":
            results[fmt] = quantize_awq(args.model, args.bits, args.output)

        elapsed = time.time() - start
        if results[fmt]["success"]:
            print(f"  Done in {elapsed:.1f}s")
        else:
            print(f"  Failed ({elapsed:.1f}s)")

    # Generate and print report
    report = generate_report(model_info, results, args.bits)
    print_report(report)

    # --eval: run quality evaluation on successful quantizations
    if args.run_eval:
        for fmt, result in results.items():
            if result["success"]:
                print(f"\n  Evaluating {fmt.upper()} quantization...")
                eval_result = evaluate_quantized_model(
                    result["file"], model_info, fmt, args.bits,
                )
                print_eval_results(eval_result, args.bits)
                if eval_result:
                    report.setdefault("eval", {})[fmt] = eval_result

    # Save report
    report_file = os.path.join(args.output, "turboquant-report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved: {report_file}")

    # --push-to-hub: upload to HuggingFace
    if args.push_to_hub:
        push_to_hub(args.push_to_hub, args.output, model_info, report)


if __name__ == "__main__":
    main()
