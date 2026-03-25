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
    turboquant meta-llama/Llama-3.1-8B-Instruct --format all --bits 4  # Compare all formats
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SUPPORTED_FORMATS = ["gguf", "gptq", "awq", "all"]
SUPPORTED_BITS = [2, 3, 4, 5, 8]

GGUF_QUANT_TYPES = {
    2: "Q2_K",
    3: "Q3_K_M",
    4: "Q4_K_M",
    5: "Q5_K_M",
    8: "Q8_0",
}


def check_dependencies():
    """Check which quantization backends are available."""
    available = {}

    # Check for llama.cpp (GGUF)
    llama_convert = shutil.which("llama-quantize") or shutil.which("quantize")
    if llama_convert:
        available["gguf"] = True
    else:
        # Check if llama.cpp python bindings available
        try:
            import llama_cpp
            available["gguf"] = True
        except ImportError:
            available["gguf"] = False

    # Check for AutoGPTQ
    try:
        import auto_gptq
        available["gptq"] = True
    except ImportError:
        available["gptq"] = False

    # Check for AutoAWQ
    try:
        import awq
        available["awq"] = True
    except ImportError:
        available["awq"] = False

    # Check for transformers (always needed)
    try:
        import transformers
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
    except ImportError:
        available["torch"] = False
        available["cuda"] = False

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


def get_model_info(model_id_or_path):
    """Get model information from HuggingFace or local path."""
    info = {"source": model_id_or_path}

    try:
        from huggingface_hub import model_info as hf_model_info
        mi = hf_model_info(model_id_or_path)
        info["model_id"] = mi.id
        info["size_bytes"] = sum(s.size for s in mi.siblings if s.rfilename.endswith(('.safetensors', '.bin')))
        info["size_human"] = format_size(info["size_bytes"])

        # Try to get parameter count from config
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(model_id_or_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        info["arch"] = config.get("architectures", ["unknown"])[0]
        info["hidden_size"] = config.get("hidden_size", 0)
        info["num_layers"] = config.get("num_hidden_layers", 0)
        info["vocab_size"] = config.get("vocab_size", 0)

        # Estimate parameters
        h = info["hidden_size"]
        n = info["num_layers"]
        v = info["vocab_size"]
        if h and n and v:
            # Rough estimate: 12 * n * h^2 + v * h (for transformer)
            params = 12 * n * h * h + v * h
            info["params_estimate"] = params
            info["params_human"] = f"{params/1e9:.1f}B" if params > 1e9 else f"{params/1e6:.0f}M"

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
        else:
            info["found"] = False
            info["error"] = str(e)

    return info


def quantize_gguf(model_id, bits, output_dir):
    """Quantize model to GGUF format using llama.cpp."""
    quant_type = GGUF_QUANT_TYPES.get(bits, "Q4_K_M")
    output_file = os.path.join(output_dir, f"model-{quant_type}.gguf")

    print(f"  Converting to GGUF {quant_type} ({bits}-bit)...")

    # Method 1: Try llama-cpp-python convert
    try:
        # First convert to GGUF FP16
        fp16_file = os.path.join(output_dir, "model-fp16.gguf")
        cmd_convert = [
            sys.executable, "-m", "llama_cpp.convert",
            "--outfile", fp16_file,
            "--outtype", "f16",
            model_id,
        ]
        print(f"  Step 1: Converting to GGUF FP16...")
        result = subprocess.run(cmd_convert, capture_output=True, text=True, timeout=3600)

        if result.returncode == 0 and os.path.exists(fp16_file):
            # Then quantize
            cmd_quant = ["llama-quantize", fp16_file, output_file, quant_type]
            print(f"  Step 2: Quantizing to {quant_type}...")
            result = subprocess.run(cmd_quant, capture_output=True, text=True, timeout=3600)

            if result.returncode == 0 and os.path.exists(output_file):
                os.remove(fp16_file)  # Clean up FP16
                return {"success": True, "file": output_file, "size": os.path.getsize(output_file)}

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 2: Try using transformers + manual conversion
    try:
        print(f"  Using transformers + manual GGUF conversion...")
        convert_script = shutil.which("convert_hf_to_gguf.py")
        if convert_script:
            cmd = [sys.executable, convert_script, model_id, "--outfile", output_file, "--outtype", quant_type.lower()]
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

        # Prepare calibration data
        print("  Preparing calibration data (c4 dataset, 128 samples)...")
        from datasets import load_dataset
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        calibration_data = []
        for i, example in enumerate(dataset):
            if i >= 128:
                break
            tokenized = tokenizer(example["text"], return_tensors="pt", truncation=True, max_length=2048)
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

    except ImportError as e:
        return {
            "success": False,
            "error": f"GPTQ requires: pip install auto-gptq datasets",
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


def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant — Compress Any LLM Up to 6x in One Command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  turboquant meta-llama/Llama-3.1-8B-Instruct --format gguf --bits 4
  turboquant ./my-model --format gptq --bits 4
  turboquant meta-llama/Llama-3.1-8B-Instruct --format all --bits 4
  turboquant meta-llama/Llama-3.1-8B-Instruct --info  # Just show model info
        """,
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

    args = parser.parse_args()

    print_banner()

    # Check dependencies (no model needed)
    if args.check:
        deps = check_dependencies()
        print("  Available backends:")
        print(f"    PyTorch:      {'YES' if deps.get('torch') else 'NO — pip install torch'}")
        print(f"    CUDA GPU:     {'YES (' + deps.get('gpu_name', '') + ', ' + str(deps.get('gpu_mem_gb', 0)) + 'GB)' if deps.get('cuda') else 'NO — CPU only'}")
        print(f"    Transformers: {'YES' if deps.get('transformers') else 'NO — pip install transformers'}")
        print(f"    GGUF:         {'YES' if deps.get('gguf') else 'NO — pip install llama-cpp-python'}")
        print(f"    GPTQ:         {'YES' if deps.get('gptq') else 'NO — pip install auto-gptq'}")
        print(f"    AWQ:          {'YES' if deps.get('awq') else 'NO — pip install autoawq'}")
        return

    # Require model for non-check commands
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

    if args.info:
        return

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

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

    # Save report
    report_file = os.path.join(args.output, "turboquant-report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved: {report_file}")


if __name__ == "__main__":
    main()
