"""
Terminal chat with Qwen2.5-1.5B-Instruct using llama-cpp-python.

Run with uv:
  uv run python src/utils/qwen_chat.py            # Default: 16-bit (FP16)
  uv run python src/utils/qwen_chat.py --bab=8    # 8-bit (Q8_0)
  uv run python src/utils/qwen_chat.py --bab=4    # 4-bit (Q4_K_M)
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

try:
    from asr import ASR
except ImportError:
    from src.utils.asr import ASR

try:
    from evals import LLMEval, calculate_metrics, print_metrics
except ImportError:
    from src.utils.evals import LLMEval, calculate_metrics, print_metrics


SYSTEM_PROMPT = "You are a helpful assistant."
MAX_NEW_TOKENS = 1024

# Official Qwen GGUF repo for 4-bit and 8-bit; Bartowski's repo includes the unquantized f16 GGUF
QUANT_MAP = {
    4: ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    8: ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q8_0.gguf"),
    16: ("bartowski/Qwen2.5-1.5B-Instruct-GGUF", "Qwen2.5-1.5B-Instruct-f16.gguf"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Llama.cpp chat with Qwen2.5")
    parser.add_argument(
        "--bab",
        type=int,
        choices=[4, 8, 16],
        default=16,
        help="Quantization bits: 4 (Q4_K_M), 8 (Q8_0), or 16 (F16, default)",
    )
    parser.add_argument(
        "--local-gguf",
        type=str,
        default=None,
        help="Optional path to a locally converted .gguf file",
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Use microphone input with ASR instead of text input",
    )
    return parser.parse_args()


def resolve_model_path(repo_id: str, filename: str) -> str:
    """
    Resolve local path for the GGUF model file.
    Tries the local cache first to ensure completely offline operation.
    Falls back to downloading online only if not cached.
    """
    # 1. Try local cache first (does not require internet)
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
    except Exception:
        pass

    # 2. Try online download if not present in cache
    print(f"Model not found in local cache. Attempting to download {filename} from {repo_id}...")
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=False)
    except Exception as e:
        raise RuntimeError(
            f"Could not load model '{filename}' from '{repo_id}'.\n"
            f"The model is not cached locally, and could not be downloaded (no internet connection).\n"
            f"Please check your internet connection or specify a local model using --local-gguf.\n"
            f"Details: {e}"
        ) from e


def main():
    args = parse_args()

    if args.local_gguf:
        if not os.path.isfile(args.local_gguf):
            raise FileNotFoundError(f"Local GGUF file not found: {args.local_gguf}")
        model_path = args.local_gguf
        print(f"Loading local GGUF: {model_path}...")
    else:
        repo_id, filename = QUANT_MAP[args.bab]
        print(f"Resolving model {filename} ({args.bab}-bit) from {repo_id}...")
        model_path = resolve_model_path(repo_id, filename)
        print(f"Loading model from local path: {model_path}...")

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,  # Offload all layers to GPU if available
        n_ctx=4096,
        verbose=False,
    )

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Ready. Type your message. (/reset to clear, /exit to quit)\n")
    asr = ASR()
    while True:
        asr_metrics = None
        try:
            if args.mic:
                user_input, asr_metrics = asr.get_text_from_mic(get_metrics=True)
                print("YOU: " + user_input)
            else:
                # uncomment the below two lines to start taking input from the mic
                # user_input, asr_metrics = asr.get_text_from_mic(get_metrics=True)
                # print("YOU: " + user_input) 
                user_input = input("YOU: ") # comment out this line to stop using text input
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print("Bye!")
            break
        if user_input.lower() == "/reset":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[history cleared]\n")
            continue

        history.append({"role": "user", "content": user_input})

        evaluator = LLMEval(asr_metrics=asr_metrics)
        evaluator.start()

        stream = llm.create_chat_completion(
            messages=history,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repeat_penalty=1.05,
            stream=True,
        )

        print("Qwen: ", end="", flush=True)
        reply = ""
        try:
            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta and delta["content"]:
                    token = delta["content"]
                    evaluator.record_token()
                    print(token, end="", flush=True)
                    reply += token
        except KeyboardInterrupt:
            print("\n[interrupted]", end="")
        print("\n")

        evaluator.print_metrics()
        print()

        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()