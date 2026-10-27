# /// script
# dependencies = [
#   "llama-cpp-python",
#   "huggingface-hub",
# ]
# ///
"""
Terminal chat with Qwen2.5-1.5B-Instruct using llama-cpp-python.

Run with uv:
  uv run qwen_chat.py            # Default: 16-bit (FP16)
  uv run qwen_chat.py --bab=8    # 8-bit (Q8_0)
  uv run qwen_chat.py --bab=4    # 4-bit (Q4_K_M)
"""

import argparse
from llama_cpp import Llama
from asr import ASR


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
    return parser.parse_args()


def main():
    args = parse_args()

    if args.local_gguf:
        print(f"Loading local GGUF: {args.local_gguf}...")
        llm = Llama(
            model_path=args.local_gguf,
            n_gpu_layers=-1,  # Offload all layers to GPU if available
            n_ctx=4096,
            verbose=False,
        )
    else:
        repo_id, filename = QUANT_MAP[args.bab]
        print(f"Loading {filename} ({args.bab}-bit) from {repo_id}...")
        llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_gpu_layers=-1,  # Offload all layers to GPU if available
            n_ctx=4096,
            verbose=False,
        )

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Ready. Type your message. (/reset to clear, /exit to quit)\n")
    asr = ASR()
    while True:
        try:
            user_input = asr.get_text_from_mic()
            print("YOU: " + user_input)
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
                    print(token, end="", flush=True)
                    reply += token
        except KeyboardInterrupt:
            print("\n[interrupted]", end="")
        print("\n")

        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()