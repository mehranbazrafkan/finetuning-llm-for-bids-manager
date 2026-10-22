"""Fast inference test for the fine-tuned BIDS-Manager assistant using AirLLM.

AirLLM streams the model's layers between CPU RAM and the GPU (or keeps them
all on CPU if no GPU is available), so the model loads with very low memory
overhead instead of materializing every weight at once.

Install AirLLM once with:

    pip install airllm

Usage:

    python test-llm-airllm.py                  # interactive chat loop
    python test-llm-airllm.py -q "question"    # single non-interactive question
"""

import argparse
import time
from pathlib import Path

import torch
from airllm import AutoModel


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "output" / "qwen3-1.7B-finetuned"

# Must match the system prompt used during fine-tuning (02-LoRA-fine-tuning.py)
SYSTEM_PROMPT = (
    "You are the BIDS-Manager assistant. BIDS-Manager is a tool for creating and validating "
    "datasets that follow the Brain Imaging Data Structure (BIDS) standard. "
    "Answer the user's question using the retrieved BIDS knowledge provided in their message. "
    "Be accurate and concise. If the provided knowledge does not contain the answer, say so "
    "instead of guessing."
)

MAX_NEW_TOKENS = 200
MAX_SEQ_LEN = 4096   # prompt + new tokens budget (AirLLM default is only 512)

# AirLLM's constructor takes `device` (e.g. "cuda:0" or "cpu"), NOT `device_map`.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ============================================================
# Load model
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Fine-tuned model not found: {MODEL_PATH}\n"
        f"Run 'python 02-LoRA-fine-tuning.py' first to train and save it."
    )

print(f"Loading model (AirLLM): {MODEL_PATH}")
print(f"Device: {DEVICE}")
print("(On the first run AirLLM also splits the checkpoint into per-layer shards)")

# AirLLM streams layer weights disk -> device on demand. Its constructor signature is:
#   (path, device="cuda:0", dtype=None, max_seq_len=512, ...)
# Passing `device_map="auto"` raises: AirLLMBaseModel.__init__() got an unexpected keyword argument
model = AutoModel.from_pretrained(
    str(MODEL_PATH),
    device=DEVICE,
    max_seq_len=MAX_SEQ_LEN,
)

tokenizer = model.get_tokenizer()
tokenizer.padding_side = "left"

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded.")


# ============================================================
# Fast-inference generation helper
# ============================================================


def generate_reply(messages):
    """Run one chat turn and return (reply_text, seconds_taken)."""

    # transformers 4.x returns a bare input_ids tensor, 5.x returns a
    # BatchEncoding (not a dict subclass). torch-free type check below
    # avoids importing torch just for that test.
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if hasattr(encoded, "keys") and "input_ids" in encoded:
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
    else:
        input_ids = encoded
        attention_mask = None

    start = time.perf_counter()

    # Greedy decoding (do_sample=False) is the fastest deterministic option:
    # no sampling overhead, no penalty computation, reproducible answers.
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,   # KV cache: required for fast autoregressive decoding
    )

    elapsed = time.perf_counter() - start

    # Depending on transformers version/generation config, generate() returns either a plain
    # tensor (default) or an object carrying `sequences` (BatchEncoding, GenerateOutput, ...).
    if hasattr(outputs, "sequences"):
        sequences = outputs["sequences"]
    else:
        sequences = outputs
    generated_tokens = sequences[0][input_ids.shape[1]:]

    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return text, elapsed


# ============================================================
# Warm-up (first call always pays layer-loading overhead in AirLLM)
# ============================================================

print("Warming up (first generation streams model layers)...")
_ = generate_reply([
    {"role": "user", "content": "Hi"},
])
print("Warm-up done.\n")


# ============================================================
# Conversation history (seeded with the fine-tuning system prompt)
# ============================================================

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
]


# ============================================================
# Main
# ============================================================

parser = argparse.ArgumentParser(description="Test the fine-tuned BIDS-Manager assistant (AirLLM).")
parser.add_argument("-q", "--question", help="Ask a single question and exit (non-interactive).")
args = parser.parse_args()

if args.question:

    # --------------------------------------------------------
    # Single-shot mode
    # --------------------------------------------------------

    messages.append({"role": "user", "content": args.question})

    reply, elapsed = generate_reply(messages)

    print(f"Assistant: {reply}")
    print(f"[{elapsed:.2f}s]")

else:

    # --------------------------------------------------------
    # Interactive loop
    # --------------------------------------------------------

    print("Type 'exit' or 'quit' to stop.\n")

    while True:

        prompt = input("You: ")

        if prompt.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not prompt.strip():
            continue

        messages.append({"role": "user", "content": prompt})

        reply, elapsed = generate_reply(messages)

        messages.append({"role": "assistant", "content": reply})

        print(f"\nAssistant: {reply}")
        print(f"[{elapsed:.2f}s]\n")
