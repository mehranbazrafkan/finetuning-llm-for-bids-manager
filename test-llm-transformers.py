import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as transformers_logging


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
TEMPERATURE = 0.7
TOP_P = 0.9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Load model
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Fine-tuned model not found: {MODEL_PATH}\n"
        f"Run 'python 02-LoRA-fine-tuning.py' first to train and save it."
    )

print(f"Loading model: {MODEL_PATH}")
print(f"Device: {DEVICE}")

# transformers <= 4.57 emits a false-positive "incorrect regex" warning for local
# non-Mistral models (see tokenization_utils_base._patch_mistral_regex): this Gemma
# tokenizer is fine, and applying the Mistral patch would actually corrupt it.
# Silencing it for the duration of the load only.
transformers_logging.set_verbosity_error()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
transformers_logging.set_verbosity_warning()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
).to(DEVICE)

model.generation_config.pad_token_id = tokenizer.pad_token_id

model.eval()

print("Model loaded.")
print("Type 'exit' or 'quit' to stop.\n")


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
# Conversation loop
# ============================================================


while True:

    # --------------------------------------------------------
    # Get user input
    # --------------------------------------------------------

    prompt = input("You: ")

    if prompt.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not prompt.strip():
        continue

    # Add user message to conversation
    messages.append({
        "role": "user",
        "content": prompt,
    })


    # --------------------------------------------------------
    # Convert conversation to model input
    # --------------------------------------------------------

    # transformers 5.x returns a BatchEncoding (which is NOT a dict subclass);
    # 4.x returns a bare input_ids tensor. torch.is_tensor() tells them apart.
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if torch.is_tensor(inputs):                       # transformers 4.x
        input_ids = inputs
        attention_mask = torch.ones_like(inputs)
    else:                                             # transformers 5.x
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)


    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )


    # --------------------------------------------------------
    # Extract only newly generated tokens
    # --------------------------------------------------------

    input_length = input_ids.shape[1]

    generated_tokens = outputs[0][input_length:]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


    # --------------------------------------------------------
    # Add model response to conversation
    # --------------------------------------------------------

    messages.append({
        "role": "assistant",
        "content": generated_text,
    })


    # --------------------------------------------------------
    # Print response
    # --------------------------------------------------------

    print(f"\nAssistant: {generated_text}\n")
