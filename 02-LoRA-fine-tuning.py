import torch
from pathlib import Path
from datasets import load_dataset
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel


# ==================================================
# Configuration
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_FILE = DATA_DIR / "knowledge-fine-tune-dataset.jsonl"

OUTPUT_DIR = BASE_DIR / "output"
# MODEL_ID = "google/gemma-3-270m-it"
MODEL_ID = "Qwen/Qwen3-1.7B"
LORA_OUTPUT_DIR = OUTPUT_DIR / "qwen3-1.7B-lora"
FINAL_OUTPUT_DIR = OUTPUT_DIR / "qwen3-1.7B-finetuned"


# Device + dtype selection. Uses CUDA when available, otherwise falls back to CPU.
#
# Dtype choice is based on GPU architecture, not just the bf16 support flag:
#   - Ampere or newer (compute >= 8.0): real bf16 tensor cores -> use bf16.
#     torch.cuda.is_bf16_supported() returns True on Turing too, but there bf16 is
#     emulated and SLOWER than fp16, so we don't rely on that flag alone.
#   - Turing and older (compute < 8.0, e.g. GTX 1650 Ti): use fp16 (native cores).
#   - CPU: fp32. fp16/bf16 must both be off on CPU or the Trainer errors out.
HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    major, _ = torch.cuda.get_device_capability(0)
    USE_BF16 = major >= 8
    USE_FP16 = major < 8
else:
    USE_BF16 = False
    USE_FP16 = False

# Load the frozen base model in bf16. This halves weight memory (~540MB vs
# ~1.08GB fp32) which is what makes a 4GB card fit. bf16 has fp32's exponent
# range, so unlike fp16 it won't overflow to nan gradients. The LoRA adapters
# train directly in this dtype (fp16/bf16 stay False in SFTConfig).
MODEL_DTYPE = torch.bfloat16 if (HAS_CUDA and not USE_FP16) else torch.float32

print(f"CUDA available : {HAS_CUDA}")
if HAS_CUDA:
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
print(f"Model dtype    : {MODEL_DTYPE}")

# Persona the agent should adopt. Trained via the system role so the merged
# model answers like a BIDS-Manager support assistant.
SYSTEM_PROMPT = (
    "You are the BIDS-Manager assistant. BIDS-Manager is a tool for creating and validating "
    "datasets that follow the Brain Imaging Data Structure (BIDS) standard. "
    "Answer the user's question using the retrieved BIDS knowledge provided in their message. "
    "Be accurate and concise. If the provided knowledge does not contain the answer, say so "
    "instead of guessing."
)


# ==================================================
# Load tokenizer
# ==================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


# ==================================================
# Load base model
# ==================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=MODEL_DTYPE,
)


# ==================================================
# Load dataset
# ==================================================

if not DATASET_FILE.exists():
    raise FileNotFoundError(
        f"Training dataset not found: {DATASET_FILE}\n"
        f"Run 'python 01-convert-kb-to-dataset.py' first to generate it."
    )

dataset = load_dataset(
    "json",
    data_files=str(DATASET_FILE),
)

print(dataset)


# ==================================================
# Convert dataset into training text
# ==================================================

def format_example(example):
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": example["prompt"]
            },
            {
                "role": "assistant",
                "content": example["completion"]
            }
        ]
    }


dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)


# ==================================================
# LoRA configuration
# ==================================================

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    bias="none",
    task_type="CAUSAL_LM",
)


# ==================================================
# Training configuration
# ==================================================

training_args = SFTConfig(
    output_dir=LORA_OUTPUT_DIR,

    num_train_epochs=4,

    # batch=4 OOMs on a 4GB card. batch=1 + grad checkpointing fits (~4GB).
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,

    learning_rate=1e-4,

    logging_steps=10,
    save_steps=10,
    save_total_limit=2,

    # fp16/bf16 need a GPU. On CPU both must be False or the Trainer raises
    # "Your setup doesn't support bf16/gpu". Auto-detected at the top of the file.
    fp16=USE_FP16,
    bf16=USE_BF16,

    # TRL 1.14 defaults to "chunked_nll", which reads `last_hidden_state` from
    # `model.base_model`. Gemma-3 sets base_model_prefix="language_model" (a name
    # that doesn't exist), so base_model resolves to the full causal LM, which only
    # returns `logits` -> AttributeError. "nll" is standard cross-entropy and works.
    loss_type="nll",

    # Tokenized samples run up to ~1878 tokens. Truncation cuts from the right,
    # which would chop off the end of the COMPLETION (the answer we want to learn).
    # Dynamic padding keeps short batches cheap, so a high ceiling is nearly free.
    max_length=2048,
)


# ==================================================
# Create trainer
# ==================================================

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    peft_config=lora_config,
)


# ==================================================
# Show trainable parameters
# ==================================================

trainer.model.print_trainable_parameters()


# ==================================================
# Train LoRA
# ==================================================

print("\nStarting LoRA training...\n")

trainer.train()


# ==================================================
# Save LoRA adapter
# ==================================================

print("\nSaving LoRA adapter...")

trainer.save_model(LORA_OUTPUT_DIR)
tokenizer.save_pretrained(LORA_OUTPUT_DIR)


# ==================================================
# Free training model from memory
# ==================================================

del trainer
del model

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ==================================================
# Reload base model
# ==================================================

print("\nReloading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=MODEL_DTYPE,
)


# ==================================================
# Load LoRA adapter
# ==================================================

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_OUTPUT_DIR,
)


# ==================================================
# Merge LoRA into the base model
# ==================================================

print("Merging LoRA into base model...")

model = model.merge_and_unload()


# ==================================================
# Save standalone model
# ==================================================

print("Saving final standalone model...")

model.save_pretrained(
    FINAL_OUTPUT_DIR,
    safe_serialization=True,
)

tokenizer.save_pretrained(
    FINAL_OUTPUT_DIR,
)


# ==================================================
# Done
# ==================================================

print("\n========================================")
print("Fine-tuning complete!")
print("Standalone model:")
print(FINAL_OUTPUT_DIR)
print("========================================")
