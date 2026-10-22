# from pathlib import Path
# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_path = Path("./output/gemma-270m-it-finetuned")
# repo_id = "mehran/gemma-3-270m-it-bids-finetuned"

# # Verify that the local model directory exists.
# if not model_path.is_dir():
#     raise FileNotFoundError(
#         f"Model directory not found: {model_path.resolve()}"
#     )

# # Load the fine-tuned model and tokenizer.
# tokenizer = AutoTokenizer.from_pretrained(str(model_path))
# model = AutoModelForCausalLM.from_pretrained(str(model_path))

# # Upload both to Hugging Face.
# tokenizer.push_to_hub(repo_id, private=True)
# model.push_to_hub(repo_id, private=True)

# print(f"Successfully uploaded model to: https://huggingface.co/{repo_id}")


from pathlib import Path
from huggingface_hub import HfApi

model_path = Path("./output/qwen3-1.7B-finetuned")
repo_id = "MehranBazrafkan/qwen3-1.7B-bids-finetuned"

# Verify the local model directory.
if not model_path.is_dir():
    raise FileNotFoundError(
        f"Model directory not found: {model_path.resolve()}"
    )

# Check that the directory contains model configuration.
if not (model_path / "config.json").is_file():
    raise FileNotFoundError(
        f"config.json not found in {model_path.resolve()}"
    )

# Create the repository and upload the existing files.
api = HfApi()
api.create_repo(
    repo_id=repo_id,
    private=True,
    exist_ok=True,
)

api.upload_folder(
    folder_path=str(model_path.resolve()),
    repo_id=repo_id,
    repo_type="model",
)

print(f"Upload completed: https://huggingface.co/{repo_id}")
