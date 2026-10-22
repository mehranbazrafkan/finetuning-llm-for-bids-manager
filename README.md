# BIDS-Manager Assistant — Fine-tuning

Fine-tunes `google/gemma-3-270m-it` (via LoRA) into a BIDS-Manager support
assistant. The model learns to answer questions about BIDS validation errors,
metadata rules, filename requirements, etc., **from retrieved knowledge**
supplied in the prompt (RAG-style), matching how the agent works in production.

## Layout

```
bids-manager-finetuning/
├── 01-convert-kb-to-dataset.py    # knowledge.jsonl -> training QA pairs
├── 02-LoRA-fine-tuning.py         # LoRA training + merge into a standalone model
├── test-llm.py                    # interactive chat with the fine-tuned model
├── requirements.txt
├── data/
│   ├── knowledge.jsonl            # ALL knowledge (2488 records, enriched merged in)
│   └── knowledge-fine-tune-dataset.jsonl  # generated training set (1766 samples)
└── output/                        # created at train time
    ├── gemma-270m-it-lora/        # LoRA adapter checkpoint
    └── gemma-270m-it-finetuned/   # merged standalone model
```

`data/knowledge.jsonl` is the single, consistent knowledge source: it is the
original `knowledge.jsonl` plus the `enriched_knowledge.jsonl` records merged
in by id (the enriched records were byte-identical supersets — they only added
an `ai_enrichment` field — so nothing was lost or changed, and duplicates were
removed). 311 records carry inline AI enrichment; the rest are plain knowledge.

All paths resolve relative to each script, so you can run from any directory
and move this whole folder to another PC or server as-is.

## Setup (GPU server)

Install the CUDA build of torch **first** — the default PyPI wheel is CPU-only
and will silently train on CPU. Pick the index matching your server's CUDA:

```bash
# e.g. CUDA 12.6
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt

# verify GPU is visible — you want "True" and a '+cu' version string
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Run

```bash
# 1. Generate the training dataset from data/knowledge.jsonl
python 01-convert-kb-to-dataset.py

# 2. Train (LoRA) and merge into a standalone model
python 02-LoRA-fine-tuning.py

# 3. Chat with the result
python test-llm.py
```

Step 1 is optional if `data/knowledge-fine-tune-dataset.jsonl` is already
present (it is, in this folder).

## How the data is built

The knowledge base is a *retrieval corpus*, not prompt/completion pairs, so
`01-convert-kb-to-dataset.py` synthesizes QA pairs in the format the agent uses
at inference time:

```
[retrieved knowledge block]  +  question   ->   helpful answer
```

It reads the single `data/knowledge.jsonl` and handles the messiness in the
source data: `$ref` chains across records, three different `Enum` shapes, and
`retrieval_text` fields that leak raw `{'$ref': ...}` dicts. Errors/warnings get
both a "what does it mean?" and a "how do I fix it?" pair; where a record carries
inline `ai_enrichment` (311 of them), the extra `common_causes` /
`resolution_guidance` detail is folded into the answer.

Useful flags:

```bash
python 01-convert-kb-to-dataset.py --limit-per-type 50    # quick small dataset
python 01-convert-kb-to-dataset.py --seed 123            # different shuffle
```

## Training notes

- The script auto-picks the dtype: **bf16** on Ampere+ GPUs, **fp16** on Turing
  and older (e.g. GTX 1650 Ti), fp32 on CPU.
- `loss_type="nll"` is set deliberately. TRL 1.14 defaults to `"chunked_nll"`,
  which breaks on Gemma-3 (its `base_model_prefix="language_model"` doesn't
  exist, so the chunked path can't find `last_hidden_state`).
- `max_length=2048` covers the longest samples (~1878 tokens). Truncation cuts
  from the right, which would chop off the end of the answer being learned.
- `num_train_epochs=10` may be too many for 1766 samples — watch the loss and
  drop to 3–5 if it flattens or rises.
