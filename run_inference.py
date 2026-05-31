# %% [markdown]
# # CSE 151B Competition — Modified Starter Notebook
# 
# Welcome to the **CSE 151B Spring 2026 Math Reasoning Competition**!  
# This notebook walks you through the full pipeline end-to-end:
# 
# 1. Setting up the Python environment with `uv`
# 2. Loading the competition dataset
# 3. Running inference with **Qwen3-4B-Thinking** via vLLM
# 5. Saving results to csv for submission
# 
# The public dataset (`public.jsonl`) contains questions **with** answers so you can measure accuracy locally.  
# The private test set used for the leaderboard does **not** include answers — for that, skip evaluation and submit the raw responses.

# %% [markdown]
# ## 1. Environment Setup
# 
# We use [`uv`](https://github.com/astral-sh/uv) for fast, reproducible package management.
# 
# The steps below:
# 1. Install `uv` into `~/.local/bin`
# 2. Create a virtual environment at `.venv/`
# 3. Install all required packages (This might take a while)
# 
# > **After running this cell, restart the kernel** so that the newly installed packages (especially `vllm` and `transformers`) are picked up by the current Python session.

# %% [markdown]
# ### Comment Out the cell below after first installation.

# %%
# pip install accelerate

# %%
# # Install uv
# !wget -qO- https://astral.sh/uv/install.sh | sh

# # Create a virtual environment
# !uv venv .venv --seed --clear

# # Install dependencies — this is fast thanks to uv's parallel resolver
# !.venv/bin/python -m pip install sympy numpy transformers vllm tqdm bitsandbytes antlr4-python3-runtime==4.11.1 ipykernel jupyter

# # Install Jupyter Kernel
# !.venv/bin/python -m ipykernel install --user --name cse151b --display-name "Python (cse151b)"

# print("Done. Restart the kernel before proceeding.")
# print("Selection process: on top right, click on current kernel '(ususally named python)' -> 'select another kernel' -> 'Jupyter Kernel' -> 'Python (cse151b)'.")

# %% [markdown]
# ### Run the cell below every time to activate the installed environment. 

# %%
# activate venv after installation. This needs to be run everytime.
!source ./.venv/bin/activate

# %% [markdown]
# ## 2. Imports & Configuration
# 
# All key settings are collected in one place.  
# - `DATA_PATH` — public dataset with ground-truth answers (use this to measure accuracy)
# - `OUTPUT_PATH` — where per-question results will be written
# - `GPU_ID` — which GPU to use (update if your machine has a different device index)
# - `MAX_TOKENS` — maximum tokens the model may generate per response

# %%
import json, os, csv, re, time
from pathlib import Path
from typing import Optional


# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"                    # CUDA_VISIBLE_DEVICES
DATA_PATH   = "data/private.jsonl"
OUTPUT_PATH = "results/private.csv"

MAX_TOKENS_MCQ  = 12288
MAX_TOKENS_MATH = 12288
MAX_MODEL_LEN   = 16384
CHUNK_SIZE      = 10
 
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0" # For VLLM to work with a RTX 6000 Blackwell
# os.environ["VLLM_USE_DEEP_GEMM"] = "0"  # For VLLLM to work with a H100


import re
import sys
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# %% [markdown]
# ## 3. Load the Dataset
# 
# The dataset is stored as newline-delimited JSON (`.jsonl`). Each line is one question with the following fields:
# 
# | Field | Description |
# |---|---|
# | `id` | Unique question identifier |
# | `question` | Problem statement |
# | `options` | List of answer choices — present for **MCQ**, absent for **free-form** |
# | `answer` | Ground-truth answer (letter for MCQ, value/list for free-form) |

# %%
data = [json.loads(line) for line in open(DATA_PATH)]

n_mcq  = sum(bool(d.get("options")) for d in data)
n_free = sum(not d.get("options")   for d in data)
print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

# %% [markdown]
# ## 4. Prompt Construction
# 
# We use two system prompts depending on the question type:
# 
# - **MCQ** — the model must select the best answer letter and wrap it in `\boxed{}`
# - **Free-form** — the model solves step-by-step and puts the final answer in `\boxed{}`
# 
# `build_prompt()` returns the appropriate `(system, user)` pair for each item.

# %%
# ── IMPROVEMENT 3: Stronger System Prompts ────────────────────────────────────
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician competing in a math olympiad. "
    "Think carefully but be concise — limit your thinking to what is necessary. "
    "Show your reasoning step-by-step, then put ONLY your final answer inside \\boxed{}. "
    "For numerical answers, simplify completely. "
    "Do NOT include units or explanations inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
    "A response without \\boxed{} is considered wrong."

)
 
SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Think carefully but be concise — limit your thinking to what is necessary. "
    "Output ONLY the letter of the correct answer inside \\boxed{}, e.g. \\boxed{C}. "
    "Do not write anything after the boxed answer."
    "A response without \\boxed{} is considered wrong."

)

def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


# %% [markdown]
# ## 5. Load Model with vLLM (for general case, vLLM is faster)
# 
# We load **Qwen3-4B-Thinking-2507** 
# 
# Key parameters:
# - `gpu_memory_utilization` — fraction of GPU VRAM reserved for the model and KV cache
# - `max_model_len` — maximum sequence length (prompt + generation)
# - `max_num_seqs` — maximum number of sequences processed in parallel

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

llm = LLM(
    model=MODEL_ID,
    dtype="float16",
    enable_prefix_caching=True,
    gpu_memory_utilization=0.90,
    max_model_len=MAX_MODEL_LEN,
    trust_remote_code=True,
    max_num_seqs=128,
    max_num_batched_tokens=MAX_MODEL_LEN
)
print("Model loaded.")

# %%
# MCQ: want greediness
sampling_params_mcq = SamplingParams(
    max_tokens=MAX_TOKENS_MCQ,
    temperature=0.0,
    top_p=1.0,
)
 
# Free-form math: want creativity
sampling_params_math = SamplingParams(
    max_tokens=MAX_TOKENS_MATH,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    repetition_penalty=1.05,
)

# %% [markdown]
# ## 6. Generate Responses
# 
# We format every question into a chat-template prompt, then call `llm.generate()` in chunks in order to mitigate damage from the frequent Datahub disconnects.  
# vLLM handles batching and scheduling internally — no manual batching needed.

# %% [markdown]
# ### Generate with vLLM

# %%
out_path = Path(OUTPUT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)

# Check what questions have been completed
completed_ids = set()
if out_path.exists():
    with open(out_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            completed_ids.add(int(row["id"]))

print(f"Resuming: {len(completed_ids)} already done, {len(data) - len(completed_ids)} remaining.")

remaining_data    = [d for d in data if d["id"] not in completed_ids]
remaining_prompts = []
remaining_params  = []

for item in remaining_data:
    is_mcq = bool(item.get("options"))
    system, user = build_prompt(item["question"], item.get("options"))
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    remaining_prompts.append(prompt_text)
    remaining_params.append(sampling_params_mcq if is_mcq else sampling_params_math)

write_header = not out_path.exists() or os.path.getsize(out_path) == 0
f_out  = open(out_path, "a", newline="", encoding="utf-8")
writer = csv.DictWriter(f_out, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL)
if write_header:
    writer.writeheader()

# Generate in chunks and show eta for completion of generation
start_total = time.time()
total_done  = len(completed_ids)
total_all   = len(data)

print(f"Generating responses for {len(remaining_prompts)} questions (chunk size={CHUNK_SIZE})...")

for chunk_start in range(0, len(remaining_prompts), CHUNK_SIZE):
    chunk_end     = min(chunk_start + CHUNK_SIZE, len(remaining_prompts))
    chunk_items   = remaining_data[chunk_start:chunk_end]
    chunk_prompts = remaining_prompts[chunk_start:chunk_end]
    chunk_params  = remaining_params[chunk_start:chunk_end]

    chunk_t = time.time()
    outputs = llm.generate(chunk_prompts, sampling_params=chunk_params)
    chunk_t = time.time() - chunk_t

    for item, out in zip(chunk_items, outputs):
        writer.writerow({"id": item["id"], "response": out.outputs[0].text.strip()})
    f_out.flush()

    total_done += len(chunk_items)
    remaining   = total_all - total_done
    avg         = (time.time() - start_total) / (total_done - len(completed_ids))
    eta         = int(avg * remaining)
    print(
        f"  [{total_done:>4}/{total_all}] "
        f"chunk took {chunk_t:.1f}s  |  "
        f"avg {avg:.1f}s/q  |  "
        f"ETA {eta//60}m {eta%60:02d}s"
    )

f_out.close()
print(f"\nDone. Results saved to {out_path}")

# %%
# Reads the saved CSV and checks how many responses are missing a \boxed{} answer. For diagnostic purposes only.

def extract_boxed(text: str) -> Optional[str]:
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None

saved_responses = []
with open(out_path, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        saved_responses.append(row["response"])

missing = [i for i, r in enumerate(saved_responses) if extract_boxed(r) is None]

if missing:
    print(f"WARNING: {len(missing)}/{len(saved_responses)} responses missing \\boxed{{}}")
    print(f"  Missing at row indices: {missing[:10]}{'...' if len(missing) > 10 else ''}")
else:
    print(f"All {len(saved_responses)} responses contain a \\boxed{{}} answer.")

print(f"\nPreview of first 3 rows:")
with open(out_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 4: break
        print(line[:120].rstrip() + ("..." if len(line) > 120 else ""))


