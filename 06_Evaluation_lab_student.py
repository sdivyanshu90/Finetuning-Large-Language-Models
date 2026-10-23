# %% [markdown]
# Source notebook: 06_Evaluation_lab_student.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

from IPython import get_ipython as _active_ipython

def get_ipython():
    """Use the active shell and expose this script namespace to line magics."""
    shell = _active_ipython()
    if shell is None:
        raise RuntimeError("This source uses IPython commands; run it with ipython instead of python.")
    import inspect
    caller = inspect.currentframe().f_back
    shell.user_ns.update(caller.f_globals)
    shell.user_ns.update(caller.f_locals)
    return shell

# %% [markdown]
# # Evaluation

# %% [markdown]
# ### Technically, there are very few steps to run it on GPUs, elsewhere (ie. on Lamini).
# ```
# finetuned_model = BasicModelRunner(
#     "lamini/lamini_docs_finetuned"
# )
# finetuned_output = finetuned_model(
#     test_dataset_list # batched!
# ) 
# ```
# 
# ### Let's look again under the hood! This is the open core code of Lamini's `llama` library :)

# %%
# Notebook cell 3
import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import logging
import difflib
import pandas as pd

import transformers
import datasets
import torch

from tqdm import tqdm
from utilities import *
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)
global_config = None

# %%
# Notebook cell 4
dataset = datasets.load_dataset("lamini/lamini_docs")

test_dataset = dataset["test"]

# %%
# Notebook cell 5
print(test_dataset[0]["question"])
print(test_dataset[0]["answer"])

# %%
# Notebook cell 6
model_name = "lamini/lamini_docs_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# %% [markdown]
# ### Setup a really basic evaluation function

# %%
# Notebook cell 8
def is_exact_match(a, b):
    return a.strip() == b.strip()

# %%
# Notebook cell 9
model.eval()

# %%
# Notebook cell 10
def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
  # Tokenize
  tokenizer.pad_token = tokenizer.eos_token
  input_ids = tokenizer.encode(
      text,
      return_tensors="pt",
      truncation=True,
      max_length=max_input_tokens
  )

  # Generate
  device = model.device
  generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device),
    max_length=max_output_tokens
  )

  # Decode
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

  # Strip the prompt
  generated_text_answer = generated_text_with_prompt[0][len(text):]

  return generated_text_answer

# %% [markdown]
# ### Run model and compare to expected answer

# %%
# Notebook cell 12
test_question = test_dataset[0]["question"]
generated_answer = inference(test_question, model, tokenizer)
print(test_question)
print(generated_answer)

# %%
# Notebook cell 13
answer = test_dataset[0]["answer"]
print(answer)

# %%
# Notebook cell 14
exact_match = is_exact_match(generated_answer, answer)
print(exact_match)

# %% [markdown]
# ### Run over entire dataset

# %%
# Notebook cell 16
n = 10
metrics = {'exact_matches': []}
predictions = []
for i, item in tqdm(enumerate(test_dataset)):
    print("i Evaluating: " + str(item))
    question = item['question']
    answer = item['answer']

    try:
      predicted_answer = inference(question, model, tokenizer)
    except:
      continue
    predictions.append([predicted_answer, answer])

    #fixed: exact_match = is_exact_match(generated_answer, answer)
    exact_match = is_exact_match(predicted_answer, answer)
    metrics['exact_matches'].append(exact_match)

    if i > n and n != -1:
      break
print('Number of exact matches: ', sum(metrics['exact_matches']))

# %%
# Notebook cell 17
df = pd.DataFrame(predictions, columns=["predicted_answer", "target_answer"])
print(df)

# %% [markdown]
# ### Evaluate all the data

# %%
# Notebook cell 19
evaluation_dataset_path = "lamini/lamini_docs_evaluation"
evaluation_dataset = datasets.load_dataset(evaluation_dataset_path)

# %%
# Notebook cell 20
pd.DataFrame(evaluation_dataset)

# %% [markdown]
# ### Try the ARC benchmark
# This can take several minutes

# %%
# Notebook cell 22
get_ipython().system('python lm-evaluation-harness/main.py --model hf-causal --model_args pretrained=lamini/lamini_docs_finetuned --tasks arc_easy --device cpu')

# %%
# Notebook cell 23

