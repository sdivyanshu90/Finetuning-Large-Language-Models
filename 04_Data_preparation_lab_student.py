# %% [markdown]
# Source notebook: 04_Data_preparation_lab_student.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Data preparation

# %%
# Notebook cell 2
import pandas as pd
import datasets

from pprint import pprint
from transformers import AutoTokenizer

# %% [markdown]
# ### Tokenizing text

# %%
# Notebook cell 4
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

# %%
# Notebook cell 5
text = "Hi, how are you?"

# %%
# Notebook cell 6
encoded_text = tokenizer(text)["input_ids"]

# %%
# Notebook cell 7
encoded_text

# %%
# Notebook cell 8
decoded_text = tokenizer.decode(encoded_text)
print("Decoded tokens back into text: ", decoded_text)

# %% [markdown]
# ### Tokenize multiple texts at once

# %%
# Notebook cell 10
list_texts = ["Hi, how are you?", "I'm good", "Yes"]
encoded_texts = tokenizer(list_texts)
print("Encoded several texts: ", encoded_texts["input_ids"])

# %% [markdown]
# ### Padding and truncation

# %%
# Notebook cell 12
tokenizer.pad_token = tokenizer.eos_token 
encoded_texts_longest = tokenizer(list_texts, padding=True)
print("Using padding: ", encoded_texts_longest["input_ids"])

# %%
# Notebook cell 13
encoded_texts_truncation = tokenizer(list_texts, max_length=3, truncation=True)
print("Using truncation: ", encoded_texts_truncation["input_ids"])

# %%
# Notebook cell 14
tokenizer.truncation_side = "left"
encoded_texts_truncation_left = tokenizer(list_texts, max_length=3, truncation=True)
print("Using left-side truncation: ", encoded_texts_truncation_left["input_ids"])

# %%
# Notebook cell 15
encoded_texts_both = tokenizer(list_texts, max_length=3, truncation=True, padding=True)
print("Using both padding and truncation: ", encoded_texts_both["input_ids"])

# %% [markdown]
# ### Prepare instruction dataset

# %%
# Notebook cell 17
import pandas as pd

filename = "lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)
examples = instruction_dataset_df.to_dict()

if "question" in examples and "answer" in examples:
  text = examples["question"][0] + examples["answer"][0]
elif "instruction" in examples and "response" in examples:
  text = examples["instruction"][0] + examples["response"][0]
elif "input" in examples and "output" in examples:
  text = examples["input"][0] + examples["output"][0]
else:
  text = examples["text"][0]

prompt_template = """### Question:
{question}

### Answer:"""

num_examples = len(examples["question"])
finetuning_dataset = []
for i in range(num_examples):
  question = examples["question"][i]
  answer = examples["answer"][i]
  text_with_prompt_template = prompt_template.format(question=question)
  finetuning_dataset.append({"question": text_with_prompt_template, "answer": answer})

from pprint import pprint
print("One datapoint in the finetuning dataset:")
pprint(finetuning_dataset[0])

# %% [markdown]
# ### Tokenize a single example

# %%
# Notebook cell 19
text = finetuning_dataset[0]["question"] + finetuning_dataset[0]["answer"]
tokenized_inputs = tokenizer(
    text,
    return_tensors="np",
    padding=True
)
print(tokenized_inputs["input_ids"])

# %%
# Notebook cell 20
max_length = 2048
max_length = min(
    tokenized_inputs["input_ids"].shape[1],
    max_length,
)

# %%
# Notebook cell 21
tokenized_inputs = tokenizer(
    text,
    return_tensors="np",
    truncation=True,
    max_length=max_length
)

# %%
# Notebook cell 22
tokenized_inputs["input_ids"]

# %% [markdown]
# ### Tokenize the instruction dataset

# %%
# Notebook cell 24
def tokenize_function(examples):
    if "question" in examples and "answer" in examples:
      text = examples["question"][0] + examples["answer"][0]
    elif "input" in examples and "output" in examples:
      text = examples["input"][0] + examples["output"][0]
    else:
      text = examples["text"][0]

    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
    )

    max_length = min(
        tokenized_inputs["input_ids"].shape[1],
        2048
    )
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=max_length
    )

    return tokenized_inputs

# %%
# Notebook cell 25
finetuning_dataset_loaded = datasets.load_dataset("json", data_files=filename, split="train")

tokenized_dataset = finetuning_dataset_loaded.map(
    tokenize_function,
    batched=True,
    batch_size=1,
    drop_last_batch=True
)

print(tokenized_dataset)

# %%
# Notebook cell 26
tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])

# %% [markdown]
# ### Prepare test/train splits

# %%
# Notebook cell 28
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=123)
print(split_dataset)

# %% [markdown]
# ### Some datasets for you to try

# %%
# Notebook cell 30
finetuning_dataset_path = "lamini/lamini_docs"
finetuning_dataset = datasets.load_dataset(finetuning_dataset_path)
print(finetuning_dataset)

# %%
# Notebook cell 31
taylor_swift_dataset = "lamini/taylor_swift"
bts_dataset = "lamini/bts"
open_llms = "lamini/open_llms"

# %%
# Notebook cell 32
dataset_swiftie = datasets.load_dataset(taylor_swift_dataset)
print(dataset_swiftie["train"][1])

# %%
# Notebook cell 33
# This is how to push your own dataset to your Huggingface hub
# !pip install huggingface_hub
# !huggingface-cli login
# split_dataset.push_to_hub(dataset_path_hf)
