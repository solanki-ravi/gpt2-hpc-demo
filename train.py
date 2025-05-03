import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from tqdm import tqdm

# Configuration
SEQ_LEN = 1024
BATCH_SIZE = 2 # Align with train_micro_batch_size_per_gpu in DeepSpeed config
EPOCHS = 3
MODEL_NAME = "gpt2"

# Load tokenizer and dataset
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    # Ensure attention_mask is created
    return tokenizer(example['text'], truncation=True, 
                     padding='max_length', max_length=SEQ_LEN,
                     return_attention_mask=True)

print("Loading dataset...")
# Load the initial portion
full_dataset = load_dataset("openwebtext", split="train[:1%]")  # ~40GB

print("Splitting dataset...")
# Split into 80% train and 20% temp (for validation + test)
ds_train_val = full_dataset.train_test_split(test_size=0.2, seed=42) 
# Split the 20% temp into 50% validation (10% of total) and 50% test (10% of total)
ds_val_test = ds_train_val['test'].train_test_split(test_size=0.5, seed=42)

train_dataset = ds_train_val['train']
val_dataset = ds_val_test['train']
test_dataset = ds_val_test['test']

# Construct the message string first
dataset_sizes_str = f"Dataset sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}"
print(dataset_sizes_str)

print("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

print("Setting format...")
data_columns = ['input_ids', 'attention_mask'] # Include attention_mask
train_dataset.set_format(type='torch', columns=data_columns)
val_dataset.set_format(type='torch', columns=data_columns)
test_dataset.set_format(type='torch', columns=data_columns)

# Model config and init
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=SEQ_LEN,
    n_ctx=SEQ_LEN,
    n_embd=768,
    n_layer=12,
    n_head=12,
    pad_token_id=tokenizer.pad_token_id
)
model = GPT2LMHeadModel(config)

# Define Optimizer - Revert to DeepSpeedCPUAdam
optimizer = DeepSpeedCPUAdam(model.parameters(), lr=1e-5) # Use DeepSpeedCPUAdam

# DeepSpeed init
# Initializes Distributed Process Group
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer, # Pass DeepSpeedCPUAdam
    model_parameters=model.parameters(),
    config="deepspeed_config.json"
)

# Add DistributedSampler and DataLoaders *after* DeepSpeed init
train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                          shuffle=False)
# No sampler for validation and test
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Checkpoint Saving Configuration ---
save_directory = "./my_gpt2_checkpoint"
steps_per_checkpoint = 1000

# Training loop
model.train()
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    # Wrap train_loader with tqdm for progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}") 
    for step, batch in enumerate(progress_bar):
        inputs = batch['input_ids'].to(model.device)
        # --- Pass attention_mask to the model --- 
        attention_mask = batch['attention_mask'].to(model.device)
        outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
        loss = outputs.loss
        model.backward(loss)
        model.step()
        
        # Update progress bar description with current loss
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Save checkpoint periodically ---
        if step > 0 and step % steps_per_checkpoint == 0:
            print(f"Saving checkpoint at step {step} of epoch {epoch}")
            # Ensure the tag identifies the step correctly, DeepSpeed might create subdirs
            tag = f"global_step{epoch * len(train_loader) + step}" # Use train_loader length
            model.save_checkpoint(save_directory, tag=tag)
            print(f"Checkpoint '{tag}' saved to {save_directory}")

    # --- Save checkpoint at the end of each epoch ---
    print(f"Saving checkpoint at end of epoch {epoch}")
    tag = f"epoch_end_{epoch}" 
    model.save_checkpoint(save_directory, tag=tag)
    print(f"Checkpoint '{tag}' saved to {save_directory}")

# --- Save final checkpoint ---
print("Saving final checkpoint")
tag = "final_checkpoint"
model.save_checkpoint(save_directory, tag=tag)
print(f"Checkpoint '{tag}' saved to {save_directory}")