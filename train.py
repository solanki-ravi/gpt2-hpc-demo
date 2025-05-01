import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam

# Configuration
SEQ_LEN = 1024
BATCH_SIZE = 50
EPOCHS = 3
MODEL_NAME = "gpt2"

# Load tokenizer and dataset
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=SEQ_LEN)

dataset = load_dataset("openwebtext", split="train[:1%]")  # ~50MB for test
dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type='torch', columns=['input_ids'])

# Add DistributedSampler
sampler = DistributedSampler(dataset)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False)

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

# Define Optimizer
optimizer = DeepSpeedCPUAdam(model.parameters(), lr=1e-5)

# DeepSpeed init
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    model_parameters=model.parameters(),
    config="deepspeed_config.json"
)

# Training loop
model.train()
for epoch in range(EPOCHS):
    for step, batch in enumerate(loader):
        inputs = batch['input_ids'].to(model.device)
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        model.backward(loss)
        model.step()
        if step % 10 == 0:
            print(f"Epoch {epoch} Step {step}: Loss = {loss.item():.4f}")
